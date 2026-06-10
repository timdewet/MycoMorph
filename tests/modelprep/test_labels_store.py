"""Tests for the labels store: append, load, latest-wins, manifest validation."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from modelprep.core.labels_store import LabelRow, LabelsStore


def _row(uid: str, name: str, **overrides) -> LabelRow:
    base = dict(
        cell_uid=uid,
        label_idx=0 if name == "good" else 1,
        label_name=name,
        area_px=100,
        session_id="s",
        source_h5="all_crops.h5",
    )
    base.update(overrides)
    return LabelRow(**base)


def test_empty_load(tmp_path: Path):
    store = LabelsStore(tmp_path / "labels")
    assert store.load().rows == []
    assert store.labeled_uids() == set()
    assert store.class_counts() == {"good": 0, "bad": 0}


def test_append_then_load(tmp_path: Path, synthetic_h5):
    store = LabelsStore(tmp_path / "labels")
    store.ensure_initialised(
        source_h5=synthetic_h5, total_cells=32, h5_schema_version="1.0",
    )
    store.append_batch([
        _row("cell_0000", "good"),
        _row("cell_0001", "bad"),
    ])
    frame = store.load()
    assert len(frame) == 2
    uids = {r.cell_uid for r in frame.rows}
    assert uids == {"cell_0000", "cell_0001"}
    counts = store.class_counts()
    assert counts == {"good": 1, "bad": 1}


def test_latest_label_wins(tmp_path: Path, synthetic_h5):
    store = LabelsStore(tmp_path / "labels")
    store.ensure_initialised(
        source_h5=synthetic_h5, total_cells=32, h5_schema_version="1.0",
    )
    earlier = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
    later = datetime.now(timezone.utc).isoformat()
    store.append_batch([_row("cell_0000", "good", labeled_at=earlier)])
    store.append_batch([_row("cell_0000", "bad", labeled_at=later)])

    latest = store.load().latest_by_uid()
    assert latest["cell_0000"].label_name == "bad"
    assert store.class_counts() == {"good": 0, "bad": 1}


def test_manifest_mismatch_raises(tmp_path: Path, synthetic_h5):
    store = LabelsStore(tmp_path / "labels")
    store.ensure_initialised(
        source_h5=synthetic_h5, total_cells=32, h5_schema_version="1.0",
    )
    # Now pretend a different HDF5 has fewer cells; reopening should reject it.
    with pytest.raises(ValueError):
        store.ensure_initialised(
            source_h5=synthetic_h5, total_cells=999, h5_schema_version="1.0",
        )


def test_to_training_dataset_round_trip(tmp_path: Path, synthetic_h5):
    from modelprep.core.crops_h5 import HDF5CropsReader

    store = LabelsStore(tmp_path / "labels")
    store.ensure_initialised(
        source_h5=synthetic_h5, total_cells=32, h5_schema_version="1.0",
    )
    store.append_batch([
        _row("cell_0000", "good"),
        _row("cell_0005", "good"),
        _row("cell_0010", "bad"),
        _row("cell_0015", "bad"),
    ])

    reader = HDF5CropsReader(synthetic_h5)
    try:
        crops, labels, uids = store.to_training_dataset(reader)
    finally:
        reader.close()

    assert crops.shape == (4, 3, 64, 64)
    assert sorted(labels.tolist()) == [0, 0, 1, 1]
    assert set(uids) == {"cell_0000", "cell_0005", "cell_0010", "cell_0015"}
