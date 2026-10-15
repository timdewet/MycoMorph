"""Tests for the labels store: append, load, latest-wins, manifest validation."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

import csv

from modelprep.core.labels_store import (
    SPLIT_TEST,
    SPLIT_TRAIN,
    LabelRow,
    LabelsStore,
)


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


# ── hold-out test split ──────────────────────────────────────────────────────

def test_split_round_trips_and_counts(tmp_path: Path, synthetic_h5):
    store = LabelsStore(tmp_path / "labels")
    store.ensure_initialised(
        source_h5=synthetic_h5, total_cells=32, h5_schema_version="1.0",
    )
    store.append_batch([
        _row("cell_0000", "good"),                       # default split = train
        _row("cell_0001", "bad", split=SPLIT_TRAIN),
        _row("cell_0002", "good", split=SPLIT_TEST),
        _row("cell_0003", "bad", split=SPLIT_TEST),
    ])
    latest = store.load().latest_by_uid()
    assert latest["cell_0000"].split == SPLIT_TRAIN
    assert latest["cell_0002"].split == SPLIT_TEST

    assert store.class_counts(SPLIT_TRAIN) == {"good": 1, "bad": 1}
    assert store.class_counts(SPLIT_TEST) == {"good": 1, "bad": 1}
    assert store.class_counts() == {"good": 2, "bad": 2}  # both splits
    assert store.split_counts() == {
        "train": {"good": 1, "bad": 1},
        "test": {"good": 1, "bad": 1},
    }


def test_train_and_eval_datasets_are_disjoint(tmp_path: Path, synthetic_h5):
    from modelprep.core.crops_h5 import HDF5CropsReader

    store = LabelsStore(tmp_path / "labels")
    store.ensure_initialised(
        source_h5=synthetic_h5, total_cells=32, h5_schema_version="1.0",
    )
    store.append_batch([
        _row("cell_0000", "good", split=SPLIT_TRAIN),
        _row("cell_0001", "bad", split=SPLIT_TRAIN),
        _row("cell_0002", "good", split=SPLIT_TEST),
        _row("cell_0003", "bad", split=SPLIT_TEST),
    ])
    reader = HDF5CropsReader(synthetic_h5)
    try:
        _, tr_labels, tr_uids = store.to_training_dataset(reader)
        ev_crops, ev_labels, ev_uids = store.to_eval_dataset(reader)
    finally:
        reader.close()

    assert set(tr_uids) == {"cell_0000", "cell_0001"}
    assert set(ev_uids) == {"cell_0002", "cell_0003"}
    assert set(tr_uids).isdisjoint(ev_uids)
    assert ev_crops.shape == (2, 3, 64, 64)
    assert sorted(ev_labels.tolist()) == [0, 1]


def test_eval_dataset_empty_without_test_rows(tmp_path: Path, synthetic_h5):
    from modelprep.core.crops_h5 import HDF5CropsReader

    store = LabelsStore(tmp_path / "labels")
    store.ensure_initialised(
        source_h5=synthetic_h5, total_cells=32, h5_schema_version="1.0",
    )
    store.append_batch([_row("cell_0000", "good")])  # train only
    reader = HDF5CropsReader(synthetic_h5)
    try:
        crops, labels, uids = store.to_eval_dataset(reader)
    finally:
        reader.close()
    assert len(labels) == 0 and len(uids) == 0
    assert crops.shape[0] == 0


def test_uids_by_split_and_restore_holdout(tmp_path: Path, synthetic_h5):
    store = LabelsStore(tmp_path / "labels")
    store.ensure_initialised(
        source_h5=synthetic_h5, total_cells=32, h5_schema_version="1.0",
    )
    store.append_batch([
        _row("cell_0000", "good", split=SPLIT_TEST),
        _row("cell_0001", "bad", split=SPLIT_TEST),
        _row("cell_0002", "good", split=SPLIT_TRAIN),
    ])
    assert store.uids_by_split(SPLIT_TEST) == {"cell_0000", "cell_0001"}

    # A hold-out cell gets accidentally moved to train and relabelled (as review did).
    store.append_batch([_row("cell_0000", "bad", split=SPLIT_TRAIN)])
    assert store.uids_by_split(SPLIT_TEST) == {"cell_0001"}

    # Restore it: back in test, keeping its *current* (corrected) label.
    assert store.restore_split(["cell_0000"], SPLIT_TEST) == 1
    assert store.uids_by_split(SPLIT_TEST) == {"cell_0000", "cell_0001"}
    assert store.load().latest_by_uid()["cell_0000"].label_name == "bad"  # correction kept
    # Already-test cells are a no-op.
    assert store.restore_split(["cell_0001"], SPLIT_TEST) == 0


def test_legacy_csv_without_split_migrates_in_place(tmp_path: Path):
    """A labels.csv written before the 'split' column upgrades on the next
    append: header gains 'split', old rows become 'train', new row keeps its
    split, and nothing is lost."""
    labels_dir = tmp_path / "labels"
    labels_dir.mkdir()
    csv_path = labels_dir / "labels.csv"
    old_cols = [
        "cell_uid", "label_idx", "label_name", "area_px", "labeled_at",
        "labeler", "session_id", "source_h5", "model_at_label",
        "confidence_good_at_label",
    ]
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(old_cols)
        w.writerow(["cell_A", 0, "good", 100, "2026-01-01", "t", "s", "h5", "none", 0.9])
        w.writerow(["cell_B", 1, "bad", 50, "2026-01-01", "t", "s", "h5", "none", 0.1])

    store = LabelsStore(labels_dir)
    store.append_batch([_row("cell_C", "good", split=SPLIT_TEST)])

    with open(csv_path, newline="") as f:
        header = next(csv.reader(f))
    assert "split" in header

    latest = store.load().latest_by_uid()
    assert latest["cell_A"].split == SPLIT_TRAIN  # back-filled
    assert latest["cell_B"].split == SPLIT_TRAIN
    assert latest["cell_C"].split == SPLIT_TEST
    assert latest["cell_A"].label_name == "good"  # data preserved
    assert store.split_counts() == {
        "train": {"good": 1, "bad": 1},
        "test": {"good": 1, "bad": 0},
    }


def test_non_ascii_labeler_and_model_round_trip(tmp_path: Path, synthetic_h5):
    """labels.csv is shared Mac <-> Windows via Dropbox, so it is always UTF-8,
    whatever the platform's default encoding."""
    store = LabelsStore(tmp_path / "labels")
    store.ensure_initialised(
        source_h5=synthetic_h5, total_cells=32, h5_schema_version="1.0",
    )
    labeler = "Zoë Łukasiewicz-Müller"
    model = "C:/Users/Zoë/ModelPrep/models/łódź_güt_v1.pth"
    store.append_batch([_row("cell_0000", "good", labeler=labeler, model_at_label=model)])

    row = store.load().latest_by_uid()["cell_0000"]
    assert row.labeler == labeler
    assert row.model_at_label == model
    assert labeler in store.csv_path.read_bytes().decode("utf-8")


@pytest.mark.parametrize("with_split", [True, False])
def test_legacy_code_page_rows_still_load(tmp_path: Path, monkeypatch, capsys,
                                          with_split):
    """Rows an older ModelPrep wrote on Windows (cp1252), mixed with UTF-8
    rows from a Mac, load with a one-line warning instead of crashing — and
    keep loading after a new append (which migrates an old-schema file)."""
    from modelprep.core import labels_store as ls

    monkeypatch.setattr(ls, "_legacy_encoding", lambda: "cp1252")
    labels_dir = tmp_path / "labels"
    labels_dir.mkdir()
    csv_path = labels_dir / "labels.csv"
    cols = list(ls.CSV_COLUMNS) if with_split else list(ls.CSV_COLUMNS)[:-1]

    def line(uid, labeler):
        vals = [uid, 0, "good", 100, "2026-01-01", labeler, "s", "h5", "none", 0.9]
        return ",".join(map(str, vals + (["train"] if with_split else []))) + "\r\n"

    csv_path.write_bytes(
        (",".join(cols) + "\r\n").encode("ascii")
        + line("cell_A", "José").encode("cp1252")  # Windows, old ModelPrep
        + line("cell_B", "Zoë").encode("utf-8")    # Mac
    )

    store = LabelsStore(labels_dir)
    latest = store.load().latest_by_uid()
    assert latest["cell_A"].labeler == "José"
    assert latest["cell_B"].labeler == "Zoë"
    err = capsys.readouterr().err
    assert err.count("WARNING") == 1 and "cp1252" in err
    store.load()
    assert capsys.readouterr().err == ""  # warned once per file

    store.append_batch([_row("cell_C", "bad", labeler="Łukasz")])
    latest = store.load().latest_by_uid()
    assert {uid: r.labeler for uid, r in latest.items()} == {
        "cell_A": "José", "cell_B": "Zoë", "cell_C": "Łukasz",
    }
    if not with_split:  # the schema migration rewrote the whole file as UTF-8
        csv_path.read_bytes().decode("utf-8")
