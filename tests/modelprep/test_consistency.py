"""Tests for the blinded label-consistency check (core)."""

from __future__ import annotations

import csv
from pathlib import Path

from modelprep.core.consistency import (
    SAMPLE_BALANCED,
    SAMPLE_RANDOM,
    _kappa_band,
    consistency_stats,
    sample_for_consistency,
    save_consistency_run,
)
from modelprep.core.crops_h5 import HDF5CropsReader
from modelprep.core.labels_store import LabelRow, LabelsStore


def _row(uid: str, name: str) -> LabelRow:
    return LabelRow(cell_uid=uid, label_idx=0 if name == "good" else 1,
                    label_name=name, area_px=100)


def test_stats_agreement_kappa_flips_and_disagreements():
    # 8 agree, 2 flip (one each direction) → 80% agreement.
    pairs = [("good", "good")] * 4 + [("bad", "bad")] * 4 + [("good", "bad"), ("bad", "good")]
    uids = [f"u{i}" for i in range(len(pairs))]
    s = consistency_stats(pairs, uids)
    assert s["n"] == 10 and s["n_agree"] == 8
    assert abs(s["raw_agreement"] - 0.8) < 1e-9
    assert abs(s["cohen_kappa"] - 0.6) < 1e-9          # known value for this table
    assert s["kappa_band"] == "substantial"
    assert s["flips"]["good"] == {"good": 4, "bad": 1}
    assert s["flips"]["bad"] == {"good": 1, "bad": 4}
    assert set(s["disagreements"]) == {"u8", "u9"}


def test_stats_empty():
    s = consistency_stats([])
    assert s["n"] == 0 and s["disagreements"] == []


def test_kappa_bands():
    assert _kappa_band(-0.1) == "poor"
    assert _kappa_band(0.1) == "slight"
    assert _kappa_band(0.5) == "moderate"
    assert _kappa_band(0.7) == "substantial"
    assert _kappa_band(0.9) == "almost perfect"


def test_sample_balanced_and_random(synthetic_h5, tmp_path):
    store = LabelsStore(tmp_path / "labels")
    store.ensure_initialised(source_h5=synthetic_h5, total_cells=32, h5_schema_version="1.0")
    # 12 good, 8 bad.
    store.append_batch([_row(f"cell_{i:04d}", "good") for i in range(12)]
                       + [_row(f"cell_{i:04d}", "bad") for i in range(12, 20)])

    reader = HDF5CropsReader(synthetic_h5)
    try:
        bal = sample_for_consistency(reader, store, n=12, mode=SAMPLE_BALANCED, seed=1)
        rnd = sample_for_consistency(reader, store, n=15, mode=SAMPLE_RANDOM, seed=1)
    finally:
        reader.close()

    # Balanced: ~equal classes (6 each here).
    bal_good = sum(1 for _, lab in bal if lab == "good")
    bal_bad = sum(1 for _, lab in bal if lab == "bad")
    assert bal_good == bal_bad == 6
    # Random: drawn only from labelled cells, no duplicates.
    rnd_uids = [rec.cell_uid for rec, _ in rnd]
    assert len(rnd_uids) == len(set(rnd_uids)) == 15
    labelled = {f"cell_{i:04d}" for i in range(20)}
    assert set(rnd_uids).issubset(labelled)


def test_save_run_writes_separate_file_not_labels_csv(tmp_path):
    labels_dir = tmp_path / "labels"
    labels_dir.mkdir()
    rows = [("u0", "good", "good"), ("u1", "good", "bad")]
    stats = consistency_stats([("good", "good"), ("good", "bad")], ["u0", "u1"])
    out = save_consistency_run(labels_dir, SAMPLE_BALANCED, rows, stats)
    assert (out / "results.csv").exists() and (out / "summary.json").exists()
    assert out.parent == labels_dir / "consistency"
    assert not (labels_dir / "labels.csv").exists()  # never touched
    with open(out / "results.csv") as f:
        assert list(csv.reader(f))[0] == ["cell_uid", "original_label", "recheck_label"]
