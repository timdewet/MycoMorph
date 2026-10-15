"""CLI parity with the GUI (review M3): train on the frozen hold-out, re-score
saved models, learning curve, consistency summary, export."""

from __future__ import annotations

import json

import numpy as np
import pytest
from typer.testing import CliRunner

from mycomorph.modelprep.core.cli import app
from mycomorph.modelprep.core.labels_store import LabelRow, LabelsStore, SPLIT_TEST, SPLIT_TRAIN

runner = CliRunner()


def _labels(synthetic_h5, tmp_path):
    """24 train (12/12) + 8 hold-out (4/4) labels over the 32 synthetic cells."""
    store = LabelsStore(tmp_path / "labels")
    store.ensure_initialised(source_h5=synthetic_h5, total_cells=32, h5_schema_version="1.0")
    rows = []
    for i in range(32):
        bad = i % 2
        rows.append(LabelRow(cell_uid=f"cell_{i:04d}", label_idx=bad,
                             label_name="bad" if bad else "good",
                             split=SPLIT_TEST if i >= 24 else SPLIT_TRAIN))
    store.append_batch(rows)
    return tmp_path / "labels"


def _invoke(*args):
    res = runner.invoke(app, [str(a) for a in args])
    assert res.exit_code == 0, res.output
    return res.output


def test_train_uses_frozen_holdout_and_evaluate_reproduces_it(synthetic_h5, tmp_path):
    ld = _labels(synthetic_h5, tmp_path)
    out = tmp_path / "run"
    text = _invoke("train", "--h5", synthetic_h5, "--labels-dir", ld, "--output", out,
                   "--epochs", 2, "--device", "cpu", "--seed", 0)
    assert "frozen hold-out: 8 cells" in text
    cfg = json.loads((out / "training_config.json").read_text())
    assert cfg["eval_split"] == "holdout" and cfg["n_holdout"] == 8
    assert cfg["lr"] == pytest.approx(1e-3)              # from scratch default
    assert cfg["provenance"]["entry_point"] == "modelprep-cli train"

    # Re-scoring the saved model on the same hold-out gives the same numbers.
    res_file = tmp_path / "eval.json"
    _invoke("evaluate", "--h5", synthetic_h5, "--labels-dir", ld, "--model", out,
            "--json", res_file)
    res = json.loads(res_file.read_text())
    assert res["n"] == 8 and res["model_type"] == "cnn"
    assert res["eval_acc"] == pytest.approx(cfg["eval_acc"])
    if cfg["roc_auc"] == cfg["roc_auc"]:
        assert res["roc_auc"] == pytest.approx(cfg["roc_auc"], abs=1e-6)
    assert len(res["holdout_sha256"]) == 64


def test_train_no_holdout_and_finetune_defaults(synthetic_h5, tmp_path):
    ld = _labels(synthetic_h5, tmp_path)
    out = tmp_path / "ft"
    _invoke("train", "--h5", synthetic_h5, "--labels-dir", ld, "--output", out,
            "--pretrained", "mtb", "--epochs", 1, "--device", "cpu", "--no-holdout")
    cfg = json.loads((out / "training_config.json").read_text())
    assert cfg["eval_split"] != "holdout" and cfg["n_holdout"] == 0
    assert cfg["lr"] == pytest.approx(1e-4)              # GUI fine-tune default


def test_evaluate_feature_model(synthetic_h5, tmp_path):
    pd = pytest.importorskip("pandas")
    from mycomorph.modelprep.core.feature_classifier import train_and_save

    ld = _labels(synthetic_h5, tmp_path)
    rng = np.random.default_rng(0)
    y = np.arange(32) % 2
    pd.DataFrame({"cell_uid": [f"cell_{i:04d}" for i in range(32)],
                  "fov": np.zeros(32, int), "centroid_y": rng.random(32) * 900,
                  "centroid_x": rng.random(32) * 900, "bbox_y0": np.zeros(32, int),
                  "bbox_x0": np.zeros(32, int), "bbox_y1": np.full(32, 20),
                  "bbox_x1": np.full(32, 20),
                  "solidity": y + rng.normal(0, 0.3, 32)}).to_parquet(
        synthetic_h5.parent / "all_features.parquet")
    s = train_and_save(synthetic_h5, ld, tmp_path / "feat")

    res_file = tmp_path / "eval.json"
    _invoke("evaluate", "--h5", synthetic_h5, "--labels-dir", ld,
            "--model", tmp_path / "feat", "--json", res_file)
    res = json.loads(res_file.read_text())
    assert res["model_type"] == "features" and res["n"] == s["n_test"]
    assert res["balanced_acc"] == pytest.approx(s["balanced_acc"])


def test_learning_curve_consistency_and_export(synthetic_h5, tmp_path):
    from mycomorph.modelprep.core.consistency import consistency_stats, save_consistency_run

    ld = _labels(synthetic_h5, tmp_path)
    curve = tmp_path / "curve"
    _invoke("learning-curve", "--h5", synthetic_h5, "--labels-dir", ld, "--points", 2,
            "--epochs", 1, "--device", "cpu", "--output", curve)
    pts = json.loads((curve / "curve.json").read_text())["points"]
    assert pts and (curve / "curve.csv").exists()

    stats = consistency_stats([("good", "good"), ("bad", "good")], uids=["a", "b"])
    save_consistency_run(ld, "balanced", [("a", "good", "good"), ("b", "bad", "good")], stats)
    text = _invoke("consistency", "--labels-dir", ld)
    assert "50.0%" in text and "bad->good 1" in text

    run = tmp_path / "models" / "demo_2026-06-14_10-00-00"
    run.mkdir(parents=True)
    (run / "best_model.pth").write_bytes(b"w")
    text = _invoke("export", "--model", run, "--dest", tmp_path / "out")
    assert (tmp_path / "out" / run.name / f"{run.name}.pth").exists()


def test_evaluate_preprocesses_like_training(synthetic_h5, tmp_path, monkeypatch):
    """The hold-out pass in train() (and MycoMorph's inference crops) min-max
    the phase channel to [0, 1] after the resize; re-scoring must too, or its
    numbers drift from the ones training reported."""
    import mycomorph.core.cell_quality_classifier as cqc
    from mycomorph.modelprep.core.evaluate import evaluate_model

    ld = _labels(synthetic_h5, tmp_path)
    seen = []

    class Spy:
        def __init__(self, *_a, **_k):
            pass

        def predict_batch(self, crops):
            seen.append(np.array(crops))
            p = np.tile([0.4, 0.6], (len(crops), 1)).astype(np.float32)
            return p.argmax(1), p

    monkeypatch.setattr(cqc, "CellQualityClassifier", Spy)
    model = tmp_path / "m.pth"
    model.write_bytes(b"")
    evaluate_model(synthetic_h5, ld, model)
    crops = np.concatenate(seen)
    assert crops.shape == (8, 3, 64, 64)
    assert np.allclose(crops[:, 0].min(axis=(1, 2)), 0.0)
    assert np.allclose(crops[:, 0].max(axis=(1, 2)), 1.0)
