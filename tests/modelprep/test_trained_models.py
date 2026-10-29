"""Tests for trained-model enumeration (data layer, no Qt)."""

from __future__ import annotations

import json
from pathlib import Path

import mycomorph.modelprep.gui.trained_models as tm
from mycomorph.modelprep.gui.trained_models import list_trained_models


def _make_run(root: Path, name: str, ts: str, **cfg) -> Path:
    d = root / f"{name}_{ts}"
    d.mkdir(parents=True)
    (d / "best_model.pth").write_bytes(b"x")
    (d / "training_config.json").write_text(json.dumps(cfg))
    return d


def test_list_populates_config_and_orders_newest_first(tmp_path, monkeypatch):
    root = tmp_path / "models"
    root.mkdir()
    monkeypatch.setattr(tm, "trained_models_root", lambda: root)

    _make_run(root, "alpha", "2026-06-11_00-00-00",
              roc_auc=0.74, balanced_acc=0.71, eval_split="holdout")
    _make_run(root, "beta", "2026-06-11_01-00-00",
              roc_auc=0.91, balanced_acc=0.86, eval_split="holdout")

    models = list_trained_models()
    assert [m.name for m in models] == ["beta", "alpha"]  # newest first
    # The full config is carried through for the comparison table.
    assert models[0].config["roc_auc"] == 0.91
    assert models[0].config["eval_split"] == "holdout"
    assert models[1].config["balanced_acc"] == 0.71


def test_list_skips_dirs_without_weights(tmp_path, monkeypatch):
    root = tmp_path / "models"
    root.mkdir()
    monkeypatch.setattr(tm, "trained_models_root", lambda: root)
    # A run dir with a config but no checkpoint is ignored.
    d = root / "incomplete_2026-06-11_02-00-00"
    d.mkdir()
    (d / "training_config.json").write_text(json.dumps({"roc_auc": 0.5}))
    assert list_trained_models() == []


def test_list_recognises_feature_runs(tmp_path, monkeypatch):
    root = tmp_path / "models"
    root.mkdir()
    monkeypatch.setattr(tm, "trained_models_root", lambda: root)

    _make_run(root, "demo", "2026-06-14_10-00-00", roc_auc=0.83)  # CNN: best_model.pth
    # Feature run: only a joblib artefact (no .pth) — exercises the OR condition.
    feat = root / "demo-features_2026-06-14_10-00-00"
    feat.mkdir()
    (feat / "feature_model.joblib").write_bytes(b"x")
    (feat / "training_config.json").write_text(json.dumps({"model_type": "features", "roc_auc": 0.85}))

    by_type = {m.model_type: m for m in list_trained_models()}
    assert set(by_type) == {"cnn", "features"}
    assert by_type["features"].weights.name == "feature_model.joblib"
    assert by_type["cnn"].weights.name == "best_model.pth"
