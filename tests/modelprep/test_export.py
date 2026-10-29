"""Export for MycoMorph (review H4)."""

from __future__ import annotations

import hashlib
import json

import pytest

from mycomorph.modelprep.core.export import export_for_mycomorph


def _run(tmp_path, name="msm_v1_2026-06-14_15-01-07"):
    d = tmp_path / "models" / name
    d.mkdir(parents=True)
    (d / "best_model.pth").write_bytes(b"best-weights")
    (d / "final_model.pth").write_bytes(b"last-epoch")
    (d / "training_config.json").write_text(json.dumps({"in_channels": 3}))
    (d / "metrics.json").write_text("{}")
    return d


def test_export_copies_best_weights_and_sidecars(tmp_path):
    run = _run(tmp_path)
    pth = export_for_mycomorph(run, tmp_path / "Documents")
    out = pth.parent
    assert out == tmp_path / "Documents" / run.name
    assert pth.name == f"{run.name}.pth"
    assert pth.read_bytes() == b"best-weights"          # never final_model.pth
    # MycoMorph reads these from the .pth's folder under their original names.
    assert json.loads((out / "training_config.json").read_text()) == {"in_channels": 3}
    assert (out / "metrics.json").exists()
    note = (out / "EXPORTED_FROM.txt").read_text()
    assert hashlib.sha256(b"best-weights").hexdigest() in note and str(run) in note


def test_export_never_overwrites(tmp_path):
    run = _run(tmp_path)
    first = export_for_mycomorph(run, tmp_path / "out")
    second = export_for_mycomorph(run / "best_model.pth", tmp_path / "out")
    assert first.parent != second.parent and first.exists() and second.exists()
    assert second.parent.name == f"{run.name} (2)"


def test_export_refuses_feature_runs(tmp_path):
    d = tmp_path / "demo-features_2026-06-14_10-00-00"
    d.mkdir()
    (d / "feature_model.joblib").write_bytes(b"")
    with pytest.raises(ValueError, match="feature-based"):
        export_for_mycomorph(d, tmp_path / "out")
