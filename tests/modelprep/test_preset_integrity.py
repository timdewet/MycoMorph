"""Integrity and loadability of the bundled classifier presets.

Each preset's sidecar (``<name>_best_model.json``) records the sha256 of the
checked-in weights, so a corrupted or accidentally replaced ``.pth`` fails CI.
The sidecar also carries the description of what the model was trained on.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

PRESETS = ("mtb-2026-06", "msm-2026-06")
DESCRIPTION_FIELDS = ("organism", "summary", "training_data", "imaging", "labels",
                      "training", "performance", "intended_use", "label_criteria")


def _models_dir() -> Path:
    import mycomorph.core.cell_quality_classifier as m
    return Path(m.__file__).parent / "models"


def test_bundled_presets_are_exactly_the_expected_set():
    from mycomorph.core.api import PRESET_MODELS
    from mycomorph.modelprep.core.cli import list_preset_models

    assert set(PRESET_MODELS) == set(PRESETS)
    assert set(list_preset_models()) == set(PRESETS)
    assert sorted(p.name for p in _models_dir().glob("*.pth")) == sorted(
        f"{n}_best_model.pth" for n in PRESETS)


@pytest.mark.parametrize("name", PRESETS)
def test_sidecar_sha_matches(name):
    models = _models_dir()
    pth = models / f"{name}_best_model.pth"
    sidecar = models / f"{name}_best_model.json"
    assert pth.exists(), f"missing bundled preset: {pth}"
    assert sidecar.exists(), f"missing preset sidecar: {sidecar}"
    meta = json.loads(sidecar.read_text(encoding="utf-8"))
    assert meta["name"] == name and meta["file"] == pth.name
    sha = hashlib.sha256(pth.read_bytes()).hexdigest()
    assert sha == meta["sha256"], (
        f"bundled {pth.name} no longer matches its recorded sha256: "
        "the preset was replaced or corrupted, or the sidecar is stale."
    )


@pytest.mark.parametrize("name", PRESETS)
def test_sidecar_describes_training(name):
    meta = json.loads((_models_dir() / f"{name}_best_model.json").read_text(encoding="utf-8"))
    missing = [f for f in DESCRIPTION_FIELDS if not str(meta.get(f, "")).strip()]
    assert not missing, f"{name} sidecar lacks: {missing}"


@pytest.mark.parametrize("name", PRESETS)
def test_preset_loads_and_runs(name):
    import numpy as np

    from mycomorph.core.cell_quality_classifier import (
        CROP_SIZE, NUM_CLASSES, CellQualityClassifier,
    )
    from mycomorph.modelprep.core.cli import resolve_classifier_preset

    p = resolve_classifier_preset(name)
    clf = CellQualityClassifier(str(p), in_channels=3, device="cpu")
    _idx, probs = clf.predict_batch(
        np.zeros((2, 3, CROP_SIZE, CROP_SIZE), dtype=np.float32)
    )
    assert probs.shape == (2, NUM_CLASSES)
