"""Integrity + loadability of the bundled ``mtb`` preset.

The committed sidecar (``mtb_best_model.json``) records the sha256 of the
checked-in weights so a corrupted or accidentally-replaced ``.pth`` fails CI.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def _models_dir() -> Path:
    import modelprep.core.cell_quality_classifier as m
    return Path(m.__file__).parent / "models"


def test_mtb_sidecar_sha_matches():
    models = _models_dir()
    pth = models / "mtb_best_model.pth"
    sidecar = models / "mtb_best_model.json"
    assert pth.exists(), f"missing bundled preset: {pth}"
    assert sidecar.exists(), f"missing preset sidecar: {sidecar}"
    meta = json.loads(sidecar.read_text())
    sha = hashlib.sha256(pth.read_bytes()).hexdigest()
    assert sha == meta["sha256"], (
        "bundled mtb_best_model.pth no longer matches its recorded sha256 — "
        "the preset was replaced/corrupted, or the sidecar is stale."
    )


def test_mtb_loads_and_runs():
    import numpy as np

    from modelprep.core.cell_quality_classifier import (
        CROP_SIZE, NUM_CLASSES, CellQualityClassifier,
    )
    from modelprep.core.cli import resolve_classifier_preset

    p = resolve_classifier_preset("mtb")
    clf = CellQualityClassifier(str(p), in_channels=3, device="cpu")
    _idx, probs = clf.predict_batch(
        np.zeros((2, 3, CROP_SIZE, CROP_SIZE), dtype=np.float32)
    )
    assert probs.shape == (2, NUM_CLASSES)
