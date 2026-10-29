"""Frozen-architecture guard for the cell-quality CNN.

Every saved ``.pth`` depends on the code between
``# ── BEGIN SHARED MODEL DEFINITION ──`` and
``# ── END SHARED MODEL DEFINITION ──`` in
``mycomorph/core/cell_quality_classifier.py``: the CNN architecture, the area
channel and the ``CellQualityClassifier`` loader. That covers the bundled
presets in ``mycomorph/core/models/`` and every model trained with ModelPrep
(``mycomorph.modelprep``), which imports this block rather than keeping a copy.
Changing the block breaks every saved ``.pth``, so this test pins its SHA-256.

``GOLDEN_SHA`` is also the hash of ModelPrep's former vendored copy (archived
ModelPrep ``release/0.2.0`` = ``b388415``), so the block is byte-identical to
the one the Fig. S2 models were trained with.

Only change the block for a deliberate, versioned change to the model format.
Then read the new SHA from this test's failure message and update
``GOLDEN_SHA``.
"""

from __future__ import annotations

import hashlib
from pathlib import Path


GOLDEN_SHA = "cb4ac9818d5e9f6eb6ee4bc9bd068a57f3dac937efc1346868c4ccaf433ed4f7"


def _hash_shared_block(py_path: Path) -> str:
    # Read as UTF-8 explicitly: the block contains box-drawing characters, and
    # the default encoding on Windows (cp1252) would mis-decode them.
    text = py_path.read_text(encoding="utf-8")
    begin = "# ── BEGIN SHARED MODEL DEFINITION ─"
    end = "# ── END SHARED MODEL DEFINITION ─"
    try:
        i = text.index(begin)
    except ValueError as e:  # pragma: no cover
        raise AssertionError(f"BEGIN marker not found in {py_path}") from e
    try:
        j = text.index(end, i)
    except ValueError as e:  # pragma: no cover
        raise AssertionError(f"END marker not found in {py_path}") from e
    block = text[i:j]
    return hashlib.sha256(block.encode("utf-8")).hexdigest()


def test_frozen_model_definition_matches_golden():
    import mycomorph.core.cell_quality_classifier as m

    sha = _hash_shared_block(Path(m.__file__))
    assert sha == GOLDEN_SHA, (
        f"The frozen model definition changed.\n"
        f"  expected: {GOLDEN_SHA}\n"
        f"  actual:   {sha}\n"
        f"Changing this block breaks every saved .pth (the bundled presets and "
        f"every model trained with ModelPrep). Revert it, or, for a deliberate "
        f"change to the model format, update GOLDEN_SHA."
    )
