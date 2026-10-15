"""Drift guard for the model definition shared with ModelPrep.

ModelPrep (the separate labeller / trainer) produces the ``.pth`` files that
the Classify stage loads, so the CNN architecture, the area channel and the
``CellQualityClassifier`` loader must match on both sides. The block between
``# ── BEGIN SHARED MODEL DEFINITION ──`` and
``# ── END SHARED MODEL DEFINITION ──`` in
``mycomorph/core/cell_quality_classifier.py`` is byte-identical to the block in
ModelPrep's ``modelprep/core/cell_quality_classifier.py``. Both repos pin the
SHA-256 of that block to the same ``GOLDEN_SHA``.

When the architecture, area channel or loader legitimately changes:
  1. Update both repos in the same change.
  2. Run this test to read the new SHA from the failure message.
  3. Paste the new SHA into both repos' ``GOLDEN_SHA``.

Set ``MYCOMORPH_SIBLING_CLF`` to the path of ModelPrep's
``cell_quality_classifier.py`` to also assert that both copies hash
identically. (ModelPrep's own test reads ``MODELPREP_SIBLING_CLF`` for the
reverse direction.)
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest


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


def test_shared_block_hash_matches_golden():
    import mycomorph.core.cell_quality_classifier as m

    sha = _hash_shared_block(Path(m.__file__))
    assert sha == GOLDEN_SHA, (
        f"Shared model definition has drifted from the golden hash.\n"
        f"  expected: {GOLDEN_SHA}\n"
        f"  actual:   {sha}\n"
        f"If this change is intentional, make the same change in ModelPrep and "
        f"update GOLDEN_SHA in BOTH repos."
    )


def test_shared_block_matches_sibling_if_present():
    """Two-sided guard: when MYCOMORPH_SIBLING_CLF points at ModelPrep's
    cell_quality_classifier.py, assert both copies hash identically. Skipped
    when the env var is unset or the path is missing (the default local run)."""
    sibling = os.environ.get("MYCOMORPH_SIBLING_CLF")
    if not sibling or not Path(sibling).exists():
        pytest.skip("MYCOMORPH_SIBLING_CLF not set / missing; sibling check skipped")

    import mycomorph.core.cell_quality_classifier as m

    ours = _hash_shared_block(Path(m.__file__))
    theirs = _hash_shared_block(Path(sibling))
    assert ours == theirs, (
        f"Shared model definition differs between MycoMorph and ModelPrep.\n"
        f"  MycoMorph: {ours}\n  ModelPrep: {theirs} ({sibling})\n"
        f"Update both copies in the same change."
    )

