"""Drift-guard test for the shared model definition block.

The block between ``# ── BEGIN SHARED MODEL DEFINITION ──`` and
``# ── END SHARED MODEL DEFINITION ──`` in
``modelprep/core/cell_quality_classifier.py`` MUST be byte-identical to the
corresponding block in MycoMorph's copy. This test computes the SHA-256 of the
ModelPrep block and compares against ``GOLDEN_SHA``.

When the architecture / area channel / classifier loader legitimately changes:
  1. Update both repos in the same commit.
  2. Run this test (with print) to read the new SHA.
  3. Paste the new SHA into both repos' ``GOLDEN_SHA``.

Set ``MODELPREP_SIBLING_CLF`` to the path of the sibling repo's
``cell_quality_classifier.py`` (MycoMorph) to also assert *both* copies
hash identically — a true two-sided guard in CI where the sibling is checked out.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest


GOLDEN_SHA = "12bcaae99b6ae0f540d9ed92ea8523a3ff2191ab04e0c9e37580f7193626b55f"


def _hash_shared_block(py_path: Path) -> str:
    text = py_path.read_text()
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
    import modelprep.core.cell_quality_classifier as m

    py_path = Path(m.__file__)
    sha = _hash_shared_block(py_path)
    if GOLDEN_SHA == "<TBD>":
        # First run: print so the dev can paste it back.
        print(f"\nFirst-run hash for shared block: {sha}")
        print("Paste this into GOLDEN_SHA in this file AND in MycoMorph's "
              "tests/test_classifier_model_def.py.")
        return
    assert sha == GOLDEN_SHA, (
        f"Shared model definition has drifted from the golden hash.\n"
        f"  expected: {GOLDEN_SHA}\n"
        f"  actual:   {sha}\n"
        f"If this change is intentional, update GOLDEN_SHA in BOTH repos."
    )


def test_shared_block_matches_sibling_if_present():
    """Two-sided guard: when MODELPREP_SIBLING_CLF points at the sibling repo's
    cell_quality_classifier.py, assert both copies hash identically. Skipped
    when the env var is unset or the path is missing (the default local run)."""
    sibling = os.environ.get("MODELPREP_SIBLING_CLF")
    if not sibling or not Path(sibling).exists():
        pytest.skip("MODELPREP_SIBLING_CLF not set / missing — sibling check skipped")

    import modelprep.core.cell_quality_classifier as m

    ours = _hash_shared_block(Path(m.__file__))
    theirs = _hash_shared_block(Path(sibling))
    assert ours == theirs, (
        f"Shared model definition differs between ModelPrep and the sibling.\n"
        f"  ModelPrep: {ours}\n  sibling:   {theirs} ({sibling})\n"
        f"Update both copies in the same commit."
    )
