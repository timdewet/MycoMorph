"""Locate ModelPrep's bundled resources (logo, icons, rubric) in both source
and frozen builds."""

from __future__ import annotations

import sys
from pathlib import Path


def resource_root() -> Path:
    """Directory holding ModelPrep's non-Python assets (logo, icons, …).

    Assets live *inside the package* (``mycomorph/modelprep/gui/assets/``) so
    they ship in wheels/sdists. The desktop bundle (``packaging/mycomorph.spec``)
    copies the package's data files to the same package path, so call sites can
    always write ``resource_root() / "logo" / "logo.svg"``. (The bundle's root
    holds MycoMorph's own logo, not ModelPrep's.)
    """
    return Path(__file__).resolve().parent / "assets"


RUBRIC_URL = "https://github.com/timdewet/MycoMorph/blob/main/docs/modelprep/labelling_rubric.md"


def rubric_location() -> str:
    """Where the labelling rubric is: the bundled/checkout copy if present
    (a local path), else its URL in the repository."""
    candidates = []
    meipass = getattr(sys, "_MEIPASS", None)
    if getattr(sys, "frozen", False) and meipass:
        # packaging/mycomorph.spec bundles docs/modelprep/ under the same path.
        candidates.append(Path(meipass) / "docs" / "modelprep" / "labelling_rubric.md")
    # Source checkout / editable install: <repo>/docs/modelprep/labelling_rubric.md.
    candidates.append(
        Path(__file__).resolve().parents[4] / "docs" / "modelprep" / "labelling_rubric.md"
    )
    for c in candidates:
        if c.is_file():
            return str(c)
    return RUBRIC_URL
