"""Locate bundled resources (logo, icons, …) in both dev and frozen builds.

When run from source, assets live under ``<repo>/assets/``. When packaged
with PyInstaller they're extracted into ``sys._MEIPASS``. ``resource_root``
returns whichever applies so call sites can write
``resource_root() / "logo" / "logo.svg"`` and not care which mode they're in.
"""

from __future__ import annotations

import sys
from pathlib import Path


def resource_root() -> Path:
    """Directory holding bundled non-Python assets (logo, icons, …).

    Assets live *inside the package* (``modelprep/gui/assets/``) so they ship in
    wheels/sdists. Under PyInstaller they're extracted to ``sys._MEIPASS`` (the
    spec bundles them at the runtime root), so call sites can always write
    ``resource_root() / "logo" / "logo.svg"``.
    """
    if getattr(sys, "frozen", False):
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            return Path(meipass)
    return Path(__file__).resolve().parent / "assets"


RUBRIC_URL = "https://github.com/timdewet/ModelPrep/blob/main/docs/labelling_rubric.md"


def rubric_location() -> str:
    """Where the labelling rubric is: the bundled/checkout copy if present
    (a local path), else its URL in the repository."""
    candidates = [resource_root() / "docs" / "labelling_rubric.md"]  # frozen bundle
    # Source checkout / editable install: <repo>/docs/labelling_rubric.md.
    candidates.append(Path(__file__).resolve().parents[3] / "docs" / "labelling_rubric.md")
    for c in candidates:
        if c.is_file():
            return str(c)
    return RUBRIC_URL
