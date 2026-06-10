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
