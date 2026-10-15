"""Regenerate the app icons from the SVG logo.

    python packaging/make_icons.py

Writes ``logo.icns`` (macOS bundle; needs ``iconutil``, i.e. macOS) and
``logo.ico`` (Windows) next to ``logo.svg``. Requires PyQt6 (QtSvg) and Pillow.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

LOGO_DIR = Path(__file__).resolve().parents[1] / "src" / "modelprep" / "gui" / "assets" / "logo"
SVG = LOGO_DIR / "logo.svg"


def _render(size: int, out: Path) -> None:
    from PyQt6.QtCore import Qt
    from PyQt6.QtGui import QImage, QPainter
    from PyQt6.QtSvg import QSvgRenderer

    img = QImage(size, size, QImage.Format.Format_ARGB32)
    img.fill(Qt.GlobalColor.transparent)
    painter = QPainter(img)
    QSvgRenderer(str(SVG)).render(painter)
    painter.end()
    img.save(str(out))


def main() -> int:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtGui import QGuiApplication

    _app = QGuiApplication.instance() or QGuiApplication(sys.argv[:1])
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        big = tmp / "logo_1024.png"
        _render(1024, big)

        from PIL import Image
        Image.open(big).save(LOGO_DIR / "logo.ico",
                             sizes=[(s, s) for s in (16, 24, 32, 48, 64, 128, 256)])

        if shutil.which("iconutil"):
            iconset = tmp / "logo.iconset"
            iconset.mkdir()
            for s in (16, 32, 128, 256, 512):
                _render(s, iconset / f"icon_{s}x{s}.png")
                _render(2 * s, iconset / f"icon_{s}x{s}@2x.png")
            subprocess.run(["iconutil", "-c", "icns", str(iconset), "-o",
                            str(LOGO_DIR / "logo.icns")], check=True)
        else:
            print("iconutil not found (not macOS): skipped logo.icns")
    print(f"wrote icons in {LOGO_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
