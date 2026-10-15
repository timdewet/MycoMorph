"""Regenerate the app icons from the SVG logos.

    python packaging/make_icons.py               # both logos
    python packaging/make_icons.py mycomorph     # or just one: mycomorph | modelprep

Writes ``logo.ico`` (Windows) and ``logo.icns`` (macOS app bundle) next to each
``logo.svg``: MycoMorph's in ``assets/logo/`` and ModelPrep's in
``src/mycomorph/modelprep/gui/assets/logo/``. The ``.icns`` comes from macOS's
``iconutil`` when it is available, otherwise from Pillow. Requires PyQt6 (QtSvg)
and Pillow.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LOGO_DIRS = {
    "mycomorph": REPO / "assets" / "logo",
    "modelprep": REPO / "src" / "mycomorph" / "modelprep" / "gui" / "assets" / "logo",
}
ICNS_SIZES = (16, 32, 64, 128, 256, 512, 1024)


def _render(svg: Path, size: int, out: Path) -> None:
    from PyQt6.QtCore import Qt
    from PyQt6.QtGui import QImage, QPainter
    from PyQt6.QtSvg import QSvgRenderer

    img = QImage(size, size, QImage.Format.Format_ARGB32)
    img.fill(Qt.GlobalColor.transparent)
    painter = QPainter(img)
    QSvgRenderer(str(svg)).render(painter)
    painter.end()
    img.save(str(out))


def _write_icns(svg: Path, out: Path, tmp: Path) -> None:
    if shutil.which("iconutil"):
        iconset = tmp / "logo.iconset"
        iconset.mkdir()
        for s in (16, 32, 128, 256, 512):
            _render(svg, s, iconset / f"icon_{s}x{s}.png")
            _render(svg, 2 * s, iconset / f"icon_{s}x{s}@2x.png")
        subprocess.run(["iconutil", "-c", "icns", str(iconset), "-o", str(out)], check=True)
        return

    from PIL import Image

    # Render every size from the SVG (sharper than downscaling the largest).
    renders = {}
    for s in ICNS_SIZES:
        renders[s] = tmp / f"icns_{s}.png"
        _render(svg, s, renders[s])
    base = Image.open(renders[max(ICNS_SIZES)])
    others = [Image.open(p) for s, p in renders.items() if s != max(ICNS_SIZES)]
    base.save(out, format="ICNS", append_images=others)


def make_icons(logo_dir: Path) -> None:
    from PIL import Image

    svg = logo_dir / "logo.svg"
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        big = tmp / "logo_1024.png"
        _render(svg, 1024, big)
        Image.open(big).save(logo_dir / "logo.ico",
                             sizes=[(s, s) for s in (16, 24, 32, 48, 64, 128, 256)])
        _write_icns(svg, logo_dir / "logo.icns", tmp)
    print(f"wrote logo.ico and logo.icns in {logo_dir}")


def main(argv: list[str]) -> int:
    names = argv or list(LOGO_DIRS)
    unknown = [n for n in names if n not in LOGO_DIRS]
    if unknown:
        print(f"unknown logo(s) {unknown}; choose from {sorted(LOGO_DIRS)}", file=sys.stderr)
        return 2

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtGui import QGuiApplication

    _app = QGuiApplication.instance() or QGuiApplication(sys.argv[:1])
    for name in names:
        make_icons(LOGO_DIRS[name])
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
