# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for MycoMorph and ModelPrep, its labeller / trainer.

Build a desktop bundle:

    pip install -c constraints/release-py312.txt -e . pyinstaller==6.16.0
    pyinstaller packaging/mycomorph.spec

Produces ``dist/MycoMorph/`` (Windows, Linux) or ``dist/MycoMorph.app``
(macOS) containing the MycoMorph executable and supporting DLLs / data
files. Zip and distribute the whole folder or app. The first launch
downloads cellpose model weights to the user's ``~/.cellpose`` directory
(a few hundred MB) — needs internet on first run only.

One executable runs both apps: ``MycoMorph`` opens MycoMorph and
``MycoMorph --modelprep`` opens ModelPrep (see ``mycomorph/gui/app.py``).

Licensing: MycoMorph's source is MIT, but the bundle contains PyQt6
(GPL v3) and Qt (LGPL v3), so the bundle as a whole is distributed under
the GPL v3. ``packaging/BUNDLE_LICENSE.txt`` says so and ships in the
bundle with ``LICENSE`` and the PyQt6 / Qt licence texts.

PyInstaller does NOT cross-compile. To produce a Windows .exe, run the
above command on a Windows machine; for the macOS app, on a Mac.
"""

import re
import sys
from importlib.metadata import distribution
from importlib.util import find_spec
from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_submodules


def _version() -> str:
    """``mycomorph.__version__``, read without importing the package."""
    init = Path(SPECPATH).parent / "src" / "mycomorph" / "__init__.py"
    m = re.search(r'^__version__ = "([^"]+)"', init.read_text(encoding="utf-8"), re.M)
    if m is None:
        raise RuntimeError(f"__version__ not found in {init}")
    return m.group(1)


def _licence_file(dist_name: str) -> str:
    """Path of the licence text an installed distribution ships."""
    dist = distribution(dist_name)
    for f in dist.files or []:
        if f.name in ("LICENSE", "LICENSE.txt", "COPYING"):
            return str(dist.locate_file(f))
    raise RuntimeError(f"No licence file found in the {dist_name} distribution")


datas: list = []
binaries: list = []
# QtNetwork drives the in-app GitHub update check; force-include it so
# the PyQt6 hook bundles the QtNetwork DLL and OpenSSL backend even
# when the import graph misses dynamic uses.
hiddenimports: list = ["PyQt6.QtNetwork"]

# Project assets — bundled at the runtime root so ``resource_root()``
# (in ``mycomorph/gui/_resources.py``) finds them.
datas.append(("../assets/logo/logo.svg", "logo"))
datas.append(("../assets/logo/logo.ico", "logo"))

# Licences (see the docstring): MycoMorph's MIT licence, the bundle's GPL
# notice, and the GPL v3 / LGPL v3 texts it points to.
datas.append(("../LICENSE", "."))
datas.append(("BUNDLE_LICENSE.txt", "."))
datas.append((_licence_file("PyQt6"), "licenses/PyQt6"))
datas.append((_licence_file("PyQt6-Qt6"), "licenses/Qt6"))

# ModelPrep's labelling rubric, linked from its Label page
# (``mycomorph/modelprep/gui/_resources.py``). ModelPrep's logo and icons are
# package data, collected with ``mycomorph`` below.
datas.append(("../docs/modelprep/labelling_rubric.md", "docs/modelprep"))

# ``collect_all`` walks each package's files (Python modules, package
# data, DLLs) so the bundle contains everything ``mycomorph.*`` imports
# at runtime.
for pkg in (
    "mycomorph",
    "cellpose",
    "torch",
    "torchvision",
    "pylibCZIrw",
    "tifffile",
    "skimage",
    "h5py",
    "pyqtgraph",
):
    if find_spec(pkg) is None:
        raise RuntimeError(f"Required package is unavailable: {pkg}")
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h

for pkg in ("skimage", "scipy", "mycomorph"):
    hiddenimports += collect_submodules(pkg)

excludes = [
    "tkinter",
    "matplotlib.tests",
    "numpy.tests",
    "pandas.tests",
    "scipy.tests",
    "skimage.tests",
    "pytest",
    "_pytest",
    "IPython",
    "notebook",
    "jupyter",
    "sphinx",
    "test",
]


a = Analysis(
    ["../src/mycomorph/__main__.py"],
    pathex=["../src"],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
)


pyz = PYZ(a.pure, a.zipped_data)


# A windowed (no console) executable. The icons are generated from
# ``assets/logo/logo.svg`` by ``packaging/make_icons.py``: a multi-resolution
# ICO (16–256 px) for the Windows .exe and an ICNS for the macOS app.
ICON = "../assets/logo/logo.icns" if sys.platform == "darwin" else "../assets/logo/logo.ico"

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="MycoMorph",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=ICON,
)


coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="MycoMorph",
)

# Windows / Linux: PyInstaller puts every data file in _internal/, so also put
# the licence notice and MycoMorph's licence next to the executable, where
# people look. (The macOS app keeps them in Contents/Resources/.)
if sys.platform != "darwin":
    import shutil

    top = Path(DISTPATH) / "MycoMorph"
    shutil.copy2(Path(SPECPATH) / "BUNDLE_LICENSE.txt", top)
    shutil.copy2(Path(SPECPATH).parent / "LICENSE", top)


# macOS: wrap the folder in MycoMorph.app, so users get an app rather than a
# folder holding a Unix executable. Not signed or notarised.
if sys.platform == "darwin":
    VERSION = _version()
    app = BUNDLE(
        coll,
        name="MycoMorph.app",
        icon=ICON,
        bundle_identifier="io.github.timdewet.mycomorph",
        version=VERSION,
        info_plist={
            "CFBundleName": "MycoMorph",
            "CFBundleDisplayName": "MycoMorph",
            "CFBundleShortVersionString": VERSION,
            "CFBundleVersion": VERSION,
            "NSHighResolutionCapable": True,
            "NSHumanReadableCopyright": (
                "© 2026 Timothy de Wet. MycoMorph's source code is MIT-licensed; "
                "this app includes PyQt6 and is distributed under the GNU GPL v3 "
                "(see BUNDLE_LICENSE.txt)."
            ),
        },
    )
