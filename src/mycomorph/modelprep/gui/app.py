"""Qt application entry point for ModelPrep."""

from __future__ import annotations

import os
import sys

from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import QApplication

from ._resources import resource_root
from .main_window import MainWindow
from .ui import theme


def _ensure_qt_plugin_path() -> None:
    """Point Qt at PyQt6's own bundled platform plugins.

    Some mixed conda environments (a pip PyQt6 wheel sitting beside a conda Qt5)
    mis-resolve Qt's plugin path, so the app aborts with
    ``Could not find the Qt platform plugin "cocoa" in ""``. Deriving the path
    from the installed PyQt6 wheel makes ``modelprep`` launch regardless of the
    environment. ``setdefault`` respects an explicit override (e.g. CI's
    offscreen path), and this must run before ``QApplication`` is constructed
    (when the platform plugin is loaded).
    """
    try:
        import PyQt6
    except Exception:  # noqa: BLE001 — defensive; nothing to do without PyQt6
        return
    from pathlib import Path

    plugins = Path(PyQt6.__file__).resolve().parent / "Qt6" / "plugins"
    if (plugins / "platforms").is_dir():
        os.environ.setdefault("QT_PLUGIN_PATH", str(plugins))
        os.environ.setdefault("QT_QPA_PLATFORM_PLUGIN_PATH", str(plugins / "platforms"))


def main() -> int:
    _ensure_qt_plugin_path()
    app = QApplication(sys.argv)
    app.setApplicationName("ModelPrep")
    app.setApplicationDisplayName("ModelPrep")
    app.setQuitOnLastWindowClosed(True)

    logo = resource_root() / "logo" / "logo.svg"
    if logo.exists():
        app.setWindowIcon(QIcon(str(logo)))

    # Install the token-driven theme (shared visual language with MycoMorph)
    # and hook the system light/dark follower.
    theme.apply_theme(app)

    win = MainWindow()
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
