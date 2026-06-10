"""Qt application entry point for ModelPrep."""

from __future__ import annotations

import sys

from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import QApplication

from ._resources import resource_root
from .main_window import MainWindow
from .ui import theme


def main() -> int:
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
