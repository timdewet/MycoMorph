"""Qt application entry point.

``mycomorph --modelprep`` (``MycoMorph --modelprep`` in the desktop bundle)
opens ModelPrep, the cell-quality labeller and trainer, on its own instead of
MycoMorph's main window.
"""

from __future__ import annotations

import argparse
import os
import sys

from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import QApplication

# QtWebEngine requires either an early import or Qt.AA_ShareOpenGLContexts
# to be set before the QApplication is constructed. Set the attribute
# upfront so the embedded interactive plot in the Analysis page works
# regardless of import order downstream.
QApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts)

from ._resources import resource_root
from .ui.theme import apply_theme


def _parse_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    """Split MycoMorph's own flags from the rest of ``argv``.

    Unknown arguments (Qt's ``-platform …``, macOS's ``-psn_…``) are passed
    on to the QApplication untouched.
    """
    parser = argparse.ArgumentParser(
        prog="mycomorph", allow_abbrev=False,
        description="MycoMorph: bacterial microscopy pre-processing pipeline.",
    )
    parser.add_argument(
        "--modelprep", action="store_true",
        help="open ModelPrep (the cell-quality labeller and trainer) instead of MycoMorph",
    )
    args, rest = parser.parse_known_args(argv[1:])
    return args, argv[:1] + rest


def _configure_mycomorph(app: QApplication) -> None:
    app.setApplicationName("MycoMorph")
    app.setApplicationDisplayName("MycoMorph")
    # Ensure closing the last window actually exits the process. Qt's
    # default is to keep the event loop alive while any QObject is
    # still attached — QWebEngineView (used by the Analysis panel)
    # spawns helper processes that occasionally outlive ``win.close()``
    # and leave the terminal hanging on Windows.
    app.setQuitOnLastWindowClosed(True)
    logo = resource_root() / "logo" / "logo.svg"
    if logo.exists():
        app.setWindowIcon(QIcon(str(logo)))
    apply_theme(app)


def _app_parts(modelprep: bool):
    """(configure-app function, main-window class) for the app to open.

    Called before the QApplication exists: MycoMorph's Analysis page imports
    QtWebEngineWidgets, which wants to load first. ModelPrep's window doesn't
    use it, so ``--modelprep`` never loads it.
    """
    if modelprep:
        from mycomorph.modelprep.gui.app import configure_app
        from mycomorph.modelprep.gui.main_window import MainWindow as ModelPrepWindow
        return configure_app, ModelPrepWindow
    from .main_window import MainWindow
    return _configure_mycomorph, MainWindow


def main(argv: list[str] | None = None) -> int:
    args, qt_argv = _parse_args(list(sys.argv if argv is None else argv))
    configure_app, window_cls = _app_parts(args.modelprep)
    app = QApplication(qt_argv)
    configure_app(app)
    win = window_cls()
    win.show()
    if os.environ.get("MYCOMORPH_SMOKE_TEST") == "1":
        # Name the window that opened so a launch check can tell the two apart.
        print(f"[mycomorph] smoke test: opened {win.windowTitle()}", flush=True)
        QTimer.singleShot(500, app.quit)
    rc = app.exec()
    # Force-quit any lingering Qt resources. ``os._exit`` bypasses
    # Python's atexit cleanup — safe here because the GUI is the
    # only thing this entry-point manages, and it's the documented
    # escape hatch for Qt apps that won't release the terminal when
    # WebEngine helper processes outlive the main loop.
    os._exit(rc)


if __name__ == "__main__":
    sys.exit(main())
