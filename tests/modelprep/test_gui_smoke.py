"""Offscreen GUI smoke tests.

Gated behind ``MODELPREP_GUI_TESTS=1`` because constructing a QApplication
aborts the process on a mis-configured local Qt install (see the project note
on QT_QPA_PLATFORM_PLUGIN_PATH). CI sets the env + an offscreen platform, where
PyQt6 loads cleanly; locally, run with::

    MODELPREP_GUI_TESTS=1 QT_QPA_PLATFORM=offscreen pytest tests/test_gui_smoke.py
"""

from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("MODELPREP_GUI_TESTS") != "1",
    reason="set MODELPREP_GUI_TESTS=1 (offscreen Qt) to run GUI smoke tests",
)


def _app():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def test_main_window_constructs_and_navigates():
    app = _app()
    from modelprep.gui.ui import theme
    from modelprep.gui.main_window import MainWindow

    theme.apply_theme(app)
    win = MainWindow()
    assert win._stack.count() == 3
    for i in range(win._stack.count()):
        win._sidebar.set_current(i)
        app.processEvents()
    win.label_train_panel.shutdown()


def test_panels_construct():
    app = _app()
    from modelprep.gui.panels.source_panel import SourcePanel
    from modelprep.gui.panels.label_train_panel import LabelTrainPanel

    sp = SourcePanel()
    assert sp is not None
    panel = LabelTrainPanel()
    # The two cards the main window pages over exist.
    assert panel.label_section is not None and panel.train_section is not None
    panel.shutdown()
    app.processEvents()
