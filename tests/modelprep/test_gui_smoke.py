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
    assert win._stack.count() == 4  # source, label, train, models
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


def _make_run(root, name, ts, **cfg):
    import json
    d = root / f"{name}_{ts}"
    d.mkdir(parents=True)
    (d / "best_model.pth").write_bytes(b"x")
    base = dict(eval_split="holdout", eval_acc=0.8, balanced_acc=0.78, roc_auc=0.85,
                average_precision=0.83, majority_baseline=0.7, decision_threshold=0.5,
                n_train=600, n_holdout=200, class_counts={"good": 200, "bad": 400})
    base.update(cfg)
    (d / "training_config.json").write_text(json.dumps(base))
    return d


def test_models_panel_lists_sorts_and_deletes(tmp_path, monkeypatch):
    app = _app()
    import modelprep.gui.trained_models as tm
    import modelprep.gui.panels.models_panel as mp
    from modelprep.gui.panels.models_panel import _COL
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import QMessageBox

    root = tmp_path / "models"
    root.mkdir()
    monkeypatch.setattr(tm, "trained_models_root", lambda: root)
    monkeypatch.setattr(mp, "trained_models_root", lambda: root)
    _make_run(root, "alpha", "2026-06-11_00-00-00", roc_auc=0.74)
    _make_run(root, "beta", "2026-06-11_01-00-00", roc_auc=0.91)

    panel = mp.ModelsPanel()
    assert panel._table.rowCount() == 2

    # Numeric sort by AUC ascending puts the weaker model first.
    panel._table.sortItems(_COL["AUC"], Qt.SortOrder.AscendingOrder)
    names = [panel._table.item(r, _COL["Name"]).text() for r in range(2)]
    assert names == ["alpha", "beta"]

    # Delete (auto-confirm) removes the run from disk and the table.
    monkeypatch.setattr(QMessageBox, "question",
                        staticmethod(lambda *a, **k: QMessageBox.StandardButton.Yes))
    changed = []
    panel.modelsChanged.connect(lambda: changed.append(1))
    panel._select_dir(str(root / "beta_2026-06-11_01-00-00"))
    panel._delete_selected()
    assert not (root / "beta_2026-06-11_01-00-00").exists()
    assert panel._table.rowCount() == 1
    assert changed == [1]
    app.processEvents()
