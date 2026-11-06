"""The CNN embeddings stage is hidden from the app unless MYCOMORPH_EMBEDDINGS=1
(``gui/features.py``): no Run-panel toggle or stepper step, no "Train
embeddings" input mode, and saved settings can't switch either back on."""

from __future__ import annotations

import pytest

pytest.importorskip("PyQt6")

from mycomorph.gui.features import EMBEDDINGS_ENV, embeddings_enabled  # noqa: E402


def test_switch_reads_the_environment(monkeypatch):
    monkeypatch.delenv(EMBEDDINGS_ENV, raising=False)
    assert not embeddings_enabled()
    monkeypatch.setenv(EMBEDDINGS_ENV, "1")
    assert embeddings_enabled()
    monkeypatch.setenv(EMBEDDINGS_ENV, "0")
    assert not embeddings_enabled()


def test_run_panel_hides_the_stage_by_default(qtbot, monkeypatch):
    monkeypatch.delenv(EMBEDDINGS_ENV, raising=False)
    from mycomorph.gui.panels.run_panel import RunPanel

    panel = RunPanel()
    qtbot.addWidget(panel)
    cb = panel.checks["Embeddings"]
    assert cb.isHidden() and not cb.isChecked()
    assert panel.stepper._steps["Embeddings"].isHidden()
    # Settings saved by an older build can't tick it back on.
    panel.restore_state({"stage_enables": {"Features": True, "Embeddings": True}})
    assert panel.state()["stage_enables"]["Features"]
    assert not panel.state()["stage_enables"]["Embeddings"]
    # MainWindow calls this on every input-mode change; leaving train-only
    # mode used to re-show every stage.
    panel.set_train_only_mode(False)
    assert cb.isHidden() and not cb.isChecked()
    assert panel.stepper._steps["Embeddings"].isHidden()


def test_run_panel_shows_the_stage_when_enabled(qtbot, monkeypatch):
    monkeypatch.setenv(EMBEDDINGS_ENV, "1")
    from mycomorph.gui.panels.run_panel import RunPanel

    panel = RunPanel()
    qtbot.addWidget(panel)
    assert not panel.checks["Embeddings"].isHidden()
    assert not panel.stepper._steps["Embeddings"].isHidden()
    panel.restore_state({"stage_enables": {"Features": True, "Embeddings": True}})
    assert panel.state()["stage_enables"]["Embeddings"]


def test_input_panel_hides_train_mode_by_default(qtbot, monkeypatch):
    monkeypatch.delenv(EMBEDDINGS_ENV, raising=False)
    from mycomorph.gui.panels.input_panel import InputMode, InputPanel

    panel = InputPanel()
    qtbot.addWidget(panel)
    assert panel._radio_train_embeddings.isHidden()
    assert panel._radio_single_plate.objectName() == "segRight"
    panel.restore_state({"mode": "train_embeddings"})
    assert panel.mode is InputMode.BULK


def test_input_panel_keeps_train_mode_when_enabled(qtbot, monkeypatch):
    monkeypatch.setenv(EMBEDDINGS_ENV, "1")
    from mycomorph.gui.panels.input_panel import InputMode, InputPanel

    panel = InputPanel()
    qtbot.addWidget(panel)
    assert not panel._radio_train_embeddings.isHidden()
    panel.restore_state({"mode": "train_embeddings"})
    assert panel.mode is InputMode.TRAIN_EMBEDDINGS
