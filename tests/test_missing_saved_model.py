"""Saved settings that name a model this version no longer offers (the ``mtb``
classifier preset after v0.6.0, an unknown Cellpose model) are flagged on the
Segment & Classify tab and block runs until the user picks one. They used to
fall back silently: to rules-only classification, or to cpsam.

These build the panels on a plain QApplication, so they run without pytest-qt.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("PyQt6")

# See test_modelprep_window.py: MycoMorph's window module pulls in
# QtWebEngineWidgets, which must load before any QApplication exists.
os.environ.setdefault("QTWEBENGINE_DISABLE_SANDBOX", "1")
from mycomorph.gui import main_window as mw  # noqa: E402


@pytest.fixture(scope="module")
def app():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


@pytest.fixture
def home(tmp_path, monkeypatch) -> Path:
    """A temp HOME, so the Classify panel never lists the user's trained runs."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    return home


def _classify(state: dict):
    from mycomorph.gui.panels.stage_panels import ClassifyPanel

    panel = ClassifyPanel()
    panel.restore_state(json.loads(json.dumps(state)))   # as MainWindow stores it
    return panel


def _segment(state: dict):
    from mycomorph.gui.panels.stage_panels import SegmentPanel

    panel = SegmentPanel()
    panel.restore_state(json.loads(json.dumps(state)))
    return panel


def _pick(combo, text: str) -> None:
    """What a user's pick does: change the index, then ``activated``."""
    i = combo.findText(text)
    assert i >= 0, text
    combo.setCurrentIndex(i)
    combo.activated.emit(i)


# ── Classify ────────────────────────────────────────────────────────────


def test_saved_preset_that_is_gone_is_flagged_not_replaced(app, home):
    panel = _classify({"preset": "mtb", "use_rules": True})

    assert panel.missing_saved_preset() == "mtb"
    assert not panel._preset_warning.isHidden()
    assert "'mtb'" in panel._preset_warning.text()
    # No stand-in model: mtb-2026-06 is a different classifier.
    assert panel.preset.currentText() == "none (rules only)"
    assert panel.opts().model_path is None
    # Closing without choosing keeps the warning for next launch.
    assert panel.state()["preset"] == "mtb"
    assert _classify(panel.state()).missing_saved_preset() == "mtb"


@pytest.mark.parametrize("choice", ["none (rules only)", "mtb-2026-06", "custom…"])
def test_any_pick_clears_it(app, home, choice):
    panel = _classify({"preset": "mtb"})
    _pick(panel.preset, choice)

    assert panel.missing_saved_preset() is None
    assert panel._preset_warning.isHidden()
    assert panel.state()["preset"] == choice


def test_picking_a_trained_run_clears_it(app, home):
    d = home / ".modelprep" / "models" / "mtb-v2_2026-10-14_09-00-00"
    d.mkdir(parents=True)
    (d / "best_model.pth").write_bytes(b"x")
    (d / "training_config.json").write_text("{}", encoding="utf-8")

    panel = _classify({"preset": "mtb"})
    panel.trained_run.setCurrentIndex(1)

    assert panel.missing_saved_preset() is None
    assert panel.opts().model_path == d / "best_model.pth"


def test_bundled_presets_restore_without_a_warning(app, home):
    from mycomorph.core.api import PRESET_MODELS

    for name, path in PRESET_MODELS.items():
        panel = _classify({"preset": name})
        assert panel.missing_saved_preset() is None, name
        assert panel._preset_warning.isHidden()
        assert panel.opts().model_path == path

    panel = _classify({"preset": "mtb"})
    panel.restore_state({"preset": "msm-2026-06"})       # restored again, now valid
    assert panel.missing_saved_preset() is None


# ── Segment ─────────────────────────────────────────────────────────────


def test_saved_cellpose_model_that_is_gone_is_flagged(app):
    panel = _segment({"model": "cpsam_v9", "diameter": 30.0})

    assert panel.missing_saved_model() == "cpsam_v9"
    assert not panel._missing_model_warning.isHidden()
    assert "'cpsam_v9'" in panel._missing_model_warning.text()
    assert panel.diameter.value() == 30.0                   # the rest still restores
    assert panel.state()["model"] == "cpsam_v9"

    _pick(panel.model, "cpsam (default)")                   # already shown; still a choice
    assert panel.missing_saved_model() is None
    assert panel._missing_model_warning.isHidden()
    assert panel.state()["model"] == "cpsam"


@pytest.mark.parametrize("state", [
    {"model": "cpsam_v2"},
    {"model": "cpsam"},
    {"model_type": "cyto3"},     # Cellpose 3 builds: ignored, as before
    {},
])
def test_known_or_absent_cellpose_model_is_not_flagged(app, state):
    panel = _segment(state)
    assert panel.missing_saved_model() is None
    assert panel._missing_model_warning.isHidden()


# ── Run preflight ───────────────────────────────────────────────────────


def _preflight(monkeypatch, tmp_path, enables, segment, classify):
    """Run MainWindow._preflight_ok against the two panels; return its
    result and the "Cannot run" text, if any."""
    out = tmp_path / "out"
    (out / "02_segment").mkdir(parents=True, exist_ok=True)
    (out / "02_segment" / "A1.tif").write_bytes(b"")       # so Classify alone has input
    shown: list[str] = []
    monkeypatch.setattr(mw.QMessageBox, "critical",
                        staticmethod(lambda _parent, _title, text: shown.append(text)))
    monkeypatch.setattr(mw.QMessageBox, "question", staticmethod(
        lambda *a, **k: pytest.fail(f"unexpected warning prompt: {a[2]}")))
    window = SimpleNamespace(
        segment_panel=segment,
        classify_panel=classify,
        run_panel=SimpleNamespace(reuse_existing=SimpleNamespace(isChecked=lambda: True)),
        input_panel=SimpleNamespace(mode=mw.InputMode.SINGLE_PLATE),
    )
    ok = mw.MainWindow._preflight_ok(window, out, enables)
    return ok, "\n\n".join(shown)


def test_run_with_classify_waits_for_a_classifier(app, home, monkeypatch, tmp_path):
    segment, classify = _segment({}), _classify({"preset": "mtb"})

    ok, text = _preflight(monkeypatch, tmp_path, {"Classify": True}, segment, classify)
    assert not ok
    assert "'mtb'" in text and "Choose a classifier" in text

    # Runs that skip Classify aren't held up.
    ok, _ = _preflight(monkeypatch, tmp_path, {"Features": True}, segment, classify)
    assert ok

    _pick(classify.preset, "mtb-2026-06")
    ok, _ = _preflight(monkeypatch, tmp_path, {"Classify": True}, segment, classify)
    assert ok


def test_run_with_segment_waits_for_a_cellpose_model(app, home, monkeypatch, tmp_path):
    segment, classify = _segment({"model": "cpsam_v9", "gpu": False}), _classify({})
    enables = {"Focus": True, "Segment": True}

    ok, text = _preflight(monkeypatch, tmp_path, enables, segment, classify)
    assert not ok
    assert "'cpsam_v9'" in text and "Choose a model" in text

    _pick(segment.model, "cpsam (default)")
    ok, _ = _preflight(monkeypatch, tmp_path, enables, segment, classify)
    assert ok
