"""The Classify page lists runs trained in ModelPrep (``~/.modelprep/models``)
by name, CNN runs only, next to Browse…, and its settings round-trip a run."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("PyQt6")


@pytest.fixture
def home(tmp_path, monkeypatch) -> Path:
    """A temp HOME, so ``modelprep_home()`` never reads the user's runs."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    return home


def _run(home: Path, dirname: str, artefact: str, **cfg) -> Path:
    d = home / ".modelprep" / "models" / dirname
    d.mkdir(parents=True)
    (d / artefact).write_bytes(b"x")
    (d / "training_config.json").write_text(json.dumps(cfg), encoding="utf-8")
    return d


def _items(combo) -> list[tuple[str, object]]:
    return [(combo.itemText(i), combo.itemData(i)) for i in range(combo.count())]


def test_lists_cnn_runs_by_name_and_skips_feature_runs(qtbot, home):
    from mycomorph.gui.panels.stage_panels import ClassifyPanel
    from mycomorph.modelprep.gui.trained_models import list_trained_models

    cnn = _run(home, "mtb-v2_2026-10-14_09-00-00", "best_model.pth",
               eval_acc=0.91, decision_threshold=0.42,
               provenance={"model_name": "mtb-v2"})
    _run(home, "mtb-v2-features_2026-10-14_09-00-00", "feature_model.joblib",
         model_type="features", eval_acc=0.88)

    panel = ClassifyPanel()
    qtbot.addWidget(panel)
    by_dir = {m.dir: m for m in list_trained_models()}
    assert _items(panel.trained_run) == [
        ("(pick a run trained in ModelPrep)", None),
        (by_dir[cnn].label(), str(cnn / "best_model.pth")),
    ]
    assert panel.trained_run.itemText(1).startswith("mtb-v2 · 2026-10-14_09-00-00")

    # Picking it selects custom… and that run's weights.
    panel.trained_run.setCurrentIndex(1)
    assert panel.preset.currentText() == "custom…"
    assert panel.custom_path.text() == str(cnn / "best_model.pth")
    assert panel.opts().model_path == cnn / "best_model.pth"
    assert panel.inspector._header.text() == f"Model: {cnn.name}"
    assert "0.420" in panel.use_model_threshold.text()   # its tuned threshold is offered

    # The placeholder is not a way to drop the model.
    panel.trained_run.setCurrentIndex(0)
    assert panel.opts().model_path == cnn / "best_model.pth"
    assert panel.trained_run.currentIndex() == 1


def test_no_runs_yet(qtbot, home):
    from mycomorph.gui.panels.stage_panels import ClassifyPanel

    panel = ClassifyPanel()
    qtbot.addWidget(panel)
    assert _items(panel.trained_run) == [("(no trained runs in ~/.modelprep/models yet)", None)]
    assert panel.opts().model_path is None


def test_settings_round_trip_a_trained_run(qtbot, home):
    from mycomorph.gui.panels.stage_panels import ClassifyPanel

    cnn = _run(home, "mtb-v2_2026-10-14_09-00-00", "best_model.pth",
               decision_threshold=0.42)
    _run(home, "older_2026-10-01_09-00-00", "best_model.pth")

    panel = ClassifyPanel()
    qtbot.addWidget(panel)
    panel.trained_run.setCurrentIndex(panel.trained_run.findData(str(cnn / "best_model.pth")))
    panel.use_model_threshold.setChecked(True)
    saved = json.loads(json.dumps(panel.state()))          # as MainWindow stores it

    restored = ClassifyPanel()
    qtbot.addWidget(restored)
    restored.restore_state(saved)
    assert restored.trained_run.currentData() == str(cnn / "best_model.pth")
    assert restored.opts().model_path == cnn / "best_model.pth"
    assert restored.opts().decision_threshold == pytest.approx(0.42)

    # Another preset hides the run (its path is kept for custom… again).
    restored.preset.setCurrentIndex(0)
    assert restored.trained_run.currentIndex() == 0
    restored.preset.setCurrentText("custom…")
    assert restored.trained_run.currentData() == str(cnn / "best_model.pth")


def test_browse_still_takes_any_pth(qtbot, home, tmp_path, monkeypatch):
    from mycomorph.gui.panels import stage_panels
    from mycomorph.gui.panels.stage_panels import ClassifyPanel

    cnn = _run(home, "mtb-v2_2026-10-14_09-00-00", "best_model.pth")
    elsewhere = tmp_path / "shared" / "lab_model.pth"
    elsewhere.parent.mkdir()
    elsewhere.write_bytes(b"x")
    monkeypatch.setattr(stage_panels.QFileDialog, "getOpenFileName",
                        staticmethod(lambda *a, **k: (str(elsewhere), "")))

    panel = ClassifyPanel()
    qtbot.addWidget(panel)
    panel.trained_run.setCurrentIndex(1)
    panel._pick_custom()
    assert panel.custom_path.text() == str(elsewhere)
    assert panel.trained_run.currentIndex() == 0            # not one of the listed runs
    assert panel.opts().model_path == elsewhere

    # Browsing to a run's own weights selects that run.
    monkeypatch.setattr(stage_panels.QFileDialog, "getOpenFileName",
                        staticmethod(lambda *a, **k: (str(cnn / "best_model.pth"), "")))
    panel._pick_custom()
    assert panel.trained_run.currentIndex() == 1


def test_runs_trained_later_appear_when_the_list_opens(qtbot, home):
    from mycomorph.gui.panels.stage_panels import ClassifyPanel

    first = _run(home, "a_2026-10-14_09-00-00", "best_model.pth")
    panel = ClassifyPanel()
    qtbot.addWidget(panel)
    panel.trained_run.setCurrentIndex(1)
    assert panel.trained_run.count() == 2

    later = _run(home, "b_2026-10-15_09-00-00", "best_model.pth")   # trained meanwhile
    panel.trained_run.showPopup()
    panel.trained_run.hidePopup()
    assert panel.trained_run.findData(str(later / "best_model.pth")) == 1   # listed
    assert panel.trained_run.currentData() == str(first / "best_model.pth")
