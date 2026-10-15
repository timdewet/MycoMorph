"""The Classify page's "Labeller / trainer" opens ModelPrep's window in
MycoMorph's process (decision D3): one instance, opened on the run's crops,
closed with MycoMorph, and a warning when training and a pipeline run would
share the GPU."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("PyQt6")

# Imported at collection, before pytest-qt creates the QApplication: MycoMorph's
# Analysis page imports QtWebEngineWidgets, which must load first.
from mycomorph.gui import main_window as mw  # noqa: E402


def _write_crops_h5(path: Path, n: int = 8) -> Path:
    """A tiny ``all_crops.h5`` in the schema the Features stage writes."""
    import h5py

    path.parent.mkdir(parents=True, exist_ok=True)
    crops = np.random.default_rng(0).random((n, 3, 32, 32), dtype=np.float32)
    crops[:, 2] = 0.0
    crops[:, 2, 8:24, 8:24] = 1.0
    str_dt = h5py.string_dtype()
    with h5py.File(str(path), "w") as f:
        f.create_dataset("crops", data=crops)
        f.create_dataset("cell_uids", data=[f"cell_{i:04d}" for i in range(n)], dtype=str_dt)
        f.create_dataset("areas_px", data=np.full(n, 256, dtype=np.int32))
        for key, value in (("condition_labels", "cond"), ("conditions", "c0"),
                           ("reporters", "rpt"), ("genes", ""), ("drugs", ""),
                           ("tiff_files", "fake.tif")):
            f.create_dataset(key, data=[value] * n, dtype=str_dt)
        f.create_dataset("fov_indices", data=np.zeros(n, dtype=np.int32))
        f.create_dataset("cell_labels", data=np.arange(1, n + 1, dtype=np.int32))
        f.attrs["channel_names"] = ["Phase", "Fluor", "Mask"]
        f.attrs["schema_version"] = "1.0"
    return path


@pytest.fixture
def home(tmp_path, monkeypatch) -> Path:
    """Temp HOME: ModelPrep's settings and runs (``~/.modelprep``) and
    MycoMorph's QSettings never touch the user's own."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(home / ".config"))
    return home


@pytest.fixture
def window(qtbot, home, tmp_path, monkeypatch):
    from PyQt6.QtCore import QSettings

    ini = tmp_path / "mycomorph.ini"
    monkeypatch.setattr(mw.MainWindow, "_settings",
                        lambda self: QSettings(str(ini), QSettings.Format.IniFormat))
    win = mw.MainWindow()
    yield win
    # Close here, while HOME and _settings are still patched (closing saves
    # MycoMorph's settings and closes ModelPrep's window), not in qtbot's
    # teardown, which runs after they are restored.
    win.close()
    win.deleteLater()
    _flush_deletes()


def _set_output_dir(win, out: Path, monkeypatch) -> None:
    from mycomorph.gui.panels import input_panel

    out.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(input_panel.QFileDialog, "getExistingDirectory",
                        staticmethod(lambda *a, **k: str(out)))
    win.input_panel._pick_out()


def _flush_deletes() -> None:
    from PyQt6.QtCore import QCoreApplication, QEvent

    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete.value)


def test_labeller_button_opens_one_modelprep_window_on_the_runs_crops(
        window, home, tmp_path, monkeypatch):
    from mycomorph.modelprep.gui.main_window import MainWindow as ModelPrepWindow
    from mycomorph.modelprep.gui.main_window import settings_file

    h5 = _write_crops_h5(tmp_path / "run" / "04_features" / "all_crops.h5")
    _set_output_dir(window, tmp_path / "run", monkeypatch)

    window.classify_panel._open_labeler_btn.click()
    mp = window._modelprep_window
    assert isinstance(mp, ModelPrepWindow)
    assert mp.isVisible()
    assert mp.source_panel.current_h5_path() == h5
    assert settings_file().is_relative_to(home)

    # A second click raises the same window.
    window.classify_panel._open_labeler_btn.click()
    assert window._modelprep_window is mp


def test_open_window_keeps_its_dataset(window, tmp_path, monkeypatch):
    # No crops yet: ModelPrep opens with no source.
    _set_output_dir(window, tmp_path / "a", monkeypatch)
    window._open_label_trainer()
    mp = window._modelprep_window
    assert mp.source_panel.current_h5_path() is None

    # Features has run since: raising the open window picks the crops up.
    h5_a = _write_crops_h5(tmp_path / "a" / "04_features" / "all_crops.h5")
    window._open_label_trainer()
    assert mp.source_panel.current_h5_path() == h5_a

    # Another output folder doesn't swap out the dataset being labelled.
    _write_crops_h5(tmp_path / "b" / "04_features" / "all_crops.h5")
    _set_output_dir(window, tmp_path / "b", monkeypatch)
    window._open_label_trainer()
    assert window._modelprep_window is mp
    assert mp.source_panel.current_h5_path() == h5_a


def test_a_closed_modelprep_window_is_rebuilt(window):
    from PyQt6 import sip

    window._open_label_trainer()
    first = window._modelprep_window
    assert first.close()
    assert window._live_modelprep_window() is None
    _flush_deletes()
    assert sip.isdeleted(first)                      # deleted on close

    window._open_label_trainer()
    second = window._modelprep_window
    assert second is not None and second is not first
    assert second.isVisible()


def test_closing_mycomorph_closes_modelprep_and_respects_cancel(window, monkeypatch):
    from mycomorph.modelprep.gui import main_window as mp_main

    window.show()
    window._open_label_trainer()
    mp = window._modelprep_window
    shutdowns: list[str] = []
    real_shutdown = mp.label_train_panel.shutdown
    monkeypatch.setattr(mp.label_train_panel, "has_unsaved_labels", lambda: True)
    monkeypatch.setattr(mp.label_train_panel, "shutdown",
                        lambda: (shutdowns.append("label_train"), real_shutdown()))

    # Cancel on ModelPrep's unsaved-labels prompt keeps both windows open.
    answer = mp_main.QMessageBox.StandardButton.Cancel
    monkeypatch.setattr(mp_main.QMessageBox, "question", staticmethod(lambda *a, **k: answer))
    assert not window.close()
    assert window.isVisible() and mp.isVisible()
    assert window._modelprep_window is mp
    assert shutdowns == []

    # Discard: ModelPrep shuts its workers down, then MycoMorph closes.
    answer = mp_main.QMessageBox.StandardButton.Discard
    assert window.close()
    assert shutdowns == ["label_train"]
    assert not mp.isVisible()
    assert window._modelprep_window is None
    assert not window.isVisible()


class _FakeTrainWorker:
    """Stands in for ModelPrep's training thread: nothing is trained."""

    def __init__(self, **_kwargs):
        from PyQt6.QtCore import QObject, pyqtSignal

        class _Signals(QObject):
            progress = pyqtSignal(float, str)
            finished = pyqtSignal(object, str)

        self._signals = _Signals()
        self.progress = self._signals.progress
        self.finished = self._signals.finished

    def start(self):
        pass

    def isRunning(self):
        return False


def test_training_during_a_pipeline_run_warns_without_blocking(
        window, tmp_path, monkeypatch):
    from PyQt6.QtCore import Qt

    from mycomorph.modelprep.gui.panels import label_train_panel as ltp

    monkeypatch.setattr(ltp, "_TrainWorker", _FakeTrainWorker)
    h5 = _write_crops_h5(tmp_path / "run" / "04_features" / "all_crops.h5")
    _set_output_dir(window, tmp_path / "run", monkeypatch)
    window._open_label_trainer()
    panel = window._modelprep_window.label_train_panel
    assert panel._h5_path == h5

    # No pipeline run: training starts quietly.
    panel._start_training()
    assert window._gpu_share_warning is None

    window._run_active = True
    panel._start_training()
    box = window._gpu_share_warning
    assert box is not None and box.isVisible()
    assert box.windowModality() == Qt.WindowModality.NonModal
    assert "GPU" in box.text()
    box.close()
    window._run_active = False
