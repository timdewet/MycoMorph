from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtCore import QCoreApplication, QEventLoop, QObject, QTimer, Qt, pyqtSignal

QCoreApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts)

from mycomorph.gui.pipeline.runner import PipelineRunner


class _Layout:
    def to_csv(self, path: Path) -> None:
        Path(path).write_text("well,condition\n")


class _SlowStage:
    name = "Slow"
    handles_own_reuse = True

    def enabled(self, _ctx) -> bool:
        return True

    def validate(self, _ctx) -> list[str]:
        return []

    def output_dir(self, ctx) -> Path:
        return ctx.output_dir / "slow"

    def run(self, _ctx, progress_cb):
        for i in range(500):
            progress_cb(i / 500, "working")
            time.sleep(0.002)
        return []


class _StopEmitter(QObject):
    stop = pyqtSignal()


def test_runner_cancels_while_worker_event_loop_is_busy(tmp_path: Path, monkeypatch):
    app = QCoreApplication.instance() or QCoreApplication([])
    ctx = SimpleNamespace(
        output_dir=tmp_path,
        manifest_path=tmp_path / "manifest.json",
        czi_path=tmp_path / "input.czi",
        layout=_Layout(),
        focus_dir=tmp_path / "focus",
        split_dir=tmp_path / "split",
        segment_dir=tmp_path / "segment",
        classify_dir=tmp_path / "classify",
        do_split=False,
        focus_opts=SimpleNamespace(filename_suffix=""),
        segment_opts=None,
        classify_opts=None,
        features_opts=None,
    )
    runner = PipelineRunner(ctx, stages=[_SlowStage()])
    emitter = _StopEmitter()
    emitter.stop.connect(
        runner.request_stop,
        type=Qt.ConnectionType.DirectConnection,
    )
    monkeypatch.setattr(runner, "_migrate_legacy_dirs", lambda: None)
    monkeypatch.setattr(runner, "_rename_outputs_for_layout_changes", lambda _cb: None)

    loop = QEventLoop()
    cancelled: list[Path] = []
    failures: list[str] = []
    runner.runCancelled.connect(cancelled.append)
    runner.runFailed.connect(failures.append)
    runner.workerStopped.connect(loop.quit)
    QTimer.singleShot(5000, loop.quit)
    runner.start()
    QTimer.singleShot(50, emitter.stop.emit)
    loop.exec()
    app.processEvents()

    assert not failures, failures
    assert runner._stop_requested
    assert cancelled == [ctx.manifest_path]
    assert ctx.manifest_path.exists()


def test_close_event_requests_stop_and_defers_close(monkeypatch):
    from mycomorph.gui.main_window import MainWindow, QMessageBox

    calls: list[str] = []
    fake = SimpleNamespace(
        _run_active=True,
        _close_when_run_ends=False,
        _runner=SimpleNamespace(request_stop=lambda: calls.append("stop")),
        _status_msg=SimpleNamespace(setText=lambda _text: None),
    )
    event = SimpleNamespace(ignore=lambda: calls.append("ignore"))
    monkeypatch.setattr(
        QMessageBox, "question",
        lambda *_args, **_kwargs: QMessageBox.StandardButton.Yes,
    )

    MainWindow.closeEvent(fake, event)

    assert fake._close_when_run_ends is True
    assert calls == ["stop", "ignore"]
