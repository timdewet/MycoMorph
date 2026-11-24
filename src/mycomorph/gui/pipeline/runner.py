"""Qt runners: the core pipeline runs on a QThread, reporting through Qt signals.

The run logic itself is in :mod:`mycomorph.core.pipeline.runner`, shared with
``mycomorph-cli run``. These classes only add the signals and the worker
thread.
"""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import QObject, QThread, pyqtSignal

from mycomorph.core.pipeline.context import BulkRunContext, RunContext
from mycomorph.core.pipeline.runner import BulkRunLogic, PlateRunLogic
from mycomorph.core.pipeline.stages import Stage


class _ThreadedRun:
    """Runs ``execute()`` on a QThread owned by the runner."""

    _thread: QThread | None = None

    def start(self) -> None:
        self._thread = QThread()
        self.moveToThread(self._thread)
        self._thread.started.connect(self._run)
        self._thread.finished.connect(self.workerStopped)
        self._thread.start()

    def _run(self) -> None:
        try:
            self.execute()
        finally:
            if self._thread is not None:
                self._thread.quit()


class PipelineRunner(_ThreadedRun, PlateRunLogic, QObject):
    """Executes enabled stages sequentially on its own thread.

    Emits, in order: runStarted → stageStarted → stageProgress* → stageFinished
    (repeated per stage) → runFinished / runFailed.
    """

    runStarted = pyqtSignal()
    stagesPlanned = pyqtSignal(list)             # names of stages that will actually run
    stageStarted = pyqtSignal(str)               # stage name
    stageProgress = pyqtSignal(str, float, str)  # stage name, fraction, message
    stageFinished = pyqtSignal(str, int)         # stage name, num_outputs
    stageCancelled = pyqtSignal(str)              # stage name
    runFinished = pyqtSignal(Path)               # manifest path
    runCancelled = pyqtSignal(Path)              # manifest path
    runFailed = pyqtSignal(str)                  # error message
    workerStopped = pyqtSignal()

    def __init__(
        self,
        ctx: RunContext,
        stages: list[Stage] | None = None,
        reuse_existing: bool = False,
    ) -> None:
        super().__init__(ctx, stages=stages, reuse_existing=reuse_existing)


class BulkPipelineRunner(_ThreadedRun, BulkRunLogic, QObject):
    """Runs Focus once per CZI, then Segment + Classify once over the
    accumulated TIFF directory. Mirrors PipelineRunner's signal surface
    so the Run panel can stay agnostic about which mode is active.
    """

    runStarted = pyqtSignal()
    stagesPlanned = pyqtSignal(list)
    stageStarted = pyqtSignal(str)
    stageProgress = pyqtSignal(str, float, str)
    stageFinished = pyqtSignal(str, int)
    stageCancelled = pyqtSignal(str)
    runFinished = pyqtSignal(Path)
    runCancelled = pyqtSignal(Path)
    runFailed = pyqtSignal(str)
    workerStopped = pyqtSignal()

    def __init__(self, ctx: BulkRunContext, reuse_existing: bool = False) -> None:
        super().__init__(ctx, reuse_existing=reuse_existing)
