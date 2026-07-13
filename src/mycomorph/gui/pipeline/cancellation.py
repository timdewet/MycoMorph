"""Thread-safe cooperative cancellation primitives for pipeline workers."""

from __future__ import annotations

import threading


class StopRequested(Exception):
    """Raised at a cooperative cancellation boundary inside a stage."""


class CancellationToken:
    """A cancellation flag that can be set safely from the GUI thread."""

    def __init__(self) -> None:
        self._event = threading.Event()

    @property
    def is_cancelled(self) -> bool:
        return self._event.is_set()

    def cancel(self) -> None:
        self._event.set()

    def raise_if_cancelled(self) -> None:
        if self.is_cancelled:
            raise StopRequested()
