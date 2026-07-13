"""Domain exceptions shared by the library, CLI, and GUI."""

from __future__ import annotations


class PipelineValidationError(RuntimeError):
    """Raised when user input cannot produce a valid pipeline run."""

