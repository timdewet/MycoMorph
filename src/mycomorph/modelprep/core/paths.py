"""ModelPrep's per-user folder.

Trained models, settings, the score cache and learning curves live under
``~/.modelprep``. It keeps that name inside MycoMorph so existing runs, and
the paths that cite them, stay valid. Every caller goes through
``modelprep_home()`` so the folder can move in one place later.
"""

from __future__ import annotations

from pathlib import Path


def modelprep_home() -> Path:
    """``~/.modelprep``, resolved per call (tests point HOME at a temp dir)."""
    return Path.home() / ".modelprep"
