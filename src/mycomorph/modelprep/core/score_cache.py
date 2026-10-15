"""Per-(model, HDF5) cache of classifier scores.

Loading a labelling pool scores every unseen cell with the chosen model, which
takes about a minute for ~10k cells. The scores only change when the model or
the HDF5 changes, so they're cached: in memory for the session, and on disk
under ``~/.modelprep/score_cache/`` so a restart doesn't rescore either.

The key covers both files' resolved path, size and modification time, so a
retrained model or a rewritten HDF5 gets a fresh entry. Unscored cells are NaN;
callers score only those and write them back, so a cancelled load keeps the
progress it made.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Optional

import numpy as np

from .paths import modelprep_home

_VERSION = "1"
_MEMORY: dict[str, np.ndarray] = {}


def score_cache_root() -> Path:
    return modelprep_home() / "score_cache"


def _file_key(path: Path) -> str:
    p = Path(path).resolve()
    st = os.stat(p)
    return f"{p}|{st.st_size}|{st.st_mtime_ns}"


def cache_key(h5_path: Path, model_path: Path) -> str:
    raw = f"v{_VERSION}|{_file_key(h5_path)}|{_file_key(model_path)}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


class ScoreCache:
    """P(good) for every cell in one HDF5 under one model (NaN = not scored)."""

    def __init__(self, key: str, scores: np.ndarray, root: Optional[Path] = None) -> None:
        self.key = key
        self.scores = scores
        self._root = root

    @classmethod
    def open(cls, h5_path: Path, model_path: Path, n_cells: int,
             root: Optional[Path] = None) -> "ScoreCache":
        key = cache_key(h5_path, model_path)
        scores = _MEMORY.get(key)
        if scores is None or len(scores) != n_cells:
            scores = None
            f = (root or score_cache_root()) / f"{key}.npy"
            if f.exists():
                try:
                    arr = np.load(f)
                    if arr.shape == (n_cells,):
                        scores = arr.astype(np.float32)
                except Exception:  # noqa: BLE001 — a bad cache file is just a miss
                    scores = None
            if scores is None:
                scores = np.full(n_cells, np.nan, dtype=np.float32)
            _MEMORY[key] = scores
        return cls(key, scores, root)

    def missing(self, indices: np.ndarray) -> np.ndarray:
        """The subset of ``indices`` that still needs scoring."""
        indices = np.asarray(indices)
        return indices[np.isnan(self.scores[indices])]

    def save(self) -> None:
        """Persist to disk (best effort: a read-only home just skips it)."""
        try:
            root = self._root or score_cache_root()
            root.mkdir(parents=True, exist_ok=True)
            tmp = root / f"{self.key}.tmp.npy"
            np.save(tmp, self.scores)
            os.replace(tmp, root / f"{self.key}.npy")
        except OSError:
            pass


def clear_memory() -> None:
    """Drop the in-memory layer (tests)."""
    _MEMORY.clear()
