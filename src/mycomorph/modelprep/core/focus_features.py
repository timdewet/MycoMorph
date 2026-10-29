"""Crop-based focus / sharpness features.

Out-of-focus cells are a quality cue a human uses but the isolated morphometrics
don't capture. Focus is just local high-frequency content, so we can measure it
straight from the brightfield in each stored crop — no need for the (huge)
original FOV TIFFs.

Caveat: crops are resized (pad-to-square → ``crop_size``), which alters
high-frequency content, so absolute sharpness is partly confounded with how much
a cell was scaled. The classifier also sees ``area_px`` etc., so trees can
partly disentangle it; we keep these as *relative* cues and let the data decide.

Reads are batched (sorted fancy-indexing) so the crops HDF5 is hit in a few
large reads rather than thousands of scattered ones.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from .crops_h5 import HDF5CropsReader


def _sharpness(bright: np.ndarray) -> tuple[float, float, float]:
    """(variance-of-Laplacian, Tenengrad gradient energy, normalised variance)."""
    from scipy.ndimage import laplace, sobel

    b = np.asarray(bright, dtype=np.float32)
    lap_var = float(laplace(b).var())
    gx = sobel(b, axis=0)
    gy = sobel(b, axis=1)
    tenengrad = float(np.mean(gx * gx + gy * gy))
    mu = float(b.mean())
    norm_var = float(b.var() / (mu * mu + 1e-9))
    return lap_var, tenengrad, norm_var


def compute_focus_features(
    h5_path: Path,
    cell_uids: Optional[Sequence[str]] = None,
    batch: int = 512,
) -> "object":
    """Per-cell focus features (DataFrame keyed by ``cell_uid``)."""
    import pandas as pd

    reader = HDF5CropsReader(h5_path)
    try:
        all_uids = [str(u) for u in reader.cell_uids]
        uid_to_idx = {u: i for i, u in enumerate(all_uids)}
        wanted = [u for u in (cell_uids if cell_uids is not None else all_uids)
                  if u in uid_to_idx]
        idx = [uid_to_idx[u] for u in wanted]
        phase_i = reader.phase_channel_idx

        rows = []
        for lo in range(0, len(idx), batch):
            chunk_idx = idx[lo:lo + batch]
            chunk_uids = wanted[lo:lo + batch]
            arr = reader._raw_batch(chunk_idx)        # (B, C, H, W), input order
            bright = arr[:, phase_i]
            for u, b in zip(chunk_uids, bright):
                lv, ten, nv = _sharpness(b)
                rows.append((u, lv, ten, nv))
    finally:
        reader.close()

    return pd.DataFrame(
        rows, columns=["cell_uid", "focus_lap_var", "focus_tenengrad", "focus_norm_var"],
    )
