"""Field-of-view context features.

The CNN and the per-cell morphometrics both see a cell **in isolation**. But a
lot of "is this cell good" is a property of its *surroundings* — is it in a
crowded clump, touching/overlapping neighbours, alone in a sparse field? Those
are exactly the cues a human uses and the isolated representations lack.

These features come straight from the morphometric table's per-cell **positions**
(``centroid_*``, ``bbox_*``, ``fov``) — no image loading — so they're cheap.
Density is computed against *all* cells in each field of view, not just the
labelled ones. Returns one row per cell to join by ``cell_uid``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np


def _features_path(h5_path: Path) -> Optional[Path]:
    base = Path(h5_path).parent
    for name in ("all_features.parquet", "all_features.csv"):
        p = base / name
        if p.exists():
            return p
    return None


def _bbox_overlap_counts(bb: np.ndarray) -> np.ndarray:
    """Count, per cell, how many other cells' bounding boxes it intersects.
    ``bb`` columns are (y0, x0, y1, x1)."""
    y0, x0, y1, x1 = bb[:, 0], bb[:, 1], bb[:, 2], bb[:, 3]
    # Pairwise overlap: boxes i,j overlap iff x ranges and y ranges both overlap.
    x_ov = (x0[:, None] < x1[None, :]) & (x1[:, None] > x0[None, :])
    y_ov = (y0[:, None] < y1[None, :]) & (y1[:, None] > y0[None, :])
    ov = x_ov & y_ov
    np.fill_diagonal(ov, False)
    return ov.sum(axis=1)


def compute_context_features(
    h5_path: Path,
    cell_uids: Optional[Sequence[str]] = None,
    radii: Sequence[float] = (50.0, 100.0),
) -> "object":
    """Per-cell FOV-context features (DataFrame keyed by ``cell_uid``)."""
    import pandas as pd
    from scipy.spatial import cKDTree

    fpath = _features_path(h5_path)
    if fpath is None:
        raise FileNotFoundError(f"No all_features table next to {Path(h5_path).name}.")
    cols = ["cell_uid", "fov", "centroid_y", "centroid_x",
            "bbox_y0", "bbox_x0", "bbox_y1", "bbox_x1"]
    df = (pd.read_parquet(fpath, columns=None) if fpath.suffix == ".parquet"
          else pd.read_csv(fpath))
    if "well" in df.columns:
        group_keys = ["well", "fov"]
    else:
        group_keys = ["fov"]

    parts = []
    for _, g in df.groupby(group_keys):
        cents = g[["centroid_y", "centroid_x"]].to_numpy(float)
        n = len(g)
        nn = np.full(n, np.nan)
        neigh = {r: np.zeros(n, dtype=int) for r in radii}
        if n > 1:
            tree = cKDTree(cents)
            d, _ = tree.query(cents, k=2)
            nn = d[:, 1]
            for r in radii:
                neigh[r] = np.array([len(tree.query_ball_point(c, r)) - 1 for c in cents])
        bb = g[["bbox_y0", "bbox_x0", "bbox_y1", "bbox_x1"]].to_numpy(float)
        out = {
            "cell_uid": g["cell_uid"].to_numpy(),
            "ctx_fov_count": n,
            "ctx_nn_dist": nn,
            "ctx_bbox_overlaps": _bbox_overlap_counts(bb),
        }
        for r in radii:
            out[f"ctx_neigh_{int(r)}"] = neigh[r]
        parts.append(pd.DataFrame(out))

    res = pd.concat(parts, ignore_index=True)
    if cell_uids is not None:
        res = res[res["cell_uid"].isin(set(cell_uids))].reset_index(drop=True)
    return res
