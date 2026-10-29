"""Segmentation-quality / "crossing fragment" features from the stored crops.

When two filaments cross, the segmenter cuts each into a fragment at the
intersection. The tell-tale signs, computable from the ``[Bright, mask]`` crop:

  * the mask **forks** (a skeleton branchpoint) where an X-merge happened, or
    has an odd number of ends;
  * the brightfield cell body **continues past the mask end** — a whole cell's
    mask ends at a rounded pole where the brightfield fades to background; a cut
    fragment's mask ends mid-cell, with the filament continuing colinearly.

The continuation test samples *along the cell's own axis*, which is what
separates a *crossing* (colinear continuation) from a cell that merely *touches*
a neighbour (off-axis body, natural pole). Returns one row per cell to join to
the morphometric feature table by ``cell_uid``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from .crops_h5 import HDF5CropsReader

# How far past the mask end to look for continuation, in units of the local
# half-width, and the side-band used for the background reference.
_MARCH_WIDTHS = 1.5
_SIDE_PAD = 3


def _skeleton_ends_and_branches(mask: np.ndarray):
    """Return (endpoints[list of (y,x)], n_branchpoints) for a binary mask."""
    from skimage.morphology import skeletonize

    skel = skeletonize(mask)
    ys, xs = np.nonzero(skel)
    if ys.size == 0:
        return [], 0
    # 8-neighbour degree of each skeleton pixel.
    from scipy.ndimage import convolve
    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
    deg = convolve(skel.astype(np.int32), kernel, mode="constant")
    deg = deg * skel
    endpoints = list(zip(*np.nonzero((deg == 1))))
    n_branch = int(np.count_nonzero(deg >= 3))
    return endpoints, n_branch, skel


def _tangent(skel: np.ndarray, end: tuple, n: int = 5) -> np.ndarray:
    """Outward unit tangent at a skeleton endpoint, from the nearest ~n skel px."""
    ys, xs = np.nonzero(skel)
    pts = np.column_stack([ys, xs]).astype(float)
    d = np.hypot(pts[:, 0] - end[0], pts[:, 1] - end[1])
    near = pts[np.argsort(d)[: max(2, n)]]
    # Direction = endpoint minus centroid of the nearby skeleton (points outward).
    v = np.array(end, float) - near.mean(axis=0)
    norm = np.hypot(*v)
    return v / norm if norm > 1e-6 else np.array([0.0, 0.0])


def _continuation_score(bright, mask, skel, endpoints, half_width):
    """Max over ends of how cell-like the brightfield is just past the mask,
    sampled along the outward tangent. ~1 = cell continues (cut fragment),
    ~0 = background (natural pole)."""
    if not endpoints or half_width <= 0:
        return float("nan")
    fg = bright[mask]
    if fg.size == 0:
        return float("nan")
    i_cell = float(np.median(fg))
    # Background reference: a ring around the mask minus the mask itself, robust
    # to the (variable) neighbour content via the median.
    from scipy.ndimage import binary_dilation
    ring = binary_dilation(mask, iterations=int(round(half_width)) + _SIDE_PAD) & ~mask
    i_bg = float(np.median(bright[ring])) if ring.any() else float(np.median(bright))
    denom = i_cell - i_bg
    if abs(denom) < 1e-6:
        return float("nan")

    h, w = bright.shape
    best = 0.0
    reach = max(2.0, _MARCH_WIDTHS * half_width)
    for end in endpoints:
        t = _tangent(skel, end)
        if not np.any(t):
            continue
        vals = []
        for step in np.linspace(1.0, reach, num=max(3, int(reach))):
            y = int(round(end[0] + t[0] * step))
            x = int(round(end[1] + t[1] * step))
            if 0 <= y < h and 0 <= x < w and not mask[y, x]:
                vals.append(bright[y, x])
        if vals:
            score = (float(np.median(vals)) - i_bg) / denom
            best = max(best, float(np.clip(score, 0.0, 1.0)))
    return best


def compute_fragment_features(
    h5_path: Path,
    cell_uids: Optional[Sequence[str]] = None,
) -> "object":
    """Return a DataFrame (cell_uid + fragment features) for the given cells
    (default: all). Requires pandas."""
    import pandas as pd

    reader = HDF5CropsReader(h5_path)
    try:
        all_uids = [str(u) for u in reader.cell_uids]
        uid_to_idx = {u: i for i, u in enumerate(all_uids)}
        wanted = list(cell_uids) if cell_uids is not None else all_uids
        idx = [uid_to_idx[u] for u in wanted if u in uid_to_idx]
        phase_i, mask_i = reader.phase_channel_idx, reader.mask_channel_idx

        rows = []
        for u, i in zip((wanted if cell_uids is None else
                         [w for w in wanted if w in uid_to_idx]), idx):
            row = reader._raw_row(i)  # (C, H, W) raw stored crop
            bright = np.asarray(row[phase_i], dtype=np.float32)
            mask = np.asarray(row[mask_i]) > 0.5
            rows.append((u, *_features_for(bright, mask)))
    finally:
        reader.close()

    return pd.DataFrame(
        rows, columns=["cell_uid", "frag_endpoints", "frag_branchpoints",
                       "frag_bright_continuation"],
    )


def _features_for(bright: np.ndarray, mask: np.ndarray):
    if mask.sum() < 5:
        return 0, 0, float("nan")
    res = _skeleton_ends_and_branches(mask)
    if len(res) == 2:  # empty skeleton
        return 0, 0, float("nan")
    endpoints, n_branch, skel = res
    # Local half-width ≈ area / skeleton length (filament cross-section radius).
    skel_len = max(int(skel.sum()), 1)
    half_width = 0.5 * mask.sum() / skel_len
    cont = _continuation_score(bright, mask, skel, endpoints, half_width)
    return len(endpoints), n_branch, cont
