"""Per-label areas and bounding boxes of a labelled mask, in one pass.

Dense fields hold thousands of cells, so per-cell code that compares the
whole mask against one label (``labeled_mask == lbl``, ``np.where``) costs
O(cells x pixels). ``label_index`` gets every label's area and tight
bounding box at once; per-cell work then only reads ``labeled_mask[sl]``.
Every pixel of a label lies inside its tight bounding box, so results
computed inside it are identical to the whole-mask versions.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

CellSlice = tuple[slice, slice]


def label_index(labeled_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[CellSlice]]:
    """Return ``(labels, areas, slices)`` for every non-zero label.

    ``labels`` are sorted and keep the mask's dtype, as ``np.unique`` gives
    them. ``areas[i]`` is the pixel count of ``labels[i]`` and ``slices[i]``
    its tight ``(rows, cols)`` bounding box, as from
    ``scipy.ndimage.find_objects``.
    """
    mask = np.asarray(labeled_mask)
    if (mask.dtype.kind in "ui" and mask.size
            and mask.min() >= 0 and mask.max() <= mask.size):
        from scipy import ndimage

        counts = np.bincount(mask.ravel().astype(np.intp, copy=False))
        labels = np.flatnonzero(counts[1:]) + 1
        objects = ndimage.find_objects(mask)
        slices = [objects[lbl - 1] for lbl in labels]
        return labels.astype(mask.dtype), counts[labels], slices

    # Non-integer, negative or very sparse labels: one scan per label.
    labels, areas = np.unique(mask, return_counts=True)
    keep = labels != 0
    labels, areas = labels[keep], areas[keep]
    slices = [label_bbox(mask, lbl) for lbl in labels]
    return labels, areas, slices


def label_bbox(labeled_mask: np.ndarray, label) -> Optional[CellSlice]:
    """Tight ``(rows, cols)`` bounding box of one label, or None if absent.

    Scans the whole mask; use ``label_index`` when looping over cells.
    """
    cell_pixels = labeled_mask == label
    rows = np.flatnonzero(cell_pixels.any(axis=1))
    if rows.size == 0:
        return None
    cols = np.flatnonzero(cell_pixels.any(axis=0))
    return (slice(int(rows[0]), int(rows[-1]) + 1),
            slice(int(cols[0]), int(cols[-1]) + 1))
