"""Shared image→RGB rendering for the GUI.

Both the labelling thumbnails and the cell inspector show the same thing: a
single-channel grayscale image, contrast-stretched, with a segmentation outline
painted on top in cyan. This module is the one place that logic lives.
"""

from __future__ import annotations

import numpy as np

CYAN = (0x21, 0xD4, 0xFD)


def normalise_gray(image: np.ndarray, percentiles: tuple[float, float] = (1, 99)) -> np.ndarray:
    """Percentile contrast-stretch a 2D array into uint8 [0, 255]."""
    img = np.asarray(image, dtype=np.float32)
    lo, hi = np.percentile(img, percentiles)
    scaled = np.clip((img - lo) / max(hi - lo, 1e-6), 0, 1)
    return (scaled * 255).astype(np.uint8)


def outline_of(mask: np.ndarray) -> np.ndarray:
    """Boolean outer boundary of a binary mask (empty mask → all-False)."""
    from skimage.segmentation import find_boundaries

    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return np.zeros(mask.shape, dtype=bool)
    return find_boundaries(mask, mode="outer")


def to_outlined_rgb(image: np.ndarray, outline: np.ndarray | None = None) -> np.ndarray:
    """Contrast-stretch ``image`` to grayscale RGB and paint ``outline`` cyan.

    ``image`` is any 2D array; ``outline`` is a boolean array of the same shape
    (e.g. from :func:`outline_of`). Returns a contiguous ``(H, W, 3)`` uint8
    array ready for ``QImage.Format_RGB888``.
    """
    gray = normalise_gray(image)
    rgb = np.stack([gray, gray, gray], axis=-1)
    if outline is not None and outline.any():
        rgb[outline] = CYAN
    return np.ascontiguousarray(rgb)
