"""Locate a labelled cell back in its original field-of-view image.

The HDF5 crops are cut tight (cell bbox + a few px, background masked out), so
they can't show a cell among its neighbours. The pixels for that still exist in
the MycoMorph pipeline outputs that sit next to the HDF5:

    <root>/01_focus/<tiff_file>     focused image,  (n_fov, C, H, W)  — brightfield = channel 0
    <root>/02_segment/<tiff_file>   binary mask,     last channel
    <root>/03_classify/<tiff_file>  labelled mask,   last channel
    <root>/04_features/all_crops.h5 the open file

This module reconstructs the exact label map MycoMorph's Features stage used —
``_label_mask(fov[-1])`` (``mycomorph/core/extract/api.py``): ``skimage.label`` of
the binary mask when it's binary, else the channel as-is. Selecting
``labeled == cell_label`` then reproduces the cell's stored ``area_px`` exactly,
which is also how we pick (and validate) the right mask source. From the cell's
centroid we cut a fixed-size window for the inspector.

Reads are per-FOV (only that FOV's pages) and LRU-cached, so the originals —
which may live on slow network storage — are touched as little as possible.
``get_window`` returns ``None`` rather than raising whenever anything is missing,
so callers can fall back to the isolated crop.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from .crops_h5 import CropRecord

_FOV_CACHE_SIZE = 4  # ~32 MB per FOV (float32 brightfield + int32 labels)


@dataclass(frozen=True)
class FovContext:
    """A square window of the original FOV centred on one cell."""
    image: np.ndarray      # (window_px, window_px) float32 brightfield
    outline: np.ndarray    # (window_px, window_px) bool — target cell boundary
    fov_index: int
    window_px: int
    tiff_file: str


def _label_mask(raw: np.ndarray) -> np.ndarray:
    """Re-label a binary or already-labelled mask channel.

    Mirrors ``mycomorph.core.extract.api._label_mask`` so the labels match the
    ``cell_label`` values stored in the HDF5.
    """
    from skimage.measure import label as sk_label

    unique_positive = np.unique(raw[raw > 0])
    if len(unique_positive) <= 1:
        return sk_label(raw > 0, connectivity=1).astype(np.int32)
    return raw.astype(np.int32)


class FovContextProvider:
    """Resolves a cell's neighbourhood from the pipeline images beside the HDF5."""

    def __init__(self, h5_path: Path) -> None:
        h5_path = Path(h5_path)
        root = h5_path.parent.parent  # .../<root>/04_features/all_crops.h5
        self.focus_dir = root / "01_focus"
        # Prefer the labelled classify mask; fall back to the binary segment mask.
        self._mask_dirs = [root / "03_classify", root / "02_segment"]
        self._mask_dirs = [d for d in self._mask_dirs if d.is_dir()]

        self.available = self.focus_dir.is_dir() and bool(self._mask_dirs)

        self._mask_source: Optional[Path] = None  # locked once area-validated
        self._geom: dict[Path, tuple[int, int, int, int]] = {}  # path → (nfov,C,H,W)
        self._cache: "OrderedDict[tuple[str, int], tuple[np.ndarray, np.ndarray]]" = OrderedDict()

    # ── public ─────────────────────────────────────────────────────────

    def get_window(self, record: CropRecord, window_px: int) -> Optional[FovContext]:
        """Return a ``window_px`` square of the original FOV centred on the cell,
        or ``None`` if the originals are missing or the cell can't be located."""
        if not self.available:
            return None
        try:
            bright, labeled = self._fov_arrays(record)
        except Exception:  # noqa: BLE001 — any I/O/decoding issue → graceful None
            return None
        if bright is None or labeled is None:
            return None

        region = labeled == int(record.cell_label)
        if not region.any():
            return None

        ys, xs = np.nonzero(region)
        cy = int(round(float(ys.mean())))
        cx = int(round(float(xs.mean())))

        image = self._cut(bright, cy, cx, window_px, fill=0.0)
        region_win = self._cut(region.astype(np.float32), cy, cx, window_px, fill=0.0) > 0.5

        from ._render import outline_of
        return FovContext(
            image=image,
            outline=outline_of(region_win),
            fov_index=int(record.fov_index),
            window_px=int(window_px),
            tiff_file=str(record.tiff_file),
        )

    # ── windowing ──────────────────────────────────────────────────────

    @staticmethod
    def _cut(arr: np.ndarray, cy: int, cx: int, side: int, fill: float) -> np.ndarray:
        """Extract a ``side``×``side`` patch centred on ``(cy, cx)``, zero-padded
        where it runs past the image edge so the output size is constant."""
        h, w = arr.shape
        half = side // 2
        top, left = cy - half, cx - half
        out = np.full((side, side), fill, dtype=arr.dtype)
        y0, y1 = max(top, 0), min(top + side, h)
        x0, x1 = max(left, 0), min(left + side, w)
        if y1 > y0 and x1 > x0:
            out[y0 - top:y1 - top, x0 - left:x1 - left] = arr[y0:y1, x0:x1]
        return out

    # ── FOV loading (cached) ───────────────────────────────────────────

    def _fov_arrays(self, record: CropRecord):
        key = (str(record.tiff_file), int(record.fov_index))
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]

        bright = self._read_channel(self.focus_dir, key[0], key[1], channel=0)
        labeled = self._read_labeled(key[0], key[1], int(record.cell_label), int(record.area_px))
        if bright is None or labeled is None:
            return None, None

        self._cache[key] = (bright, labeled)
        self._cache.move_to_end(key)
        while len(self._cache) > _FOV_CACHE_SIZE:
            self._cache.popitem(last=False)
        return bright, labeled

    def _read_labeled(
        self, tiff_file: str, fov_index: int, cell_label: int, area_px: int
    ) -> Optional[np.ndarray]:
        # Source already locked by an earlier exact area match.
        if self._mask_source is not None:
            raw = self._read_channel(self._mask_source, tiff_file, fov_index, channel=-1)
            return None if raw is None else _label_mask(raw)

        # First time: try each candidate, lock on an exact area_px match.
        fallback: Optional[np.ndarray] = None
        for d in self._mask_dirs:
            raw = self._read_channel(d, tiff_file, fov_index, channel=-1)
            if raw is None:
                continue
            labeled = _label_mask(raw)
            area = int((labeled == cell_label).sum())
            if area == area_px:
                self._mask_source = d
                return labeled
            if area > 0 and fallback is None:
                fallback = labeled  # cell present but area differs — best effort
        return fallback

    def _read_channel(
        self, directory: Path, tiff_file: str, fov_index: int, channel: int
    ) -> Optional[np.ndarray]:
        """Read a single channel of a single FOV, decoding only that FOV's pages."""
        import tifffile

        path = directory / tiff_file
        if not path.exists():
            return None
        geom = self._tiff_geom(path)
        if geom is None:
            return None
        nfov, C, _, _ = geom
        if not (0 <= fov_index < nfov):
            return None
        stack = tifffile.imread(path, key=range(fov_index * C, (fov_index + 1) * C))
        plane = stack[channel] if stack.ndim == 3 else stack
        return np.asarray(plane)

    def _tiff_geom(self, path: Path) -> Optional[tuple[int, int, int, int]]:
        """Return ``(n_fov, channels, H, W)`` from TIFF metadata (no pixel read)."""
        if path in self._geom:
            return self._geom[path]
        import tifffile

        with tifffile.TiffFile(str(path)) as tf:
            shape = tf.series[0].shape
        if len(shape) == 4:
            nfov, C, H, W = (int(x) for x in shape)
        elif len(shape) == 3:  # (n_fov, H, W) — single channel per FOV
            nfov, H, W = (int(x) for x in shape)
            C = 1
        else:
            return None
        self._geom[path] = (nfov, C, H, W)
        return self._geom[path]
