"""Classical foci detectors: DoG, LoG, top-hat preprocessing, trackpy.

These four share the same scale-space-blob idea: find local maxima after
filtering, then optionally fit a 2D Gaussian for sub-pixel position +
sigma. DoG and LoG come from ``skimage.feature``; trackpy uses a
centroid-based locator (Crocker–Grier) with its own sub-pixel
refinement. Top-hat is a preprocessing wrapper around DoG that
suppresses slowly-varying background before the blob filter runs.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .._types import DetectorOpts, Focus
from ._common import build_focus, normalise_image


# ─────────────────────────────────────────────────────────────────────────────
# DoG (Difference of Gaussians)
# ─────────────────────────────────────────────────────────────────────────────

class DogDetector:
    name = "dog"

    def fit(self, run_stack):  # noqa: D401
        """No-op — DoG has no per-run state."""
        return None

    def detect(
        self,
        image: np.ndarray,
        labeled_mask: Optional[np.ndarray] = None,
        opts: Optional[DetectorOpts] = None,
    ) -> list[Focus]:
        from skimage.feature import blob_dog

        opts = opts or DetectorOpts()
        norm = normalise_image(image)
        blobs = blob_dog(
            norm,
            min_sigma=opts.min_sigma,
            max_sigma=opts.max_sigma,
            sigma_ratio=1.6,
            threshold=opts.threshold,
            overlap=opts.overlap,
        )
        out: list[Focus] = []
        for y, x, sigma in blobs:
            focus = build_focus(
                image, float(y), float(x), float(sigma),
                labeled_mask=labeled_mask,
                refine=opts.refine,
                refine_window=opts.refine_window,
                snr_min=opts.snr_min,
            )
            if focus is not None:
                out.append(focus)
        return out


# ─────────────────────────────────────────────────────────────────────────────
# LoG (Laplacian of Gaussian)
# ─────────────────────────────────────────────────────────────────────────────

class LogDetector:
    name = "log"

    def fit(self, run_stack):
        return None

    def detect(
        self,
        image: np.ndarray,
        labeled_mask: Optional[np.ndarray] = None,
        opts: Optional[DetectorOpts] = None,
    ) -> list[Focus]:
        from skimage.feature import blob_log

        opts = opts or DetectorOpts()
        norm = normalise_image(image)
        blobs = blob_log(
            norm,
            min_sigma=opts.min_sigma,
            max_sigma=opts.max_sigma,
            num_sigma=opts.num_sigma,
            threshold=opts.threshold,
            overlap=opts.overlap,
        )
        out: list[Focus] = []
        for y, x, sigma in blobs:
            focus = build_focus(
                image, float(y), float(x), float(sigma),
                labeled_mask=labeled_mask,
                refine=opts.refine,
                refine_window=opts.refine_window,
                snr_min=opts.snr_min,
            )
            if focus is not None:
                out.append(focus)
        return out


# ─────────────────────────────────────────────────────────────────────────────
# Top-hat + DoG (background-suppressing preprocessing)
# ─────────────────────────────────────────────────────────────────────────────

class TophatDogDetector:
    name = "tophat_dog"

    def __init__(self) -> None:
        self._inner = DogDetector()

    def fit(self, run_stack):
        return None

    def detect(
        self,
        image: np.ndarray,
        labeled_mask: Optional[np.ndarray] = None,
        opts: Optional[DetectorOpts] = None,
    ) -> list[Focus]:
        from skimage.morphology import white_tophat, disk

        opts = opts or DetectorOpts()
        radius = max(1, int(opts.tophat_radius_px))
        tophat = white_tophat(image.astype(np.float32), footprint=disk(radius))
        return self._inner.detect(tophat, labeled_mask=labeled_mask, opts=opts)


# ─────────────────────────────────────────────────────────────────────────────
# Trackpy (Crocker–Grier with sub-pixel centroid + Gaussian)
# ─────────────────────────────────────────────────────────────────────────────

class TrackpyDetector:
    name = "trackpy"

    def fit(self, run_stack):
        return None

    def detect(
        self,
        image: np.ndarray,
        labeled_mask: Optional[np.ndarray] = None,
        opts: Optional[DetectorOpts] = None,
    ) -> list[Focus]:
        try:
            import trackpy as tp
        except ImportError as exc:  # noqa: BLE001
            raise ImportError(
                "trackpy is not installed. `pip install trackpy` to use this detector."
            ) from exc

        opts = opts or DetectorOpts()
        # trackpy needs an odd diameter and works best on the raw or
        # background-subtracted image. We feed the normalised image so
        # the minmass threshold is interpretable across image scales.
        norm = normalise_image(image)
        diameter = int(opts.trackpy_diameter_px)
        if diameter % 2 == 0:
            diameter += 1

        # Suppress trackpy's "loud" reporting.
        tp.quiet(suppress=True)
        df = tp.locate(
            norm,
            diameter=diameter,
            minmass=opts.trackpy_minmass,
            percentile=64,        # keep things permissive; SNR cut filters later
            preprocess=True,
        )
        out: list[Focus] = []
        for _, row in df.iterrows():
            # trackpy already gives sub-pixel y, x and a size proxy.
            y_sub = float(row["y"])
            x_sub = float(row["x"])
            size_proxy = float(row.get("size", opts.min_sigma * 1.5))
            mass = float(row.get("mass", 0.0))
            # Build the focus with refinement OFF (trackpy has done it
            # already) but compute SNR/cell_label consistently.
            focus = build_focus(
                image, y_sub, x_sub, size_proxy,
                labeled_mask=labeled_mask,
                refine=False,
                snr_min=opts.snr_min,
                intensity_override=mass,
            )
            if focus is not None:
                out.append(focus)
        return out
