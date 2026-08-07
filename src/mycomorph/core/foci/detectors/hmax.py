"""h-maxima (morphological prominence) foci detector.

Detection by grayscale reconstruction: a peak survives iff it rises at
least ``h`` above its surroundings, regardless of the absolute intensity
it sits on. This is the "h-dome" family from the Smal et al. spot-
detection benchmark (IEEE TMI 2010), where it ranked alongside wavelets,
and it is exactly what ImageJ/Fiji's *Find Maxima* does with its
"prominence" parameter — so results are directly comparable to a manual
Fiji workflow.

Why it complements DoG/LoG/wavelet:

- Prominence is a **local** criterion — a dim focus on a dim cell and a
  bright focus on a bright cell both survive the same ``h``, where a
  global relative threshold (DoG/LoG ``threshold``) trades one off
  against the other.
- No scale race: plateaus and slightly-elongated peaks that scale-space
  detectors split or miss survive reconstruction intact.

``h`` is specified in robust-noise units (``hmax_h_mad`` ×
MAD-estimated σ of the smoothed image), so the same setting transfers
across exposure times and channels.

Algorithm:

1. Percentile-normalise, then lightly Gaussian-smooth (σ = ``min_sigma/2``)
   to suppress single-pixel salt noise.
2. ``h = hmax_h_mad * robust_sigma_mad(smoothed)``.
3. ``skimage.morphology.h_maxima`` → binary plateaus; connected-component
   centroids are the candidate peaks.
4. Greedy min-distance suppression at ``2·min_sigma`` px (brightest wins).
5. ``build_focus`` per candidate: optional 2D-Gaussian sub-pixel
   refinement, SNR gating, cell-label assignment.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .._types import DetectorOpts, Focus
from ._common import (
    build_focus,
    normalise_image,
    robust_sigma_mad,
    suppress_close_foci,
)


class HMaxDetector:
    name = "hmax"

    def fit(self, run_stack):  # noqa: D401
        """No-op — h-maxima has no per-run state."""
        return None

    def detect(
        self,
        image: np.ndarray,
        labeled_mask: Optional[np.ndarray] = None,
        opts: Optional[DetectorOpts] = None,
    ) -> list[Focus]:
        from scipy import ndimage
        from scipy.ndimage import gaussian_filter
        from skimage.morphology import h_maxima

        opts = opts or DetectorOpts()
        norm = normalise_image(image)
        if norm.size == 0 or float(norm.max()) <= 0.0:
            return []

        smooth_sigma = max(0.5, 0.5 * float(opts.min_sigma))
        sm = gaussian_filter(norm.astype(np.float64), sigma=smooth_sigma)

        noise = robust_sigma_mad(sm)
        h = max(float(opts.hmax_h_mad) * noise, 1e-6)

        maxima = h_maxima(sm, h)
        if not maxima.any():
            return []
        lbl, n = ndimage.label(maxima, structure=np.ones((3, 3), dtype=int))
        centers = ndimage.center_of_mass(sm, lbl, index=range(1, n + 1))

        # Greedy min-distance suppression, brightest plateau first.
        peaks = sorted(
            ((float(y), float(x)) for y, x in centers),
            key=lambda p: sm[int(round(p[0])), int(round(p[1]))],
            reverse=True,
        )
        min_dist = max(1.0, 2.0 * float(opts.min_sigma))
        kept: list[tuple[float, float]] = []
        for y, x in peaks:
            if any((y - ky) ** 2 + (x - kx) ** 2 < min_dist ** 2 for ky, kx in kept):
                continue
            kept.append((y, x))

        out: list[Focus] = []
        init_sigma = max(float(opts.min_sigma), 1.2)
        for y, x in kept:
            focus = build_focus(
                image, y, x, init_sigma,
                labeled_mask=labeled_mask,
                refine=opts.refine,
                refine_window=opts.refine_window,
                snr_min=opts.snr_min,
            )
            if focus is not None:
                out.append(focus)
        # Refinement can pull two plateaus of an unresolved doublet onto
        # nearly the same point — collapse those to one detection.
        return suppress_close_foci(out, min_dist)
