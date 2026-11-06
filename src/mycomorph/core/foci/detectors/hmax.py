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

1. Percentile-normalise (1st → 0, 99.9th → 1, with the top left
   unclipped), then lightly Gaussian-smooth (σ = ``min_sigma/2``) to
   suppress single-pixel salt noise.
2. ``h = max(hmax_h_mad * robust_sigma_mad(smoothed), threshold)`` — the
   ``opts.threshold`` floor keeps a degenerate (near-flat) channel from
   collapsing h to ~0 and detecting every noise ripple.
3. ``skimage.morphology.h_maxima`` → binary plateaus; connected-component
   centroids are the candidate peaks.
4. Sub-PSF impulse gate: drop candidates whose smoothed peak has no
   4-neighbour ≥ 25% of its (median-subtracted) height — single-pixel
   spikes are narrower than any real focus.
5. Greedy min-distance suppression at ``2·min_sigma`` px (brightest wins).
6. ``build_focus`` per candidate: optional 2D-Gaussian sub-pixel
   refinement, SNR gating, cell-label assignment.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .._types import DetectorOpts, Focus
from ._common import (
    build_focus,
    greedy_nms_indices,
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
        # Leave the top unclipped: in a frame with no bright foci the
        # 99.9th percentile sits on dim in-cell signal, so capping at 1
        # flattens every focus above it while h (in MAD units) can exceed
        # 1 — nothing could pass and the frame went blind.
        norm = normalise_image(image, clip_top=False)
        if norm.size == 0 or float(norm.max()) <= 0.0:
            return []

        smooth_sigma = max(0.5, 0.5 * float(opts.min_sigma))
        sm = gaussian_filter(norm.astype(np.float64), sigma=smooth_sigma)

        noise = robust_sigma_mad(sm)
        # MAD scaling assumes the smoothed background has healthy
        # texture. On a degenerate channel (e.g. mostly-zero residue
        # left by an aggressive upstream subtraction) the MAD is ~0,
        # which used to send h → ~0 and turn every noise ripple into a
        # plateau (observed in production: 60k "detections" on one
        # blank FOV). Floor the prominence at ``opts.threshold`` — the
        # same fraction-of-normalised-range meaning that parameter has
        # for the scale-space detectors.
        h = max(float(opts.hmax_h_mad) * noise, float(opts.threshold), 1e-6)

        # Grayscale reconstruction runs ~1.5x faster on integers, and
        # the h threshold spans thousands of 16-bit quanta, so the
        # quantisation step is far below the noise floor the threshold
        # is defined in — detections are unchanged up to sub-noise
        # boundary wobble.
        sm_max = float(sm.max())
        if sm_max <= 0.0:
            return []
        scale = 65535.0 / sm_max
        sm_q = (np.clip(sm, 0.0, None) * scale).astype(np.uint16)
        h_q = max(1, int(round(h * scale)))
        maxima = h_maxima(sm_q, h_q)
        if not maxima.any():
            return []
        lbl, n = ndimage.label(maxima, structure=np.ones((3, 3), dtype=int))
        centers = ndimage.center_of_mass(sm, lbl, index=range(1, n + 1))

        pts = np.array([[float(y), float(x)] for y, x in centers])
        yi = np.clip(np.round(pts[:, 0]).astype(int), 0, sm.shape[0] - 1)
        xi = np.clip(np.round(pts[:, 1]).astype(int), 0, sm.shape[1] - 1)
        scores = sm[yi, xi]

        # Sub-PSF impulse gate: a diffraction-limited focus (σ ≳ 1 px)
        # keeps well over half its smoothed peak in the immediate
        # 4-neighbourhood, while a single-pixel spike (hot pixel, shot
        # noise, clipped-subtraction residue) drops to ~exp(-2) ≈ 0.14
        # under the σ = min_sigma/2 smoothing. Structure narrower than
        # the PSF is not optical signal. Amplitudes are taken above the
        # global median, which is exact on the flat-background images
        # this targets; on elevated local background the ratio only
        # grows, so the gate fails open and keeps the candidate.
        med = float(np.median(sm))
        neigh = np.maximum.reduce([
            sm[np.clip(yi - 1, 0, sm.shape[0] - 1), xi],
            sm[np.clip(yi + 1, 0, sm.shape[0] - 1), xi],
            sm[yi, np.clip(xi - 1, 0, sm.shape[1] - 1)],
            sm[yi, np.clip(xi + 1, 0, sm.shape[1] - 1)],
        ])
        solid = (neigh - med) >= 0.25 * np.maximum(scores - med, 1e-12)
        pts = pts[solid]
        scores = scores[solid]
        if not len(pts):
            return []

        # Greedy min-distance suppression, brightest plateau first
        # (KD-tree NMS — plateau counts explode on noisy images and a
        # naive pairwise scan goes quadratic).
        min_dist = max(1.0, 2.0 * float(opts.min_sigma))
        kept = [tuple(pts[i]) for i in greedy_nms_indices(pts, scores, min_dist)]

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
