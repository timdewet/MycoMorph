"""Radial-symmetry foci detector (Parthasarathy 2012).

Reference: Parthasarathy, *Nature Methods* 9:724–726 (2012), "Rapid,
accurate particle tracking by calculation of radial symmetry centers" —
the localisation engine inside RS-FISH (Bahry et al., Nature Methods
2022).

The idea: at every pixel of a radially symmetric spot, the intensity
gradient points through the centre. The centre is therefore the point
minimising the weighted squared distance to all gradient-direction
lines — a closed-form 2×2 linear solve, no iterative fitting. Compared
to least-squares Gaussian refinement this is

- more accurate on **dim** spots (no fit to diverge, no offset/σ
  trade-off eating the localisation budget),
- non-iterative and fast, and
- model-free — it assumes symmetry, not Gaussian-ness.

Pipeline: candidate peaks come from a lightly smoothed, MAD-thresholded
local-maxima pass (``rs_threshold_mad`` × robust noise σ above the
median); each candidate is then localised by the radial-symmetry solve
on the raw image. ``opts.refine`` is ignored — radial symmetry *is* the
sub-pixel step here. σ is estimated from background-subtracted second
moments (for the ``Focus`` size proxy and SNR annulus geometry).
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


def radial_symmetry_center(
    patch: np.ndarray,
    y0: int,
    x0: int,
) -> Optional[tuple[float, float]]:
    """Radial-symmetry centre of ``patch`` in absolute image coordinates.

    ``(y0, x0)`` is the absolute position of ``patch[0, 0]``. Returns
    ``None`` when the gradient field is degenerate (flat patch /
    singular normal equations); the caller falls back to the integer
    candidate peak.
    """
    p = np.asarray(patch, dtype=np.float64)
    if p.shape[0] < 3 or p.shape[1] < 3:
        return None

    gy, gx = np.gradient(p)
    m2 = gx * gx + gy * gy
    total = float(m2.sum())
    if total <= 0.0:
        return None

    yy, xx = np.mgrid[y0:y0 + p.shape[0], x0:x0 + p.shape[1]].astype(np.float64)

    # Down-weight pixels far from the gradient-magnitude centroid so
    # neighbouring structure at the window edge doesn't drag the centre.
    cy = float((m2 * yy).sum() / total)
    cx = float((m2 * xx).sum() / total)
    dist = np.hypot(yy - cy, xx - cx)
    w = m2 / (dist + 1.0)

    # Keep only pixels with a meaningful gradient direction.
    keep = m2 > (1e-4 * float(m2.max()))
    if keep.sum() < 4:
        return None
    w = np.where(keep, w, 0.0)
    inv_m = np.zeros_like(m2)
    inv_m[keep] = 1.0 / np.sqrt(m2[keep])
    u = gx * inv_m   # unit gradient, x component
    v = gy * inv_m   # unit gradient, y component

    # Least squares for the point minimising Σ w · d²(c, line through
    # r along ĝ):  Σ w (I − ĝĝᵀ) c = Σ w (I − ĝĝᵀ) r,  a 2×2 system.
    muu = w * (1.0 - u * u)
    mvv = w * (1.0 - v * v)
    muv = w * (-u * v)
    a11 = float(muu.sum());  a12 = float(muv.sum())
    a22 = float(mvv.sum())
    b1 = float((muu * xx + muv * yy).sum())
    b2 = float((muv * xx + mvv * yy).sum())
    det = a11 * a22 - a12 * a12
    if abs(det) < 1e-12:
        return None
    x_c = (a22 * b1 - a12 * b2) / det
    y_c = (a11 * b2 - a12 * b1) / det
    if not (np.isfinite(x_c) and np.isfinite(y_c)):
        return None
    return float(y_c), float(x_c)


def _moment_sigma(patch: np.ndarray, y_c: float, x_c: float, y0: int, x0: int) -> float:
    """σ estimate from background-subtracted second moments about (y_c, x_c).

    For a 2D Gaussian, E[r²] = 2σ². Background is the median of the
    patch border. Returns 0.0 when there is no positive signal.
    """
    p = np.asarray(patch, dtype=np.float64)
    border = np.concatenate([p[0, :], p[-1, :], p[1:-1, 0], p[1:-1, -1]])
    bg = float(np.median(border)) if border.size else float(p.min())
    w = np.clip(p - bg, 0.0, None)
    total = float(w.sum())
    if total <= 0.0:
        return 0.0
    yy, xx = np.mgrid[y0:y0 + p.shape[0], x0:x0 + p.shape[1]].astype(np.float64)
    r2 = (yy - y_c) ** 2 + (xx - x_c) ** 2
    return float(np.sqrt(max(float((w * r2).sum() / total) / 2.0, 1e-12)))


class RadialSymmetryDetector:
    name = "radial_symmetry"

    def fit(self, run_stack):  # noqa: D401
        """No-op — radial symmetry has no per-run state."""
        return None

    def detect(
        self,
        image: np.ndarray,
        labeled_mask: Optional[np.ndarray] = None,
        opts: Optional[DetectorOpts] = None,
    ) -> list[Focus]:
        from scipy.ndimage import gaussian_filter
        from skimage.feature import peak_local_max

        opts = opts or DetectorOpts()
        norm = normalise_image(image)
        if norm.size == 0 or float(norm.max()) <= 0.0:
            return []

        sm = gaussian_filter(
            norm.astype(np.float64), sigma=max(0.8, 0.8 * float(opts.min_sigma)),
        )
        thr = float(np.median(sm)) + float(opts.rs_threshold_mad) * robust_sigma_mad(sm)
        peaks = peak_local_max(
            sm,
            min_distance=max(1, int(round(opts.min_sigma * 2))),
            threshold_abs=thr,
            exclude_border=False,
        )
        if peaks.size == 0:
            return []

        img = np.asarray(image, dtype=np.float64)
        half = max(3, int(opts.refine_window))
        out: list[Focus] = []
        for y_pk, x_pk in peaks:
            y_lo = max(0, int(y_pk) - half)
            y_hi = min(img.shape[0], int(y_pk) + half + 1)
            x_lo = max(0, int(x_pk) - half)
            x_hi = min(img.shape[1], int(x_pk) + half + 1)
            patch = img[y_lo:y_hi, x_lo:x_hi]

            centre = radial_symmetry_center(patch, y_lo, x_lo)
            if centre is None:
                y_c, x_c = float(y_pk), float(x_pk)
            else:
                y_c, x_c = centre
                # Degenerate gradient fields can push the solution out of
                # the window — fall back to the candidate pixel.
                if not (y_lo - 1 <= y_c <= y_hi and x_lo - 1 <= x_c <= x_hi):
                    y_c, x_c = float(y_pk), float(x_pk)

            sigma_est = _moment_sigma(patch, y_c, x_c, y_lo, x_lo)
            if sigma_est <= 0.0:
                sigma_est = max(float(opts.min_sigma), 1.2)
            sigma_est = float(np.clip(sigma_est, 0.3, 1.5 * float(opts.max_sigma)))

            # refine=False: radial symmetry has already done the sub-pixel
            # step; build_focus contributes SNR gating + cell label.
            focus = build_focus(
                image, y_c, x_c, sigma_est,
                labeled_mask=labeled_mask,
                refine=False,
                snr_min=opts.snr_min,
            )
            if focus is not None:
                out.append(focus)
        # The symmetry solve can pull two candidates of an unresolved
        # doublet onto nearly the same point — collapse those to one.
        return suppress_close_foci(out, max(1.0, 2.0 * float(opts.min_sigma)))
