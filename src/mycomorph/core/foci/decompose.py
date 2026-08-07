"""Two-Gaussian refit that splits merged focus pairs.

Two diffraction-limited foci closer than ~2σ (adjacent replisomes in a
small cell are the canonical case) merge into one detection under every
point detector in the registry — blob overlap-merging and local-maxima
spacing both resolve them as a single peak. That undercounts exactly the
per-cell focus-count distributions downstream analysis cares about.

This module is the light-weight alternative to full multi-emitter
fitting: for each detected focus, refit its patch with a two-Gaussian
mixture (shared σ — same PSF — and shared offset) seeded from the
single-Gaussian residual. The split is accepted only when the pair model
genuinely earns it:

- ΔR² over the single-Gaussian fit ≥ ``opts.split_min_improvement``,
- the pair model fits well in absolute terms (R² ≥ 0.5 — on noise blobs
  a two-Gaussian "improves" a near-zero single-Gaussian R² for free, so
  the relative criterion alone would double weak false positives),
- fitted separation is resolvable (≥ 1.5·``min_sigma``) but still inside
  the fit window,
- both amplitudes are positive and both children clear ``opts.snr_min``,
- no *different* already-detected focus lies near the segment between
  the two children (a detector that half-resolved the doublet, or a
  resolved neighbour the second Gaussian latched onto, would otherwise
  yield duplicates).

Anything else keeps the original detection, so the step is conservative:
enabling it can only refine counts upward where the evidence is strong.

Wired behind ``DetectorOpts.split_merged`` and applied *after* the
detector by the pipeline stage / live preview, so it composes with every
detector key.
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np

from ._types import DetectorOpts, Focus
from .detectors._common import _gaussian_2d, label_at, local_snr

# Absolute R² floor for the accepted two-Gaussian model. Genuine merged
# pairs fit ≥ ~0.9; noise blobs rarely exceed ~0.3 even with 8 params.
_R2_DOUBLE_FLOOR = 0.5


def _two_gaussian_2d(coords, a1, x1, y1, a2, x2, y2, sigma, offset):
    x, y = coords
    s2 = 2.0 * sigma * sigma
    return (
        offset
        + a1 * np.exp(-(((x - x1) ** 2 + (y - y1) ** 2) / s2))
        + a2 * np.exp(-(((x - x2) ** 2 + (y - y2) ** 2) / s2))
    )


def _point_segment_dist(
    py: float, px: float,
    ay: float, ax: float,
    by: float, bx: float,
) -> float:
    """Distance from point (py, px) to the segment (ay, ax)–(by, bx)."""
    vy, vx = by - ay, bx - ax
    length_sq = vy * vy + vx * vx
    if length_sq <= 0.0:
        return float(np.hypot(py - ay, px - ax))
    t = ((py - ay) * vy + (px - ax) * vx) / length_sq
    t = max(0.0, min(1.0, t))
    return float(np.hypot(py - (ay + t * vy), px - (ax + t * vx)))


def _r_squared(patch: np.ndarray, model: np.ndarray) -> float:
    ss_res = float(((patch - model) ** 2).sum())
    ss_tot = float(((patch - patch.mean()) ** 2).sum())
    if ss_tot < 1e-12:
        return 0.0
    return 1.0 - ss_res / ss_tot


def _patch_elongation(patch: np.ndarray) -> float:
    """Eigenvalue ratio (≥ 1) of the second-moment matrix of the patch's
    bright core — a merged pair is elongated along the pair axis, a
    single focus is round. Used as a cheap pre-screen so the expensive
    two-Gaussian machinery only runs on plausibly-merged candidates.

    Moments are restricted to pixels above half the (background-
    subtracted) patch maximum: bacterial cells are rods, so the
    cytoplasm inside the window is itself elongated and total-mass
    moments would flag nearly every in-cell focus. The half-max core
    isolates the focus/doublet shape from the cell body.
    """
    p = np.asarray(patch, dtype=np.float64)
    border = np.concatenate([p[0, :], p[-1, :], p[1:-1, 0], p[1:-1, -1]])
    bg = float(np.median(border)) if border.size else float(p.min())
    w = np.clip(p - bg, 0.0, None)
    w_max = float(w.max())
    if w_max <= 0.0:
        return 1.0
    w = np.where(w >= 0.5 * w_max, w, 0.0)
    # A diffraction-limited focus core (σ ≳ 1.2 px) covers ≥ ~7 px at
    # half-max; a noise spike's "core" is 1–3 scattered pixels. Tiny
    # cores can't be resolvable doublets — report round so the caller
    # skips them.
    if int(np.count_nonzero(w)) < 5:
        return 1.0
    total = float(w.sum())
    if total <= 0.0:
        return 1.0
    yy, xx = np.mgrid[0:p.shape[0], 0:p.shape[1]].astype(np.float64)
    cy = float((w * yy).sum() / total)
    cx = float((w * xx).sum() / total)
    myy = float((w * (yy - cy) ** 2).sum() / total)
    mxx = float((w * (xx - cx) ** 2).sum() / total)
    mxy = float((w * (yy - cy) * (xx - cx)).sum() / total)
    trace = myy + mxx
    disc = (myy - mxx) ** 2 + 4.0 * mxy * mxy
    if disc < 0.0:
        return 1.0
    sqrt_d = float(np.sqrt(disc))
    lam_hi = 0.5 * (trace + sqrt_d)
    lam_lo = 0.5 * (trace - sqrt_d)
    if lam_lo <= 1e-9:
        return 1.0
    return lam_hi / lam_lo


def _try_split(
    image: np.ndarray,
    focus: Focus,
    opts: DetectorOpts,
) -> Optional[tuple[tuple[float, float, float], tuple[float, float, float], float]]:
    """Attempt the two-Gaussian refit for one focus.

    Returns ``((y1, x1, amp1), (y2, x2, amp2), sigma)`` when the pair
    model wins, else ``None``.
    """
    from scipy.optimize import curve_fit, OptimizeWarning

    sigma0 = max(float(focus.sigma), float(opts.min_sigma), 0.8)
    half = min(12, max(int(opts.refine_window) + 2, int(np.ceil(3.0 * sigma0))))
    y_r, x_r = int(round(focus.y)), int(round(focus.x))
    y_lo = max(0, y_r - half); y_hi = min(image.shape[0], y_r + half + 1)
    x_lo = max(0, x_r - half); x_hi = min(image.shape[1], x_r + half + 1)
    patch = image[y_lo:y_hi, x_lo:x_hi].astype(np.float64)
    if patch.shape[0] < 5 or patch.shape[1] < 5:
        return None

    # Pre-screen: only elongated bright cores can be merged pairs.
    # Measured on synthetic σ≈1.3 spots (noise σ 4): singles' half-max
    # moment ratio runs p50 1.45 / p99 2.6 (small-sample pixel moments
    # are that noisy), resolvable pairs (separation ≥ 3 px ≈ the
    # widefield Rayleigh limit) sit above 3.4. The 2.4 cut skips ~99%
    # of singles; pairs closer than ~2.5 px fall below it, but those
    # are sub-resolution — splitting them would be overfitting, and
    # they were already marginal against the ΔR²/separation criteria.
    if _patch_elongation(patch) < 2.4:
        return None

    yy, xx = np.mgrid[y_lo:y_hi, x_lo:x_hi].astype(np.float64)
    coords = (xx.ravel(), yy.ravel())

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", OptimizeWarning)

        # Single-Gaussian baseline (gives R²_1 and the residual seed).
        p0_single = (
            float(patch.max() - patch.min()),
            float(focus.x), float(focus.y),
            sigma0,
            float(patch.min()),
        )
        try:
            popt1, _ = curve_fit(
                _gaussian_2d, coords, patch.ravel(), p0=p0_single, maxfev=300,
            )
        except Exception:  # noqa: BLE001
            return None
        amp1_s, x_s, y_s, sig_s, off_s = (float(v) for v in popt1)
        model1 = _gaussian_2d(coords, *popt1).reshape(patch.shape)
        r2_single = _r_squared(patch, model1)

        # Seed the second component at the strongest residual pixel.
        residual = patch - model1
        ry, rx = np.unravel_index(int(np.argmax(residual)), residual.shape)
        amp2_seed = float(residual[ry, rx])
        if amp2_seed <= 0.0:
            return None

        sig_seed = float(np.clip(abs(sig_s), 0.5, 2.0 * float(opts.max_sigma)))
        p0_double = (
            max(amp1_s, 1e-6), x_s, y_s,
            amp2_seed, float(x_lo + rx), float(y_lo + ry),
            sig_seed, off_s,
        )
        bounds = (
            [0.0, x_lo - 1.0, y_lo - 1.0, 0.0, x_lo - 1.0, y_lo - 1.0,
             0.3, -np.inf],
            [np.inf, x_hi + 1.0, y_hi + 1.0, np.inf, x_hi + 1.0, y_hi + 1.0,
             2.0 * float(opts.max_sigma), np.inf],
        )
        try:
            popt2, _ = curve_fit(
                _two_gaussian_2d, coords, patch.ravel(),
                p0=p0_double, bounds=bounds, maxfev=600,
            )
        except Exception:  # noqa: BLE001
            return None

    a1, x1, y1, a2, x2, y2, sig2, _off2 = (float(v) for v in popt2)
    if not np.isfinite([a1, x1, y1, a2, x2, y2, sig2]).all():
        return None
    model2 = _two_gaussian_2d(coords, *popt2).reshape(patch.shape)
    r2_double = _r_squared(patch, model2)

    if r2_double < _R2_DOUBLE_FLOOR:
        return None
    if (r2_double - r2_single) < float(opts.split_min_improvement):
        return None
    separation = float(np.hypot(x1 - x2, y1 - y2))
    min_sep = max(1.5, 1.5 * float(opts.min_sigma))
    if separation < min_sep or separation > 2.0 * half:
        return None
    if a1 <= 0.0 or a2 <= 0.0:
        return None
    return (y1, x1, a1), (y2, x2, a2), sig2


def split_merged_foci(
    image: np.ndarray,
    foci: list[Focus],
    opts: Optional[DetectorOpts] = None,
    labeled_mask: Optional[np.ndarray] = None,
) -> list[Focus]:
    """Return ``foci`` with confidently-merged pairs replaced by two foci.

    Order is preserved; children take the parent's slot. Foci whose
    two-Gaussian refit does not clear every acceptance criterion pass
    through unchanged.
    """
    opts = opts or DetectorOpts()
    if not foci:
        return list(foci)

    img = np.asarray(image, dtype=np.float64)
    min_sep = max(1.5, 1.5 * float(opts.min_sigma))
    out: list[Focus] = []
    for i, focus in enumerate(foci):
        split = _try_split(img, focus, opts)
        if split is None:
            out.append(focus)
            continue
        (y1, x1, a1), (y2, x2, a2), sig2 = split

        # Reject when a different, already-existing detection lies near
        # the segment between the two children: either the second
        # Gaussian latched onto a resolved neighbour, or the detector
        # half-resolved this doublet itself (a second detection sitting
        # between the children) — splitting would duplicate it.
        near_other = any(
            _point_segment_dist(other.y, other.x, y1, x1, y2, x2) < min_sep
            for j, other in enumerate(foci)
            if j != i
        )
        if near_other:
            out.append(focus)
            continue

        children: list[Focus] = []
        for yc, xc, amp in ((y1, x1, a1), (y2, x2, a2)):
            if not (0 <= yc < image.shape[0] and 0 <= xc < image.shape[1]):
                break
            snr = local_snr(img, yc, xc, sig2)
            if snr < float(opts.snr_min):
                break
            children.append(Focus(
                y=float(yc), x=float(xc),
                sigma=float(sig2),
                intensity=float(amp),
                snr=float(snr),
                cell_label=label_at(labeled_mask, yc, xc),
            ))
        if len(children) == 2:
            out.extend(children)
        else:
            out.append(focus)
    return out
