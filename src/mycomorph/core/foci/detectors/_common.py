"""Shared utilities used by multiple detectors.

- ``normalise_image``: rescale to [0, 1] so threshold parameters mean
  the same thing across image scales (uint16 raw vs float vs deconvolved).
- ``refine_gaussian_peak``: 2D Gaussian sub-pixel refinement of an
  integer-pixel peak.
- ``local_snr``: peak intensity / (background-std within a ring around
  the peak).
- ``label_at``: bilinear-rounded lookup of the cell label at a sub-pixel
  position.
- ``build_focus``: factory that wraps refinement + SNR + cell-label
  assignment + optional SNR cutoff.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .._types import Focus


def normalise_image(image: np.ndarray, p_lo: float = 1.0, p_hi: float = 99.9) -> np.ndarray:
    """Robust percentile rescale to [0, 1] in float32.

    Uses the same recipe as most foci detectors expect — the integer
    image scale (uint16 in MycoMorph) gets mapped so the ``threshold``
    parameter is interpretable as "fraction of dynamic range".
    """
    img = np.asarray(image, dtype=np.float32)
    if img.size == 0:
        return img
    lo, hi = np.percentile(img, [p_lo, p_hi])
    if hi <= lo:
        return np.zeros_like(img)
    out = (img - lo) / (hi - lo)
    return np.clip(out, 0.0, 1.0)


def suppress_close_foci(foci: list[Focus], min_dist: float) -> list[Focus]:
    """Greedy near-duplicate suppression — the brightest focus wins and
    anything within ``min_dist`` of an already-kept focus is dropped.

    Sub-pixel refinement can converge two nearby candidates onto (almost)
    the same peak, e.g. both flanks of an unresolved doublet; run this
    *after* refinement to collapse them back into one detection.
    """
    if len(foci) <= 1:
        return list(foci)
    from scipy.spatial import cKDTree

    # Greedy NMS via KD-tree adjacency (O(n log n) instead of the naive
    # O(n²) pairwise scan): walking candidates brightest-first, keeping
    # one suppresses its neighbours — identical output to checking each
    # candidate against the kept list.
    pts = np.array([[f.y, f.x] for f in foci], dtype=np.float64)
    pairs = cKDTree(pts).query_pairs(float(min_dist), output_type="ndarray")
    if pairs.size:
        # query_pairs includes distance == min_dist; the suppression
        # criterion is strictly-less-than, so drop boundary-equal pairs.
        d2 = ((pts[pairs[:, 0]] - pts[pairs[:, 1]]) ** 2).sum(axis=1)
        pairs = pairs[d2 < float(min_dist) ** 2]
    adjacency: dict[int, list[int]] = {}
    for a, b in pairs:
        adjacency.setdefault(int(a), []).append(int(b))
        adjacency.setdefault(int(b), []).append(int(a))

    order = sorted(
        range(len(foci)), key=lambda i: foci[i].intensity, reverse=True,
    )
    suppressed = np.zeros(len(foci), dtype=bool)
    kept: list[Focus] = []
    for i in order:
        if suppressed[i]:
            continue
        kept.append(foci[i])
        for j in adjacency.get(i, ()):
            suppressed[j] = True
    return kept


def robust_sigma_mad(image: np.ndarray) -> float:
    """Robust noise-σ estimate via median absolute deviation.

    The 1.4826 factor makes the MAD consistent with the standard
    deviation for Gaussian noise. Insensitive to the bright foci pixels
    that would inflate a plain ``std``.
    """
    flat = np.asarray(image, dtype=np.float64).ravel()
    if flat.size == 0:
        return 0.0
    med = float(np.median(flat))
    mad = float(np.median(np.abs(flat - med)))
    return 1.4826 * mad


def _gaussian_2d(coords, amp, x0, y0, sigma, offset):
    x, y = coords
    return offset + amp * np.exp(-(((x - x0) ** 2 + (y - y0) ** 2) / (2.0 * sigma * sigma)))


def refine_gaussian_peak(
    image: np.ndarray,
    y_peak: float,
    x_peak: float,
    half_window: int = 4,
    init_sigma: float = 1.5,
) -> tuple[float, float, float, float]:
    """Return ``(y_sub, x_sub, sigma_fit, amplitude)`` from a 2D Gaussian fit.

    Falls back to the input peak (with the init sigma and the peak intensity)
    if the fit fails or the window is too small.
    """
    import warnings
    from scipy.optimize import curve_fit, OptimizeWarning

    y_round, x_round = int(round(y_peak)), int(round(x_peak))
    y_lo = max(0, y_round - half_window)
    y_hi = min(image.shape[0], y_round + half_window + 1)
    x_lo = max(0, x_round - half_window)
    x_hi = min(image.shape[1], x_round + half_window + 1)
    patch = image[y_lo:y_hi, x_lo:x_hi].astype(np.float64)
    if patch.size < 9:
        return float(y_peak), float(x_peak), float(init_sigma), float(patch.max() if patch.size else 0.0)

    yy, xx = np.mgrid[y_lo:y_hi, x_lo:x_hi]
    p0 = (
        float(patch.max() - patch.min()),
        float(x_peak),
        float(y_peak),
        float(init_sigma),
        float(patch.min()),
    )
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", OptimizeWarning)
            popt, _ = curve_fit(
                _gaussian_2d,
                (xx.ravel(), yy.ravel()),
                patch.ravel(),
                p0=p0,
                maxfev=200,
            )
    except Exception:  # noqa: BLE001
        return float(y_peak), float(x_peak), float(init_sigma), float(patch.max())

    amp, x_fit, y_fit, sigma_fit, _offset = popt
    if not np.isfinite([amp, x_fit, y_fit, sigma_fit]).all():
        return float(y_peak), float(x_peak), float(init_sigma), float(patch.max())
    # Reject runaway fits — caller should fall back to integer peak.
    if (
        not (y_lo - 2 <= y_fit <= y_hi + 2)
        or not (x_lo - 2 <= x_fit <= x_hi + 2)
        or abs(sigma_fit) > 5 * init_sigma
        or abs(sigma_fit) < 0.3
    ):
        return float(y_peak), float(x_peak), float(init_sigma), float(patch.max())

    return float(y_fit), float(x_fit), float(abs(sigma_fit)), float(amp)


def local_snr(image: np.ndarray, y: float, x: float, sigma: float) -> float:
    """Peak amplitude / std of an annulus around the peak.

    The annulus is ``[2σ, 4σ]`` so it avoids the focus itself but stays
    close enough to capture local background variation.
    """
    y_round, x_round = int(round(y)), int(round(x))
    inner = max(2, int(round(2 * sigma)))
    outer = max(inner + 2, int(round(4 * sigma)))
    yy, xx = np.mgrid[
        max(0, y_round - outer): min(image.shape[0], y_round + outer + 1),
        max(0, x_round - outer): min(image.shape[1], x_round + outer + 1),
    ]
    if yy.size == 0:
        return 0.0
    r = np.hypot(yy - y_round, xx - x_round)
    annulus = (r >= inner) & (r <= outer)
    sub = image[yy.min():yy.max() + 1, xx.min():xx.max() + 1]
    bg = sub[annulus]
    if bg.size < 8:
        return 0.0
    bg_std = float(bg.std())
    bg_med = float(np.median(bg))
    peak_y = max(0, min(image.shape[0] - 1, y_round))
    peak_x = max(0, min(image.shape[1] - 1, x_round))
    peak_val = float(image[peak_y, peak_x])
    if bg_std < 1e-9:
        return 0.0
    return (peak_val - bg_med) / bg_std


def label_at(labeled_mask: Optional[np.ndarray], y: float, x: float) -> int:
    if labeled_mask is None:
        return 0
    yi = int(round(y)); xi = int(round(x))
    if not (0 <= yi < labeled_mask.shape[0] and 0 <= xi < labeled_mask.shape[1]):
        return 0
    return int(labeled_mask[yi, xi])


def build_focus(
    image: np.ndarray,
    y0: float,
    x0: float,
    init_sigma: float,
    *,
    labeled_mask: Optional[np.ndarray] = None,
    refine: bool = True,
    refine_window: int = 4,
    snr_min: float = 0.0,
    intensity_override: Optional[float] = None,
) -> Optional[Focus]:
    """Assemble a Focus around an integer-pixel peak.

    Returns ``None`` when the focus fails the SNR cutoff. ``intensity_override``
    lets caller pass a pre-computed peak intensity (e.g. when the detector
    has already done its own fit, like trackpy).
    """
    if refine:
        # Cheap pre-gate before the (comparatively expensive) Gaussian
        # fit: permissive detectors hand us thousands of noise candidates
        # that sit far below the SNR floor, and fitting every one of them
        # dominates detection time. Refinement only nudges the peak by a
        # pixel or two, so a candidate at less than half the floor never
        # climbs past it — gate at 0.5× to keep borderline candidates on
        # the accurate (post-fit) path.
        if snr_min > 0.0:
            if local_snr(image, y0, x0, init_sigma) < 0.5 * snr_min:
                return None
        y_sub, x_sub, sigma_fit, amp = refine_gaussian_peak(
            image, y0, x0, half_window=refine_window, init_sigma=init_sigma,
        )
    else:
        y_sub, x_sub, sigma_fit = float(y0), float(x0), float(init_sigma)
        yi = max(0, min(image.shape[0] - 1, int(round(y0))))
        xi = max(0, min(image.shape[1] - 1, int(round(x0))))
        amp = float(image[yi, xi])

    if intensity_override is not None:
        amp = float(intensity_override)

    snr = local_snr(image, y_sub, x_sub, sigma_fit)
    if snr < snr_min:
        return None

    return Focus(
        y=y_sub,
        x=x_sub,
        sigma=sigma_fit,
        intensity=amp,
        snr=snr,
        cell_label=label_at(labeled_mask, y_sub, x_sub),
    )
