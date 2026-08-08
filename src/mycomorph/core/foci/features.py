"""Per-focus feature computation for filtering / classification.

Detection alone uses 1–2 features (intensity, scale). Many false
positives (dim cytoplasmic bumps, segmentation-edge artefacts,
single-pixel salt-noise) survive any single-feature threshold because
they share one property with real foci while differing on another.

A high-recall detector (e.g., wavelet at low ``threshold_mad``) generates
many candidates, most of which aren't foci. These features score each
candidate across multiple complementary criteria. Real foci tend to
score well on **all** of them; specific failure modes score well on
some and badly on others:

- ``sigma``                — fitted Gaussian σ (PSF-plausible range)
- ``intensity``            — peak amplitude
- ``snr_annulus``          — peak / std of annulus around peak (from detector)
- ``patch_snr``            — peak / std of patch perimeter
- ``gaussian_r2``          — R² of 2D Gaussian fit to the patch
  (real foci ≈ 0.7–0.95; salt noise / edges drop below 0.5)
- ``hessian_symmetry``     — |λ_min| / |λ_max| of the Hessian at peak
  (1.0 = perfectly round; 0.0 = streak/edge)
- ``cell_*``               — per-cell intensity statistics
- ``prominence_p90``       — peak / cell's p90 pixel (false-positive killer
  for cells with no real focus)

Each feature is computed from a small patch around the candidate and
the cell's pixel mask. The output DataFrame plugs into either:

(a) hand-tuned thresholds — set ``prominence_p90 > 1.3 AND
    gaussian_r2 > 0.5 AND hessian_symmetry > 0.4``, etc., or

(b) a trained classifier — label a subset of patches yes/no, train a
    random forest or small CNN on the features, predict on the rest.
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd

from ._types import Focus


def extract_patch(
    image: np.ndarray,
    focus: Focus,
    half_size: int = 8,
) -> tuple[np.ndarray, tuple[int, int]]:
    """Return a ``(2*half_size+1, 2*half_size+1)`` patch and its ``(y0, x0)`` origin.

    The patch is clipped at the image edges, so corner foci yield smaller
    asymmetric patches — downstream features handle this gracefully.
    """
    yc = int(round(focus.y))
    xc = int(round(focus.x))
    y0 = max(0, yc - half_size); y1 = min(image.shape[0], yc + half_size + 1)
    x0 = max(0, xc - half_size); x1 = min(image.shape[1], xc + half_size + 1)
    return image[y0:y1, x0:x1].astype(np.float64), (y0, x0)


def _gauss_2d(coords, amp, x0, y0, sigma, offset):
    x, y = coords
    return offset + amp * np.exp(-(((x - x0) ** 2 + (y - y0) ** 2) / (2.0 * sigma * sigma)))


def gaussian_2d_fit_r2(patch: np.ndarray, sigma_init: float = 1.4) -> float:
    """R² of a 2D Gaussian fit to the patch.

    Real diffraction-limited foci fit a Gaussian well — R² typically
    0.7–0.95 (limited by Poisson noise). Salt noise, edges, and broad
    cytoplasmic blobs all fit poorly (R² < 0.5).
    """
    from scipy.optimize import curve_fit, OptimizeWarning

    if patch.size < 9 or patch.shape[0] < 5 or patch.shape[1] < 5:
        return 0.0

    yy, xx = np.mgrid[:patch.shape[0], :patch.shape[1]]
    yc_init = patch.shape[0] / 2.0
    xc_init = patch.shape[1] / 2.0
    p0 = (
        float(patch.max() - patch.min()),
        float(xc_init), float(yc_init),
        float(sigma_init),
        float(patch.min()),
    )
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", OptimizeWarning)
            popt, _ = curve_fit(
                _gauss_2d,
                (xx.ravel(), yy.ravel()),
                patch.ravel(),
                p0=p0, maxfev=300,
            )
    except Exception:  # noqa: BLE001
        return 0.0

    fitted = _gauss_2d((xx.ravel(), yy.ravel()), *popt).reshape(patch.shape)
    ss_res = float(((patch - fitted) ** 2).sum())
    ss_tot = float(((patch - patch.mean()) ** 2).sum())
    if ss_tot < 1e-9:
        return 0.0
    r2 = 1.0 - ss_res / ss_tot
    return float(max(0.0, min(1.0, r2)))


def hessian_eigenvalue_ratio(patch: np.ndarray) -> float:
    """``|λ_min| / |λ_max|`` of the Hessian at the patch centre.

    1.0 = perfectly symmetric (round Gaussian focus).
    0.0 = streak/edge — one direction dominates curvature.
    Useful for rejecting elongated artefacts (membrane edges, line noise).
    """
    if patch.shape[0] < 3 or patch.shape[1] < 3:
        return 0.0

    yc, xc = patch.shape[0] // 2, patch.shape[1] // 2
    p = patch
    H_yy = p[yc + 1, xc] - 2.0 * p[yc, xc] + p[yc - 1, xc]
    H_xx = p[yc, xc + 1] - 2.0 * p[yc, xc] + p[yc, xc - 1]
    H_xy = 0.25 * (p[yc + 1, xc + 1] - p[yc + 1, xc - 1]
                   - p[yc - 1, xc + 1] + p[yc - 1, xc - 1])

    trace = H_yy + H_xx
    det = H_yy * H_xx - H_xy * H_xy
    discriminant = trace * trace - 4.0 * det
    if discriminant < 0:
        return 0.0
    sqrt_d = float(np.sqrt(discriminant))
    lam_a = 0.5 * (trace - sqrt_d)
    lam_b = 0.5 * (trace + sqrt_d)
    a_max = max(abs(lam_a), abs(lam_b))
    if a_max < 1e-9:
        return 0.0
    return float(min(abs(lam_a), abs(lam_b)) / a_max)


def patch_snr(patch: np.ndarray, exclude_centre_radius: int = 2) -> float:
    """Peak (centre) / std of patch perimeter pixels (centre region excluded)."""
    if patch.size == 0:
        return 0.0
    yc, xc = patch.shape[0] // 2, patch.shape[1] // 2
    yy, xx = np.mgrid[:patch.shape[0], :patch.shape[1]]
    r2 = (yy - yc) ** 2 + (xx - xc) ** 2
    bg_mask = r2 > (exclude_centre_radius ** 2)
    if bg_mask.sum() < 4:
        return 0.0
    bg_pixels = patch[bg_mask]
    bg_std = float(bg_pixels.std())
    bg_med = float(np.median(bg_pixels))
    if bg_std < 1e-9:
        return 0.0
    return float((patch[yc, xc] - bg_med) / bg_std)


def cell_stats_from_values(cell_values: np.ndarray) -> dict[str, float]:
    """Per-cell intensity statistics consumed by :func:`compute_features`."""
    return {
        "cell_p90":    float(np.percentile(cell_values, 90)),
        "cell_p95":    float(np.percentile(cell_values, 95)),
        "cell_median": float(np.median(cell_values)),
        "cell_std":    float(cell_values.std()),
    }


def compute_features(
    image: np.ndarray,
    focus: Focus,
    *,
    patch_half_size: int = 8,
    cell_pixels: Optional[np.ndarray] = None,
    cell_stats: Optional[dict[str, float]] = None,
) -> dict[str, float]:
    """Compute all features for one candidate Focus.

    Cell-relative features come from ``cell_stats`` (a precomputed
    :func:`cell_stats_from_values` dict — the fast path used by
    :func:`features_dataframe`, which computes each cell's stats once)
    or, failing that, from ``cell_pixels`` (a boolean mask of the
    containing cell — recomputes the stats for this call). With
    neither, cell-relative features are NaN.
    """
    patch, _ = extract_patch(image, focus, half_size=patch_half_size)

    features: dict[str, float] = {
        "sigma":             float(focus.sigma),
        "intensity":         float(focus.intensity),
        "snr_annulus":       float(focus.snr),
        "patch_snr":         patch_snr(patch),
        "gaussian_r2":       gaussian_2d_fit_r2(patch, sigma_init=float(focus.sigma)),
        "hessian_symmetry":  hessian_eigenvalue_ratio(patch),
    }

    if cell_stats is None and cell_pixels is not None and cell_pixels.any():
        cell_stats = cell_stats_from_values(
            image[cell_pixels].astype(np.float64)
        )
    if cell_stats is not None:
        features.update(cell_stats)
        if features["cell_p90"] > 0:
            features["prominence_p90"] = features["intensity"] / features["cell_p90"]
        else:
            features["prominence_p90"] = 0.0
        if features["cell_median"] > 0:
            features["prominence_median"] = features["intensity"] / features["cell_median"]
        else:
            features["prominence_median"] = 0.0
    else:
        for key in ("cell_p90", "cell_p95", "cell_median", "cell_std",
                    "prominence_p90", "prominence_median"):
            features[key] = float("nan")

    return features


def features_dataframe(
    image: np.ndarray,
    labeled_mask: np.ndarray,
    foci: list[Focus],
    *,
    patch_half_size: int = 8,
    detector: str = "",
    well: str = "",
    fov_index: int = 0,
    channel: str = "",
) -> pd.DataFrame:
    """Compute features for every focus and return a long-form DataFrame.

    Each row also carries identification (detector / well / fov / channel /
    cell_label / y / x) so rows can be joined back to detections after
    classification or threshold filtering.

    Per-cell statistics are computed once per cell on its bounding-box
    crop — a full-frame mask comparison per focus made this the
    dominant cost on large FOVs (O(image area × n_foci)).
    """
    from scipy import ndimage

    cell_slices = ndimage.find_objects(labeled_mask) if foci else []
    cell_stats_cache: dict[int, Optional[dict[str, float]]] = {}

    def _stats_for(cid: int) -> Optional[dict[str, float]]:
        if cid in cell_stats_cache:
            return cell_stats_cache[cid]
        stats: Optional[dict[str, float]] = None
        if 0 < cid <= len(cell_slices) and cell_slices[cid - 1] is not None:
            sl = cell_slices[cid - 1]
            in_cell = labeled_mask[sl] == cid
            if in_cell.any():
                stats = cell_stats_from_values(
                    image[sl][in_cell].astype(np.float64)
                )
        cell_stats_cache[cid] = stats
        return stats

    rows: list[dict] = []
    for f in foci:
        cid = int(f.cell_label)
        feats = compute_features(
            image, f,
            patch_half_size=patch_half_size,
            cell_stats=_stats_for(cid) if cid > 0 else None,
        )
        feats.update({
            "detector":  detector,
            "well":      well,
            "fov_index": int(fov_index),
            "channel":   channel,
            "cell_label": cid,
            "y":         float(f.y),
            "x":         float(f.x),
        })
        rows.append(feats)
    return pd.DataFrame(rows)


def extract_patches_array(
    image: np.ndarray,
    foci: list[Focus],
    patch_half_size: int = 8,
) -> np.ndarray:
    """Return a ``(N, 2H+1, 2H+1)`` float32 array of patches, one per focus.

    Patches are zero-padded when a focus is near the image edge. Useful
    as direct input to a CNN classifier.
    """
    side = 2 * patch_half_size + 1
    out = np.zeros((len(foci), side, side), dtype=np.float32)
    for i, f in enumerate(foci):
        patch, _ = extract_patch(image, f, half_size=patch_half_size)
        if patch.shape == (side, side):
            out[i] = patch.astype(np.float32)
        else:
            # Edge case: pad to fixed size
            py, px = patch.shape
            y_off = (side - py) // 2
            x_off = (side - px) // 2
            out[i, y_off:y_off + py, x_off:x_off + px] = patch.astype(np.float32)
    return out
