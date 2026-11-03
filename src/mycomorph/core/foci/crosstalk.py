"""Spectral crosstalk estimation and subtraction between fluor channels.

Bright structures in one fluorophore's channel can bleed a proportional
ghost into another channel (e.g. mCherry foci leaking ~5-10% into an
EGFP acquisition). Because the foci detectors threshold *relative* to
local background, even a few-percent ghost over a flat background is
detected as a genuine focus — producing phantom detections that sit
sub-pixel-perfectly on the source channel's foci.

The correction model is a single linear coefficient per channel pair:

    target_corrected = max(target - k * source, 0)

``estimate_crosstalk_k`` recovers ``k`` from the images themselves: it
finds bright peaks in the source channel, measures the local
background-subtracted amplitude of BOTH channels at those sites, and
takes a robust (Theil–Sen) slope of target-vs-source amplitude. Genuine
biological colocalisation at a minority of sites shows up as outliers
above the crosstalk line and does not move the median-based slope.

Identifiability limit: when the two reporters colocalise at *most*
source peaks (e.g. DnaN-mCherry with ImuB-eGFP under DNA damage), the
median slope IS the colocalisation slope and auto estimation cannot
tell it apart from bleed — correlation quality doesn't help, because
biology correlates at least as well as optics. Real filter-set bleed
is small (~5–10%), so a fitted slope above ``max_plausible_k`` is
treated as biology and auto estimation refuses (``k = 0``,
``mode = "implausible-slope"``); measure a fixed ``k`` from
single-fluorophore control wells instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "CrosstalkEstimate",
    "estimate_crosstalk_k",
    "subtract_crosstalk",
]


@dataclass
class CrosstalkEstimate:
    """Result of a per-well crosstalk fit (or a fixed user coefficient)."""

    k: float
    n_sites: int = 0
    spearman_rho: float = float("nan")
    # "auto" | "fixed" | "insufficient-sites" | "implausible-slope"
    mode: str = "auto"
    # Raw Theil–Sen slope before any plausibility decision — kept even
    # when the estimate is refused, so the sidecar shows what auto saw.
    fitted_slope: float = float("nan")
    # Per-FOV site counts, for the sidecar JSON / debugging.
    sites_per_fov: list[int] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "k": float(self.k),
            "n_sites": int(self.n_sites),
            "spearman_rho": (
                None if np.isnan(self.spearman_rho) else float(self.spearman_rho)
            ),
            "mode": self.mode,
            "fitted_slope": (
                None if np.isnan(self.fitted_slope) else float(self.fitted_slope)
            ),
            "sites_per_fov": [int(n) for n in self.sites_per_fov],
        }


def _local_amplitude(
    img: np.ndarray, yx: np.ndarray, r_in: int = 2, r_out: int = 8,
) -> np.ndarray:
    """Peak-minus-annulus amplitude at each (y, x) site.

    Peak is the 3x3 max around the site; background is the median of the
    (2*r_out+1)^2 window with the inner (2*r_in+1)^2 core masked out.
    Sites closer than ``r_out`` to the frame edge get NaN.
    """
    H, W = img.shape
    out = np.full(len(yx), np.nan, dtype=np.float32)
    for i, (y, x) in enumerate(yx):
        y, x = int(y), int(x)
        if not (r_out <= y < H - r_out and r_out <= x < W - r_out):
            continue
        peak = float(img[y - 1:y + 2, x - 1:x + 2].max())
        ring = img[y - r_out:y + r_out + 1, x - r_out:x + r_out + 1].astype(
            np.float32, copy=True,
        )
        ring[r_out - r_in:r_out + r_in + 1, r_out - r_in:r_out + r_in + 1] = np.nan
        out[i] = peak - float(np.nanmedian(ring))
    return out


def _source_peaks(
    frame: np.ndarray,
    mask: np.ndarray | None,
    *,
    tophat_radius_px: int,
    min_distance: int,
    max_sites: int,
) -> np.ndarray:
    """Bright, isolated peaks in one source-channel frame → (N, 2) [y, x].

    Peaks are found on the white top-hat of the frame (so slow-varying
    background doesn't rank sites) with a robust MAD-based floor, and are
    restricted to ``mask`` when given. The brightest ``max_sites`` are
    returned.
    """
    from skimage.feature import peak_local_max
    from skimage.morphology import disk, white_tophat

    img = frame.astype(np.float32, copy=False)
    th = white_tophat(img, footprint=disk(int(tophat_radius_px)))
    sel = th[mask] if (mask is not None and mask.any()) else th.ravel()
    med = float(np.median(sel))
    mad = float(np.median(np.abs(sel - med))) or 1e-6
    floor = med + 6.0 * 1.4826 * mad
    coords = peak_local_max(
        th,
        min_distance=int(min_distance),
        threshold_abs=floor,
        num_peaks=int(max_sites),
        labels=(mask.astype(np.uint8) if mask is not None else None),
    )
    return coords


def estimate_crosstalk_k(
    source_stack: np.ndarray,
    target_stack: np.ndarray,
    mask_stack: np.ndarray | None = None,
    *,
    tophat_radius_px: int = 5,
    min_distance: int = 5,
    max_sites_per_fov: int = 300,
    min_sites: int = 20,
    max_plausible_k: float = 0.25,
) -> CrosstalkEstimate:
    """Estimate the source→target bleed coefficient for one well.

    ``source_stack`` / ``target_stack`` are (N, Y, X) FOV stacks of the
    raw channels (2D single frames are accepted too). ``mask_stack``
    optionally restricts peak sites to in-cell pixels — recommended, so
    extracellular debris that is genuinely bright in both channels
    doesn't inflate the fit.

    Returns ``k = 0`` with ``mode = "insufficient-sites"`` when fewer
    than ``min_sites`` usable peaks exist (e.g. a source channel with no
    foci) — subtraction with k=0 is then a safe no-op.

    Returns ``k = 0`` with ``mode = "implausible-slope"`` when the
    fitted slope exceeds ``max_plausible_k``: filter-set bleed runs a
    few percent, so a large slope means the reporters colocalise and
    the estimate would subtract genuine signal (observed in production:
    DnaN-mCherry/ImuB-eGFP fitted slope ≥ 1.0 with ρ = 0.63, which
    wiped the EGFP channel and downstream detection exploded on the
    clipped residue). The refused slope is kept in ``fitted_slope``.
    """
    from scipy.stats import spearmanr, theilslopes

    src = np.asarray(source_stack)
    tgt = np.asarray(target_stack)
    if src.ndim == 2:
        src = src[None]
        tgt = tgt[None]
        if mask_stack is not None and np.asarray(mask_stack).ndim == 2:
            mask_stack = np.asarray(mask_stack)[None]
    if src.shape != tgt.shape:
        raise ValueError(
            f"source/target shapes differ: {src.shape} vs {tgt.shape}"
        )

    src_amps: list[np.ndarray] = []
    tgt_amps: list[np.ndarray] = []
    sites_per_fov: list[int] = []
    for i in range(src.shape[0]):
        mask = None
        if mask_stack is not None:
            mask = np.asarray(mask_stack[i]) > 0
            if not mask.any():
                mask = None
        coords = _source_peaks(
            src[i], mask,
            tophat_radius_px=tophat_radius_px,
            min_distance=min_distance,
            max_sites=max_sites_per_fov,
        )
        sites_per_fov.append(len(coords))
        if not len(coords):
            continue
        a_src = _local_amplitude(src[i], coords)
        a_tgt = _local_amplitude(tgt[i], coords)
        ok = np.isfinite(a_src) & np.isfinite(a_tgt) & (a_src > 0)
        src_amps.append(a_src[ok])
        tgt_amps.append(a_tgt[ok])

    if not src_amps:
        return CrosstalkEstimate(
            k=0.0, n_sites=0, mode="insufficient-sites",
            sites_per_fov=sites_per_fov,
        )
    a_src = np.concatenate(src_amps)
    a_tgt = np.concatenate(tgt_amps)
    if len(a_src) < min_sites:
        return CrosstalkEstimate(
            k=0.0, n_sites=len(a_src), mode="insufficient-sites",
            sites_per_fov=sites_per_fov,
        )

    slope = float(theilslopes(a_tgt, a_src)[0])
    rho = float(spearmanr(a_src, a_tgt)[0])
    if slope > float(max_plausible_k):
        return CrosstalkEstimate(
            k=0.0, n_sites=len(a_src), spearman_rho=rho,
            mode="implausible-slope", fitted_slope=slope,
            sites_per_fov=sites_per_fov,
        )
    # Negative slopes mean "no measurable bleed" (noise); never add
    # signal back.
    k = float(max(slope, 0.0))
    return CrosstalkEstimate(
        k=k, n_sites=len(a_src), spearman_rho=rho, mode="auto",
        fitted_slope=slope, sites_per_fov=sites_per_fov,
    )


def subtract_crosstalk(
    target: np.ndarray, source: np.ndarray, k: float,
) -> np.ndarray:
    """``max(target - k*max(source - median(source), 0), 0)`` as float32.

    ``k`` scales the source's *signal above its per-frame median*, not
    the raw frame: bleed is proportional to source photons, while the
    raw frame also carries the camera offset (and diffuse floor), which
    the target channel owns outright. Subtracting ``k * raw`` used to
    take ``k × offset`` out of the whole target frame — harmless at
    k ≈ 0.075 under top-hat, but at large k it drove the background
    negative and the clip to zero turned entire FOVs into sparse noise
    speckle. Scaling only above-median signal means no ``k`` can zero a
    region the source is dark in. This also matches the amplitude space
    ``estimate_crosstalk_k`` fits in (peak minus local background).

    ``source`` may be one frame (Y, X) or a stack (N, Y, X); the median
    is taken per frame. Shapes must match.
    """
    tgt = np.asarray(target, dtype=np.float32)
    src = np.asarray(source, dtype=np.float32)
    if tgt.shape != src.shape:
        raise ValueError(
            f"target/source shapes differ: {tgt.shape} vs {src.shape}"
        )
    if not k:
        return tgt.copy()
    src_bg = np.median(src, axis=(-2, -1), keepdims=True).astype(np.float32)
    ghost = np.clip(src - src_bg, 0.0, None)
    return np.clip(tgt - np.float32(k) * ghost, 0.0, None)
