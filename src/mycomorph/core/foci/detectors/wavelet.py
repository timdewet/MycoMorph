"""À trous wavelet (undecimated B3-spline) detector for dim foci.

Reference: Olivo-Marin, *Pattern Recognition* 35:1989–1996 (2002),
"Extraction of spots in biological images using multiscale products".
The same engine sits inside CellProfiler's ``SpeckleCounting`` and the
"LoG-via-wavelet" backend of Big-FISH.

Why it beats DoG/LoG on dim signal:

- Background variation lives in the **low-frequency** wavelet planes,
  which are discarded — no per-image threshold tuning required.
- Noise is suppressed **per scale** by a robust MAD threshold, so a
  scale that contains genuine signal isn't penalised by noise at a
  different scale.
- Foci-sized features survive intact with their full peak amplitude
  even when the SNR vs. raw background is marginal.

Algorithm:

1. Convolve the image with the separable B3-spline kernel ``[1 4 6 4 1] / 16``
   at dyadic scales ``2^k`` (à trous = "with holes" — kernel taps spaced
   ``2^k`` pixels apart, no decimation).
2. Wavelet plane ``W_k = I_{k-1} - I_k``.
3. Per-plane MAD-based threshold ``T_k = threshold_mad * 1.4826 * MAD(W_k)``.
   Zero out everything in ``W_k`` below ``T_k``.
4. Sum the thresholded foci-size planes (default ``k = 1, 2`` — i.e.
   features 2–6 px wide, matching the Mtb widefield PSF).
5. Local maxima of the summed map are foci.
6. Sub-pixel refine with a 2D Gaussian fit on the raw image (so peak
   coordinates and SNR are reported on the original scale).
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from .._types import DetectorOpts, Focus
from ._common import build_focus, normalise_image


def _atrous_kernel_1d() -> np.ndarray:
    return np.array([1, 4, 6, 4, 1], dtype=np.float64) / 16.0


def _atrous_decompose(
    image: np.ndarray,
    n_scales: int,
) -> tuple[list[np.ndarray], np.ndarray]:
    """Return ``(planes, residual)`` where ``sum(planes) + residual == image``.

    Each ``planes[k]`` is the wavelet plane ``I_k - I_{k+1}`` for the
    dyadically dilated B3-spline (à trous). The decomposition is
    undecimated, so every plane has the same shape as the input.
    """
    from scipy.ndimage import convolve1d

    h = _atrous_kernel_1d()
    smoothed = [image.astype(np.float64, copy=False)]
    planes: list[np.ndarray] = []
    for k in range(n_scales):
        step = 2 ** k
        if step == 1:
            kernel = h
        else:
            kernel = np.zeros(len(h) + (len(h) - 1) * (step - 1))
            kernel[::step] = h
        smoothed_next = convolve1d(smoothed[-1], kernel, axis=0, mode="reflect")
        smoothed_next = convolve1d(smoothed_next, kernel, axis=1, mode="reflect")
        planes.append(smoothed[-1] - smoothed_next)
        smoothed.append(smoothed_next)
    return planes, smoothed[-1]


def _mad_std(plane: np.ndarray) -> float:
    """Robust σ estimate via median absolute deviation (Gaussian-consistent)."""
    flat = plane.ravel()
    med = float(np.median(flat))
    mad = float(np.median(np.abs(flat - med)))
    return 1.4826 * mad


class WaveletAtrousDetector:
    """Olivo-Marin 2002 à trous wavelet spot detector.

    Default ``scales = (1, 2)`` covers feature widths ~2–6 px, the right
    band for a widefield diffraction-limited Mtb focus (PSF σ ≈ 1.3–1.8 px).
    Adjust ``scales`` higher for larger foci or wider PSFs.
    """

    name = "wavelet"

    def __init__(
        self,
        scales: Sequence[int] = (1, 2),
        threshold_mad: float = 3.0,
        n_scales_total: int = 4,
    ) -> None:
        if not scales:
            raise ValueError("WaveletAtrousDetector: need at least one scale")
        if min(scales) < 0:
            raise ValueError("WaveletAtrousDetector: scales must be non-negative")
        self._scales = tuple(int(s) for s in scales)
        self._threshold_mad = float(threshold_mad)
        self._n_scales = max(int(n_scales_total), max(self._scales) + 1)

    def fit(self, run_stack):  # noqa: D401
        """No-op — wavelet has no per-run state."""
        return None

    def _foci_response(
        self,
        image: np.ndarray,
        scales: Optional[Sequence[int]] = None,
        threshold_mad: Optional[float] = None,
    ) -> np.ndarray:
        """Build the recombined foci response map.

        Public-ish so notebooks can visualise the intermediate map.
        """
        scales = tuple(scales) if scales is not None else self._scales
        thr_mad = self._threshold_mad if threshold_mad is None else float(threshold_mad)
        n_scales = max(self._n_scales, max(scales) + 1)

        norm = normalise_image(image)
        planes, _ = _atrous_decompose(norm, n_scales=n_scales)
        recombined = np.zeros_like(norm, dtype=np.float64)
        for k in scales:
            if k >= len(planes):
                continue
            plane = planes[k]
            thr = thr_mad * _mad_std(plane)
            recombined += np.where(plane > thr, plane, 0.0)
        return recombined

    def detect(
        self,
        image: np.ndarray,
        labeled_mask: Optional[np.ndarray] = None,
        opts: Optional[DetectorOpts] = None,
    ) -> list[Focus]:
        from skimage.feature import peak_local_max

        opts = opts or DetectorOpts()
        # DetectorOpts overrides the constructor defaults when provided.
        scales = tuple(opts.wavelet_scales) if opts.wavelet_scales else self._scales
        thr_mad = float(opts.wavelet_threshold_mad)
        response = self._foci_response(image, scales=scales, threshold_mad=thr_mad)

        if response.max() <= 0:
            return []

        peaks = peak_local_max(
            response,
            min_distance=max(1, int(round(opts.min_sigma * 2))),
            threshold_abs=0.0,
            exclude_border=False,
        )

        out: list[Focus] = []
        init_sigma = max(opts.min_sigma, 1.2)
        for peak in peaks:
            focus = build_focus(
                image,
                float(peak[0]), float(peak[1]),
                init_sigma=init_sigma,
                labeled_mask=labeled_mask,
                refine=opts.refine,
                refine_window=opts.refine_window,
                snr_min=opts.snr_min,
            )
            if focus is not None:
                out.append(focus)
        return out
