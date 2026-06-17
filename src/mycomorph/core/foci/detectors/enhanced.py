"""Detectors that stack preprocessing (denoise / deconvolve) in front
of the à trous wavelet spot detector.

Three variants — pick whichever combination most boosts your dim
signal:

- ``DeconvolveWaveletDetector`` (key ``decon_wavelet``): Richardson-Lucy
  → wavelet. Sharpens diffraction-limited foci; cheapest of the three.
- ``Bm3dWaveletDetector`` (key ``bm3d_wavelet``): BM3D denoising →
  wavelet. Boosts SNR without sharpening, good when foci are mostly
  buried in noise rather than blurred.
- ``DeconvolveBm3dWaveletDetector`` (key ``decon_bm3d_wavelet``):
  BM3D → Richardson-Lucy → wavelet. The full stack. Most expensive
  (~10 s / FOV at 2K × 2K) but typically the best on very dim signal.

All three accept the same ``DetectorOpts`` as the bare wavelet detector,
so ``wavelet_threshold_mad`` / ``wavelet_scales`` knobs transfer 1-for-1.
After preprocessing the image is cleaner, so you can usually **lower**
``wavelet_threshold_mad`` (e.g. 2.5 instead of 4.0) and gain recall on
dim foci without picking up more false positives.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .._types import DetectorOpts, Focus
from .. import preprocess as pp
from .wavelet import WaveletAtrousDetector


class DeconvolveWaveletDetector:
    name = "decon_wavelet"

    def __init__(
        self,
        iterations: int = 30,
        psf: Optional[np.ndarray] = None,
    ) -> None:
        self._iter = int(iterations)
        self._psf = psf
        self._inner = WaveletAtrousDetector()

    def fit(self, run_stack):
        return None

    def detect(
        self,
        image: np.ndarray,
        labeled_mask: Optional[np.ndarray] = None,
        opts: Optional[DetectorOpts] = None,
    ) -> list[Focus]:
        decon = pp.richardson_lucy(image, psf=self._psf, iterations=self._iter)
        return self._inner.detect(decon, labeled_mask=labeled_mask, opts=opts)


class Bm3dWaveletDetector:
    name = "bm3d_wavelet"

    def __init__(self, bm3d_sigma: Optional[float] = None) -> None:
        self._sigma = bm3d_sigma
        self._inner = WaveletAtrousDetector()

    def fit(self, run_stack):
        return None

    def detect(
        self,
        image: np.ndarray,
        labeled_mask: Optional[np.ndarray] = None,
        opts: Optional[DetectorOpts] = None,
    ) -> list[Focus]:
        denoised = pp.bm3d_denoise(image, sigma=self._sigma)
        return self._inner.detect(denoised, labeled_mask=labeled_mask, opts=opts)


class DeconvolveBm3dWaveletDetector:
    name = "decon_bm3d_wavelet"

    def __init__(
        self,
        iterations: int = 30,
        psf: Optional[np.ndarray] = None,
        bm3d_sigma: Optional[float] = None,
    ) -> None:
        self._iter = int(iterations)
        self._psf = psf
        self._sigma = bm3d_sigma
        self._inner = WaveletAtrousDetector()

    def fit(self, run_stack):
        return None

    def detect(
        self,
        image: np.ndarray,
        labeled_mask: Optional[np.ndarray] = None,
        opts: Optional[DetectorOpts] = None,
    ) -> list[Focus]:
        # Denoise first so RL doesn't amplify residual high-freq noise.
        denoised = pp.bm3d_denoise(image, sigma=self._sigma)
        decon = pp.richardson_lucy(denoised, psf=self._psf, iterations=self._iter)
        return self._inner.detect(decon, labeled_mask=labeled_mask, opts=opts)
