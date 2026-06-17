"""Illumination-correction preprocessing detectors.

Two flavours, both followed by a plain DoG detector:

- ``BasicDogDetector`` — BaSiC flatfield + darkfield (Peng et al.,
  Nature Communications 2017; BaSiCPy, bioRxiv 2026). Best-in-class
  shading correction but pulls JAX and currently pins
  ``scipy < 1.13`` / ``jaxlib <= 0.4.23`` — incompatible with
  Python 3.13 as of 2026-05. Stays in the registry behind a lazy
  import; will work as soon as basicpy ships a Python 3.13 release.

- ``GaussianNormDogDetector`` — single-image low-pass-ratio shading
  correction: ``raw / gaussian_blur(raw, sigma=large)``. No compiled
  deps, no run-wide fit; works per FOV. Captures the gross illumination
  pattern without estimating darkfield. Functionally similar to the
  classical "rolling-ball over a very large radius" trick. Useful as a
  benchmark-only stand-in while basicpy is wedged on Python 3.13.

The point of both is to isolate whether illumination correction moves
the foci-count needle vs raw DoG.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .._types import DetectorOpts, Focus
from .classical import DogDetector


class BasicDogDetector:
    name = "basic_dog"

    def __init__(self) -> None:
        self._flatfields: dict[int, np.ndarray] = {}     # channel index -> flatfield
        self._darkfields: dict[int, np.ndarray] = {}     # channel index -> darkfield
        self._inner = DogDetector()
        self._fit_done = False

    def fit(self, run_stack: dict[int, list[np.ndarray]]) -> None:
        """Fit BaSiC per channel.

        ``run_stack`` maps channel-index to a list of 2D FOV images (raw
        intensities, same channel, same shape). One BaSiC fit per channel.
        """
        try:
            from basicpy import BaSiC
        except ImportError as exc:  # noqa: BLE001
            raise ImportError(
                "basicpy not installed. `pip install basicpy` to use this detector."
            ) from exc

        for ch, fovs in run_stack.items():
            if not fovs:
                continue
            stack = np.stack([np.asarray(f, dtype=np.float32) for f in fovs], axis=0)
            basic = BaSiC(get_darkfield=True, smoothness_flatfield=1.0)
            basic.fit(stack)
            self._flatfields[int(ch)] = np.asarray(basic.flatfield, dtype=np.float32)
            self._darkfields[int(ch)] = np.asarray(basic.darkfield, dtype=np.float32)
        self._fit_done = True

    def correct(self, image: np.ndarray, channel_index: int) -> np.ndarray:
        """Return BaSiC-corrected image (or raw if no fit exists for the channel)."""
        flat = self._flatfields.get(int(channel_index))
        dark = self._darkfields.get(int(channel_index))
        if flat is None:
            return image.astype(np.float32, copy=False)
        if dark is None:
            dark = np.zeros_like(flat)
        # Guard against zero flatfield pixels.
        flat_safe = np.where(flat > 1e-6, flat, 1.0)
        corrected = (image.astype(np.float32) - dark) / flat_safe
        # Clamp at zero — negative values are physically meaningless after
        # darkfield subtraction.
        return np.clip(corrected, 0.0, None)

    def detect(
        self,
        image: np.ndarray,
        labeled_mask: Optional[np.ndarray] = None,
        opts: Optional[DetectorOpts] = None,
        *,
        channel_index: int = 0,
    ) -> list[Focus]:
        corrected = self.correct(image, channel_index=channel_index)
        return self._inner.detect(corrected, labeled_mask=labeled_mask, opts=opts)

    @property
    def flatfields(self) -> dict[int, np.ndarray]:
        return dict(self._flatfields)

    @property
    def darkfields(self) -> dict[int, np.ndarray]:
        return dict(self._darkfields)


# ─────────────────────────────────────────────────────────────────────────────
# Gaussian low-pass ratio (basicpy-free shading correction)
# ─────────────────────────────────────────────────────────────────────────────

class GaussianNormDogDetector:
    """Single-image shading correction by dividing by a strong Gaussian blur.

    Recipe:
        shading = gaussian_filter(raw, sigma=sigma_shading)
        corrected = raw / max(shading, eps)
        → DoG on the corrected image

    ``sigma_shading`` defaults to 1/8 of the shorter image side (i.e.
    much wider than any PSF) so the blur captures only the slow-varying
    illumination, leaving foci-scale signal in the ratio. Works per
    FOV — no run-wide fit needed. Cheap.
    """

    name = "gaussian_norm_dog"

    def __init__(self, sigma_shading: Optional[float] = None) -> None:
        self._sigma = sigma_shading
        self._inner = DogDetector()

    def fit(self, run_stack):
        return None

    def correct(self, image: np.ndarray) -> np.ndarray:
        from .. import preprocess as pp
        return pp.gaussian_low_pass_normalise(image, sigma_shading=self._sigma)

    def detect(
        self,
        image: np.ndarray,
        labeled_mask: Optional[np.ndarray] = None,
        opts: Optional[DetectorOpts] = None,
    ) -> list[Focus]:
        return self._inner.detect(self.correct(image), labeled_mask=labeled_mask, opts=opts)


# ─────────────────────────────────────────────────────────────────────────────
# Gaussian low-pass ratio + wavelet à trous detector
# ─────────────────────────────────────────────────────────────────────────────

class GaussianNormWaveletDetector:
    """Gaussian low-pass shading correction → à trous wavelet detection.

    Combines the multiplicative-shading correction of ``gaussian_norm_dog``
    with the per-scale MAD noise rejection of the bare ``wavelet`` detector.

    Why this stacks well:

    - Gaussian-norm removes multiplicative illumination shading
      (vignetting, lamp drift) — wavelet's implicit additive background
      subtraction can't do this on its own, so foci in vignetted corners
      get unfairly suppressed in the bare wavelet detector.
    - Wavelet then handles noise structurally via per-scale MAD
      thresholding — far better than DoG, especially on dim signal.

    Best for widefield images that have **both** uneven illumination
    **and** dim foci — the common case. Cheap (~1 Gaussian filter +
    wavelet) so worth always including in a benchmark.
    """

    name = "gaussian_norm_wavelet"

    def __init__(self, sigma_shading: Optional[float] = None) -> None:
        from .wavelet import WaveletAtrousDetector
        self._sigma = sigma_shading
        self._inner = WaveletAtrousDetector()

    def fit(self, run_stack):
        return None

    def correct(self, image: np.ndarray) -> np.ndarray:
        from .. import preprocess as pp
        return pp.gaussian_low_pass_normalise(image, sigma_shading=self._sigma)

    def detect(
        self,
        image: np.ndarray,
        labeled_mask: Optional[np.ndarray] = None,
        opts: Optional[DetectorOpts] = None,
    ) -> list[Focus]:
        return self._inner.detect(self.correct(image), labeled_mask=labeled_mask, opts=opts)
