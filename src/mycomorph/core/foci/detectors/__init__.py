"""Foci detector implementations sharing a uniform ``detect`` contract.

All detectors expose:
    name: str
    fit(run_stack)       optional pre-processing fit across the whole run
    detect(image, mask, opts) -> list[Focus]

The ``REGISTRY`` maps short string keys to factory callables so the
notebook can iterate over a selected subset without importing every
optional dependency upfront. Heavy / optional deps (BaSiC, trackpy)
are imported lazily inside the relevant detector's body.
"""

from __future__ import annotations

from typing import Callable

from .classical import DogDetector, LogDetector, TophatDogDetector, TrackpyDetector
from .basic import (
    BasicDogDetector,
    GaussianNormDogDetector,
    GaussianNormWaveletDetector,
)
from .bacteroidal import BacteroidalDetector
from .wavelet import WaveletAtrousDetector
from .enhanced import (
    Bm3dWaveletDetector,
    DeconvolveBm3dWaveletDetector,
    DeconvolveWaveletDetector,
)


REGISTRY: dict[str, Callable[[], object]] = {
    "dog": DogDetector,
    "log": LogDetector,
    "trackpy": TrackpyDetector,
    "tophat_dog": TophatDogDetector,
    "gaussian_norm_dog": GaussianNormDogDetector,
    "basic_dog": BasicDogDetector,
    "wavelet": WaveletAtrousDetector,
    "gaussian_norm_wavelet": GaussianNormWaveletDetector,
    "decon_wavelet": DeconvolveWaveletDetector,
    "bm3d_wavelet": Bm3dWaveletDetector,
    "decon_bm3d_wavelet": DeconvolveBm3dWaveletDetector,
    "bacteroidal": BacteroidalDetector,
}


CLASSICAL_BASELINE_KEYS = ["dog", "log", "trackpy"]
NORMALISATION_KEYS = ["tophat_dog", "gaussian_norm_dog", "basic_dog"]
DIM_SIGNAL_KEYS = ["wavelet"]
ENHANCED_DIM_SIGNAL_KEYS = [
    "gaussian_norm_wavelet", "decon_wavelet", "bm3d_wavelet", "decon_bm3d_wavelet",
]
BACTERIAL_SPECIFIC_KEYS = ["bacteroidal"]

ALL_KEYS = (
    CLASSICAL_BASELINE_KEYS
    + NORMALISATION_KEYS
    + DIM_SIGNAL_KEYS
    + ENHANCED_DIM_SIGNAL_KEYS
    + BACTERIAL_SPECIFIC_KEYS
)

__all__ = [
    "DogDetector",
    "LogDetector",
    "TrackpyDetector",
    "TophatDogDetector",
    "GaussianNormDogDetector",
    "GaussianNormWaveletDetector",
    "BasicDogDetector",
    "WaveletAtrousDetector",
    "DeconvolveWaveletDetector",
    "Bm3dWaveletDetector",
    "DeconvolveBm3dWaveletDetector",
    "BacteroidalDetector",
    "REGISTRY",
    "ALL_KEYS",
    "CLASSICAL_BASELINE_KEYS",
    "NORMALISATION_KEYS",
    "DIM_SIGNAL_KEYS",
    "ENHANCED_DIM_SIGNAL_KEYS",
    "BACTERIAL_SPECIFIC_KEYS",
]
