"""Foci-detection library: detectors, preprocessing, features, normalisers.

The detectors share a uniform ``detect(image, labeled_mask, opts) -> list[Focus]``
contract and are exposed through :data:`mycomorph.core.foci.detectors.REGISTRY`.
Per-FOV normalisers (top-hat, BaSiC, Richardson-Lucy, BM3D, …) share an analogous
:data:`mycomorph.core.foci.normalise.NORMALISER_REGISTRY`.

All heavy / optional dependencies (``bm3d``, ``trackpy``, ``spotiflow``, ``basicpy``)
are lazy-imported inside detector / normaliser bodies, so ``import mycomorph.core.foci``
succeeds without the ``[foci]`` optional extra installed.
"""

from ._types import DetectorOpts, Focus, FovBundle
from . import detectors, features, normalise, preprocess

__all__ = [
    "DetectorOpts",
    "Focus",
    "FovBundle",
    "detectors",
    "features",
    "normalise",
    "preprocess",
]
