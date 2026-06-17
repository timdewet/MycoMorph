"""Fluorescent-channel normalisers with a uniform ``apply(image)`` contract.

Mirrors the structure of :mod:`mycomorph.core.foci.detectors`: a registry of
short string keys → factory callables, so the pipeline stage and the GUI panel
can iterate over a selected method without knowing about every optional
dependency upfront.

Heavy / optional dependencies (``basicpy``, ``bm3d``) are lazy-imported inside
the relevant class's body.

All normalisers preserve the input dtype's intensity scale: input goes in as
``uint16`` or ``float``, output comes back as ``float32`` in roughly the same
intensity range so downstream detectors keep their thresholds meaningful.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Protocol

import numpy as np

from . import preprocess as pp


@dataclass
class NormaliserOpts:
    """Knobs every normaliser may consume. Individual implementations are free to
    ignore any field that doesn't apply to them."""
    tophat_radius_px: int = 5
    gaussian_lp_sigma: Optional[float] = None        # None → auto (≈ min_dim / 8)
    rl_iterations: int = 30
    rl_psf_sigma: float = 1.5
    bm3d_sigma: Optional[float] = None               # None → MAD-estimate
    basic_max_workers: int = 0                       # passed through to basicpy if supported


class Normaliser(Protocol):
    name: str

    def fit(self, run_stack: Optional[np.ndarray]) -> None:
        """Optional run-level fit (BaSiC). ``run_stack`` is a stack of FOVs for one
        channel (N, Y, X). No-op by default."""
        ...

    def apply(self, image: np.ndarray) -> np.ndarray:
        """Per-FOV transform, image-in → image-out."""
        ...


class NoOp:
    name = "none"

    def __init__(self, opts: NormaliserOpts | None = None) -> None:  # noqa: ARG002
        pass

    def fit(self, run_stack: Optional[np.ndarray]) -> None:  # noqa: ARG002
        return None

    def apply(self, image: np.ndarray) -> np.ndarray:
        return image.astype(np.float32, copy=False)


class TopHat:
    """White top-hat: subtracts a slow-varying background estimate while leaving
    foci-sized peaks intact."""

    name = "tophat"

    def __init__(self, opts: NormaliserOpts | None = None) -> None:
        self.opts = opts or NormaliserOpts()

    def fit(self, run_stack: Optional[np.ndarray]) -> None:  # noqa: ARG002
        return None

    def apply(self, image: np.ndarray) -> np.ndarray:
        from skimage.morphology import white_tophat, disk
        img = image.astype(np.float32, copy=False)
        return white_tophat(img, footprint=disk(int(self.opts.tophat_radius_px))).astype(np.float32)


class GaussianLowPass:
    """Multiplicative shading correction by dividing out a Gaussian-blurred copy.

    Wraps :func:`mycomorph.core.foci.preprocess.gaussian_low_pass_normalise`.
    """

    name = "gaussian_lp"

    def __init__(self, opts: NormaliserOpts | None = None) -> None:
        self.opts = opts or NormaliserOpts()

    def fit(self, run_stack: Optional[np.ndarray]) -> None:  # noqa: ARG002
        return None

    def apply(self, image: np.ndarray) -> np.ndarray:
        return pp.gaussian_low_pass_normalise(
            image, sigma_shading=self.opts.gaussian_lp_sigma
        )


class RichardsonLucy:
    """Iterative deconvolution. Pre-fits nothing; PSF is a fixed Gaussian unless
    swapped out."""

    name = "richardson_lucy"

    def __init__(self, opts: NormaliserOpts | None = None) -> None:
        self.opts = opts or NormaliserOpts()

    def fit(self, run_stack: Optional[np.ndarray]) -> None:  # noqa: ARG002
        return None

    def apply(self, image: np.ndarray) -> np.ndarray:
        psf = pp.gaussian_psf(sigma=float(self.opts.rl_psf_sigma))
        return pp.richardson_lucy(image, psf=psf, iterations=int(self.opts.rl_iterations))


class Bm3dDenoise:
    """BM3D denoising. Requires the optional ``bm3d`` package."""

    name = "bm3d"

    def __init__(self, opts: NormaliserOpts | None = None) -> None:
        self.opts = opts or NormaliserOpts()

    def fit(self, run_stack: Optional[np.ndarray]) -> None:  # noqa: ARG002
        return None

    def apply(self, image: np.ndarray) -> np.ndarray:
        return pp.bm3d_denoise(image, sigma=self.opts.bm3d_sigma)


class DeconBm3d:
    """Denoise → deconvolve composition. Recommended order from the benchmark
    notebook (BM3D first so RL doesn't amplify noise)."""

    name = "decon_bm3d"

    def __init__(self, opts: NormaliserOpts | None = None) -> None:
        self.opts = opts or NormaliserOpts()
        self._bm3d = Bm3dDenoise(self.opts)
        self._rl = RichardsonLucy(self.opts)

    def fit(self, run_stack: Optional[np.ndarray]) -> None:
        self._bm3d.fit(run_stack)
        self._rl.fit(run_stack)

    def apply(self, image: np.ndarray) -> np.ndarray:
        return self._rl.apply(self._bm3d.apply(image))


class BasicFlatField:
    """BaSiC shading correction (Peng et al. 2017).

    Needs a run-level ``fit`` across all FOVs of one channel to learn the
    flat-field / dark-field profiles. Lazy-imports ``basicpy``.
    """

    name = "basic"

    def __init__(self, opts: NormaliserOpts | None = None) -> None:
        self.opts = opts or NormaliserOpts()
        self._basic = None

    def fit(self, run_stack: Optional[np.ndarray]) -> None:
        if run_stack is None or run_stack.size == 0:
            return None
        from basicpy import BaSiC

        kwargs = {}
        if self.opts.basic_max_workers:
            kwargs["max_workers"] = int(self.opts.basic_max_workers)
        basic = BaSiC(**kwargs)
        basic.fit(run_stack.astype(np.float32, copy=False))
        self._basic = basic
        return None

    def apply(self, image: np.ndarray) -> np.ndarray:
        if self._basic is None:
            # Fallback when fit() was never called (e.g. single-FOV preview).
            return image.astype(np.float32, copy=False)
        corrected = self._basic.transform(image.astype(np.float32, copy=False)[None, ...])
        return np.asarray(corrected[0], dtype=np.float32)


NORMALISER_REGISTRY: dict[str, Callable[[NormaliserOpts | None], Normaliser]] = {
    NoOp.name: NoOp,
    TopHat.name: TopHat,
    GaussianLowPass.name: GaussianLowPass,
    RichardsonLucy.name: RichardsonLucy,
    Bm3dDenoise.name: Bm3dDenoise,
    DeconBm3d.name: DeconBm3d,
    BasicFlatField.name: BasicFlatField,
}


def build(method: str, opts: NormaliserOpts | None = None) -> Normaliser:
    """Factory: look up a normaliser by key, raise on unknown keys."""
    try:
        factory = NORMALISER_REGISTRY[method]
    except KeyError as exc:  # noqa: BLE001
        raise ValueError(
            f"Unknown normalisation method {method!r}. "
            f"Available: {sorted(NORMALISER_REGISTRY)}"
        ) from exc
    return factory(opts)


def requires_run_level_fit(method: str) -> bool:
    """True for methods (currently just BaSiC) that need all-FOV access to fit
    before per-FOV ``apply`` calls produce meaningful output."""
    return method == BasicFlatField.name


__all__ = [
    "Normaliser",
    "NormaliserOpts",
    "NoOp",
    "TopHat",
    "GaussianLowPass",
    "RichardsonLucy",
    "Bm3dDenoise",
    "DeconBm3d",
    "BasicFlatField",
    "NORMALISER_REGISTRY",
    "build",
    "requires_run_level_fit",
]
