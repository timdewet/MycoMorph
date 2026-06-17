"""Image preprocessing for foci detection on dim signal.

Two complementary tools that lift effective SNR before any detector
runs. They compose cleanly — typical order is denoise → deconvolve so
that deconvolution doesn't amplify high-frequency noise.

- **Richardson-Lucy deconvolution** (``richardson_lucy``): iterative
  maximum-likelihood deconvolution with a known PSF. Widefield benefits
  dramatically — out-of-focus blur gets pushed back toward its origin,
  diffraction-limited foci become brighter relative to their local
  background. Ships with ``scikit-image``.

- **BM3D denoising** (``bm3d_denoise``): block-matching 3D filtering
  (Dabov et al. 2007). The state-of-the-art classical denoiser, model-
  free, exploits non-local self-similarity in natural images. Typically
  gives a 2–3× effective SNR boost on microscopy with no parameter
  tuning beyond the noise σ (which we auto-estimate by MAD).

Both functions are pure-function preprocessing — feed the result to any
detector (``wavelet``, ``dog``, ``spotiflow``, …).

Default PSF is a 9×9 Gaussian with σ=1.5 px, which approximates a
widefield Mtb PSF at typical binning. For a more accurate PSF, generate
one with ``psfmodels`` (Gibson-Lanni) from your microscope's NA /
wavelength / pixel-size and pass it as ``psf=``.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def gaussian_psf(sigma: float = 1.5, size: int = 9) -> np.ndarray:
    """Return a normalised 2D Gaussian PSF kernel.

    ``sigma=1.5`` px is a reasonable widefield-Mtb default. Replace with
    a bead-measured PSF or a Gibson-Lanni model for production.
    """
    if size % 2 == 0:
        size += 1
    half = size // 2
    yy, xx = np.mgrid[-half:half + 1, -half:half + 1]
    psf = np.exp(-(yy ** 2 + xx ** 2) / (2.0 * sigma * sigma))
    psf /= psf.sum()
    return psf.astype(np.float32)


def estimate_noise_sigma(image: np.ndarray) -> float:
    """Robust noise σ via MAD on a high-frequency residual.

    Computes ``residual = image - gaussian_blur(image, σ=2)`` and returns
    the Gaussian-consistent MAD-scale of the residual. Good enough for
    BM3D's σ argument in the shot-noise-limited regime where your widefield
    images sit.
    """
    from scipy.ndimage import gaussian_filter

    img = image.astype(np.float32)
    smoothed = gaussian_filter(img, sigma=2.0)
    residual = img - smoothed
    med = float(np.median(residual))
    mad = float(np.median(np.abs(residual - med)))
    return 1.4826 * mad


def richardson_lucy(
    image: np.ndarray,
    psf: Optional[np.ndarray] = None,
    iterations: int = 30,
    clip: bool = False,
) -> np.ndarray:
    """Richardson-Lucy deconvolution.

    Wraps ``skimage.restoration.richardson_lucy`` with sensible defaults:
    Gaussian PSF, 30 iterations, no clipping (we restore the original
    intensity scale afterwards so downstream detectors see comparable
    pixel values).

    Parameters
    ----------
    image
        2D float / uint image.
    psf
        2D PSF kernel summing to 1. Falls back to a Gaussian σ=1.5.
    iterations
        RL is iterative. 20–50 is the usual widefield range; more
        iterations sharpen further at the cost of ringing artefacts.
    clip
        See :func:`skimage.restoration.richardson_lucy`. Leave False
        for foci detection — clipping at [0, 1] truncates the dynamic
        range we need.

    Returns
    -------
    np.ndarray
        Deconvolved image as ``float32``, rescaled to roughly the input's
        intensity range.
    """
    from skimage.restoration import richardson_lucy as rl

    if psf is None:
        psf = gaussian_psf(sigma=1.5, size=9)

    img = image.astype(np.float32, copy=True)
    img_min = float(img.min())
    img_max = float(img.max())
    if img_max <= img_min:
        return img

    # RL expects non-negative input in roughly [0, 1].
    normalised = (img - img_min) / (img_max - img_min)
    deconvolved = rl(normalised, psf, num_iter=int(iterations), clip=clip)
    # Restore original intensity scale so SNR / threshold knobs that
    # were tuned on raw images still mean roughly the same thing.
    return (deconvolved * (img_max - img_min) + img_min).astype(np.float32)


def bm3d_denoise(
    image: np.ndarray,
    sigma: Optional[float] = None,
) -> np.ndarray:
    """BM3D denoising.

    Lazy-imports ``bm3d`` (``pip install bm3d``). If ``sigma`` is None,
    auto-estimates via :func:`estimate_noise_sigma`.

    Parameters
    ----------
    image
        2D image.
    sigma
        Noise standard deviation **in the raw intensity scale**. The
        function rescales internally for BM3D's [0, 1] convention.

    Returns
    -------
    np.ndarray
        Denoised image as ``float32`` in the original intensity scale.
    """
    try:
        import bm3d
    except ImportError as exc:  # noqa: BLE001
        raise ImportError(
            "bm3d not installed. `pip install bm3d` to use BM3D denoising."
        ) from exc

    img = image.astype(np.float32, copy=True)
    if sigma is None:
        sigma = estimate_noise_sigma(img)

    img_min = float(img.min())
    img_max = float(img.max())
    if img_max <= img_min:
        return img

    normalised = (img - img_min) / (img_max - img_min)
    sigma_norm = float(sigma) / (img_max - img_min)
    denoised = bm3d.bm3d(normalised, sigma_psd=sigma_norm)
    return (denoised * (img_max - img_min) + img_min).astype(np.float32)


def gaussian_low_pass_normalise(
    image: np.ndarray,
    sigma_shading: Optional[float] = None,
) -> np.ndarray:
    """Multiplicative shading correction via division by a Gaussian-blurred copy.

    ``shading = gaussian_filter(raw, σ_shading)`` captures the slow-
    varying illumination. ``corrected = raw / shading × mean(shading)``
    keeps roughly the input dynamic range and divides out the shading.

    Unlike additive background subtraction (what wavelet à trous does
    implicitly), this is the right correction for multiplicative
    illumination effects like lamp vignetting, sample tilt, or uneven
    laser profiles in widefield. ``σ_shading`` defaults to ~⅛ of the
    shorter image dimension — large enough to capture only the slow-
    varying component, leaving foci intact.

    Cheap (one Gaussian filter pass), per-FOV (no run-wide fit needed),
    compositions cleanly with downstream detectors.
    """
    from scipy.ndimage import gaussian_filter

    img = image.astype(np.float32, copy=False)
    sigma = (
        float(sigma_shading)
        if sigma_shading is not None
        else max(8.0, min(img.shape) / 8.0)
    )
    shading = gaussian_filter(img, sigma=sigma)
    mean_shading = float(shading.mean()) if shading.size else 1.0
    eps = max(1e-6, mean_shading * 1e-3)
    return (img * (mean_shading / np.maximum(shading, eps))).astype(np.float32)


def gaussian_psf_3d(
    sigma_xy: float = 1.5,
    sigma_z: float = 2.5,
    size_xy: int = 11,
    size_z: int = 7,
) -> np.ndarray:
    """Normalised 3D Gaussian PSF in ``(Z, Y, X)`` order.

    Defaults approximate widefield Mtb at typical binning
    (~70 nm/px XY, ~250 nm Z step). For better accuracy, generate a
    Gibson-Lanni 3D PSF with ``psfmodels.scalar.gibson_lanni()`` using
    your microscope's NA, emission wavelength, and pixel sizes — then
    pass it as ``psf=`` to :func:`richardson_lucy_3d`.
    """
    if size_xy % 2 == 0:
        size_xy += 1
    if size_z % 2 == 0:
        size_z += 1
    half_xy = size_xy // 2
    half_z = size_z // 2
    zz, yy, xx = np.mgrid[
        -half_z: half_z + 1,
        -half_xy: half_xy + 1,
        -half_xy: half_xy + 1,
    ]
    psf = np.exp(
        -(yy ** 2 + xx ** 2) / (2.0 * sigma_xy * sigma_xy)
        - (zz ** 2) / (2.0 * sigma_z * sigma_z)
    )
    psf /= psf.sum()
    return psf.astype(np.float32)


def richardson_lucy_3d(
    stack_zyx: np.ndarray,
    psf: Optional[np.ndarray] = None,
    iterations: int = 30,
    sigma_xy: float = 1.5,
    sigma_z: float = 2.5,
) -> np.ndarray:
    """3D Richardson-Lucy deconvolution on a ``(Z, Y, X)`` stack.

    Qualitatively different from 2D RL on widefield: the 3D PSF lets
    the algorithm explicitly model and remove out-of-focus light
    contamination — the dominant source of background. Typical SNR
    gain on widefield bacterial fluorescence: 5–10×.

    Parameters
    ----------
    stack_zyx
        3D array shaped ``(Z, Y, X)``.
    psf
        3D PSF kernel shaped ``(Pz, Py, Px)``, summing to 1. If None,
        falls back to a Gaussian via :func:`gaussian_psf_3d` with the
        provided ``sigma_xy`` / ``sigma_z``.
    iterations
        20–40 is the usual widefield range. More sharpens further at
        the cost of ringing.
    """
    from skimage.restoration import richardson_lucy as rl

    if psf is None:
        psf = gaussian_psf_3d(sigma_xy=sigma_xy, sigma_z=sigma_z)

    img = stack_zyx.astype(np.float32, copy=True)
    img_min = float(img.min())
    img_max = float(img.max())
    if img_max <= img_min:
        return img

    normalised = (img - img_min) / (img_max - img_min)
    deconvolved = rl(normalised, psf, num_iter=int(iterations), clip=False)
    return (deconvolved * (img_max - img_min) + img_min).astype(np.float32)


def pick_focused_plane(stack_zyx: np.ndarray) -> int:
    """Return the Z index of the sharpest plane (Laplacian-variance metric)."""
    from scipy.ndimage import laplace

    scores = [
        float(np.var(laplace(plane.astype(np.float32))))
        for plane in stack_zyx
    ]
    return int(np.argmax(scores))


def preprocess(
    image: np.ndarray,
    *,
    denoise: bool = False,
    deconvolve: bool = False,
    psf: Optional[np.ndarray] = None,
    decon_iterations: int = 30,
    bm3d_sigma: Optional[float] = None,
) -> np.ndarray:
    """Apply preprocessing steps in canonical order: denoise first
    (so deconvolution doesn't amplify residual noise), then deconvolve.
    """
    img = image.astype(np.float32, copy=True)
    if denoise:
        img = bm3d_denoise(img, sigma=bm3d_sigma)
    if deconvolve:
        img = richardson_lucy(img, psf=psf, iterations=decon_iterations)
    return img
