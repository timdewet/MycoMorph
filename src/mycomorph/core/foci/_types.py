"""Shared dataclasses for the foci-detection benchmark.

Kept deliberately minimal — the detector contract is a uniform
``detect(image, mask) -> list[Focus]`` returning sub-pixel positions plus
the bits we need for downstream metrics (intensity, sigma, SNR, cell
label). All other metadata (well, FOV, channel, detector name) is
attached by the caller when results are flattened into a long-form
DataFrame.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class Focus:
    """A single foci detection on one image."""
    y: float                # sub-pixel row coordinate
    x: float                # sub-pixel column coordinate
    sigma: float            # Gaussian sigma in pixels (size proxy)
    intensity: float        # peak amplitude (raw image scale)
    snr: float = 0.0        # peak / local-background-std
    cell_label: int = 0     # 0 = outside any cell mask
    focus_id: int = -1      # stable per (well, fov, channel, detector); assigned by the writer


@dataclass
class DetectorOpts:
    """Common parameters detectors consume; individual detectors may ignore
    any field that doesn't apply to them.

    Sigma ranges are in **pixels**. For DnaN-mCherry / ImuB-GFP on a
    typical widefield Mtb image (~0.07 µm/px after binning, diffraction-
    limited focus ≈ 250 nm), expect sigma_psf ≈ 1.3–1.8 px.
    """
    min_sigma: float = 1.0
    max_sigma: float = 3.5
    num_sigma: int = 6          # LoG only
    threshold: float = 0.02     # relative intensity threshold (0..1 of normalised image)
    overlap: float = 0.5        # skimage blob overlap-merge fraction
    snr_min: float = 3.0        # discard foci whose peak/local-bg-std falls below this
    refine: bool = True         # 2D Gaussian sub-pixel refinement
    refine_window: int = 4      # half-window around peak for Gaussian fit
    # Top-hat preprocessing
    tophat_radius_px: int = 5
    # Trackpy
    trackpy_diameter_px: int = 7        # must be odd
    trackpy_minmass: Optional[float] = None
    # Spotiflow
    spotiflow_model: str = "general"    # pretrained model name
    spotiflow_min_distance: int = 2
    # Wavelet à trous (Olivo-Marin 2002) — best classical option for dim foci
    wavelet_scales: tuple[int, ...] = (1, 2)    # foci-size scales to keep
    wavelet_threshold_mad: float = 3.0          # per-plane MAD threshold


@dataclass
class FovBundle:
    """One field-of-view extracted from a MycoMorph segmented hyperstack.

    ``image_channels`` has the mask stripped out (it's exposed
    separately as ``labeled_mask``). Channel-name order matches
    ``image_channels`` first axis.
    """
    well: str
    fov_index: int
    image_channels: np.ndarray          # (C, Y, X), mask removed
    labeled_mask: np.ndarray            # (Y, X) int32
    channel_names: list[str]
    condition_fields: dict[str, str] = field(default_factory=dict)
    source_tiff: Optional[Path] = None
    pixels_per_um: Optional[float] = None
    fluorescence_channel_indices: list[int] = field(default_factory=list)

    @property
    def n_cells(self) -> int:
        return int(self.labeled_mask.max())
