"""Bacteroidal background subtraction + 2D Gaussian fit.

Reimplementation of the algorithm benchmarked in Wang et al., Biophysical
Journal 2015 (`Quantitative Analysis of Intracellular Fluorescent Foci
in Live Bacteria <https://pmc.ncbi.nlm.nih.gov/articles/PMC4564679/>`__)
as best-performing for bacterial fluorescent foci.

Algorithm per cell:
1. Compute the median intensity of pixels **inside the cell mask** (the
   "bacteroidal" background).
2. Subtract the cell-local background from the in-cell pixels.
3. Find local maxima in the background-subtracted image (limited to the
   cell ROI) using ``skimage.feature.peak_local_max``.
4. Fit a 2D Gaussian to each peak via :func:`refine_gaussian_peak`.
5. Compute SNR against the within-cell std (rather than annulus) since
   bacterial cells are small and the annulus can leave the cell.

A focus is accepted iff its fitted amplitude is above
``snr_min * cell_std``.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .._types import DetectorOpts, Focus
from ._common import label_at, refine_gaussian_peak


class BacteroidalDetector:
    name = "bacteroidal"

    def fit(self, run_stack):
        return None

    def detect(
        self,
        image: np.ndarray,
        labeled_mask: Optional[np.ndarray] = None,
        opts: Optional[DetectorOpts] = None,
    ) -> list[Focus]:
        from skimage.feature import peak_local_max

        opts = opts or DetectorOpts()
        if labeled_mask is None:
            # Without a mask we have no bacteroidal background; degrade to
            # a single global median.
            labeled_mask = np.ones_like(image, dtype=np.int32)

        out: list[Focus] = []
        n_cells = int(labeled_mask.max())
        for cell_id in range(1, n_cells + 1):
            cell_pixels = labeled_mask == cell_id
            if not cell_pixels.any():
                continue
            cell_values = image[cell_pixels].astype(np.float64)
            cell_median = float(np.median(cell_values))
            cell_std = float(cell_values.std())
            if cell_std < 1e-9:
                continue

            # Local bbox crop, background-subtracted in-cell, zeroed elsewhere.
            ys, xs = np.where(cell_pixels)
            y0, y1 = int(ys.min()), int(ys.max()) + 1
            x0, x1 = int(xs.min()), int(xs.max()) + 1
            mask_crop = cell_pixels[y0:y1, x0:x1]
            sub = image[y0:y1, x0:x1].astype(np.float64) - cell_median
            sub = np.where(mask_crop, sub, 0.0)

            # Detect peaks above threshold (in cell-std units).
            min_amp = opts.snr_min * cell_std
            peaks = peak_local_max(
                sub,
                min_distance=max(1, int(round(opts.min_sigma * 2))),
                threshold_abs=min_amp,
                exclude_border=False,
            )
            for peak in peaks:
                y_peak = float(peak[0] + y0)
                x_peak = float(peak[1] + x0)
                if opts.refine:
                    y_sub, x_sub, sigma_fit, amp = refine_gaussian_peak(
                        image, y_peak, x_peak,
                        half_window=opts.refine_window,
                        init_sigma=opts.min_sigma * 1.2,
                    )
                else:
                    y_sub, x_sub = y_peak, x_peak
                    sigma_fit = opts.min_sigma * 1.2
                    amp = float(sub[int(peak[0]), int(peak[1])])
                snr = (amp - cell_median) / cell_std if cell_std > 0 else 0.0
                if snr < opts.snr_min:
                    continue
                out.append(Focus(
                    y=y_sub,
                    x=x_sub,
                    sigma=sigma_fit,
                    intensity=amp,
                    snr=snr,
                    cell_label=label_at(labeled_mask, y_sub, x_sub),
                ))
        return out
