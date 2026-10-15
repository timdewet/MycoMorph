"""Read pipeline output hyperstacks and their mask channel.

Pipeline stages write ImageJ ZCYX hyperstacks (one Z slice per FOV) with the
segmentation mask as one of the channels. These helpers load them back as
``(N_FOV, C, Y, X)`` and turn the mask channel into labelled cells.
"""

import numpy as np
import tifffile


def load_hyperstack(tiff_path):
    """
    Load a pipeline output TIFF (ImageJ ZCYX hyperstack).

    tifffile may return the data as a flat (Z*C, Y, X) array rather than
    the expected (Z, C, Y, X), depending on version.  We use the ImageJ
    metadata ('slices' and 'channels') to reshape correctly.

    Returns (N_FOV, C, Y, X) and metadata dict.
    """
    with tifffile.TiffFile(str(tiff_path)) as tif:
        data = tif.asarray()
        metadata = dict(tif.imagej_metadata) if tif.imagej_metadata else {}

    if data.ndim == 4:
        # Already (Z, C, Y, X) — nothing to do
        pass
    elif data.ndim == 3:
        n_ch     = int(metadata.get("channels", 1))
        n_slices = int(metadata.get("slices", 1))

        if n_ch > 1 and n_slices > 1:
            # Flat (Z*C, Y, X) → reshape to (Z, C, Y, X)
            expected_frames = n_slices * n_ch
            if data.shape[0] == expected_frames:
                data = data.reshape(n_slices, n_ch, data.shape[1], data.shape[2])
            else:
                # Fallback: can't reconcile, treat as single-FOV with many channels
                data = data[np.newaxis]
        elif n_ch > 1:
            # Single Z, multiple C: (C, Y, X) → (1, C, Y, X)
            data = data[np.newaxis]
        elif n_slices > 1:
            # Multiple Z, single C: (Z, Y, X) → (Z, 1, Y, X)
            data = data[:, np.newaxis]
        else:
            # Single Z, single C: (1, Y, X) → (1, 1, Y, X)
            data = data[np.newaxis]
    elif data.ndim == 2:
        data = data[np.newaxis, np.newaxis]

    return data, metadata


def get_labeled_mask_from_fov(fov_data, mask_channel):
    """Re-label a binary mask channel with connected components."""
    from skimage.measure import label as sk_label
    binary = fov_data[mask_channel] > 0
    labeled = sk_label(binary, connectivity=1)
    return labeled, int(labeled.max())
