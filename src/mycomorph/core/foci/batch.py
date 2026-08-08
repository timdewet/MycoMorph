"""Process-pool helpers for the batch Foci Detection stage.

Detection + per-focus features for one (FOV, channel, detector) unit is
pure CPU work on plain numpy arrays and embarrassingly parallel across
units — a plate run is hundreds of independent units, and Python holds
the GIL through most of the per-unit work (scipy fits, feature loops),
so threads don't help but processes do.

This module is Qt-free and holds the module-level entry point so the
``spawn`` start method can import it in worker processes. Workers
import ~0.8 s of numpy/scipy/pandas once each and are reused for every
unit thereafter.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def detect_unit(
    image: np.ndarray,
    labeled_mask: np.ndarray,
    det_key: str,
    det_opts: Any,
    *,
    well: str = "",
    fov_index: int = 0,
    channel_index: int = 0,
    channel_name: str = "",
    normalisation: str = "",
):
    """Detect + featurise one (FOV, channel, detector) unit.

    Returns the finished long-form DataFrame for this unit — the same
    columns the Foci Detection stage writes to parquet — or an empty
    DataFrame when nothing was found.
    """
    from mycomorph.core.foci.decompose import split_merged_foci
    from mycomorph.core.foci.detectors import REGISTRY
    from mycomorph.core.foci.features import features_dataframe

    detector = REGISTRY[det_key]()
    foci = detector.detect(image, labeled_mask, det_opts)
    if getattr(det_opts, "split_merged", False):
        foci = split_merged_foci(
            image, foci, det_opts, labeled_mask=labeled_mask,
        )
    # Stable focus_id per (well, fov, channel, detector).
    for fid, f in enumerate(foci):
        f.focus_id = fid
    df = features_dataframe(
        image, labeled_mask, foci,
        detector=det_key,
        well=well,
        fov_index=fov_index,
        channel=channel_name,
    )
    if not df.empty:
        df["fov"] = int(fov_index)
        df["channel_index"] = int(channel_index)
        df["channel_name"] = channel_name
        df["normalisation"] = normalisation
        df["cell_id"] = df["cell_label"].astype(int)
        df["focus_id"] = [int(f.focus_id) for f in foci]
    return df


def resolve_worker_count(n_tasks: int) -> int:
    """Worker-process count for the stage.

    Honours ``MYCOMORPH_FOCI_WORKERS`` (a value of 1 forces the inline
    single-process path — useful for debugging); otherwise CPU count
    minus two so the GUI and OS stay responsive, capped at the number
    of tasks.
    """
    import os

    raw = os.environ.get("MYCOMORPH_FOCI_WORKERS", "").strip()
    try:
        n = int(raw) if raw else 0
    except ValueError:
        n = 0
    if n <= 0:
        n = max(1, (os.cpu_count() or 2) - 2)
    return max(1, min(n, max(n_tasks, 1)))
