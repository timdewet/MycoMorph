"""Per-cell features read from the stored crops — focus + fragment cues — in a
single batched pass.

Both the focus (sharpness) and fragment (crossing) features need the brightfield
(and mask) crop. Computing them together, with batched ``_raw_batch`` reads,
keeps the (network-bound) HDF5 hit to a few large reads instead of two passes or
thousands of scattered single-row reads.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

from .crops_h5 import HDF5CropsReader


def compute_crop_features(
    h5_path: Path,
    cell_uids: Optional[Sequence[str]] = None,
    focus: bool = True,
    fragment: bool = True,
    batch: int = 512,
) -> "object":
    """Return a DataFrame (cell_uid + requested crop features)."""
    import numpy as np
    import pandas as pd

    from .focus_features import _sharpness
    from .fragment_features import _features_for

    if not (focus or fragment):
        return pd.DataFrame({"cell_uid": list(cell_uids or [])})

    reader = HDF5CropsReader(h5_path)
    try:
        all_uids = [str(u) for u in reader.cell_uids]
        uid_to_idx = {u: i for i, u in enumerate(all_uids)}
        wanted = [u for u in (cell_uids if cell_uids is not None else all_uids)
                  if u in uid_to_idx]
        idx = [uid_to_idx[u] for u in wanted]
        phase_i, mask_i = reader.phase_channel_idx, reader.mask_channel_idx

        rows = []
        for lo in range(0, len(idx), batch):
            arr = reader._raw_batch(idx[lo:lo + batch])      # (B, C, H, W), input order
            chunk_uids = wanted[lo:lo + batch]
            for u, row in zip(chunk_uids, arr):
                rec = {"cell_uid": u}
                bright = np.asarray(row[phase_i], dtype=np.float32)
                if focus:
                    lv, ten, nv = _sharpness(bright)
                    rec.update(focus_lap_var=lv, focus_tenengrad=ten, focus_norm_var=nv)
                if fragment:
                    mask = np.asarray(row[mask_i]) > 0.5
                    eps, branch, cont = _features_for(bright, mask)
                    rec.update(frag_endpoints=eps, frag_branchpoints=branch,
                               frag_bright_continuation=cont)
                rows.append(rec)
    finally:
        reader.close()

    return pd.DataFrame(rows)
