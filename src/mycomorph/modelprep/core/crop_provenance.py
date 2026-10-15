"""How were these crops made? Checks that catch train/inference mismatches.

MycoMorph's Features stage can replace everything outside each cell's mask
with the median (``mask_background``, on by default in its GUI). The crops its
classifier sees at inference are *not* masked, so a model trained on masked
crops is trained on different images from the ones it will judge.

The setting is read from the HDF5 itself when MycoMorph records it (an
``mask_background`` attribute, or a JSON ``crop_options`` attribute), else from
the pipeline's ``run_manifest.json`` (beside the HDF5 or one folder up, i.e.
two levels above the file), whose ``Features`` stage lists its options. The
manifest also shows whether Classify ran before Features, which means the
HDF5 probably holds only cells the previous model kept.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

_OPTION_ATTRS = ("crop_options", "crop_opts", "extract_options")


def _as_bool(v: Any) -> Optional[bool]:
    import numpy as np

    if isinstance(v, (bool, int, np.bool_, np.integer)):
        return bool(v)
    if isinstance(v, (bytes, str)):
        s = v.decode() if isinstance(v, bytes) else v
        if s.strip().lower() in ("true", "1", "yes"):
            return True
        if s.strip().lower() in ("false", "0", "no"):
            return False
    return None


def _from_h5(h5_path: Path) -> Optional[dict]:
    import h5py

    try:
        with h5py.File(str(h5_path), "r") as f:
            attrs = dict(f.attrs)
    except OSError:
        return None
    if "mask_background" in attrs:
        return {"mask_background": _as_bool(attrs["mask_background"]), "source": "HDF5 attribute"}
    for key in _OPTION_ATTRS:
        if key in attrs:
            raw = attrs[key]
            try:
                opts = json.loads(raw.decode() if isinstance(raw, bytes) else str(raw))
            except ValueError:
                continue
            if isinstance(opts, dict) and "mask_background" in opts:
                return {"mask_background": _as_bool(opts["mask_background"]),
                        "source": f"HDF5 attribute {key}"}
    return None


def find_run_manifest(h5_path: Path) -> Optional[Path]:
    """``run_manifest.json`` beside the HDF5 or in the folder above it."""
    h5_path = Path(h5_path)
    for d in (h5_path.parent, h5_path.parent.parent):
        p = d / "run_manifest.json"
        if p.is_file():
            return p
    return None


def _from_manifest(manifest: Path) -> dict:
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    stages = data.get("stages") if isinstance(data, dict) else None
    if not isinstance(stages, list):
        return {}
    names = [str(s.get("name", "")).lower() for s in stages if isinstance(s, dict)]
    out: dict = {"source": manifest.name}
    if "features" in names:
        feat = stages[names.index("features")]
        opts = feat.get("opts") or {}
        if "mask_background" in opts:
            out["mask_background"] = _as_bool(opts["mask_background"])
        if "classify" in names[: names.index("features")]:
            cls = stages[names.index("classify")]
            out["classify_before_features"] = not cls.get("skipped", False)
    return out


def crop_options(h5_path: Path) -> dict:
    """What's known about how the crops were extracted: ``mask_background``
    (True/False/None), ``classify_before_features`` and where each came from.
    Read-only; never raises for a missing or malformed manifest."""
    info: dict = {}
    manifest = find_run_manifest(h5_path)
    if manifest is not None:
        info.update(_from_manifest(manifest))
    h5 = _from_h5(h5_path)
    if h5 is not None:  # the HDF5's own record wins over the manifest
        info.update(h5)
    return info


def crop_warnings(h5_path: Path, info: Optional[dict] = None) -> list[str]:
    """One-line warnings for the Source page and ``modelprep-cli info``."""
    info = crop_options(h5_path) if info is None else info
    out = []
    if info.get("mask_background") is True:
        out.append(
            "These crops were extracted with background masking ON (mask_background, "
            f"per {info.get('source', 'the run')}): everything outside each cell is "
            "blanked, but MycoMorph classifies unmasked crops, so a model trained on "
            "them won't see what it sees at inference. Re-extract with masking off.")
    if info.get("classify_before_features"):
        out.append(
            "Classify ran before Features in this pipeline run, so the HDF5 probably "
            "holds only cells the previous model kept: cells it wrongly rejected never "
            "reach the labeller, and hold-out metrics are conditional on that filter.")
    return out
