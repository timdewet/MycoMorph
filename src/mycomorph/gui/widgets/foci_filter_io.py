"""IO helpers for persisting foci filter thresholds.

The FociDetectionPanel hosts the histograms + threshold sliders inline.
When the user clicks "Save filter" the panel calls
:func:`save_foci_filter`, which writes:

- ``05_foci_filters/<timestamp>__<run_id>.json`` — the threshold dict
  the user picked, with input / pass counts for auditability.
- ``04d_foci_filtered/<well>.parquet`` — one filtered subset per source
  well, where each row passes every threshold. Only written when the
  run has saved foci-detection parquets at ``04c_foci_detection/`` to
  apply the filter to; otherwise we just write the JSON.

Filter semantics: every threshold is a **minimum** — a focus passes when
``feature >= threshold``. NaN values fail any threshold by definition.
AND-combined across features.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


# Features shown in the inline histograms (must match the panel + the
# parquet columns produced by ``mycomorph.core.foci.features.features_dataframe``).
FILTER_FEATURES: list[tuple[str, str, bool]] = [
    ("gaussian_r2",      "gaussian_r2",      True),
    ("hessian_symmetry", "hessian_symmetry", True),
    ("patch_snr",        "patch_snr",        True),
    ("prominence_p90",   "prominence_p90",   True),
    ("sigma",            "sigma",            True),
    ("intensity",        "intensity",        True),
]


@dataclass
class FilterSaveResult:
    json_path: Path
    filtered_parquets: list[Path]
    n_input: int
    n_pass: int


def compute_pass_mask(df: pd.DataFrame, thresholds: dict[str, float]) -> np.ndarray:
    """AND-combine each ``feature >= threshold``; NaN fails."""
    mask = np.ones(len(df), dtype=bool)
    for col, thr in thresholds.items():
        if col not in df.columns:
            continue
        vals = df[col].to_numpy(dtype=np.float64, copy=False)
        mask &= np.isfinite(vals) & (vals >= float(thr))
    return mask


def save_foci_filter(
    output_dir: Path,
    run_id: str,
    thresholds: dict[str, float],
) -> FilterSaveResult:
    """Persist the threshold dict + write filtered parquets.

    Parameters
    ----------
    output_dir : Path
        The run's output directory. Reads from ``04c_foci_detection/`` if
        present and writes both ``04d_foci_filtered/<well>.parquet`` and
        ``05_foci_filters/<timestamp>__<run_id>.json`` under it.
    run_id : str
        Identifier baked into the JSON filename.
    thresholds : dict[str, float]
        Per-feature minimum thresholds.
    """
    output_dir = Path(output_dir)
    foci_dir = output_dir / "04c_foci_detection"
    filtered_dir = output_dir / "04d_foci_filtered"
    filters_dir = output_dir / "05_foci_filters"
    filters_dir.mkdir(parents=True, exist_ok=True)

    n_input = 0
    n_pass = 0
    filtered_parquets: list[Path] = []
    if foci_dir.exists():
        filtered_dir.mkdir(parents=True, exist_ok=True)
        for parquet in sorted(foci_dir.glob("*.parquet")):
            well_df = pd.read_parquet(parquet)
            n_input += len(well_df)
            mask = compute_pass_mask(well_df, thresholds)
            n_pass += int(mask.sum())
            out_path = filtered_dir / parquet.name
            well_df.loc[mask].to_parquet(out_path, index=False)
            filtered_parquets.append(out_path)

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    iso = datetime.now().isoformat(timespec="seconds")
    payload = {
        "run_id": run_id,
        "saved_at": iso,
        "thresholds": {
            col: {"min": float(val)} for col, val in thresholds.items()
        },
        "n_input": int(n_input),
        "n_pass": int(n_pass),
        "pct_pass": (n_pass / n_input) if n_input else 0.0,
    }
    json_path = filters_dir / f"{ts}__{run_id or 'run'}.json"
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)

    return FilterSaveResult(
        json_path=json_path,
        filtered_parquets=filtered_parquets,
        n_input=n_input,
        n_pass=n_pass,
    )
