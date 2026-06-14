"""Blinded intra-rater label-consistency check.

Re-labelling a sample of already-labelled cells *without seeing the prior
label* measures how consistent the human is with themselves. That self-agreement
is the practical ceiling on any classifier — a model can't learn a sharper
good/bad boundary than the labels define — and the disagreements are exactly the
ambiguous cases worth tightening in the labelling rubric.

This module samples cells, scores a re-label against the original, and persists
the run to a **separate** file. It never touches ``labels.csv``.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Optional

import numpy as np

from .crops_h5 import CropRecord, HDF5CropsReader
from .labels_store import LABEL_NAMES, LabelsStore

SAMPLE_BALANCED = "balanced"
SAMPLE_RANDOM = "random"


def _kappa_band(k: float) -> str:
    """Landis & Koch interpretation band for Cohen's κ."""
    if k != k:           # NaN
        return "n/a"
    if k < 0.0:
        return "poor"
    if k < 0.20:
        return "slight"
    if k < 0.40:
        return "fair"
    if k < 0.60:
        return "moderate"
    if k < 0.80:
        return "substantial"
    return "almost perfect"


def sample_for_consistency(
    reader: HDF5CropsReader,
    store: LabelsStore,
    n: int,
    mode: str = SAMPLE_BALANCED,
    seed: Optional[int] = None,
) -> list[tuple[CropRecord, str]]:
    """Return ``(record, original_label)`` pairs to re-label blind.

    ``balanced`` draws ~n/2 of each class; ``random`` draws n across all labels
    (reflecting the true class ratio). Cells whose uid isn't in the HDF5 are
    skipped.
    """
    latest = store.load().latest_by_uid()
    uid_to_idx = {uid: i for i, uid in enumerate(reader.cell_uids)}

    by_class: dict[str, list[str]] = {name: [] for name in LABEL_NAMES}
    for uid, row in latest.items():
        if row.label_name in by_class and uid in uid_to_idx:
            by_class[row.label_name].append(uid)

    rng = np.random.default_rng(seed)
    picked: list[str] = []
    if mode == SAMPLE_BALANCED:
        per = max(1, n // len(LABEL_NAMES))
        for name in LABEL_NAMES:
            pool = by_class[name]
            take = min(per, len(pool))
            if take:
                picked.extend(rng.choice(pool, size=take, replace=False).tolist())
    else:  # random
        allu = [u for pool in by_class.values() for u in pool]
        take = min(n, len(allu))
        if take:
            picked.extend(rng.choice(allu, size=take, replace=False).tolist())

    rng.shuffle(picked)  # fresh order, so the re-label can't anchor on grouping
    return [(reader.get(uid_to_idx[uid]), latest[uid].label_name) for uid in picked]


def consistency_stats(pairs: list[tuple[str, str]], uids: Optional[list[str]] = None) -> dict:
    """Score re-labels against originals.

    ``pairs`` = ``[(original_label, recheck_label), …]`` over the cells that were
    re-labelled (skipped cells excluded by the caller). ``uids`` (optional, same
    order) lets the result list which cells were flipped.
    """
    n = len(pairs)
    if n == 0:
        return {"n": 0, "raw_agreement": float("nan"), "cohen_kappa": float("nan"),
                "kappa_band": "n/a", "flips": {}, "disagreements": []}

    agree = sum(1 for o, r in pairs if o == r)
    raw = agree / n

    from sklearn.metrics import cohen_kappa_score
    orig = [o for o, _ in pairs]
    recheck = [r for _, r in pairs]
    try:
        kappa = float(cohen_kappa_score(orig, recheck, labels=list(LABEL_NAMES)))
    except Exception:  # noqa: BLE001 — single-class etc.
        kappa = float("nan")

    # Per-class flip counts: how each originally-X cell was re-labelled.
    flips: dict[str, dict[str, int]] = {
        a: {b: 0 for b in LABEL_NAMES} for a in LABEL_NAMES
    }
    disagreements: list[str] = []
    for i, (o, r) in enumerate(pairs):
        if o in flips and r in flips[o]:
            flips[o][r] += 1
        if o != r and uids is not None:
            disagreements.append(uids[i])

    return {
        "n": n,
        "n_agree": agree,
        "raw_agreement": raw,
        "cohen_kappa": kappa,
        "kappa_band": _kappa_band(kappa),
        "flips": flips,
        "disagreements": disagreements,
    }


def save_consistency_run(
    labels_dir: Path,
    mode: str,
    rows: list[tuple[str, str, str]],
    stats: dict,
) -> Path:
    """Persist a run under ``<labels_dir>/consistency/<timestamp>/`` — a separate
    location that never collides with ``labels.csv``. ``rows`` =
    ``[(cell_uid, original, recheck), …]``. Returns the run directory."""
    from datetime import datetime

    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out = Path(labels_dir) / "consistency" / stamp
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cell_uid", "original_label", "recheck_label"])
        w.writerows(rows)
    (out / "summary.json").write_text(json.dumps(
        {"mode": mode, **{k: v for k, v in stats.items() if k != "disagreements"},
         "n_disagreements": len(stats.get("disagreements", []))},
        indent=2,
    ))
    return out
