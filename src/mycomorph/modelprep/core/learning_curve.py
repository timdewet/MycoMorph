"""Learning curve: does labelling more cells still help?

Trains the classifier on increasing, nested, class-stratified subsamples of the
labelled **train** split and evaluates every one on the same frozen **hold-out**
test set. Plotting AUC / AP / balanced-accuracy against training-set size shows
where the curve plateaus — i.e. where more labelling stops paying off.

Each point is trained **from scratch** (``pretrained_path=None``) for a clean
curve, to a throwaway temp directory so nothing lands in ``~/.modelprep/models``.
The trainer's external-hold-out path (``eval_crops``/``eval_labels``) is reused so
every point is scored on the identical fixed set.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Callable, Optional

import numpy as np

# Minimum cells per class at the smallest point, so the trainer's internal
# train/val split (and class-weighting) stays valid.
_MIN_PER_CLASS = 8


def _nested_stratified_indices(
    labels: np.ndarray, fractions: np.ndarray, seed: int,
) -> list[np.ndarray]:
    """One index array per fraction. Each is a class-balanced prefix of a
    per-class shuffle, so smaller subsets nest inside larger ones and every
    subset keeps the full good/bad ratio."""
    rng = np.random.default_rng(seed)
    per_class: dict[int, np.ndarray] = {}
    for c in np.unique(labels):
        idx = np.flatnonzero(labels == c)
        rng.shuffle(idx)
        per_class[int(c)] = idx

    out: list[np.ndarray] = []
    for f in fractions:
        parts = []
        ok = True
        for c, idx in per_class.items():
            n = int(round(len(idx) * float(f)))
            n = max(n, min(_MIN_PER_CLASS, len(idx)))
            if n < _MIN_PER_CLASS:
                ok = False
                break
            parts.append(idx[:n])
        out.append(np.concatenate(parts) if ok and parts else np.array([], dtype=int))
    return out


def compute_learning_curve(
    train_crops: np.ndarray,
    train_labels: np.ndarray,
    eval_crops: np.ndarray,
    eval_labels: np.ndarray,
    *,
    n_points: int = 5,
    epochs: int = 25,
    seed: int = 42,
    device: Optional[str] = None,
    progress_cb: Optional[Callable[[float, str], None]] = None,
    should_cancel: Optional[Callable[[], bool]] = None,
) -> list[dict]:
    """Return one dict per trained size (skipped sizes are omitted).

    Keys: ``n``, ``n_good``, ``n_bad``, ``roc_auc``, ``average_precision``,
    ``balanced_acc``, ``eval_acc``.
    """
    from .train_classifier import train as run_train

    train_labels = np.asarray(train_labels)
    if eval_labels is None or len(eval_labels) == 0:
        raise ValueError("Learning curve needs a hold-out test set to evaluate on.")
    if len(np.unique(train_labels)) < 2:
        raise ValueError("Need both 'good' and 'bad' labels in the train split.")

    fractions = np.linspace(0.1, 1.0, max(2, int(n_points)))
    subsets = _nested_stratified_indices(train_labels, fractions, seed)

    points: list[dict] = []
    with tempfile.TemporaryDirectory(prefix="modelprep_lc_") as tmp:
        for i, sel in enumerate(subsets):
            if should_cancel is not None and should_cancel():
                break
            if sel.size == 0:
                continue  # too few cells at this fraction — skip
            sub_labels = train_labels[sel]
            if len(np.unique(sub_labels)) < 2:
                continue

            n = int(sel.size)
            base = i / len(subsets)

            def _point_progress(frac: float, msg: str, _n=n, _base=base) -> None:
                if progress_cb is not None:
                    overall = _base + (max(0.0, min(1.0, frac)) / len(subsets))
                    progress_cb(overall, f"n={_n:,}: {msg}")

            summary = run_train(
                crops=train_crops[sel],
                labels=sub_labels,
                eval_crops=eval_crops,
                eval_labels=eval_labels,
                output_dir=Path(tmp) / f"point_{i}",
                epochs=epochs,
                seed=seed,
                device=device,
                pretrained_path=None,
                progress_cb=_point_progress,
            )
            points.append({
                "n": n,
                "n_good": int((sub_labels == 0).sum()),
                "n_bad": int((sub_labels == 1).sum()),
                "roc_auc": summary.get("roc_auc"),
                "average_precision": summary.get("average_precision"),
                "balanced_acc": summary.get("balanced_acc"),
                "eval_acc": summary.get("eval_acc"),
            })
            if progress_cb is not None:
                progress_cb((i + 1) / len(subsets),
                            f"n={n:,}: AUC {summary.get('roc_auc', float('nan')):.3f}")

    return points
