"""Feature-based cell-quality classifier.

An alternative to the CNN: gradient-boosted trees on **measured** per-cell
features — MycoMorph's morphometrics (``all_features.parquet``, sibling of the
crops HDF5) plus the segmentation-quality / "crossing fragment" cues from
:mod:`fragment_features`. Trained on the labels' ``train`` split and evaluated
on the frozen ``test`` hold-out, so its numbers are directly comparable to the
CNN's.

The point: when the good/bad signal is geometric (e.g. "a crossing split the
cell into a fragment"), measured features beat a CNN on an isolated, downsized
crop — and the importances say *which* features carry the decision.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from .labels_store import LABEL_TO_IDX, LabelsStore

# Identifier / positional columns dropped from the feature matrix (not morphology,
# and some would leak FOV position rather than cell quality).
_DROP_COLS = {
    "cell_uid", "run_id", "well", "fov", "cell_id", "source_czi",
    "plate_acquisition_datetime", "acquisition_time",
    "centroid_y", "centroid_x", "bbox_y0", "bbox_x0", "bbox_y1", "bbox_x1",
    "orientation_rad", "label", "split",
}


def _features_path(h5_path: Path) -> Optional[Path]:
    base = Path(h5_path).parent
    for name in ("all_features.parquet", "all_features.csv"):
        p = base / name
        if p.exists():
            return p
    return None


def load_feature_matrix(
    h5_path: Path,
    labels_dir: Path,
    with_fragment: bool = True,
    with_context: bool = True,
    with_focus: bool = True,
):
    """Return a DataFrame of numeric features + ``label`` (0/1) + ``split`` for
    every labelled cell, joining the morphometric table to the labels.

    ``with_context`` adds FOV-context features (neighbour density etc.);
    ``with_fragment`` adds the crossing-fragment cues from the crops.
    """
    import pandas as pd

    fpath = _features_path(h5_path)
    if fpath is None:
        raise FileNotFoundError(
            f"No all_features.parquet/.csv next to {Path(h5_path).name}. "
            "The feature-based model needs MycoMorph's 04_features table."
        )
    feats = pd.read_parquet(fpath) if fpath.suffix == ".parquet" else pd.read_csv(fpath)

    latest = LabelsStore(labels_dir).load().latest_by_uid()
    if not latest:
        raise ValueError(f"No labels in {labels_dir}.")
    labels = pd.DataFrame(
        [(uid, LABEL_TO_IDX[r.label_name], r.split)
         for uid, r in latest.items() if r.label_name in LABEL_TO_IDX],
        columns=["cell_uid", "label", "split"],
    )

    matrix = feats.merge(labels, on="cell_uid", how="inner")
    if with_context:
        from .context_features import compute_context_features
        ctx = compute_context_features(h5_path, cell_uids=matrix["cell_uid"].tolist())
        matrix = matrix.merge(ctx, on="cell_uid", how="left")
    if with_focus or with_fragment:
        from .crop_features import compute_crop_features
        crop = compute_crop_features(h5_path, cell_uids=matrix["cell_uid"].tolist(),
                                     focus=with_focus, fragment=with_fragment)
        matrix = matrix.merge(crop, on="cell_uid", how="left")
    return matrix


def _feature_columns(matrix) -> list[str]:
    import pandas as pd  # noqa: F401
    return [c for c in matrix.columns
            if c not in _DROP_COLS
            and np.issubdtype(matrix[c].dtype, np.number)]


def train_feature_classifier(matrix, seed: int = 42, n_importance_repeats: int = 10,
                             return_estimator: bool = False) -> dict:
    """Fit a HistGradientBoosting classifier on the train split, evaluate on the
    frozen hold-out, and return metrics + permutation importances.

    With ``return_estimator`` the result also carries the fitted model and the
    hold-out arrays (keys prefixed ``_``) so a caller can persist it / draw curves.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.inspection import permutation_importance
    from sklearn.metrics import (
        average_precision_score,
        balanced_accuracy_score,
        confusion_matrix,
        roc_auc_score,
    )

    cols = _feature_columns(matrix)
    tr = matrix[matrix["split"] == "train"]
    te = matrix[matrix["split"] == "test"]
    if len(te) == 0:
        raise ValueError("Feature model needs a hold-out test set (no 'test' split labels).")
    if tr["label"].nunique() < 2:
        raise ValueError("Train split has only one class.")

    Xtr, ytr = tr[cols].to_numpy(dtype=np.float64), tr["label"].to_numpy()
    Xte, yte = te[cols].to_numpy(dtype=np.float64), te["label"].to_numpy()

    clf = HistGradientBoostingClassifier(class_weight="balanced", random_state=seed)
    clf.fit(Xtr, ytr)

    proba_bad = clf.predict_proba(Xte)[:, 1]   # P(class 1 = bad), matches the CNN
    preds = (proba_bad >= 0.5).astype(int)

    roc = float(roc_auc_score(yte, proba_bad)) if len(np.unique(yte)) > 1 else float("nan")
    ap = float(average_precision_score(yte, proba_bad, pos_label=1)) if len(np.unique(yte)) > 1 else float("nan")
    bal = float(balanced_accuracy_score(yte, preds))
    acc = float((preds == yte).mean())
    tn, fp, fn, tp = (int(v) for v in confusion_matrix(yte, preds, labels=[0, 1]).ravel())
    _, te_counts = np.unique(yte, return_counts=True)
    majority_baseline = float(te_counts.max() / te_counts.sum())

    imp = permutation_importance(clf, Xte, yte, n_repeats=n_importance_repeats,
                                 random_state=seed, scoring="balanced_accuracy")
    order = np.argsort(imp.importances_mean)[::-1]
    top = [(cols[i], float(imp.importances_mean[i]), float(imp.importances_std[i]))
           for i in order[:15] if imp.importances_mean[i] > 0]

    summary = {
        "n_train": int(len(tr)), "n_test": int(len(te)), "n_features": len(cols),
        "roc_auc": roc, "average_precision": ap,
        "balanced_acc": bal, "eval_acc": acc, "majority_baseline": majority_baseline,
        "confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "top_importances": top,
        "with_fragment": any(c.startswith("frag_") for c in cols),
        "with_context": any(c.startswith("ctx_") for c in cols),
        "with_focus": any(c.startswith("focus_") for c in cols),
    }
    if return_estimator:
        summary["_clf"] = clf
        summary["_feature_cols"] = cols
        summary["_yte"] = yte
        summary["_proba_bad"] = proba_bad
        summary["_train_class_counts"] = {
            "good": int((tr["label"] == 0).sum()), "bad": int((tr["label"] == 1).sum()),
        }
    return summary


def train_and_save(h5_path: Path, labels_dir: Path, output_dir: Path,
                   seed: int = 42, provenance: Optional[dict] = None) -> dict:
    """Train the all-features model and persist a run dir that mirrors the CNN's
    artefact shape (so it lists in the Models tab + renders in the inspector).
    Returns the JSON-safe summary (with ``model_path`` + ``model_type``)."""
    import json
    import joblib

    from .cell_quality_classifier import CLASS_NAMES
    from .train_classifier import _binary_curves

    matrix = load_feature_matrix(h5_path, labels_dir)  # all 4 groups by default
    s = train_feature_classifier(matrix, seed=seed, return_estimator=True)
    clf = s.pop("_clf")
    cols = s.pop("_feature_cols")
    yte = s.pop("_yte")
    proba = s.pop("_proba_bad")
    class_counts = s.pop("_train_class_counts")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": clf, "feature_columns": cols}, output_dir / "feature_model.joblib")

    curves = _binary_curves(np.asarray(yte), np.asarray(proba))
    (output_dir / "metrics.json").write_text(json.dumps({
        "num_classes": 2, "class_names": list(CLASS_NAMES), "eval_split": "holdout",
        "decision_threshold": 0.5, "binary": curves,
    }, indent=2))

    config = {
        "model_type": "features",
        "class_names": list(CLASS_NAMES),
        "n_train": s["n_train"], "n_val": 0, "n_test": 0, "n_holdout": s["n_test"],
        "eval_split": "holdout",
        "class_counts": class_counts,
        "eval_acc": s["eval_acc"], "balanced_acc": s["balanced_acc"],
        "majority_baseline": s["majority_baseline"],
        "roc_auc": s["roc_auc"], "average_precision": s["average_precision"],
        "decision_threshold": 0.5, "confusion": s["confusion"],
        "n_features": s["n_features"], "top_importances": s["top_importances"],
        "feature_groups": {k: s[k] for k in ("with_context", "with_focus", "with_fragment")},
        "provenance": provenance or {},
    }
    (output_dir / "training_config.json").write_text(json.dumps(config, indent=2))

    s["model_path"] = str(output_dir / "feature_model.joblib")
    s["model_type"] = "features"
    return s
