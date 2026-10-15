"""Score a saved model on the *current* frozen hold-out.

Training reports metrics on the hold-out as it was at training time. When the
hold-out changes (or to compare runs against one fixed yardstick), re-score
saved models here: a CNN (``best_model.pth``) or a feature model
(``feature_model.joblib``), from the ``test`` split of ``labels.csv``.

The result carries ``holdout_sha256``, a fingerprint of the (cell_uid, label)
pairs scored, so two numbers are only compared when they were measured on the
same hold-out.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Optional

import numpy as np


def holdout_fingerprint(uids, labels) -> str:
    """SHA-256 over sorted ``cell_uid<TAB>label`` lines."""
    h = hashlib.sha256()
    for uid, lab in sorted(zip((str(u) for u in uids), (int(x) for x in labels))):
        h.update(f"{uid}\t{lab}\n".encode("utf-8"))
    return h.hexdigest()


def _metrics(y: np.ndarray, p_bad: np.ndarray, pred: np.ndarray) -> dict:
    from sklearn.metrics import (
        average_precision_score,
        balanced_accuracy_score,
        confusion_matrix,
        roc_auc_score,
    )

    from .cell_quality_classifier import CLASS_NAMES
    from .train_classifier import single_class_warning

    counts = {CLASS_NAMES[i]: int((y == i).sum()) for i in range(len(CLASS_NAMES))}
    two = len(np.unique(y)) > 1
    tn, fp, fn, tp = (int(v) for v in confusion_matrix(y, pred, labels=[0, 1]).ravel())
    _, c = np.unique(y, return_counts=True)
    return {
        "n": int(len(y)),
        "eval_class_counts": counts,
        "eval_acc": float((pred == y).mean()),
        "balanced_acc": float(balanced_accuracy_score(y, pred)),
        "majority_baseline": float(c.max() / c.sum()),
        "roc_auc": float(roc_auc_score(y, p_bad)) if two else float("nan"),
        "average_precision": float(average_precision_score(y, p_bad, pos_label=1)) if two else float("nan"),
        "confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "warning": single_class_warning(counts),
    }


def _resolve(model: Path) -> Path:
    model = Path(model)
    if model.is_dir():
        for name in ("best_model.pth", "feature_model.joblib"):
            if (model / name).is_file():
                return model / name
        raise FileNotFoundError(f"No best_model.pth or feature_model.joblib in {model}")
    if not model.is_file():
        raise FileNotFoundError(f"Model not found: {model}")
    return model


def _read_config(model_file: Path) -> dict:
    cfg = model_file.parent / "training_config.json"
    try:
        return json.loads(cfg.read_text()) if cfg.is_file() else {}
    except (OSError, ValueError):
        return {}


def evaluate_model(h5_path: Path, labels_dir: Path, model: Path,
                   device: Optional[str] = "cpu", batch_size: int = 256) -> dict:
    """Metrics for ``model`` (a run dir, ``.pth`` or ``.joblib``) on the current
    hold-out of ``labels_dir``. Accuracy uses argmax (P(bad) >= 0.5), exactly
    as MycoMorph classifies; AUC/AP are threshold-free."""
    model_file = _resolve(model)
    cfg = _read_config(model_file)

    if model_file.suffix == ".joblib":
        y, p_bad, uids = _score_feature_model(h5_path, labels_dir, model_file, cfg)
        model_type = "features"
    else:
        y, p_bad, uids = _score_cnn(h5_path, labels_dir, model_file, cfg, device, batch_size)
        model_type = "cnn"

    pred = (p_bad >= 0.5).astype(int)
    out = {
        "model_path": str(model_file),
        "model_type": model_type,
        "h5": str(h5_path),
        "labels_dir": str(labels_dir),
        "holdout_sha256": holdout_fingerprint(uids, y),
        "trained_on_holdout_n": cfg.get("n_holdout"),
    }
    out.update(_metrics(y, p_bad, pred))
    return out


def _score_cnn(h5_path, labels_dir, model_file, cfg, device, batch_size):
    from .cell_quality_classifier import CellQualityClassifier
    from .crops_h5 import HDF5CropsReader
    from .labels_store import LabelsStore

    with HDF5CropsReader(h5_path) as reader:
        crops, labels, uids = LabelsStore(labels_dir).to_eval_dataset(reader)
    if len(labels) == 0:
        raise ValueError(f"No hold-out test set in {labels_dir} (no 'test' split labels).")
    # Same preprocessing as training's validation/hold-out pass (and MycoMorph's
    # inference crops): re-normalise the phase channel(s) of each crop to [0, 1]
    # after the 128 -> 64 px resize. Without it the numbers drift from the ones
    # train() reported.
    from .train_classifier import _normalise_phase_inplace, _phase_count

    crops = np.array(crops, dtype=np.float32, copy=True)
    n_phase = _phase_count(crops.shape[1])
    for x in crops:
        _normalise_phase_inplace(x, n_phase)
    clf = CellQualityClassifier(str(model_file), in_channels=int(cfg.get("in_channels", 3)),
                                device=device)
    probs = np.concatenate([clf.predict_batch(crops[i:i + batch_size])[1]
                            for i in range(0, len(crops), batch_size)])
    return np.asarray(labels, dtype=int), probs[:, 1], list(uids)


def _score_feature_model(h5_path, labels_dir, model_file, cfg):
    import joblib

    from .feature_classifier import load_feature_matrix

    bundle = joblib.load(model_file)
    groups = cfg.get("feature_groups") or {}
    matrix = load_feature_matrix(
        h5_path, labels_dir,
        with_fragment=groups.get("with_fragment", True),
        with_context=groups.get("with_context", True),
        with_focus=groups.get("with_focus", True),
        phase_only=groups.get("phase_only", False),
    )
    te = matrix[matrix["split"] == "test"]
    if te.empty:
        raise ValueError(f"No hold-out test set in {labels_dir} (no 'test' split labels "
                         "with a row in all_features).")
    cols = list(bundle["feature_columns"])
    missing = [c for c in cols if c not in te.columns]
    if missing:
        raise ValueError(f"Feature table lacks {len(missing)} column(s) the model was "
                         f"trained on, e.g. {missing[0]!r}.")
    p_bad = bundle["model"].predict_proba(te[cols].to_numpy(dtype=np.float64))[:, 1]
    return te["label"].to_numpy(dtype=int), p_bad, te["cell_uid"].tolist()
