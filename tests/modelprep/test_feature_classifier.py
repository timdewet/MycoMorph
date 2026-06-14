"""Tests for the feature-based classifier + fragment features."""

from __future__ import annotations

import numpy as np
import pytest

pd = pytest.importorskip("pandas")

from modelprep.core.feature_classifier import (
    _feature_columns,
    load_feature_matrix,
    train_feature_classifier,
)
from modelprep.core.fragment_features import _features_for


def _separable_matrix(n=120, seed=0):
    rng = np.random.default_rng(seed)
    # signal feature separates the classes; noise feature doesn't.
    label = np.array([0] * (n // 2) + [1] * (n // 2))
    signal = label + rng.normal(0, 0.3, n)
    noise = rng.normal(0, 1, n)
    split = np.array(["train"] * (n // 2) + ["test"] * (n // 2))
    # interleave train/test across both classes
    split = np.where(np.arange(n) % 2 == 0, "train", "test")
    return pd.DataFrame({
        "cell_uid": [f"c{i}" for i in range(n)],
        "signal": signal, "noise": noise,
        "label": label, "split": split,
    })


def test_feature_columns_drop_ids_and_targets():
    m = _separable_matrix()
    cols = _feature_columns(m)
    assert "signal" in cols and "noise" in cols
    assert "cell_uid" not in cols and "label" not in cols and "split" not in cols


def test_train_feature_classifier_learns_signal():
    m = _separable_matrix()
    s = train_feature_classifier(m, n_importance_repeats=5)
    assert s["n_train"] > 0 and s["n_test"] > 0
    assert s["balanced_acc"] > 0.8          # signal is easily separable
    # The signal feature should rank above pure noise.
    top = {name for name, *_ in s["top_importances"]}
    assert "signal" in top


def test_train_feature_classifier_requires_holdout():
    m = _separable_matrix()
    m["split"] = "train"     # no test rows
    with pytest.raises(ValueError):
        train_feature_classifier(m)


def test_fragment_features_distinguish_cut_from_clean():
    H = W = 128

    def crop(extend):
        bright = np.full((H, W), 0.8, np.float32)
        mask = np.zeros((H, W), bool)
        mask[60:68, 30:80] = True
        bright[60:68, 30:80] = 0.2          # dark cell under the mask
        if extend:
            bright[60:68, 80:110] = 0.2      # brightfield continues past mask end
        return bright, mask

    _, _, cont_clean = _features_for(*crop(False))
    _, _, cont_cut = _features_for(*crop(True))
    assert cont_cut > 0.5 and cont_clean < 0.2
    # A forked mask is flagged via branchpoints.
    bright = np.full((H, W), 0.8, np.float32)
    mask = np.zeros((H, W), bool)
    mask[60:68, 30:80] = True
    mask[40:68, 52:60] = True               # a T-branch
    bright[mask] = 0.2
    eps, branch, _ = _features_for(bright, mask)
    assert branch > 0 and eps >= 3


def test_load_feature_matrix_joins_and_drops(synthetic_h5, tmp_path):
    from modelprep.core.labels_store import LabelRow, LabelsStore, SPLIT_TEST, SPLIT_TRAIN

    # A morphometric table next to the crops h5.
    uids = [f"cell_{i:04d}" for i in range(32)]
    feats = pd.DataFrame({
        "cell_uid": uids,
        "run_id": ["r"] * 32,                 # id col → dropped
        "solidity": np.linspace(0.5, 1.0, 32),
        "width_variation": np.linspace(0, 1, 32),
    })
    feats.to_parquet(synthetic_h5.parent / "all_features.parquet")

    store = LabelsStore(tmp_path / "labels")
    store.ensure_initialised(source_h5=synthetic_h5, total_cells=32, h5_schema_version="1.0")
    store.append_batch(
        [LabelRow(cell_uid=uids[i], label_idx=0, label_name="good",
                  split=SPLIT_TRAIN if i % 2 else SPLIT_TEST) for i in range(0, 16)]
        + [LabelRow(cell_uid=uids[i], label_idx=1, label_name="bad",
                    split=SPLIT_TRAIN if i % 2 else SPLIT_TEST) for i in range(16, 32)]
    )

    m = load_feature_matrix(synthetic_h5, tmp_path / "labels",
                            with_fragment=False, with_context=False)
    assert len(m) == 32
    assert {"label", "split", "solidity", "width_variation"}.issubset(m.columns)
    cols = _feature_columns(m)
    assert "solidity" in cols and "run_id" not in cols


def test_context_features_neighbours_and_overlap(tmp_path):
    from modelprep.core.context_features import compute_context_features

    # FOV 0: two cells 10px apart whose bboxes overlap. FOV 1: one lonely cell.
    df = pd.DataFrame({
        "cell_uid": ["a", "b", "c"],
        "well": ["w", "w", "w"],
        "fov": [0, 0, 1],
        "centroid_y": [100.0, 110.0, 500.0],
        "centroid_x": [100.0, 100.0, 500.0],
        "bbox_y0": [90, 100, 490], "bbox_x0": [90, 90, 490],
        "bbox_y1": [115, 125, 515], "bbox_x1": [110, 110, 510],
    })
    df.to_parquet(tmp_path / "all_features.parquet")

    ctx = compute_context_features(tmp_path / "all_crops.h5", radii=(50.0, 100.0))
    ctx = ctx.set_index("cell_uid")
    # a and b are 10px apart → each other's neighbour; c is alone.
    assert ctx.loc["a", "ctx_neigh_50"] == 1 and ctx.loc["b", "ctx_neigh_50"] == 1
    assert ctx.loc["c", "ctx_neigh_50"] == 0
    assert ctx.loc["a", "ctx_nn_dist"] == pytest.approx(10.0)
    assert ctx.loc["a", "ctx_bbox_overlaps"] == 1   # a & b overlap
    assert ctx.loc["c", "ctx_bbox_overlaps"] == 0
    assert ctx.loc["a", "ctx_fov_count"] == 2 and ctx.loc["c", "ctx_fov_count"] == 1


def test_focus_sharpness_ranks_sharp_above_blurred():
    from scipy.ndimage import gaussian_filter
    from modelprep.core.focus_features import _sharpness

    rng = np.random.default_rng(0)
    sharp = rng.random((128, 128)).astype(np.float32)
    blurred = gaussian_filter(sharp, sigma=3)
    assert _sharpness(sharp)[0] > _sharpness(blurred)[0]      # lap_var
    assert _sharpness(sharp)[1] > _sharpness(blurred)[1]      # tenengrad


def test_focus_features_from_h5(synthetic_h5):
    from modelprep.core.focus_features import compute_focus_features

    foc = compute_focus_features(synthetic_h5)
    assert len(foc) == 32
    assert {"focus_lap_var", "focus_tenengrad", "focus_norm_var"}.issubset(foc.columns)
    assert foc["focus_lap_var"].notna().all()


def test_crop_features_unified_pass(synthetic_h5):
    from modelprep.core.crop_features import compute_crop_features

    df = compute_crop_features(synthetic_h5, focus=True, fragment=True)
    assert len(df) == 32
    assert {"focus_lap_var", "frag_branchpoints", "frag_bright_continuation"}.issubset(df.columns)


def _planted_dataset(synthetic_h5, tmp_path):
    """A morphometric parquet with a separable signal + labels with a hold-out."""
    from modelprep.core.labels_store import LabelRow, LabelsStore, SPLIT_TEST, SPLIT_TRAIN

    uids = [f"cell_{i:04d}" for i in range(32)]
    y = np.array([0] * 16 + [1] * 16)
    rng = np.random.default_rng(0)
    feats = pd.DataFrame({
        "cell_uid": uids, "well": ["w"] * 32, "fov": np.zeros(32, int),
        "centroid_y": rng.random(32) * 1000, "centroid_x": rng.random(32) * 1000,
        "bbox_y0": np.zeros(32, int), "bbox_x0": np.zeros(32, int),
        "bbox_y1": np.full(32, 20), "bbox_x1": np.full(32, 20),
        "solidity": y + rng.normal(0, 0.3, 32), "width_variation": rng.normal(0, 1, 32),
    })
    feats.to_parquet(synthetic_h5.parent / "all_features.parquet")
    store = LabelsStore(tmp_path / "labels")
    store.ensure_initialised(source_h5=synthetic_h5, total_cells=32, h5_schema_version="1.0")
    store.append_batch([
        LabelRow(cell_uid=uids[i], label_idx=int(y[i]),
                 label_name="good" if y[i] == 0 else "bad",
                 split=SPLIT_TRAIN if i % 3 else SPLIT_TEST) for i in range(32)])
    return tmp_path / "labels"


def test_train_and_save_writes_feature_run(synthetic_h5, tmp_path):
    import json
    from modelprep.core.feature_classifier import train_and_save

    labels_dir = _planted_dataset(synthetic_h5, tmp_path)
    out = tmp_path / "demo-features_2026-06-14_10-00-00"
    s = train_and_save(synthetic_h5, labels_dir, out, provenance={"model_name": "demo"})

    assert s["model_type"] == "features"
    assert (out / "feature_model.joblib").exists()
    assert (out / "metrics.json").exists()
    cfg = json.loads((out / "training_config.json").read_text())
    assert cfg["model_type"] == "features"
    assert "roc_auc" in cfg and "balanced_acc" in cfg and cfg["eval_split"] == "holdout"
