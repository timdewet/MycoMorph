"""Unit + CPU-smoke tests for the trainer."""

from __future__ import annotations

import numpy as np
import pytest

from modelprep.core.train_classifier import (
    _make_splits,
    _resolve_device,
    compute_class_weights,
    make_augmentation_transforms,
    train,
)


def _synthetic(n_good: int, n_bad: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    n = n_good + n_bad
    crops = rng.random((n, 3, 64, 64), dtype=np.float32)
    crops[:, 1] = (crops[:, 1] > 0.5).astype(np.float32)  # binary mask
    crops[:, 2] = 0.3                                      # constant area plane
    labels = np.array([0] * n_good + [1] * n_bad, dtype=np.int64)
    return crops, labels


def test_compute_class_weights_inverse_frequency():
    import torch
    w = compute_class_weights(np.array([0, 0, 0, 1]), 2)
    assert torch.is_tensor(w) and len(w) == 2
    assert w[1] > w[0]                              # rarer class weighted heavier
    assert abs(float(w.sum()) - 2.0) < 1e-5         # sums to num_classes


def test_resolve_device():
    assert _resolve_device("cpu") == "cpu"
    assert _resolve_device(None) in ("cpu", "mps", "cuda")


def test_make_splits_three_way_disjoint():
    labels = np.array([0] * 20 + [1] * 20)
    tr, va, te, note = _make_splits(labels, 0.2, 0.15, 42)
    assert len(te) > 0 and "3-way" in note
    s_tr, s_va, s_te = set(tr.tolist()), set(va.tolist()), set(te.tolist())
    assert s_tr & s_va == set() and s_tr & s_te == set() and s_va & s_te == set()
    assert len(s_tr | s_va | s_te) == 40


def test_make_splits_tiny_falls_back_to_two_way():
    labels = np.array([0] * 12 + [1] * 3)   # rarer class < 4 → no test split
    tr, va, te, note = _make_splits(labels, 0.2, 0.15, 42)
    assert len(te) == 0 and "2-way" in note


def test_phase_only_augmentation_protects_mask_and_area():
    import torch
    train_tf, _ = make_augmentation_transforms()
    crops, _ = _synthetic(1, 0)
    torch.manual_seed(0)
    y = train_tf(torch.from_numpy(crops[0].copy()))
    assert set(torch.unique(y[1]).tolist()) <= {0.0, 1.0}      # mask stays binary
    assert float(y[2].max() - y[2].min()) < 1e-6               # area stays constant


def test_train_cpu_smoke(tmp_path):
    crops, labels = _synthetic(20, 20)
    summary = train(crops=crops, labels=labels, output_dir=tmp_path,
                    epochs=2, device="cpu", seed=0)
    for fname in ("best_model.pth", "final_model.pth",
                  "training_config.json", "metrics.json", "train.log"):
        assert (tmp_path / fname).exists(), f"missing artefact: {fname}"
    assert summary["eval_split"] == "test"
    assert "decision_threshold" in summary and "confusion" in summary


def test_train_reproducible(tmp_path):
    crops, labels = _synthetic(20, 20)
    a = train(crops=crops, labels=labels, output_dir=tmp_path / "a",
              epochs=2, device="cpu", seed=7)
    b = train(crops=crops, labels=labels, output_dir=tmp_path / "b",
              epochs=2, device="cpu", seed=7)
    assert abs(a["best_val_acc"] - b["best_val_acc"]) < 1e-9


def test_train_single_class_raises(tmp_path):
    crops, labels = _synthetic(10, 0)
    with pytest.raises(ValueError):
        train(crops=crops, labels=labels, output_dir=tmp_path, epochs=1, device="cpu")
