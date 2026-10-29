"""Tests for the learning-curve computation."""

from __future__ import annotations

import numpy as np

from mycomorph.modelprep.core.learning_curve import (
    _nested_stratified_indices,
    compute_learning_curve,
)


def _synthetic(n_good: int, n_bad: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    n = n_good + n_bad
    crops = rng.random((n, 3, 64, 64), dtype=np.float32)
    crops[:, 1] = (crops[:, 1] > 0.5).astype(np.float32)
    crops[:, 2] = 0.3
    labels = np.array([0] * n_good + [1] * n_bad, dtype=np.int64)
    crops[labels == 0, 0] += 0.3  # weak separable signal
    return crops, labels


def test_nested_stratified_subsets_are_nested_and_balanced():
    labels = np.array([0] * 40 + [1] * 60)
    fracs = np.linspace(0.1, 1.0, 4)
    subsets = _nested_stratified_indices(labels, fracs, seed=1)

    sizes = [s.size for s in subsets]
    assert sizes == sorted(sizes)                    # non-decreasing
    for a, b in zip(subsets, subsets[1:]):
        assert set(a.tolist()).issubset(set(b.tolist()))  # nested
    # Each subset keeps both classes.
    for s in subsets:
        cls = labels[s]
        assert (cls == 0).any() and (cls == 1).any()


def test_compute_learning_curve_returns_increasing_points():
    train_crops, train_labels = _synthetic(40, 60, seed=1)
    eval_crops, eval_labels = _synthetic(20, 20, seed=2)

    seen = []
    points = compute_learning_curve(
        train_crops, train_labels, eval_crops, eval_labels,
        n_points=2, epochs=1, seed=0, device="cpu",
        progress_cb=lambda f, m: seen.append(f),
    )
    assert len(points) == 2
    assert [p["n"] for p in points] == sorted(p["n"] for p in points)
    for p in points:
        for key in ("roc_auc", "average_precision", "balanced_acc", "eval_acc"):
            assert key in p
        assert p["n"] == p["n_good"] + p["n_bad"]
    assert seen and max(seen) <= 1.0


def test_compute_learning_curve_requires_holdout():
    train_crops, train_labels = _synthetic(20, 20)
    empty_crops = np.empty((0, 3, 64, 64), dtype=np.float32)
    empty_labels = np.empty((0,), dtype=np.int64)
    try:
        compute_learning_curve(train_crops, train_labels, empty_crops, empty_labels,
                               n_points=2, epochs=1, device="cpu")
        assert False, "expected ValueError for empty hold-out"
    except ValueError:
        pass
