"""Batch foci-detection helpers: per-unit worker, pool round-trip, and
the per-cell feature-stats fast path."""

from __future__ import annotations

import concurrent.futures as cf
import multiprocessing as mp

import numpy as np
import pytest

from mycomorph.core.foci import DetectorOpts
from mycomorph.core.foci.batch import detect_unit, resolve_worker_count
from mycomorph.core.foci.features import compute_features, features_dataframe
from mycomorph.core.foci.detectors import REGISTRY


def _scene(seed=0):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:96, 0:96].astype(np.float64)
    img = np.full((96, 96), 100.0)
    mask = np.zeros((96, 96), dtype=np.int32)
    mask[10:26, 10:40] = 1
    mask[50:66, 40:80] = 2
    img[mask > 0] += 30.0
    for y, x in ((18.0, 24.0), (58.0, 60.0), (58.0, 70.0)):
        img += 70.0 * np.exp(-((yy - y) ** 2 + (xx - x) ** 2) / (2 * 1.4**2))
    img += rng.normal(0, 2.0, img.shape)
    return img.astype(np.float32), mask


EXPECTED_META_COLS = {
    "detector", "well", "fov_index", "channel", "cell_label", "y", "x",
    "fov", "channel_index", "channel_name", "normalisation", "cell_id",
    "focus_id",
}


def test_detect_unit_matches_inline_composition():
    img, mask = _scene()
    det_opts = DetectorOpts(snr_min=2.0)
    df = detect_unit(
        img, mask, "hmax", det_opts,
        well="B2", fov_index=3, channel_index=1,
        channel_name="GFP", normalisation="tophat",
    )
    assert not df.empty
    assert EXPECTED_META_COLS <= set(df.columns)
    assert set(df["well"]) == {"B2"}
    assert set(df["fov"]) == {3}
    assert set(df["channel_index"]) == {1}
    assert set(df["channel_name"]) == {"GFP"}
    assert set(df["normalisation"]) == {"tophat"}
    assert list(df["focus_id"]) == list(range(len(df)))

    # Same detections/features as running the pieces by hand.
    foci = REGISTRY["hmax"]().detect(img, mask, det_opts)
    ref = features_dataframe(
        img, mask, foci, detector="hmax", well="B2", fov_index=3,
        channel="GFP",
    )
    assert len(ref) == len(df)
    np.testing.assert_allclose(
        df["intensity"].to_numpy(), ref["intensity"].to_numpy(),
    )
    np.testing.assert_allclose(df["y"].to_numpy(), ref["y"].to_numpy())


def test_detect_unit_round_trips_through_spawned_pool():
    img, mask = _scene()
    det_opts = DetectorOpts(snr_min=2.0)
    with cf.ProcessPoolExecutor(
        max_workers=2, mp_context=mp.get_context("spawn"),
    ) as pool:
        futures = [
            pool.submit(
                detect_unit, img, mask, "hmax", det_opts,
                well="W", fov_index=fov, channel_index=1,
                channel_name="GFP", normalisation="none",
            )
            for fov in (0, 1)
        ]
        dfs = [f.result(timeout=120) for f in futures]
    assert all(not d.empty for d in dfs)
    # Identical inputs → identical results regardless of worker.
    np.testing.assert_allclose(
        dfs[0]["intensity"].to_numpy(), dfs[1]["intensity"].to_numpy(),
    )
    assert set(dfs[0]["fov"]) == {0} and set(dfs[1]["fov"]) == {1}


def test_resolve_worker_count_env(monkeypatch):
    monkeypatch.setenv("MYCOMORPH_FOCI_WORKERS", "1")
    assert resolve_worker_count(100) == 1
    monkeypatch.setenv("MYCOMORPH_FOCI_WORKERS", "4")
    assert resolve_worker_count(100) == 4
    assert resolve_worker_count(2) == 2       # capped at task count
    monkeypatch.setenv("MYCOMORPH_FOCI_WORKERS", "garbage")
    assert resolve_worker_count(100) >= 1
    monkeypatch.delenv("MYCOMORPH_FOCI_WORKERS")
    assert 1 <= resolve_worker_count(100) <= 100


def test_cell_stats_fast_path_matches_direct_computation():
    """features_dataframe's once-per-cell bbox stats must equal the
    per-focus full-mask computation they replaced."""
    img, mask = _scene()
    det_opts = DetectorOpts(snr_min=2.0)
    foci = REGISTRY["hmax"]().detect(img, mask, det_opts)
    assert foci
    df = features_dataframe(img, mask, foci)
    for f, (_, row) in zip(foci, df.iterrows()):
        cid = int(f.cell_label)
        cell_pixels = (mask == cid) if cid > 0 else None
        ref = compute_features(img, f, cell_pixels=cell_pixels)
        for key in ("cell_p90", "cell_p95", "cell_median", "cell_std",
                    "prominence_p90", "prominence_median"):
            if np.isnan(ref[key]):
                assert np.isnan(row[key])
            else:
                assert row[key] == pytest.approx(ref[key], rel=1e-12)
