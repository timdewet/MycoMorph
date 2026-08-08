"""The live preview's foci side-stage runs on the worker thread.

These tests pin the worker-side behavior of ``_maybe_run_foci``: it
detects on the (optionally normalised) fluorescence channels, honours
both cache keys (norm + detection), and returns ``None`` when everything
is already cached — the controller then repaints from cache without any
main-thread compute.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from mycomorph.gui.pipeline.context import (
    FluorescentNormalisationOpts,
    FociDetectionOpts,
)
from mycomorph.gui.widgets.live_preview.cache import CacheEntry
from mycomorph.gui.widgets.live_preview.worker import (
    FociPayload,
    PreviewWorker,
    RenderRequest,
)


def _scene() -> tuple[np.ndarray, np.ndarray]:
    """(C, Y, X) channels (phase + one fluor with 2 spots) and a mask."""
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[0:64, 0:64].astype(np.float64)
    fluor = np.full((64, 64), 100.0)
    for y, x in ((20.0, 20.0), (44.0, 40.0)):
        fluor += 80.0 * np.exp(-((yy - y) ** 2 + (xx - x) ** 2) / (2 * 1.4**2))
    fluor += rng.normal(0, 2.0, fluor.shape)
    phase = np.full((64, 64), 500.0)
    channels = np.stack([phase, fluor]).astype(np.float32)
    mask = np.zeros((64, 64), dtype=np.int32)
    mask[12:28, 12:28] = 1
    mask[36:52, 32:48] = 2
    return channels, mask


def _worker(entry: CacheEntry, fluor_method: str = "none") -> PreviewWorker:
    req = RenderRequest(
        sample_path=Path("/nonexistent/sample.tif"),
        fov_index=0,
        target_stage="classify",
        phase_channel=0,
        segment_opts=None,
        foci_tab="foci_det",
        fluor_norm_opts=FluorescentNormalisationOpts(method=fluor_method),
        foci_det_opts=FociDetectionOpts(detector_keys=["wavelet"]),
        cached_entry=entry,
    )
    return PreviewWorker(req)


def test_worker_foci_stage_detects():
    channels, mask = _scene()
    entry = CacheEntry(focus_key=("focus", "k"))
    payload = _worker(entry)._maybe_run_foci(channels, mask)
    assert isinstance(payload, FociPayload)
    assert payload.foci_key is not None
    assert payload.det_key == "wavelet"
    df = payload.df
    assert df is not None and not df.empty
    # Both spots found, on the fluor channel, inside their cells.
    assert set(df["channel_index"]) == {1}
    assert set(df["cell_label"]) >= {1, 2}
    # Pass-through normalisation → no norm stack delivered.
    assert payload.norm_key is None and payload.norm_channels is None


def test_worker_foci_stage_cache_hit_returns_none():
    channels, mask = _scene()
    entry = CacheEntry(focus_key=("focus", "k"))
    worker = _worker(entry)
    first = worker._maybe_run_foci(channels, mask)
    assert first is not None
    # Simulate the controller landing the payload into the cache.
    entry.foci_key = first.foci_key
    entry.foci_df = first.df
    assert _worker(entry)._maybe_run_foci(channels, mask) is None


def test_worker_foci_stage_normalises_and_caches():
    channels, mask = _scene()
    entry = CacheEntry(focus_key=("focus", "k"))
    payload = _worker(entry, fluor_method="tophat")._maybe_run_foci(
        channels, mask,
    )
    assert payload is not None
    assert payload.norm_key is not None
    assert payload.norm_channels is not None
    assert payload.norm_channels.shape == channels.shape
    # Phase channel untouched; fluor channel background-suppressed.
    np.testing.assert_allclose(payload.norm_channels[0], channels[0])
    assert float(np.median(payload.norm_channels[1])) < float(
        np.median(channels[1])
    )

    # Land both results, then a second pass is a full cache hit.
    entry.norm_key = payload.norm_key
    entry.norm_channels = payload.norm_channels
    entry.foci_key = payload.foci_key
    entry.foci_df = payload.df
    assert (
        _worker(entry, fluor_method="tophat")._maybe_run_foci(channels, mask)
        is None
    )


def test_worker_foci_stage_norm_only_on_fluor_norm_tab():
    channels, mask = _scene()
    entry = CacheEntry(focus_key=("focus", "k"))
    req = RenderRequest(
        sample_path=Path("/nonexistent/sample.tif"),
        fov_index=0,
        target_stage="classify",
        phase_channel=0,
        segment_opts=None,
        foci_tab="fluor_norm",
        fluor_norm_opts=FluorescentNormalisationOpts(method="tophat"),
        foci_det_opts=None,
        cached_entry=entry,
    )
    payload = PreviewWorker(req)._maybe_run_foci(channels, mask)
    assert payload is not None
    assert payload.norm_channels is not None
    # No detection on the fluor_norm tab.
    assert payload.foci_key is None and payload.df is None
