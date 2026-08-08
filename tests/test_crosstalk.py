from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import tifffile

from mycomorph.core.foci import DetectorOpts
from mycomorph.core.foci.crosstalk import (
    estimate_crosstalk_k,
    subtract_crosstalk,
)
from mycomorph.core.provenance import artifact_signature_path


def _add_spot(img: np.ndarray, y: int, x: int, amp: float, sigma: float = 1.5) -> None:
    yy, xx = np.mgrid[-6:7, -6:7]
    img[y - 6:y + 7, x - 6:x + 7] += amp * np.exp(
        -(yy**2 + xx**2) / (2 * sigma**2)
    )


def _synthetic_pair(k_true: float = 0.08, seed: int = 0):
    """Two-FOV red/green stacks: red foci bleed ``k_true`` into green;
    green also has two genuine foci, one of them colocalised with a red
    focus (an outlier the robust fit must shrug off)."""
    rng = np.random.default_rng(seed)
    red = np.full((2, 256, 256), 100.0, dtype=np.float32)
    green_true = np.full((2, 256, 256), 60.0, dtype=np.float32)
    spots = [(y, x) for y in (40, 100, 160, 220) for x in (40, 100, 160, 220)]
    for f in range(2):
        for j, (y, x) in enumerate(spots):
            _add_spot(red[f], y, x, 500 + 170 * j)
        # Genuine green-only focus, away from red foci.
        _add_spot(green_true[f], 70, 190, 800)
        # Genuine green focus colocalised with a red focus (outlier).
        _add_spot(green_true[f], 40, 40, 600)
    red += rng.normal(0, 4, red.shape).astype(np.float32)
    green = green_true + k_true * red
    green += rng.normal(0, 3, green.shape).astype(np.float32)
    return red, green


def test_estimator_recovers_bleed_coefficient():
    red, green = _synthetic_pair(k_true=0.08)
    est = estimate_crosstalk_k(red, green)
    assert est.mode == "auto"
    assert est.n_sites >= 20
    assert est.k == pytest.approx(0.08, abs=0.02)


def test_estimator_insufficient_sites_is_safe_noop():
    rng = np.random.default_rng(1)
    flat = rng.normal(100, 3, (2, 128, 128)).astype(np.float32)
    est = estimate_crosstalk_k(flat, flat + 5)
    assert est.mode == "insufficient-sites"
    assert est.k == 0.0
    corrected = subtract_crosstalk(flat + 5, flat, est.k)
    np.testing.assert_allclose(corrected, flat + 5)


def test_subtraction_removes_ghosts_and_keeps_real_foci():
    red, green = _synthetic_pair(k_true=0.08)
    est = estimate_crosstalk_k(red, green)
    corrected = subtract_crosstalk(green, red, est.k)

    def amp(img, y, x):
        peak = img[y - 1:y + 2, x - 1:x + 2].max()
        ring = img[y - 8:y + 9, x - 8:x + 9].astype(float).copy()
        ring[6:11, 6:11] = np.nan
        return float(peak) - float(np.nanmedian(ring))

    # Pure-ghost site (red-only focus): residual green amplitude should
    # collapse to noise scale. Red amp here is ~2000 → ghost was ~160.
    ghost_before = amp(green[0], 220, 220)
    ghost_after = amp(corrected[0], 220, 220)
    assert ghost_before > 100
    assert ghost_after < 30
    # Genuine green-only focus survives essentially unchanged.
    real_before = amp(green[0], 70, 190)
    real_after = amp(corrected[0], 70, 190)
    assert real_after == pytest.approx(real_before, rel=0.05)
    # Colocalised genuine focus keeps its true (bleed-stripped) amplitude.
    coloc_after = amp(corrected[0], 40, 40)
    assert coloc_after == pytest.approx(600, rel=0.25)


def _write_hyperstack(path: Path, data: np.ndarray) -> None:
    tifffile.imwrite(
        str(path), data, imagej=True,
        metadata={
            "axes": "ZCYX",
            "channels": data.shape[1],
            "slices": data.shape[0],
            "hyperstack": True,
            "mode": "composite",
        },
    )


def _stage_ctx(tmp_path: Path, opts) -> SimpleNamespace:
    return SimpleNamespace(
        czi_entries=[],  # bulk-style context → layout filter passes all
        do_fluorescent_normalisation=True,
        classify_dir=tmp_path / "03_classify",
        segment_dir=tmp_path / "02_segment",
        fluorescent_normalisation_dir=tmp_path / "04b_fluorescent_normalisation",
        focus_opts=SimpleNamespace(filename_suffix=""),
        fluorescent_normalisation_opts=opts,
        phase_channel=0,
        channel_labels=["Phase", "mCher", "EGFP"],
        reuse_existing=False,
    )


def _classify_stack() -> np.ndarray:
    rng = np.random.default_rng(2)
    z, c, h, w = 2, 4, 96, 96
    data = np.zeros((z, c, h, w), dtype=np.float32)
    data[:, 0] = 500.0                                   # phase
    data[:, 1] = 100.0 + rng.normal(0, 2, (z, h, w))     # red
    for f in range(z):
        _add_spot(data[f, 1], 48, 48, 2000)
    data[:, 2] = 60.0 + 0.1 * data[:, 1]                 # green = bleed only
    data[:, 3] = 1.0                                     # mask
    return data


def test_fluor_norm_stage_applies_fixed_k_and_writes_sidecars(tmp_path: Path):
    from mycomorph.gui.pipeline.context import FluorescentNormalisationOpts
    from mycomorph.gui.pipeline.stages import FluorescentNormalisationStage

    data = _classify_stack()
    opts = FluorescentNormalisationOpts(
        method="none",
        crosstalk_enabled=True,
        crosstalk_source_channel=1,
        crosstalk_target_channel=2,
        crosstalk_k=0.1,
    )
    ctx = _stage_ctx(tmp_path, opts)
    ctx.classify_dir.mkdir(parents=True)
    _write_hyperstack(ctx.classify_dir / "well.tif", data)

    messages: list[str] = []
    stage = FluorescentNormalisationStage()
    assert stage.validate(ctx) == []
    outputs = stage.run(ctx, lambda f, m: messages.append(m))
    assert len(outputs) == 1
    out = outputs[0]

    from mycomorph.core.label_cells import load_hyperstack
    got, _ = load_hyperstack(out)
    expected_green = np.clip(data[:, 2] - 0.1 * data[:, 1], 0, None)
    np.testing.assert_allclose(got[:, 2], expected_green, atol=1e-3)
    # Red and mask channels pass through unchanged.
    np.testing.assert_allclose(got[:, 1], data[:, 1], atol=1e-3)
    np.testing.assert_allclose(got[:, 3], data[:, 3], atol=1e-3)

    sidecar = out.with_name(out.stem + "__crosstalk.json")
    payload = json.loads(sidecar.read_text())
    assert payload["mode"] == "fixed"
    assert payload["k"] == pytest.approx(0.1)
    assert payload["source_channel"] == 1
    assert payload["target_channel"] == 2
    assert artifact_signature_path(out).exists()


def test_fluor_norm_stage_reuse_honours_crosstalk_options(tmp_path: Path):
    from mycomorph.gui.pipeline.context import FluorescentNormalisationOpts
    from mycomorph.gui.pipeline.stages import FluorescentNormalisationStage

    data = _classify_stack()
    opts = FluorescentNormalisationOpts(
        method="none",
        crosstalk_enabled=True,
        crosstalk_source_channel=1,
        crosstalk_target_channel=2,
        crosstalk_k=0.1,
    )
    ctx = _stage_ctx(tmp_path, opts)
    ctx.classify_dir.mkdir(parents=True)
    _write_hyperstack(ctx.classify_dir / "well.tif", data)

    stage = FluorescentNormalisationStage()
    stage.run(ctx, lambda f, m: None)

    # Same options + reuse → skip.
    ctx.reuse_existing = True
    messages: list[str] = []
    stage.run(ctx, lambda f, m: messages.append(m))
    assert any("Reused" in m for m in messages)

    # Changed k + reuse → rebuild with the new coefficient.
    ctx.fluorescent_normalisation_opts = FluorescentNormalisationOpts(
        method="none",
        crosstalk_enabled=True,
        crosstalk_source_channel=1,
        crosstalk_target_channel=2,
        crosstalk_k=0.2,
    )
    messages.clear()
    outputs = stage.run(ctx, lambda f, m: messages.append(m))
    assert not any("Reused" in m for m in messages)
    from mycomorph.core.label_cells import load_hyperstack
    got, _ = load_hyperstack(outputs[0])
    expected_green = np.clip(data[:, 2] - 0.2 * data[:, 1], 0, None)
    np.testing.assert_allclose(got[:, 2], expected_green, atol=1e-3)


def test_foci_detection_signature_survives_parallel_path(
    tmp_path: Path, monkeypatch,
):
    """Foci Detection writes a provenance signature and honours it on
    reuse. Guards the merge of signature-checked reuse into the
    process-pool detection path (single worker keeps the test fast)."""
    from mycomorph.gui.pipeline.context import (
        FluorescentNormalisationOpts, FociDetectionOpts,
    )
    from mycomorph.gui.pipeline.stages import FociDetectionStage

    monkeypatch.setenv("MYCOMORPH_FOCI_WORKERS", "1")

    rng = np.random.default_rng(3)
    z, c, h, w = 1, 4, 64, 64
    data = np.zeros((z, c, h, w), dtype=np.float32)
    data[:, 1] = rng.normal(10, 1, (z, h, w))
    data[:, 2] = rng.normal(10, 1, (z, h, w))
    _add_spot(data[0, 2], 32, 32, 400)
    data[:, 3] = 1.0  # single cell covering the frame

    ctx = _stage_ctx(tmp_path, FluorescentNormalisationOpts(method="none"))
    ctx.do_foci_detection = True
    ctx.foci_detection_dir = tmp_path / "04c_foci_detection"
    ctx.foci_detection_opts = FociDetectionOpts(detector_keys=["hmax"])
    ctx.fluorescent_normalisation_dir.mkdir(parents=True)
    _write_hyperstack(ctx.fluorescent_normalisation_dir / "well.tif", data)

    stage = FociDetectionStage()
    outputs = stage.run(ctx, lambda f, m: None)
    assert len(outputs) == 1
    assert artifact_signature_path(outputs[0]).exists()

    # Same options + reuse → skip.
    ctx.reuse_existing = True
    messages: list[str] = []
    stage.run(ctx, lambda f, m: messages.append(m))
    assert any("Reused" in m for m in messages)

    # Changed detector opts + reuse → rebuild.
    ctx.foci_detection_opts = FociDetectionOpts(
        detector_keys=["hmax"],
        detector_opts=DetectorOpts(hmax_h_mad=6.0),
    )
    messages.clear()
    stage.run(ctx, lambda f, m: messages.append(m))
    assert not any("Reused" in m for m in messages)


def test_fluor_norm_validate_rejects_bad_crosstalk_config(tmp_path: Path):
    from mycomorph.gui.pipeline.context import FluorescentNormalisationOpts
    from mycomorph.gui.pipeline.stages import FluorescentNormalisationStage

    opts = FluorescentNormalisationOpts(
        method="none",
        crosstalk_enabled=True,
        crosstalk_source_channel=1,
        crosstalk_target_channel=1,
    )
    ctx = _stage_ctx(tmp_path, opts)
    ctx.classify_dir.mkdir(parents=True)
    issues = FluorescentNormalisationStage().validate(ctx)
    assert issues and "must differ" in issues[0]
