"""Foci Detection labels cells exactly as Features does.

The 04c ``cell_id`` must join straight onto the 04_features ``cell_id``,
and two Classify cells that touch must stay two cells. Before this was
fixed the stage re-ran connected components on the mask, which renumbered
every cell and merged touching ones.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import tifffile
from skimage.measure import label as sk_label

from mycomorph.core.extract import api as extract_api
from mycomorph.core.extract import per_cell
from mycomorph.core.foci import DetectorOpts
from mycomorph.core.foci.batch import detect_unit
from mycomorph.core.foci.detectors import REGISTRY
from mycomorph.core.hyperstack import label_mask_channel, load_hyperstack

PER_CELL_COLS = [
    "cell_p90", "cell_p95", "cell_median", "cell_std",
    "prominence_p90", "prominence_median",
]

# Classify output: gaps where cells were rejected, numbering that doesn't
# follow raster order, and cells 9 and 12 sharing an edge.
CELLS = {
    20: (slice(6, 20), slice(6, 30)),
    9: (slice(30, 44), slice(10, 40)),
    12: (slice(30, 44), slice(40, 70)),
    5: (slice(48, 60), slice(70, 90)),
}
TOUCHING = {9, 12}
CELL_BG = {20: 30.0, 9: 10.0, 12: 80.0, 5: 30.0}
FOCI = {20: (13.0, 18.0), 9: (37.0, 25.0), 12: (37.0, 55.0), 5: (54.0, 80.0)}
SHAPE = (64, 96)


def _classify_mask() -> np.ndarray:
    mask = np.zeros(SHAPE, dtype=np.float32)
    for lbl, sl in CELLS.items():
        mask[sl] = lbl
    return mask


def _gauss(y: float, x: float, amp: float, sigma: float = 1.4) -> np.ndarray:
    yy, xx = np.mgrid[0:SHAPE[0], 0:SHAPE[1]].astype(np.float64)
    return amp * np.exp(-((yy - y) ** 2 + (xx - x) ** 2) / (2 * sigma ** 2))


def _fluor(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = np.full(SHAPE, 100.0)
    for lbl, sl in CELLS.items():
        img[sl] += CELL_BG[lbl]
    for y, x in FOCI.values():
        img += _gauss(y, x, 300.0)
    img += rng.normal(0, 2.0, SHAPE)
    return img.astype(np.float32)


def _nearest(df: pd.DataFrame, y: float, x: float) -> pd.Series:
    d = np.hypot(df["y"] - y, df["x"] - x)
    assert d.min() < 1.5, f"no detection near ({y}, {x})"
    return df.loc[d.idxmin()]


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


def test_foci_stage_cell_ids_join_features(tmp_path: Path, monkeypatch):
    """Classify → Fluorescent Normalisation → Foci Detection: every focus
    carries the Features cell_id of the cell it sits in."""
    from mycomorph.gui.pipeline.context import (
        FluorescentNormalisationOpts, FociDetectionOpts,
    )
    from mycomorph.gui.pipeline.stages import (
        FluorescentNormalisationStage, FociDetectionStage,
    )

    monkeypatch.setenv("MYCOMORPH_FOCI_WORKERS", "1")
    classify = np.stack([
        np.full(SHAPE, 500.0, dtype=np.float32),   # phase
        _fluor(),
        _classify_mask(),
    ])[np.newaxis]                                 # (1 FOV, C=3, Y, X)

    ctx = SimpleNamespace(
        czi_entries=[],  # bulk-style context → layout filter passes all
        do_fluorescent_normalisation=True,
        do_foci_detection=True,
        do_split=False,
        do_focus=False,
        split_dir=tmp_path / "01_split",
        focus_dir=tmp_path / "01_focus",
        segment_dir=tmp_path / "02_segment",
        classify_dir=tmp_path / "03_classify",
        fluorescent_normalisation_dir=tmp_path / "04b_fluorescent_normalisation",
        foci_detection_dir=tmp_path / "04c_foci_detection",
        focus_opts=SimpleNamespace(filename_suffix=""),
        fluorescent_normalisation_opts=FluorescentNormalisationOpts(method="none"),
        foci_detection_opts=FociDetectionOpts(),   # default detector
        phase_channel=0,
        channel_labels=["Phase", "GFP"],
        reuse_existing=False,
    )
    ctx.classify_dir.mkdir(parents=True)
    _write_hyperstack(ctx.classify_dir / "well.tif", classify)

    FluorescentNormalisationStage().run(ctx, lambda f, m: None)
    (parquet,) = FociDetectionStage().run(ctx, lambda f, m: None)
    foci = pd.read_parquet(parquet)

    # Features as the Features stage computes them on the Classify TIFF.
    fov = load_hyperstack(ctx.classify_dir / "well.tif")[0][0]
    features = per_cell.extract_fov_features(
        image_channels=fov[:-1],
        labeled_mask=extract_api._label_mask(fov[-1]),
        channel_names=["Phase", "GFP"],
        pixels_per_um=1.0,
        run_id="r", well="well", fov_index=0, source_czi="",
        plate_acquisition_datetime=None, fov_acquisition_time=None,
        midline_features=False,
    )
    assert sorted(features["cell_id"]) == sorted(CELLS)

    for lbl, (y, x) in FOCI.items():
        row = _nearest(foci, y, x)
        assert row["cell_id"] == lbl
        assert row["cell_label"] == lbl

    in_cell = foci[foci["cell_id"] > 0]
    joined = in_cell.merge(features, on="cell_id", how="left", validate="m:1")
    assert joined["area_px"].notna().all()
    inside_bbox = (
        (joined["bbox_y0"] <= joined["y"].round())
        & (joined["y"].round() < joined["bbox_y1"])
        & (joined["bbox_x0"] <= joined["x"].round())
        & (joined["x"].round() < joined["bbox_x1"])
    )
    assert inside_bbox.all()


@pytest.mark.parametrize(
    "det_key", [k for k in REGISTRY if k != "bacteroidal"],
)
def test_classify_labels_move_no_detections(det_key: str):
    """Swapping connected components for Classify labels changes only ids
    and the per-cell statistics of touching cells — never where foci are
    found or their focus-level features. (``bacteroidal`` is the exception:
    see the next test.)"""
    img = _fluor()
    classify = _classify_mask()
    old_labels = sk_label(classify > 0, connectivity=1).astype(np.int32)
    new_labels = label_mask_channel(classify)
    opts = DetectorOpts(split_merged=True)

    try:
        df_old = detect_unit(img, old_labels, det_key, opts)
    except ImportError as exc:
        pytest.skip(f"{det_key}: {exc}")
    df_new = detect_unit(img, new_labels, det_key, opts)

    assert len(df_new) == len(df_old)
    assert TOUCHING <= set(df_new["cell_id"])
    unchanged = [
        c for c in df_new.columns
        if c not in {"cell_label", "cell_id", *PER_CELL_COLS}
    ]
    pd.testing.assert_frame_equal(df_old[unchanged], df_new[unchanged])

    yi = df_new["y"].round().astype(int).clip(0, SHAPE[0] - 1)
    xi = df_new["x"].round().astype(int).clip(0, SHAPE[1] - 1)
    np.testing.assert_array_equal(
        df_new["cell_id"].to_numpy(), classify[yi, xi].astype(int),
    )

    isolated = ~df_new["cell_id"].isin(TOUCHING)
    pd.testing.assert_frame_equal(
        df_old.loc[isolated, PER_CELL_COLS], df_new.loc[isolated, PER_CELL_COLS],
    )
    # The merged pair pooled a dim and a bright cell; each now gets its own.
    touching = df_new["cell_id"].isin(TOUCHING).to_numpy()
    assert not np.allclose(
        df_old.loc[touching, "cell_median"], df_new.loc[touching, "cell_median"],
    )
    for lbl in TOUCHING:
        here = (df_new["cell_id"] == lbl).to_numpy()
        assert df_new.loc[here, "cell_median"].iloc[0] == pytest.approx(
            np.median(img[new_labels == lbl]), rel=1e-6,
        )


def test_bacteroidal_background_is_per_classify_cell():
    """``bacteroidal`` thresholds each cell against its own median and std,
    so merging a dim cell with a bright neighbour hid the dim cell's focus.
    With Classify labels each cell has its own background and the focus is
    found — detections from this detector can change, not just ids."""
    rng = np.random.default_rng(1)
    img = np.zeros(SHAPE)
    img[CELLS[9]] += 10.0       # dim cell …
    img[CELLS[12]] += 100.0     # … touching a bright one
    y, x = FOCI[9]
    img += _gauss(y, x, 40.0)
    img += rng.normal(0, 1.0, SHAPE)
    img = img.astype(np.float32)

    classify = np.zeros(SHAPE, dtype=np.float32)
    for lbl in TOUCHING:
        classify[CELLS[lbl]] = lbl
    old_labels = sk_label(classify > 0, connectivity=1).astype(np.int32)
    new_labels = label_mask_channel(classify)
    opts = DetectorOpts()

    df_new = detect_unit(img, new_labels, "bacteroidal", opts)
    row = _nearest(df_new, y, x)
    assert row["cell_id"] == 9

    df_old = detect_unit(img, old_labels, "bacteroidal", opts)
    if not df_old.empty:
        assert np.hypot(df_old["y"] - y, df_old["x"] - x).min() > 1.5
