"""Tests for FovContextProvider — locating a cell back in its original FOV.

Builds a synthetic pipeline layout (``01_focus`` / ``02_segment`` beside a
``04_features`` dir) with a known blob, then checks the provider reproduces the
cell's ``area_px``, centres the window on the blob, keeps a constant window size
by zero-padding at edges, and degrades gracefully when originals are missing.
"""

from __future__ import annotations

import numpy as np
import pytest

from modelprep.core.crops_h5 import CropRecord
from modelprep.core.fov_context import FovContextProvider

tifffile = pytest.importorskip("tifffile")

TIFF = "well__reporter__cond__R1_focused.tif"
H = W = 200
NFOV = 2
BRIGHT_FILL = 1000.0


def _record(cell_label: int, area_px: int, fov_index: int = 1) -> CropRecord:
    dummy = np.zeros((3, 8, 8), dtype=np.float32)
    return CropRecord(
        cell_uid=f"uid_{cell_label}",
        crop=dummy,
        full_crop=np.zeros((2, 8, 8), dtype=np.float32),
        area_px=area_px,
        condition_label="cond",
        gene="g",
        drug="",
        reporter="r",
        tiff_file=TIFF,
        fov_index=fov_index,
        cell_label=cell_label,
        source_h5_index=0,
    )


def _build_dataset(tmp_path, blob_center, blob_half=6, with_classify=False):
    """Write focus + segment (and optionally classify) TIFFs; return (h5_path,
    blob_area). The blob lives in FOV 1; FOV 0 is empty."""
    root = tmp_path / "training_data"
    (root / "01_focus").mkdir(parents=True)
    (root / "02_segment").mkdir(parents=True)
    (root / "04_features").mkdir(parents=True)

    # Focus: (NFOV, 2, H, W) — channel 0 brightfield (constant so padding shows
    # as 0), channel 1 reporter.
    focus = np.zeros((NFOV, 2, H, W), dtype=np.uint16)
    focus[:, 0] = int(BRIGHT_FILL)

    # Binary foreground blob in FOV 1.
    cy, cx = blob_center
    binary = np.zeros((NFOV, H, W), dtype=np.uint16)
    binary[1, cy - blob_half:cy + blob_half, cx - blob_half:cx + blob_half] = 255
    blob_area = int((binary[1] > 0).sum())

    segment = np.zeros((NFOV, 3, H, W), dtype=np.uint16)
    segment[:, 0] = focus[:, 0]
    segment[:, 2] = binary  # last channel = mask
    # photometric='minisblack' stores each (fov, channel) as a separate page —
    # matching how MycoMorph writes these stacks (one page per channel, not RGB).
    tifffile.imwrite(root / "01_focus" / TIFF, focus, photometric="minisblack")
    tifffile.imwrite(root / "02_segment" / TIFF, segment, photometric="minisblack")

    if with_classify:
        # Already-labelled mask: blob == label 1 (matches skimage labelling of
        # the single FOV-1 blob), so 03_classify is the preferred exact match.
        classify = np.zeros((NFOV, 3, H, W), dtype=np.uint16)
        classify[:, 0] = focus[:, 0]
        classify[1, 2, cy - blob_half:cy + blob_half, cx - blob_half:cx + blob_half] = 1
        (root / "03_classify").mkdir()
        tifffile.imwrite(root / "03_classify" / TIFF, classify, photometric="minisblack")

    return root / "04_features" / "all_crops.h5", blob_area


def test_available_false_without_originals(tmp_path):
    (tmp_path / "training_data" / "04_features").mkdir(parents=True)
    prov = FovContextProvider(tmp_path / "training_data" / "04_features" / "all_crops.h5")
    assert prov.available is False
    assert prov.get_window(_record(1, 100), 64) is None


def test_locates_blob_and_reproduces_area(tmp_path):
    h5_path, area = _build_dataset(tmp_path, blob_center=(120, 60))
    prov = FovContextProvider(h5_path)
    assert prov.available is True

    ctx = prov.get_window(_record(cell_label=1, area_px=area), window_px=64)
    assert ctx is not None
    assert ctx.image.shape == (64, 64)
    assert ctx.outline.shape == (64, 64)
    assert ctx.outline.any()
    # Exact area_px match locked the binary segment mask as the source.
    assert prov._mask_source is not None
    assert prov._mask_source.name == "02_segment"

    # The blob is centred in the window (outline centroid near the middle).
    ys, xs = np.nonzero(ctx.outline)
    assert abs(ys.mean() - 32) <= 3
    assert abs(xs.mean() - 32) <= 3


def test_prefers_classify_when_present(tmp_path):
    h5_path, area = _build_dataset(tmp_path, blob_center=(120, 60), with_classify=True)
    prov = FovContextProvider(h5_path)
    ctx = prov.get_window(_record(cell_label=1, area_px=area), window_px=64)
    assert ctx is not None
    assert prov._mask_source.name == "03_classify"


def test_edge_cell_zero_pads_to_constant_size(tmp_path):
    # Blob hard against the top-left corner; window must still be 64×64 with the
    # out-of-image region zero-padded.
    h5_path, area = _build_dataset(tmp_path, blob_center=(6, 6), blob_half=5)
    prov = FovContextProvider(h5_path)
    ctx = prov.get_window(_record(cell_label=1, area_px=area), window_px=64)
    assert ctx is not None
    assert ctx.image.shape == (64, 64)
    # Top-left corner is padding (brightfield fill is 1000, so 0 == padded).
    assert ctx.image[0, 0] == 0
    # The cell is still centred, where brightfield is present.
    assert ctx.image[32, 32] == pytest.approx(BRIGHT_FILL)
    assert ctx.outline.any()


def test_missing_label_returns_none(tmp_path):
    h5_path, area = _build_dataset(tmp_path, blob_center=(120, 60))
    prov = FovContextProvider(h5_path)
    # cell_label 999 doesn't exist in the mask → graceful None.
    assert prov.get_window(_record(cell_label=999, area_px=area), 64) is None


def test_online_only_detection():
    """Cloud placeholders (e.g. Dropbox online-only) are recognised from stat
    flags alone, so the inspector can avoid downloading 2 GB TIFFs (review L3)."""
    from types import SimpleNamespace

    from modelprep.core.fov_context import is_online_only

    assert is_online_only(SimpleNamespace(st_flags=0x40000060))      # macOS dataless
    assert not is_online_only(SimpleNamespace(st_flags=0x40))         # local
    assert is_online_only(SimpleNamespace(st_file_attributes=0x400000))  # Windows
    assert not is_online_only(SimpleNamespace())


def test_online_only_bytes_zero_for_local_files(tmp_path):
    h5_path, _ = _build_dataset(tmp_path, blob_center=(120, 60))
    assert FovContextProvider(h5_path).online_only_bytes() == 0
