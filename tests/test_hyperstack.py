"""Pipeline hyperstack I/O: ZCYX round-trip and mask-channel labelling."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import tifffile

from mycomorph.core.extract import api as extract_api
from mycomorph.core.hyperstack import get_labeled_mask_from_fov, load_hyperstack


def _write_hyperstack(path: Path, data: np.ndarray) -> Path:
    # Same ImageJ metadata the pipeline writes (cellpose_pipeline.save_hyperstack).
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
    return path


@pytest.mark.parametrize("n_fov", [3, 1])
def test_load_hyperstack_round_trips_zcyx(tmp_path: Path, n_fov: int):
    # n_fov=1 exercises the (C, Y, X) → (1, C, Y, X) reshape.
    rng = np.random.default_rng(0)
    data = rng.random((n_fov, 4, 24, 32)).astype(np.float32)
    path = _write_hyperstack(tmp_path / "well.tif", data)

    got, meta = load_hyperstack(path)

    assert got.shape == (n_fov, 4, 24, 32)
    np.testing.assert_array_equal(got, data)
    assert int(meta["channels"]) == 4


def test_load_hyperstack_single_plane_becomes_one_fov_one_channel(tmp_path: Path):
    data = np.arange(24 * 32, dtype=np.uint16).reshape(24, 32)
    path = tmp_path / "plane.tif"
    tifffile.imwrite(str(path), data)

    got, meta = load_hyperstack(path)

    assert got.shape == (1, 1, 24, 32)
    np.testing.assert_array_equal(got[0, 0], data)
    assert meta == {}


def _fov_with_mask(tmp_path: Path, mask: np.ndarray) -> np.ndarray:
    phase = np.full(mask.shape, 0.5, dtype=np.float32)
    fov_stack = np.stack([phase, phase * 2, mask])[np.newaxis]  # (1, C=3, Y, X)
    return load_hyperstack(_write_hyperstack(tmp_path / "well.tif", fov_stack))[0][0]


def test_get_labeled_mask_from_fov_labels_binary_segment_mask(tmp_path: Path):
    # Segment writes a binary 0/255 mask: cells come from connected components.
    mask = np.zeros((20, 20), dtype=np.float32)
    mask[2:6, 2:8] = 255.0      # cell A
    mask[10:15, 10:14] = 255.0  # cell B
    mask[17, 17] = 255.0        # C and D touch only diagonally, so with
    mask[18, 18] = 255.0        # 4-connectivity they stay separate cells
    fov = _fov_with_mask(tmp_path, mask)

    labeled, max_label = get_labeled_mask_from_fov(fov, mask_channel=2)

    assert max_label == 4
    assert labeled.shape == (20, 20)
    assert set(np.unique(labeled)) == {0, 1, 2, 3, 4}
    np.testing.assert_array_equal(labeled > 0, mask > 0)
    assert labeled[17, 17] != labeled[18, 18]
    assert len(np.unique(labeled[2:6, 2:8])) == 1
    np.testing.assert_array_equal(labeled, extract_api._label_mask(fov[2]))


def test_get_labeled_mask_from_fov_keeps_classify_labels(tmp_path: Path):
    # Classify writes its own cell labels: gaps where cells were rejected,
    # numbering that doesn't follow raster order, and touching cells.
    mask = np.zeros((20, 20), dtype=np.float32)
    mask[2:6, 2:8] = 14.0
    mask[10:15, 4:9] = 3.0
    mask[10:15, 9:14] = 8.0     # shares an edge with cell 3
    fov = _fov_with_mask(tmp_path, mask)

    labeled, max_label = get_labeled_mask_from_fov(fov, mask_channel=2)

    np.testing.assert_array_equal(labeled, mask.astype(np.int32))
    assert max_label == 14
    assert labeled[12, 8] == 3 and labeled[12, 9] == 8
    # Same labels the Features stage gives the cells, so foci cell_id
    # joins straight onto the features table.
    np.testing.assert_array_equal(labeled, extract_api._label_mask(fov[2]))

