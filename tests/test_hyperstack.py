"""Pipeline hyperstack I/O: ZCYX round-trip and mask-channel labelling."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import tifffile

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


def test_get_labeled_mask_from_fov_labels_mask_channel(tmp_path: Path):
    mask = np.zeros((20, 20), dtype=np.float32)
    mask[2:6, 2:8] = 1.0        # cell A
    mask[10:15, 10:14] = 7.0    # cell B: any non-zero value counts as foreground
    mask[17, 17] = 1.0          # C and D touch only diagonally, so with
    mask[18, 18] = 1.0          # 4-connectivity they stay separate cells
    phase = np.full((20, 20), 0.5, dtype=np.float32)
    fov_stack = np.stack([phase, phase * 2, mask])[np.newaxis]  # (1, C=3, Y, X)
    fov = load_hyperstack(_write_hyperstack(tmp_path / "well.tif", fov_stack))[0][0]

    labeled, n = get_labeled_mask_from_fov(fov, mask_channel=2)

    assert n == 4
    assert labeled.shape == (20, 20)
    assert set(np.unique(labeled)) == {0, 1, 2, 3, 4}
    np.testing.assert_array_equal(labeled > 0, mask > 0)
    assert labeled[17, 17] != labeled[18, 18]
    assert len(np.unique(labeled[2:6, 2:8])) == 1


def test_label_cells_reexports_hyperstack_helpers():
    # Kept for old importers until Phase E retires label_cells.
    from mycomorph.core import hyperstack, label_cells

    assert label_cells.load_hyperstack is hyperstack.load_hyperstack
    assert label_cells.get_labeled_mask_from_fov is hyperstack.get_labeled_mask_from_fov
