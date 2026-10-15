"""Smoke tests for HDF5CropsReader against a synthetic all_crops.h5."""

from __future__ import annotations

import numpy as np

from modelprep.core.crops_h5 import HDF5CropsReader


def test_reader_opens_and_reports_summary(synthetic_h5):
    with HDF5CropsReader(synthetic_h5) as r:
        assert len(r) == 32
        assert r.n_channels == 3
        assert r.channel_names == ["Phase", "Fluor", "Mask"]
        assert r.phase_channel_idx == 0
        assert r.mask_channel_idx == 2
        s = r.summary()
        assert "32" in s
        assert "3 channels" in s


def test_get_returns_classifier_ready_record(synthetic_h5):
    with HDF5CropsReader(synthetic_h5) as r:
        rec = r.get(0)
        assert rec.cell_uid == "cell_0000"
        # (phase, mask, area) at 64×64
        assert rec.crop.shape == (3, 64, 64)
        assert rec.crop.dtype == np.float32
        # Phase channel is in [0, 1]
        assert 0.0 <= rec.crop[0].min() and rec.crop[0].max() <= 1.0
        # Mask channel is strictly binarised
        unique_mask = np.unique(rec.crop[1])
        assert set(unique_mask.tolist()).issubset({0.0, 1.0})
        # Area channel is a constant plane
        area_plane = rec.crop[2]
        assert np.isclose(area_plane.min(), area_plane.max())
        # Full crop is the raw (3, 128, 128)
        assert rec.full_crop.shape == (3, 128, 128)
        assert rec.area_px > 0


def test_batch_read_matches_single(synthetic_h5):
    with HDF5CropsReader(synthetic_h5) as r:
        indices = [0, 5, 10, 17]
        batch = r.get_classifier_batch(indices)
        assert batch.shape == (4, 3, 64, 64)
        for i, idx in enumerate(indices):
            single = r.get(idx).crop
            np.testing.assert_allclose(batch[i], single, rtol=1e-5, atol=1e-6)


def test_cell_uids_property(synthetic_h5):
    with HDF5CropsReader(synthetic_h5) as r:
        uids = r.cell_uids
        assert uids.shape == (32,)
        assert str(uids[0]) == "cell_0000"
        assert str(uids[-1]) == "cell_0031"


def test_fov_groups_prefer_global_fov_ids(synthetic_h5):
    """fov_indices restart per TIFF, so diversity sampling groups by the global
    fov_ids when present, else by (tiff_file, fov_index) (review L2)."""
    import h5py

    n = 32
    tiffs = ["a.tif"] * 16 + ["b.tif"] * 16
    with h5py.File(synthetic_h5, "a") as f:
        del f["tiff_files"]
        f.create_dataset("tiff_files", data=tiffs, dtype=h5py.string_dtype())
    with HDF5CropsReader(synthetic_h5) as r:
        assert set(r.fov_indices.tolist()) == {0}          # both TIFFs: FOV 0
        g = r.fov_groups
        assert len(set(g[:16].tolist())) == 1 and len(set(g[16:].tolist())) == 1
        assert g[0] != g[16]                                # …but different fields

    with h5py.File(synthetic_h5, "a") as f:
        f.create_dataset("fov_ids", data=np.arange(n, dtype=np.int32) // 4)
    with HDF5CropsReader(synthetic_h5) as r:
        assert r.fov_groups.tolist() == (np.arange(n) // 4).tolist()
