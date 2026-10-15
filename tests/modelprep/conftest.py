"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def synthetic_h5(tmp_path: Path) -> Path:
    """Create a tiny ``all_crops.h5`` file matching MycoMorph's schema.

    Schema is the minimum the ModelPrep reader expects: crops dataset,
    cell_uids, areas_px, and basic metadata.
    """
    import h5py

    h5_path = tmp_path / "all_crops.h5"
    n = 32
    c = 3
    h = w = 128

    rng = np.random.default_rng(0)
    crops = rng.random((n, c, h, w), dtype=np.float32)
    # Channel 2 is the mask — make it a binary disk in each crop.
    yy, xx = np.mgrid[:h, :w]
    cy, cx = h // 2, w // 2
    disk = ((yy - cy) ** 2 + (xx - cx) ** 2) < 30 ** 2
    crops[:, 2, :, :] = disk.astype(np.float32)

    cell_uids = np.array([f"cell_{i:04d}" for i in range(n)], dtype=object)
    areas_px = np.array([int(disk.sum())] * n, dtype=np.int32)

    with h5py.File(str(h5_path), "w") as f:
        f.create_dataset("crops", data=crops)
        str_dt = h5py.string_dtype()
        f.create_dataset("cell_uids", data=cell_uids.tolist(), dtype=str_dt)
        f.create_dataset("areas_px", data=areas_px)
        f.create_dataset("condition_labels", data=["cond"] * n, dtype=str_dt)
        f.create_dataset("conditions", data=["c0"] * n, dtype=str_dt)
        f.create_dataset("reporters", data=["rpt"] * n, dtype=str_dt)
        f.create_dataset("genes", data=[""] * n, dtype=str_dt)
        f.create_dataset("drugs", data=[""] * n, dtype=str_dt)
        f.create_dataset("tiff_files", data=["fake.tif"] * n, dtype=str_dt)
        f.create_dataset("fov_indices", data=np.zeros(n, dtype=np.int32))
        f.create_dataset("cell_labels", data=np.arange(1, n + 1, dtype=np.int32))
        f.attrs["crop_size"] = 128
        f.attrs["n_channels"] = c
        f.attrs["channel_names"] = ["Phase", "Fluor", "Mask"]
        f.attrs["total_cells"] = n
        f.attrs["n_fovs"] = 1
        f.attrs["schema_version"] = "1.0"
    return h5_path


# Gene / condition layout of ``multi_condition_h5``: 48 cells, 12 per well
# group, each group split 6/6 between an uninduced and an induced condition.
MULTI_GENES = ("dnaN", "inhA", "NT1", "")          # "" = control-like well
MULTI_LABELS = ("dnaN", "inhA", "NT1", "empty")    # condition_labels
MULTI_CONDITIONS = ("ATc+MMC-", "ATc+MMC+")


@pytest.fixture
def multi_condition_h5(tmp_path: Path) -> Path:
    """A session-wide ``all_crops.h5`` with several genes and conditions, laid
    out like MycoMorph writes them (``condition_labels`` = mutant name,
    ``conditions`` = treatment, ``genes`` blank for control-like wells)."""
    import h5py

    h5_path = tmp_path / "multi" / "all_crops.h5"
    h5_path.parent.mkdir()
    n, c, h, w = 48, 3, 32, 32
    rng = np.random.default_rng(1)
    crops = rng.random((n, c, h, w), dtype=np.float32)
    crops[:, 2] = 0.0
    crops[:, 2, 8:24, 8:24] = 1.0

    group = [i % 4 for i in range(n)]
    cond = [MULTI_CONDITIONS[(i // 4) % 2] for i in range(n)]
    genes = [MULTI_GENES[g] for g in group]
    labels = [MULTI_LABELS[g] for g in group]
    tiffs = [f"{cd}__rep__{lb}__R1_focused.tif" for cd, lb in zip(cond, labels)]

    with h5py.File(str(h5_path), "w") as f:
        str_dt = h5py.string_dtype()
        f.create_dataset("crops", data=crops)
        f.create_dataset("cell_uids", data=[f"cell_{i:04d}" for i in range(n)], dtype=str_dt)
        f.create_dataset("areas_px", data=np.full(n, 256, dtype=np.int32))
        f.create_dataset("conditions", data=cond, dtype=str_dt)
        f.create_dataset("condition_labels", data=labels, dtype=str_dt)
        f.create_dataset("genes", data=genes, dtype=str_dt)
        f.create_dataset("reporters", data=["rpt"] * n, dtype=str_dt)
        f.create_dataset("drugs", data=[""] * n, dtype=str_dt)
        f.create_dataset("tiff_files", data=tiffs, dtype=str_dt)
        f.create_dataset("fov_indices", data=np.arange(n, dtype=np.int32) % 3)
        f.create_dataset("cell_labels", data=np.arange(1, n + 1, dtype=np.int32))
        f.attrs["channel_names"] = ["Phase", "Fluor", "Mask"]
        f.attrs["schema_version"] = "1.0"
    return h5_path
