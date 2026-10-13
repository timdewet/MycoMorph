"""TIFF inputs for Single-file / Bulk mode: reading, focus pooling, batch layout."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import tifffile

from mycomorph.core.focus import io_source, io_tiff
from mycomorph.core.focus.pipeline import Options, process_sample
from mycomorph.gui.pipeline.bulk_layout import BulkLayout, group_entries


def _stack(shape, seed=0) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 4000, size=shape, dtype=np.uint16)


def _write_imagej(path: Path, arr: np.ndarray, axes: str, **metadata) -> Path:
    tifffile.imwrite(str(path), arr, imagej=True, metadata={"axes": axes, **metadata})
    return path


# ─────────────────────────────────────────────────────────────── reading


def test_imagej_zcyx_is_one_scene_of_focal_planes(tmp_path: Path):
    arr = _stack((5, 3, 32, 40))
    path = _write_imagej(tmp_path / "zstack.tif", arr, "ZCYX")

    assert io_tiff.list_scene_indices(path) == [0]
    scene = io_tiff.read_scene(path, 0)
    np.testing.assert_array_equal(scene.array_zcyx, arr)
    assert scene.channel_names == ["C0", "C1", "C2"]


def test_imagej_frames_become_separate_scenes(tmp_path: Path):
    arr = _stack((2, 4, 3, 32, 40))
    path = _write_imagej(tmp_path / "frames.tif", arr, "TZCYX")

    assert io_tiff.list_scene_indices(path) == [0, 1]
    np.testing.assert_array_equal(io_tiff.read_scene(path, 1).array_zcyx, arr[1])
    with pytest.raises(IndexError):
        io_tiff.read_scene(path, 2)


def test_imagej_channel_labels_are_used(tmp_path: Path):
    arr = _stack((3, 32, 40))
    path = _write_imagej(tmp_path / "labelled.tif", arr, "CYX", Labels=["Phase", "GFP", "RFP"])

    scene = io_tiff.read_scene(path, 0)
    assert scene.array_zcyx.shape == (1, 3, 32, 40)
    assert scene.channel_names == ["Phase", "GFP", "RFP"]
    assert io_source.channel_names(path) == ["Phase", "GFP", "RFP"]


def test_single_plane_tiff(tmp_path: Path):
    arr = _stack((32, 40))
    path = tmp_path / "plane.tif"
    tifffile.imwrite(str(path), arr)

    scene = io_tiff.read_scene(path, 0)
    assert scene.array_zcyx.shape == (1, 1, 32, 40)
    np.testing.assert_array_equal(scene.array_zcyx[0, 0], arr)


def test_unlabelled_pages_are_channels(tmp_path: Path):
    arr = _stack((4, 32, 40))
    path = tmp_path / "pages.tif"
    tifffile.imwrite(str(path), arr, photometric="minisblack")
    with tifffile.TiffFile(str(path)) as tif:
        assert tif.series[0].axes[0] in "IQ"

    scene = io_tiff.read_scene(path, 0)
    assert scene.array_zcyx.shape == (1, 4, 32, 40)
    np.testing.assert_array_equal(scene.array_zcyx[0], arr)


def test_ome_axes_are_reordered_to_zcyx(tmp_path: Path):
    arr = _stack((3, 5, 32, 40))  # C, Z, Y, X
    path = tmp_path / "czyx.ome.tif"
    tifffile.imwrite(str(path), arr, metadata={"axes": "CZYX"})

    scene = io_tiff.read_scene(path, 0)
    np.testing.assert_array_equal(scene.array_zcyx, arr.transpose(1, 0, 2, 3))


def test_ome_positions_channels_and_pixel_size(tmp_path: Path):
    path = tmp_path / "positions.ome.tif"
    fovs = [_stack((2, 32, 40), seed=i) for i in range(3)]
    with tifffile.TiffWriter(str(path), ome=True) as w:
        for fov in fovs:
            w.write(fov, metadata={
                "axes": "CYX",
                "Channel": {"Name": ["Phase", "mCherry"]},
                "PhysicalSizeX": 0.065, "PhysicalSizeY": 0.065,
            })

    assert io_source.list_scene_indices(path) == [0, 1, 2]
    scene = io_source.read_scene(path, 2)
    np.testing.assert_array_equal(scene.array_zcyx[0], fovs[2])
    assert scene.channel_names == ["Phase", "mCherry"]
    assert scene.pixel_size_um[0] == pytest.approx(0.065)
    assert io_source.pixels_per_um(path) == pytest.approx(1 / 0.065)


def test_tiffs_have_no_well_metadata(tmp_path: Path):
    path = _write_imagej(tmp_path / "x.tif", _stack((2, 16, 16)), "CYX")
    assert io_source.is_tiff(path)
    assert io_source.list_scene_wells(path) == {}
    assert not io_source.is_tiff(tmp_path / "x.czi")


# ──────────────────────────────────────────────────────────── focus pooling


def _focus_stack(n_fov: int, sharp_z: int, seed: int) -> np.ndarray:
    """(T, Z, C, Y, X): textured phase at ``sharp_z``, flat elsewhere."""
    rng = np.random.default_rng(seed)
    arr = np.full((n_fov, 5, 2, 48, 48), 1000, dtype=np.uint16)
    for t in range(n_fov):
        arr[t, sharp_z, 0] = rng.integers(0, 4000, size=(48, 48))
        arr[t, :, 1] = rng.integers(0, 500, size=(5, 48, 48))
    return arr


def test_process_sample_pools_fovs_across_tiffs(tmp_path: Path):
    a = _focus_stack(2, sharp_z=3, seed=1)
    b = _focus_stack(1, sharp_z=1, seed=2)
    pa = _write_imagej(tmp_path / "a.tif", a, "TZCYX", Labels=["Phase", "GFP"] * 10)
    pb = _write_imagej(tmp_path / "b.tif", b[0], "ZCYX", Labels=["Phase", "GFP"] * 5)
    out_dir = tmp_path / "focus"

    result = process_sample(
        [pa, pb], "WT__GFP__DMSO",
        out_dir=out_dir,
        opts=Options(phase_channel=0, focus_mode="whole"),
        filename_suffix="_focused",
    )

    out = out_dir / "WT__GFP__DMSO_focused.tif"
    assert result.well_results[0].output_path == out
    assert result.well_results[0].chosen_z == [3, 3, 1]
    written = tifffile.imread(str(out))
    assert written.shape == (3, 2, 48, 48)
    np.testing.assert_array_equal(written[0], a[0, 3])
    np.testing.assert_array_equal(written[2], b[0, 1])

    sidecar = json.loads((out_dir / "WT__GFP__DMSO_focused__acquisition.json").read_text())
    assert sidecar["fov_source_files"] == ["a.tif", "a.tif", "b.tif"]
    assert sidecar["source_czi"] == "a.tif; b.tif"
    assert sidecar["scene_indices"] == [0, 1, 0]


def test_process_sample_single_file_sidecar_unchanged(tmp_path: Path):
    a = _focus_stack(2, sharp_z=2, seed=3)
    pa = _write_imagej(tmp_path / "a.tif", a, "TZCYX")

    process_sample([pa], "S", out_dir=tmp_path / "out",
                   opts=Options(phase_channel=0, focus_mode="edf"))

    sidecar = json.loads((tmp_path / "out" / "S__acquisition.json").read_text())
    assert sidecar["source_czi"] == "a.tif"
    assert "fov_source_files" not in sidecar


def test_process_sample_rejects_mismatched_image_sizes(tmp_path: Path):
    pa = _write_imagej(tmp_path / "a.tif", _stack((2, 32, 40)), "CYX")
    pb = _write_imagej(tmp_path / "b.tif", _stack((2, 32, 48)), "CYX")

    with pytest.raises(ValueError, match="same channels and image size"):
        process_sample([pa, pb], "S", out_dir=tmp_path / "out",
                       opts=Options(phase_channel=0))


# ───────────────────────────────────────────────────────────── batch layout


def _row(path: Path, condition: str, replica: str = "") -> dict:
    return {"czi_path": str(path), "condition": condition, "reporter": "GFP",
            "mutant_or_drug": "", "replica": replica, "notes": ""}


def test_add_folder_picks_up_czi_and_tiff(tmp_path: Path):
    for name in ("a.czi", "b.tif", "sub/c.TIFF", "d.png", ".hidden.tif"):
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_bytes(b"x")

    layout = BulkLayout.empty()
    assert layout.add_folder(tmp_path) == 3
    assert sorted(Path(p).name for p in layout.df["czi_path"]) == ["a.czi", "b.tif", "c.TIFF"]


def test_rows_with_same_labels_pool_into_one_sample(tmp_path: Path):
    files = []
    for name in ("f1.tif", "f2.tif", "f3.tif"):
        (tmp_path / name).write_bytes(b"x")
        files.append(tmp_path / name)
    import pandas as pd
    layout = BulkLayout(df=pd.DataFrame([
        _row(files[0], "WT"), _row(files[1], "MUT"), _row(files[2], "WT"),
    ]))

    assert layout.validate() == []
    samples = layout.samples()
    assert [len(paths) for _, paths in samples] == [2, 1]
    assert samples[0][1] == [files[0], files[2]]
    assert samples == group_entries(layout.active_rows().to_dict(orient="records"))


def test_validate_flags_same_file_twice_in_a_sample(tmp_path: Path):
    f = tmp_path / "f.tif"
    f.write_bytes(b"x")
    import pandas as pd
    layout = BulkLayout(df=pd.DataFrame([_row(f, "WT"), _row(f, "WT")]))

    issues = layout.validate()
    assert any("listed twice" in i for i in issues)


def test_validate_flags_unsupported_file_type(tmp_path: Path):
    f = tmp_path / "f.png"
    f.write_bytes(b"x")
    import pandas as pd
    layout = BulkLayout(df=pd.DataFrame([_row(f, "WT")]))

    assert any("Unsupported file type" in i for i in layout.validate())
