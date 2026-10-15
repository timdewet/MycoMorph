"""The ``all_crops.h5`` contract with ModelPrep.

ModelPrep (the labeller / trainer for the cell-quality CNN) reads the crops
HDF5 that the Features stage writes and accepts only ``schema_version`` "1.0".
New information is therefore added as extra root attributes, never by bumping
the version: the crop options (so a consumer can warn about background-masked
crops, which the Classify stage never shows the model) and whether Classify had
already filtered the masks the crops were cut from.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import h5py
import numpy as np
import pytest

from mycomorph.core import api
from mycomorph.core import cell_quality_classifier as cqc
from mycomorph.core.extract import crops as crops_mod
from mycomorph.core.extract.api import (
    ExtractOpts,
    _input_provenance,
    consolidate_crops,
    extract_features_tiff,
)


def test_schema_version_is_pinned():
    # ModelPrep's crops_h5 loader rejects anything but "1.0". Changing this
    # needs a coordinated ModelPrep release.
    assert crops_mod.HDF5_SCHEMA_VERSION == "1.0"


def _segmented_tiff(path: Path) -> Path:
    """One FOV, phase + binary mask, two well-separated cells."""
    from mycomorph.core.cellpose_pipeline import save_hyperstack

    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.zeros((1, 2, 64, 64), dtype=np.uint16)
    rng = np.random.default_rng(1)
    data[0, 0] = rng.integers(50, 200, size=(64, 64), dtype=np.uint16)
    data[0, 1, 10:20, 10:24] = 1
    data[0, 1, 40:50, 36:52] = 1
    save_hyperstack(data, path, "WT_GFP_rep1", ["fov_000"], ["Phase", "BinaryMask"], 13.8767)
    return path


def _extract(tiff: Path, out_dir: Path, **crop_kwargs) -> Path:
    opts = ExtractOpts(
        midline_features=False,
        refine_contour=False,
        save_csv=False,
        make_qc_plots=False,
        save_crops=True,
        crop_size=32,
        **crop_kwargs,
    )
    out = out_dir / (tiff.stem + ".parquet")
    extract_features_tiff(tiff, out, opts)
    h5 = out.with_name(out.stem + "__crops.h5")
    assert h5.exists()
    return h5


def _attrs(h5_path: Path) -> dict:
    with h5py.File(h5_path, "r") as f:
        return dict(f.attrs)


@pytest.mark.parametrize("mask_background", [True, False])
def test_crop_options_recorded_per_well_and_consolidated(tmp_path, mask_background):
    tiff = _segmented_tiff(tmp_path / "out" / "02_segment" / "A01.tif")
    well_h5 = _extract(
        tiff, tmp_path / "out" / "04_features",
        mask_background=mask_background, crop_pad=7, normalise_per_crop=False,
    )
    all_h5 = consolidate_crops([well_h5], tmp_path / "out" / "04_features" / "all_crops.h5")

    for path in (well_h5, all_h5):
        a = _attrs(path)
        assert a["schema_version"] == "1.0"
        assert bool(a["mask_background"]) is mask_background
        assert int(a["dilate_px"]) == crops_mod.CropOpts().dilate_px
        assert int(a["pad"]) == 7
        assert int(a["crop_size"]) == 32
        assert bool(a["normalise_per_crop"]) is False
        assert bool(a["include_mask_channel"]) is True
        assert a["input_stage"] == "segment"
        assert a["classifier_prefilter"] == "none"
        assert a["classifier_prefilter_model_sha256"] == ""
        assert int(a["total_cells"]) == 2


def test_cnn_prefilter_recorded_from_classify_provenance(tmp_path, monkeypatch):
    seg = _segmented_tiff(tmp_path / "out" / "02_segment" / "A01.tif")
    model = tmp_path / "models" / "best_model.pth"
    model.parent.mkdir()
    model.write_bytes(b"pretend weights")

    def keep_everything(*, labeled_mask, **_kwargs):
        return labeled_mask, {"total_cells": 2, "kept": 2, "removed_edge": 0,
                              "removed_debris": 0, "removed_cnn": 0}

    monkeypatch.setattr(cqc, "classify_and_filter_mask", keep_everything)
    classified = tmp_path / "out" / "03_classify" / "A01.tif"
    api.classify_filter_tiff(seg, classified, phase_channel=0,
                             opts=api.ClassifyOpts(model_path=model))

    a = _attrs(_extract(classified, tmp_path / "out" / "04_features"))
    assert a["input_stage"] == "classify"
    assert a["classifier_prefilter"] == "cnn"
    assert a["classifier_prefilter_model_sha256"] == hashlib.sha256(
        model.read_bytes()).hexdigest()


def test_rules_only_classify_recorded(tmp_path, monkeypatch):
    seg = _segmented_tiff(tmp_path / "out" / "02_segment" / "A01.tif")
    monkeypatch.setattr(
        cqc, "classify_and_filter_mask",
        lambda *, labeled_mask, **_k: (labeled_mask, {"total_cells": 2, "kept": 2}),
    )
    classified = tmp_path / "out" / "03_classify" / "A01.tif"
    api.classify_filter_tiff(seg, classified, phase_channel=0, opts=api.ClassifyOpts())
    assert _input_provenance(classified) == {
        "input_stage": "classify",
        "classifier_prefilter": "rules",
        "classifier_prefilter_model_sha256": "",
    }


@pytest.mark.parametrize(
    "folder,expected_stage,expected_prefilter",
    [
        ("03_classify", "classify", "unknown"),   # can't tell if a CNN ran
        ("02_segment", "segment", "none"),
        ("my_masks", "unknown", "unknown"),
    ],
)
def test_unsigned_input_falls_back_to_folder_name(
        tmp_path, folder, expected_stage, expected_prefilter):
    tiff = tmp_path / folder / "A01.tif"
    tiff.parent.mkdir()
    tiff.write_bytes(b"x")
    prov = _input_provenance(tiff)
    assert prov["input_stage"] == expected_stage
    assert prov["classifier_prefilter"] == expected_prefilter
    assert prov["classifier_prefilter_model_sha256"] == ""


def _well(path: Path, **attrs) -> Path:
    f, crops = crops_mod.open_well_h5(path, crop_size=8, n_channels=2,
                                      channel_names=["Phase", "mask"], attrs=attrs)
    crops.resize(1, axis=0)
    crops[0] = np.zeros((2, 8, 8), dtype=np.float32)
    meta = {
        "cell_uid": "u", "tiff_file": path.stem, "fov_index": 0, "cell_label": 1,
        "area_px": 10, "condition": "", "reporter": "", "replica": "",
        "condition_label": "", "condition_type": "", "gene": "", "drug": "",
        "concentration": "", "is_control": False, "is_drug": False,
        "acquisition_time": "", "source_czi": "",
    }
    crops_mod.finalise_well_h5(f, [meta])
    return path


def test_consolidation_keeps_only_attrs_all_wells_agree_on(tmp_path):
    base = {**crops_mod.crop_opts_attrs(crops_mod.CropOpts()),
            "input_stage": "segment", "classifier_prefilter": "none",
            "classifier_prefilter_model_sha256": ""}
    a = _well(tmp_path / "a.h5", **base)
    b = _well(tmp_path / "b.h5", **{**base, "mask_background": False})
    legacy = _well(tmp_path / "legacy.h5")       # written before these attrs

    out = _attrs(consolidate_crops([a, b], tmp_path / "ab.h5"))
    assert "mask_background" not in out          # wells disagree -> unknown
    assert int(out["dilate_px"]) == 3
    assert out["input_stage"] == "segment"
    assert out["schema_version"] == "1.0"

    out = _attrs(consolidate_crops([a, legacy], tmp_path / "a_legacy.h5"))
    assert not set(crops_mod.CARRIED_ATTRS) & set(out)
    assert out["schema_version"] == "1.0"

    out = _attrs(consolidate_crops([a, a], tmp_path / "aa.h5"))
    assert set(crops_mod.CARRIED_ATTRS) <= set(out)

