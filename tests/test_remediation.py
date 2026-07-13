from __future__ import annotations

import inspect
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import tifffile
from typer.testing import CliRunner

from mycomorph.core import api
from mycomorph.core.errors import PipelineValidationError
from mycomorph.core.extract.api import extract_features_tiff
from mycomorph.core.extract.feature_library import FeatureLibrary
from mycomorph.core.naming import build_output_filename
from mycomorph.core.provenance import (
    artifact_is_in_progress,
    checkpoint_signature,
    clear_artifact_in_progress,
    mark_artifact_in_progress,
    prepare_checkpoint_dir,
)
from mycomorph.core import split_czi_plate as split_mod


@pytest.mark.parametrize(
    "function,parameter",
    [
        (api.run_focus, "opts"),
        (api.segment_tiff, "opts"),
        (api.classify_filter_tiff, "opts"),
        (extract_features_tiff, "opts"),
    ],
)
def test_public_option_defaults_are_none(function, parameter):
    assert inspect.signature(function).parameters[parameter].default is None


def test_pixel_calibration_ignores_imagej_z_spacing(tmp_path: Path):
    path = tmp_path / "calibrated.tif"
    tifffile.imwrite(
        path,
        np.zeros((2, 8, 8), dtype=np.uint16),
        imagej=True,
        metadata={"axes": "ZYX", "spacing": 2.0, "unit": "um"},
        resolution=(100_000, 100_000),  # 10 px/um, expressed as px/cm
        resolutionunit="CENTIMETER",
    )
    assert api._read_pixels_per_um(path) == pytest.approx(10.0)


def test_ome_physical_size_precedes_tiff_resolution(tmp_path: Path):
    path = tmp_path / "ome.tif"
    tifffile.imwrite(
        path,
        np.zeros((8, 8), dtype=np.uint16),
        ome=True,
        metadata={
            "axes": "YX",
            "PhysicalSizeX": 0.2,
            "PhysicalSizeXUnit": "um",
        },
        resolution=(10_000, 10_000),  # conflicting fallback: 1 px/um
        resolutionunit="CENTIMETER",
    )
    assert api._read_pixels_per_um(path) == pytest.approx(5.0)


@pytest.mark.parametrize(
    "label",
    ["WT/knockdown", "drug:control", "../outside", "CON", "bad\x01label"],
)
def test_output_names_are_portable_and_cannot_traverse(label: str):
    name = build_output_filename(label, "reporter", "mutant")
    assert Path(name).name == name
    assert "/" not in name and "\\" not in name and ":" not in name
    assert "\x01" not in name
    assert name.endswith(".tif")


def _patch_split_io(monkeypatch, scene_wells: dict[int, str]) -> None:
    monkeypatch.setattr(split_mod, "extract_scene_well_map", lambda _p: scene_wells)
    monkeypatch.setattr(
        split_mod,
        "read_czi_scenes",
        lambda _p: {i: np.zeros((1, 4, 4), dtype=np.uint16) for i in scene_wells},
    )
    monkeypatch.setattr(split_mod, "extract_channel_names", lambda _p: ["Phase"])
    monkeypatch.setattr(split_mod, "_read_czi_acquisition_time", lambda _p: None)
    monkeypatch.setattr(split_mod, "_read_czi_per_scene_acquisition_times", lambda _p: {})

    def fake_save(_stacked, output_path, *_args, **_kwargs):
        Path(output_path).write_bytes(b"tiff")

    monkeypatch.setattr(split_mod, "save_hyperstack", fake_save)


def test_split_validation_raises_library_exception(monkeypatch, tmp_path: Path):
    _patch_split_io(monkeypatch, {0: "A1"})
    layout = {
        "B1": {"condition": "c", "reporter": "r", "mutant_or_drug": "m"},
    }
    with pytest.raises(PipelineValidationError, match="No matching wells"):
        split_mod.split_and_save(tmp_path / "plate.czi", layout, tmp_path / "out")


def test_split_returns_only_exact_writes_and_disambiguates_collisions(
    monkeypatch, tmp_path: Path,
):
    _patch_split_io(monkeypatch, {0: "A1", 1: "A2"})
    out = tmp_path / "out"
    out.mkdir()
    (out / "stale.tif").write_bytes(b"old")
    row = {"condition": "c", "reporter": "r", "mutant_or_drug": "m"}
    written = split_mod.split_and_save(
        tmp_path / "plate.czi", {"A1": dict(row), "A2": dict(row)}, out,
    )
    assert len(written) == 2
    assert {p.name for p in written} == {
        "c__r__m__WA1.tif",
        "c__r__m__WA2.tif",
    }
    assert out / "stale.tif" not in written


def test_delegated_cli_forwards_unknown_options_without_separator(monkeypatch):
    from mycomorph.core import cli

    captured: list[str] = []
    monkeypatch.setattr(cli, "_delegate", lambda _module, argv: captured.extend(argv))
    result = CliRunner().invoke(
        cli.app,
        ["segment", "--input", "images", "--output", "results", "--gpu"],
    )
    assert result.exit_code == 0, result.output
    assert captured == ["--input", "images", "--output", "results", "--gpu"]


def test_delegated_cli_help_reaches_argparse():
    from mycomorph.core.cli import app

    result = CliRunner().invoke(app, ["segment", "--help"])
    assert result.exit_code == 0
    assert "--phase-channel" in result.output


def test_changed_or_corrupt_checkpoint_signature_is_discarded(tmp_path: Path):
    source = tmp_path / "input.tif"
    source.write_bytes(b"input")
    partial = tmp_path / "result.partial"
    first = checkpoint_signature(stage="features", input_path=source, options={"crop": 64})
    assert prepare_checkpoint_dir(partial, first) is False
    marker = partial / "fov_000.npz"
    marker.write_bytes(b"checkpoint")

    changed = checkpoint_signature(stage="features", input_path=source, options={"crop": 128})
    assert prepare_checkpoint_dir(partial, changed) is False
    assert not marker.exists()

    (partial / "checkpoint_signature.json").write_text("not-json")
    marker.write_bytes(b"corrupt")
    assert prepare_checkpoint_dir(partial, changed) is False
    assert not marker.exists()


def test_in_progress_artifact_marker_survives_until_explicit_success(tmp_path: Path):
    output = tmp_path / "output.tif"
    mark_artifact_in_progress(output)
    assert artifact_is_in_progress(output)
    clear_artifact_in_progress(output)
    assert not artifact_is_in_progress(output)


def test_checkpoint_signature_binds_channels_phase_and_model_hash(tmp_path: Path):
    source = tmp_path / "input.tif"
    model = tmp_path / "model.pth"
    source.write_bytes(b"image")
    model.write_bytes(b"weights-v1")
    base = checkpoint_signature(
        stage="classify", input_path=source, options={"threshold": 0.5},
        channels=["Phase", "GFP"], phase_channel=0, model_path=model,
    )
    changed_channel = checkpoint_signature(
        stage="classify", input_path=source, options={"threshold": 0.5},
        channels=["GFP", "Phase"], phase_channel=1, model_path=model,
    )
    model.write_bytes(b"weights-v2")
    changed_model = checkpoint_signature(
        stage="classify", input_path=source, options={"threshold": 0.5},
        channels=["Phase", "GFP"], phase_channel=0, model_path=model,
    )
    assert base["digest"] != changed_channel["digest"]
    assert base["digest"] != changed_model["digest"]


def test_hyperstack_write_does_not_publish_partial_file(monkeypatch, tmp_path: Path):
    from mycomorph.core import cellpose_pipeline

    output = tmp_path / "result.tif"
    output.write_bytes(b"previous-good-file")

    def interrupted(path, *_args, **_kwargs):
        Path(path).write_bytes(b"partial")
        raise RuntimeError("interrupted")

    monkeypatch.setattr(cellpose_pipeline.tifffile, "imwrite", interrupted)
    with pytest.raises(RuntimeError, match="interrupted"):
        cellpose_pipeline.save_hyperstack(
            np.zeros((1, 1, 4, 4), dtype=np.uint16), output, "condition",
            ["fov_000"], ["Phase"], 10.0,
        )
    assert output.read_bytes() == b"previous-good-file"
    assert not output.with_name(f".{output.name}.tmp").exists()


def _synthetic_segmented_tiff(path: Path, pixels_per_um: float = 2.0) -> None:
    from mycomorph.core.cellpose_pipeline import save_hyperstack

    data = np.zeros((1, 2, 32, 32), dtype=np.uint16)
    data[0, 0, 10:14, 10:14] = 100
    data[0, 1, 10:14, 10:14] = 1
    save_hyperstack(
        data, path, "condition", ["fov_000"], ["Phase", "BinaryMask"],
        pixels_per_um,
    )


def test_segmentation_tiff_adapter_preserves_fovs_and_scale(monkeypatch, tmp_path: Path):
    from mycomorph.core import cellpose_pipeline
    from mycomorph.core.label_cells import load_hyperstack

    source = tmp_path / "input.tif"
    _synthetic_segmented_tiff(source)
    output = tmp_path / "segmented.tif"

    class FakeModel:
        def __init__(self, gpu: bool):
            self.gpu = gpu

    fake_models = types.ModuleType("cellpose.models")
    fake_models.CellposeModel = FakeModel
    fake_cellpose = types.ModuleType("cellpose")
    fake_cellpose.models = fake_models
    monkeypatch.setitem(sys.modules, "cellpose", fake_cellpose)
    monkeypatch.setitem(sys.modules, "cellpose.models", fake_models)

    stacked = np.zeros((2, 2, 16, 16), dtype=np.uint16)
    stacked[:, 0, 5:9, 5:9] = 100
    stacked[:, 1, 5:9, 5:9] = 255
    monkeypatch.setattr(
        cellpose_pipeline,
        "process_tiff_unit",
        lambda **_kwargs: (stacked, ["fov_000", "fov_001"], 2),
    )

    api.segment_tiff(source, output, phase_channel=0, opts=api.SegmentOpts(gpu=False))
    loaded, _ = load_hyperstack(output)
    assert loaded.shape == (2, 2, 16, 16)
    assert api._read_pixels_per_um(output) == pytest.approx(2.0)


def test_classification_adapter_uses_physical_scale(monkeypatch, tmp_path: Path):
    from mycomorph.core import cell_quality_classifier
    from mycomorph.core.label_cells import load_hyperstack

    source = tmp_path / "segmented.tif"
    output = tmp_path / "classified.tif"
    _synthetic_segmented_tiff(source, pixels_per_um=2.0)
    observed: dict[str, float] = {}

    def fake_filter(*, labeled_mask, pixels_per_um, **_kwargs):
        observed["pixels_per_um"] = pixels_per_um
        return labeled_mask, {
            "total_cells": 1, "kept": 1, "removed_edge": 0,
            "removed_debris": 0, "removed_cnn": 0,
        }

    monkeypatch.setattr(cell_quality_classifier, "classify_and_filter_mask", fake_filter)
    api.classify_filter_tiff(source, output, phase_channel=0)
    loaded, _ = load_hyperstack(output)
    assert loaded.shape == (1, 2, 32, 32)
    assert observed["pixels_per_um"] == pytest.approx(2.0)


def test_feature_extraction_uses_physical_area(tmp_path: Path):
    from mycomorph.core.extract.api import ExtractOpts

    source = tmp_path / "segmented.tif"
    output = tmp_path / "features.parquet"
    _synthetic_segmented_tiff(source, pixels_per_um=2.0)
    extract_features_tiff(
        source,
        output,
        ExtractOpts(
            midline_features=False,
            refine_contour=False,
            save_csv=False,
            save_crops=False,
        ),
    )
    features = pd.read_parquet(output)
    assert features.loc[0, "area_px"] == 16
    assert features.loc[0, "area_um2"] == pytest.approx(4.0)


def test_wavelet_foci_detection_on_synthetic_spot():
    from mycomorph.core.foci import DetectorOpts
    from mycomorph.core.foci.detectors.wavelet import WaveletAtrousDetector
    from mycomorph.core.foci.features import features_dataframe

    yy, xx = np.mgrid[:64, :64]
    rng = np.random.default_rng(5)
    image = rng.normal(5.0, 0.25, size=(64, 64))
    image += 50.0 * np.exp(-((yy - 31.0) ** 2 + (xx - 34.0) ** 2) / (2 * 1.4 ** 2))
    mask = np.ones((64, 64), dtype=np.int32)
    foci = WaveletAtrousDetector().detect(
        image, mask,
        DetectorOpts(snr_min=1.0, refine=False, wavelet_threshold_mad=4.0),
    )
    assert any(abs(f.y - 31) < 2 and abs(f.x - 34) < 2 for f in foci)
    table = features_dataframe(image, mask, foci, detector="wavelet", well="A1")
    assert not table.empty
    assert {"gaussian_r2", "hessian_symmetry", "cell_label"} <= set(table.columns)


def test_library_uses_safe_storage_keys_for_display_names(tmp_path: Path):
    source = tmp_path / "features.parquet"
    pd.DataFrame({"well": ["c__r__m"], "area": [1.0]}).to_parquet(source)
    library = FeatureLibrary(tmp_path / "library")
    library.register_run("../outside", source, "species", "knockdown")

    index = library.list_runs()
    stored = str(index.iloc[0]["features_file"])
    assert index.iloc[0]["run_id"] == "../outside"
    assert ".." not in stored
    assert (library.library_dir / stored).is_file()
    assert not (tmp_path / "outside.parquet").exists()

    model = tmp_path / "model.pth"
    model.write_bytes(b"weights")
    destination = library.register_model("../model", model, "ae", [], 1, 0.1)
    assert destination.parent == library.models_dir
    assert destination.name.startswith("model_")


def test_bulk_duplicate_names_match_downstream_stage_filter(tmp_path: Path):
    from mycomorph.gui.pipeline.stages import _expected_stems_for_layout

    entries = [
        {
            "czi_path": str(tmp_path / name),
            "condition": "WT",
            "reporter": "GFP",
            "mutant_or_drug": "control",
            "replica": "",
        }
        for name in ("one.czi", "two.czi")
    ]
    expected = _expected_stems_for_layout(
        SimpleNamespace(czi_entries=entries),
        focus_suffix="_focused",
    )
    assert len(expected) == 2
    assert all(stem.endswith("_focused") for stem in expected)
    assert all("__W" in stem for stem in expected)
