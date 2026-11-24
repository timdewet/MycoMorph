"""``mycomorph-cli run``: the whole pipeline from a config file, without the GUI.

The parity tests run the same options twice on small synthetic images, once
through the command and once through the Qt runner the GUI uses (driven
synchronously, so no display is needed), and compare every file the two
write: images, tables, crops, ``.provenance.json`` sidecars and
``run_manifest.json``. Cellpose is replaced with a threshold; every other
stage runs for real.
"""

from __future__ import annotations

import json
import subprocess
import sys
import types
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tifffile
from typer.testing import CliRunner

from mycomorph.core.api import (
    PRESET_MODELS,
    ClassifyOpts,
    ExtractOpts,
    FocusOpts,
    SegmentOpts,
)
from mycomorph.core.cli import app
from mycomorph.core.foci import DetectorOpts
from mycomorph.core.pipeline.bulk_layout import BulkLayout
from mycomorph.core.pipeline.config import ConfigError, plan_from_config
from mycomorph.core.pipeline.context import (
    BulkRunContext,
    EmbeddingOpts,
    FluorescentNormalisationOpts,
    FociDetectionOpts,
    RunContext,
)
from mycomorph.core.pipeline.layout import PlateLayout

cli = CliRunner()

CHANNELS = ["Phase", "GFP"]
# Stage options used on both sides of the parity tests, as they would be
# typed into a config. Anything not listed keeps its default.
SEGMENT = {"gpu": False, "min_size": 20}
FEATURES = {"make_qc_plots": False, "crop_size": 64}
FOCI = {"detector_opts": {"snr_min": 2.0}}


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic data and the Cellpose stand-in
# ─────────────────────────────────────────────────────────────────────────────

def _fov(seed: int, shape=(120, 160)) -> np.ndarray:
    """One field of view, (C=2, Y, X): dark rods on a bright phase background,
    and a fluorescent spot inside each rod."""
    rng = np.random.default_rng(seed)
    phase = rng.normal(3000.0, 20.0, shape)
    gfp = rng.normal(200.0, 3.0, shape)
    yy, xx = np.mgrid[: shape[0], : shape[1]]
    for cy, cx in ((30, 40), (30, 110), (80, 50), (85, 115)):
        phase[cy - 5: cy + 5, cx - 20: cx + 20] -= 1500.0
        gfp += 900.0 * np.exp(-((yy - cy) ** 2 + (xx - cx - 6) ** 2) / (2 * 1.5 ** 2))
    return np.clip(np.stack([phase, gfp]), 0, 65535).astype(np.uint16)


def _zstack_tiff(path: Path, seed: int) -> Path:
    """A 3-plane Z-stack whose middle plane is the sharp one."""
    from scipy.ndimage import gaussian_filter

    sharp = _fov(seed).astype(np.float32)
    blurred = np.stack([gaussian_filter(c, 3.0) for c in sharp])
    stack = np.stack([blurred, sharp, blurred]).astype(np.uint16)  # (Z, C, Y, X)
    tifffile.imwrite(str(path), stack, imagej=True,
                     metadata={"axes": "ZCYX", "Labels": CHANNELS * 3})
    return path


@pytest.fixture
def fake_cellpose(monkeypatch):
    """Segment by thresholding the phase channel instead of loading Cellpose."""
    from scipy import ndimage

    from mycomorph.core import cellpose_pipeline

    class FakeModel:
        def __init__(self, gpu: bool, pretrained_model: str):
            pass

    fake_models = types.ModuleType("cellpose.models")
    fake_models.CellposeModel = FakeModel
    fake = types.ModuleType("cellpose")
    fake.models = fake_models
    monkeypatch.setitem(sys.modules, "cellpose", fake)
    monkeypatch.setitem(sys.modules, "cellpose.models", fake_models)

    def segment_phase(phase, _model, **_kwargs):
        labels, _n = ndimage.label(phase < 2300)
        return labels.astype(np.int32)

    monkeypatch.setattr(cellpose_pipeline, "segment_phase", segment_phase)
    # One process: the stage's worker pool would not see the patches above.
    monkeypatch.setenv("MYCOMORPH_FOCI_WORKERS", "1")


# ─────────────────────────────────────────────────────────────────────────────
# Comparing two output folders
# ─────────────────────────────────────────────────────────────────────────────

# Wall-clock values, and the digest that hashes the input's path and mtime.
_VOLATILE = {"started_at", "finished_at", "elapsed_s", "mtime_ns", "digest"}


def _scrub(value, root: Path):
    if isinstance(value, dict):
        return {k: _scrub(v, root) for k, v in value.items() if k not in _VOLATILE}
    if isinstance(value, list):
        return [_scrub(v, root) for v in value]
    if isinstance(value, str):
        return value.replace(str(root), "<run>")
    return value


def _h5_contents(path: Path, root: Path) -> dict:
    import h5py

    out: dict = {}

    def visit(name, node):
        attrs = {k: _scrub(_plain(v), root) for k, v in node.attrs.items()}
        if attrs:
            out[f"{name}@attrs"] = attrs
        if isinstance(node, h5py.Dataset):
            out[name] = _scrub(_plain(node[()]), root)

    with h5py.File(path, "r") as f:
        visit("/", f)
        f.visititems(visit)
    return out


def _plain(value):
    """h5py values as plain Python, so they compare with ``==``."""
    if isinstance(value, bytes):
        return value.decode()
    if isinstance(value, np.ndarray):
        if value.dtype.kind in "OSU":
            return [_plain(v) for v in value.tolist()]
        return ("array", value.dtype.str, value.shape, value.tobytes())
    if isinstance(value, np.generic):
        return value.item()
    return value


def _contents(path: Path, root: Path):
    suffix = path.suffix.lower()
    if suffix == ".tif":
        with tifffile.TiffFile(str(path)) as tif:
            meta = _scrub(dict(tif.imagej_metadata or {}), root)
            data = tif.asarray()
        return ("tiff", data.dtype.str, data.shape, data.tobytes(), meta)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".json":
        return _scrub(json.loads(path.read_text()), root)
    if suffix == ".csv":
        return path.read_text().replace(str(root), "<run>")
    if suffix == ".h5":
        return _h5_contents(path, root)
    return path.read_bytes()


def _assert_same_outputs(a: Path, b: Path) -> list[str]:
    """Every file under ``a`` has an identical twin under ``b`` and vice versa.
    Returns the relative paths compared."""
    files_a = sorted(str(p.relative_to(a)) for p in a.rglob("*") if p.is_file())
    files_b = sorted(str(p.relative_to(b)) for p in b.rglob("*") if p.is_file())
    assert files_a == files_b
    for rel in files_a:
        got, want = _contents(a / rel, a), _contents(b / rel, b)
        if isinstance(want, pd.DataFrame):
            got, want = (
                df.apply(lambda col: col.map(lambda v: _scrub(v, root)))
                for df, root in ((got, a), (want, b))
            )
            pd.testing.assert_frame_equal(got, want, obj=rel)
        else:
            assert got == want, f"{rel} differs"
    return files_a


def _run_cli(*args: str):
    res = cli.invoke(app, ["run", *args])
    assert res.exit_code == 0, res.output
    return res


_qt_app = None


def _run_gui_runner(runner_class, ctx) -> None:
    """Drive a Qt runner to completion in this thread, as its QThread would.
    Needs a QCoreApplication (no display), alive before the runner is made."""
    from PyQt6.QtCore import QCoreApplication

    global _qt_app
    _qt_app = QCoreApplication.instance() or QCoreApplication([])
    runner = runner_class(ctx, reuse_existing=True)
    finished, failures = [], []
    runner.runFinished.connect(finished.append)
    runner.runFailed.connect(failures.append)
    runner._run()
    assert not failures, failures
    assert finished


# ─────────────────────────────────────────────────────────────────────────────
# Parity: bulk mode, Focus → Foci Detection
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def bulk_inputs(tmp_path: Path) -> dict:
    """Three Z-stacks: two pooled into sample WT, one for sample KO."""
    root = tmp_path.resolve()
    raw = root / "raw"
    raw.mkdir()
    rows = []
    for i, (name, condition) in enumerate(
        (("wt_a.tif", "WT"), ("wt_b.tif", "WT"), ("ko_a.tif", "KO"))
    ):
        _zstack_tiff(raw / name, seed=i)
        rows.append({"czi_path": name, "condition": condition, "reporter": "GFP",
                     "mutant_or_drug": "DMSO", "replica": "1", "notes": ""})
    pd.DataFrame(rows).to_csv(raw / "entries.csv", index=False)
    return {"root": root, "raw": raw, "rows": rows}


def _bulk_config(inputs: dict, out: Path) -> Path:
    config = {
        "mode": "bulk",
        "bulk_entries_csv": "raw/entries.csv",
        "output_dir": str(out),
        "stages": [
            {"name": "Focus", "opts": {"mode": "whole", "metric": "vollath_f4"}},
            {"name": "Segment", "opts": SEGMENT},
            {"name": "Classify"},
            {"name": "Features", "opts": FEATURES},
            {"name": "Fluorescent Normalisation"},
            {"name": "Foci Detection", "opts": FOCI},
        ],
    }
    path = inputs["root"] / "config.json"
    path.write_text(json.dumps(config, indent=2))
    return path


def _bulk_gui_context(inputs: dict, out: Path) -> BulkRunContext:
    """The context MainWindow builds for the same settings: the entries table
    from the Input page, each panel's options, the Run page's ticks."""
    layout = BulkLayout.empty()
    layout.add_files([inputs["raw"] / r["czi_path"] for r in inputs["rows"]])
    for column in ("condition", "reporter", "mutant_or_drug", "replica"):
        layout.df[column] = [r[column] for r in inputs["rows"]]
    assert layout.validate() == []
    return BulkRunContext(
        czi_entries=layout.active_rows().to_dict(orient="records"),
        output_dir=out,
        do_split=False,
        do_focus=True,
        do_segment=True,
        do_classify=True,
        do_features=True,
        do_fluorescent_normalisation=True,
        do_foci_detection=True,
        focus_opts=FocusOpts(mode="whole", metric="vollath_f4", phase_channel=0),
        segment_opts=SegmentOpts(**SEGMENT),
        classify_opts=ClassifyOpts(),
        features_opts=ExtractOpts(**FEATURES),
        fluorescent_normalisation_opts=FluorescentNormalisationOpts(),
        foci_detection_opts=FociDetectionOpts(detector_opts=DetectorOpts(snr_min=2.0)),
        phase_channel=0,
        channel_labels=list(CHANNELS),
    )


def test_bulk_run_matches_the_gui_runner(bulk_inputs, fake_cellpose):
    pytest.importorskip("PyQt6")
    from mycomorph.gui.pipeline.runner import BulkPipelineRunner

    root = bulk_inputs["root"]
    cli_out, gui_out = root / "cli" / "run1", root / "gui" / "run1"

    res = _run_cli("--config", str(_bulk_config(bulk_inputs, cli_out)))
    assert "Run complete." in res.output
    assert "Enabled stages: Focus, Segment, Classify, Features, " \
           "Fluorescent Normalisation, Foci Detection" in res.output

    _run_gui_runner(BulkPipelineRunner, _bulk_gui_context(bulk_inputs, gui_out))

    compared = _assert_same_outputs(cli_out, gui_out)

    # The comparison covered what a run is meant to leave behind.
    for expected in (
        "run_manifest.json",
        "bulk_entries.csv",
        "01_focus/WT__GFP__DMSO__R1_focused.tif",
        "02_segment/WT__GFP__DMSO__R1_focused.tif",
        "02_segment/WT__GFP__DMSO__R1_focused.tif.provenance.json",
        "03_classify/KO__GFP__DMSO__R1_focused.tif",
        "03_classify/KO__GFP__DMSO__R1_focused.tif.provenance.json",
        "04_features/WT__GFP__DMSO__R1_focused.parquet",
        "04_features/WT__GFP__DMSO__R1_focused.parquet.provenance.json",
        "04_features/all_features.parquet",
        "04_features/all_crops.h5",
        "04b_fluorescent_normalisation/WT__GFP__DMSO__R1_focused.tif.provenance.json",
        "04c_foci_detection/WT__GFP__DMSO__R1_focused.parquet",
        "04c_foci_detection/WT__GFP__DMSO__R1_focused.parquet.provenance.json",
    ):
        assert expected in compared
    # ...and the stages did real work: two pooled fields of four cells, each
    # with one focus.
    features = pd.read_parquet(cli_out / "04_features" / "WT__GFP__DMSO__R1_focused.parquet")
    assert len(features) == 8
    foci = pd.read_parquet(cli_out / "04c_foci_detection" / "WT__GFP__DMSO__R1_focused.parquet")
    assert len(foci) == 8

    manifest = json.loads((cli_out / "run_manifest.json").read_text())
    assert [s["name"] for s in manifest["stages"]] == [
        "Focus", "Segment", "Classify", "Features",
        "Fluorescent Normalisation", "Foci Detection",
    ]
    assert manifest["stopped_early"] is False


def test_reuse_and_no_reuse(bulk_inputs, fake_cellpose):
    out = bulk_inputs["root"] / "out" / "run1"
    config = _bulk_config(bulk_inputs, out)
    _run_cli("--config", str(config), "--quiet")
    segmented = out / "02_segment" / "WT__GFP__DMSO__R1_focused.tif"
    first = segmented.stat().st_mtime_ns

    res = _run_cli("--config", str(config))  # reuse is the default
    assert "Reused WT__GFP__DMSO__R1_focused.tif" in res.output
    assert segmented.stat().st_mtime_ns == first

    res = _run_cli("--config", str(config), "--no-reuse")
    assert "Reused" not in res.output
    assert "reuse is off" in res.output  # the overwrite warning
    assert segmented.stat().st_mtime_ns != first


def test_gui_manifest_replays_from_the_command_line(bulk_inputs, fake_cellpose):
    """A manifest written by a GUI run is a config that repeats the run."""
    pytest.importorskip("PyQt6")
    from mycomorph.gui.pipeline.runner import BulkPipelineRunner

    root = bulk_inputs["root"]
    gui_out, replay_out = root / "gui" / "run1", root / "replay" / "run1"
    _run_gui_runner(BulkPipelineRunner, _bulk_gui_context(bulk_inputs, gui_out))

    _run_cli("--config", str(gui_out / "run_manifest.json"),
             "--output-dir", str(replay_out), "--quiet")

    _assert_same_outputs(replay_out, gui_out)


# ─────────────────────────────────────────────────────────────────────────────
# Parity: plate mode, Split → Features
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def fake_plate(tmp_path: Path, monkeypatch) -> dict:
    """A "plate CZI" whose reading is stubbed: scenes 0-1 in A1, 2 in A2, and
    3 in B1, a well the layout leaves unlabelled."""
    from mycomorph.core import split_czi_plate as split_mod

    root = tmp_path.resolve()
    czi = root / "raw" / "plate.czi"
    czi.parent.mkdir()
    czi.write_bytes(b"stub")
    scene_wells = {0: "A1", 1: "A1", 2: "A2", 3: "B1"}
    monkeypatch.setattr(split_mod, "extract_scene_well_map", lambda _p: scene_wells)
    monkeypatch.setattr(split_mod, "read_czi_scenes",
                        lambda _p: {i: _fov(seed=10 + i) for i in scene_wells})
    monkeypatch.setattr(split_mod, "extract_channel_names", lambda _p: list(CHANNELS))
    monkeypatch.setattr(split_mod, "_read_czi_acquisition_time", lambda _p: None)
    monkeypatch.setattr(split_mod, "_read_czi_per_scene_acquisition_times", lambda _p: {})

    layout_csv = root / "raw" / "layout.csv"
    layout_csv.write_text(
        "well,condition,reporter,mutant_or_drug,replica\n"
        "A1,ATc+,GFP,rpoB,1\n"
        "A2,ATc+,GFP,NT,1\n"
    )
    return {"root": root, "czi": czi, "layout_csv": layout_csv}


def test_plate_run_matches_the_gui_runner(fake_plate, fake_cellpose):
    pytest.importorskip("PyQt6")
    from mycomorph.gui.pipeline.runner import PipelineRunner

    root = fake_plate["root"]
    cli_out, gui_out = root / "cli" / "plate1", root / "gui" / "plate1"

    # Channel labels and the phase channel are left to be read from the file.
    config = root / "raw" / "config.json"
    config.write_text(json.dumps({
        "mode": "plate",
        "czi_paths": ["plate.czi"],
        "plate_layout_csv": "layout.csv",
        "output_dir": str(cli_out),
        "stages": [
            {"name": "Split"},
            {"name": "Segment", "opts": SEGMENT},
            {"name": "Classify"},
            {"name": "Features", "opts": FEATURES},
        ],
    }))
    res = _run_cli("--config", str(config))
    assert "Channels: Phase, GFP (phase channel 0)" in res.output

    # The GUI's side: the Plate layout page reads wells from the CZI and the
    # user labels them; the Input page reads the channels.
    layout = PlateLayout.from_czis([fake_plate["czi"]])
    layout.set_wells(["A1"], condition="ATc+", reporter="GFP",
                     mutant_or_drug="rpoB", replica="1")
    layout.set_wells(["A2"], condition="ATc+", reporter="GFP",
                     mutant_or_drug="NT", replica="1")
    ctx = RunContext(
        czi_path=fake_plate["czi"],
        czi_paths=[fake_plate["czi"]],
        output_dir=gui_out,
        layout=layout,
        do_split=True,
        do_focus=False,
        do_segment=True,
        do_classify=True,
        do_features=True,
        focus_opts=FocusOpts(phase_channel=0),
        segment_opts=SegmentOpts(**SEGMENT),
        classify_opts=ClassifyOpts(),
        features_opts=ExtractOpts(**FEATURES),
        phase_channel=0,
        channel_labels=list(CHANNELS),
    )
    _run_gui_runner(PipelineRunner, ctx)

    compared = _assert_same_outputs(cli_out, gui_out)
    for expected in (
        "run_manifest.json",
        "plate_layout.csv",
        "01_split/ATc+__GFP__rpoB__R1.tif",
        "02_segment/ATc+__GFP__NT__R1.tif.provenance.json",
        "03_classify/ATc+__GFP__rpoB__R1.tif.provenance.json",
        "04_features/ATc+__GFP__rpoB__R1.parquet.provenance.json",
        "04_features/all_features.parquet",
    ):
        assert expected in compared
    # The GUI run's manifest, replayed by the command, gives the same again.
    replay_out = root / "replay" / "plate1"
    _run_cli("--config", str(gui_out / "run_manifest.json"),
             "--output-dir", str(replay_out), "--quiet")
    _assert_same_outputs(replay_out, gui_out)

    # The unlabelled well B1 was not processed.
    assert len(list((cli_out / "01_split").glob("*.tif"))) == 2
    features = pd.read_parquet(cli_out / "04_features" / "ATc+__GFP__rpoB__R1.parquet")
    assert len(features) == 8


# ─────────────────────────────────────────────────────────────────────────────
# The manifest's config block
# ─────────────────────────────────────────────────────────────────────────────

def _context_fields(ctx) -> dict:
    """Everything in a context that decides what a run does."""
    data = {k: v for k, v in asdict(ctx).items() if k not in ("layout", "reuse_existing")}
    if hasattr(ctx, "layout"):
        data["layout"] = ctx.layout.disambiguated_active_rows().to_dict(orient="records")
    return data


def test_manifest_config_block_rebuilds_the_same_context(bulk_inputs, monkeypatch):
    """Non-default options of every stage, ticked or not, survive the round
    trip through the manifest: tuples, paths and the nested detector options."""
    from mycomorph.core.pipeline import runner as core_runner

    out = bulk_inputs["root"] / "out"
    ctx = _bulk_gui_context(bulk_inputs, out)
    ctx.do_split = True                      # ticked, but subsumed by Focus
    ctx.do_foci_detection = False            # unticked: its options still count
    ctx.focus_opts = FocusOpts(mode="tiles", tile_grid=(2, 4), phase_channel="Phase",
                               filename_suffix="_f")
    ctx.segment_opts = SegmentOpts(diameter=25.0, gpu=False, model="cpsam_v2")
    ctx.classify_opts = ClassifyOpts(
        model_path=PRESET_MODELS["mtb-2026-06"],
        keep_classes=("good", "ok"), decision_threshold=0.3,
    )
    ctx.features_opts = ExtractOpts(fluorescence_channels=[1], library_dir=out / "lib",
                                    control_labels="NT, WT")
    ctx.fluorescent_normalisation_opts = FluorescentNormalisationOpts(
        method="tophat", crosstalk_enabled=True, crosstalk_source_channel=1,
        crosstalk_target_channel=2, crosstalk_k=0.075,
    )
    ctx.foci_detection_opts = FociDetectionOpts(
        detector_keys=["hmax", "log"],
        detector_opts=DetectorOpts(wavelet_scales=(1, 2, 3), trackpy_minmass=5.0),
        apply_to_channels=[1],
    )

    # Only the manifest is wanted here, not the stages' work.
    for stage in ("SegmentStage", "ClassifyStage", "ExtractStage",
                  "FluorescentNormalisationStage"):
        monkeypatch.setattr(getattr(core_runner, stage), "run", lambda *_a: [])
    monkeypatch.setattr("mycomorph.core.api.run_focus_sample", lambda *_a, **_k: None)
    run = core_runner.BulkRun(ctx)
    failures: list[str] = []
    run.runFailed.connect(failures.append)
    run.execute()
    assert not failures, failures

    manifest = json.loads(ctx.manifest_path.read_text())
    assert [s["name"] for s in manifest["config"]["stages"] if s["enabled"]] == [
        "Split", "Focus", "Segment", "Classify", "Features",
        "Fluorescent Normalisation",
    ]
    plan = plan_from_config(manifest, base_dir=out)
    assert plan.mode == "bulk"
    assert _context_fields(plan.ctx) == _context_fields(ctx)


def test_embedding_options_round_trip(tmp_path: Path):
    """The hidden Embeddings stage is recorded only when it is ticked."""
    from mycomorph.core.pipeline.runner import _requested_config

    ctx = BulkRunContext(czi_entries=[], output_dir=tmp_path)
    assert "Embeddings" not in [s["name"] for s in _requested_config(ctx, "bulk")["stages"]]

    ctx.do_embeddings = True
    ctx.embeddings_opts = EmbeddingOpts(model_type="Lightweight", epochs=3,
                                        library_dir=tmp_path / "lib")
    block = json.loads(json.dumps(_requested_config(ctx, "bulk"), default=str))
    block["entries"] = [{"czi_path": "a.tif", "condition": "WT"}]
    plan = plan_from_config(block, base_dir=tmp_path)
    assert plan.ctx.do_embeddings
    assert plan.ctx.embeddings_opts == ctx.embeddings_opts


def test_older_plate_manifest_keeps_its_split_and_focused_folder(tmp_path: Path):
    """Before the config block, a manifest listed only the stages that ran.
    Focus runs in place of a ticked Split, and that tick names the folder."""
    layout_csv = tmp_path / "plate_layout.csv"
    layout_csv.write_text("well,condition\nA1,WT\n")
    manifest = {
        "started_at": 1.0,
        "software": {"mycomorph": "0.6.1"},
        "czi": str(tmp_path / "plate.czi"),
        "output_dir": str(tmp_path / "out"),
        "stages": [
            {"name": "Focus", "reused": False, "elapsed_s": 2.0,
             "opts": {"metric": "ensemble", "mode": "edf", "tile_grid": [3, 3],
                      "phase_channel": 0, "save_zmaps": False, "save_mip": False,
                      "filename_suffix": "_focused"},
             "outputs": [str(tmp_path / "out" / "01_split_and_focused" / "WT_focused.tif")]},
            {"name": "Segment", "reused": False, "elapsed_s": 3.0, "opts": {},
             "outputs": []},
        ],
        "finished_at": 9.0,
        "stopped_early": False,
        "plate_layout_csv": str(layout_csv),
    }
    plan = plan_from_config(manifest, base_dir=tmp_path)
    assert plan.mode == "plate"
    assert plan.enabled_stages == ["Split", "Focus", "Segment"]
    assert plan.ctx.focus_dir == tmp_path / "out" / "01_split_and_focused"
    assert plan.ctx.focus_opts == FocusOpts(phase_channel=0)

    manifest["stages"][0]["outputs"] = [str(tmp_path / "out" / "01_focus" / "WT_focused.tif")]
    plan = plan_from_config(manifest, base_dir=tmp_path)
    assert plan.enabled_stages == ["Focus", "Segment"]
    assert plan.ctx.focus_dir == tmp_path / "out" / "01_focus"


# ─────────────────────────────────────────────────────────────────────────────
# Reading configs
# ─────────────────────────────────────────────────────────────────────────────

def _minimal(tmp_path: Path, **overrides) -> dict:
    config = {
        "mode": "bulk",
        "entries": [{"czi_path": "a.tif", "condition": "WT"}],
        "output_dir": "out",
        "stages": [{"name": "Segment"}],
    }
    config.update(overrides)
    return config


def test_config_defaults_match_the_gui_defaults(tmp_path: Path):
    plan = plan_from_config(_minimal(tmp_path), base_dir=tmp_path)
    ctx = plan.ctx
    assert ctx.output_dir == tmp_path / "out"
    assert ctx.czi_entries == [{
        "czi_path": str(tmp_path / "a.tif"), "condition": "WT", "reporter": "",
        "mutant_or_drug": "", "replica": "", "notes": "",
    }]
    assert plan.enabled_stages == ["Segment"]
    assert ctx.segment_opts == SegmentOpts()
    assert ctx.classify_opts == ClassifyOpts()
    assert ctx.features_opts == ExtractOpts()
    # No readable input to take channel names from: unset, phase channel 0,
    # which Focus is given too, as in the app.
    assert ctx.channel_labels is None and ctx.phase_channel == 0
    assert ctx.focus_opts == FocusOpts(phase_channel=0)


def test_channels_are_read_from_the_first_input(tmp_path: Path):
    tifffile.imwrite(str(tmp_path / "a.tif"), np.zeros((2, 8, 8), np.uint16),
                     imagej=True, metadata={"axes": "CYX", "Labels": ["mCherry", "TL"]})
    plan = plan_from_config(_minimal(tmp_path), base_dir=tmp_path)
    assert plan.ctx.channel_labels == ["mCherry", "TL"]
    assert plan.ctx.phase_channel == 1
    assert plan.ctx.focus_opts.phase_channel == 1

    explicit = _minimal(tmp_path, channel_labels=["a", "b"], phase_channel=0)
    plan = plan_from_config(explicit, base_dir=tmp_path)
    assert plan.ctx.channel_labels == ["a", "b"] and plan.ctx.phase_channel == 0


def test_stage_names_are_forgiving_and_options_are_typed(tmp_path: Path):
    config = _minimal(tmp_path, stages=[
        "focus",
        {"name": "fluorescent_normalization", "enabled": False,
         "opts": {"method": "tophat"}},
        {"name": "Foci Detection", "opts": {"detector_opts": {"wavelet_scales": [1, 3]}}},
        {"name": "Segment", "opts": {"diameter": 25}},
        {"name": "Classify", "opts": {"model_path": "msm-2026-06", "keep_classes": ["good"]}},
    ])
    ctx = plan_from_config(config, base_dir=tmp_path).ctx
    assert (ctx.do_focus, ctx.do_segment, ctx.do_classify, ctx.do_foci_detection) == (
        True, True, True, True)
    assert not ctx.do_fluorescent_normalisation
    assert ctx.fluorescent_normalisation_opts.method == "tophat"
    assert ctx.foci_detection_opts.detector_opts.wavelet_scales == (1, 3)
    assert ctx.segment_opts.diameter == 25.0 and isinstance(ctx.segment_opts.diameter, float)
    assert ctx.classify_opts.model_path.name == "msm-2026-06_best_model.pth"
    assert ctx.classify_opts.keep_classes == ("good",)


@pytest.mark.parametrize("change, message", [
    ({"stages": [{"name": "Segment", "opts": {"diamter": 20}}]}, "unknown option(s) diamter"),
    ({"stages": [{"name": "Segment", "opts": {"gpu": "no"}}]}, "Segment.gpu: expected true or false"),
    ({"stages": [{"name": "Focus", "opts": {"tile_grid": [3]}}]}, "expected 2 values"),
    ({"stages": [{"name": "Polish"}]}, "unknown stage 'Polish'"),
    ({"stages": [{"name": "Segment"}, {"name": "segment"}]}, "listed twice"),
    ({"stages": [{"name": "Split", "opts": {"x": 1}}]}, "takes no options"),
    ({"stages": [{"name": "Segment", "enabled": False}]}, "No stage is enabled"),
    ({"stages": []}, 'needs a "stages" list'),
    ({"stages": [{"name": "Classify", "opts": {"model_path": "nope.pth"}}]},
     "neither an existing file nor a bundled preset"),
    ({"output_dir": None}, 'needs an "output_dir"'),
    ({"ouput_dir": "x"}, "Unknown config key(s): ouput_dir"),
    ({"mode": "single"}, '"mode" must be "plate" or "bulk"'),
    ({"phase_channel": 3, "channel_labels": ["a", "b"]}, "only 2 channel labels"),
    ({"entries": [{"czi_path": "a.tif"}]}, "No entry has a condition"),
    ({"entries": [{"czi_path": "a.tif", "condtion": "WT"}]}, "unknown key(s) condtion"),
    ({"bulk_entries_csv": "entries.csv"}, "needs one of"),
    ({"mode": "plate"}, 'needs "czi_paths"'),
    ({"mode": "plate", "czi_paths": ["p.czi"]}, 'needs "plate_layout_csv"'),
])
def test_config_errors_say_what_to_fix(tmp_path: Path, change: dict, message: str):
    with pytest.raises(ConfigError) as err:
        plan_from_config(_minimal(tmp_path, **change), base_dir=tmp_path)
    assert message in str(err.value)


def test_invalid_config_exits_2(tmp_path: Path):
    config = tmp_path / "config.json"
    config.write_text(json.dumps(_minimal(tmp_path, stages=[{"name": "Polish"}])))
    res = cli.invoke(app, ["run", "--config", str(config)])
    assert res.exit_code == 2
    assert "Config error:" in res.output and "unknown stage 'Polish'" in res.output

    config.write_text("{not json")
    res = cli.invoke(app, ["run", "--config", str(config)])
    assert res.exit_code == 2
    assert "is not valid JSON" in res.output


def test_missing_inputs_stop_the_run_before_it_starts(tmp_path: Path):
    """The checks the GUI makes when Run is pressed."""
    config = tmp_path / "config.json"
    config.write_text(json.dumps(_minimal(tmp_path, stages=[{"name": "Classify"}])))
    res = cli.invoke(app, ["run", "--config", str(config)])
    assert res.exit_code == 1
    assert "Cannot run:" in res.output
    assert "Classify is enabled but Segment is disabled" in res.output
    assert "File not found" in res.output and "a.tif" in res.output
    assert not (tmp_path / "out").exists()


def test_missing_plate_czi_is_reported_before_the_run(tmp_path: Path):
    (tmp_path / "layout.csv").write_text("well,condition\nA1,WT\n")
    config = tmp_path / "config.json"
    config.write_text(json.dumps({
        "czi_paths": ["gone.czi"], "plate_layout_csv": "layout.csv",
        "output_dir": "out", "stages": ["Split", "Focus", "Segment"],
    }))
    res = cli.invoke(app, ["run", "--config", str(config), "--dry-run"])
    assert res.exit_code == 1
    assert "plate run" in res.output
    assert "CZI file not found" in res.output and "gone.czi" in res.output


def test_dry_run_checks_without_running(bulk_inputs):
    out = bulk_inputs["root"] / "out"
    res = _run_cli("--config", str(_bulk_config(bulk_inputs, out)), "--dry-run")
    assert "Channel labels read from wt_a.tif: Phase, GFP" in res.output
    assert "Dry run: config is valid" in res.output
    assert not out.exists()


def test_failed_run_exits_1(bulk_inputs, monkeypatch):
    from mycomorph.core.pipeline import stages

    def boom(**_kwargs):
        raise RuntimeError("segmentation exploded")

    monkeypatch.setattr(stages, "segment_tiff", boom)
    res = cli.invoke(app, ["run", "--config",
                           str(_bulk_config(bulk_inputs, bulk_inputs["root"] / "out"))])
    assert res.exit_code == 1
    assert "Run failed: segmentation exploded" in res.output


@pytest.mark.skipif(sys.platform == "win32", reason="sends SIGINT to itself")
def test_ctrl_c_stops_at_the_next_safe_point(bulk_inputs, monkeypatch):
    """Ctrl-C is the GUI's Stop button: the run ends at the next progress
    report and the manifest is still written."""
    import os
    import signal

    from mycomorph.core.pipeline import stages

    def interrupted(*, progress_cb, **_kwargs):
        os.kill(os.getpid(), signal.SIGINT)
        progress_cb(0.5, "still segmenting")
        raise AssertionError("the run should have stopped at the progress report")

    monkeypatch.setattr(stages, "segment_tiff", interrupted)
    out = bulk_inputs["root"] / "out"
    before = signal.getsignal(signal.SIGINT)
    res = cli.invoke(app, ["run", "--config", str(_bulk_config(bulk_inputs, out))])

    assert res.exit_code == 130, res.output
    assert "Stopping at the next safe point" in res.output
    assert "Run stopped. Manifest:" in res.output
    assert signal.getsignal(signal.SIGINT) is before
    manifest = json.loads((out / "run_manifest.json").read_text())
    assert manifest["stopped_early"] is True
    assert [(s["name"], s.get("stopped", False)) for s in manifest["stages"]] == [
        ("Focus", False), ("Segment", True),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# The command itself
# ─────────────────────────────────────────────────────────────────────────────

def test_run_help_documents_the_options():
    import re

    res = cli.invoke(app, ["run", "--help"])
    assert res.exit_code == 0
    text = " ".join(re.sub(r"\x1b\[[0-9;]*m", "", res.output).split())
    for option in ("--config", "--reuse", "--no-reuse", "--output-dir", "--dry-run", "--quiet"):
        assert option in text
    assert "run_manifest.json" in text
    assert '"stages"' in text and '"plate_layout_csv"' in text


def test_run_never_imports_qt_or_the_gui(bulk_inputs):
    """The command works where no GUI can: nothing it loads touches Qt."""
    config = _bulk_config(bulk_inputs, bulk_inputs["root"] / "out")
    script = (
        "import sys\n"
        "from mycomorph.core.cli import app\n"
        "app(['run', '--config', sys.argv[1], '--dry-run'], standalone_mode=False)\n"
        "bad = sorted(m for m in sys.modules if m.startswith(('PyQt6', 'mycomorph.gui')))\n"
        "assert not bad, bad\n"
    )
    res = subprocess.run(
        [sys.executable, "-c", script, str(config)],
        capture_output=True, text=True, timeout=300,
    )
    assert res.returncode == 0, res.stdout + res.stderr
    assert "Dry run: config is valid" in res.stdout
