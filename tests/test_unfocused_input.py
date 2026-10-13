"""Running the pipeline on single-plane images without the Focus stage."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import tifffile

from mycomorph.core.api import FocusOpts, split_sample
from mycomorph.gui.pipeline import stages
from mycomorph.gui.pipeline.bulk_layout import sample_stem


def _plane_tiff(path: Path, shape=(2, 32, 40), seed=0) -> Path:
    arr = np.random.default_rng(seed).integers(0, 4000, size=shape, dtype=np.uint16)
    tifffile.imwrite(str(path), arr, imagej=True,
                     metadata={"axes": "CYX", "Labels": ["Phase", "GFP"]})
    return path


def test_split_sample_writes_planes_unchanged(tmp_path: Path):
    a = _plane_tiff(tmp_path / "a.tif", seed=1)
    b = _plane_tiff(tmp_path / "b.tif", seed=2)
    out_dir = tmp_path / "01_split"

    out = split_sample([a, b], out_dir, "WT__GFP")

    assert out == out_dir / "WT__GFP.tif"
    written = tifffile.imread(str(out))
    assert written.shape == (2, 2, 32, 40)
    np.testing.assert_array_equal(written[1], tifffile.imread(str(b)))
    # Nothing was focus-scored, so no score / z-map files.
    assert sorted(p.name for p in out_dir.iterdir()) == [
        "WT__GFP.tif", "WT__GFP__acquisition.json",
    ]


def test_split_sample_rejects_z_stacks(tmp_path: Path):
    zstack = tmp_path / "z.tif"
    tifffile.imwrite(str(zstack), np.zeros((3, 2, 16, 16), np.uint16),
                     imagej=True, metadata={"axes": "ZCYX"})

    with pytest.raises(ValueError, match="3 Z-planes.*Enable Focus"):
        split_sample([zstack], tmp_path / "out", "S")


def _ctx(tmp_path: Path, *, do_split: bool, do_focus: bool) -> SimpleNamespace:
    return SimpleNamespace(
        do_split=do_split, do_focus=do_focus,
        focus_dir=tmp_path / "01_focus", split_dir=tmp_path / "01_split",
        focus_opts=FocusOpts(),
    )


def test_segment_source_follows_this_runs_stage(tmp_path: Path):
    ctx = _ctx(tmp_path, do_split=True, do_focus=False)
    # A stale focus dir from an earlier run must not win over this run's Split.
    ctx.focus_dir.mkdir()
    (ctx.focus_dir / "old_focused.tif").write_bytes(b"x")
    assert stages._segment_source(ctx) == (ctx.split_dir, "")

    ctx.do_focus = True
    assert stages._segment_source(ctx) == (ctx.focus_dir, "_focused")


def test_segment_only_rerun_uses_earlier_output(tmp_path: Path):
    ctx = _ctx(tmp_path, do_split=False, do_focus=False)
    assert stages._segment_source(ctx) == (ctx.split_dir, "")

    ctx.focus_dir.mkdir()
    (ctx.focus_dir / "s_focused.tif").write_bytes(b"x")
    assert stages._segment_source(ctx) == (ctx.focus_dir, "_focused")


def _run_bulk(ctx):
    from PyQt6.QtCore import QCoreApplication

    from mycomorph.gui.pipeline.runner import BulkPipelineRunner

    QCoreApplication.instance() or QCoreApplication([])
    runner = BulkPipelineRunner(ctx)
    planned, failures = [], []
    runner.stagesPlanned.connect(planned.append)
    runner.runFailed.connect(failures.append)
    runner._run()
    return planned, failures


def test_bulk_split_without_focus_feeds_segment(tmp_path: Path, monkeypatch):
    pytest.importorskip("PyQt6")
    from mycomorph.gui.pipeline.context import BulkRunContext

    files = [_plane_tiff(tmp_path / f"fov{i}.tif", seed=i) for i in range(2)]
    entries = [{"czi_path": str(f), "condition": "WT", "reporter": "GFP"} for f in files]
    ctx = BulkRunContext(
        czi_entries=entries, output_dir=tmp_path / "out",
        do_split=True, do_focus=False, do_segment=True, do_classify=False,
    )

    segmented: list[Path] = []

    def fake_segment(*, tiff_path, out_path, **_kw):
        segmented.append(tiff_path)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_bytes(b"x")

    monkeypatch.setattr(stages, "segment_tiff", fake_segment)

    planned, failures = _run_bulk(ctx)

    assert not failures, failures
    assert planned == [["Split", "Segment"]]
    stem = sample_stem(entries[0])
    assert segmented == [ctx.split_dir / f"{stem}.tif"]
    assert tifffile.imread(str(segmented[0])).shape == (2, 2, 32, 40)
    assert not ctx.focus_dir.exists()


def test_bulk_focus_subsumes_split(tmp_path: Path, monkeypatch):
    pytest.importorskip("PyQt6")
    import mycomorph.core.api as api
    from mycomorph.gui.pipeline.context import BulkRunContext

    f = _plane_tiff(tmp_path / "a.tif")
    ctx = BulkRunContext(
        czi_entries=[{"czi_path": str(f), "condition": "WT"}],
        output_dir=tmp_path / "out",
        do_split=True, do_focus=True, do_segment=False, do_classify=False,
    )
    calls: list[str] = []
    monkeypatch.setattr(api, "run_focus_sample", lambda *a, **k: calls.append("focus"))
    monkeypatch.setattr(api, "split_sample", lambda *a, **k: calls.append("split"))

    planned, failures = _run_bulk(ctx)

    assert not failures, failures
    assert planned == [["Focus"]]
    assert calls == ["focus"]
