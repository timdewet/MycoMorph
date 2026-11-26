"""Focus options: metric names shared by GUI and core, and the two focus modes.

The Focus page once offered ``laplacian`` and ``vollath`` while the core's
score keys are ``laplacian_variance`` and ``vollath_f4``, so picking either
with whole-field focus raised ``KeyError`` mid-run. These tests pin that the
GUI lists the core's own names, that the old spellings still load, and that
the removed tiled mode fails with a message that says what to use instead.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import tifffile
from scipy.ndimage import gaussian_filter

from mycomorph.core.api import FocusOpts, run_focus_sample
from mycomorph.core.focus import focus

SHARP_Z = 3
N_Z = 6


def _zstack(sharp_z: int = SHARP_Z, n_z: int = N_Z, seed: int = 7) -> np.ndarray:
    """(Z, Y, X) phase-like stack: blocky texture, blurred more away from ``sharp_z``."""
    rng = np.random.default_rng(seed)
    blocks = rng.integers(0, 2, size=(16, 16)).astype(np.float64)
    sharp = 500.0 + 3000.0 * np.kron(blocks, np.ones((4, 4)))
    planes = [gaussian_filter(sharp, sigma=1.5 * abs(z - sharp_z)) for z in range(n_z)]
    return np.stack(planes).astype(np.uint16)


def _zstack_tiff(path: Path) -> Path:
    """Two-channel ZCYX TIFF whose phase channel (0) is sharpest at ``SHARP_Z``."""
    phase = _zstack()
    fluor = np.random.default_rng(1).integers(0, 500, size=phase.shape, dtype=np.uint16)
    arr = np.stack([phase, fluor], axis=1)
    tifffile.imwrite(str(path), arr, imagej=True, metadata={"axes": "ZCYX"})
    return path


def _focused_phase(work_dir: Path, opts: FocusOpts) -> np.ndarray:
    """Focus the synthetic stack through the API and return its phase plane."""
    work_dir.mkdir(parents=True, exist_ok=True)
    src = _zstack_tiff(work_dir / "stack.tif")
    out = run_focus_sample([src], work_dir / "out", "S", opts=opts)
    written = tifffile.imread(str(out))          # one FOV: (C, Y, X)
    assert written.shape == (2, 64, 64)
    return written[0]


# ───────────────────────────────────────────────────────────── metric names


@pytest.mark.parametrize("old_name", ["laplacian", "vollath"])
def test_metric_names_the_gui_used_to_offer_run_in_whole_mode(tmp_path: Path, old_name: str):
    """Reproduces the bug: these two names raised ``KeyError`` in whole mode."""
    phase = _focused_phase(
        tmp_path, FocusOpts(mode="whole", metric=old_name, phase_channel=0)
    )
    np.testing.assert_array_equal(phase, _zstack()[SHARP_Z])


def test_every_metric_the_gui_offers_is_accepted_by_the_core(qtbot):
    pytest.importorskip("PyQt6")
    from mycomorph.gui.panels.stage_panels import FocusPanel

    panel = FocusPanel()
    qtbot.addWidget(panel)
    offered = [panel.metric.itemText(i) for i in range(panel.metric.count())]

    scores = focus.score_stack(_zstack())
    assert sorted(offered) == sorted(scores), "GUI list and core score keys differ"
    for name in offered:
        assert focus.pick_best_slice(scores, metric=name) == SHARP_Z, name


def test_saved_settings_with_the_old_metric_names_still_load(qtbot):
    pytest.importorskip("PyQt6")
    from mycomorph.gui.panels.stage_panels import FocusPanel

    panel = FocusPanel()
    qtbot.addWidget(panel)
    for saved, expected in (("laplacian", "laplacian_variance"), ("vollath", "vollath_f4")):
        panel.restore_state({"mode": "whole", "metric": saved})
        assert panel.opts().metric == expected
        assert panel.state()["metric"] == expected


def test_unknown_metric_fails_before_any_image_is_read():
    from mycomorph.core.focus.pipeline import Options

    with pytest.raises(ValueError, match="unknown focus metric 'sharpness'"):
        Options(metric="sharpness")
    assert Options(metric="vollath").metric == "vollath_f4"


def test_preview_accepts_the_same_metric_names(tmp_path: Path):
    from mycomorph.gui.widgets.live_preview import focus_single_fov

    src = _zstack_tiff(tmp_path / "stack.tif")
    for name in ("laplacian_variance", "vollath"):
        planes, _names, phase_idx = focus_single_fov.run(
            src, 0, FocusOpts(mode="whole", metric=name), 0
        )
        np.testing.assert_array_equal(planes[phase_idx], _zstack()[SHARP_Z])


def test_ensemble_is_the_unsmoothed_mean_of_minmax_normalised_curves():
    """Pins the arithmetic the ``focus`` module docstring describes."""
    scores = focus.score_stack(_zstack())
    raw = [scores[m] for m in focus.METRIC_NAMES if m != "ensemble"]
    assert len(raw) == 6
    expected = np.mean([(r - r.min()) / (r.max() - r.min()) for r in raw], axis=0)
    np.testing.assert_allclose(scores["ensemble"], expected, rtol=0, atol=1e-12)


# ───────────────────────────────────────────────────────────── focus modes


def test_the_gui_offers_two_modes_and_the_core_accepts_both(qtbot):
    pytest.importorskip("PyQt6")
    from mycomorph.core.focus.pipeline import Options
    from mycomorph.gui.panels.stage_panels import FocusPanel

    panel = FocusPanel()
    qtbot.addWidget(panel)
    modes = [panel.mode.itemText(i) for i in range(panel.mode.count())]

    assert modes == ["edf", "whole"]
    assert panel.opts().mode == "edf"
    for mode in modes:
        assert Options(focus_mode=mode).focus_mode == mode
    assert "tile" not in " ".join(panel.state()).lower()
    assert not hasattr(panel, "tile_rows")


def test_both_modes_still_focus_a_stack(tmp_path: Path):
    stack = _zstack()
    whole = _focused_phase(tmp_path / "whole", FocusOpts(mode="whole", phase_channel=0))
    np.testing.assert_array_equal(whole, stack[SHARP_Z])

    edf = _focused_phase(tmp_path / "edf", FocusOpts(mode="edf", phase_channel=0))
    z_map = focus.pick_per_pixel_z(stack)
    np.testing.assert_array_equal(edf, focus.assemble_edf_plane(stack, z_map))


def test_tiles_mode_fails_with_a_message_that_names_the_alternatives(tmp_path: Path):
    from mycomorph.core.focus.pipeline import Options

    with pytest.raises(ValueError, match="'tiles'.*removed.*'edf'.*'whole'"):
        Options(focus_mode="tiles")

    src = _zstack_tiff(tmp_path / "stack.tif")
    with pytest.raises(ValueError, match="'tiles'.*removed"):
        run_focus_sample([src], tmp_path / "out", "S", opts=FocusOpts(mode="tiles"))
    assert not (tmp_path / "out" / "S_focused.tif").exists()


def test_focus_opts_no_longer_takes_a_tile_grid():
    from dataclasses import fields

    from mycomorph.core.focus.pipeline import Options

    assert "tile_grid" not in {f.name for f in fields(FocusOpts)}
    assert "tile_grid" not in {f.name for f in fields(Options)}
    with pytest.raises(TypeError):
        FocusOpts(tile_grid=(3, 3))


def test_saved_tiles_setting_falls_back_to_edf_and_says_so(qtbot):
    pytest.importorskip("PyQt6")
    from mycomorph.gui.panels.stage_panels import FocusPanel

    panel = FocusPanel()
    qtbot.addWidget(panel)
    assert panel._mode_notice.isHidden()

    panel.mode.setCurrentText("whole")
    panel.restore_state({"mode": "tiles", "tile_rows": 2, "tile_cols": 4,
                         "metric": "ensemble"})
    assert panel.opts().mode == "edf"
    assert not panel._mode_notice.isHidden()
    assert "removed" in panel._mode_notice.text()

    # Choosing a mode by hand dismisses the notice.
    panel.mode.setCurrentText("whole")
    assert panel._mode_notice.isHidden()


def test_cli_rejects_tiles_and_no_longer_documents_it(tmp_path: Path):
    from typer.testing import CliRunner

    from mycomorph.core.focus.cli import app

    runner = CliRunner()
    src = _zstack_tiff(tmp_path / "stack.tif")

    result = runner.invoke(app, ["process", str(src), "--focus-mode", "tiles", "--no-archive"])
    assert result.exit_code == 2
    assert "'tiles'" in result.output and "removed" in result.output
    assert sorted(p.name for p in tmp_path.iterdir()) == ["stack.tif"]

    assert runner.invoke(app, ["process", str(src), "--tile-grid", "2x2"]).exit_code == 2
    assert "tile" not in runner.invoke(app, ["process", "--help"]).output.lower()
