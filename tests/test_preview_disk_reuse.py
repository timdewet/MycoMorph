"""Live-preview reuse of the pipeline's on-disk segmentation output.

The preview skips cellpose when a previous run already segmented the same
FOV at the same options. Soundness rests entirely on the provenance
signature matching, so these tests pin both directions: a matching
signature reuses, and every way the inputs can diverge falls back to a
live run.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mycomorph.core.api import SegmentOpts
from mycomorph.core.cellpose_pipeline import save_hyperstack
from mycomorph.core.provenance import checkpoint_signature, write_artifact_signature
from mycomorph.gui.widgets.live_preview.worker import PreviewWorker, RenderRequest


CHANNEL_LABELS = ["Phase", "GFP"]
PHASE_CHANNEL = 0
N_FOV = 3


def _write_focus_tiff(path: Path) -> None:
    """A stand-in for 01_split_and_focused output: (N_FOV, C, Y, X)."""
    data = np.zeros((N_FOV, len(CHANNEL_LABELS), 40, 50), dtype=np.uint16)
    # Vary content per FOV so a wrong-FOV read would be visible.
    for f in range(N_FOV):
        data[f, 0, f * 5 : f * 5 + 4, 0:4] = 1000
    save_hyperstack(
        stacked=data,
        output_path=path,
        condition_name=path.stem,
        filenames=[f"fov_{i:03d}" for i in range(N_FOV)],
        channel_labels=CHANNEL_LABELS,
    )


def _mask_for_fov(fov: int) -> np.ndarray:
    """Binary 0/255 mask with ``fov + 1`` well-separated blobs.

    Binary (not labelled) matches what SegmentStage actually writes, so
    the worker's re-labelling is exercised.
    """
    mask = np.zeros((40, 50), dtype=np.uint16)
    for i in range(fov + 1):
        mask[5 + i * 10 : 10 + i * 10, 5:12] = 255
    return mask


def _write_segment_output(path: Path, opts: SegmentOpts, source: Path) -> None:
    """Write a 02_segment artifact plus the sidecar SegmentStage would."""
    stacked = np.zeros((N_FOV, len(CHANNEL_LABELS) + 1, 40, 50), dtype=np.uint16)
    for f in range(N_FOV):
        stacked[f, -1] = _mask_for_fov(f)
    save_hyperstack(
        stacked=stacked,
        output_path=path,
        condition_name=source.stem,
        filenames=[f"fov_{i:03d}" for i in range(N_FOV)],
        channel_labels=CHANNEL_LABELS + ["BinaryMask"],
    )
    write_artifact_signature(
        path,
        checkpoint_signature(
            stage="segment",
            input_path=source,
            options=opts,
            channels=CHANNEL_LABELS,
            phase_channel=PHASE_CHANNEL,
        ),
    )


@pytest.fixture
def project(tmp_path: Path):
    """A focused TIFF plus a matching segmented artifact, as a run leaves them."""
    focus_dir = tmp_path / "01_split_and_focused"
    segment_dir = tmp_path / "02_segment"
    focus_dir.mkdir()
    segment_dir.mkdir()

    opts = SegmentOpts(diameter=30.0, min_size=15)
    sample = focus_dir / "condA__gfp__wt__R1_focused.tif"
    _write_focus_tiff(sample)
    _write_segment_output(segment_dir / sample.name, opts, sample)
    return sample, segment_dir, opts


def _request(sample: Path, segment_dir: Path | None, opts: SegmentOpts,
             **overrides) -> RenderRequest:
    kwargs = dict(
        sample_path=sample,
        fov_index=0,
        target_stage="segment",
        phase_channel=PHASE_CHANNEL,
        segment_opts=opts,
        channel_labels=list(CHANNEL_LABELS),
        segment_dir=segment_dir,
        use_disk_focus=True,
    )
    kwargs.update(overrides)
    return RenderRequest(**kwargs)


def _try(sample, segment_dir, opts, **overrides):
    worker = PreviewWorker(_request(sample, segment_dir, opts, **overrides))
    return worker._try_disk_segment(("segment", "key"))


# ── Reuse happens when the signature matches ────────────────────────────────

@pytest.mark.parametrize("fov,expected_cells", [(0, 1), (1, 2), (2, 3)])
def test_matching_signature_reuses_the_stored_mask(project, fov, expected_cells):
    sample, segment_dir, opts = project
    payload = _try(sample, segment_dir, opts, fov_index=fov)

    assert payload is not None
    assert payload.reused is True
    assert payload.key == ("segment", "key")
    # Binary 0/255 on disk must come back as unique integer cell IDs.
    assert payload.n_cells == expected_cells
    assert sorted(np.unique(payload.mask)) == list(range(expected_cells + 1))


def test_reused_mask_matches_the_requested_fov(project):
    sample, segment_dir, opts = project
    payload = _try(sample, segment_dir, opts, fov_index=2)

    assert payload is not None
    # FOV 2's third blob sits at rows 25..30 — absent from FOVs 0 and 1.
    assert payload.mask[25:30, 5:12].min() > 0


# ── Every divergence falls back to a live run ───────────────────────────────

def test_changed_options_do_not_reuse(project):
    sample, segment_dir, _opts = project
    changed = SegmentOpts(diameter=45.0, min_size=15)
    assert _try(sample, segment_dir, changed) is None


def test_roi_does_not_reuse(project):
    """A crop is scored differently from the full frame the pipeline saw."""
    sample, segment_dir, opts = project
    assert _try(sample, segment_dir, opts, roi=(0, 0, 20, 20)) is None


def test_raw_czi_preview_does_not_reuse(project):
    """In-memory focus was never written to disk, so nothing can match it."""
    sample, segment_dir, opts = project
    assert _try(sample, segment_dir, opts, use_disk_focus=False) is None


def test_no_segment_dir_does_not_reuse(project):
    sample, _segment_dir, opts = project
    assert _try(sample, None, opts) is None


def test_different_phase_channel_does_not_reuse(project):
    sample, segment_dir, opts = project
    assert _try(sample, segment_dir, opts, phase_channel=1) is None


def test_unresolved_phase_channel_does_not_reuse(project):
    """The 'auto' sentinel can't be signed — resolve it or segment live."""
    sample, segment_dir, opts = project
    assert _try(sample, segment_dir, opts, phase_channel=None) is None


def test_renamed_channels_do_not_reuse(project):
    sample, segment_dir, opts = project
    assert _try(sample, segment_dir, opts, channel_labels=["Phase", "RFP"]) is None


def test_missing_artifact_does_not_reuse(project):
    sample, segment_dir, opts = project
    (segment_dir / sample.name).unlink()
    assert _try(sample, segment_dir, opts) is None


def test_unsigned_legacy_artifact_does_not_reuse(project):
    """No sidecar means unknown provenance — don't vouch for it."""
    sample, segment_dir, opts = project
    from mycomorph.core.provenance import artifact_signature_path

    artifact_signature_path(segment_dir / sample.name).unlink()
    assert _try(sample, segment_dir, opts) is None


def test_edited_input_file_does_not_reuse(project):
    """The signature fingerprints the input, so re-focusing invalidates it."""
    sample, segment_dir, opts = project
    _write_focus_tiff(sample)  # rewrites → new mtime/size
    assert _try(sample, segment_dir, opts) is None


def test_in_progress_artifact_does_not_reuse(project):
    """A run is mid-write; the file on disk may be half-finished."""
    sample, segment_dir, opts = project
    from mycomorph.core.provenance import mark_artifact_in_progress

    mark_artifact_in_progress(segment_dir / sample.name)
    assert _try(sample, segment_dir, opts) is None


def test_fov_beyond_the_artifact_does_not_reuse(project):
    sample, segment_dir, opts = project
    assert _try(sample, segment_dir, opts, fov_index=N_FOV) is None


def test_corrupt_artifact_does_not_reuse(project):
    """Reuse is an optimisation — a bad read must never surface as an error."""
    sample, segment_dir, opts = project
    (segment_dir / sample.name).write_bytes(b"not a tiff")
    assert _try(sample, segment_dir, opts) is None
