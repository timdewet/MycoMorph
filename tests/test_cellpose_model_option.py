"""The Cellpose model option: cpsam by default, cpsam_v2 opt-in with a warning.

Cellpose 4.2 switched its own default weights to cpsam_v2, so MycoMorph names
the model explicitly. These tests pin that the default stays cpsam, that
outputs signed before the option existed remain reusable, and that choosing
cpsam_v2 warns about the classifiers being trained on cpsam masks.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from mycomorph.core import api
from mycomorph.core.cellpose_pipeline import save_hyperstack
from mycomorph.core.provenance import (
    artifact_signature_matches,
    checkpoint_signature,
    write_artifact_signature,
)
from mycomorph.gui.widgets.live_preview.cache import segment_key


def _source(tmp_path: Path) -> Path:
    path = tmp_path / "focused.tif"
    path.write_bytes(b"stand-in input")
    return path


def _signature(source: Path, options) -> dict:
    return checkpoint_signature(
        stage="segment", input_path=source, options=options,
        channels=["Phase"], phase_channel=0,
    )


def test_default_model_is_cpsam():
    assert api.SegmentOpts().model == "cpsam"
    assert api.DEFAULT_CELLPOSE_MODEL == api.CLASSIFIER_TRAINING_CELLPOSE_MODEL == "cpsam"
    assert set(api.CELLPOSE_MODELS) == {"cpsam", "cpsam_v2"}


def test_cpsam_signature_matches_outputs_signed_before_the_option(tmp_path: Path):
    """A pre-option sidecar (options without ``model``) must stay reusable."""
    source = _source(tmp_path)
    legacy_options = {k: v for k, v in asdict(api.SegmentOpts()).items() if k != "model"}
    output = tmp_path / "segmented.tif"
    output.write_bytes(b"x")
    write_artifact_signature(output, _signature(source, legacy_options))

    assert artifact_signature_matches(output, _signature(source, api.SegmentOpts())) is True
    assert artifact_signature_matches(
        output, _signature(source, api.SegmentOpts(model="cpsam_v2"))
    ) is False


def test_preview_cache_key_depends_on_model():
    focus_k = ("focus", "x.tif", 0)
    assert segment_key(focus_k, api.SegmentOpts(), 0) != segment_key(
        focus_k, api.SegmentOpts(model="cpsam_v2"), 0
    )


def test_segment_tiff_rejects_unknown_model(tmp_path: Path):
    with pytest.raises(ValueError, match="Unknown Cellpose model"):
        api.segment_tiff(
            _source(tmp_path), tmp_path / "out.tif", phase_channel=0,
            opts=api.SegmentOpts(model="cyto3"),
        )


def test_segmentation_model_of(tmp_path: Path):
    output = tmp_path / "segmented.tif"
    output.write_bytes(b"x")
    assert api.segmentation_model_of(output) is None  # unsigned legacy output

    signature = _signature(_source(tmp_path), api.SegmentOpts(model="cpsam_v2"))
    signature["segmentation"] = {"cellpose_model": "cpsam_v2", "cellpose_version": "4.2.1.1"}
    write_artifact_signature(output, signature)
    assert api.segmentation_model_of(output) == "cpsam_v2"


def _empty_segmented_tiff(path: Path, model: str) -> None:
    data = np.zeros((1, 2, 16, 16), dtype=np.uint16)
    save_hyperstack(data, path, "condition", ["fov_000"], ["Phase", "BinaryMask"])
    path.with_name(path.name + ".provenance.json").write_text(json.dumps({
        "digest": "x", "segmentation": {"cellpose_model": model},
    }))


def test_classify_warns_on_cpsam_v2_masks_with_a_cnn(tmp_path: Path):
    source = tmp_path / "segmented.tif"
    _empty_segmented_tiff(source, "cpsam_v2")
    messages: list[str] = []
    with pytest.warns(UserWarning, match="trained on cpsam masks"):
        api.classify_filter_tiff(
            source, tmp_path / "classified.tif", phase_channel=0,
            opts=api.ClassifyOpts(model_path=tmp_path / "model.pth"),
            progress_cb=lambda _f, m: messages.append(m),
        )
    assert any(m.startswith("WARNING") and "cpsam_v2" in m for m in messages)


@pytest.mark.parametrize("model, model_path", [("cpsam", "model.pth"), ("cpsam_v2", None)])
def test_classify_does_not_warn_otherwise(tmp_path: Path, recwarn, model, model_path):
    """cpsam masks, or rules-only filtering, need no warning."""
    source = tmp_path / "segmented.tif"
    _empty_segmented_tiff(source, model)
    api.classify_filter_tiff(
        source, tmp_path / "classified.tif", phase_channel=0,
        opts=api.ClassifyOpts(model_path=tmp_path / model_path if model_path else None),
    )
    assert not [w for w in recwarn if "trained on cpsam masks" in str(w.message)]
