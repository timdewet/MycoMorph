"""Crop-extraction checks: background masking and Classify pre-filtering
(review H5)."""

from __future__ import annotations

import json

from mycomorph.modelprep.core.crop_provenance import crop_options, crop_warnings


def _manifest(path, mask_background, classify=True):
    stages = [{"name": "Focus", "opts": {}}, {"name": "Segment", "opts": {}}]
    if classify:
        stages.append({"name": "Classify", "opts": {}})
    stages.append({"name": "Features", "opts": {"mask_background": mask_background,
                                                "crop_size": 128}})
    path.write_text(json.dumps({"stages": stages}))


def test_no_record_means_unknown_and_no_warning(synthetic_h5):
    assert crop_options(synthetic_h5).get("mask_background") is None
    assert crop_warnings(synthetic_h5) == []


def test_manifest_two_levels_up(synthetic_h5, tmp_path):
    # <run>/04_features/all_crops.h5 with <run>/run_manifest.json
    run = tmp_path / "run"
    (run / "04_features").mkdir(parents=True)
    h5 = run / "04_features" / "all_crops.h5"
    h5.write_bytes(synthetic_h5.read_bytes())
    _manifest(run / "run_manifest.json", mask_background=True)

    info = crop_options(h5)
    assert info["mask_background"] is True and info["classify_before_features"]
    warnings = crop_warnings(h5)
    assert len(warnings) == 2
    assert "masking ON" in warnings[0] and "Classify ran before Features" in warnings[1]

    _manifest(run / "run_manifest.json", mask_background=False, classify=False)
    assert crop_warnings(h5) == []


def test_h5_attribute_wins_over_manifest(synthetic_h5):
    import h5py

    _manifest(synthetic_h5.parent / "run_manifest.json", mask_background=False, classify=False)
    with h5py.File(synthetic_h5, "a") as f:
        f.attrs["crop_options"] = json.dumps({"mask_background": True})
    info = crop_options(synthetic_h5)
    assert info["mask_background"] is True and "HDF5" in info["source"]

    with h5py.File(synthetic_h5, "a") as f:
        del f.attrs["crop_options"]
        f.attrs["mask_background"] = False
    assert crop_options(synthetic_h5)["mask_background"] is False


def test_classifier_prefilter_attr_overrides_manifest(synthetic_h5):
    """MycoMorph records classifier_prefilter on the HDF5; a known value wins
    over run_manifest.json, and 'unknown' falls back to it."""
    import h5py

    _manifest(synthetic_h5.parent / "run_manifest.json", mask_background=False, classify=True)
    with h5py.File(synthetic_h5, "a") as f:
        f.attrs["classifier_prefilter"] = "none"     # Classify ran but filtered nothing
    assert crop_warnings(synthetic_h5) == []

    with h5py.File(synthetic_h5, "a") as f:
        f.attrs["classifier_prefilter"] = "cnn"
        f.attrs["classifier_prefilter_model_sha256"] = "ab" * 32
    (w,) = crop_warnings(synthetic_h5)
    assert "CNN removed cells" in w and "abababababab" in w

    with h5py.File(synthetic_h5, "a") as f:
        f.attrs["classifier_prefilter"] = "unknown"
    (w,) = crop_warnings(synthetic_h5)
    assert "Classify ran before Features" in w        # manifest fallback
