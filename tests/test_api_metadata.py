"""TIFF metadata helpers exposed by ``mycomorph.core.api``.

Covers ``_read_pixels_per_um`` (ImageJ ``spacing`` and TIFF
``XResolution``/``ResolutionUnit`` paths) and ``_read_imagej_labels``.
These feed downstream area filters and channel-name displays, so a silent
unit-conversion bug here propagates everywhere.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import tifffile

from mycomorph.core.api import _read_imagej_labels, _read_pixels_per_um


def _write_imagej_tiff(
    path: Path,
    *,
    spacing: float | None = None,
    labels: list[str] | None = None,
    resolution: tuple[float, float] | None = None,
    resolutionunit: str | None = None,
) -> Path:
    """Write a small ImageJ-style hyperstack with optional metadata fields."""
    data = np.zeros((2, 8, 8), dtype=np.uint16)  # (C, Y, X)
    meta: dict = {"axes": "CYX"}
    if spacing is not None:
        meta["spacing"] = spacing
    if labels is not None:
        meta["Labels"] = labels
    kwargs: dict = {"imagej": True, "metadata": meta}
    if resolution is not None:
        kwargs["resolution"] = resolution
    if resolutionunit is not None:
        kwargs["resolutionunit"] = resolutionunit
    tifffile.imwrite(str(path), data, **kwargs)
    return path


class TestReadPixelsPerUm:
    def test_imagej_spacing_is_not_xy_calibration(self, tmp_path):
        # ImageJ spacing is the Z interval and must not be interpreted as XY.
        path = _write_imagej_tiff(tmp_path / "spacing.tif", spacing=0.072)
        assert _read_pixels_per_um(path) is None

    def test_xresolution_cm_path(self, tmp_path):
        # XResolution = 138767 px/cm ⇒ 13.8767 px/µm.
        path = _write_imagej_tiff(
            tmp_path / "cm.tif",
            resolution=(138_767.0, 138_767.0),
            resolutionunit="CENTIMETER",
        )
        result = _read_pixels_per_um(path)
        assert result == pytest.approx(13.8767, rel=1e-3)

    def test_xresolution_inch_path(self, tmp_path):
        # XResolution = 25_400 px/inch ⇒ 1 px/µm.
        path = _write_imagej_tiff(
            tmp_path / "inch.tif",
            resolution=(25_400.0, 25_400.0),
            resolutionunit="INCH",
        )
        result = _read_pixels_per_um(path)
        assert result == pytest.approx(1.0, rel=1e-3)

    @pytest.mark.parametrize("unit, expected", [
        ("micron", 15.0), ("um", 15.0), ("\\u00B5m", 15.0), ("nm", 15_000.0),
    ])
    def test_fiji_calibration_without_tiff_unit(self, tmp_path, unit, expected):
        # Fiji writes XResolution in px per ImageJ ``unit`` with
        # ResolutionUnit = NONE.
        path = tmp_path / "fiji.tif"
        tifffile.imwrite(
            str(path), np.zeros((2, 8, 8), dtype=np.uint16), imagej=True,
            metadata={"axes": "CYX", "unit": unit},
            resolution=(15.0, 15.0), resolutionunit="NONE",
        )
        assert _read_pixels_per_um(path) == pytest.approx(expected)

    def test_unitless_resolution_without_imagej_unit_returns_none(self, tmp_path):
        path = tmp_path / "unitless.tif"
        tifffile.imwrite(
            str(path), np.zeros((8, 8), dtype=np.uint16),
            resolution=(15.0, 15.0), resolutionunit="NONE",
        )
        assert _read_pixels_per_um(path) is None

    def test_missing_metadata_returns_none(self, tmp_path):
        # Plain TIFF, no ImageJ metadata at all.
        path = tmp_path / "plain.tif"
        tifffile.imwrite(str(path), np.zeros((4, 4), dtype=np.uint16))
        assert _read_pixels_per_um(path) is None

    def test_nonexistent_file_returns_none(self, tmp_path):
        assert _read_pixels_per_um(tmp_path / "does_not_exist.tif") is None


class TestReadImagejLabels:
    def test_returns_labels_list_when_present(self, tmp_path):
        path = _write_imagej_tiff(
            tmp_path / "labels.tif",
            labels=["Phase", "GFP"],
        )
        assert _read_imagej_labels(path) == ["Phase", "GFP"]

    def test_returns_none_when_labels_missing(self, tmp_path):
        path = _write_imagej_tiff(tmp_path / "no_labels.tif")
        assert _read_imagej_labels(path) is None

    def test_returns_none_for_nonexistent_file(self, tmp_path):
        assert _read_imagej_labels(tmp_path / "missing.tif") is None
