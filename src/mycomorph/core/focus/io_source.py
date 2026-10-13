"""Format dispatch for raw image inputs (CZI or TIFF).

Callers that only need scenes and basic metadata go through here so a
CZI and a TIFF are interchangeable inputs. Plate-specific metadata
(Zeiss scene → well maps) remains CZI-only and lives in
``split_czi_plate``.
"""

from __future__ import annotations

from pathlib import Path

from . import io_czi, io_tiff
from .io_czi import Scene

CZI_SUFFIXES = (".czi",)
TIFF_SUFFIXES = (".tif", ".tiff")
SUPPORTED_SUFFIXES = CZI_SUFFIXES + TIFF_SUFFIXES

# Qt file-dialog filter offering every supported input format.
FILE_DIALOG_FILTER = (
    "Images (*.czi *.tif *.tiff);;CZI files (*.czi);;"
    "TIFF files (*.tif *.tiff);;All files (*)"
)


def is_tiff(path: Path) -> bool:
    return Path(path).suffix.lower() in TIFF_SUFFIXES


def is_supported(path: Path) -> bool:
    return Path(path).suffix.lower() in SUPPORTED_SUFFIXES


def _reader(path: Path):
    return io_tiff if is_tiff(path) else io_czi


def list_scene_indices(path: Path) -> list[int]:
    return _reader(path).list_scene_indices(Path(path))


def list_scene_wells(path: Path) -> dict[int, str]:
    return _reader(path).list_scene_wells(Path(path))


def read_scene(path: Path, scene_index: int) -> Scene:
    return _reader(path).read_scene(Path(path), scene_index)


def channel_names(path: Path) -> list[str]:
    """Channel names as stored in the file's metadata."""
    path = Path(path)
    if is_tiff(path):
        return list(io_tiff.read_layout(path).channel_names)
    from ..split_czi_plate import extract_channel_names
    return list(extract_channel_names(path) or [])


def pixels_per_um(path: Path) -> float | None:
    """Pixel calibration from the file's metadata, or None if absent."""
    path = Path(path)
    if is_tiff(path):
        from ..api import _read_pixels_per_um
        return _read_pixels_per_um(path)
    from ..split_czi_plate import _read_czi_pixels_per_um
    return _read_czi_pixels_per_um(path)


def per_scene_acquisition_times(path: Path) -> dict[int, str]:
    """Per-scene timestamps where the format records them separately.

    CZIs keep them in subblock metadata; TIFF scenes already carry
    their timestamp on the :class:`Scene` itself, so this is empty.
    """
    if is_tiff(path):
        return {}
    try:
        from ..split_czi_plate import _read_czi_per_scene_acquisition_times
        return _read_czi_per_scene_acquisition_times(Path(path))
    except Exception:  # noqa: BLE001
        return {}
