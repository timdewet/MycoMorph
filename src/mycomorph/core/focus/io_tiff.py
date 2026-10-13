"""Read TIFF / OME-TIFF / ImageJ-hyperstack inputs as focus-pipeline scenes.

Mirrors the reading API of :mod:`io_czi` (``list_scene_indices``,
``list_scene_wells``, ``read_scene``) so the Focus stage can consume a
TIFF exactly the way it consumes a CZI: a list of scenes (fields of
view), each a ``(Z, C, Y, X)`` array.

Axis interpretation follows the labels ``tifffile`` derives from the
file's OME / ImageJ metadata:

- ``Y`` / ``X``                  → image plane
- ``Z``                          → focal planes (collapsed by Focus)
- ``C`` / ``S`` (RGB samples)    → channels
- ``T`` and position-like axes   → separate fields of view (scenes)
- unlabelled axes (``I`` / ``Q``) → channels — a plain multi-page TIFF
  with no metadata is most often a channel stack

Multi-series files (e.g. OME-TIFF with one series per stage position)
contribute one or more scenes per series. Series whose shape differs
from the first (thumbnails, previews) are ignored.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Iterator
from xml.etree import ElementTree as ET

import numpy as np
import tifffile

from .io_czi import Scene

_CHANNEL_AXES = frozenset("CS")
_UNLABELLED_AXES = frozenset("IQ")
_Z_AXES = frozenset("Z")
_PLANE_AXES = frozenset("YX")


@dataclass(frozen=True)
class _SeriesLayout:
    series_index: int
    axes: str
    shape: tuple[int, ...]
    n_scenes: int


@dataclass(frozen=True)
class TiffLayout:
    """Cheap, pixel-free description of a TIFF input."""

    series: tuple[_SeriesLayout, ...]
    n_z: int
    n_channels: int
    channel_names: list[str]
    pixel_size_um: tuple[float | None, float | None]
    acquisition_times: tuple[str | None, ...]  # one per series

    @property
    def n_scenes(self) -> int:
        return sum(s.n_scenes for s in self.series)

    def locate(self, scene_index: int) -> tuple[_SeriesLayout, int]:
        """Map a global scene index to ``(series, scene index within series)``."""
        if scene_index < 0:
            raise IndexError(f"scene index {scene_index} out of range")
        remaining = scene_index
        for s in self.series:
            if remaining < s.n_scenes:
                return s, remaining
            remaining -= s.n_scenes
        raise IndexError(
            f"scene index {scene_index} out of range (file has {self.n_scenes} scenes)"
        )


def _axis_role(axis: str) -> str:
    if axis in _PLANE_AXES:
        return "plane"
    if axis in _Z_AXES:
        return "z"
    if axis in _CHANNEL_AXES or axis in _UNLABELLED_AXES:
        return "channel"
    return "scene"


def _role_sizes(axes: str, shape: tuple[int, ...]) -> dict[str, int]:
    sizes = {"scene": 1, "z": 1, "channel": 1}
    for ax, n in zip(axes, shape):
        role = _axis_role(ax)
        if role != "plane":
            sizes[role] *= int(n)
    return sizes


def _to_szcyx(arr: np.ndarray, axes: str) -> np.ndarray:
    """Reorder an array with ``tifffile`` axes into ``(scene, Z, C, Y, X)``."""
    if arr.ndim != len(axes):
        raise ValueError(f"array has {arr.ndim} dims but axes are {axes!r}")
    if axes.count("Y") != 1 or axes.count("X") != 1:
        raise ValueError(f"TIFF series axes {axes!r} lack a single Y and X axis")
    order: list[int] = []
    for role in ("scene", "z", "channel"):
        order.extend(i for i, ax in enumerate(axes) if _axis_role(ax) == role)
    order.extend([axes.index("Y"), axes.index("X")])
    moved = np.transpose(arr, order)
    sizes = _role_sizes(axes, arr.shape)
    y, x = arr.shape[axes.index("Y")], arr.shape[axes.index("X")]
    return moved.reshape(sizes["scene"], sizes["z"], sizes["channel"], y, x)


def _ome_root(tif: tifffile.TiffFile) -> ET.Element | None:
    xml = tif.ome_metadata
    if not xml:
        return None
    try:
        return ET.fromstring(xml)
    except ET.ParseError:
        return None


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _ome_images(root: ET.Element) -> list[ET.Element]:
    return [el for el in root.iter() if _local(el.tag) == "Image"]


def _ome_channel_names(root: ET.Element) -> list[str]:
    images = _ome_images(root)
    if not images:
        return []
    names: list[str] = []
    for el in images[0].iter():
        if _local(el.tag) == "Channel":
            names.append(str(el.attrib.get("Name") or "").strip())
    return names


def _ome_acquisition_dates(root: ET.Element) -> list[str | None]:
    out: list[str | None] = []
    for img in _ome_images(root):
        date = None
        for el in img:
            if _local(el.tag) == "AcquisitionDate" and el.text:
                date = el.text.strip()
                break
        out.append(date)
    return out


def _tiff_datetime(tif: tifffile.TiffFile) -> str | None:
    """Baseline TIFF ``DateTime`` tag (``YYYY:MM:DD HH:MM:SS``) as ISO 8601."""
    try:
        tag = tif.pages[0].tags.get("DateTime")
    except Exception:  # noqa: BLE001
        return None
    if tag is None:
        return None
    text = str(tag.value).strip()
    try:
        return datetime.strptime(text, "%Y:%m:%d %H:%M:%S").isoformat()
    except ValueError:
        return text or None


def _resolve_channel_names(tif: tifffile.TiffFile, n_channels: int) -> list[str]:
    ij = tif.imagej_metadata or {}
    labels = ij.get("Labels")
    if isinstance(labels, (list, tuple)) and len(labels) >= n_channels:
        # ImageJ stores one label per plane; for channel-only stacks the
        # first ``n_channels`` are the channel labels.
        names = [str(x).strip() for x in labels[:n_channels]]
        if all(names):
            return names
    root = _ome_root(tif)
    if root is not None:
        names = _ome_channel_names(root)
        if len(names) == n_channels and all(names):
            return names
    return [f"C{i}" for i in range(n_channels)]


def _pixel_size_um(path: Path) -> tuple[float | None, float | None]:
    from ..api import _read_pixels_per_um

    px_per_um = _read_pixels_per_um(path)
    if not px_per_um:
        return (None, None)
    um = 1.0 / px_per_um
    return (um, um)


def _cache_key(path: Path) -> tuple[str, int]:
    path = Path(path)
    return (str(path.resolve()), path.stat().st_mtime_ns)


@lru_cache(maxsize=32)
def _layout_cached(path_str: str, _mtime_ns: int) -> TiffLayout:
    path = Path(path_str)
    with tifffile.TiffFile(path_str) as tif:
        if not tif.series:
            raise ValueError(f"{path.name}: no image data found")
        first = tif.series[0]
        ref_axes, ref_shape = first.axes, tuple(first.shape)
        series: list[_SeriesLayout] = []
        for i, s in enumerate(tif.series):
            if s.axes != ref_axes or tuple(s.shape) != ref_shape:
                continue
            sizes = _role_sizes(s.axes, tuple(s.shape))
            series.append(_SeriesLayout(i, s.axes, tuple(s.shape), sizes["scene"]))
        sizes = _role_sizes(ref_axes, ref_shape)
        channel_names = _resolve_channel_names(tif, sizes["channel"])

        root = _ome_root(tif)
        dates = _ome_acquisition_dates(root) if root is not None else []
        fallback = _tiff_datetime(tif)
        acq = tuple(
            (dates[s.series_index] if s.series_index < len(dates) else None) or fallback
            for s in series
        )
    return TiffLayout(
        series=tuple(series),
        n_z=sizes["z"],
        n_channels=sizes["channel"],
        channel_names=channel_names,
        pixel_size_um=_pixel_size_um(path),
        acquisition_times=acq,
    )


def read_layout(path: Path) -> TiffLayout:
    """Describe ``path`` (scenes, Z, channels, metadata) without reading pixels."""
    return _layout_cached(*_cache_key(path))


@lru_cache(maxsize=2)
def _series_szcyx(path_str: str, _mtime_ns: int, series_index: int) -> np.ndarray:
    with tifffile.TiffFile(path_str) as tif:
        s = tif.series[series_index]
        arr = s.asarray()
        axes = s.axes
    out = _to_szcyx(arr, axes)
    out.setflags(write=False)
    return out


def list_scene_indices(path: Path) -> list[int]:
    return list(range(read_layout(path).n_scenes))


def list_scene_wells(path: Path) -> dict[int, str]:
    """TIFFs carry no plate-well metadata; every scene is unassigned."""
    return {}


def read_scene(path: Path, scene_index: int) -> Scene:
    layout = read_layout(path)
    series, local = layout.locate(int(scene_index))
    key = _cache_key(path)
    stack = _series_szcyx(key[0], key[1], series.series_index)
    pos = layout.series.index(series)
    return Scene(
        index=int(scene_index),
        array_zcyx=np.array(stack[local]),
        channel_names=list(layout.channel_names),
        pixel_size_um=layout.pixel_size_um,
        acquisition_time=layout.acquisition_times[pos],
    )


def iter_scenes(path: Path) -> Iterator[Scene]:
    for idx in list_scene_indices(path):
        yield read_scene(path, idx)
