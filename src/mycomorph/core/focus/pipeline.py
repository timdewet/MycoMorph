"""Per-input orchestration: read, score, pick, write TIFF, log scores, archive.

Inputs are CZIs or TIFFs; both are read through :mod:`io_source`.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import tifffile

from . import archive, channel_id, focus, io_ometiff, io_source, tiling
from .io_czi import Scene

# "none" writes single-plane inputs through unchanged (no focus picking);
# it refuses Z-stacks rather than silently choosing a plane.
FocusMode = Literal["whole", "tiles", "edf", "none"]


@dataclass
class Options:
    phase_channel: str | int | None = None
    metric: str = focus.DEFAULT_METRIC
    # Move the source CZI into a ``raw/`` subfolder after a successful run.
    # Default OFF: the move breaks restorable session state (the input
    # panel stores absolute paths in QSettings; on reopen those paths are
    # gone) and complicates multi-CZI workflows where you add new CZIs to
    # a plate that's been partially processed already. The pipeline
    # already records "this CZI was processed" in run_manifest.json and
    # the existence of 01_split_and_focused/ outputs, so the move is just
    # cosmetic. Users who want the tidy-up can enable it explicitly.
    archive_original: bool = False
    crop_fraction: float = 1.0
    preblur_sigma: float = 0.0
    focus_mode: FocusMode = "whole"
    tile_grid: tuple[int, int] = (2, 2)
    # When True, each fluorescence channel gets a companion ``<name>_MIP``
    # max-intensity projection channel in addition to the focused slice.
    # Default is False so focus output has the same channel count as the CZI.
    save_mip: bool = False


@dataclass
class SceneResult:
    scene_index: int
    chosen_z: int
    output_path: Path
    scores_csv: Path | None
    well: str | None = None


@dataclass
class WellResult:
    well: str
    scene_indices: list[int]
    chosen_z: list[int]
    output_path: Path
    scores_csv: Path | None


@dataclass
class ProcessResult:
    scene_results: list[SceneResult] = field(default_factory=list)
    well_results: list[WellResult] = field(default_factory=list)


def _assemble_output_planes(
    array_zcyx: np.ndarray,
    channel_names: list[str],
    phase_idx: int,
    chosen_z: int,
    save_mip: bool = False,
) -> tuple[np.ndarray, list[str]]:
    """Build the output (C, Y, X) array for ``whole`` mode.

    The phase channel contributes one slice (the chosen Z). Each fluorescence
    channel contributes the slice at the chosen Z; if ``save_mip`` is True
    it also contributes a ``<name>_MIP`` max-intensity projection.
    """
    planes: list[np.ndarray] = []
    names: list[str] = []
    n_channels = array_zcyx.shape[1]
    for c in range(n_channels):
        cname = channel_names[c]
        if c == phase_idx:
            planes.append(array_zcyx[chosen_z, c])
            names.append(cname)
        else:
            planes.append(array_zcyx[chosen_z, c])
            names.append(f"{cname}@z" if save_mip else cname)
            if save_mip:
                planes.append(array_zcyx[:, c].max(axis=0))
                names.append(f"{cname}_MIP")
    return np.stack(planes, axis=0), names


def _assemble_tiled_planes(
    array_zcyx: np.ndarray,
    channel_names: list[str],
    phase_idx: int,
    z_per_tile: dict[tiling.TileCoord, int],
    grid: tiling.GridSpec,
    save_mip: bool = False,
) -> tuple[np.ndarray, list[str]]:
    """Build the output (C, Y, X) array for ``tiles`` mode.

    Phase: stitched per-tile from the tile-Z map. Each fluorescence channel
    is also stitched from the same tile-Z map; if ``save_mip`` is True a
    companion ``<name>_MIP`` max-intensity projection is appended.
    """
    planes: list[np.ndarray] = []
    names: list[str] = []
    n_channels = array_zcyx.shape[1]
    for c in range(n_channels):
        cname = channel_names[c]
        stitched = tiling.assemble_tiled_plane(array_zcyx[:, c], z_per_tile, grid)
        if c == phase_idx:
            planes.append(stitched)
            names.append(cname)
        else:
            planes.append(stitched)
            names.append(f"{cname}@z" if save_mip else cname)
            if save_mip:
                planes.append(array_zcyx[:, c].max(axis=0))
                names.append(f"{cname}_MIP")
    return np.stack(planes, axis=0), names


def _assemble_edf_planes(
    array_zcyx: np.ndarray,
    channel_names: list[str],
    phase_idx: int,
    z_map: np.ndarray,
    save_mip: bool = False,
) -> tuple[np.ndarray, list[str]]:
    """Build the output (C, Y, X) array for ``edf`` mode.

    Phase: per-pixel selection from the phase Z-map. Each fluorescence
    channel: per-pixel-selected from the same Z-map (channels stay
    aligned); if ``save_mip`` is True a companion ``<name>_MIP`` is
    appended.
    """
    planes: list[np.ndarray] = []
    names: list[str] = []
    n_channels = array_zcyx.shape[1]
    for c in range(n_channels):
        cname = channel_names[c]
        gathered = focus.assemble_edf_plane(array_zcyx[:, c], z_map)
        if c == phase_idx:
            planes.append(gathered)
            names.append(cname)
        else:
            planes.append(gathered)
            names.append(f"{cname}@z" if save_mip else cname)
            if save_mip:
                planes.append(array_zcyx[:, c].max(axis=0))
                names.append(f"{cname}_MIP")
    return np.stack(planes, axis=0), names


def _write_scores_csv(
    path: Path,
    scene_index: int,
    scores: dict[str, np.ndarray],
    chosen_z: int,
) -> None:
    metric_names = list(focus.METRIC_NAMES)
    n_z = len(next(iter(scores.values())))
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["scene", "z", *metric_names, "chosen"])
        for z in range(n_z):
            row = [scene_index, z]
            row.extend(f"{scores[m][z]:.6g}" for m in metric_names)
            row.append("1" if z == chosen_z else "0")
            writer.writerow(row)


def _write_well_scores_csv(
    path: Path,
    well: str,
    rows: list[tuple[int, int, dict[str, np.ndarray], int]],
) -> None:
    """Write a per-well CSV with one row per (FOV, Z) combination."""
    metric_names = list(focus.METRIC_NAMES)
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            ["well", "fov_in_well", "scene", "z", *metric_names, "chosen"]
        )
        for fov_in_well, (scene_index, _chosen, scores, chosen_z) in enumerate(rows):
            n_z = len(next(iter(scores.values())))
            for z in range(n_z):
                row = [well, fov_in_well, scene_index, z]
                row.extend(f"{scores[m][z]:.6g}" for m in metric_names)
                row.append("1" if z == chosen_z else "0")
                writer.writerow(row)


def _write_tile_scores_csv(
    path: Path,
    scene_index: int,
    tile_picks: dict[tiling.TileCoord, tuple[int, dict[str, np.ndarray]]],
    grid: tiling.GridSpec,
) -> None:
    """One row per (tile, z): tile_y, tile_x, scene, z, <metrics>, chosen."""
    metric_names = list(focus.METRIC_NAMES)
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["scene", "tile_y", "tile_x", "z", *metric_names, "chosen"])
        for (i, j), (chosen_z, scores) in sorted(tile_picks.items()):
            n_z = len(next(iter(scores.values())))
            for z in range(n_z):
                row = [scene_index, i, j, z]
                row.extend(f"{scores[m][z]:.6g}" for m in metric_names)
                row.append("1" if z == chosen_z else "0")
                writer.writerow(row)


def _write_well_tile_scores_csv(
    path: Path,
    well: str,
    rows: list[tuple[int, dict[tiling.TileCoord, tuple[int, dict[str, np.ndarray]]]]],
    grid: tiling.GridSpec,
) -> None:
    """Per-well CSV with one row per (FOV, tile, z)."""
    metric_names = list(focus.METRIC_NAMES)
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            ["well", "fov_in_well", "scene", "tile_y", "tile_x", "z", *metric_names, "chosen"]
        )
        for fov_in_well, (scene_index, tile_picks) in enumerate(rows):
            for (i, j), (chosen_z, scores) in sorted(tile_picks.items()):
                n_z = len(next(iter(scores.values())))
                for z in range(n_z):
                    row = [well, fov_in_well, scene_index, i, j, z]
                    row.extend(f"{scores[m][z]:.6g}" for m in metric_names)
                    row.append("1" if z == chosen_z else "0")
                    writer.writerow(row)


def _write_acquisition_sidecar(
    tiff_path: Path,
    source_czi: Path,
    plate_acquisition_datetime: str | None,
    fov_acquisition_times: list[str | None],
    scene_indices: list[int] | None = None,
    fov_source_files: list[str] | None = None,
) -> None:
    """Write a JSON sidecar carrying source-file provenance for downstream stages.

    The Features stage reads this to populate per-cell ``source_czi`` and
    ``acquisition_time`` columns. Path is ``<tiff_stem>__acquisition.json``
    next to the TIFF. ``source_czi`` keeps its historical name but may be
    a TIFF; when a sample pools FOVs from several files, it lists them all
    and ``fov_source_files`` records the file each FOV came from.
    """
    sidecar = tiff_path.with_name(tiff_path.stem + "__acquisition.json")
    sources = list(dict.fromkeys(fov_source_files or [])) or [source_czi.name]
    payload: dict = {
        "source_czi": "; ".join(sources),
        "plate_acquisition_datetime": plate_acquisition_datetime,
        "fov_acquisition_times": list(fov_acquisition_times),
    }
    if scene_indices is not None:
        payload["scene_indices"] = list(scene_indices)
    if fov_source_files is not None:
        payload["fov_source_files"] = list(fov_source_files)
    try:
        from ..provenance import atomic_write_text
        atomic_write_text(sidecar, json.dumps(payload, indent=2))
    except OSError:
        pass


def _write_zmap_tif(path: Path, z_map: np.ndarray) -> None:
    """Write a single (Y, X) Z-index map as a uint8 TIFF (Fiji-friendly)."""
    arr = np.clip(z_map, 0, 255).astype(np.uint8)
    from ..provenance import atomic_output_path
    tmp = atomic_output_path(path)
    try:
        tifffile.imwrite(str(tmp), arr, photometric="minisblack")
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def _write_well_zmaps_tif(path: Path, z_maps: list[np.ndarray]) -> None:
    """Write a (T, Y, X) stack of Z-maps for all FOVs in one well."""
    arr = np.stack([np.clip(zm, 0, 255).astype(np.uint8) for zm in z_maps], axis=0)
    from ..provenance import atomic_output_path
    tmp = atomic_output_path(path)
    try:
        tifffile.imwrite(str(tmp), arr, photometric="minisblack")
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


@dataclass
class _ProcessedScene:
    """Internal aggregate of everything one scene produced, mode-agnostic."""

    planes: np.ndarray
    channel_names: list[str]
    mode: FocusMode
    summary_z: int  # representative scalar Z (chosen_z for whole; mean for tiles/edf)
    scores: dict[str, np.ndarray] | None = None  # whole only
    tile_picks: dict[tiling.TileCoord, tuple[int, dict[str, np.ndarray]]] | None = None  # tiles only
    z_map: np.ndarray | None = None  # edf only
    description_extra: str = ""


def _process_scene(
    scene: Scene,
    phase_idx: int,
    opts: Options,
) -> _ProcessedScene:
    """Score the phase stack and assemble output planes per the chosen mode."""
    phase_stack = scene.array_zcyx[:, phase_idx]

    if opts.focus_mode == "whole":
        scores = focus.score_stack(
            phase_stack,
            crop_fraction=opts.crop_fraction,
            preblur_sigma=opts.preblur_sigma,
        )
        chosen_z = focus.pick_best_slice(scores, metric=opts.metric)
        planes, names = _assemble_output_planes(
            scene.array_zcyx, scene.channel_names, phase_idx, chosen_z,
            save_mip=opts.save_mip,
        )
        return _ProcessedScene(
            planes=planes,
            channel_names=names,
            mode="whole",
            summary_z=chosen_z,
            scores=scores,
            description_extra=f"chosen_z={chosen_z} mode=whole",
        )

    if opts.focus_mode == "tiles":
        tile_picks = tiling.pick_best_z_per_tile(
            phase_stack, opts.tile_grid, metric=opts.metric
        )
        z_per_tile = {coord: chosen for coord, (chosen, _) in tile_picks.items()}
        planes, names = _assemble_tiled_planes(
            scene.array_zcyx, scene.channel_names, phase_idx, z_per_tile, opts.tile_grid,
            save_mip=opts.save_mip,
        )
        zs = list(z_per_tile.values())
        summary_z = int(round(sum(zs) / len(zs))) if zs else 0
        rows, cols = opts.tile_grid
        return _ProcessedScene(
            planes=planes,
            channel_names=names,
            mode="tiles",
            summary_z=summary_z,
            tile_picks=tile_picks,
            description_extra=f"mode=tiles grid={rows}x{cols} mean_z={summary_z}",
        )

    if opts.focus_mode == "edf":
        z_map = focus.pick_per_pixel_z(phase_stack)
        planes, names = _assemble_edf_planes(
            scene.array_zcyx, scene.channel_names, phase_idx, z_map,
            save_mip=opts.save_mip,
        )
        summary_z = int(round(float(z_map.mean())))
        z_min, z_max = int(z_map.min()), int(z_map.max())
        return _ProcessedScene(
            planes=planes,
            channel_names=names,
            mode="edf",
            summary_z=summary_z,
            z_map=z_map,
            description_extra=f"mode=edf z_min={z_min} z_max={z_max} mean_z={summary_z}",
        )

    if opts.focus_mode == "none":
        if scene.array_zcyx.shape[0] != 1:
            raise ValueError(
                f"scene {scene.index} has {scene.array_zcyx.shape[0]} Z-planes; "
                "focus_mode 'none' only accepts single-plane images"
            )
        return _ProcessedScene(
            planes=scene.array_zcyx[0],
            channel_names=list(scene.channel_names),
            mode="none",
            summary_z=0,
            description_extra="mode=none",
        )

    raise ValueError(f"unknown focus_mode {opts.focus_mode!r}")


@dataclass(frozen=True)
class _SceneRef:
    """One field of view: a scene inside a specific input file."""

    path: Path
    index: int


def _resolve_phase_idx(
    refs: list[_SceneRef],
    opts: Options,
    scene_cache: dict[_SceneRef, Scene],
) -> int:
    """Decide the phase channel once for a set of FOVs.

    Scenes read along the way are left in ``scene_cache`` for reuse.
    """
    if opts.phase_channel is not None:
        first = io_source.read_scene(refs[0].path, refs[0].index)
        scene_cache[refs[0]] = first
        return channel_id.resolve_phase_channel(
            first.array_zcyx, first.channel_names, opts.phase_channel
        )
    # Pool stats from scenes spread evenly across the inputs rather than the
    # first N — corner-of-plate FOVs can have unrepresentative channel
    # statistics (e.g. weak fluorescence that mimics phase contrast).
    sample_count = min(10, len(refs))
    step = max(1, len(refs) // sample_count)
    positions = sorted({min(i * step, len(refs) - 1) for i in range(sample_count)})
    sample_refs = [refs[p] for p in positions]
    sample = [io_source.read_scene(r.path, r.index) for r in sample_refs]
    scene_cache.update(zip(sample_refs, sample))
    return channel_id.detect_phase_channel_multi([s.array_zcyx for s in sample])


def _well_base(well_filenames: dict[str, str] | None, well: str, filename_suffix: str) -> str:
    """Output stem for a well: caller-supplied name → well id, plus suffix."""
    base = (well_filenames or {}).get(well, well)
    # Strip any caller-supplied extension; we own the suffix.
    for ext in (".ome.tiff", ".tiff", ".tif"):
        if base.endswith(ext):
            base = base[:-len(ext)]
            break
    return base + filename_suffix


def _process_well(
    well: str,
    refs: list[_SceneRef],
    *,
    well_dir: Path,
    base: str,
    phase_idx: int,
    opts: Options,
    scene_cache: dict[_SceneRef, Scene],
    per_scene_times: dict[Path, dict[int, str]],
    save_zmaps: bool,
    on_fov,
) -> WellResult:
    """Focus every FOV in ``refs`` and write them as one per-well TIFF."""
    fov_planes: list[np.ndarray] = []
    channel_names_out: list[str] | None = None
    first_ref: _SceneRef | None = None
    chosen_zs: list[int] = []
    whole_score_rows: list[tuple[int, int, dict[str, np.ndarray], int]] = []
    tile_score_rows: list[tuple[int, dict[tiling.TileCoord, tuple[int, dict[str, np.ndarray]]]]] = []
    edf_z_maps: list[np.ndarray] = []
    pixel_size = (None, None)
    fov_acq_times: list[str | None] = []
    fov_sources: list[str] = []
    plate_acq_dt: str | None = None

    for ref in refs:
        scene = scene_cache.pop(ref, None) or io_source.read_scene(ref.path, ref.index)
        n_z = scene.array_zcyx.shape[0]
        if opts.focus_mode == "none" and n_z != 1:
            raise ValueError(
                f"{ref.path.name} (scene {ref.index}) has {n_z} Z-planes. "
                "Enable Focus to collapse Z-stacks; Split without Focus only "
                "accepts single-plane images."
            )
        processed = _process_scene(scene, phase_idx, opts)
        if first_ref is None:
            first_ref = ref
            channel_names_out = processed.channel_names
        elif processed.planes.shape != fov_planes[0].shape:
            raise ValueError(
                f"Cannot combine fields of view for {base}: "
                f"{ref.path.name} (scene {ref.index}) is {processed.planes.shape} "
                f"(channels, Y, X) but {first_ref.path.name} is "
                f"{fov_planes[0].shape}. Files pooled into one sample must "
                "have the same channels and image size."
            )
        fov_planes.append(processed.planes)
        chosen_zs.append(processed.summary_z)
        # Prefer the per-scene subblock timestamp; fall back to the
        # file-level master if the subblock didn't carry one.
        per_scene_t = per_scene_times.get(ref.path, {}).get(int(ref.index))
        fov_acq_times.append(per_scene_t or scene.acquisition_time)
        fov_sources.append(ref.path.name)
        if plate_acq_dt is None:
            plate_acq_dt = scene.acquisition_time
        if processed.mode == "whole":
            whole_score_rows.append(
                (ref.index, processed.summary_z, processed.scores, processed.summary_z)
            )
        elif processed.mode == "tiles":
            tile_score_rows.append((ref.index, processed.tile_picks))
        elif processed.mode == "edf":
            edf_z_maps.append(processed.z_map)
        pixel_size = scene.pixel_size_um
        on_fov(len(chosen_zs))

    stack_tcyx = np.stack(fov_planes, axis=0)  # (T, C, Y, X)
    out_path = well_dir / f"{base}.tif"
    sources = list(dict.fromkeys(fov_sources))

    mode_tag = opts.focus_mode
    if mode_tag == "tiles":
        rows, cols = opts.tile_grid
        mode_tag = f"tiles({rows}x{cols})"
    description = (
        f"FocusPicker source={', '.join(sources)} well={well} "
        f"n_fov={len(refs)} metric={opts.metric} mode={mode_tag}"
    )
    io_ometiff.write_tcyx(
        out_path,
        stack_tcyx,
        channel_names_out or [],
        pixel_size,
        description=description,
    )

    _write_acquisition_sidecar(
        out_path,
        source_czi=refs[0].path,
        plate_acquisition_datetime=plate_acq_dt,
        fov_acquisition_times=fov_acq_times,
        scene_indices=[r.index for r in refs],
        fov_source_files=fov_sources if len(sources) > 1 else None,
    )

    if opts.focus_mode == "whole":
        scores_path = well_dir / f"{base}_focus_scores.csv"
        _write_well_scores_csv(scores_path, well, whole_score_rows)
    elif opts.focus_mode == "tiles":
        scores_path = well_dir / f"{base}_focus_scores.csv"
        _write_well_tile_scores_csv(scores_path, well, tile_score_rows, opts.tile_grid)
    elif opts.focus_mode == "edf":
        scores_path = well_dir / f"{base}_zmaps.tif"
        if save_zmaps:
            _write_well_zmaps_tif(scores_path, edf_z_maps)
    else:  # none — nothing was scored
        scores_path = None

    return WellResult(
        well=well,
        scene_indices=[r.index for r in refs],
        chosen_z=chosen_zs,
        output_path=out_path,
        scores_csv=scores_path,
    )


def process_czi(
    czi_path: Path,
    opts: Options | None = None,
    out_dir: Path | None = None,
    well_filenames: dict[str, str] | None = None,
    save_zmaps: bool = True,
    filename_suffix: str = "",
    progress_cb=None,
    single_bucket_label: str | None = None,
) -> ProcessResult:
    """Process every scene of a CZI (or TIFF).

    Args:
        czi_path: the CZI or TIFF to read.
        opts: focus-picking options.
        out_dir: where to write output. If None, write next to the CZI and
            nest under a per-CZI folder (legacy behaviour). If given, files
            are written directly into ``out_dir`` (flat).
        well_filenames: optional mapping ``{well_id → base_filename}`` that
            overrides the default ``{well}.ome.tiff`` naming, allowing the
            caller (e.g. the GUI) to supply condition/reporter-based names.
        save_zmaps: if False, EDF-mode z-map TIFFs are not written.
        filename_suffix: appended to each output stem before the extension,
            e.g. "_focused".
        single_bucket_label: bulk-mode escape hatch. If set, all scenes in
            the CZI are grouped into a single virtual well named by this
            label, regardless of any plate metadata in the file. Output is
            ``out_dir / f"{single_bucket_label}{filename_suffix}.tif"``.
            Use this for CZIs that aren't multi-well plates (each one is
            its own sample). See :func:`process_sample` to pool several
            files into one sample.
    """
    czi_path = Path(czi_path)
    if single_bucket_label is not None:
        return process_sample(
            [czi_path],
            single_bucket_label,
            opts=opts,
            out_dir=out_dir if out_dir is not None else czi_path.parent / czi_path.stem,
            save_zmaps=save_zmaps,
            filename_suffix=filename_suffix,
            progress_cb=progress_cb,
        )

    opts = opts or Options()
    flat_out = out_dir is not None
    out_dir = Path(out_dir) if out_dir is not None else czi_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = czi_path.stem
    result = ProcessResult()

    def _emit(frac: float, msg: str) -> None:
        if progress_cb is not None:
            progress_cb(frac, msg)

    all_indices = io_source.list_scene_indices(czi_path)
    if not all_indices:
        return result

    _emit(0.02, f"Resolving phase channel over {min(10, len(all_indices))} sample scenes")

    # Per-scene AcquisitionTime — pulled once via czifile subblock metadata
    # (the master CZI XML only carries the plate-level AcquisitionDateAndTime,
    # which is identical for every scene). Empty dict on any failure;
    # downstream code falls back to the file-level timestamp.
    per_scene_times = {czi_path: io_source.per_scene_acquisition_times(czi_path)}

    well_map = io_source.list_scene_wells(czi_path)
    use_wells = bool(well_map)

    # Decide phase channel once for the whole file, with up to 10 scenes pooled.
    all_refs = [_SceneRef(czi_path, idx) for idx in all_indices]
    scene_cache: dict[_SceneRef, Scene] = {}
    phase_idx = _resolve_phase_idx(all_refs, opts, scene_cache)

    if use_wells:
        # Group scene indices by well, preserving original ordering.
        wells: dict[str, list[_SceneRef]] = defaultdict(list)
        for ref in all_refs:
            wells[well_map.get(ref.index) or "unassigned"].append(ref)
        # When an explicit out_dir was supplied (the GUI case), write files
        # flat into it. Otherwise use the legacy per-CZI nested folder.
        well_dir = out_dir if flat_out else (out_dir / stem)
        well_dir.mkdir(parents=True, exist_ok=True)

        well_list = sorted(wells)
        total_scenes = len(all_refs)
        done_scenes = 0
        for w_idx, well in enumerate(well_list):
            refs = wells[well]
            _emit(0.05 + 0.9 * (done_scenes / max(total_scenes, 1)),
                  f"Processing well {well} ({w_idx+1}/{len(well_list)})")

            def on_fov(n: int, _well=well, _n=len(refs), _done=done_scenes) -> None:
                _emit(0.05 + 0.9 * ((_done + n) / max(total_scenes, 1)),
                      f"Well {_well}: FOV {n}/{_n}")

            result.well_results.append(
                _process_well(
                    well,
                    refs,
                    well_dir=well_dir,
                    base=_well_base(well_filenames, well, filename_suffix),
                    phase_idx=phase_idx,
                    opts=opts,
                    scene_cache=scene_cache,
                    per_scene_times=per_scene_times,
                    save_zmaps=save_zmaps,
                    on_fov=on_fov,
                )
            )
            done_scenes += len(refs)
    else:
        for scene_pos, scene_index in enumerate(all_indices):
            _emit(
                0.05 + 0.9 * (scene_pos / max(len(all_indices), 1)),
                f"Processing FOV {scene_pos + 1}/{len(all_indices)}",
            )
            scene = scene_cache.pop(
                _SceneRef(czi_path, scene_index), None
            ) or io_source.read_scene(czi_path, scene_index)
            processed = _process_scene(scene, phase_idx, opts)

            scene_tag = f"_scene{scene.index:02d}"
            out_path = out_dir / f"{stem}{scene_tag}_focus.tif"

            description = (
                f"FocusPicker source={czi_path.name} scene={scene.index} "
                f"metric={opts.metric} {processed.description_extra} "
                f"phase_channel={scene.channel_names[phase_idx]} "
                f"acquired={scene.acquisition_time or 'unknown'}"
            )
            io_ometiff.write(
                out_path,
                processed.planes,
                processed.channel_names,
                scene.pixel_size_um,
                description=description,
            )

            if processed.mode == "whole":
                scores_path = out_dir / f"{stem}{scene_tag}_focus_scores.csv"
                _write_scores_csv(scores_path, scene.index, processed.scores, processed.summary_z)
            elif processed.mode == "tiles":
                scores_path = out_dir / f"{stem}{scene_tag}_focus_scores.csv"
                _write_tile_scores_csv(
                    scores_path, scene.index, processed.tile_picks, opts.tile_grid
                )
            elif processed.mode == "edf":
                scores_path = out_dir / f"{stem}{scene_tag}_zmap.tif"
                if save_zmaps:
                    _write_zmap_tif(scores_path, processed.z_map)
            else:  # none — nothing was scored
                scores_path = None

            result.scene_results.append(
                SceneResult(
                    scene_index=scene.index,
                    chosen_z=processed.summary_z,
                    output_path=out_path,
                    scores_csv=scores_path,
                )
            )
            _emit(
                0.05 + 0.9 * ((scene_pos + 1) / max(len(all_indices), 1)),
                f"Completed FOV {scene_pos + 1}/{len(all_indices)}",
            )

    if opts.archive_original and (result.scene_results or result.well_results):
        archive.move_to_raw(czi_path)

    return result


def process_sample(
    paths: list[Path],
    label: str,
    *,
    out_dir: Path,
    opts: Options | None = None,
    save_zmaps: bool = True,
    filename_suffix: str = "",
    progress_cb=None,
) -> ProcessResult:
    """Focus one sample whose FOVs come from one or more input files.

    Every scene of every file in ``paths`` (CZI or TIFF, in order) becomes
    a field of view of a single virtual well, written to
    ``out_dir / f"{label}{filename_suffix}.tif"``. Plate-well metadata is
    ignored. Used by bulk mode, where batch rows sharing the same labels
    are pooled into one sample.
    """
    opts = opts or Options()
    paths = [Path(p) for p in paths]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = ProcessResult()

    def _emit(frac: float, msg: str) -> None:
        if progress_cb is not None:
            progress_cb(frac, msg)

    refs = [_SceneRef(p, idx) for p in paths for idx in io_source.list_scene_indices(p)]
    if not refs:
        return result

    if opts.focus_mode != "none":
        _emit(0.02, f"Resolving phase channel over {min(10, len(refs))} sample FOVs")
    per_scene_times = {p: io_source.per_scene_acquisition_times(p) for p in paths}
    scene_cache: dict[_SceneRef, Scene] = {}
    # Without focus picking nothing reads the phase channel.
    phase_idx = 0 if opts.focus_mode == "none" else _resolve_phase_idx(refs, opts, scene_cache)

    files_note = f" from {len(paths)} files" if len(paths) > 1 else ""
    _emit(0.05, f"Processing {len(refs)} FOV(s){files_note}")

    def on_fov(n: int) -> None:
        _emit(0.05 + 0.9 * (n / len(refs)), f"FOV {n}/{len(refs)}{files_note}")

    result.well_results.append(
        _process_well(
            "_BULK_",
            refs,
            well_dir=out_dir,
            base=_well_base(None, label, filename_suffix),
            phase_idx=phase_idx,
            opts=opts,
            scene_cache=scene_cache,
            per_scene_times=per_scene_times,
            save_zmaps=save_zmaps,
            on_fov=on_fov,
        )
    )

    if opts.archive_original:
        for p in paths:
            archive.move_to_raw(p)

    return result
