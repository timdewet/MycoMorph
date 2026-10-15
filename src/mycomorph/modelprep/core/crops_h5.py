"""HDF5 crops adapter — projects MycoMorph's ``all_crops.h5`` into the
classifier-ready ``(3, 64, 64)`` = [phase, mask, area] shape.

This is the single boundary between MycoMorph and ModelPrep. The schema is the
one written by ``mycomorph.core.extract.crops``:

    crops             (N, C, 128, 128)  float32, per-channel min-max into [0, 1]
    cell_uids         (N,)              str       — primary key
    areas_px          (N,)              int32     — true mask pixel count
    channel_names     attr              list[str] — e.g. ["Phase", "ParB", "Mask"]
    crop_size         attr              int (typically 128)
    schema_version    attr              str ("1.0")
    + parallel metadata: condition_labels, genes, drugs, reporters, tiff_files,
      fov_indices, cell_labels, fov_ids, is_control, is_drug, ...

The reader exposes random-access ``get(idx)`` and batched
``get_classifier_batch(indices)`` for active-learning workers.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Sequence

import numpy as np

from .cell_quality_classifier import (
    CROP_SIZE,
    _append_area_channel_batch,
    _append_area_channel_single,
)


SUPPORTED_SCHEMA_VERSIONS = ("1.0",)


@dataclass(frozen=True)
class CropRecord:
    """One cell normalised for the labeller + active-learning pipeline.

    ``crop`` is classifier-ready (phase + mask + area, 64×64). ``full_crop`` is
    the raw 128×128 row from the HDF5 — handed to the UI when the user wants
    to inspect the original channel set.
    """
    cell_uid: str
    crop: np.ndarray            # (3, 64, 64) float32
    full_crop: np.ndarray       # (C, 128, 128) float32
    area_px: int
    condition_label: str
    gene: str
    drug: str
    reporter: str
    tiff_file: str
    fov_index: int
    cell_label: int
    source_h5_index: int


class HDF5CropsReader:
    """Random-access reader for an ``all_crops.h5`` produced by MycoMorph.

    The reader is read-only: it never mutates the HDF5. Open files use
    ``swmr=True`` so MycoMorph can keep writing if it ever does (today it
    doesn't, but the cost is zero).

    ``classifier_size`` controls the resize applied to the [phase, mask] pair
    before the area channel is appended. The bundled MycoMorph classifier was
    trained at 64×64, which is also ``CROP_SIZE``. Don't change this unless
    you're training a new architecture.
    """

    def __init__(
        self,
        h5_path: os.PathLike | str,
        *,
        classifier_size: int = CROP_SIZE,
        max_cache_bytes: Optional[int] = None,
    ) -> None:
        import h5py

        self.path = Path(h5_path)
        if not self.path.exists():
            raise FileNotFoundError(f"HDF5 crops file not found: {self.path}")

        self._h5 = h5py.File(str(self.path), "r", swmr=True)

        # Schema validation
        schema = str(self._h5.attrs.get("schema_version", "")) or "unknown"
        if schema not in SUPPORTED_SCHEMA_VERSIONS:
            self._h5.close()
            raise ValueError(
                f"{self.path}: schema_version={schema!r}, supported="
                f"{SUPPORTED_SCHEMA_VERSIONS}"
            )

        if "crops" not in self._h5:
            self._h5.close()
            raise ValueError(f"{self.path}: missing required dataset 'crops'")
        if "cell_uids" not in self._h5:
            self._h5.close()
            raise ValueError(f"{self.path}: missing required dataset 'cell_uids'")

        self._crops_ds = self._h5["crops"]
        n, c, h, w = self._crops_ds.shape
        self._n = int(n)
        self._n_channels = int(c)
        self._native_h = int(h)
        self._native_w = int(w)
        self._classifier_size = int(classifier_size)

        # Channel-name resolution mirrors the pattern in
        # mycomorph.core.autoencoder.dataset._select_channels.
        raw_names = self._h5.attrs.get("channel_names", [])
        self._channel_names = [_decode(s) for s in list(raw_names)]
        lower = [n.lower() for n in self._channel_names]

        self._mask_channel_idx = next(
            (i for i, n in enumerate(lower) if "mask" in n), -1,
        )
        if self._mask_channel_idx < 0:
            # Conventionally the mask is the last channel even if name doesn't
            # contain "mask" — fall back to that.
            self._mask_channel_idx = self._n_channels - 1

        # Prefer an explicit "phase" name; else first non-mask channel.
        phase_hit = next(
            (i for i, n in enumerate(lower) if n == "phase"), None,
        )
        if phase_hit is None:
            phase_hit = next(
                (i for i, n in enumerate(lower) if "phase" in n), None,
            )
        if phase_hit is None:
            phase_hit = next(
                (i for i in range(self._n_channels) if i != self._mask_channel_idx),
                0,
            )
        self._phase_channel_idx = int(phase_hit)

        # Metadata — read once into memory (strings are small).
        def _read_str(key: str) -> np.ndarray:
            if key in self._h5:
                return np.array([_decode(s) for s in self._h5[key][:]], dtype=object)
            return np.array([""] * self._n, dtype=object)

        def _read_int(key: str, dtype=np.int32) -> np.ndarray:
            if key in self._h5:
                return np.asarray(self._h5[key][:], dtype=dtype)
            return np.zeros(self._n, dtype=dtype)

        self._cell_uids = _read_str("cell_uids")
        self._condition_labels = _read_str("condition_labels")
        self._genes = _read_str("genes")
        self._drugs = _read_str("drugs")
        self._reporters = _read_str("reporters")
        self._tiff_files = _read_str("tiff_files")
        self._fov_indices = _read_int("fov_indices")
        # Global FOV ids (unique across TIFFs/wells). Older files lack them; then
        # a (tiff_file, fov_index) pair identifies the field instead, since
        # fov_indices restart at 0 in every TIFF.
        if "fov_ids" in self._h5:
            self._fov_groups = np.asarray(self._h5["fov_ids"][:], dtype=np.int64)
        else:
            pairs = [f"{t}\x00{i}" for t, i in zip(self._tiff_files, self._fov_indices)]
            self._fov_groups = np.unique(np.asarray(pairs, dtype=object),
                                         return_inverse=True)[1].astype(np.int64)
        self._cell_labels = _read_int("cell_labels")
        self._areas_px = _read_int("areas_px")

        # Optional caching: only honour if it fits comfortably and the user
        # asks for it. Default behaviour is streaming — typical plates exceed
        # any reasonable cap.
        bytes_per_row = c * h * w * 4  # float32
        total_bytes = self._n * bytes_per_row
        env_cap = os.environ.get("MODELPREP_CROP_CACHE_BYTES")
        if max_cache_bytes is None and env_cap is not None:
            try:
                max_cache_bytes = int(env_cap)
            except ValueError:
                max_cache_bytes = None
        if max_cache_bytes is not None and total_bytes <= max_cache_bytes:
            self._cache = np.asarray(self._crops_ds[:])
        else:
            self._cache = None

    # ── lifecycle ────────────────────────────────────────────────────────

    def close(self) -> None:
        try:
            self._h5.close()
        except Exception:
            pass

    def __enter__(self) -> "HDF5CropsReader":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __len__(self) -> int:
        return self._n

    # ── properties ───────────────────────────────────────────────────────

    @property
    def n_channels(self) -> int:
        return self._n_channels

    @property
    def channel_names(self) -> list[str]:
        return list(self._channel_names)

    @property
    def phase_channel_idx(self) -> int:
        return self._phase_channel_idx

    @property
    def mask_channel_idx(self) -> int:
        return self._mask_channel_idx

    @property
    def cell_uids(self) -> np.ndarray:
        return self._cell_uids

    @property
    def fov_indices(self) -> np.ndarray:
        """Per-cell field-of-view index *within its TIFF* (restarts per file)."""
        return self._fov_indices

    @property
    def fov_groups(self) -> np.ndarray:
        """Per-cell id of its field of view across the whole HDF5: the global
        ``fov_ids`` when present, else one id per (tiff_file, fov_index). Used
        for diversity-aware sampling."""
        return self._fov_groups

    @property
    def schema_version(self) -> str:
        return str(self._h5.attrs.get("schema_version", "unknown"))

    @property
    def native_crop_size(self) -> int:
        return self._native_h

    @property
    def classifier_size(self) -> int:
        return self._classifier_size

    def summary(self) -> str:
        names = ", ".join(self._channel_names) if self._channel_names else "(none)"
        return (
            f"{self._n:,} cells · {self._n_channels} channels [{names}] · "
            f"{self._native_h}×{self._native_w} · schema v{self.schema_version}"
        )

    # ── reads ────────────────────────────────────────────────────────────

    def _raw_row(self, idx: int) -> np.ndarray:
        if self._cache is not None:
            return np.asarray(self._cache[idx], dtype=np.float32)
        return np.asarray(self._crops_ds[idx], dtype=np.float32)

    def _raw_batch(self, indices: Sequence[int]) -> np.ndarray:
        idx = np.asarray(list(indices), dtype=np.int64)
        if self._cache is not None:
            return np.asarray(self._cache[idx], dtype=np.float32)
        # h5py supports fancy indexing but requires sorted unique indices for
        # speed; sort, fetch, then unsort.
        order = np.argsort(idx)
        sorted_idx = idx[order]
        # Find duplicate-collapsed unique indices.
        unique, inv = np.unique(sorted_idx, return_inverse=True)
        batch_unique = np.asarray(self._crops_ds[list(unique)], dtype=np.float32)
        batch_sorted = batch_unique[inv]
        out = np.empty_like(batch_sorted)
        out[order] = batch_sorted
        return out

    def get(self, idx: int) -> CropRecord:
        """Random-access read, fully projected into classifier-ready shape."""
        if not (0 <= idx < self._n):
            raise IndexError(idx)
        row = self._raw_row(int(idx))
        crop_64 = self._project_single(row, int(self._areas_px[idx]))
        return CropRecord(
            cell_uid=str(self._cell_uids[idx]),
            crop=crop_64,
            full_crop=row,
            area_px=int(self._areas_px[idx]),
            condition_label=str(self._condition_labels[idx]),
            gene=str(self._genes[idx]),
            drug=str(self._drugs[idx]),
            reporter=str(self._reporters[idx]),
            tiff_file=str(self._tiff_files[idx]),
            fov_index=int(self._fov_indices[idx]),
            cell_label=int(self._cell_labels[idx]),
            source_h5_index=int(idx),
        )

    def get_classifier_batch(self, indices: Sequence[int]) -> np.ndarray:
        """Return ``(N, 3, classifier_size, classifier_size)`` ready for
        ``CellQualityClassifier.predict_batch``."""
        idx = np.asarray(list(indices), dtype=np.int64)
        if idx.size == 0:
            return np.zeros((0, 3, self._classifier_size, self._classifier_size),
                            dtype=np.float32)
        rows = self._raw_batch(idx)
        return self._project_batch(rows, self._areas_px[idx])

    def iter_records(
        self,
        indices: Optional[Sequence[int]] = None,
    ) -> Iterator[CropRecord]:
        if indices is None:
            it = range(self._n)
        else:
            it = (int(i) for i in indices)
        for i in it:
            yield self.get(i)

    # ── projection ───────────────────────────────────────────────────────

    def _project_single(self, row: np.ndarray, area_px: int) -> np.ndarray:
        """Project one (C, 128, 128) row into the (3, 64, 64) classifier shape."""
        from skimage.transform import resize

        pair = np.stack(
            [row[self._phase_channel_idx], row[self._mask_channel_idx]],
            axis=0,
        )  # (2, native, native)

        if pair.shape[-1] != self._classifier_size:
            resized = np.zeros(
                (2, self._classifier_size, self._classifier_size),
                dtype=np.float32,
            )
            for c in range(2):
                resized[c] = resize(
                    pair[c],
                    (self._classifier_size, self._classifier_size),
                    order=1,
                    preserve_range=True,
                    anti_aliasing=True,
                )
            pair = resized

        # Re-binarise the mask after the resize.
        pair[1] = (pair[1] > 0.5).astype(np.float32)
        # Phase already in [0, 1] in the HDF5, but clip defensively.
        pair[0] = np.clip(pair[0], 0.0, 1.0).astype(np.float32)

        return _append_area_channel_single(pair, true_area_px=area_px)

    def _project_batch(
        self,
        rows: np.ndarray,
        areas_px: np.ndarray,
    ) -> np.ndarray:
        """Vectorised version of ``_project_single`` over a batch."""
        from skimage.transform import resize

        n = rows.shape[0]
        pair = np.stack(
            [
                rows[:, self._phase_channel_idx],
                rows[:, self._mask_channel_idx],
            ],
            axis=1,
        )  # (N, 2, native, native)

        if pair.shape[-1] != self._classifier_size:
            resized = np.zeros(
                (n, 2, self._classifier_size, self._classifier_size),
                dtype=np.float32,
            )
            for i in range(n):
                for c in range(2):
                    resized[i, c] = resize(
                        pair[i, c],
                        (self._classifier_size, self._classifier_size),
                        order=1,
                        preserve_range=True,
                        anti_aliasing=True,
                    )
            pair = resized

        pair[:, 1] = (pair[:, 1] > 0.5).astype(np.float32)
        pair[:, 0] = np.clip(pair[:, 0], 0.0, 1.0).astype(np.float32)
        return _append_area_channel_batch(pair, true_areas_px=areas_px)


def _decode(s) -> str:
    if isinstance(s, bytes):
        return s.decode("utf-8", errors="replace")
    return str(s)
