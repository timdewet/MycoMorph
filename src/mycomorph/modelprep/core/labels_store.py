"""Labels-on-disk for ModelPrep.

Layout (chosen to never mutate the HDF5):

    <all_crops.h5 parent>/
    ├── all_crops.h5
    └── modelprep_labels/
        ├── labels.csv       # append-only; latest cell_uid row wins
        └── manifest.json    # source HDF5 fingerprint + session history

CSV columns:
    cell_uid, label_idx, label_name, area_px, labeled_at, labeler,
    session_id, source_h5, model_at_label, confidence_good_at_label, split

``split`` is "train" (default) or "test". Test rows form a frozen hold-out set:
they are never trained on (``to_training_dataset`` skips them) and are the fixed
evaluation set (``to_eval_dataset``). Files written before ``split`` existed are
migrated in place on the next write (see ``_ensure_csv_schema``).
"""

from __future__ import annotations

import csv
import getpass
import hashlib
import json
import os
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

import numpy as np


LABELS_SCHEMA_VERSION = "1.1"
LABELS_FILENAME = "labels.csv"
MANIFEST_FILENAME = "manifest.json"
DEFAULT_LABELS_DIRNAME = "modelprep_labels"

LABEL_NAMES = ("good", "bad")
LABEL_TO_IDX = {n: i for i, n in enumerate(LABEL_NAMES)}

SPLIT_TRAIN = "train"
SPLIT_TEST = "test"
SPLITS = (SPLIT_TRAIN, SPLIT_TEST)


CSV_COLUMNS = (
    "cell_uid",
    "label_idx",
    "label_name",
    "area_px",
    "labeled_at",
    "labeler",
    "session_id",
    "source_h5",
    "model_at_label",
    "confidence_good_at_label",
    "split",
)


@dataclass
class LabelRow:
    cell_uid: str
    label_idx: int
    label_name: str
    area_px: int = 0
    labeled_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    labeler: str = field(default_factory=lambda: os.getenv("USER") or getpass.getuser() or "unknown")
    session_id: str = ""
    source_h5: str = ""
    model_at_label: str = "none"
    confidence_good_at_label: float = float("nan")
    split: str = SPLIT_TRAIN

    def as_row(self) -> list:
        return [getattr(self, col) for col in CSV_COLUMNS]


def default_labels_dir_for(h5_path: os.PathLike | str) -> Path:
    """Default labels dir alongside an ``all_crops.h5``."""
    return Path(h5_path).resolve().parent / DEFAULT_LABELS_DIRNAME


def _head_hash(path: Path, n_bytes: int = 1024) -> str:
    """Cheap fingerprint for the source HDF5 — first ``n_bytes`` bytes.

    NOTE: for an HDF5 file the first KB is the superblock / root-group
    metadata, which is identical across files written by the same MycoPrep
    version. It is kept only as a fast pre-check; ``cell_uids_sha256`` below is
    the authoritative, content-bearing fingerprint.
    """
    with open(path, "rb") as f:
        head = f.read(n_bytes)
    return hashlib.sha256(head).hexdigest()


def cell_uids_sha256(cell_uids: Iterable) -> str:
    """Content fingerprint of an HDF5 over its ``cell_uids`` (order included).

    Two different plates that happen to share a cell count produce different
    uid hashes, so this catches label/dataset mismatches that ``_head_hash``
    (superblock-only) misses.
    """
    h = hashlib.sha256()
    for uid in cell_uids:
        h.update(str(uid).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


class LabelsStore:
    """Read/write/merge for ``modelprep_labels/labels.csv``.

    The CSV is append-only. On read, duplicate ``cell_uid`` rows collapse to
    the one with the latest ``labeled_at`` — this gives undo-by-relabelling
    without a separate edit op.
    """

    def __init__(self, labels_dir: os.PathLike | str) -> None:
        self.dir = Path(labels_dir)
        self.csv_path = self.dir / LABELS_FILENAME
        self.manifest_path = self.dir / MANIFEST_FILENAME

    # ── manifest ────────────────────────────────────────────────────────

    def ensure_initialised(
        self,
        *,
        source_h5: os.PathLike | str,
        total_cells: int,
        h5_schema_version: str,
        cell_uids: Optional[Iterable] = None,
    ) -> None:
        """Create the labels dir + manifest if missing; validate if present.

        When ``cell_uids`` is supplied it is hashed and used as the
        authoritative content fingerprint (catches different-plate /
        same-count collisions that the superblock head hash misses). Old
        manifests lacking the uid hash are upgraded in place rather than
        rejected.
        """
        source_h5 = Path(source_h5)
        self.dir.mkdir(parents=True, exist_ok=True)
        expected_uids = cell_uids_sha256(cell_uids) if cell_uids is not None else None

        if self.manifest_path.exists():
            with open(self.manifest_path) as f:
                manifest = json.load(f)
            expected_head = _head_hash(source_h5)
            expected_total = int(total_cells)
            stored_head = manifest.get("source_h5_sha256_head_kb", "")
            stored_total = int(manifest.get("source_h5_total_cells", -1))
            stored_uids = manifest.get("source_h5_cell_uids_sha256", "")

            if expected_uids is not None and stored_uids:
                # Authoritative content check.
                if stored_uids != expected_uids:
                    raise ValueError(
                        f"Labels at {self.dir} were created against a different "
                        f"HDF5 file (cell-uid fingerprint mismatch: stored "
                        f"{stored_uids[:12]}..., current {expected_uids[:12]}...). "
                        f"Move or remove the existing labels dir to start fresh."
                    )
                # Content matches — refresh head/total if the file was re-saved.
                changed = False
                if stored_total != expected_total:
                    manifest["source_h5_total_cells"] = expected_total
                    changed = True
                if stored_head != expected_head:
                    manifest["source_h5_sha256_head_kb"] = expected_head
                    changed = True
                if changed:
                    self._write_manifest(manifest)
                return

            # No uid fingerprint to compare against — fall back to head+total.
            if stored_total != expected_total or stored_head != expected_head:
                raise ValueError(
                    f"Labels at {self.dir} were created against a different "
                    f"HDF5 file (stored total_cells={stored_total}, "
                    f"head_hash={stored_head[:12]}...; current "
                    f"total_cells={expected_total}, head_hash={expected_head[:12]}...). "
                    f"Move or remove the existing labels dir to start fresh."
                )
            # Upgrade an older manifest with the content fingerprint for next time.
            if expected_uids is not None and not stored_uids:
                manifest["source_h5_cell_uids_sha256"] = expected_uids
                self._write_manifest(manifest)
            return

        manifest = {
            "schema_version": LABELS_SCHEMA_VERSION,
            "source_h5": source_h5.name,
            "source_h5_total_cells": int(total_cells),
            "source_h5_schema_version": str(h5_schema_version),
            "source_h5_sha256_head_kb": _head_hash(source_h5),
            "source_h5_cell_uids_sha256": expected_uids or "",
            "first_labeled_at": None,
            "last_labeled_at": None,
            "sessions": [],
        }
        with open(self.manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

    def _read_manifest(self) -> dict:
        if not self.manifest_path.exists():
            return {}
        with open(self.manifest_path) as f:
            return json.load(f)

    def _write_manifest(self, manifest: dict) -> None:
        with open(self.manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

    # ── read ────────────────────────────────────────────────────────────

    def load(self) -> "LabelsFrame":
        """Read all rows. Returns a LabelsFrame with the latest row per cell_uid."""
        rows: list[LabelRow] = []
        n_skipped = 0
        if not self.csv_path.exists():
            return LabelsFrame(rows=[])

        with open(self.csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for r in reader:
                try:
                    rows.append(LabelRow(
                        cell_uid=r["cell_uid"],
                        label_idx=int(r["label_idx"]),
                        label_name=r.get("label_name", LABEL_NAMES[int(r["label_idx"])]),
                        area_px=int(r.get("area_px") or 0),
                        labeled_at=r.get("labeled_at", ""),
                        labeler=r.get("labeler", ""),
                        session_id=r.get("session_id", ""),
                        source_h5=r.get("source_h5", ""),
                        model_at_label=r.get("model_at_label", "none"),
                        confidence_good_at_label=_safe_float(r.get("confidence_good_at_label")),
                        split=(r.get("split") or SPLIT_TRAIN),
                    ))
                except (KeyError, ValueError):
                    n_skipped += 1
                    continue

        if n_skipped:
            import sys
            print(
                f"[modelprep] WARNING: skipped {n_skipped} malformed row(s) while "
                f"reading {self.csv_path} — the file may be corrupt or partially "
                f"written; those labels were NOT loaded.",
                file=sys.stderr,
            )
        return LabelsFrame(rows=rows, n_skipped=n_skipped)

    def labeled_uids(self) -> set[str]:
        return set(self.load().latest_by_uid().keys())

    def class_counts(self, split: Optional[str] = None) -> dict[str, int]:
        """good/bad counts, optionally restricted to one split ("train"|"test")."""
        latest = self.load().latest_by_uid()
        counts = {n: 0 for n in LABEL_NAMES}
        for row in latest.values():
            if split is not None and row.split != split:
                continue
            if row.label_name in counts:
                counts[row.label_name] += 1
        return counts

    def split_counts(self) -> dict[str, dict[str, int]]:
        """Per-split good/bad counts: ``{"train": {...}, "test": {...}}``."""
        latest = self.load().latest_by_uid()
        out = {s: {n: 0 for n in LABEL_NAMES} for s in SPLITS}
        for row in latest.values():
            bucket = out.get(row.split if row.split in out else SPLIT_TRAIN)
            if row.label_name in bucket:
                bucket[row.label_name] += 1
        return out

    # ── write ───────────────────────────────────────────────────────────

    @staticmethod
    def _lock(f) -> None:
        """Best-effort exclusive advisory lock (POSIX). No-op where unsupported."""
        try:
            import fcntl
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        except Exception:  # noqa: BLE001 — non-POSIX / unsupported FS
            pass

    @staticmethod
    def _unlock(f) -> None:
        try:
            import fcntl
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        except Exception:  # noqa: BLE001
            pass

    def _ensure_csv_schema(self) -> None:
        """One-time, in-place migration: if ``labels.csv`` predates a column
        (e.g. ``split``), rewrite it with the current header and back-fill
        defaults. No-op when the file is absent or already current.

        Without this, appending current-schema rows under an old, shorter
        header would misalign columns on the next read.
        """
        if not self.csv_path.exists():
            return
        with open(self.csv_path, newline="") as f:
            reader = csv.reader(f)
            try:
                header = next(reader)
            except StopIteration:
                return  # empty file — append_batch will write a fresh header
            if list(header) == list(CSV_COLUMNS):
                return
            old_rows = list(csv.DictReader(f, fieldnames=header))

        defaults = {col: "" for col in CSV_COLUMNS}
        defaults["split"] = SPLIT_TRAIN
        defaults["model_at_label"] = "none"
        tmp = self.csv_path.with_suffix(".csv.tmp")
        with open(tmp, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(CSV_COLUMNS)
            for r in old_rows:
                writer.writerow([(r.get(col) if r.get(col) not in (None, "") else defaults[col])
                                 for col in CSV_COLUMNS])
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self.csv_path)

    def append_batch(self, rows: Iterable[LabelRow]) -> int:
        """Append rows to ``labels.csv`` (creating it with a header if empty).

        Appends in place — O(rows written), not O(file) — under a best-effort
        advisory lock plus an fsync, so a long labelling session stays fast and
        concurrent writers (a second window, or the CLI labeller) serialise
        instead of clobbering each other's rows.
        """
        rows = list(rows)
        if not rows:
            return 0
        self.dir.mkdir(parents=True, exist_ok=True)
        self._ensure_csv_schema()

        with open(self.csv_path, "a", newline="") as f:
            self._lock(f)
            try:
                # st_size under the lock is authoritative regardless of the
                # append-mode stream position.
                need_header = os.fstat(f.fileno()).st_size == 0
                writer = csv.writer(f)
                if need_header:
                    writer.writerow(CSV_COLUMNS)
                for r in rows:
                    writer.writerow(r.as_row())
                f.flush()
                os.fsync(f.fileno())
            finally:
                self._unlock(f)

        # Update manifest timestamps.
        manifest = self._read_manifest()
        if manifest:
            now = datetime.now(timezone.utc).isoformat()
            if not manifest.get("first_labeled_at"):
                manifest["first_labeled_at"] = now
            manifest["last_labeled_at"] = now
            self._write_manifest(manifest)

        return len(rows)

    def record_session(
        self,
        session_id: str,
        started_at: str,
        ended_at: str,
        n_labels: int,
    ) -> None:
        manifest = self._read_manifest()
        if not manifest:
            return
        manifest.setdefault("sessions", []).append({
            "id": session_id,
            "started_at": started_at,
            "ended_at": ended_at,
            "n_labels": int(n_labels),
        })
        self._write_manifest(manifest)

    # ── training dataset assembly ───────────────────────────────────────

    def to_training_dataset(
        self,
        reader,
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """Return ``(crops, labels, cell_uids_used)`` for the **train** split,
        ready for ``train()``. Hold-out test rows are excluded.

        Cells whose ``cell_uid`` is in labels but not in the HDF5 are skipped.
        The latest row per uid wins.
        """
        return self._to_dataset(reader, split=SPLIT_TRAIN, require=True)

    def to_eval_dataset(
        self,
        reader,
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """Return ``(crops, labels, cell_uids_used)`` for the frozen **test**
        split, or empty arrays when no hold-out set has been labelled."""
        return self._to_dataset(reader, split=SPLIT_TEST, require=False)

    def _to_dataset(
        self,
        reader,
        split: str,
        require: bool,
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        latest = self.load().latest_by_uid()
        rows = {uid: row for uid, row in latest.items() if row.split == split}
        if not rows:
            if require:
                raise ValueError(
                    f"No '{split}' labels found in {self.csv_path}. "
                    f"Run `modelprep-cli label` first."
                )
            empty_crops = np.empty(
                (0, 3, reader.classifier_size, reader.classifier_size), dtype=np.float32
            )
            return empty_crops, np.empty((0,), dtype=np.int64), []

        h5_uid_to_idx: dict[str, int] = {
            uid: i for i, uid in enumerate(reader.cell_uids)
        }
        keep_indices: list[int] = []
        keep_labels: list[int] = []
        keep_uids: list[str] = []
        skipped: list[str] = []
        for uid, row in rows.items():
            idx = h5_uid_to_idx.get(uid)
            if idx is None:
                skipped.append(uid)
                continue
            keep_indices.append(idx)
            keep_labels.append(row.label_idx)
            keep_uids.append(uid)

        if not keep_indices:
            raise ValueError(
                f"None of the {len(rows)} '{split}' labelled cells could be found in "
                f"{reader.path.name}. Did you point at the right HDF5?"
            )

        crops = reader.get_classifier_batch(keep_indices)
        labels = np.asarray(keep_labels, dtype=np.int64)

        if skipped:
            print(f"  Warning: skipped {len(skipped)} labeled cell(s) "
                  f"missing from {reader.path.name} (e.g. {skipped[0]!r})")

        return crops, labels, keep_uids


@dataclass
class LabelsFrame:
    """Lightweight in-memory view of all loaded labels."""
    rows: list[LabelRow]
    n_skipped: int = 0  # malformed CSV rows dropped during load()

    def latest_by_uid(self) -> dict[str, LabelRow]:
        """Collapse to one row per ``cell_uid`` (latest ``labeled_at`` wins)."""
        latest: dict[str, LabelRow] = {}
        for row in self.rows:
            prior = latest.get(row.cell_uid)
            if prior is None or row.labeled_at >= prior.labeled_at:
                latest[row.cell_uid] = row
        return latest

    def __len__(self) -> int:
        return len(self.rows)


def _safe_float(v) -> float:
    if v is None or v == "":
        return float("nan")
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def make_session_id() -> str:
    return uuid.uuid4().hex[:12]
