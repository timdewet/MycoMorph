"""Signatures and atomic-file helpers for resumable pipeline stages."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

CHECKPOINT_SCHEMA_VERSION = 1
SIGNATURE_FILE = "checkpoint_signature.json"
ARTIFACT_SIGNATURE_SUFFIX = ".provenance.json"
ARTIFACT_IN_PROGRESS_SUFFIX = ".inprogress"


def _jsonable(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in sorted(value.items(), key=lambda x: str(x[0]))}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def file_fingerprint(path: Path, *, content_hash: bool = False) -> dict[str, Any]:
    path = Path(path)
    stat = path.stat()
    result: dict[str, Any] = {
        "path": str(path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }
    if content_hash:
        digest = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                digest.update(chunk)
        result["sha256"] = digest.hexdigest()
    return result


def checkpoint_signature(
    *,
    stage: str,
    input_path: Path,
    options: Any,
    channels: Any = None,
    phase_channel: int | None = None,
    model_path: Path | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "stage": stage,
        "input": file_fingerprint(input_path),
        "options": _jsonable(options),
        "channels": _jsonable(channels),
        "phase_channel": phase_channel,
    }
    if model_path is not None and Path(model_path).exists():
        payload["model"] = file_fingerprint(Path(model_path), content_hash=True)
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    payload["digest"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return payload


def prepare_checkpoint_dir(directory: Path, signature: dict[str, Any]) -> bool:
    """Prepare a checkpoint directory and return whether it is reusable."""
    import shutil

    directory = Path(directory)
    signature_path = directory / SIGNATURE_FILE
    reusable = False
    if signature_path.exists():
        try:
            previous = json.loads(signature_path.read_text())
            reusable = previous.get("digest") == signature.get("digest")
        except (OSError, ValueError, TypeError):
            reusable = False
    if directory.exists() and not reusable:
        shutil.rmtree(directory)
    directory.mkdir(parents=True, exist_ok=True)
    atomic_write_text(signature_path, json.dumps(signature, indent=2, sort_keys=True))
    return reusable


def atomic_write_text(path: Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    try:
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def atomic_output_path(path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.with_name(f".{path.name}.tmp")


def artifact_signature_path(path: Path) -> Path:
    path = Path(path)
    return path.with_name(path.name + ARTIFACT_SIGNATURE_SUFFIX)


def artifact_in_progress_path(path: Path) -> Path:
    path = Path(path)
    return path.with_name(f".{path.name}{ARTIFACT_IN_PROGRESS_SUFFIX}")


def mark_artifact_in_progress(path: Path) -> None:
    atomic_write_text(artifact_in_progress_path(path), "in progress\n")


def clear_artifact_in_progress(path: Path) -> None:
    artifact_in_progress_path(path).unlink(missing_ok=True)


def artifact_is_in_progress(path: Path) -> bool:
    return artifact_in_progress_path(path).exists()


def write_artifact_signature(path: Path, signature: dict[str, Any]) -> None:
    atomic_write_text(
        artifact_signature_path(path),
        json.dumps(signature, indent=2, sort_keys=True),
    )


def artifact_signature_matches(path: Path, signature: dict[str, Any]) -> bool | None:
    """Return True/False for signed artifacts, or None for a legacy artifact."""
    sidecar = artifact_signature_path(path)
    if not sidecar.exists():
        return None
    try:
        previous = json.loads(sidecar.read_text())
    except (OSError, ValueError, TypeError):
        return False
    return previous.get("digest") == signature.get("digest")


def validate_tiff(path: Path) -> bool:
    try:
        import tifffile
        with tifffile.TiffFile(path) as tif:
            return bool(tif.pages) and all(page.shape for page in tif.pages)
    except Exception:  # noqa: BLE001
        return False


def validate_parquet(path: Path) -> bool:
    try:
        import pyarrow.parquet as pq
        metadata = pq.read_metadata(path)
        return metadata.num_rows >= 0 and metadata.num_columns >= 0
    except Exception:  # noqa: BLE001
        return False


def validate_hdf5(path: Path) -> bool:
    try:
        import h5py
        with h5py.File(path, "r") as handle:
            return bool(handle.keys())
    except Exception:  # noqa: BLE001
        return False
