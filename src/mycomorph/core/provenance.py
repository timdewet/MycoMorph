"""Signatures and atomic-file helpers for resumable pipeline stages."""

from __future__ import annotations

import functools
import hashlib
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

CHECKPOINT_SCHEMA_VERSION = 1
SIGNATURE_FILE = "checkpoint_signature.json"
ARTIFACT_SIGNATURE_SUFFIX = ".provenance.json"
ARTIFACT_IN_PROGRESS_SUFFIX = ".inprogress"

# Dependencies whose versions change results, recorded alongside MycoMorph's.
_RECORDED_DEPENDENCIES = ("cellpose", "torch")


def source_commit(package_dir: Path) -> str | None:
    """Git commit of the source checkout ``package_dir`` was imported from.

    Only recognises the repository's own layout (``<repo>/src/mycomorph``
    next to ``<repo>/pyproject.toml``), so a copy installed into some other
    project's virtualenv is never attributed to that project's repository.
    Reads ``.git`` files directly rather than running git, so it is fast and
    safe inside the frozen app. Handles worktrees and packed refs. It does
    not detect uncommitted changes.
    """
    package_dir = Path(package_dir)
    if package_dir.parent.name != "src":
        return None
    repo = package_dir.parent.parent
    dot_git = repo / ".git"
    if not (repo / "pyproject.toml").is_file() or not dot_git.exists():
        return None
    try:
        if dot_git.is_file():
            # Worktree or submodule: ".git" holds "gitdir: <path>".
            text = dot_git.read_text(encoding="utf-8").strip()
            if not text.startswith("gitdir:"):
                return None
            git_dir = (repo / text.split(":", 1)[1].strip()).resolve()
        else:
            git_dir = dot_git
        head = (git_dir / "HEAD").read_text(encoding="utf-8").strip()
        if not head.startswith("ref:"):
            return head or None  # detached HEAD holds the commit itself
        ref = head.split(":", 1)[1].strip()
        common_dir = git_dir
        commondir_file = git_dir / "commondir"
        if commondir_file.is_file():
            common_dir = (git_dir / commondir_file.read_text(encoding="utf-8").strip()).resolve()
        for base in (git_dir, common_dir):
            ref_file = base / ref
            if ref_file.is_file():
                return ref_file.read_text(encoding="utf-8").strip() or None
        packed = common_dir / "packed-refs"
        if packed.is_file():
            for line in packed.read_text(encoding="utf-8").splitlines():
                sha, _, name = line.partition(" ")
                if name == ref and not line.startswith(("#", "^")):
                    return sha
    except OSError:
        return None
    return None


@functools.lru_cache(maxsize=1)
def _software_info_cached() -> str:
    import platform
    import sys
    from importlib.metadata import PackageNotFoundError, version

    import mycomorph

    frozen = bool(getattr(sys, "frozen", False))
    info: dict[str, Any] = {
        "mycomorph": mycomorph.__version__,
        "frozen_app": frozen,
        "python": platform.python_version(),
    }
    if not frozen:
        commit = source_commit(Path(mycomorph.__file__).resolve().parent)
        if commit:
            info["mycomorph_git_commit"] = commit
    for dist in _RECORDED_DEPENDENCIES:
        try:
            info[dist] = version(dist)
        except PackageNotFoundError:
            pass
    return json.dumps(info, sort_keys=True)


def software_info() -> dict[str, Any]:
    """Versions of MycoMorph and key dependencies, for run and output records.

    Written outside signature digests, so a new version never invalidates
    reusable outputs. Computed once per process.
    """
    return json.loads(_software_info_cached())


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
    # Options objects can choose what they hash (e.g. leave out a new field
    # at its default so older signatures stay valid).
    if hasattr(options, "signature_payload"):
        options = options.signature_payload()
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
    atomic_write_text(signature_path, json.dumps(_with_software(signature), indent=2, sort_keys=True))
    return reusable


def _with_software(signature: dict[str, Any]) -> dict[str, Any]:
    """Copy of ``signature`` with the software versions added.

    Reuse checks compare only ``digest``, so this block documents the run
    without affecting whether its outputs are reused.
    """
    return {**signature, "software": software_info()}


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
        json.dumps(_with_software(signature), indent=2, sort_keys=True),
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
