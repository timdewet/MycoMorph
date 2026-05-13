"""Persistence helpers for optimal-transport analysis caches.

Keeping cache validation and storage separate from the plotting renderer makes
the large plotting module easier to evolve and gives every sidecar the same
atomic-write guarantees as the rest of the pipeline.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


def ot_sidecar_path(html_path: Path) -> Path:
    """Return the cached OT distance sidecar for a rendered HTML file."""
    return Path(html_path).with_suffix(".ot_distance.parquet")


def save_ot_sidecar(
    html_path: Path,
    distances: "np.ndarray",
    group_meta: list[dict],
    params: dict | None = None,
) -> None:
    """Persist an OT distance matrix and its parameters atomically.

    Analysis caches are best-effort: failures leave the previous complete
    cache intact and never prevent the main imaging pipeline from finishing.
    """
    import pandas as pd

    from ..provenance import atomic_output_path, atomic_write_text

    sidecar = ot_sidecar_path(html_path)
    tmp = atomic_output_path(sidecar)
    try:
        meta_df = pd.DataFrame(group_meta)
        for column in range(distances.shape[0]):
            meta_df[f"d_{column}"] = distances[:, column]
        meta_df.to_parquet(tmp, index=False)
        tmp.replace(sidecar)
        if params is not None:
            atomic_write_text(
                sidecar.with_suffix(sidecar.suffix + ".params.json"),
                json.dumps(params, sort_keys=True, default=str),
            )
    except Exception:  # noqa: BLE001
        pass
    finally:
        tmp.unlink(missing_ok=True)


def embedding_ot_cache_path(emb_path: Path) -> Path:
    """Return the stable default-parameter cache beside an embedding table."""
    emb_path = Path(emb_path)
    return emb_path.with_name(emb_path.stem + ".ot_default.ot_distance.parquet")


def features_ot_cache_path(library_dir: Path | None, species: str) -> Path:
    """Return the species-specific feature OT cache in the feature library."""
    from .feature_library import FeatureLibrary

    library = FeatureLibrary(library_dir)
    cache_dir = library.library_dir / "features_ot_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    safe_species = "".join(c if c.isalnum() else "_" for c in (species or "all"))
    return cache_dir / f"{safe_species}.ot_distance.parquet"


def variant_cache_path(
    canonical_path: Path,
    *,
    batch_correct: bool,
    merge_replicates: bool,
) -> Path:
    """Return the cache slot for one batch/replicate toggle combination."""
    canonical_path = Path(canonical_path)
    batch = "on" if batch_correct else "off"
    merge = "on" if merge_replicates else "off"
    variants_dir = canonical_path.parent / f"{canonical_path.stem}.variants"
    variants_dir.mkdir(parents=True, exist_ok=True)
    return variants_dir / f"bc-{batch}__merge-{merge}.ot_distance.parquet"


def copy_to_variant_cache(
    canonical_path: Path,
    *,
    batch_correct: bool,
    merge_replicates: bool,
) -> Path | None:
    """Atomically snapshot a canonical cache into its matching variant slot."""
    import shutil

    from ..provenance import atomic_output_path

    canonical_path = Path(canonical_path)
    if not canonical_path.exists():
        return None
    variant = variant_cache_path(
        canonical_path,
        batch_correct=batch_correct,
        merge_replicates=merge_replicates,
    )
    tmp = atomic_output_path(variant)
    params_source = canonical_path.with_suffix(canonical_path.suffix + ".params.json")
    params_target = variant.with_suffix(variant.suffix + ".params.json")
    params_tmp = atomic_output_path(params_target)
    try:
        shutil.copy2(canonical_path, tmp)
        tmp.replace(variant)
        if params_source.exists():
            shutil.copy2(params_source, params_tmp)
            params_tmp.replace(params_target)
        return variant
    except OSError:
        return None
    finally:
        tmp.unlink(missing_ok=True)
        params_tmp.unlink(missing_ok=True)


def try_load_ot_cache(
    cache_sidecar: Path,
    requested_params: dict,
    *,
    miss_reason: list[str] | None = None,
) -> "tuple[np.ndarray, list[dict]] | None":
    """Load a cache only when it is readable and all parameters match."""
    import numpy as np
    import pandas as pd

    def report(reason: str) -> None:
        if miss_reason is not None:
            miss_reason.append(reason)

    cache_sidecar = Path(cache_sidecar)
    if not cache_sidecar.exists():
        report(f"cache file not found at {cache_sidecar.name}")
        return None

    params_path = cache_sidecar.with_suffix(cache_sidecar.suffix + ".params.json")
    if not params_path.exists():
        report("params.json missing — old cache layout, will recompute")
        return None
    try:
        stored = json.loads(params_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        report(f"params.json unreadable: {exc}")
        return None
    for key, value in requested_params.items():
        if str(stored.get(key)) != str(value):
            report(
                f"param mismatch on {key!r}: "
                f"requested={value!r} cached={stored.get(key)!r}"
            )
            return None

    try:
        frame = pd.read_parquet(cache_sidecar)
    except Exception as exc:  # noqa: BLE001
        report(f"parquet unreadable: {exc}")
        return None
    distance_columns = sorted(
        [column for column in frame.columns if column.startswith("d_")],
        key=lambda column: int(column.split("_", 1)[1]),
    )
    if not distance_columns:
        report("cache parquet has no d_* columns")
        return None
    distances = frame[distance_columns].to_numpy(dtype=np.float64)
    metadata = frame.drop(columns=distance_columns).to_dict("records")
    return distances, metadata
