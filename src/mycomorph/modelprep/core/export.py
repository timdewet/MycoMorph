"""Export a trained CNN run for MycoMorph's Custom-model picker.

Trained runs live in a hidden folder (``~/.modelprep/models/<run>/``) beside a
``final_model.pth`` that's easy to pick by mistake. This copies the one file
MycoMorph needs, ``best_model.pth``, into a visible folder of the user's
choosing under a readable name, together with the sidecars MycoMorph reads from
the same folder (``training_config.json`` for ``in_channels``; ``metrics.json``
and ``roc_pr_curves.png`` for its model inspector), plus a note recording where
the weights came from and their SHA-256.
"""

from __future__ import annotations

import hashlib
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

_SIDECARS = ("training_config.json", "metrics.json", "roc_pr_curves.png")
_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _unique_dir(path: Path) -> Path:
    if not path.exists():
        return path
    i = 2
    while (path.parent / f"{path.name} ({i})").exists():
        i += 1
    return path.parent / f"{path.name} ({i})"


def export_for_mycomorph(run: Path, dest_parent: Path, name: Optional[str] = None) -> Path:
    """Copy a CNN run's ``best_model.pth`` (+ sidecars) to
    ``<dest_parent>/<name>/<name>.pth`` and return that ``.pth`` path.

    ``run`` is the run directory or any file in it. ``name`` defaults to the
    run directory's name (``<model name>_<date>``). An existing folder is never
    overwritten; a numbered sibling is created instead.
    """
    run_dir = Path(run)
    if run_dir.is_file():
        run_dir = run_dir.parent
    weights = run_dir / "best_model.pth"
    if not weights.is_file():
        if (run_dir / "feature_model.joblib").exists():
            raise ValueError("This is a feature-based model; MycoMorph can only "
                             "use CNN runs (best_model.pth).")
        raise FileNotFoundError(f"No best_model.pth in {run_dir}")

    stem = _UNSAFE.sub("-", (name or run_dir.name).strip()).strip("-.") or "model"
    out = _unique_dir(Path(dest_parent) / stem)
    out.mkdir(parents=True)
    pth = out / f"{stem}.pth"
    shutil.copy2(weights, pth)
    for side in _SIDECARS:
        if (run_dir / side).is_file():
            shutil.copy2(run_dir / side, out / side)
    (out / "EXPORTED_FROM.txt").write_text(
        f"Exported by ModelPrep on {datetime.now().isoformat(timespec='seconds')}\n"
        f"source run: {run_dir}\n"
        f"source file: best_model.pth (the validation-selected weights)\n"
        f"sha256: {_sha256(pth)}\n"
        f"\nIn MycoMorph: Classify > Custom model > choose {pth.name}\n",
        encoding="utf-8",
    )
    return pth
