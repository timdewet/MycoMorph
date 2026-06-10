"""Naming + enumeration of trained models under ``~/.modelprep/models/``.

A run lives in a directory named ``<slug>_<YYYY-MM-DD_HH-MM-SS>`` — human
readable and grouped by the name chosen on the Source page (the chosen name is
also recorded in ``training_config.json`` under ``provenance.model_name``, which
is the authoritative source when listing). Legacy ``trained_<unix-ts>`` dirs are
still recognised.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional


def trained_models_root() -> Path:
    return Path.home() / ".modelprep" / "models"


_SLUG_RE = re.compile(r"[^A-Za-z0-9._-]+")
# Trailing human timestamp produced by human_timestamp() — used to split a run
# directory name back into (name, timestamp).
_TS_RE = re.compile(r"_(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})$")


def slugify(name: str, default: str = "model") -> str:
    """Filesystem-safe slug for a user-typed model name."""
    s = _SLUG_RE.sub("-", (name or "").strip()).strip("-_.")
    return s or default


def human_timestamp(dt: Optional[datetime] = None) -> str:
    """Readable, sortable, second-unique timestamp for a run directory."""
    return (dt or datetime.now()).strftime("%Y-%m-%d_%H-%M-%S")


def new_run_dir(name: str) -> Path:
    """Directory for a fresh run: ``<slug>_<human timestamp>``."""
    return trained_models_root() / f"{slugify(name)}_{human_timestamp()}"


def _split_dir_name(dirname: str) -> tuple[str, str]:
    """(name, created) parsed from a run directory name."""
    m = _TS_RE.search(dirname)
    if m:
        return dirname[: m.start()], m.group(1)
    if dirname.startswith("trained_"):  # legacy trained_<unix-ts>
        return "trained", dirname[len("trained_"):]
    return dirname, ""


@dataclass(frozen=True)
class TrainedModel:
    dir: Path
    name: str                 # chosen model name (or parsed from dir)
    created: str              # human timestamp string, "" if unknown
    best_val_acc: Optional[float]
    eval_acc: Optional[float]

    @property
    def weights(self) -> Path:
        return self.dir / "best_model.pth"

    def label(self) -> str:
        """Display label for a picker entry."""
        bits = [self.name]
        if self.created:
            bits.append(self.created)
        acc = self.eval_acc if self.eval_acc is not None else self.best_val_acc
        suffix = f" (acc {acc:.2f})" if isinstance(acc, (int, float)) else ""
        return " · ".join(bits) + suffix


def _read_config(d: Path) -> dict:
    cfg = d / "training_config.json"
    if cfg.exists():
        try:
            return json.loads(cfg.read_text())
        except Exception:  # noqa: BLE001
            return {}
    return {}


def list_trained_models() -> list[TrainedModel]:
    """All trained models under the root, newest first."""
    root = trained_models_root()
    if not root.exists():
        return []
    out: list[TrainedModel] = []
    for d in sorted(root.glob("*"), reverse=True):
        if not (d / "best_model.pth").exists():
            continue
        cfg = _read_config(d)
        prov = cfg.get("provenance") or {}
        name, created = _split_dir_name(d.name)
        out.append(TrainedModel(
            dir=d,
            name=str(prov.get("model_name") or name),
            created=created,
            best_val_acc=cfg.get("best_val_acc"),
            eval_acc=cfg.get("eval_acc"),
        ))
    return out


def list_model_names() -> list[str]:
    """Distinct, user-meaningful model names (most-recent first)."""
    seen: set[str] = set()
    names: list[str] = []
    for m in list_trained_models():
        if m.name and m.name != "trained" and m.name not in seen:
            seen.add(m.name)
            names.append(m.name)
    return names
