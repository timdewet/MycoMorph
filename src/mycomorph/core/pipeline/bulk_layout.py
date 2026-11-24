"""Pandas-backed model for non-plate (bulk) runs over CZI or TIFF files.

One row per input file. Rows sharing the same labels (condition,
reporter, mutant/drug, replica) are pooled into one sample: each file's
fields of view are combined into a single output TIFF. Mirrors the
public surface of `PlateLayout` so the rest of the pipeline (filename
templates, `active_rows()`, `validate()`) behaves the same way
regardless of source.

The path column keeps its historical name ``czi_path`` so existing
``bulk_batch.csv`` files load unchanged; it may point at a TIFF.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from mycomorph.core.focus.io_source import SUPPORTED_SUFFIXES

COLUMNS = [
    "czi_path",
    "condition",
    "reporter",
    "mutant_or_drug",
    "replica",
    "notes",
]


def _output_filename(
    condition: str,
    reporter: str,
    mutant: str,
    replica: str,
    *,
    stable_identifier: str | None = None,
) -> str:
    """The same template all stage adapters expect (matches PlateLayout)."""
    from mycomorph.core.naming import build_output_filename
    return build_output_filename(
        condition, reporter, mutant, replica,
        stable_identifier=stable_identifier,
    )


def sample_stem(entry) -> str:
    """Output stem (no extension) for a batch row; shared by pooled rows."""
    return _output_filename(
        str(entry.get("condition", "")),
        str(entry.get("reporter", "")),
        str(entry.get("mutant_or_drug", "")),
        str(entry.get("replica", "")),
    ).removesuffix(".tif")


def group_entries(entries: Iterable[Any]) -> list[tuple[str, list[Path]]]:
    """Pool batch rows into samples: ``[(stem, [paths…]), …]``.

    Rows whose labels produce the same output stem form one sample, in
    first-appearance order, with their files in row order.
    """
    groups: dict[str, list[Path]] = {}
    for entry in entries:
        groups.setdefault(sample_stem(entry), []).append(Path(str(entry["czi_path"])))
    return list(groups.items())


def _is_input_file(path: Path) -> bool:
    return (
        path.is_file()
        and path.suffix.lower() in SUPPORTED_SUFFIXES
        and not path.name.startswith(".")
    )


@dataclass
class BulkLayout:
    df: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=COLUMNS))

    # ----------------------------------------------------------------- builders

    @classmethod
    def empty(cls) -> "BulkLayout":
        return cls(df=pd.DataFrame(columns=COLUMNS))

    @classmethod
    def from_csv(cls, csv_path: Path) -> "BulkLayout":
        df = pd.read_csv(csv_path, dtype=str).fillna("")
        for c in COLUMNS:
            if c not in df.columns:
                df[c] = ""
        return cls(df=df[COLUMNS].copy())

    # ------------------------------------------------------------------ writers

    def to_csv(self, csv_path: Path) -> None:
        self.df.to_csv(csv_path, index=False, quoting=csv.QUOTE_MINIMAL)

    # ------------------------------------------------------------ row mutations

    def add_files(self, paths: list[Path]) -> int:
        """Append files that aren't already in the layout. Returns # added."""
        existing = set(self.df["czi_path"].astype(str))
        new = []
        for p in paths:
            sp = str(Path(p))
            if sp in existing:
                continue
            existing.add(sp)
            row = {c: "" for c in COLUMNS}
            row["czi_path"] = sp
            new.append(row)
        if not new:
            return 0
        self.df = pd.concat([self.df, pd.DataFrame(new, columns=COLUMNS)], ignore_index=True)
        return len(new)

    def add_folder(self, folder: Path) -> int:
        """Recursively add every CZI / TIFF under `folder`."""
        return self.add_files(sorted(p for p in Path(folder).rglob("*") if _is_input_file(p)))

    def remove_rows(self, indices: list[int]) -> int:
        if not indices:
            return 0
        keep = [i for i in range(len(self.df)) if i not in set(indices)]
        self.df = self.df.iloc[keep].reset_index(drop=True)
        return len(indices)

    def set_single(self, czi_path: Path, **fields: str) -> None:
        """Single-file mode: collapse to a single row, overwrite labels."""
        row = {c: "" for c in COLUMNS}
        row["czi_path"] = str(Path(czi_path))
        for k, v in fields.items():
            if k in COLUMNS:
                row[k] = v
        self.df = pd.DataFrame([row], columns=COLUMNS)

    # ----------------------------------------------------------------- queries

    def active_rows(self) -> pd.DataFrame:
        """Rows the runner should actually process — those with a condition."""
        if self.df.empty:
            return self.df.copy()
        mask = self.df["condition"].astype(str).str.strip().str.len() > 0
        return self.df[mask].copy()

    def samples(self) -> list[tuple[str, list[Path]]]:
        """Active rows pooled into samples; see :func:`group_entries`."""
        return group_entries(self.active_rows().to_dict(orient="records"))

    def validate(self) -> list[str]:
        """Return human-readable errors that should block a run."""
        issues: list[str] = []
        active = self.active_rows()
        if active.empty:
            issues.append("No files have a condition assigned — nothing to process.")
            return issues

        for _, r in active.iterrows():
            p = Path(str(r["czi_path"]))
            if not p.exists():
                issues.append(f"File not found: {p}")
            elif p.suffix.lower() not in SUPPORTED_SUFFIXES:
                issues.append(f"Unsupported file type (expected CZI or TIFF): {p.name}")

        # Rows sharing labels are pooled; the same file twice in one
        # sample would double-count its fields of view.
        for stem, paths in self.samples():
            seen: set[str] = set()
            for p in paths:
                key = str(p.resolve()) if p.exists() else str(p)
                if key in seen:
                    issues.append(f"'{p.name}' is listed twice for sample '{stem}'.")
                seen.add(key)
        return issues

    # ------------------------------------------------------------- conveniences

    def output_label_for(self, row) -> str:
        """Stem (no extension) used as Focus output filename for `row`."""
        return sample_stem(row)
