"""Restrict labelling to cells from chosen genes / conditions / wells.

A :class:`CellFilter` maps metadata fields to the set of values to keep. Values
within a field are OR'd (exact string match); fields are AND'd. A field that is
absent or has no values places no restriction, so an empty filter is "all
cells". Example — the uninduced controls of an M. smegmatis plate::

    CellFilter.from_mapping({
        "gene": ["backbone_275", "cntl2_276", "cntl3_277"],
        "condition": ["ATc+MMC-"],
    })

Field → HDF5 dataset (see ``crops_h5`` for how MycoMorph fills them):

    gene             genes             mutant/knockdown, "" for control-like wells
    condition        conditions        treatment, e.g. "ATc+" or "ATc+MMC-"
    condition_label  condition_labels  mutant/drug name, else the condition
    reporter         reporters
    drug             drugs
    tiff_file        tiff_files        one per well / acquisition
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Optional

import numpy as np


# field key → (HDF5 dataset, display name). Order is the display order.
FILTER_FIELDS: dict[str, tuple[str, str]] = {
    "gene": ("genes", "Gene"),
    "condition": ("conditions", "Condition"),
    "condition_label": ("condition_labels", "Condition label"),
    "reporter": ("reporters", "Reporter"),
    "drug": ("drugs", "Drug"),
    "tiff_file": ("tiff_files", "TIFF / well"),
}


def _natural_key(s: str):
    """Sort "NT2" before "NT10" and "MSMEG_4482_41" before "MSMEG_4482_100"."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def distinct_values(reader, field_key: str) -> list[tuple[str, int]]:
    """``[(value, n_cells), ...]`` for one filter field, naturally sorted."""
    dataset = FILTER_FIELDS[field_key][0]
    values, counts = np.unique(reader.metadata(dataset).astype(str), return_counts=True)
    pairs = [(str(v), int(c)) for v, c in zip(values, counts)]
    return sorted(pairs, key=lambda vc: _natural_key(vc[0]))


@dataclass(frozen=True)
class CellFilter:
    """Which cells may be sampled. Build with :meth:`from_mapping`."""

    values: Mapping[str, frozenset[str]] = field(default_factory=dict)

    @classmethod
    def from_mapping(
        cls, mapping: Optional[Mapping[str, Iterable[str]]] = None,
    ) -> "CellFilter":
        """Validate field names and drop fields with no values."""
        clean: dict[str, frozenset[str]] = {}
        for key, vals in (mapping or {}).items():
            if key not in FILTER_FIELDS:
                raise ValueError(
                    f"Unknown filter field {key!r}; expected one of {list(FILTER_FIELDS)}"
                )
            vals = frozenset(str(v) for v in vals)
            if vals:
                clean[key] = vals
        return cls(values=clean)

    def is_empty(self) -> bool:
        return not self.values

    def mask(self, reader) -> np.ndarray:
        """Boolean mask over the reader's cells: True where every restricted
        field's value is one of the chosen values."""
        keep = np.ones(len(reader), dtype=bool)
        for key, vals in self.values.items():
            column = reader.metadata(FILTER_FIELDS[key][0]).astype(str)
            keep &= np.isin(column, sorted(vals))
        return keep

    def unknown_values(self, reader) -> dict[str, list[str]]:
        """Chosen values that no cell in ``reader`` has (likely typos), by field."""
        out: dict[str, list[str]] = {}
        for key, vals in self.values.items():
            present = {v for v, _ in distinct_values(reader, key)}
            missing = sorted(vals - present, key=_natural_key)
            if missing:
                out[key] = missing
        return out

    def to_dict(self) -> dict[str, list[str]]:
        """JSON-friendly form (fields in display order, values sorted)."""
        return {key: sorted(self.values[key], key=_natural_key)
                for key in FILTER_FIELDS if key in self.values}

    def describe(self) -> str:
        """Human-readable form, e.g. ``gene ∈ {dnaN, inhA} AND condition = ATc+``."""
        if self.is_empty():
            return "all cells"
        parts = []
        for key, vals in self.to_dict().items():
            shown = [v if v else "(blank)" for v in vals]
            parts.append(f"{key} = {shown[0]}" if len(shown) == 1
                         else f"{key} ∈ {{{', '.join(shown)}}}")
        return " AND ".join(parts)
