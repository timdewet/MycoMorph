"""Reading plate layouts from CSV."""

from __future__ import annotations

from pathlib import Path

from mycomorph.core.pipeline.layout import PlateLayout


def test_layout_csv_needs_only_wells_and_labels(tmp_path: Path):
    """A hand-written CSV has no ``scene_indices`` or ``source_czi`` column;
    those are filled in from the CZI. It used to fail to import."""
    csv = tmp_path / "layout.csv"
    csv.write_text(
        "well,condition,reporter,mutant_or_drug,replica\n"
        "A01,ATc+,GFP,rpoB,1\n"
        "B2,ATc-,GFP,NT,2\n"
    )
    layout = PlateLayout.from_csv(csv)

    active = layout.active_rows()
    assert list(active["well"]) == ["A1", "B2"]
    assert list(active["mutant_or_drug"]) == ["rpoB", "NT"]
    assert all(scenes == [] for scenes in layout.df["scene_indices"])
    assert (layout.rows, layout.cols) == (2, 2)

