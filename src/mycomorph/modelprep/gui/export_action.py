""""Export for MycoMorph…" — shared by the Fine-tune page and the Models tab."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PyQt6.QtWidgets import QFileDialog, QMessageBox, QWidget

from ..core.export import export_for_mycomorph

EXPORT_TOOLTIP = (
    "Copy this model's weights (best_model.pth) and its training record to a "
    "folder you choose, under a readable name, ready for MycoMorph's "
    "Classify > Custom model."
)


def export_run_for_mycomorph(parent: QWidget, run_dir: Path) -> Optional[Path]:
    """Ask for a destination folder, export ``run_dir`` there, and report.
    Returns the exported ``.pth`` (None if cancelled or failed)."""
    docs = Path.home() / "Documents"
    dest = QFileDialog.getExistingDirectory(
        parent, "Export for MycoMorph: choose a folder",
        str(docs if docs.is_dir() else Path.home()),
    )
    if not dest:
        return None
    try:
        pth = export_for_mycomorph(run_dir, Path(dest))
    except (ValueError, OSError) as e:
        QMessageBox.warning(parent, "Export failed", str(e))
        return None
    QMessageBox.information(
        parent, "Exported for MycoMorph",
        f"Saved {pth.name}\nin {pth.parent}\n\n"
        "In MycoMorph: Classify > Custom model > choose this file. Keep the "
        "files beside it: MycoMorph reads training_config.json from the same folder.",
    )
    return pth
