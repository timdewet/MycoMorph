"""SourcePanel — pick an ``all_crops.h5`` + labels dir, show summary stats."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ...core.crops_h5 import HDF5CropsReader
from ...core.labels_store import default_labels_dir_for


class SourcePanel(QGroupBox):
    """Top-of-window file picker. Emits ``sourceChanged`` whenever the
    HDF5 or labels-dir selection changes."""

    sourceChanged = pyqtSignal(object, object)  # (h5_path | None, labels_dir | None)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__("Source", parent)
        self._h5_path: Optional[Path] = None
        self._labels_dir: Optional[Path] = None
        self._build_ui()

    # ── UI ─────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 18, 12, 12)
        layout.setSpacing(6)

        # HDF5 row
        row_h5 = QHBoxLayout()
        row_h5.addWidget(QLabel("HDF5 crops:"))
        self._h5_edit = QLineEdit()
        self._h5_edit.setReadOnly(True)
        self._h5_edit.setPlaceholderText("(pick an all_crops.h5 to begin)")
        row_h5.addWidget(self._h5_edit, stretch=1)
        browse_h5 = QPushButton("Browse…")
        browse_h5.clicked.connect(self.browse_h5)
        row_h5.addWidget(browse_h5)
        layout.addLayout(row_h5)

        # Labels-dir row
        row_lbl = QHBoxLayout()
        row_lbl.addWidget(QLabel("Labels dir:"))
        self._lbl_edit = QLineEdit()
        self._lbl_edit.setReadOnly(True)
        self._lbl_edit.setPlaceholderText("(auto: <HDF5 parent>/modelprep_labels)")
        row_lbl.addWidget(self._lbl_edit, stretch=1)
        browse_lbl = QPushButton("Browse…")
        browse_lbl.clicked.connect(self._browse_labels_dir)
        row_lbl.addWidget(browse_lbl)
        layout.addLayout(row_lbl)

        # Info
        self._info_label = QLabel("")
        self._info_label.setObjectName("muted")
        self._info_label.setWordWrap(True)
        layout.addWidget(self._info_label)

    # ── public ─────────────────────────────────────────────────────────

    def current_h5_path(self) -> Optional[Path]:
        return self._h5_path

    def current_labels_dir(self) -> Optional[Path]:
        return self._labels_dir

    def set_h5_path(self, path: Path) -> None:
        self._set_h5(path)

    # ── actions ────────────────────────────────────────────────────────

    def browse_h5(self) -> None:
        start = str(self._h5_path.parent) if self._h5_path else ""
        path, _ = QFileDialog.getOpenFileName(
            self, "Pick an all_crops.h5 file", start,
            "HDF5 files (*.h5 *.hdf5)",
        )
        if path:
            self._set_h5(Path(path))

    def _browse_labels_dir(self) -> None:
        start = str(self._labels_dir) if self._labels_dir else (
            str(self._h5_path.parent) if self._h5_path else ""
        )
        path = QFileDialog.getExistingDirectory(
            self, "Pick a labels directory", start,
        )
        if path:
            self._labels_dir = Path(path)
            self._lbl_edit.setText(str(self._labels_dir))
            self.sourceChanged.emit(self._h5_path, self._labels_dir)

    # ── internals ──────────────────────────────────────────────────────

    def _set_h5(self, path: Path) -> None:
        if not path.exists():
            self._info_label.setText(f"File not found: {path}")
            return
        try:
            with HDF5CropsReader(path) as reader:
                summary = reader.summary()
        except Exception as e:
            self._info_label.setText(f"Could not open {path.name}: {e}")
            return

        self._h5_path = path
        self._h5_edit.setText(str(path))
        self._labels_dir = default_labels_dir_for(path)
        self._lbl_edit.setText(str(self._labels_dir))
        self._info_label.setText(summary)
        self.sourceChanged.emit(self._h5_path, self._labels_dir)
