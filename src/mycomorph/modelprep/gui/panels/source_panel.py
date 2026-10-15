"""SourcePanel — pick an ``all_crops.h5`` + labels dir, show summary stats."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
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
from ..trained_models import list_model_names


class SourcePanel(QGroupBox):
    """Top-of-window file picker. Emits ``sourceChanged`` whenever the
    HDF5 or labels-dir selection changes, and ``modelNameChanged`` when the
    model name is edited."""

    sourceChanged = pyqtSignal(object, object)  # (h5_path | None, labels_dir | None)
    modelNameChanged = pyqtSignal(str)

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

        # Model-name row — names the trained model. A human-readable timestamp
        # is appended at train time, so each run lands in <name>_<time>/. Pick
        # an existing name to add a new dated version, or type a new one.
        row_model = QHBoxLayout()
        row_model.addWidget(QLabel("Model name:"))
        self._model_name_combo = QComboBox()
        self._model_name_combo.setEditable(True)
        self._model_name_combo.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self._model_name_combo.lineEdit().setPlaceholderText(
            "(name this model, e.g. mtb-v2 — a date is added automatically)"
        )
        row_model.addWidget(self._model_name_combo, stretch=1)
        self._model_ts_hint = QLabel("")
        self._model_ts_hint.setObjectName("caption")
        row_model.addWidget(self._model_ts_hint)
        layout.addLayout(row_model)
        # Connect + populate only after both widgets exist (refresh touches the
        # hint label).
        self._model_name_combo.currentTextChanged.connect(self._on_name_changed)
        self.refresh_models()

        # Info
        self._info_label = QLabel("")
        self._info_label.setObjectName("muted")
        self._info_label.setWordWrap(True)
        layout.addWidget(self._info_label)

        self._warn_label = QLabel("")
        self._warn_label.setWordWrap(True)
        self._warn_label.setVisible(False)
        layout.addWidget(self._warn_label)

    # ── public ─────────────────────────────────────────────────────────

    def current_h5_path(self) -> Optional[Path]:
        return self._h5_path

    def current_labels_dir(self) -> Optional[Path]:
        return self._labels_dir

    def set_h5_path(self, path: Path) -> None:
        self._set_h5(path)

    # ── model name ──────────────────────────────────────────────────────

    def current_model_name(self) -> str:
        return self._model_name_combo.currentText().strip()

    def set_model_name(self, name: str) -> None:
        self._model_name_combo.setCurrentText(name or "")

    def refresh_models(self) -> None:
        """Repopulate the existing-model-name suggestions, preserving the
        current text."""
        current = self._model_name_combo.currentText()
        self._model_name_combo.blockSignals(True)
        self._model_name_combo.clear()
        self._model_name_combo.addItems(list_model_names())
        self._model_name_combo.setCurrentText(current)
        self._model_name_combo.blockSignals(False)
        self._update_ts_hint(current)

    def _on_name_changed(self, text: str) -> None:
        self._update_ts_hint(text)
        self.modelNameChanged.emit(text)

    def _update_ts_hint(self, text: str) -> None:
        from ..trained_models import slugify
        self._model_ts_hint.setText(f"→ {slugify(text)}_<date>")

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

    def _show_crop_warnings(self, path: Path) -> None:
        """Flag crops that won't match what MycoMorph classifies (background
        masking) or that the previous model pre-filtered."""
        from ...core.crop_provenance import crop_warnings
        from ..ui import tokens

        try:
            warnings = crop_warnings(path)
        except Exception:  # noqa: BLE001 — a check, never a blocker
            warnings = []
        self._warn_label.setVisible(bool(warnings))
        self._warn_label.setText("\n".join(f"⚠ {w}" for w in warnings))
        self._warn_label.setStyleSheet(f"color: {tokens.active().warning};")

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
        self._show_crop_warnings(path)
        self.sourceChanged.emit(self._h5_path, self._labels_dir)
