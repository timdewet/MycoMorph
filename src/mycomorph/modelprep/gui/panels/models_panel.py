"""Models comparison tab — list every trained run and compare performance.

Reads the runs under ``~/.modelprep/models/`` (via
``trained_models.list_trained_models``) into a sortable table of metrics, with
the per-model ROC/PR curves shown below (reusing ``ModelInspector``). Actions:
use a model for labelling, open its run folder, or delete a run.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import Qt, QUrl, pyqtSignal
from PyQt6.QtGui import QColor, QDesktopServices
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..trained_models import TrainedModel, list_trained_models, trained_models_root
from ..ui import tokens
from ..widgets.model_inspector import ModelInspector

# Column order. Numeric columns get numeric-aware sort items.
_COLUMNS = [
    "Name", "Created", "Eval on", "Acc", "Balanced", "AUC", "AP",
    "Baseline", "Thr", "Train", "Test/hold-out", "good/bad",
]
_COL = {name: i for i, name in enumerate(_COLUMNS)}
# Columns whose best (max) value is highlighted.
_HIGHLIGHT = ("AUC", "Balanced")
_DIR_ROLE = Qt.ItemDataRole.UserRole


class _NumItem(QTableWidgetItem):
    """Table cell that sorts by a numeric key rather than its display text."""

    def __init__(self, text: str, key: float) -> None:
        super().__init__(text)
        self._key = key
        self.setFlags(self.flags() & ~Qt.ItemFlag.ItemIsEditable)
        self.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

    def __lt__(self, other: object) -> bool:
        if isinstance(other, _NumItem):
            return self._key < other._key
        return super().__lt__(other)  # type: ignore[arg-type]


def _opt_float(v) -> Optional[float]:
    try:
        f = float(v)
        return f if f == f else None  # drop NaN
    except (TypeError, ValueError):
        return None


def _num_cell(v, fmt: str = "{:.3f}") -> _NumItem:
    """Numeric cell; missing/NaN renders as '—' and sorts to the bottom."""
    f = _opt_float(v)
    if f is None:
        return _NumItem("—", float("-inf"))
    return _NumItem(fmt.format(f), f)


def _int_cell(v) -> _NumItem:
    f = _opt_float(v)
    if f is None:
        return _NumItem("—", float("-inf"))
    return _NumItem(f"{int(f):,}", f)


def _text_cell(text: str) -> QTableWidgetItem:
    item = QTableWidgetItem(text)
    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
    return item


class ModelsPanel(QWidget):
    """A sortable table of trained models + curves for the selected one."""

    useModelRequested = pyqtSignal(object)  # Path to the .pth
    modelsChanged = pyqtSignal()            # after a delete

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._models: list[TrainedModel] = []
        self._build_ui()
        self.refresh()

    # ── UI ─────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(tokens.S3)

        # Action bar.
        bar = QHBoxLayout()
        self._refresh_btn = QPushButton("Refresh")
        self._refresh_btn.clicked.connect(self.refresh)
        self._use_btn = QPushButton("Use for labelling")
        self._use_btn.setObjectName("primary")
        self._use_btn.clicked.connect(self._use_selected)
        self._open_btn = QPushButton("Open folder")
        self._open_btn.clicked.connect(self._open_selected)
        self._delete_btn = QPushButton("Delete run")
        self._delete_btn.clicked.connect(self._delete_selected)
        bar.addWidget(self._refresh_btn)
        bar.addStretch(1)
        bar.addWidget(self._use_btn)
        bar.addWidget(self._open_btn)
        bar.addWidget(self._delete_btn)
        root.addLayout(bar)

        # Empty-state placeholder (shown when there are no runs).
        self._empty = QLabel("No trained models yet — train one on the Fine-tune tab.")
        self._empty.setObjectName("muted")
        self._empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root.addWidget(self._empty)

        # Table (top) + inspector (bottom) in a splitter.
        self._splitter = QSplitter(Qt.Orientation.Vertical)
        self._table = QTableWidget(0, len(_COLUMNS))
        self._table.setHorizontalHeaderLabels(_COLUMNS)
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.setAlternatingRowColors(True)
        self._table.setSortingEnabled(True)
        self._table.verticalHeader().setVisible(False)
        hdr = self._table.horizontalHeader()
        hdr.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(_COL["Name"], QHeaderView.ResizeMode.Stretch)
        self._table.itemSelectionChanged.connect(self._on_selection_changed)
        self._table.itemDoubleClicked.connect(lambda *_: self._use_selected())
        self._splitter.addWidget(self._table)

        self._inspector = ModelInspector()
        self._splitter.addWidget(self._inspector)
        self._splitter.setStretchFactor(0, 3)
        self._splitter.setStretchFactor(1, 2)
        root.addWidget(self._splitter, stretch=1)

    # ── data ───────────────────────────────────────────────────────────

    def refresh(self) -> None:
        """Reload the model list, preserving the selected run if still present."""
        prev_dir = self._selected_dir()
        self._models = list_trained_models()

        self._table.setSortingEnabled(False)
        self._table.setRowCount(0)
        for m in self._models:
            self._add_row(m)
        self._highlight_best()
        self._table.setSortingEnabled(True)

        has_models = bool(self._models)
        self._splitter.setVisible(has_models)
        self._empty.setVisible(not has_models)

        if prev_dir and not self._select_dir(prev_dir):
            self._inspector.set_model(None)
        elif not prev_dir and has_models:
            self._table.selectRow(0)
        self._on_selection_changed()

    def _add_row(self, m: TrainedModel) -> None:
        cfg = m.config or {}
        cc = cfg.get("class_counts") or {}
        # Evaluation set size is the hold-out if present, else the internal test.
        n_holdout = cfg.get("n_holdout") or 0
        n_eval = n_holdout or cfg.get("n_test") or 0

        r = self._table.rowCount()
        self._table.insertRow(r)

        name_item = _text_cell(m.name)
        name_item.setData(_DIR_ROLE, str(m.dir))
        name_item.setToolTip(str(m.dir))

        cells = {
            "Name": name_item,
            "Created": _text_cell(m.created or "—"),
            "Eval on": _text_cell(str(cfg.get("eval_split", "—"))),
            "Acc": _num_cell(cfg.get("eval_acc")),
            "Balanced": _num_cell(cfg.get("balanced_acc")),
            "AUC": _num_cell(cfg.get("roc_auc")),
            "AP": _num_cell(cfg.get("average_precision")),
            "Baseline": _num_cell(cfg.get("majority_baseline")),
            "Thr": _num_cell(cfg.get("decision_threshold"), "{:.2f}"),
            "Train": _int_cell(cfg.get("n_train")),
            "Test/hold-out": _int_cell(n_eval),
            "good/bad": _text_cell(f"{cc.get('good', 0)}/{cc.get('bad', 0)}"),
        }
        for name, item in cells.items():
            self._table.setItem(r, _COL[name], item)

    def _highlight_best(self) -> None:
        """Tint the best (max) cell in each highlighted column."""
        tint = QColor(tokens.active().success)
        tint.setAlpha(48)
        for col_name in _HIGHLIGHT:
            col = _COL[col_name]
            best_row, best_val = -1, float("-inf")
            for r in range(self._table.rowCount()):
                item = self._table.item(r, col)
                if isinstance(item, _NumItem) and item._key > best_val:
                    best_val, best_row = item._key, r
            if best_row >= 0 and best_val > float("-inf"):
                self._table.item(best_row, col).setBackground(tint)

    # ── selection + actions ────────────────────────────────────────────

    def _selected_dir(self) -> Optional[str]:
        rows = self._table.selectionModel().selectedRows() if self._table.selectionModel() else []
        if not rows:
            return None
        item = self._table.item(rows[0].row(), _COL["Name"])
        return item.data(_DIR_ROLE) if item else None

    def _select_dir(self, target: str) -> bool:
        for r in range(self._table.rowCount()):
            item = self._table.item(r, _COL["Name"])
            if item and item.data(_DIR_ROLE) == target:
                self._table.selectRow(r)
                return True
        return False

    def _selected_model(self) -> Optional[TrainedModel]:
        target = self._selected_dir()
        if not target:
            return None
        return next((m for m in self._models if str(m.dir) == target), None)

    def _on_selection_changed(self) -> None:
        m = self._selected_model()
        for btn in (self._use_btn, self._open_btn, self._delete_btn):
            btn.setEnabled(m is not None)
        self._inspector.set_model(m.weights if m else None)

    def _use_selected(self) -> None:
        m = self._selected_model()
        if m is not None:
            self.useModelRequested.emit(m.weights)

    def _open_selected(self) -> None:
        m = self._selected_model()
        if m is not None and m.dir.exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(m.dir)))

    def _delete_selected(self) -> None:
        m = self._selected_model()
        if m is None:
            return
        ans = QMessageBox.question(
            self, "Delete run",
            f"Permanently delete this trained run?\n\n{m.name} · {m.created}\n{m.dir}",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        if ans != QMessageBox.StandardButton.Yes:
            return
        # Safety: only ever delete inside the models root.
        run = m.dir.resolve()
        root = trained_models_root().resolve()
        if root not in run.parents:
            QMessageBox.warning(self, "Refused",
                                f"Run directory is outside {root}; not deleting.")
            return
        try:
            shutil.rmtree(run)
        except Exception as e:  # noqa: BLE001
            QMessageBox.warning(self, "Delete failed", str(e))
            return
        self.refresh()
        self.modelsChanged.emit()
