"""Models comparison tab — list every trained run and compare performance.

Reads the runs under ``~/.modelprep/models/`` (via
``trained_models.list_trained_models``) into a sortable table of metrics, with
the per-model ROC/PR curves shown below (reusing ``ModelInspector``). Actions:
use a model for labelling, open its run folder, export it for MycoMorph, or
delete a run.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import Qt, QThread, QUrl, pyqtSignal
from PyQt6.QtGui import QColor, QDesktopServices
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ...core.crops_h5 import HDF5CropsReader
from ...core.labels_store import SPLIT_TEST, LabelsStore
from ..trained_models import (
    TrainedModel,
    human_timestamp,
    list_trained_models,
    trained_models_root,
)
from ..export_action import EXPORT_TOOLTIP, export_run_for_mycomorph
from ..ui import tokens
from ..widgets.model_inspector import ModelInspector

# Column order. Numeric columns get numeric-aware sort items.
_COLUMNS = [
    "Name", "Type", "Created", "Eval on", "Acc", "Balanced", "AUC", "AP",
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


def _fmt(v) -> str:
    f = _opt_float(v)
    return "—" if f is None else f"{f:.3f}"


class _LearningCurveWorker(QThread):
    """Train on increasing fractions of the train split, scoring each on the
    frozen hold-out, off the UI thread."""

    progress = pyqtSignal(float, str)
    finished = pyqtSignal(object, str)  # list[point dict], error

    def __init__(self, h5_path: Path, labels_dir: Path, epochs: int, n_points: int) -> None:
        super().__init__()
        self.h5_path = h5_path
        self.labels_dir = labels_dir
        self.epochs = epochs
        self.n_points = n_points
        self._cancel = False

    def cancel(self) -> None:
        self._cancel = True

    def run(self) -> None:
        try:
            from ...core.learning_curve import compute_learning_curve

            reader = HDF5CropsReader(self.h5_path)
            store = LabelsStore(self.labels_dir)
            try:
                tr_crops, tr_labels, _ = store.to_training_dataset(reader)
                ev_crops, ev_labels, _ = store.to_eval_dataset(reader)
            finally:
                reader.close()

            if len(ev_labels) == 0:
                self.finished.emit([], "No hold-out test set — build one on the Label cells tab.")
                return
            if len(set(ev_labels.tolist())) < 2:
                self.finished.emit([], "The hold-out has only one class, so AUC/AP can't "
                                       "be computed. Label both good and bad cells into it.")
                return

            points = compute_learning_curve(
                tr_crops, tr_labels, ev_crops, ev_labels,
                n_points=self.n_points, epochs=self.epochs,
                progress_cb=lambda f, m: self.progress.emit(f, m),
                should_cancel=lambda: self._cancel,
            )
            if self._cancel and not points:
                self.finished.emit([], "Cancelled.")
                return
            self.finished.emit(points, "")
        except Exception as e:  # noqa: BLE001
            self.finished.emit([], str(e))


class ModelsPanel(QWidget):
    """A sortable table of trained models + curves for the selected one."""

    useModelRequested = pyqtSignal(object)  # Path to the .pth
    modelsChanged = pyqtSignal()            # after a delete

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._models: list[TrainedModel] = []
        self._h5_path: Optional[Path] = None
        self._labels_dir: Optional[Path] = None
        self._lc_worker: Optional[_LearningCurveWorker] = None
        self._lc_canvas = None
        self._lc_fig = None
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
        self._export_btn = QPushButton("Export for MycoMorph…")
        self._export_btn.clicked.connect(self._export_selected)
        self._delete_btn = QPushButton("Delete run")
        self._delete_btn.clicked.connect(self._delete_selected)
        bar.addWidget(self._refresh_btn)
        bar.addStretch(1)
        bar.addWidget(self._use_btn)
        bar.addWidget(self._open_btn)
        bar.addWidget(self._export_btn)
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

        self._build_learning_curve(root)

    # ── learning curve ─────────────────────────────────────────────────

    def _build_learning_curve(self, root: QVBoxLayout) -> None:
        # Disclosure toggle (collapsed by default — keeps the tab clean).
        self._lc_toggle = QPushButton("Learning curve ▸")
        self._lc_toggle.setObjectName("ghost")
        self._lc_toggle.setCheckable(True)
        self._lc_toggle.setToolTip(
            "Train on increasing fractions of the labelled train split and score "
            "each on the frozen hold-out — shows whether more labelling still helps."
        )
        self._lc_toggle.toggled.connect(self._on_lc_toggled)
        root.addWidget(self._lc_toggle)

        self._lc_box = QFrame()
        self._lc_box.setObjectName("card")
        self._lc_box.setVisible(False)
        v = QVBoxLayout(self._lc_box)
        v.setContentsMargins(8, 8, 8, 8)
        v.setSpacing(6)

        ctrls = QHBoxLayout()
        self._lc_epochs = QSpinBox(); self._lc_epochs.setRange(5, 200); self._lc_epochs.setValue(25)
        self._lc_epochs.setToolTip("Epochs per point (lower = faster, less faithful).")
        self._lc_points = QSpinBox(); self._lc_points.setRange(3, 8); self._lc_points.setValue(5)
        self._lc_points.setToolTip("Number of training-set sizes to sample along the curve.")
        self._lc_run = QPushButton("Run learning curve")
        self._lc_run.setObjectName("primary")
        self._lc_run.clicked.connect(self._run_curve)
        self._lc_cancel = QPushButton("Cancel")
        self._lc_cancel.clicked.connect(self._cancel_curve)
        self._lc_cancel.setEnabled(False)
        ctrls.addWidget(QLabel("Epochs:")); ctrls.addWidget(self._lc_epochs)
        ctrls.addWidget(QLabel("Points:")); ctrls.addWidget(self._lc_points)
        ctrls.addWidget(self._lc_run)
        ctrls.addWidget(self._lc_cancel)
        ctrls.addStretch(1)
        v.addLayout(ctrls)

        self._lc_progress = QProgressBar()
        self._lc_progress.setRange(0, 1000); self._lc_progress.setValue(0)
        v.addWidget(self._lc_progress)

        self._lc_status = QLabel(
            "Trains a model per size from scratch (slow) and scores each on the "
            "hold-out. Absolute values may sit below your fine-tuned models — the "
            "shape / plateau is the signal."
        )
        self._lc_status.setObjectName("muted")
        self._lc_status.setWordWrap(True)
        v.addWidget(self._lc_status)

        # Live plot (matplotlib, lazy like ModelInspector).
        try:
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
            from matplotlib.figure import Figure
            self._lc_fig = Figure(figsize=(5.2, 2.4), tight_layout=True, facecolor="#ffffff")
            self._lc_canvas = FigureCanvasQTAgg(self._lc_fig)
            self._lc_canvas.setMinimumHeight(240)
            self._lc_canvas.setMaximumWidth(760)
            self._lc_canvas.setVisible(False)
            v.addWidget(self._lc_canvas, alignment=Qt.AlignmentFlag.AlignLeft)
        except Exception:  # noqa: BLE001
            self._lc_canvas = None

        root.addWidget(self._lc_box)
        self._update_lc_enabled()

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
            "Type": _text_cell("features" if m.model_type == "features" else "CNN"),
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
        for btn in (self._open_btn, self._delete_btn):
            btn.setEnabled(m is not None)
        # Active learning scores crops with a CNN, so a tabular feature model
        # can't be used as the labelling model.
        is_cnn = m is not None and m.model_type != "features"
        self._use_btn.setEnabled(is_cnn)
        self._use_btn.setToolTip(
            "" if is_cnn else "Feature models can't drive the crop-based labeller."
        )
        self._export_btn.setEnabled(is_cnn)
        self._export_btn.setToolTip(
            EXPORT_TOOLTIP if is_cnn or m is None
            else "Feature models can't be used in MycoMorph; only CNN runs export."
        )
        self._inspector.set_model(m.weights if m else None)

    def _use_selected(self) -> None:
        m = self._selected_model()
        if m is not None:
            self.useModelRequested.emit(m.weights)

    def _open_selected(self) -> None:
        m = self._selected_model()
        if m is not None and m.dir.exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(m.dir)))

    def _export_selected(self) -> None:
        m = self._selected_model()
        if m is not None:
            export_run_for_mycomorph(self, m.dir)

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

    # ── learning curve: source + run ────────────────────────────────────

    def set_source(self, h5_path: Optional[Path], labels_dir: Optional[Path]) -> None:
        """Point the learning-curve tool at the active dataset."""
        self._h5_path = Path(h5_path) if h5_path else None
        self._labels_dir = Path(labels_dir) if labels_dir else None
        self._update_lc_enabled()

    def _has_holdout(self) -> bool:
        if not (self._h5_path and self._labels_dir):
            return False
        try:
            te = LabelsStore(self._labels_dir).split_counts().get(SPLIT_TEST, {})
            return (te.get("good", 0) + te.get("bad", 0)) > 0
        except Exception:  # noqa: BLE001
            return False

    def _update_lc_enabled(self) -> None:
        running = self._lc_worker is not None and self._lc_worker.isRunning()
        has = self._has_holdout()
        self._lc_run.setEnabled(has and not running)
        self._lc_run.setToolTip(
            "" if has else "Build a hold-out test set first (Label cells tab)."
        )

    def _on_lc_toggled(self, checked: bool) -> None:
        self._lc_box.setVisible(checked)
        self._lc_toggle.setText("Learning curve ▾" if checked else "Learning curve ▸")
        if checked:
            self._update_lc_enabled()

    def _run_curve(self) -> None:
        if not self._has_holdout():
            return
        self._lc_worker = _LearningCurveWorker(
            self._h5_path, self._labels_dir,
            self._lc_epochs.value(), self._lc_points.value(),
        )
        self._lc_worker.progress.connect(self._on_curve_progress)
        self._lc_worker.finished.connect(self._on_curve_finished)
        self._lc_run.setEnabled(False)
        self._lc_cancel.setEnabled(True)
        self._lc_progress.setValue(0)
        self._lc_status.setText("Running… training a model per size; this can take a while.")
        self._lc_worker.start()

    def _cancel_curve(self) -> None:
        if self._lc_worker is not None and self._lc_worker.isRunning():
            self._lc_worker.cancel()
            self._lc_status.setText("Cancelling after the current size finishes…")
            self._lc_cancel.setEnabled(False)

    def _on_curve_progress(self, frac: float, msg: str) -> None:
        if self.sender() is not self._lc_worker:
            return
        self._lc_progress.setValue(int(max(0.0, min(1.0, frac)) * 1000))
        self._lc_status.setText(msg)

    def _on_curve_finished(self, points: object, error: str) -> None:
        if self.sender() is not self._lc_worker:
            return
        self._lc_cancel.setEnabled(False)
        self._update_lc_enabled()
        self._lc_progress.setValue(1000)
        if error:
            self._lc_status.setText(f"Learning curve failed: {error}")
            return
        points = list(points or [])
        if not points:
            self._lc_status.setText("No curve produced — too few labels to train any size.")
            return
        self._plot_curve(points)
        saved = self._save_curve(points)
        last = points[-1]
        msg = (f"Done · {len(points)} sizes · final n={last['n']:,} "
               f"AUC {_fmt(last['roc_auc'])} AP {_fmt(last['average_precision'])} "
               f"balanced {_fmt(last['balanced_acc'])}")
        if saved:
            msg += f"  →  {saved}"
        self._lc_status.setText(msg)

    def _plot_curve(self, points: list[dict]) -> None:
        if self._lc_canvas is None:
            return
        ns = [p["n"] for p in points]
        self._lc_fig.clear()
        ax = self._lc_fig.add_subplot(111)
        series = [
            ("roc_auc", "AUC", "#2166ac"),
            ("average_precision", "AP", "#b2182b"),
            ("balanced_acc", "Balanced acc", "#1a9850"),
        ]
        for key, label, color in series:
            ys = [p.get(key) if p.get(key) is not None else float("nan") for p in points]
            ax.plot(ns, ys, "o-", color=color, lw=1.4, ms=4, label=label)
        ax.set_xlabel("training cells (n)", fontsize=8)
        ax.set_ylabel("score", fontsize=8)
        ax.set_title("Learning curve · evaluated on hold-out", fontsize=9)
        ax.set_ylim(0.5, 1.0)
        ax.grid(True, alpha=0.25)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=7, frameon=False, loc="lower right")
        self._lc_canvas.setVisible(True)
        self._lc_canvas.draw_idle()

    def _save_curve(self, points: list[dict]) -> Optional[str]:
        try:
            stem = self._h5_path.stem if self._h5_path else "curve"
            out = Path.home() / ".modelprep" / "learning_curves" / f"{stem}_{human_timestamp()}"
            out.mkdir(parents=True, exist_ok=True)
            (out / "curve.json").write_text(json.dumps(
                {"h5": str(self._h5_path), "points": points}, indent=2))
            if self._lc_fig is not None:
                self._lc_fig.savefig(out / "curve.png", dpi=150)
            return str(out)
        except Exception:  # noqa: BLE001
            return None

    def shutdown(self) -> None:
        """Cancel + wait for the learning-curve worker (called on app close)."""
        if self._lc_worker is not None and self._lc_worker.isRunning():
            self._lc_worker.cancel()
            self._lc_worker.wait(5000)
