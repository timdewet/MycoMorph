"""Active-learning labeller + classifier fine-tuning panel — ModelPrep.

Workflow:
  1. SourcePanel emits ``sourceChanged(h5_path, labels_dir)`` →
     ``LabelTrainPanel.set_source(...)`` opens the HDF5.
  2. User picks a base classifier (a bundled preset, a previously-trained
     model under ``~/.modelprep/models/``, or "none").
  3. Click "Load pool" — runs the model over every unlabelled cell, ranks
     by uncertainty (|0.5 - P(good)|, optionally diversified across fields of
     view), buffers the top N in memory.
  4. Click a thumbnail to cycle ``unlabeled → good → bad → unlabeled``, or use
     the keyboard on the focused cell: ``g`` = good, ``b`` = bad (both advance
     to the next cell), ``u`` = unlabeled, arrow keys move between cells.
  5. "Save & next page" persists labels to ``labels.csv`` then advances.
  6. The training section fine-tunes the classifier from the labels, writes
     artefacts to ``~/.modelprep/models/trained_<ts>/``, and shows the
     resulting curves + metrics inline.
"""

from __future__ import annotations

import math
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
from PyQt6.QtCore import QSize, Qt, QThread, QTimer, QUrl, pyqtSignal
from PyQt6.QtGui import QDesktopServices
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ...core._render import outline_of, to_outlined_rgb
from ...core.cli import list_preset_models
from ...core.crops_h5 import CropRecord, HDF5CropsReader
from ...core.fov_context import FovContextProvider
from ...core.labels_store import (
    LABEL_TO_IDX,
    LabelRow,
    LabelsStore,
    default_labels_dir_for,
    make_session_id,
)
from ..palette import COLORS, LABEL_COLORS
from ..ui import tokens
from ..widgets.model_inspector import ModelInspector
from .cell_detail_dialog import CellDetailView


# Where GUI-trained models land (kept in sync with _start_training).
def _trained_models_root() -> Path:
    return Path.home() / ".modelprep" / "models"


# ─────────────────────────────────────────────────────────────────────────────
# Active-learning worker (HDF5)
# ─────────────────────────────────────────────────────────────────────────────

class _ActiveLearningWorker(QThread):
    """Score every unlabelled cell in the HDF5, build a ranked pool."""

    progress = pyqtSignal(float, str)
    finished = pyqtSignal(object, str)  # list[CropRecord+score], error

    def __init__(
        self,
        h5_path: Path,
        model_path: Optional[Path],
        n_target: int,
        seen_uids: set[str],
        diversify: bool = True,
        *,
        review: bool = False,
        uncertain_only: bool = False,
        confidence_threshold: float = 0.85,
    ) -> None:
        super().__init__()
        self.h5_path = h5_path
        self.model_path = model_path
        self.n_target = n_target
        self.seen_uids = seen_uids
        self.diversify = diversify
        self.review = review
        self.uncertain_only = uncertain_only
        self.confidence_threshold = confidence_threshold
        self._cancel = False

    def cancel(self) -> None:
        self._cancel = True

    def _diversified(self, reader, ranked_idx, ranked_scores):
        """Spread the most-uncertain candidates across fields of view so a
        single noisy FOV can't dominate a page. Falls back to plain order."""
        try:
            from collections import OrderedDict, deque
            k = min(len(ranked_idx), self.n_target * 5)
            cand = ranked_idx[:k]
            cand_scores = ranked_scores[:k]
            fovs = reader.fov_indices[cand]
            buckets: "OrderedDict[int, deque]" = OrderedDict()
            for j, f in enumerate(fovs):
                buckets.setdefault(int(f), deque()).append(j)
            order: list[int] = []
            queues = list(buckets.values())
            while len(order) < self.n_target and any(queues):
                for q in queues:
                    if q:
                        order.append(q.popleft())
                        if len(order) >= self.n_target:
                            break
            order = order[: self.n_target]
            return cand[order], cand_scores[order]
        except Exception:  # noqa: BLE001 — diversity is best-effort
            return ranked_idx[: self.n_target], ranked_scores[: self.n_target]

    def run(self) -> None:  # noqa: D401
        try:
            reader = HDF5CropsReader(self.h5_path)
        except Exception as e:
            self.finished.emit([], f"Could not open {self.h5_path}: {e}")
            return

        try:
            unseen_mask = np.array(
                [uid not in self.seen_uids for uid in reader.cell_uids],
                dtype=bool,
            )
            unseen_idx = np.flatnonzero(unseen_mask)
            if unseen_idx.size == 0:
                self.finished.emit([], "No unseen cells in this HDF5.")
                return

            if self.model_path is None:
                rng = np.random.default_rng(42)
                rng.shuffle(unseen_idx)
                picked = unseen_idx[: self.n_target]
                scores = np.full(len(picked), 0.5, dtype=np.float32)
            else:
                from ...core.cell_quality_classifier import CellQualityClassifier

                clf = CellQualityClassifier(str(self.model_path), in_channels=3)
                scores_all = np.empty(len(unseen_idx), dtype=np.float32)
                B = 256
                for lo in range(0, len(unseen_idx), B):
                    if self._cancel:
                        self.finished.emit([], "Cancelled.")
                        return
                    hi = min(lo + B, len(unseen_idx))
                    batch = reader.get_classifier_batch(unseen_idx[lo:hi].tolist())
                    _, probs = clf.predict_batch(batch)
                    scores_all[lo:hi] = probs[:, 0]
                    self.progress.emit(
                        hi / len(unseen_idx),
                        f"Scoring cells: {hi}/{len(unseen_idx)}",
                    )
                # max-class confidence per cell (== how sure the model is).
                maxconf = np.maximum(scores_all, 1.0 - scores_all)
                if self.uncertain_only:
                    keep = np.flatnonzero(maxconf < self.confidence_threshold)
                    if keep.size == 0:
                        self.finished.emit(
                            [], f"No cells below confidence {self.confidence_threshold:.2f}. "
                                f"Raise the threshold or turn off 'uncertain only'.")
                        return
                    sub_idx = unseen_idx[keep]
                    sub_scores = scores_all[keep]
                    sub_conf = maxconf[keep]
                else:
                    sub_idx, sub_scores, sub_conf = unseen_idx, scores_all, maxconf
                # Least-confident first (identical to |0.5 - P(good)| ascending).
                rank = np.argsort(sub_conf)
                ranked_idx = sub_idx[rank]
                ranked_scores = sub_scores[rank]
                if self.diversify and not self.review:
                    picked, scores = self._diversified(reader, ranked_idx, ranked_scores)
                else:
                    picked = ranked_idx[: self.n_target]
                    scores = ranked_scores[: self.n_target]

            if self._cancel:
                self.finished.emit([], "Cancelled.")
                return
            self.progress.emit(1.0, f"Loading {len(picked)} crops…")
            records = [
                (reader.get(int(i)), float(s))
                for i, s in zip(picked.tolist(), scores.tolist())
            ]
            self.progress.emit(1.0, f"Loaded {len(records)} cells for labelling")
            self.finished.emit(records, "")
        except Exception as e:
            self.finished.emit([], str(e))
        finally:
            reader.close()


# ─────────────────────────────────────────────────────────────────────────────
# Cell thumbnail widget
# ─────────────────────────────────────────────────────────────────────────────

class _CellThumb(QPushButton):
    """Square button showing one cell, recoloured by current label.

    Single click cycles the label; double click emits ``expandRequested`` so the
    panel can open the large inspector. When the thumb holds keyboard focus,
    ``g``/``b``/``u`` classify it and the arrow keys move focus (handled by the
    panel via ``classifyKey`` / ``navKey``).
    """

    SIDE_PX = 96

    expandRequested = pyqtSignal()
    classifyKey = pyqtSignal(str)      # "good" | "bad" | "unlabeled"
    navKey = pyqtSignal(int, int)      # (dx, dy)

    def __init__(
        self,
        record: CropRecord,
        score: float,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.record = record
        self.score = score
        self.label = "unlabeled"
        self.setFixedSize(self.SIDE_PX, self.SIDE_PX)
        self.setIconSize(self._icon_size())
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setToolTip(self._tooltip())
        self._render()

        self._click_timer = QTimer(self)
        self._click_timer.setSingleShot(True)
        self._click_timer.setInterval(QApplication.doubleClickInterval())
        self._click_timer.timeout.connect(self._cycle)

    def mouseReleaseEvent(self, event) -> None:
        super().mouseReleaseEvent(event)
        if event.button() == Qt.MouseButton.LeftButton:
            self._click_timer.start()

    def mouseDoubleClickEvent(self, event) -> None:
        self._click_timer.stop()
        self.expandRequested.emit()

    def keyPressEvent(self, event) -> None:
        key = event.key()
        if key == Qt.Key.Key_G:
            self.classifyKey.emit("good")
        elif key == Qt.Key.Key_B:
            self.classifyKey.emit("bad")
        elif key == Qt.Key.Key_U:
            self.classifyKey.emit("unlabeled")
        elif key == Qt.Key.Key_Left:
            self.navKey.emit(-1, 0)
        elif key == Qt.Key.Key_Right:
            self.navKey.emit(1, 0)
        elif key == Qt.Key.Key_Up:
            self.navKey.emit(0, -1)
        elif key == Qt.Key.Key_Down:
            self.navKey.emit(0, 1)
        else:
            super().keyPressEvent(event)

    def _icon_size(self):
        return QSize(self.SIDE_PX - 12, self.SIDE_PX - 12)

    def _tooltip(self) -> str:
        r = self.record
        conf = "n/a" if math.isnan(self.score) else f"{self.score:.3f}"
        return (
            f"{r.cell_uid}\n"
            f"  {r.condition_label}  gene={r.gene}  drug={r.drug}\n"
            f"  reporter={r.reporter}  fov={r.fov_index}  area={r.area_px}px\n"
            f"  P(good)={conf}\n"
            f"  (click cycles · g/b/u on keyboard · double-click to enlarge)"
        )

    def _render(self) -> None:
        from PyQt6.QtGui import QImage, QPixmap, QIcon

        # crop layout: (3, H, W) = [phase, mask, area]
        phase = self.record.crop[0]
        mask = self.record.crop[1] > 0.5

        rgb = to_outlined_rgb(phase, outline_of(mask))
        h, w = rgb.shape[:2]

        qimg = QImage(rgb.tobytes(), w, h, w * 3, QImage.Format.Format_RGB888)
        pix = QPixmap.fromImage(qimg).scaled(
            self._icon_size().width(), self._icon_size().height(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.setIcon(QIcon(pix))

        ring = LABEL_COLORS[self.label]
        self.setStyleSheet(
            f"QPushButton {{ background: {COLORS['surface']}; "
            f"border: 3px solid {ring}; border-radius: 6px; padding: 2px; }}"
            f"QPushButton:hover {{ border: 3px solid {COLORS['primary']}; }}"
            f"QPushButton:focus {{ border: 3px solid {COLORS['primary']}; }}"
        )

    def _cycle(self) -> None:
        order = ["unlabeled", "good", "bad"]
        self.label = order[(order.index(self.label) + 1) % len(order)]
        self._render()

    def set_label(self, value: str) -> None:
        if value in LABEL_COLORS:
            self.label = value
            self._render()


# ─────────────────────────────────────────────────────────────────────────────
# Training worker
# ─────────────────────────────────────────────────────────────────────────────

class _TrainWorker(QThread):
    progress = pyqtSignal(float, str)
    finished = pyqtSignal(object, str)

    def __init__(
        self,
        h5_path: Path,
        labels_dir: Path,
        output_dir: Path,
        *,
        epochs: int,
        lr: float,
        batch_size: int,
        patience: int,
        val_fraction: float,
        test_fraction: float,
        seed: int,
        device: Optional[str],
        freeze_backbone: bool,
        pretrained: Optional[Path],
    ) -> None:
        super().__init__()
        self.h5_path = h5_path
        self.labels_dir = labels_dir
        self.output_dir = output_dir
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.patience = patience
        self.val_fraction = val_fraction
        self.test_fraction = test_fraction
        self.seed = seed
        self.device = device
        self.freeze_backbone = freeze_backbone
        self.pretrained = pretrained

    def run(self) -> None:
        try:
            from ...core.train_classifier import train as run_train
            from ...core.labels_store import LABELS_SCHEMA_VERSION, cell_uids_sha256
            from ...core.cell_quality_classifier import CLASS_NAMES, NUM_CLASSES

            reader = HDF5CropsReader(self.h5_path)
            store = LabelsStore(self.labels_dir)
            try:
                crops, labels, used_uids = store.to_training_dataset(reader)
            finally:
                reader.close()

            class_counts = {CLASS_NAMES[i]: int((labels == i).sum())
                            for i in range(NUM_CLASSES)}
            provenance = {
                "source_h5": self.h5_path.name,
                "labels_dir": str(self.labels_dir),
                "n_labelled": int(len(used_uids)),
                "used_uids_sha256": cell_uids_sha256(used_uids),
                "class_counts": class_counts,
                "labels_schema_version": LABELS_SCHEMA_VERSION,
            }

            self.output_dir.mkdir(parents=True, exist_ok=True)
            summary = run_train(
                crops=crops,
                labels=labels,
                output_dir=self.output_dir,
                epochs=self.epochs,
                batch_size=self.batch_size,
                lr=self.lr,
                val_fraction=self.val_fraction,
                test_fraction=self.test_fraction,
                patience=self.patience,
                seed=self.seed,
                device=self.device,
                freeze_backbone=self.freeze_backbone,
                pretrained_path=str(self.pretrained) if self.pretrained else None,
                provenance=provenance,
                progress_cb=lambda f, m: self.progress.emit(f, m),
            )
            self.progress.emit(1.0, "Training finished.")
            self.finished.emit(summary or {}, "")
        except Exception as e:
            self.finished.emit({}, str(e))


# ─────────────────────────────────────────────────────────────────────────────
# Panel
# ─────────────────────────────────────────────────────────────────────────────

class LabelTrainPanel(QWidget):
    """The single workhorse panel: active-learning labeller + fine-tune trainer."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._h5_path: Optional[Path] = None
        self._labels_dir: Optional[Path] = None
        self._reader: Optional[HDF5CropsReader] = None
        self._store: Optional[LabelsStore] = None
        self._context_provider: Optional[FovContextProvider] = None

        self._pool: list[tuple[CropRecord, float]] = []
        self._page_index = 0
        self._page_labels: dict[int, str] = {}  # absolute pool index → label string
        # Last label persisted to disk per pool index, so we can tell what is
        # genuinely unsaved (and avoid re-writing rows that already match).
        self._saved_labels: dict[int, str] = {}
        self._thumbs: list[_CellThumb] = []
        self._focused_pos = 0
        self._cols = 8
        self._review_mode = False  # showing model predictions to accept/flip
        self._session_id: str = make_session_id()
        self._session_started_at = datetime.now(timezone.utc).isoformat()
        self._session_seen_uids: set[str] = set()
        self._session_label_count = 0
        self._last_model_dir: Optional[Path] = None

        self._al_worker: Optional[_ActiveLearningWorker] = None
        self._train_worker: Optional[_TrainWorker] = None

        self._build_ui()
        self._refresh_dataset_stats()

    # ── public ─────────────────────────────────────────────────────────

    def set_source(self, h5_path: Optional[Path], labels_dir: Optional[Path]) -> None:
        # Auto-save any unsaved labels to the OLD store before swapping, and
        # stop the in-flight pool scorer so it can't emit into a stale dataset.
        if self._store is not None and self._pool and self.has_unsaved_labels():
            self._save_batch(silent_if_empty=True)
        self._record_session_if_any()
        self._stop_worker(self._al_worker)
        self._al_worker = None

        if self._reader is not None:
            self._reader.close()
            self._reader = None
        if h5_path is None:
            self._h5_path = None
            self._labels_dir = None
            self._store = None
            self._context_provider = None
            self._info_label.setText("(no source — pick an HDF5)")
            self._refresh_dataset_stats()
            return

        self._h5_path = Path(h5_path)
        self._labels_dir = Path(labels_dir) if labels_dir else default_labels_dir_for(self._h5_path)
        try:
            self._reader = HDF5CropsReader(self._h5_path)
        except Exception as e:
            self._info_label.setText(f"Could not open {self._h5_path.name}: {e}")
            return
        self._store = LabelsStore(self._labels_dir)
        try:
            self._store.ensure_initialised(
                source_h5=self._h5_path,
                total_cells=len(self._reader),
                h5_schema_version=self._reader.schema_version,
                cell_uids=self._reader.cell_uids,
            )
        except ValueError as e:
            self._info_label.setText(str(e))
            self._reader.close()
            self._reader = None
            self._store = None
            return

        # Optional: locate the pipeline originals beside the HDF5 so the cell
        # inspector can show each cell in its full field-of-view context.
        self._context_provider = FovContextProvider(self._h5_path)

        self._info_label.setText(
            f"Loaded {len(self._reader):,} cells from {self._h5_path.name} · "
            f"labels at {self._labels_dir}"
        )
        # Fresh labelling session for this dataset.
        self._pool = []
        self._page_index = 0
        self._page_labels.clear()
        self._saved_labels.clear()
        self._session_seen_uids.clear()
        self._session_id = make_session_id()
        self._session_started_at = datetime.now(timezone.utc).isoformat()
        self._session_label_count = 0
        self._review_mode = False
        self._populate_grid_from_page()
        self._save_btn.setEnabled(False)
        self._save_next_btn.setEnabled(False)
        self._prev_btn.setEnabled(False)
        self._next_btn.setEnabled(False)
        self._refresh_dataset_stats()

    def has_unsaved_labels(self) -> bool:
        """True if any good/bad mark differs from what's been written to disk."""
        self._capture_current_page()
        return any(
            lbl in ("good", "bad") and self._saved_labels.get(idx) != lbl
            for idx, lbl in self._page_labels.items()
        )

    def save_pending(self) -> int:
        """Persist any unsaved labels (used by the main window on close)."""
        if self._store is None:
            return 0
        return self._save_batch(silent_if_empty=True)

    # ── UI build ──────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        # The two sections are exposed as standalone cards so the main window
        # can place them on separate sidebar pages (Label cells / Fine-tune).
        # This panel object stays alive as the controller that owns those
        # widgets and the HDF5 reader / labels store; it is never shown as a
        # whole, so it carries no layout of its own.
        self.label_section = self._build_label_section()
        self.train_section = self._build_train_section()

    def _build_label_section(self) -> QWidget:
        box = QGroupBox("Label cells")
        v = QVBoxLayout(box)
        v.setContentsMargins(12, 18, 12, 12)
        v.setSpacing(6)

        bar = QHBoxLayout()

        self._model_combo = QComboBox()
        self._populate_model_combo()
        self._model_combo.activated.connect(self._on_model_combo_activated)

        self._pool_size = QSpinBox(); self._pool_size.setRange(48, 5000); self._pool_size.setValue(1000)
        self._pool_size.setToolTip(
            "Cells to preload per fetch. Higher = fewer waits, more upfront time."
        )
        self._page_size = QSpinBox(); self._page_size.setRange(8, 256); self._page_size.setValue(48)
        self._page_size.setToolTip(
            "Cells shown per page. Smaller pages render faster."
        )
        self._diversify = QCheckBox("Diversify")
        self._diversify.setChecked(True)
        self._diversify.setToolTip(
            "Spread the most-uncertain cells across fields of view so one noisy "
            "FOV can't fill a page with near-duplicates."
        )
        self._uncertain_only = QCheckBox("Uncertain only")
        self._uncertain_only.setToolTip(
            "Only load cells the model is unsure about (max-class confidence "
            "below the threshold). Requires a model."
        )
        self._conf_thresh = QDoubleSpinBox()
        self._conf_thresh.setRange(0.5, 1.0); self._conf_thresh.setSingleStep(0.05)
        self._conf_thresh.setValue(0.85); self._conf_thresh.setDecimals(2)
        self._conf_thresh.setToolTip("Confidence cutoff for 'uncertain only' / Review.")

        self._load_btn = QPushButton("Load pool")
        self._load_btn.setToolTip(
            "Score every unlabelled cell with the chosen model and load the "
            "most-uncertain into memory. Subsequent pages turn instantly."
        )
        self._load_btn.clicked.connect(lambda: self._load_pool(review=False))

        self._review_btn = QPushButton("Review predictions")
        self._review_btn.setToolTip(
            "Load cells with the model's prediction pre-applied (including "
            "already-labelled ones), so you only flip the wrong ones with g/b, "
            "then Save. Requires a model."
        )
        self._review_btn.clicked.connect(lambda: self._load_pool(review=True))

        self._save_btn = QPushButton("Save labels")
        self._save_btn.clicked.connect(lambda: self._save_batch())
        self._save_btn.setEnabled(False)

        self._prev_btn = QPushButton("← Prev")
        self._prev_btn.clicked.connect(self._prev_page)
        self._prev_btn.setEnabled(False)
        self._next_btn = QPushButton("Next →")
        self._next_btn.clicked.connect(self._next_page)
        self._next_btn.setEnabled(False)

        self._save_next_btn = QPushButton("Save && next page")
        self._save_next_btn.setObjectName("primary")
        self._save_next_btn.clicked.connect(self._save_then_next_page)
        self._save_next_btn.setEnabled(False)

        self._reset_btn = QPushButton("Reset session")
        self._reset_btn.setToolTip(
            "Forget which cells have been shown this session, so previously-skipped "
            "cells become eligible again."
        )
        self._reset_btn.clicked.connect(self._reset_session)

        bar.addWidget(QLabel("Model:"))
        bar.addWidget(self._model_combo)
        bar.addWidget(QLabel("Pool:"))
        bar.addWidget(self._pool_size)
        bar.addWidget(QLabel("Page:"))
        bar.addWidget(self._page_size)
        bar.addWidget(self._diversify)
        bar.addWidget(self._uncertain_only)
        bar.addWidget(QLabel("conf<"))
        bar.addWidget(self._conf_thresh)
        bar.addWidget(self._load_btn)
        bar.addWidget(self._review_btn)
        bar.addWidget(self._save_btn)
        bar.addWidget(self._prev_btn)
        bar.addWidget(self._next_btn)
        bar.addWidget(self._save_next_btn)
        bar.addWidget(self._reset_btn)
        v.addLayout(bar)

        # Secondary row: folder actions.
        actions = QHBoxLayout()
        self._open_labels_btn = QPushButton("Open labels folder")
        self._open_labels_btn.setObjectName("ghost")
        self._open_labels_btn.clicked.connect(self._open_labels_folder)
        self._export_btn = QPushButton("Export labels…")
        self._export_btn.setObjectName("ghost")
        self._export_btn.clicked.connect(self._export_labels)
        self._balance_label = QLabel("")
        self._balance_label.setObjectName("caption")
        actions.addWidget(self._open_labels_btn)
        actions.addWidget(self._export_btn)
        actions.addStretch(1)
        actions.addWidget(self._balance_label)
        v.addLayout(actions)

        self._info_label = QLabel("(no source — pick an HDF5)")
        self._info_label.setObjectName("muted")
        self._info_label.setWordWrap(True)
        v.addWidget(self._info_label)

        self._status = QLabel("Pick an HDF5 and click 'Load pool'.")
        self._status.setObjectName("muted")
        self._status.setWordWrap(True)
        v.addWidget(self._status)

        self._progress = QProgressBar()
        self._progress.setRange(0, 1000); self._progress.setValue(0)
        v.addWidget(self._progress)

        hint = QLabel(
            "Click a thumbnail to cycle unlabeled → good → bad. On the focused "
            "cell, press g (good) / b (bad) — both jump to the next cell — or u "
            "(unlabeled); arrow keys move between cells. Double-click to enlarge. "
            "Outer border = your label · cyan outline = segmented cell."
        )
        hint.setObjectName("muted")
        hint.setWordWrap(True)
        v.addWidget(hint)

        # Grid
        self._grid_host = QWidget()
        self._grid = QGridLayout(self._grid_host)
        self._grid.setSpacing(6)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self._grid_host)
        v.addWidget(scroll, stretch=1)

        return box

    def _build_train_section(self) -> QWidget:
        box = QGroupBox("Fine-tune classifier")
        v = QVBoxLayout(box)
        v.setContentsMargins(12, 18, 12, 12)
        v.setSpacing(6)

        self._stats_label = QLabel("")
        self._stats_label.setObjectName("muted")
        v.addWidget(self._stats_label)

        bar = QHBoxLayout()
        self._epochs = QSpinBox(); self._epochs.setRange(1, 500); self._epochs.setValue(40)
        self._lr = QDoubleSpinBox()
        self._lr.setRange(1e-6, 1.0); self._lr.setDecimals(6)
        self._lr.setValue(1e-4); self._lr.setSingleStep(5e-5)
        self._fine_tune = QCheckBox("Fine-tune from selected base model")
        self._fine_tune.setChecked(True)
        self._freeze = QCheckBox("Freeze backbone")
        self._freeze.setToolTip(
            "Train only the classifier head (freeze the conv layers). Often "
            "better when fine-tuning from a preset on a small label set."
        )
        self._train_btn = QPushButton("Start training")
        self._train_btn.setObjectName("primary")
        self._train_btn.clicked.connect(self._start_training)

        bar.addWidget(QLabel("Epochs:")); bar.addWidget(self._epochs)
        bar.addWidget(QLabel("LR:"));     bar.addWidget(self._lr)
        bar.addWidget(self._fine_tune)
        bar.addWidget(self._freeze)
        bar.addStretch(1)
        bar.addWidget(self._train_btn)
        v.addLayout(bar)

        # Advanced row.
        adv = QHBoxLayout()
        self._device_combo = QComboBox()
        self._device_combo.addItem("Auto", userData=None)
        self._device_combo.addItem("CPU", userData="cpu")
        self._device_combo.addItem("MPS", userData="mps")
        self._device_combo.addItem("CUDA", userData="cuda")
        self._batch = QSpinBox(); self._batch.setRange(1, 512); self._batch.setValue(32)
        self._patience = QSpinBox(); self._patience.setRange(1, 200); self._patience.setValue(15)
        self._val_frac = QDoubleSpinBox(); self._val_frac.setRange(0.05, 0.5)
        self._val_frac.setSingleStep(0.05); self._val_frac.setValue(0.2)
        self._test_frac = QDoubleSpinBox(); self._test_frac.setRange(0.0, 0.5)
        self._test_frac.setSingleStep(0.05); self._test_frac.setValue(0.15)
        self._test_frac.setToolTip(
            "Held-out test fraction for unbiased final metrics. 0 = no test set "
            "(auto-disabled on tiny datasets)."
        )
        self._seed = QSpinBox(); self._seed.setRange(0, 2**31 - 1); self._seed.setValue(42)
        adv.addWidget(QLabel("Device:")); adv.addWidget(self._device_combo)
        adv.addWidget(QLabel("Batch:")); adv.addWidget(self._batch)
        adv.addWidget(QLabel("Patience:")); adv.addWidget(self._patience)
        adv.addWidget(QLabel("Val:")); adv.addWidget(self._val_frac)
        adv.addWidget(QLabel("Test:")); adv.addWidget(self._test_frac)
        adv.addWidget(QLabel("Seed:")); adv.addWidget(self._seed)
        adv.addStretch(1)
        self._open_model_btn = QPushButton("Open model folder")
        self._open_model_btn.setObjectName("ghost")
        self._open_model_btn.clicked.connect(self._open_model_folder)
        self._open_model_btn.setEnabled(False)
        adv.addWidget(self._open_model_btn)
        v.addLayout(adv)

        self._train_progress = QProgressBar()
        self._train_progress.setRange(0, 1000); self._train_progress.setValue(0)
        v.addWidget(self._train_progress)

        self._train_status = QLabel("")
        self._train_status.setObjectName("muted")
        self._train_status.setWordWrap(True)
        v.addWidget(self._train_status)

        self._inspector = ModelInspector()
        v.addWidget(self._inspector)

        return box

    # ── model picker ──────────────────────────────────────────────────

    def _list_trained_models(self) -> list[tuple[Path, Optional[float]]]:
        """(<trained_dir>, best_val_acc) for every model under ~/.modelprep/models."""
        root = _trained_models_root()
        out: list[tuple[Path, Optional[float]]] = []
        if not root.exists():
            return out
        import json
        for d in sorted(root.glob("*"), reverse=True):
            best = d / "best_model.pth"
            if not best.exists():
                continue
            acc = None
            cfg = d / "training_config.json"
            if cfg.exists():
                try:
                    acc = json.loads(cfg.read_text()).get("best_val_acc")
                except Exception:  # noqa: BLE001
                    acc = None
            out.append((d, acc))
        return out

    def _populate_model_combo(self, select_path: Optional[str] = None) -> None:
        prev = select_path if select_path is not None else self._model_combo.currentData()
        self._model_combo.blockSignals(True)
        self._model_combo.clear()
        self._model_combo.addItem("none (random order)", userData=None)
        for name, path in list_preset_models().items():
            self._model_combo.addItem(f"preset: {name}", userData=str(path))
        for d, acc in self._list_trained_models():
            label = f"trained: {d.name}"
            if acc is not None:
                label += f" (acc {acc:.2f})"
            self._model_combo.addItem(label, userData=str(d / "best_model.pth"))
        self._model_combo.addItem("Browse .pth…", userData="__browse__")
        # Restore a sensible selection.
        if prev and prev != "__browse__":
            idx = self._model_combo.findData(prev)
            if idx >= 0:
                self._model_combo.setCurrentIndex(idx)
        self._model_combo.blockSignals(False)

    def _on_model_combo_activated(self, idx: int) -> None:
        if self._model_combo.itemData(idx) != "__browse__":
            return
        path, _ = QFileDialog.getOpenFileName(
            self.label_section, "Pick a classifier .pth", "",
            "PyTorch models (*.pth)",
        )
        if path:
            self._model_combo.insertItem(idx, f"custom: {Path(path).name}", userData=path)
            self._model_combo.setCurrentIndex(idx)
        else:
            self._model_combo.setCurrentIndex(0)

    def _current_model_path(self) -> Optional[Path]:
        data = self._model_combo.currentData()
        if not data or data == "__browse__":
            return None
        return Path(data)

    # ── events: label flow ────────────────────────────────────────────

    def _load_pool(self, review: bool = False) -> None:
        if self._reader is None or self._h5_path is None or self._store is None:
            QMessageBox.warning(self.label_section, "No source", "Pick an HDF5 first.")
            return
        model_path = self._current_model_path()
        uncertain_only = self._uncertain_only.isChecked()
        if (review or uncertain_only) and model_path is None:
            QMessageBox.warning(
                self.label_section, "No model",
                "Pick a model first — Review and 'uncertain only' need a model "
                "to score cells.")
            return
        # Don't blow away an unsaved page when switching into review.
        if self._pool and self.has_unsaved_labels():
            self._save_batch(silent_if_empty=True)
        self._stop_worker(self._al_worker)
        self._review_mode = review

        # Review audits the model on cells not yet seen this session — including
        # already-labelled ones, so earlier mistakes can be corrected. Normal
        # labelling also skips cells already saved on disk.
        if review:
            seen_uids = set(self._session_seen_uids)
        else:
            seen_uids = self._store.labeled_uids() | self._session_seen_uids

        self._load_btn.setEnabled(False)
        self._review_btn.setEnabled(False)
        self._save_btn.setEnabled(False)
        self._save_next_btn.setEnabled(False)
        self._progress.setValue(0)
        what = "predictions to review" if review else "cells"
        self._status.setText(
            f"Loading up to {self._pool_size.value()} {what} "
            f"(skipping {len(seen_uids)} seen)…"
        )

        self._al_worker = _ActiveLearningWorker(
            h5_path=self._h5_path,
            model_path=model_path,
            n_target=self._pool_size.value(),
            seen_uids=seen_uids,
            diversify=self._diversify.isChecked() and model_path is not None,
            review=review,
            uncertain_only=uncertain_only,
            confidence_threshold=self._conf_thresh.value(),
        )
        self._al_worker.progress.connect(self._on_al_progress)
        self._al_worker.finished.connect(self._on_al_finished)
        self._al_worker.start()

    def _on_al_progress(self, frac: float, msg: str) -> None:
        if self.sender() is not self._al_worker:
            return
        self._progress.setValue(int(max(0.0, min(1.0, frac)) * 1000))
        self._status.setText(msg)

    def _on_al_finished(self, records, error: str) -> None:
        if self.sender() is not self._al_worker:
            return  # a superseded worker — ignore its late result
        self._load_btn.setEnabled(True)
        self._review_btn.setEnabled(True)
        if error:
            self._status.setText(f"Failed: {error}")
            return

        self._pool = list(records)
        self._page_index = 0
        self._page_labels.clear()
        self._saved_labels.clear()  # pool indices now map to different cells

        for rec, _score in self._pool:
            self._session_seen_uids.add(rec.cell_uid)

        # In review mode, pre-apply the model's prediction to every cell so the
        # user only has to flip the wrong ones (g/b), then Save.
        if self._review_mode:
            for i, (_rec, score) in enumerate(self._pool):
                self._page_labels[i] = "good" if score >= 0.5 else "bad"

        if not self._pool:
            self._populate_grid_from_page()
            self._save_btn.setEnabled(False)
            self._save_next_btn.setEnabled(False)
            self._status.setText(
                "No more unseen cells. Use 'Reset session' to revisit skipped cells."
            )
            return

        self._populate_grid_from_page()
        self._save_btn.setEnabled(True)
        self._save_next_btn.setEnabled(True)
        self._update_page_status()
        if self._review_mode:
            self._status.setText(
                self._status.text()
                + "  ·  Reviewing predictions: flip wrong ones with g/b, then Save."
            )

    def _populate_grid_from_page(self) -> None:
        for i in reversed(range(self._grid.count())):
            item = self._grid.itemAt(i)
            if item and item.widget():
                item.widget().setParent(None)
        self._thumbs.clear()

        page_size = self._page_size.value()
        start = self._page_index * page_size
        end = min(start + page_size, len(self._pool))

        cols = self._cols
        for i, (rec, score) in enumerate(self._pool[start:end]):
            pool_idx = start + i
            thumb = _CellThumb(rec, score)
            cached = self._page_labels.get(pool_idx)
            if cached:
                thumb.set_label(cached)
            thumb.expandRequested.connect(
                lambda idx=pool_idx: self._open_detail(idx)
            )
            thumb.classifyKey.connect(
                lambda lbl, pos=i: self._grid_classify(pos, lbl)
            )
            thumb.navKey.connect(
                lambda dx, dy, pos=i: self._grid_nav(pos, dx, dy)
            )
            self._thumbs.append(thumb)
            self._grid.addWidget(thumb, i // cols, i % cols)

        self._prev_btn.setEnabled(self._page_index > 0)
        self._next_btn.setEnabled(end < len(self._pool))
        # Put keyboard focus on the first cell so g/b/u work immediately.
        if self._thumbs:
            self._focused_pos = 0
            self._thumbs[0].setFocus()
        self._refresh_balance()

    def _capture_current_page(self) -> None:
        page_size = self._page_size.value()
        start = self._page_index * page_size
        for i, t in enumerate(self._thumbs):
            self._page_labels[start + i] = t.label

    # ── keyboard-driven grid labelling ─────────────────────────────────

    def _focus_thumb(self, pos: int) -> None:
        if 0 <= pos < len(self._thumbs):
            self._focused_pos = pos
            self._thumbs[pos].setFocus()

    def _grid_classify(self, pos: int, label: str) -> None:
        if not (0 <= pos < len(self._thumbs)):
            return
        self._thumbs[pos].set_label(label)
        page_size = self._page_size.value()
        self._page_labels[self._page_index * page_size + pos] = label
        self._refresh_balance()
        # Good/bad advance to the next cell (matching the inspector's rhythm);
        # 'unlabeled' is a correction, so it holds.
        if label in ("good", "bad") and pos + 1 < len(self._thumbs):
            self._focus_thumb(pos + 1)

    def _grid_nav(self, pos: int, dx: int, dy: int) -> None:
        if not self._thumbs:
            return
        cols = self._cols
        row, col = pos // cols, pos % cols
        target = (row + dy) * cols + (col + dx)
        if 0 <= target < len(self._thumbs):
            self._focus_thumb(target)

    # ── large inspector ───────────────────────────────────────────────

    def _pool_label(self, pool_idx: int) -> str:
        return self._page_labels.get(pool_idx, "unlabeled")

    def _set_pool_label(self, pool_idx: int, label: str) -> None:
        self._page_labels[pool_idx] = label
        # Keep the grid in sync if this cell is on the visible page.
        start = self._page_index * self._page_size.value()
        thumb_pos = pool_idx - start
        if 0 <= thumb_pos < len(self._thumbs):
            self._thumbs[thumb_pos].set_label(label)
        self._refresh_balance()

    def _open_detail(self, pool_idx: int) -> None:
        if self._reader is None or not (0 <= pool_idx < len(self._pool)):
            return
        # Persist current thumbnails so the dialog reads authoritative labels.
        self._capture_current_page()
        dialog = CellDetailView(
            pool=self._pool,
            start_index=pool_idx,
            phase_idx=self._reader.phase_channel_idx,
            mask_idx=self._reader.mask_channel_idx,
            get_label=self._pool_label,
            set_label=self._set_pool_label,
            parent=self,
            context_provider=self._context_provider,
        )
        dialog.exec()

    def _save_then_next_page(self) -> None:
        self._capture_current_page()
        self._save_batch(silent_if_empty=True)
        self._page_index += 1
        page_size = self._page_size.value()
        if self._page_index * page_size >= len(self._pool):
            self._populate_grid_from_page()
            self._save_btn.setEnabled(False)
            self._save_next_btn.setEnabled(False)
            self._next_btn.setEnabled(False)
            self._status.setText(
                f"Pool exhausted ({len(self._pool)} cells cycled). "
                "Click 'Load pool' for more."
            )
            return
        self._populate_grid_from_page()
        self._update_page_status()

    def _prev_page(self) -> None:
        if self._page_index <= 0:
            return
        self._capture_current_page()
        self._page_index -= 1
        self._populate_grid_from_page()
        self._update_page_status()

    def _next_page(self) -> None:
        page_size = self._page_size.value()
        if (self._page_index + 1) * page_size >= len(self._pool):
            return
        self._capture_current_page()
        self._page_index += 1
        self._populate_grid_from_page()
        self._update_page_status()

    def _reset_session(self) -> None:
        n_seen = len(self._session_seen_uids)
        n_cached = len(self._page_labels)
        self._session_seen_uids.clear()
        self._page_labels.clear()
        self._status.setText(
            f"Cleared {n_seen} session-shown cells and {n_cached} cached labels. "
            "Saved labels on disk are still excluded."
        )

    def _update_page_status(self) -> None:
        page_size = self._page_size.value()
        start = self._page_index * page_size
        end = min(start + page_size, len(self._pool))
        n_pages = max(1, (len(self._pool) + page_size - 1) // page_size)
        self._status.setText(
            f"Pool: {len(self._pool)} cells · page {self._page_index + 1}/{n_pages} "
            f"(cells {start + 1}–{end})."
        )

    # ── save ──────────────────────────────────────────────────────────

    def _save_batch(self, silent_if_empty: bool = False) -> int:
        if self._store is None or self._reader is None:
            return 0
        self._capture_current_page()
        to_save = [
            (idx, lbl) for idx, lbl in self._page_labels.items()
            if lbl in ("good", "bad") and 0 <= idx < len(self._pool)
            and self._saved_labels.get(idx) != lbl
        ]
        if not to_save:
            if not silent_if_empty:
                QMessageBox.information(self.label_section, "Nothing to save",
                                        "Label some cells first.")
            return 0

        model_path = self._current_model_path()
        model_at_label = str(model_path) if model_path else "none"
        rows: list[LabelRow] = []
        for pool_idx, lbl_name in to_save:
            rec, score = self._pool[pool_idx]
            # With no model the "score" is a UI-ordering placeholder, not a real
            # P(good); persist NaN so it isn't mistaken for a 0.5 confidence.
            conf = score if model_path is not None else float("nan")
            rows.append(LabelRow(
                cell_uid=rec.cell_uid,
                label_idx=LABEL_TO_IDX[lbl_name],
                label_name=lbl_name,
                area_px=rec.area_px,
                labeled_at=datetime.now(timezone.utc).isoformat(),
                session_id=self._session_id,
                source_h5=self._h5_path.name if self._h5_path else "",
                model_at_label=model_at_label,
                confidence_good_at_label=conf,
            ))
        n_saved = self._store.append_batch(rows)
        self._session_label_count += n_saved
        # Remember what we persisted so these cells aren't re-written and don't
        # read as "unsaved" while their thumbnails stay coloured on screen.
        for idx, lbl in to_save:
            self._saved_labels[idx] = lbl
        self._status.setText(
            f"Saved {n_saved} labels → {self._store.csv_path}."
        )
        self._refresh_dataset_stats()
        return n_saved

    # ── folder actions ─────────────────────────────────────────────────

    def _open_labels_folder(self) -> None:
        if self._labels_dir and Path(self._labels_dir).exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._labels_dir)))
        else:
            QMessageBox.information(self.label_section, "No labels yet",
                                    "Pick a source and save some labels first.")

    def _open_model_folder(self) -> None:
        if self._last_model_dir and self._last_model_dir.exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._last_model_dir)))

    def _export_labels(self) -> None:
        if self._store is None or not self._store.csv_path.exists():
            QMessageBox.information(self.label_section, "Nothing to export",
                                    "Save some labels first.")
            return
        dest, _ = QFileDialog.getSaveFileName(
            self.label_section, "Export labels.csv",
            str(Path.home() / "labels.csv"), "CSV files (*.csv)",
        )
        if not dest:
            return
        try:
            shutil.copyfile(self._store.csv_path, dest)
            self._status.setText(f"Exported labels → {dest}")
        except Exception as e:  # noqa: BLE001
            QMessageBox.warning(self.label_section, "Export failed", str(e))

    # ── balance feedback ───────────────────────────────────────────────

    def _refresh_balance(self) -> None:
        if self._store is None:
            self._balance_label.setText("")
            return
        counts = self._store.class_counts()
        g, b = counts.get("good", 0), counts.get("bad", 0)
        self._capture_current_page()
        pend_g = sum(1 for idx, v in self._page_labels.items()
                     if v == "good" and self._saved_labels.get(idx) != "good")
        pend_b = sum(1 for idx, v in self._page_labels.items()
                     if v == "bad" and self._saved_labels.get(idx) != "bad")
        txt = f"Saved: good {g} · bad {b} (total {g + b})"
        if pend_g or pend_b:
            txt += f"  ·  +page good {pend_g} / bad {pend_b}"
        total = g + b
        minority = min(g, b)
        warn = total >= 10 and minority / total < 0.2
        color = tokens.active().warning if warn else tokens.active().text_subtle
        if warn:
            txt += "  ⚠ class imbalance — label more of the rarer class"
        self._balance_label.setStyleSheet(f"color: {color};")
        self._balance_label.setText(txt)

    # ── training ──────────────────────────────────────────────────────

    def _refresh_dataset_stats(self) -> None:
        if self._store is None:
            self._stats_label.setText("(no source — pick an HDF5)")
            self._train_btn.setEnabled(False)
            self._refresh_balance()
            return
        counts = self._store.class_counts()
        total = counts.get("good", 0) + counts.get("bad", 0)
        self._stats_label.setText(
            f"{total} labeled crops · good={counts.get('good', 0)} "
            f"bad={counts.get('bad', 0)}   ·   {self._store.csv_path}"
        )
        self._train_btn.setEnabled(total >= 20)
        self._refresh_balance()

    def _start_training(self) -> None:
        if self._h5_path is None or self._labels_dir is None:
            QMessageBox.warning(self.train_section, "No source", "Pick an HDF5 first.")
            return
        self._stop_worker(self._train_worker)
        out_dir = _trained_models_root() / f"trained_{int(time.time())}"

        pretrained = self._current_model_path() if self._fine_tune.isChecked() else None
        device = self._device_combo.currentData()

        self._train_btn.setEnabled(False)
        self._train_progress.setValue(0)
        self._train_status.setText(
            f"Training → {out_dir} (epochs={self._epochs.value()}, "
            f"lr={self._lr.value()}, device={device or 'auto'}, "
            f"fine-tune={'yes' if pretrained else 'no'}, "
            f"freeze={'yes' if self._freeze.isChecked() else 'no'})…"
        )

        self._train_worker = _TrainWorker(
            h5_path=self._h5_path,
            labels_dir=self._labels_dir,
            output_dir=out_dir,
            epochs=self._epochs.value(),
            lr=self._lr.value(),
            batch_size=self._batch.value(),
            patience=self._patience.value(),
            val_fraction=self._val_frac.value(),
            test_fraction=self._test_frac.value(),
            seed=self._seed.value(),
            device=device,
            freeze_backbone=self._freeze.isChecked(),
            pretrained=pretrained,
        )
        self._train_worker.progress.connect(self._on_train_progress)
        self._train_worker.finished.connect(self._on_train_finished)
        self._train_worker.start()

    def _on_train_progress(self, frac: float, msg: str) -> None:
        if self.sender() is not self._train_worker:
            return
        self._train_progress.setValue(int(max(0.0, min(1.0, frac)) * 1000))
        self._train_status.setText(msg)

    def _on_train_finished(self, summary: dict, error: str) -> None:
        if self.sender() is not self._train_worker:
            return
        self._train_btn.setEnabled(True)
        self._train_progress.setValue(1000)
        if error:
            self._train_status.setText(f"Training failed: {error}")
            return

        model_path = summary.get("model_path", "(unknown)")
        split = summary.get("eval_split", "val")
        eval_acc = summary.get("eval_acc")
        best_acc = summary.get("best_val_acc", 0.0)
        auc = summary.get("roc_auc")
        ap = summary.get("average_precision")
        thr = summary.get("decision_threshold", 0.5)
        conf = summary.get("confusion")
        n_train = summary.get("n_train", 0)
        n_val = summary.get("n_val", 0)
        n_test = summary.get("n_test", 0)

        parts = [f"Done · best val acc {best_acc:.3f}"]
        if eval_acc is not None and eval_acc == eval_acc:
            label = "test" if split == "test" else "val"
            parts.append(f"{label} acc {eval_acc:.3f}")
        if auc is not None and auc == auc:
            parts.append(f"AUC {auc:.3f}")
        if ap is not None and ap == ap:
            parts.append(f"AP {ap:.3f}")
        parts.append(f"thr {thr:.2f}")
        if conf:
            parts.append(f"conf tn={conf['tn']} fp={conf['fp']} fn={conf['fn']} tp={conf['tp']}")
        parts.append(f"(train {n_train} / val {n_val}" + (f" / test {n_test}" if n_test else "") + ")")
        msg = " · ".join(parts)

        warning = ""
        if split != "test":
            warning = ("  ⚠ no held-out test set (small dataset) — metrics are "
                       "validation, i.e. optimistic. Label more cells.")
        elif (auc is not None and auc == auc and auc <= 0.6) or best_acc <= 0.55:
            warning = ("  ⚠ near chance — the model didn't learn anything "
                       "generalisable. Label many more cells, then retrain.")
        elif n_train + n_val + n_test < 100:
            warning = "  ⚠ small dataset — results are noisy. Label more and retrain."
        self._train_status.setText(msg + warning + f"  →  {model_path}")

        # Surface the curves + metrics inline.
        try:
            self._last_model_dir = Path(model_path).parent
            self._inspector.set_model(Path(model_path))
            self._open_model_btn.setEnabled(True)
        except Exception:  # noqa: BLE001
            pass

        # Make the just-trained model available + selected for the next
        # active-learning round (re-rank the pool with the improved model).
        self._populate_model_combo(select_path=str(model_path))

    # ── lifecycle ─────────────────────────────────────────────────────

    def _stop_worker(self, worker: Optional[QThread]) -> None:
        """Cancel + wait for a worker so it can't emit into stale state."""
        if worker is None or not worker.isRunning():
            return
        if hasattr(worker, "cancel"):
            worker.cancel()
        worker.wait(5000)

    def _record_session_if_any(self) -> None:
        if self._store is not None and self._session_label_count > 0:
            try:
                self._store.record_session(
                    self._session_id,
                    self._session_started_at,
                    datetime.now(timezone.utc).isoformat(),
                    self._session_label_count,
                )
            except Exception:  # noqa: BLE001
                pass
            self._session_label_count = 0

    def shutdown(self) -> None:
        """Stop workers, record the session, release the HDF5 reader. Called by
        the main window on close (this controller panel is never shown, so its
        own ``closeEvent`` would not fire)."""
        self._stop_worker(self._al_worker)
        self._stop_worker(self._train_worker)
        self._record_session_if_any()
        if self._reader is not None:
            self._reader.close()
            self._reader = None

    def closeEvent(self, event) -> None:
        self.shutdown()
        super().closeEvent(event)
