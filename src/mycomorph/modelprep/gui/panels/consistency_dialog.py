"""Blinded label-consistency check dialog.

Samples already-labelled cells, presents them blind (no prior label, fresh
random order) in the familiar thumbnail grid for re-labelling, then scores the
re-label against the original — raw agreement, Cohen's κ, and the per-class
flips. The disagreements are the ambiguous rubric cases worth reviewing.

Results are written to ``<labels_dir>/consistency/<ts>/``; ``labels.csv`` is
never touched.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QGridLayout,
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

from ...core.consistency import (
    SAMPLE_BALANCED,
    SAMPLE_RANDOM,
    consistency_stats,
    sample_for_consistency,
    save_consistency_run,
)
from ...core.crops_h5 import HDF5CropsReader
from ...core.fov_context import FovContextProvider
from ...core.labels_store import LabelsStore
from ..palette import COLORS
from .cell_detail_dialog import CellDetailView
from .label_train_panel import _CellThumb


class _SampleWorker(QThread):
    """Open the HDF5 and load the sampled records off the UI thread."""

    done = pyqtSignal(object, int, int, str)  # pairs, phase_idx, mask_idx, error

    def __init__(self, h5_path: Path, labels_dir: Path, n: int, mode: str) -> None:
        super().__init__()
        self.h5_path = h5_path
        self.labels_dir = labels_dir
        self.n = n
        self.mode = mode

    def run(self) -> None:
        try:
            reader = HDF5CropsReader(self.h5_path)
            try:
                store = LabelsStore(self.labels_dir)
                pairs = sample_for_consistency(reader, store, self.n, self.mode)
                phase_idx = reader.phase_channel_idx
                mask_idx = reader.mask_channel_idx
            finally:
                reader.close()
            if not pairs:
                self.done.emit([], 0, 0, "No labelled cells to sample.")
                return
            self.done.emit(pairs, phase_idx, mask_idx, "")
        except Exception as e:  # noqa: BLE001
            self.done.emit([], 0, 0, str(e))


class ConsistencyCheckDialog(QDialog):
    """Re-label a blind sample, then score self-consistency."""

    COLS = 8

    def __init__(self, h5_path: Path, labels_dir: Path, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._h5_path = Path(h5_path)
        self._labels_dir = Path(labels_dir)
        self._pairs: list = []          # [(record, original_label)]
        self._thumbs: list[_CellThumb] = []
        self._phase_idx = 0
        self._mask_idx = 1
        self._provider: Optional[FovContextProvider] = None
        self._worker: Optional[_SampleWorker] = None

        self.setWindowTitle("Label consistency check")
        self.setModal(True)
        self.resize(900, 720)
        self._build_ui()

    # ── UI ─────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(8)

        intro = QLabel(
            "Re-label this blind sample of your own cells as you see them — your "
            "previous labels are hidden. Self-agreement is the practical ceiling "
            "for any model; the disagreements are your fuzzy rubric cases."
        )
        intro.setObjectName("muted")
        intro.setWordWrap(True)
        root.addWidget(intro)

        cfg = QHBoxLayout()
        self._size = QSpinBox(); self._size.setRange(20, 2000); self._size.setValue(200)
        self._mode = QComboBox()
        self._mode.addItem("Balanced (good vs bad)", userData=SAMPLE_BALANCED)
        self._mode.addItem("Random (real prevalence)", userData=SAMPLE_RANDOM)
        self._start = QPushButton("Start")
        self._start.setObjectName("primary")
        self._start.clicked.connect(self._on_start)
        cfg.addWidget(QLabel("Sample size:")); cfg.addWidget(self._size)
        cfg.addWidget(QLabel("Mode:")); cfg.addWidget(self._mode)
        cfg.addWidget(self._start)
        cfg.addStretch(1)
        root.addLayout(cfg)

        self._progress = QProgressBar()
        self._progress.setRange(0, 0)  # indeterminate while sampling
        self._progress.setVisible(False)
        root.addWidget(self._progress)

        self._status = QLabel("Pick a sample size + mode, then Start.")
        self._status.setObjectName("muted")
        self._status.setWordWrap(True)
        root.addWidget(self._status)

        # Blind grid.
        self._grid_host = QWidget()
        self._grid = QGridLayout(self._grid_host)
        self._grid.setSpacing(6)
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setWidget(self._grid_host)
        root.addWidget(self._scroll, stretch=1)

        # Results (hidden until scored).
        self._results = QLabel("")
        self._results.setTextFormat(Qt.TextFormat.RichText)
        self._results.setWordWrap(True)
        self._results.setVisible(False)
        root.addWidget(self._results)

        bottom = QHBoxLayout()
        self._counter = QLabel("")
        self._counter.setObjectName("muted")
        self._review_btn = QPushButton("Review disagreements")
        self._review_btn.clicked.connect(self._review_disagreements)
        self._review_btn.setVisible(False)
        self._score_btn = QPushButton("Score")
        self._score_btn.setObjectName("primary")
        self._score_btn.clicked.connect(self._score)
        self._score_btn.setEnabled(False)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        bottom.addWidget(self._counter)
        bottom.addStretch(1)
        bottom.addWidget(self._review_btn)
        bottom.addWidget(self._score_btn)
        bottom.addWidget(close_btn)
        root.addLayout(bottom)

    # ── sampling ───────────────────────────────────────────────────────

    def _on_start(self) -> None:
        if self._thumbs and not self._confirm_restart():
            return
        self._clear_grid()
        self._results.setVisible(False)
        self._review_btn.setVisible(False)
        self._score_btn.setEnabled(False)
        self._start.setEnabled(False)
        self._progress.setVisible(True)
        self._status.setText("Sampling cells…")
        mode = self._mode.currentData()
        self._worker = _SampleWorker(self._h5_path, self._labels_dir, self._size.value(), mode)
        self._worker.done.connect(self._on_sampled)
        self._worker.start()

    def _confirm_restart(self) -> bool:
        return QMessageBox.question(
            self, "Restart?", "Discard the current blind labels and resample?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        ) == QMessageBox.StandardButton.Yes

    def _on_sampled(self, pairs, phase_idx, mask_idx, error: str) -> None:
        self._progress.setVisible(False)
        self._start.setEnabled(True)
        if error:
            self._status.setText(f"Failed: {error}")
            return
        self._pairs = list(pairs)
        self._phase_idx, self._mask_idx = phase_idx, mask_idx
        try:
            self._provider = FovContextProvider(self._h5_path)
        except Exception:  # noqa: BLE001
            self._provider = None

        for i, (record, _orig) in enumerate(self._pairs):
            thumb = _CellThumb(record, float("nan"))
            thumb.classifyKey.connect(lambda lbl, t=thumb: t.set_label(lbl))
            thumb.expandRequested.connect(lambda idx=i: self._open_detail(idx))
            thumb.labelChanged.connect(self._update_counter)
            self._thumbs.append(thumb)
            self._grid.addWidget(thumb, i // self.COLS, i % self.COLS)

        self._status.setText(
            f"{len(self._pairs)} cells — label each good/bad as you see it "
            "(click cycles · g / b / u · double-click to enlarge), then Score."
        )
        self._update_counter()

    def _clear_grid(self) -> None:
        for t in self._thumbs:
            t.setParent(None)
        self._thumbs.clear()
        self._pairs = []

    # ── labelling ──────────────────────────────────────────────────────

    def _labelled_count(self) -> int:
        return sum(1 for t in self._thumbs if t.label in ("good", "bad"))

    def _update_counter(self) -> None:
        done = self._labelled_count()
        self._counter.setText(f"Labelled {done} / {len(self._thumbs)}")
        self._score_btn.setEnabled(done >= 5)

    def _open_detail(self, idx: int) -> None:
        pool = [(rec, float("nan")) for rec, _ in self._pairs]
        dialog = CellDetailView(
            pool=pool, start_index=idx,
            phase_idx=self._phase_idx, mask_idx=self._mask_idx,
            get_label=lambda i: self._thumbs[i].label,
            set_label=lambda i, lbl: self._thumbs[i].set_label(lbl),
            parent=self, context_provider=self._provider,
        )
        dialog.exec()

    # ── scoring ────────────────────────────────────────────────────────

    def _score(self) -> None:
        scored, uids, rows = [], [], []
        for (record, orig), thumb in zip(self._pairs, self._thumbs):
            if thumb.label in ("good", "bad"):
                scored.append((orig, thumb.label))
                uids.append(record.cell_uid)
                rows.append((record.cell_uid, orig, thumb.label))
        if len(scored) < 5:
            QMessageBox.information(self, "Label more",
                                    "Label at least a handful of cells before scoring.")
            return

        stats = consistency_stats(scored, uids)
        self._disagree_uids = set(stats.get("disagreements", []))
        saved = ""
        try:
            saved = str(save_consistency_run(self._labels_dir, self._mode.currentData(), rows, stats))
        except Exception:  # noqa: BLE001
            saved = ""

        self._results.setText(self._results_html(stats, saved))
        self._results.setVisible(True)
        self._review_btn.setVisible(bool(self._disagree_uids))

    def _results_html(self, stats: dict, saved: str) -> str:
        raw = stats["raw_agreement"]
        kappa = stats["cohen_kappa"]
        flips = stats["flips"]
        n = stats["n"]
        g, b = flips.get("good", {}), flips.get("bad", {})
        good_total = g.get("good", 0) + g.get("bad", 0)
        bad_total = b.get("good", 0) + b.get("bad", 0)
        takeaway = (
            "A model can't reliably beat this — a balanced accuracy near it means "
            "you're at the label-noise ceiling, so tighten the rubric rather than "
            "labelling more."
        )
        rows = [
            f"<b>Self-agreement: {raw:.1%}</b> over {n} re-labelled cells",
            f"Cohen's κ: <b>{kappa:.2f}</b> ({stats['kappa_band']})",
            f"good → kept good {g.get('good', 0)} / flipped to bad {g.get('bad', 0)}"
            + (f" ({g.get('bad', 0)/good_total:.0%} flipped)" if good_total else ""),
            f"bad → kept bad {b.get('bad', 0)} / flipped to good {b.get('good', 0)}"
            + (f" ({b.get('good', 0)/bad_total:.0%} flipped)" if bad_total else ""),
            f"<span style='color:{COLORS['muted_text']}'>{takeaway}</span>",
        ]
        if saved:
            rows.append(f"<span style='color:{COLORS['muted_text']}'>saved → {saved}</span>")
        return "<br>".join(rows)

    def _review_disagreements(self) -> None:
        idxs = [i for i, (rec, _) in enumerate(self._pairs)
                if rec.cell_uid in getattr(self, "_disagree_uids", set())]
        if not idxs:
            return
        self._open_detail(idxs[0])  # land on the first; arrow keys walk the pool

    # ── lifecycle ──────────────────────────────────────────────────────

    def closeEvent(self, event) -> None:
        if self._worker is not None and self._worker.isRunning():
            self._worker.wait(3000)
        super().closeEvent(event)
