"""Modal dialog for labelling foci detected by the Foci Detection stage.

Loads parquets from ``output_dir/04c_foci_detection`` and the corresponding
normalised hyperstacks from ``output_dir/04b_fluorescent_normalisation``,
renders a paginated 4×5 grid of single-cell crops with foci as a clickable
scatter overlay, and writes one CSV per session to
``output_dir/05_foci_labels/<timestamp>__<run_id>.csv``.

Click a focus to cycle its label: ``''`` → ``correct`` → ``incorrect`` → ``''``.
"""

from __future__ import annotations

import csv
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from mycomorph.core.label_cells import (
    get_labeled_mask_from_fov,
    load_hyperstack,
)

from ..ui import tokens

# Grid / pagination
GRID_ROWS = 4
GRID_COLS = 5
CELLS_PER_PAGE = GRID_ROWS * GRID_COLS
CROP_PAD = 5    # pixels of padding around the cell's bounding box

# Label cycling order. '' = unlabelled.
LABEL_ORDER: list[str] = ["", "correct", "incorrect"]
LABEL_COLOR_RGBA: dict[str, tuple[int, int, int, int]] = {
    "":          (200, 200, 200, 220),   # grey
    "correct":   ( 60, 200,  90, 240),   # green
    "incorrect": (220,  70,  70, 240),   # red
}


@dataclass
class _FocusKey:
    """Identifier for one focus in the label dictionary."""
    well: str
    fov: int
    channel_index: int
    detector: str
    cell_id: int
    focus_id: int

    def as_tuple(self) -> tuple:
        return (
            self.well, self.fov, self.channel_index, self.detector,
            self.cell_id, self.focus_id,
        )


class FociLabelDialog(QDialog):
    """Modal foci-labelling dialog.

    Parameters
    ----------
    output_dir : Path
        The pipeline run's output directory. Must contain
        ``04b_fluorescent_normalisation/`` and ``04c_foci_detection/``;
        ``05_foci_labels/`` is created on save.
    channel_labels : list[str] | None
        Names of the image channels (mask not included). Used to label the
        channel combobox; falls back to ``channel_<index>`` strings.
    run_id : str
        Identifier baked into the output CSV filename.
    parent : QWidget | None
        Parent widget — usually the MainWindow so the dialog inherits its
        position and is properly modal against it.
    """

    def __init__(
        self,
        output_dir: Path,
        channel_labels: Optional[list[str]] = None,
        run_id: str = "",
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Label foci")
        self.setModal(True)
        self.resize(1200, 800)

        self.output_dir = Path(output_dir)
        self.foci_dir = self.output_dir / "04c_foci_detection"
        self.norm_dir = self.output_dir / "04b_fluorescent_normalisation"
        self.labels_dir = self.output_dir / "05_foci_labels"
        self.channel_labels = list(channel_labels or [])
        self.run_id = run_id or self.output_dir.name or "run"

        # Per-session CSV path. Stamped at construction so multiple saves
        # in the same dialog session land in the same file.
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        self._csv_path = self.labels_dir / f"{ts}__{self.run_id}.csv"

        # Persisted UI state.
        # labels: key → ('', 'correct', 'incorrect')
        self.labels: dict[tuple, str] = {}
        # Foci data for the current view, as a sub-DataFrame.
        self._view_df: pd.DataFrame = pd.DataFrame()
        # Cached normalised hyperstack: (well, fov) → (channels_CYX, mask_YX)
        self._fov_cache: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]] = {}
        # Per-cell list of ScatterPlotItems on the current page so click
        # handlers can find their owning cell.
        self._page_scatters: list[pg.ScatterPlotItem] = []
        # Pagination state.
        self._cell_ids_in_view: list[int] = []
        self._page_index: int = 0

        self._build_ui()
        self._populate_wells()

    # ─────────────────────────────────────────────────────────────────
    # UI
    # ─────────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(tokens.S4, tokens.S4, tokens.S4, tokens.S4)
        root.setSpacing(tokens.S3)

        # Top toolbar: pickers + Save / Close
        tb = QHBoxLayout()
        tb.setSpacing(tokens.S3)

        self.well_combo = QComboBox()
        self.fov_combo = QComboBox()
        self.ch_combo = QComboBox()
        self.det_combo = QComboBox()
        tb.addWidget(QLabel("Well:"));     tb.addWidget(self.well_combo)
        tb.addWidget(QLabel("FOV:"));      tb.addWidget(self.fov_combo)
        tb.addWidget(QLabel("Channel:"));  tb.addWidget(self.ch_combo)
        tb.addWidget(QLabel("Detector:")); tb.addWidget(self.det_combo)
        tb.addStretch(1)

        self.prev_btn = QPushButton("◀ Prev")
        self.prev_btn.clicked.connect(self._on_prev)
        self.next_btn = QPushButton("Next ▶")
        self.next_btn.clicked.connect(self._on_next)
        self.page_label = QLabel("Page 0 / 0")
        self.page_label.setObjectName("muted")
        tb.addWidget(self.prev_btn)
        tb.addWidget(self.page_label)
        tb.addWidget(self.next_btn)

        self.save_btn = QPushButton("Save labels")
        self.save_btn.setObjectName("primary")
        self.save_btn.clicked.connect(self._on_save)
        self.close_btn = QPushButton("Close")
        self.close_btn.clicked.connect(self.reject)
        tb.addWidget(self.save_btn)
        tb.addWidget(self.close_btn)
        root.addLayout(tb)

        # Grid canvas
        self.canvas = pg.GraphicsLayoutWidget()
        try:
            self.canvas.setBackground(tokens.active().bg)
        except Exception:  # noqa: BLE001
            self.canvas.setBackground("#222")
        root.addWidget(self.canvas, stretch=1)

        # Status bar
        self.status = QLabel("Nothing loaded.")
        self.status.setObjectName("muted")
        root.addWidget(self.status)

        # Combobox wiring (set up last so populate_* doesn't fire reloads).
        self.well_combo.currentTextChanged.connect(self._on_well_changed)
        self.fov_combo.currentIndexChanged.connect(self._reload_view)
        self.ch_combo.currentIndexChanged.connect(self._reload_view)
        self.det_combo.currentTextChanged.connect(self._reload_view)

    # ─────────────────────────────────────────────────────────────────
    # Population
    # ─────────────────────────────────────────────────────────────────

    def _populate_wells(self) -> None:
        if not self.foci_dir.exists():
            self.status.setText(
                f"No foci-detection output found at {self.foci_dir}"
            )
            return
        wells = sorted(p.stem for p in self.foci_dir.glob("*.parquet"))
        if not wells:
            self.status.setText(
                f"No parquets in {self.foci_dir}. Run Foci Detection first."
            )
            return
        self.well_combo.blockSignals(True)
        self.well_combo.clear()
        for w in wells:
            self.well_combo.addItem(w)
        self.well_combo.blockSignals(False)
        self._on_well_changed(self.well_combo.currentText())

    def _on_well_changed(self, well: str) -> None:
        if not well:
            return
        parquet = self.foci_dir / f"{well}.parquet"
        if not parquet.exists():
            self.status.setText(f"Missing {parquet}")
            return
        try:
            df = pd.read_parquet(parquet)
        except Exception as e:  # noqa: BLE001
            self.status.setText(f"Failed to read {parquet.name}: {e}")
            return

        # Populate fov / channel / detector combos from the parquet contents.
        self.fov_combo.blockSignals(True)
        self.ch_combo.blockSignals(True)
        self.det_combo.blockSignals(True)
        self.fov_combo.clear()
        self.ch_combo.clear()
        self.det_combo.clear()

        for fov in sorted(df["fov"].unique()):
            self.fov_combo.addItem(str(int(fov)), int(fov))
        # Channel: show "index: name" if we have labels, else just "index".
        for ch in sorted(df["channel_index"].unique()):
            ch = int(ch)
            name = (
                self.channel_labels[ch]
                if 0 <= ch < len(self.channel_labels)
                else f"channel_{ch}"
            )
            self.ch_combo.addItem(f"{ch}: {name}", ch)
        for det in sorted(df["detector"].unique()):
            self.det_combo.addItem(str(det))

        self.fov_combo.blockSignals(False)
        self.ch_combo.blockSignals(False)
        self.det_combo.blockSignals(False)

        self._well_df = df
        self._reload_view()

    # ─────────────────────────────────────────────────────────────────
    # View loading + rendering
    # ─────────────────────────────────────────────────────────────────

    def _current_filters(self) -> Optional[tuple[str, int, int, str]]:
        well = self.well_combo.currentText()
        fov_data = self.fov_combo.currentData()
        ch_data = self.ch_combo.currentData()
        det = self.det_combo.currentText()
        if not well or fov_data is None or ch_data is None or not det:
            return None
        return well, int(fov_data), int(ch_data), det

    def _reload_view(self, *_args) -> None:
        sel = self._current_filters()
        if sel is None or getattr(self, "_well_df", None) is None:
            self._clear_canvas()
            self.status.setText("Pick a well, FOV, channel, and detector.")
            return
        well, fov, ch, det = sel
        df = self._well_df
        mask = (
            (df["fov"] == fov)
            & (df["channel_index"] == ch)
            & (df["detector"] == det)
        )
        view = df.loc[mask].copy()
        self._view_df = view

        # Group by cell_id. Cells with no foci aren't shown — labelling
        # is per-focus, so cells with zero candidates have nothing to do.
        cell_ids = sorted(int(c) for c in view["cell_id"].unique() if int(c) > 0)
        self._cell_ids_in_view = cell_ids
        # Reset to page 0 whenever the filter changes.
        self._page_index = 0
        self._render_page()

    def _render_page(self) -> None:
        self._clear_canvas()
        cells = self._cell_ids_in_view
        if not cells:
            self.status.setText("No cells with detected foci in this view.")
            self.page_label.setText("Page 0 / 0")
            self.prev_btn.setEnabled(False)
            self.next_btn.setEnabled(False)
            return

        n_pages = (len(cells) + CELLS_PER_PAGE - 1) // CELLS_PER_PAGE
        self._page_index = max(0, min(self._page_index, n_pages - 1))
        start = self._page_index * CELLS_PER_PAGE
        page_cells = cells[start:start + CELLS_PER_PAGE]

        sel = self._current_filters()
        assert sel is not None
        well, fov, ch, _det = sel
        try:
            channels, labeled_mask = self._fetch_fov(well, fov)
        except Exception as e:  # noqa: BLE001
            self.status.setText(f"Failed to load FOV image: {e}")
            return

        n_channels = channels.shape[0]
        if ch >= n_channels:
            self.status.setText(
                f"Channel {ch} out of range ({n_channels} channels in the TIFF)."
            )
            return
        image = channels[ch]

        for slot, cell_id in enumerate(page_cells):
            row, col = divmod(slot, GRID_COLS)
            vb = self.canvas.addViewBox(row=row, col=col, lockAspect=True)
            vb.invertY(True)
            vb.setMouseEnabled(x=False, y=False)
            vb.setMenuEnabled(False)
            vb.setBackgroundColor((20, 20, 20))

            crop, (y_min, _, x_min, _) = self._cell_crop(
                image, labeled_mask, cell_id,
            )
            img_item = pg.ImageItem(crop)
            img_item.setLevels((float(crop.min()), float(crop.max())))
            vb.addItem(img_item)

            # Foci on this cell, in crop coords.
            cell_foci = self._view_df.loc[
                self._view_df["cell_id"] == cell_id
            ]
            xs, ys, brushes, pen_colors, focus_ids = [], [], [], [], []
            for _, frow in cell_foci.iterrows():
                fy = float(frow["y"]) - y_min
                fx = float(frow["x"]) - x_min
                xs.append(fx)
                ys.append(fy)
                key = self._key_for(int(frow["focus_id"]), int(cell_id))
                lbl = self.labels.get(key, "")
                brushes.append(LABEL_COLOR_RGBA.get(lbl, LABEL_COLOR_RGBA[""]))
                pen_colors.append((0, 0, 0, 220))
                focus_ids.append(int(frow["focus_id"]))

            scatter = pg.ScatterPlotItem(
                size=11,
                pen=pg.mkPen((0, 0, 0, 220), width=1),
            )
            spots = [
                {
                    "pos": (xs[i], ys[i]),
                    "brush": pg.mkBrush(*brushes[i]),
                    "data": (int(cell_id), focus_ids[i]),
                }
                for i in range(len(xs))
            ]
            scatter.addPoints(spots)
            scatter.sigClicked.connect(self._on_scatter_clicked)
            vb.addItem(scatter)
            self._page_scatters.append(scatter)

            # Small in-corner label so the user can see which cell this is.
            txt = pg.TextItem(
                f"cell {cell_id} ({len(cell_foci)} foci)",
                color=(200, 200, 200, 220),
                anchor=(0, 0),
            )
            txt.setPos(2, 2)
            vb.addItem(txt)

        self.page_label.setText(
            f"Page {self._page_index + 1} / {n_pages} — {len(cells)} cells"
        )
        self.prev_btn.setEnabled(self._page_index > 0)
        self.next_btn.setEnabled(self._page_index < n_pages - 1)
        self._update_status()

    def _clear_canvas(self) -> None:
        self.canvas.clear()
        self._page_scatters = []

    # ─────────────────────────────────────────────────────────────────
    # Data fetching
    # ─────────────────────────────────────────────────────────────────

    def _fetch_fov(
        self, well: str, fov: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Load (channels_CYX, labeled_mask_YX) for the given FOV.

        Cached across renders so paginating through cells of the same FOV
        doesn't re-read the TIFF.
        """
        key = (well, int(fov))
        if key in self._fov_cache:
            return self._fov_cache[key]
        tiff = self.norm_dir / f"{well}.tif"
        if not tiff.exists():
            raise FileNotFoundError(
                f"Normalised TIFF not found: {tiff}. "
                "Did the Fluorescent Normalisation stage run?"
            )
        data, _meta = load_hyperstack(tiff)
        if fov >= data.shape[0]:
            raise IndexError(
                f"FOV {fov} out of range ({data.shape[0]} FOVs in {tiff.name})"
            )
        channels = data[fov]
        mask_channel = channels.shape[0] - 1
        labeled_mask, _ = get_labeled_mask_from_fov(channels, mask_channel)
        # Drop the mask channel from the channels array so 'channel index'
        # in the parquet aligns one-to-one with the array.
        result = (channels, labeled_mask)
        self._fov_cache[key] = result
        return result

    def _cell_crop(
        self, image: np.ndarray, labeled_mask: np.ndarray, cell_id: int,
    ) -> tuple[np.ndarray, tuple[int, int, int, int]]:
        h, w = labeled_mask.shape
        ys, xs = np.where(labeled_mask == cell_id)
        if ys.size == 0:
            return np.zeros((10, 10), dtype=np.float32), (0, 10, 0, 10)
        y0 = max(int(ys.min()) - CROP_PAD, 0)
        y1 = min(int(ys.max()) + CROP_PAD + 1, h)
        x0 = max(int(xs.min()) - CROP_PAD, 0)
        x1 = min(int(xs.max()) + CROP_PAD + 1, w)
        crop = image[y0:y1, x0:x1].astype(np.float32)
        return crop, (y0, y1, x0, x1)

    # ─────────────────────────────────────────────────────────────────
    # Interaction
    # ─────────────────────────────────────────────────────────────────

    def _on_scatter_clicked(self, _scatter, points) -> None:
        if not points:
            return
        for pt in points:
            data = pt.data()
            if not data:
                continue
            cell_id, focus_id = data
            key = self._key_for(int(focus_id), int(cell_id))
            current = self.labels.get(key, "")
            i = LABEL_ORDER.index(current) if current in LABEL_ORDER else 0
            next_label = LABEL_ORDER[(i + 1) % len(LABEL_ORDER)]
            if next_label == "":
                self.labels.pop(key, None)
            else:
                self.labels[key] = next_label
            pt.setBrush(pg.mkBrush(*LABEL_COLOR_RGBA[next_label]))
        self._update_status()

    def _key_for(self, focus_id: int, cell_id: int) -> tuple:
        sel = self._current_filters()
        assert sel is not None
        well, fov, ch, det = sel
        return _FocusKey(
            well=well, fov=fov, channel_index=ch, detector=det,
            cell_id=cell_id, focus_id=focus_id,
        ).as_tuple()

    def _on_prev(self) -> None:
        self._page_index = max(0, self._page_index - 1)
        self._render_page()

    def _on_next(self) -> None:
        self._page_index += 1
        self._render_page()

    def _update_status(self) -> None:
        n_correct = sum(1 for v in self.labels.values() if v == "correct")
        n_incorrect = sum(1 for v in self.labels.values() if v == "incorrect")
        total = len(self._view_df) if not self._view_df.empty else 0
        n_unlabelled = max(total - n_correct - n_incorrect, 0)
        self.status.setText(
            f"{n_correct} correct, {n_incorrect} incorrect, "
            f"{n_unlabelled} unlabelled (this view) — "
            f"{len(self.labels)} labels across all views"
        )

    # ─────────────────────────────────────────────────────────────────
    # Save
    # ─────────────────────────────────────────────────────────────────

    def _on_save(self) -> None:
        if not self.labels:
            QMessageBox.information(
                self, "Nothing to save", "No labels yet — click a focus first."
            )
            return
        try:
            self.labels_dir.mkdir(parents=True, exist_ok=True)
            self._write_csv()
            QMessageBox.information(
                self, "Saved",
                f"Wrote {len(self.labels)} labels to "
                f"{self._csv_path.relative_to(self.output_dir)}.",
            )
        except Exception as e:  # noqa: BLE001
            QMessageBox.warning(
                self, "Save failed", f"Could not write labels CSV:\n{e}",
            )

    def _write_csv(self) -> None:
        """Rewrite the session CSV with the current label set.

        Each save call overwrites the same per-session file so the CSV
        always reflects the latest in-memory state. The filename embeds a
        timestamp + the run_id so re-opening the dialog in a later
        session produces a fresh file.
        """
        ts_iso = datetime.now().isoformat(timespec="seconds")
        with open(self._csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "well", "fov", "channel_index", "detector",
                "cell_id", "focus_id", "label", "labeled_at",
            ])
            for key, lbl in sorted(self.labels.items()):
                well, fov, ch, det, cell_id, focus_id = key
                w.writerow([
                    well, int(fov), int(ch), det,
                    int(cell_id), int(focus_id), lbl, ts_iso,
                ])
