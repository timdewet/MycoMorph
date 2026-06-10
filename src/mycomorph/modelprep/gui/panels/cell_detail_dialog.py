"""Large single-cell inspector for ModelPrep's labelling grid.

Opens from the grid (double-click a thumbnail) to show one cell at high
resolution. Unlike the 96px thumbnail — which renders only the downsampled
64×64 classifier crop — this view draws the raw 128×128 ``full_crop``, so there
is more image around the cell ("surrounding context").

Navigation walks the whole loaded pool with ◀ / ▶ buttons or the arrow keys.
Labels applied here (``g`` / ``b`` / ``u``) flow straight back to the grid via
the ``get_label`` / ``set_label`` callbacks supplied by the panel. Classifying
a cell good or bad auto-advances to the next one so the pool can be labelled in
a single rhythm; ``u`` (a correction) holds on the current cell.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QKeyEvent, QPixmap
from PyQt6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ...core.crops_h5 import CropRecord
from ..palette import COLORS, LABEL_COLORS

_LABEL_ORDER = ["unlabeled", "good", "bad"]


class CellDetailView(QDialog):
    """Full-resolution viewer for one cell, with pool navigation + labelling."""

    IMAGE_PX = 512

    def __init__(
        self,
        pool: list[tuple[CropRecord, float]],
        start_index: int,
        phase_idx: int,
        mask_idx: int,
        get_label: Callable[[int], str],
        set_label: Callable[[int, str], None],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._pool = pool
        self._idx = start_index
        self._phase_idx = phase_idx
        self._mask_idx = mask_idx
        self._get_label = get_label
        self._set_label = set_label

        self.setWindowTitle("Cell inspector")
        self.setModal(True)
        self._build_ui()
        self._render()

    # ── UI build ───────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QHBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(12)

        # Left: the large image.
        self._image = QLabel()
        self._image.setFixedSize(self.IMAGE_PX, self.IMAGE_PX)
        self._image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root.addWidget(self._image)

        # Right: metadata + controls.
        side = QVBoxLayout()
        side.setSpacing(8)

        self._counter = QLabel()
        self._counter.setStyleSheet("font-weight: bold;")
        side.addWidget(self._counter)

        self._meta = QLabel()
        self._meta.setTextFormat(Qt.TextFormat.RichText)
        self._meta.setWordWrap(True)
        self._meta.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self._meta.setStyleSheet(f"color: {COLORS['muted_text']};")
        side.addWidget(self._meta)

        side.addStretch(1)

        # Label buttons.
        side.addWidget(QLabel("Label (g / b / u):"))
        self._label_btns: dict[str, QPushButton] = {}
        for name in _LABEL_ORDER:
            btn = QPushButton(name.capitalize())
            btn.setCheckable(True)
            btn.clicked.connect(lambda _checked, n=name: self._apply_label(n))
            self._label_btns[name] = btn
            side.addWidget(btn)

        side.addSpacing(8)

        # Navigation.
        nav = QHBoxLayout()
        self._prev_btn = QPushButton("◀ Prev")
        self._prev_btn.clicked.connect(self._prev)
        self._next_btn = QPushButton("Next ▶")
        self._next_btn.clicked.connect(self._next)
        nav.addWidget(self._prev_btn)
        nav.addWidget(self._next_btn)
        side.addLayout(nav)

        hint = QLabel("g / b classify + advance · u clears · ← / → navigate · Esc closes")
        hint.setStyleSheet(f"color: {COLORS['muted_text']};")
        side.addWidget(hint)

        root.addLayout(side, stretch=1)

    # ── rendering ──────────────────────────────────────────────────────

    def _render(self) -> None:
        record, score = self._pool[self._idx]
        label = self._get_label(self._idx)

        self._image.setPixmap(self._make_pixmap(record))
        self._image.setStyleSheet(
            f"background: {COLORS['surface']}; "
            f"border: 4px solid {LABEL_COLORS[label]}; border-radius: 6px;"
        )

        self._counter.setText(f"Cell {self._idx + 1} of {len(self._pool)}")
        self._meta.setText(self._meta_html(record, score))

        for name, btn in self._label_btns.items():
            btn.setChecked(name == label)

        self._prev_btn.setEnabled(self._idx > 0)
        self._next_btn.setEnabled(self._idx < len(self._pool) - 1)

    def _make_pixmap(self, record: CropRecord) -> QPixmap:
        from skimage.segmentation import find_boundaries

        # full_crop layout: (C, 128, 128) — raw HDF5 channels.
        phase = record.full_crop[self._phase_idx]
        mask = record.full_crop[self._mask_idx] > 0.5

        lo, hi = np.percentile(phase, (1, 99))
        scaled = np.clip((phase - lo) / max(hi - lo, 1e-6), 0, 1)
        gray = (scaled * 255).astype(np.uint8)
        h, w = gray.shape
        rgb = np.stack([gray, gray, gray], axis=-1)
        if mask.any():
            boundary = find_boundaries(mask, mode="outer")
            rgb[boundary] = (0x21, 0xD4, 0xFD)  # cyan
        rgb = np.ascontiguousarray(rgb)

        qimg = QImage(rgb.tobytes(), w, h, w * 3, QImage.Format.Format_RGB888)
        return QPixmap.fromImage(qimg).scaled(
            self.IMAGE_PX - 16,
            self.IMAGE_PX - 16,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )

    def _meta_html(self, record: CropRecord, score: float) -> str:
        rows = [
            ("cell_uid", record.cell_uid),
            ("condition", record.condition_label),
            ("gene", record.gene),
            ("drug", record.drug),
            ("reporter", record.reporter),
            ("tiff_file", record.tiff_file),
            ("fov_index", record.fov_index),
            ("cell_label", record.cell_label),
            ("area_px", f"{record.area_px} px"),
            ("P(good)", f"{score:.3f}"),
        ]
        body = "".join(
            f"<tr><td style='padding-right:10px; color:{COLORS['border_strong']};'>"
            f"{k}</td><td>{v}</td></tr>"
            for k, v in rows
        )
        return f"<table>{body}</table>"

    # ── label flow ─────────────────────────────────────────────────────

    def _apply_label(self, name: str) -> None:
        self._set_label(self._idx, name)
        # Auto-advance after a good/bad classification so the user can rip
        # through the pool continuously; 'unlabeled' is a correction, so hold
        # on the current cell. At the end of the pool we stay put.
        if name in ("good", "bad") and self._idx < len(self._pool) - 1:
            self._idx += 1
        self._render()

    # ── navigation ─────────────────────────────────────────────────────

    def _prev(self) -> None:
        if self._idx > 0:
            self._idx -= 1
            self._render()

    def _next(self) -> None:
        if self._idx < len(self._pool) - 1:
            self._idx += 1
            self._render()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        key = event.key()
        if key == Qt.Key.Key_Left:
            self._prev()
        elif key == Qt.Key.Key_Right:
            self._next()
        elif key == Qt.Key.Key_G:
            self._apply_label("good")
        elif key == Qt.Key.Key_B:
            self._apply_label("bad")
        elif key == Qt.Key.Key_U:
            self._apply_label("unlabeled")
        else:
            super().keyPressEvent(event)
