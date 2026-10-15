"""Large single-cell inspector for ModelPrep's labelling grid.

Opens from the grid (double-click a thumbnail) to show one cell at high
resolution. Two views:

* **Context** — the cell located back in its original field-of-view image
  (`FovContextProvider`), so its real neighbours are visible. This is the
  default whenever the pipeline originals sit next to the HDF5. A fixed-size
  window (default 256 px) keeps the scale consistent cell-to-cell and is
  adjustable live with the zoom slider or ``+`` / ``-``.
* **Crop** — the isolated `full_crop` stored in the HDF5 (background masked
  out). Always available; the fallback when the originals can't be found.

Navigation walks the whole loaded pool with ◀ / ▶ buttons or the arrow keys.
Labels applied here (``g`` / ``b`` / ``u``) flow straight back to the grid via
the ``get_label`` / ``set_label`` callbacks. Classifying good/bad auto-advances
to the next cell so the pool can be labelled in one rhythm; ``u`` holds.

The original images may live on slow network storage, so context windows are
fetched on a worker thread (`_ContextWorker`); stale results from fast
navigation are dropped by request id, and the view falls back to the crop
whenever a window can't be produced.
"""

from __future__ import annotations

from typing import Callable, Optional

from PyQt6.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QImage, QKeyEvent, QPixmap
from PyQt6.QtWidgets import (
    QCheckBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from ...core._render import outline_of, to_outlined_rgb
from ...core.crops_h5 import CropRecord
from ...core.fov_context import FovContext, FovContextProvider
from ..palette import COLORS, LABEL_COLORS

_LABEL_ORDER = ["unlabeled", "good", "bad"]

_WINDOW_MIN = 96
_WINDOW_MAX = 1024
_WINDOW_DEFAULT = 256
_WINDOW_STEP = 64


class _ContextWorker(QThread):
    """Fetch one FOV-context window off the UI thread."""

    done = pyqtSignal(int, object)  # request_id, FovContext | None

    def __init__(
        self,
        provider: FovContextProvider,
        record: CropRecord,
        window_px: int,
        request_id: int,
    ) -> None:
        super().__init__()
        self._provider = provider
        self._record = record
        self._window_px = window_px
        self._request_id = request_id

    def run(self) -> None:
        try:
            ctx = self._provider.get_window(self._record, self._window_px)
        except Exception:  # noqa: BLE001 — never let a worker crash the dialog
            ctx = None
        self.done.emit(self._request_id, ctx)


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
        context_provider: Optional[FovContextProvider] = None,
    ) -> None:
        super().__init__(parent)
        self._pool = pool
        self._idx = start_index
        self._phase_idx = phase_idx
        self._mask_idx = mask_idx
        self._get_label = get_label
        self._set_label = set_label
        self._provider = context_provider

        self._context_ok = bool(context_provider and context_provider.available)
        # Online-only (cloud placeholder) originals would be downloaded in full
        # (~2 GB per TIFF) the moment context is shown, so don't default to it.
        self._online_bytes = context_provider.online_only_bytes() if self._context_ok else 0
        self._mode = "context" if self._context_ok and not self._online_bytes else "crop"
        self._window_px = _WINDOW_DEFAULT
        self._img_key: Optional[tuple] = None  # (idx, mode, window_px) shown
        self._req_id = 0
        self._inflight: list[_ContextWorker] = []
        self._src_rgb = None  # last rendered image, kept so resize can rescale it

        self.setWindowTitle("Cell inspector")
        self.setModal(True)
        # Start roughly square around IMAGE_PX with room for the side panel; the
        # image grows as the user enlarges the window.
        self.resize(self.IMAGE_PX + 360, self.IMAGE_PX + 120)
        self.setMinimumSize(560, 440)
        self._build_ui()
        self._sync_view()

    # ── UI build ───────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QHBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(12)

        # Left: the large image + a status line. The image expands to fill the
        # window so making it larger gives a bigger view.
        left = QVBoxLayout()
        left.setSpacing(6)
        self._image = QLabel()
        self._image.setMinimumSize(256, 256)
        self._image.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        left.addWidget(self._image, stretch=1)
        self._view_status = QLabel("")
        self._view_status.setStyleSheet(f"color: {COLORS['muted_text']};")
        self._view_status.setWordWrap(True)
        left.addWidget(self._view_status)
        root.addLayout(left, stretch=1)

        # Right: metadata + controls.
        side = QVBoxLayout()
        side.setSpacing(8)

        self._counter = QLabel()
        self._counter.setStyleSheet("font-weight: bold;")
        side.addWidget(self._counter)

        self._meta = QLabel()
        self._meta.setTextFormat(Qt.TextFormat.RichText)
        self._meta.setWordWrap(True)
        self._meta.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self._meta.setStyleSheet(f"color: {COLORS['muted_text']};")
        side.addWidget(self._meta)

        side.addStretch(1)

        # View toggle + zoom.
        self._ctx_toggle = QCheckBox("Show original-image context")
        self._ctx_toggle.setChecked(self._mode == "context")
        self._ctx_toggle.setEnabled(self._context_ok)
        if not self._context_ok:
            self._ctx_toggle.setToolTip(
                "Not available: the original pipeline images (01_focus plus "
                "02_segment or 03_classify) aren't in the folder above this HDF5's."
            )
        elif self._online_bytes:
            self._ctx_toggle.setToolTip(
                "The original images are online-only (cloud storage). Turning this "
                f"on downloads them, about {self._online_bytes / 1e9:.1f} GB."
            )
        self._ctx_toggle.toggled.connect(self._on_toggle_context)
        side.addWidget(self._ctx_toggle)

        self._zoom_row = QWidget()
        zoom = QHBoxLayout(self._zoom_row)
        zoom.setContentsMargins(0, 0, 0, 0)
        self._zoom_label = QLabel()
        self._zoom = QSlider(Qt.Orientation.Horizontal)
        self._zoom.setRange(_WINDOW_MIN, _WINDOW_MAX)
        self._zoom.setSingleStep(_WINDOW_STEP)
        self._zoom.setPageStep(_WINDOW_STEP)
        self._zoom.setValue(self._window_px)
        self._zoom.valueChanged.connect(self._on_zoom_changed)
        zoom.addWidget(QLabel("Zoom:"))
        zoom.addWidget(self._zoom, stretch=1)
        zoom.addWidget(self._zoom_label)
        side.addWidget(self._zoom_row)

        # Debounce zoom reloads so dragging the slider doesn't spawn a worker
        # per pixel.
        self._zoom_timer = QTimer(self)
        self._zoom_timer.setSingleShot(True)
        self._zoom_timer.setInterval(120)
        self._zoom_timer.timeout.connect(self._sync_view)

        side.addSpacing(8)

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

        hint = QLabel(
            "g / b classify + advance · u clears · ← / → navigate · "
            "+ / − zoom · Esc closes"
        )
        hint.setStyleSheet(f"color: {COLORS['muted_text']};")
        hint.setWordWrap(True)
        side.addWidget(hint)

        root.addLayout(side)

    # ── view sync ──────────────────────────────────────────────────────

    def _sync_view(self) -> None:
        """Refresh metadata/controls, then (re)load the image if what it should
        show has changed."""
        self._render_meta()
        key = (self._idx, self._mode, self._window_px)
        if key != self._img_key:
            self._img_key = key
            self._load_image()

    def _render_meta(self) -> None:
        record, score = self._pool[self._idx]
        label = self._get_label(self._idx)

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

        self._zoom_row.setVisible(self._mode == "context")
        self._zoom_label.setText(f"{self._window_px}px")

    # ── image loading ──────────────────────────────────────────────────

    def _load_image(self) -> None:
        record, _ = self._pool[self._idx]
        if self._mode == "crop" or self._provider is None:
            self._show_crop(record)
            return

        # Context mode: fetch off-thread; keep showing whatever's there with a
        # loading note until the window arrives.
        self._view_status.setText("Loading context…")
        self._req_id += 1
        worker = _ContextWorker(self._provider, record, self._window_px, self._req_id)
        worker.done.connect(self._on_context_ready)
        worker.finished.connect(lambda w=worker: self._inflight.remove(w) if w in self._inflight else None)
        self._inflight.append(worker)
        worker.start()

    def _on_context_ready(self, request_id: int, ctx: object) -> None:
        if request_id != self._req_id:
            return  # superseded by newer navigation/zoom
        if isinstance(ctx, FovContext):
            self._set_image_rgb(to_outlined_rgb(ctx.image, ctx.outline))
            self._view_status.setText(
                f"Original image · FOV {ctx.fov_index} · {ctx.window_px}px window"
            )
        else:
            # Couldn't locate this cell in the originals — fall back to the crop
            # for this cell without leaving context mode.
            record, _ = self._pool[self._idx]
            self._show_crop(record)
            self._view_status.setText(
                "Context unavailable for this cell — showing isolated crop."
            )

    def _show_crop(self, record: CropRecord) -> None:
        phase = record.full_crop[self._phase_idx]
        mask = record.full_crop[self._mask_idx] > 0.5
        self._set_image_rgb(to_outlined_rgb(phase, outline_of(mask)))
        if self._mode == "crop":
            self._view_status.setText("Isolated crop (HDF5)")

    def _set_image_rgb(self, rgb) -> None:
        """Cache the source image and render it at the current label size."""
        self._src_rgb = rgb
        self._apply_pixmap()

    def _apply_pixmap(self) -> None:
        """Scale the cached source image to fit the image label (square,
        aspect-preserved). Called on every render and on window resize."""
        if self._src_rgb is None:
            return
        rgb = self._src_rgb
        h, w = rgb.shape[:2]
        qimg = QImage(rgb.tobytes(), w, h, w * 3, QImage.Format.Format_RGB888)
        avail = self._image.size()
        side = max(64, min(avail.width(), avail.height()) - 16)
        self._image.setPixmap(
            QPixmap.fromImage(qimg).scaled(
                side,
                side,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._apply_pixmap()

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
            ("P(good)", "not scored" if score != score else f"{score:.3f}"),
        ]
        body = "".join(
            f"<tr><td style='padding-right:10px; color:{COLORS['border_strong']};'>"
            f"{k}</td><td>{v}</td></tr>"
            for k, v in rows
        )
        return f"<table>{body}</table>"

    # ── controls ───────────────────────────────────────────────────────

    def _on_toggle_context(self, checked: bool) -> None:
        self._mode = "context" if checked else "crop"
        self._sync_view()

    def _on_zoom_changed(self, value: int) -> None:
        self._window_px = int(value)
        self._zoom_label.setText(f"{self._window_px}px")
        if self._mode == "context":
            self._zoom_timer.start()  # debounce → _sync_view

    def _nudge_zoom(self, delta: int) -> None:
        self._zoom.setValue(
            max(_WINDOW_MIN, min(_WINDOW_MAX, self._zoom.value() + delta))
        )

    # ── label flow ─────────────────────────────────────────────────────

    def _apply_label(self, name: str) -> None:
        self._set_label(self._idx, name)
        # Auto-advance after a good/bad classification so the user can rip
        # through the pool continuously; 'unlabeled' is a correction, so hold.
        if name in ("good", "bad") and self._idx < len(self._pool) - 1:
            self._idx += 1
        self._sync_view()

    # ── navigation ─────────────────────────────────────────────────────

    def _prev(self) -> None:
        if self._idx > 0:
            self._idx -= 1
            self._sync_view()

    def _next(self) -> None:
        if self._idx < len(self._pool) - 1:
            self._idx += 1
            self._sync_view()

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
        elif key in (Qt.Key.Key_Plus, Qt.Key.Key_Equal) and self._mode == "context":
            self._nudge_zoom(_WINDOW_STEP)
        elif key == Qt.Key.Key_Minus and self._mode == "context":
            self._nudge_zoom(-_WINDOW_STEP)
        else:
            super().keyPressEvent(event)

    def closeEvent(self, event) -> None:
        # Let any in-flight context workers finish before teardown so their
        # signals don't fire into a half-destroyed dialog.
        for w in list(self._inflight):
            w.wait(2000)
        super().closeEvent(event)
