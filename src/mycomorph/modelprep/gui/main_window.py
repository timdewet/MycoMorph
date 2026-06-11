"""ModelPrep main window — MycoMorph-style chrome.

A header bar (breadcrumb + brand + logo), a left navigation sidebar with a
live system-info footer, and a stacked set of cards for the three steps of
the workflow (Source → Label cells → Fine-tune). Shares its visual language
with MycoMorph via the ported ``gui.ui`` design system.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import QSize, Qt
from PyQt6.QtGui import QAction, QIcon
from PyQt6.QtWidgets import (
    QApplication,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QScrollArea,
    QSizePolicy,
    QStackedWidget,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from ._resources import resource_root
from ..core.device import device_label, gpu_available
from .panels.label_train_panel import LabelTrainPanel
from .panels.models_panel import ModelsPanel
from .panels.source_panel import SourcePanel
from .ui import tokens
from .ui.elevation import apply_shadow
from .ui.nav_sidebar import NavEntry, NavSidebar


SETTINGS_FILE = Path.home() / ".modelprep" / "settings.json"


# The three workflow steps. ``pipeline=False`` keeps the sidebar clean — no
# status dots or group dividers, since ModelPrep has no per-stage run state to
# surface there (label/train carry their own progress bars).
NAV_ENTRIES = [
    NavEntry("source", "Source",      "folder",   pipeline=False),
    NavEntry("label",  "Label cells", "label",    pipeline=False),
    NavEntry("train",  "Fine-tune",   "model",    pipeline=False),
    NavEntry("models", "Models",      "analysis", pipeline=False),
]


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("ModelPrep")

        self._logo_path = resource_root() / "logo" / "logo.svg"
        if self._logo_path.exists():
            self.setWindowIcon(QIcon(str(self._logo_path)))

        # Size to a comfortable default, capped to 90% of the screen so the
        # window fits on smaller laptop displays.
        screen = self.screen() or QApplication.primaryScreen()
        avail = screen.availableGeometry() if screen else None
        target_w, target_h = 1280, 880
        if avail is not None:
            target_w = min(target_w, int(avail.width() * 0.9))
            target_h = min(target_h, int(avail.height() * 0.9))
        self.resize(target_w, target_h)

        self.source_panel = SourcePanel()
        self.label_train_panel = LabelTrainPanel()
        self.models_panel = ModelsPanel()

        # Wire signals before chrome so callbacks can reach widgets.
        self.source_panel.sourceChanged.connect(self._on_source_changed)
        self.source_panel.modelNameChanged.connect(self._on_model_name_changed)
        self.models_panel.useModelRequested.connect(self._on_use_model)
        self.models_panel.modelsChanged.connect(self._on_models_changed)

        self._build_chrome()
        self._apply_card_shadows()
        self._build_menu()
        self._show_device_status()

        # Keep the (never-shown) controller panel anchored under this window so
        # its modal dialogs (cell detail, message boxes) centre correctly.
        self.label_train_panel.setParent(self)
        self.label_train_panel.hide()

        # Seed the trainer with the initial model name from the Source page.
        self.label_train_panel.set_model_name(self.source_panel.current_model_name())

        self._restore_settings()

    # ── chrome ───────────────────────────────────────────────────────────

    def _build_chrome(self) -> None:
        # Header: breadcrumb on the left, brand + logo on the right.
        self._header = QFrame()
        self._header.setObjectName("header")
        self._header.setFixedHeight(tokens.HEADER_HEIGHT)
        hl = QHBoxLayout(self._header)
        hl.setContentsMargins(tokens.S5, 0, tokens.S5, 0)
        hl.setSpacing(tokens.S3)

        self._breadcrumb = QLabel(f"Step 1 of {len(NAV_ENTRIES)} · Source")
        self._breadcrumb.setObjectName("breadcrumb")
        hl.addWidget(self._breadcrumb)
        hl.addStretch(1)

        brand_name = QLabel("ModelPrep")
        brand_name.setObjectName("brandName")
        hl.addWidget(brand_name)
        self._add_header_logo(hl)

        header_div = QFrame()
        header_div.setObjectName("headerDivider")

        # Sidebar nav.
        self._sidebar = NavSidebar(NAV_ENTRIES, brand="ModelPrep")
        self._sidebar.set_gpu_available(gpu_available())
        self._sidebar.currentChanged.connect(self._on_nav_changed)

        # Stacked pages, one per nav entry.
        self._stack = QStackedWidget()
        self._stack.addWidget(self._wrap_scroll(self.source_panel))
        self._stack.addWidget(self._wrap_fill(self.label_train_panel.label_section))
        self._stack.addWidget(self._wrap_scroll(self.label_train_panel.train_section))
        self._stack.addWidget(self._wrap_fill(self.models_panel))

        # Body: sidebar | divider | stack.
        body = QWidget()
        body_l = QHBoxLayout(body)
        body_l.setContentsMargins(0, 0, 0, 0)
        body_l.setSpacing(0)
        body_l.addWidget(self._sidebar)
        divider = QFrame()
        divider.setObjectName("sidebarDivider")
        divider.setFrameShape(QFrame.Shape.NoFrame)
        body_l.addWidget(divider)
        body_l.addWidget(self._stack, stretch=1)

        # Compose header + body.
        wrap = QWidget()
        v = QVBoxLayout(wrap)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)
        v.addWidget(self._header)
        v.addWidget(header_div)
        v.addWidget(body, stretch=1)
        self.setCentralWidget(wrap)

        # Status bar: status message (left) + trained-models hint (right).
        sb = QStatusBar()
        self._status_msg = QLabel("Ready.")
        self._status_msg.setObjectName("muted")
        sb.addWidget(self._status_msg, 1)
        self._status_models = QLabel("")
        self._status_models.setObjectName("caption")
        sb.addPermanentWidget(self._status_models)
        self.setStatusBar(sb)

    def _add_header_logo(self, layout: QHBoxLayout) -> None:
        if not self._logo_path.exists():
            return
        try:
            from PyQt6.QtGui import QPainter, QPixmap
            from PyQt6.QtSvg import QSvgRenderer
        except Exception:  # noqa: BLE001
            return  # QtSvg unavailable — brand text alone is fine.
        logo_lbl = QLabel()
        size = QSize(44, 44)
        logo_lbl.setFixedSize(size)
        renderer = QSvgRenderer(str(self._logo_path))
        pix = QPixmap(size)
        pix.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pix)
        renderer.render(painter)
        painter.end()
        logo_lbl.setPixmap(pix)
        logo_lbl.setStyleSheet("background: transparent;")
        layout.addWidget(logo_lbl)

    def _wrap_scroll(self, widget: QWidget) -> QWidget:
        """Top-pinned scroll page — forms hug the top, no empty stretch."""
        wrap = QWidget()
        wv = QVBoxLayout(wrap)
        wv.setContentsMargins(tokens.S5, tokens.S5, tokens.S5, tokens.S5)
        wv.setSpacing(tokens.S3)
        scroll = QScrollArea()
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        inner = QWidget()
        iv = QVBoxLayout(inner)
        iv.setContentsMargins(0, 0, 0, 0)
        iv.setSpacing(tokens.S3)
        widget.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        iv.addWidget(widget)
        iv.addStretch(1)
        scroll.setWidget(inner)
        wv.addWidget(scroll)
        return wrap

    def _wrap_fill(self, widget: QWidget) -> QWidget:
        """Fill page — the widget owns its full vertical space (the grid)."""
        wrap = QWidget()
        wv = QVBoxLayout(wrap)
        wv.setContentsMargins(tokens.S5, tokens.S5, tokens.S5, tokens.S5)
        wv.setSpacing(tokens.S3)
        widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        wv.addWidget(widget, stretch=1)
        return wrap

    def _apply_card_shadows(self) -> None:
        """Soft drop shadows on the form-style cards.

        The label card is skipped on purpose: it hosts the scrolling thumbnail
        grid, and a QGraphicsEffect over a large scroll viewport is needless
        overhead — the QSS border + radius already reads as a card. MycoMorph
        treats its big scrolling surfaces the same way.
        """
        for card in (self.source_panel, self.label_train_panel.train_section):
            if isinstance(card, QGroupBox):
                apply_shadow(card, level=1)

    def _build_menu(self) -> None:
        menu = self.menuBar()
        file_menu = menu.addMenu("&File")

        open_action = QAction("&Open HDF5…", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self.source_panel.browse_h5)
        file_menu.addAction(open_action)

        file_menu.addSeparator()
        quit_action = QAction("&Quit", self)
        quit_action.setShortcut("Ctrl+Q")
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

    # ── nav + source ──────────────────────────────────────────────────────

    def _on_nav_changed(self, idx: int) -> None:
        if not (0 <= idx < len(NAV_ENTRIES)):
            return
        self._stack.setCurrentIndex(idx)
        entry = NAV_ENTRIES[idx]
        if entry.key == "source":
            # Pick up names of models trained since this page was last shown.
            self.source_panel.refresh_models()
        elif entry.key == "models":
            # Pick up runs trained since this page was last shown.
            self.models_panel.refresh()
        self._breadcrumb.setText(
            f"Step {idx + 1} of {len(NAV_ENTRIES)} · {entry.label}"
        )

    # ── models tab ──────────────────────────────────────────────────────────

    def _on_use_model(self, weights: Path) -> None:
        """Models tab → make this model active and jump to Label cells."""
        self.label_train_panel.select_model(weights)
        try:
            label_idx = next(i for i, e in enumerate(NAV_ENTRIES) if e.key == "label")
        except StopIteration:
            return
        self._sidebar.set_current(label_idx)

    def _on_models_changed(self) -> None:
        """A run was deleted — drop it from the model pickers too."""
        self.label_train_panel.refresh_models()
        self.source_panel.refresh_models()

    def _on_source_changed(
        self, h5_path: Optional[Path], labels_dir: Optional[Path]
    ) -> None:
        self.label_train_panel.set_source(h5_path, labels_dir)
        self._save_settings()

    def _on_model_name_changed(self, name: str) -> None:
        self.label_train_panel.set_model_name(name)
        self._save_settings()

    # ── status bar ─────────────────────────────────────────────────────────

    def _show_device_status(self) -> None:
        self._status_msg.setText(f"Ready · compute: {device_label()}")
        self._status_models.setText("trained models: ~/.modelprep/models/")

    # ── persistence ─────────────────────────────────────────────────────────

    def _restore_settings(self) -> None:
        if not SETTINGS_FILE.exists():
            return
        try:
            data = json.loads(SETTINGS_FILE.read_text())
        except Exception:  # noqa: BLE001
            return
        name = data.get("last_model_name")
        if name:
            self.source_panel.set_model_name(name)
        last_h5 = data.get("last_h5")
        if last_h5:
            p = Path(last_h5)
            if p.exists():
                # Restoring kicks off _on_source_changed via the signal.
                self.source_panel.set_h5_path(p)

    def _save_settings(self) -> None:
        SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
        h5 = self.source_panel.current_h5_path()
        try:
            SETTINGS_FILE.write_text(json.dumps({
                "last_h5": str(h5) if h5 else "",
                "last_model_name": self.source_panel.current_model_name(),
            }, indent=2))
        except Exception:  # noqa: BLE001
            pass

    def closeEvent(self, event) -> None:  # noqa: N802 (Qt API)
        if self.label_train_panel.has_unsaved_labels():
            ans = QMessageBox.question(
                self, "Unsaved labels",
                "You have labelled cells on the current page that aren't saved "
                "yet. Save them before quitting?",
                QMessageBox.StandardButton.Save
                | QMessageBox.StandardButton.Discard
                | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Save,
            )
            if ans == QMessageBox.StandardButton.Cancel:
                event.ignore()
                return
            if ans == QMessageBox.StandardButton.Save:
                self.label_train_panel.save_pending()
        self._save_settings()
        self.label_train_panel.shutdown()
        super().closeEvent(event)
