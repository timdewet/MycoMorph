"""Per-stage option panels — small forms wrapping api.py dataclasses."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
import pyqtgraph as pg

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from mycomorph.core.api import (
    PRESET_MODELS,
    ClassifyOpts,
    FocusOpts,
    SegmentOpts,
    resolve_classifier_preset,
)
from mycomorph.core.foci import DetectorOpts
from mycomorph.core.foci.detectors import (
    BACTERIAL_SPECIFIC_KEYS,
    CLASSICAL_BASELINE_KEYS,
    DEEP_LEARNING_KEYS,
    DIM_SIGNAL_KEYS,
)
from mycomorph.core.foci.normalise import NORMALISER_REGISTRY

from ..pipeline.context import (
    FluorescentNormalisationOpts,
    FociDetectionOpts,
)
from ..ui import icons, tokens
from ..ui.labeled_slider import LabeledSlider
from ..widgets.foci_filter_io import (
    FILTER_FEATURES,
    compute_pass_mask,
    save_foci_filter,
)


def _with_helper(widget: QWidget, helper: str) -> QWidget:
    """Wrap a form-row widget with helper text below it."""
    from PyQt6.QtWidgets import QVBoxLayout
    holder = QWidget()
    v = QVBoxLayout(holder)
    v.setContentsMargins(0, 0, 0, 0)
    v.setSpacing(2)
    v.addWidget(widget)
    if helper:
        cap = QLabel(helper)
        cap.setObjectName("caption")
        cap.setStyleSheet(
            f"color: {tokens.active().text_subtle}; font-size: {tokens.FS_CAPTION}px;"
        )
        cap.setWordWrap(True)
        v.addWidget(cap)
    return holder


class FocusPanel(QWidget):
    # Fires whenever any option widget on this panel changes value. The
    # live preview controller listens for this and re-runs focus +
    # downstream stages. Suppressed during ``restore_state`` via the
    # ``_loading`` guard so app startup doesn't cause spurious renders.
    optionsChanged = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        from PyQt6.QtWidgets import QGroupBox, QVBoxLayout

        self._loading = False
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(tokens.S3)

        opt_box = QGroupBox("Focus options")
        self._form = QFormLayout(opt_box)
        self._form.setContentsMargins(tokens.S4, tokens.S5, tokens.S4, tokens.S4)
        self._form.setHorizontalSpacing(tokens.S4)
        self._form.setVerticalSpacing(tokens.S3)
        root.addWidget(opt_box)
        root.addStretch(1)

        self.mode = QComboBox()
        self.mode.addItems(["edf", "tiles", "whole"])      # EDF is the default
        self.metric = QComboBox()
        self.metric.addItems(["ensemble", "normalized_variance", "brenner",
                              "tenengrad", "laplacian", "sml", "vollath"])
        self.metric.setCurrentText("ensemble")             # tested default

        # Tile-grid row, hidden unless mode == "tiles"
        self.tile_rows = QSpinBox(); self.tile_rows.setRange(1, 16); self.tile_rows.setValue(3)
        self.tile_cols = QSpinBox(); self.tile_cols.setRange(1, 16); self.tile_cols.setValue(3)
        tile_row = QHBoxLayout()
        tile_row.addWidget(self.tile_rows); tile_row.addWidget(QLabel("×")); tile_row.addWidget(self.tile_cols)
        tile_row.addStretch(1)
        self._tile_wrap = QWidget(); self._tile_wrap.setLayout(tile_row)
        self._tile_label = QLabel("Tile grid (rows × cols):")

        self._form.addRow("Mode:", _with_helper(
            self.mode,
            "edf = stack the per-tile in-focus pixels (recommended). "
            "tiles = pick best Z per tile. whole = single best Z per FOV.",
        ))
        self._form.addRow("Metric:", _with_helper(
            self.metric,
            "Sharpness measure used to score Z-planes. 'ensemble' averages "
            "several robust metrics and is a safe default.",
        ))
        self._form.addRow(self._tile_label, self._tile_wrap)

        self.save_zmaps = QCheckBox("Save per-FOV Z-map TIFFs (EDF debug output)")
        self.save_zmaps.setChecked(False)
        self._form.addRow("", self.save_zmaps)

        self.save_mip = QCheckBox("Add MIP companion channel for each fluorescence channel")
        self.save_mip.setChecked(False)
        self._form.addRow("", self.save_mip)

        self.mode.currentTextChanged.connect(self._refresh_tile_visibility)
        self._refresh_tile_visibility(self.mode.currentText())

        # External phase-channel override — set from the Input tab.
        self._phase_channel: int | str | None = None

        # Fan in every option-widget value-changed signal into the
        # single ``optionsChanged`` signal the controller listens to.
        for w in (self.mode, self.metric):
            w.currentIndexChanged.connect(self._emit_options_changed)
        for w in (self.tile_rows, self.tile_cols):
            w.valueChanged.connect(self._emit_options_changed)
        for w in (self.save_zmaps, self.save_mip):
            w.toggled.connect(self._emit_options_changed)

    def _emit_options_changed(self, *_args) -> None:
        if not self._loading:
            self.optionsChanged.emit()

    def _refresh_tile_visibility(self, mode: str) -> None:
        show = mode == "tiles"
        self._tile_wrap.setVisible(show)
        self._tile_label.setVisible(show)

    def set_phase_channel(self, phase: int | str | None) -> None:
        self._phase_channel = phase

    def opts(self) -> FocusOpts:
        return FocusOpts(
            metric=self.metric.currentText(),
            mode=self.mode.currentText(),
            tile_grid=(self.tile_rows.value(), self.tile_cols.value()),
            phase_channel=self._phase_channel,
            save_zmaps=self.save_zmaps.isChecked(),
            save_mip=self.save_mip.isChecked(),
        )

    def state(self) -> dict:
        return {
            "metric": self.metric.currentText(),
            "mode": self.mode.currentText(),
            "tile_rows": self.tile_rows.value(),
            "tile_cols": self.tile_cols.value(),
            "save_zmaps": self.save_zmaps.isChecked(),
            "save_mip": self.save_mip.isChecked(),
        }

    def restore_state(self, s: dict) -> None:
        if not isinstance(s, dict):
            return
        self._loading = True
        try:
            if "mode" in s:
                i = self.mode.findText(str(s["mode"]))
                if i >= 0:
                    self.mode.setCurrentIndex(i)
            if "metric" in s:
                i = self.metric.findText(str(s["metric"]))
                if i >= 0:
                    self.metric.setCurrentIndex(i)
            if "tile_rows" in s:
                self.tile_rows.setValue(int(s["tile_rows"]))
            if "tile_cols" in s:
                self.tile_cols.setValue(int(s["tile_cols"]))
            if "save_zmaps" in s:
                self.save_zmaps.setChecked(bool(s["save_zmaps"]))
            if "save_mip" in s:
                self.save_mip.setChecked(bool(s["save_mip"]))
        finally:
            self._loading = False


class SegmentPanel(QWidget):
    """Pure options widget for the Segment stage.

    The single-FOV preview now lives in the persistent live preview
    column to the right of this tab; this panel only carries options.
    """

    optionsChanged = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        from PyQt6.QtWidgets import QGroupBox, QVBoxLayout

        self._loading = False
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(8)

        opt_box = QGroupBox("Segmentation options")
        form = QFormLayout(opt_box)
        form.setContentsMargins(tokens.S4, tokens.S5, tokens.S4, tokens.S4)
        form.setHorizontalSpacing(tokens.S4)
        form.setVerticalSpacing(tokens.S3)
        self.diameter = QDoubleSpinBox(); self.diameter.setRange(0.0, 500.0); self.diameter.setSpecialValueText("auto")
        # Default to a concrete value (rather than ``auto``) so cellpose's
        # rescale-by-diameter pass lands on a sensible image size — the
        # cpsam ViT-SAM backbone has fixed-size positional embeddings
        # and crashes on unusually large rescaled grids. 25 px matches
        # typical *Mtb* imaging at our standard 100×/1.4 objective; the
        # user can override if their imaging differs.
        self.diameter.setValue(25.0)
        self.gpu = QCheckBox("Use GPU")
        self.gpu.setChecked(True)
        self.pixels_per_um = QDoubleSpinBox()
        self.pixels_per_um.setRange(0.0, 10000.0)
        self.pixels_per_um.setValue(13.8767)
        self.pixels_per_um.setDecimals(4)
        self._pixel_hint = QLabel("(auto-detect from CZI once a file is loaded)")
        self._pixel_hint.setObjectName("muted")
        self._user_overrode_pixels = False
        self.pixels_per_um.valueChanged.connect(self._mark_user_override)

        # Cellpose tunables — sliders for thresholds (visual feedback for
        # tuning) plus precise numeric input.
        self.flow_threshold = LabeledSlider(
            "Flow threshold", 0.0, 3.0, 0.4, step=0.05, decimals=2,
            helper="Higher → keep more marginal masks. Default 0.4. Try 0.6–1.0 if cells are missed.",
        )
        self.cellprob_threshold = LabeledSlider(
            "Cell-prob threshold", -6.0, 6.0, 0.0, step=0.1, decimals=2,
            helper="Lower (more negative) → keep more low-probability cells. Default 0. Try −1 to −2 if cells are missed.",
        )

        self.min_size = QSpinBox()
        self.min_size.setRange(0, 5000); self.min_size.setValue(15)
        self.min_size.setToolTip("Minimum mask area in pixels. Default 15. Set to 0 to disable.")

        form.addRow("Diameter (px):", _with_helper(
            self.diameter,
            "Median cell diameter in pixels; 0 = let Cellpose estimate.",
        ))
        form.addRow("Min size (px):", _with_helper(
            self.min_size,
            "Drop masks smaller than this many pixels. 0 disables the filter.",
        ))
        form.addRow("", self.gpu)
        form.addRow("Pixels per µm:", _with_helper(
            self.pixels_per_um, ""  # hint label below carries the helper
        ))
        form.addRow("", self._pixel_hint)
        # Sliders sit below the form, full-width, with their own helper text.
        sliders_label = QLabel("Mask-quality thresholds")
        sliders_label.setObjectName("h3")
        from PyQt6.QtWidgets import QVBoxLayout as _QVB
        opt_box_layout = opt_box.layout()  # the form
        # Move out of QFormLayout: stack the two sliders below
        _holder = QWidget()
        _holder_l = _QVB(_holder)
        _holder_l.setContentsMargins(0, tokens.S2, 0, 0)
        _holder_l.setSpacing(tokens.S3)
        _holder_l.addWidget(sliders_label)
        _holder_l.addWidget(self.flow_threshold)
        _holder_l.addWidget(self.cellprob_threshold)
        # Add sliders as a wide form row spanning both columns
        form.addRow(_holder)

        root.addWidget(opt_box)
        root.addStretch(1)

        # Fan in option-widget signals → optionsChanged.
        self.diameter.valueChanged.connect(self._emit_options_changed)
        self.min_size.valueChanged.connect(self._emit_options_changed)
        self.gpu.toggled.connect(self._emit_options_changed)
        self.pixels_per_um.valueChanged.connect(self._emit_options_changed)
        self.flow_threshold.valueChanged.connect(self._emit_options_changed)
        self.cellprob_threshold.valueChanged.connect(self._emit_options_changed)

    def _emit_options_changed(self, *_args) -> None:
        if not self._loading:
            self.optionsChanged.emit()

    def _mark_user_override(self, _value: float) -> None:
        # Any manual edit sticks; auto-detection from CZI won't clobber it.
        self._user_overrode_pixels = True

    def set_detected_pixels_per_um(self, value: float | None) -> None:
        """Called from the Input panel when a CZI is selected."""
        if value is None:
            self._pixel_hint.setText("(pixel size could not be read from CZI)")
            return
        self._pixel_hint.setText(f"(auto-detected from CZI: {value:.3f} px/µm)")
        if not self._user_overrode_pixels:
            self.pixels_per_um.blockSignals(True)
            self.pixels_per_um.setValue(value)
            self.pixels_per_um.blockSignals(False)

    def opts(self) -> SegmentOpts:
        diameter = None if self.diameter.value() == 0.0 else self.diameter.value()
        return SegmentOpts(
            diameter=diameter,
            flow_threshold=self.flow_threshold.value(),
            cellprob_threshold=self.cellprob_threshold.value(),
            min_size=self.min_size.value(),
            gpu=self.gpu.isChecked(),
            pixels_per_um=self.pixels_per_um.value(),
        )

    def state(self) -> dict:
        return {
            "diameter": self.diameter.value(),
            "flow_threshold": self.flow_threshold.value(),
            "cellprob_threshold": self.cellprob_threshold.value(),
            "min_size": self.min_size.value(),
            "gpu": self.gpu.isChecked(),
            "pixels_per_um": self.pixels_per_um.value(),
            "user_overrode_pixels": self._user_overrode_pixels,
        }

    def restore_state(self, s: dict) -> None:
        if not isinstance(s, dict):
            return
        self._loading = True
        try:
            # ``model_type`` was a dropdown in older builds; ignore it if
            # present in saved state — cellpose 4.x ships only cpsam.
            if "diameter" in s:
                self.diameter.setValue(float(s["diameter"]))
            if "flow_threshold" in s:
                self.flow_threshold.setValue(float(s["flow_threshold"]))
            if "cellprob_threshold" in s:
                self.cellprob_threshold.setValue(float(s["cellprob_threshold"]))
            if "min_size" in s:
                self.min_size.setValue(int(s["min_size"]))
            if "gpu" in s:
                self.gpu.setChecked(bool(s["gpu"]))
            if "pixels_per_um" in s:
                # Block the override flag while restoring; the user's manual
                # state is captured separately below.
                self.pixels_per_um.blockSignals(True)
                self.pixels_per_um.setValue(float(s["pixels_per_um"]))
                self.pixels_per_um.blockSignals(False)
            if "user_overrode_pixels" in s:
                self._user_overrode_pixels = bool(s["user_overrode_pixels"])
        finally:
            self._loading = False


class ClassifyPanel(QWidget):
    """Classifier options + small model-inspector. The combined
    Segment & Classify tab embeds this directly; the inspector lives
    inside this panel so it stays adjacent to the threshold control.
    """

    # Emitted when the user clicks one of the "Show …" buttons.
    # Connected by MainWindow to pop the corresponding standalone window.
    openLabelTrainerRequested = pyqtSignal()
    showModelDetailsRequested = pyqtSignal()
    optionsChanged = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        from PyQt6.QtWidgets import QGroupBox, QVBoxLayout
        from ..widgets.model_inspector import ModelInspector

        self._loading = False
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(8)

        # ── Options group ────────────────────────────────────────────────
        opt_box = QGroupBox("Classifier options")
        form = QFormLayout(opt_box)
        form.setContentsMargins(tokens.S4, tokens.S5, tokens.S4, tokens.S4)
        form.setHorizontalSpacing(tokens.S4)
        form.setVerticalSpacing(tokens.S3)

        self.preset = QComboBox()
        self.preset.addItems(["none (rules only)"] + sorted(PRESET_MODELS.keys()) + ["custom…"])
        self.preset.currentIndexChanged.connect(self._on_preset)

        self.custom_path = QLineEdit()
        self.custom_path.setReadOnly(True)
        self.custom_path.setPlaceholderText("(no custom model selected)")
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._pick_custom)
        path_row = QHBoxLayout()
        path_row.addWidget(self.custom_path); path_row.addWidget(browse)
        path_wrap = QWidget(); path_wrap.setLayout(path_row)

        self.use_rules = QCheckBox("Apply rule-based pre-filter (edge / debris)")
        self.use_rules.setChecked(True)

        self.confidence = LabeledSlider(
            "Confidence threshold", 0.0, 1.0, 0.5, step=0.01, decimals=2,
            helper="Cells classified as 'good' below this score are rejected. "
                   "Higher = stricter. Default 0.5.",
        )
        self.confidence.valueChanged.connect(self._on_confidence_changed)

        form.addRow("Classifier:", self.preset)
        form.addRow("Custom model (.pth):", path_wrap)
        form.addRow("", self.use_rules)
        from PyQt6.QtWidgets import QVBoxLayout as _QVB
        _conf_holder = QWidget()
        _conf_l = _QVB(_conf_holder)
        _conf_l.setContentsMargins(0, tokens.S2, 0, 0)
        _conf_l.addWidget(self.confidence)
        form.addRow(_conf_holder)
        root.addWidget(opt_box)

        # Pop-out buttons grouped in a card: model details and the
        # labeller / trainer live in their own windows so this tab can
        # stay focused on the controls that drive the pipeline. The
        # single-FOV segmentation preview now lives in the always-on
        # live preview column to the right of this tab.
        from PyQt6.QtWidgets import QFrame
        actions_card = QFrame()
        actions_card.setObjectName("card")
        button_row = QHBoxLayout(actions_card)
        button_row.setContentsMargins(tokens.S3, tokens.S2, tokens.S3, tokens.S2)
        button_row.setSpacing(tokens.S2)

        self._show_details_btn = QPushButton("  Model details")
        self._show_details_btn.setIcon(icons.icon("model", role="muted"))
        self._show_details_btn.setToolTip(
            "Show ROC / precision-recall curves and training stats for the "
            "currently-selected classifier model in a separate window. The live "
            "threshold marker on the plot tracks this tab's confidence spinner."
        )
        self._show_details_btn.clicked.connect(self.showModelDetailsRequested.emit)

        self._open_labeler_btn = QPushButton("  Labeller / trainer")
        self._open_labeler_btn.setIcon(icons.icon("label", role="muted"))
        self._open_labeler_btn.setToolTip(
            "Open the active-learning labeller and classifier fine-tuner in a "
            "separate window. Use it to build training data from the current "
            "segmented output and retrain the model."
        )
        self._open_labeler_btn.clicked.connect(self.openLabelTrainerRequested.emit)

        button_row.addWidget(self._show_details_btn)
        button_row.addWidget(self._open_labeler_btn)
        button_row.addStretch(1)
        root.addWidget(actions_card)
        root.addStretch(1)

        # The inspector widget is still constructed (and kept fed with the
        # current model and threshold), but it lives in a pop-up window
        # rather than in this panel — see MainWindow._open_model_details.
        self.inspector = ModelInspector()

        self.custom_path.textChanged.connect(self._refresh_inspector)
        self._refresh_inspector()

        # Fan in option-widget signals → optionsChanged.
        self.preset.currentIndexChanged.connect(self._emit_options_changed)
        self.custom_path.textChanged.connect(self._emit_options_changed)
        self.use_rules.toggled.connect(self._emit_options_changed)
        self.confidence.valueChanged.connect(self._emit_options_changed)

    def _emit_options_changed(self, *_args) -> None:
        if not self._loading:
            self.optionsChanged.emit()

    def _on_preset(self) -> None:
        text = self.preset.currentText()
        self.custom_path.setEnabled(text == "custom…")
        self._refresh_inspector()

    def _pick_custom(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select classifier .pth", "", "PyTorch (*.pth)")
        if path:
            self.custom_path.setText(path)

    def _on_confidence_changed(self, value: float) -> None:
        self.inspector.set_threshold(value)

    def _refresh_inspector(self) -> None:
        """Point the inspector at the currently-selected model (or None)."""
        text = self.preset.currentText()
        if text.startswith("none"):
            self.inspector.set_model(None)
            return
        if text == "custom…":
            p = self.custom_path.text().strip()
            self.inspector.set_model(Path(p) if p else None)
            return
        try:
            self.inspector.set_model(resolve_classifier_preset(text))
        except (KeyError, FileNotFoundError):
            self.inspector.set_model(None)
        self.inspector.set_threshold(self.confidence.value())

    def opts(self) -> ClassifyOpts:
        text = self.preset.currentText()
        model_path: Path | None = None
        if text.startswith("none"):
            model_path = None
        elif text == "custom…":
            p = self.custom_path.text().strip()
            model_path = Path(p) if p else None
        else:
            try:
                model_path = resolve_classifier_preset(text)
            except (KeyError, FileNotFoundError):
                model_path = None
        return ClassifyOpts(
            model_path=model_path,
            use_rules=self.use_rules.isChecked(),
            confidence_threshold=self.confidence.value(),
        )

    def state(self) -> dict:
        return {
            "preset": self.preset.currentText(),
            "custom_path": self.custom_path.text(),
            "use_rules": self.use_rules.isChecked(),
            "confidence": self.confidence.value(),
        }

    def restore_state(self, s: dict) -> None:
        if not isinstance(s, dict):
            return
        self._loading = True
        try:
            if "preset" in s:
                i = self.preset.findText(str(s["preset"]))
                if i >= 0:
                    self.preset.setCurrentIndex(i)
            if "custom_path" in s:
                self.custom_path.setText(str(s["custom_path"]))
            if "use_rules" in s:
                self.use_rules.setChecked(bool(s["use_rules"]))
            if "confidence" in s:
                self.confidence.setValue(float(s["confidence"]))
        finally:
            self._loading = False


class SegmentClassifyPanel(QWidget):
    """Combined Segment + Classify tab.

    Stacks SegmentPanel above ClassifyPanel — both option groups feed
    the live preview column on the right. Re-exposes the inner panels
    as ``.segment_panel`` and ``.classify_panel`` so MainWindow can wire
    signals/options through.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        from PyQt6.QtWidgets import QVBoxLayout

        self.segment_panel = SegmentPanel()
        self.classify_panel = ClassifyPanel()

        # Vertical stack: Segment above, Classify below. The combined tab
        # used to put these side-by-side, but the live preview reclaims
        # the right side of the window so the options column is now too
        # narrow for two cards across.
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(tokens.S4)
        root.addWidget(self.segment_panel)
        root.addWidget(self.classify_panel)
        root.addStretch(1)

    def set_detected_pixels_per_um(self, value):
        self.segment_panel.set_detected_pixels_per_um(value)


# ─────────────────────────────────────────────────────────────────────────────
# Foci-detection panels
# ─────────────────────────────────────────────────────────────────────────────


class _ChannelIndexSelect(QWidget):
    """Compact multi-select listing fluorescence channels by ``index: name``.

    Channel names are populated lazily via ``set_channels``; the selection
    persists as the *original* channel index (so toggling which channel is
    phase doesn't reshuffle saved selections). The phase channel is hidden
    from the list — these panels only operate on fluorescence.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        self._list = QListWidget()
        self._list.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        self._list.setMaximumHeight(110)
        layout.addWidget(self._list)
        self._channels: list[str] = []
        self._exclude: set[int] = set()
        # ``_index_at_row[row]`` → original channel index for the item at
        # that visible row. Lets selected_indices() return the user-stable
        # original indices even though phase rows are filtered out.
        self._index_at_row: list[int] = []
        self._pending_selection: list[int] | None = None

    def set_channels(
        self,
        names: list[str],
        exclude_indices: list[int] | None = None,
    ) -> None:
        self._channels = list(names)
        self._exclude = set(exclude_indices or [])
        prev_idx = self.selected_indices() or self._pending_selection
        self._list.clear()
        self._index_at_row = []
        for i, name in enumerate(self._channels):
            if i in self._exclude:
                continue
            item = QListWidgetItem(f"{i}: {name}")
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsSelectable)
            self._list.addItem(item)
            self._index_at_row.append(i)
        if prev_idx:
            self.set_selected_indices(list(prev_idx))
            self._pending_selection = None

    def selected_indices(self) -> list[int]:
        return [
            self._index_at_row[self._list.row(item)]
            for item in self._list.selectedItems()
        ]

    def set_selected_indices(self, indices: list[int]) -> None:
        if self._list.count() == 0:
            self._pending_selection = list(indices)
            return
        self._list.clearSelection()
        for orig_idx in indices:
            try:
                row = self._index_at_row.index(int(orig_idx))
            except ValueError:
                continue
            self._list.item(row).setSelected(True)


class _DetectorKeySelect(QWidget):
    """Single-select of one detector REGISTRY key, grouped by family.

    Normalisation is now a separate upstream stage (Fluorescent
    Normalisation), so the panel only surfaces *pure* detectors —
    bundled "normalise + detect" variants (e.g. ``tophat_dog``,
    ``decon_bm3d_wavelet``) would double-apply normalisation when the
    upstream stage already handled it. They stay in the library for the
    benchmark notebook but aren't exposed here.

    Non-selectable header rows segment the list into "classical
    baselines", "dim signal", "bacterial-specific", "deep learning".
    """

    _GROUPS: list[tuple[str, list[str]]] = [
        ("Classical baselines",    CLASSICAL_BASELINE_KEYS),
        ("Dim signal",             DIM_SIGNAL_KEYS),
        ("Bacterial-specific",     BACTERIAL_SPECIFIC_KEYS),
        ("Deep learning",          DEEP_LEARNING_KEYS),
    ]

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        self._list = QListWidget()
        self._list.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        self._list.setMaximumHeight(180)
        layout.addWidget(self._list)
        self._key_rows: dict[str, int] = {}
        self._build_items()

    def _build_items(self) -> None:
        self._list.clear()
        self._key_rows.clear()
        for header, keys in self._GROUPS:
            hdr = QListWidgetItem(f"— {header} —")
            hdr.setFlags(Qt.ItemFlag.NoItemFlags)
            hdr.setForeground(Qt.GlobalColor.gray)
            self._list.addItem(hdr)
            for key in keys:
                item = QListWidgetItem(key)
                self._list.addItem(item)
                self._key_rows[key] = self._list.row(item)

    def selected_key(self) -> str | None:
        items = self._list.selectedItems()
        if not items:
            return None
        item = items[0]
        if not (item.flags() & Qt.ItemFlag.ItemIsSelectable):
            return None
        return item.text()

    def set_selected_key(self, key: str | None) -> None:
        self._list.clearSelection()
        if key is None:
            return
        row = self._key_rows.get(key)
        if row is not None:
            self._list.item(row).setSelected(True)


class FluorescentNormalisationPanel(QWidget):
    """Options for the Fluorescent Normalisation stage.

    A method combobox plus per-method parameter widgets that show / hide
    based on the selected normaliser. ``apply_to_channels`` lets the user
    restrict normalisation to a subset of fluorescence channels (empty
    selection → every channel that isn't phase or mask).
    """

    optionsChanged = pyqtSignal()

    # Method key → list of param-widget keys to show. Anything not in the
    # list is hidden for that method.
    _METHOD_PARAMS: dict[str, list[str]] = {
        "none":            [],
        "tophat":          ["tophat_radius_px"],
        "gaussian_lp":     ["gaussian_lp_sigma"],
        "richardson_lucy": ["rl_iterations", "rl_psf_sigma"],
        "bm3d":            ["bm3d_sigma"],
        "decon_bm3d":      ["rl_iterations", "rl_psf_sigma", "bm3d_sigma"],
        "basic":           [],
    }

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._loading = False
        self._channel_names: list[str] = []
        self._phase_channel: int | None = None
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(tokens.S3)

        box = QGroupBox("Fluorescent Normalisation options")
        form = QFormLayout(box)
        form.setContentsMargins(tokens.S4, tokens.S5, tokens.S4, tokens.S4)
        form.setHorizontalSpacing(tokens.S4)
        form.setVerticalSpacing(tokens.S3)
        root.addWidget(box)
        root.addStretch(1)

        self.method = QComboBox()
        for key in NORMALISER_REGISTRY:
            self.method.addItem(key)
        # Default to top-hat — fast and effective for most fluor channels.
        idx = self.method.findText("tophat")
        if idx >= 0:
            self.method.setCurrentIndex(idx)
        form.addRow("Method:", _with_helper(
            self.method,
            "Per-FOV normaliser applied to fluorescent channels before "
            "foci detection. `none` passes images through unchanged.",
        ))

        # Top-hat: structuring-element radius in pixels.
        self.tophat_radius_px = QSpinBox()
        self.tophat_radius_px.setRange(1, 100)
        self.tophat_radius_px.setValue(5)
        self._row_tophat = QLabel("Top-hat radius (px):")
        form.addRow(self._row_tophat, self.tophat_radius_px)

        # Gaussian low-pass: sigma + auto checkbox.
        self.gaussian_lp_sigma = QDoubleSpinBox()
        self.gaussian_lp_sigma.setRange(0.1, 200.0)
        self.gaussian_lp_sigma.setDecimals(2)
        self.gaussian_lp_sigma.setValue(20.0)
        self.gaussian_lp_auto = QCheckBox("auto (min_dim / 8)")
        self.gaussian_lp_auto.setChecked(True)
        lp_row = QHBoxLayout()
        lp_row.setContentsMargins(0, 0, 0, 0)
        lp_row.addWidget(self.gaussian_lp_sigma)
        lp_row.addWidget(self.gaussian_lp_auto)
        lp_row.addStretch(1)
        self._lp_wrap = QWidget()
        self._lp_wrap.setLayout(lp_row)
        self._row_gauss = QLabel("Low-pass σ:")
        form.addRow(self._row_gauss, self._lp_wrap)

        # Richardson-Lucy params: iterations + PSF sigma.
        self.rl_iterations = QSpinBox()
        self.rl_iterations.setRange(1, 500)
        self.rl_iterations.setValue(30)
        self._row_rl_iter = QLabel("Richardson–Lucy iterations:")
        form.addRow(self._row_rl_iter, self.rl_iterations)

        self.rl_psf_sigma = QDoubleSpinBox()
        self.rl_psf_sigma.setRange(0.1, 10.0)
        self.rl_psf_sigma.setDecimals(2)
        self.rl_psf_sigma.setSingleStep(0.1)
        self.rl_psf_sigma.setValue(1.5)
        self._row_rl_psf = QLabel("Richardson–Lucy PSF σ (px):")
        form.addRow(self._row_rl_psf, self.rl_psf_sigma)

        # BM3D sigma + auto checkbox.
        self.bm3d_sigma = QDoubleSpinBox()
        self.bm3d_sigma.setRange(0.0, 65535.0)
        self.bm3d_sigma.setDecimals(2)
        self.bm3d_sigma.setValue(0.0)
        self.bm3d_auto = QCheckBox("auto (MAD estimate)")
        self.bm3d_auto.setChecked(True)
        bm3d_row = QHBoxLayout()
        bm3d_row.setContentsMargins(0, 0, 0, 0)
        bm3d_row.addWidget(self.bm3d_sigma)
        bm3d_row.addWidget(self.bm3d_auto)
        bm3d_row.addStretch(1)
        self._bm3d_wrap = QWidget()
        self._bm3d_wrap.setLayout(bm3d_row)
        self._row_bm3d = QLabel("BM3D σ:")
        form.addRow(self._row_bm3d, self._bm3d_wrap)

        self.channels = _ChannelIndexSelect()
        form.addRow("Apply to channels:", _with_helper(
            self.channels,
            "Empty → every channel that isn't phase or mask.",
        ))

        # All params present on the form — track widget pairs so we can
        # show/hide whole rows by method.
        self._param_rows: dict[str, tuple[QWidget, QWidget]] = {
            "tophat_radius_px":  (self._row_tophat,  self.tophat_radius_px),
            "gaussian_lp_sigma": (self._row_gauss,   self._lp_wrap),
            "rl_iterations":     (self._row_rl_iter, self.rl_iterations),
            "rl_psf_sigma":      (self._row_rl_psf,  self.rl_psf_sigma),
            "bm3d_sigma":        (self._row_bm3d,    self._bm3d_wrap),
        }

        self.method.currentTextChanged.connect(self._refresh_param_visibility)
        self.gaussian_lp_auto.toggled.connect(
            lambda on: self.gaussian_lp_sigma.setEnabled(not on)
        )
        self.bm3d_auto.toggled.connect(
            lambda on: self.bm3d_sigma.setEnabled(not on)
        )
        self.gaussian_lp_sigma.setEnabled(not self.gaussian_lp_auto.isChecked())
        self.bm3d_sigma.setEnabled(not self.bm3d_auto.isChecked())
        self._refresh_param_visibility(self.method.currentText())

        for w in (self.method,):
            w.currentIndexChanged.connect(self._emit_options_changed)
        for w in (
            self.tophat_radius_px, self.rl_iterations,
        ):
            w.valueChanged.connect(self._emit_options_changed)
        for w in (
            self.gaussian_lp_sigma, self.rl_psf_sigma, self.bm3d_sigma,
        ):
            w.valueChanged.connect(self._emit_options_changed)
        for w in (self.gaussian_lp_auto, self.bm3d_auto):
            w.toggled.connect(self._emit_options_changed)
        self.channels._list.itemSelectionChanged.connect(self._emit_options_changed)

    def _emit_options_changed(self, *_args) -> None:
        if not self._loading:
            self.optionsChanged.emit()

    def _refresh_param_visibility(self, method: str) -> None:
        visible_keys = set(self._METHOD_PARAMS.get(method, []))
        for key, (label, widget) in self._param_rows.items():
            on = key in visible_keys
            label.setVisible(on)
            widget.setVisible(on)

    # ── external wiring ─────────────────────────────────────────────
    def set_channels(self, names: list[str]) -> None:
        self._channel_names = list(names)
        self._refresh_channel_list()

    def set_phase_channel(self, phase: int | None) -> None:
        self._phase_channel = phase if isinstance(phase, int) else None
        self._refresh_channel_list()

    def _refresh_channel_list(self) -> None:
        exclude: list[int] = []
        if isinstance(self._phase_channel, int):
            exclude.append(self._phase_channel)
        self.channels.set_channels(self._channel_names, exclude_indices=exclude)

    # ── opts / persistence ──────────────────────────────────────────
    def opts(self) -> FluorescentNormalisationOpts:
        selected = self.channels.selected_indices()
        return FluorescentNormalisationOpts(
            method=self.method.currentText(),
            tophat_radius_px=int(self.tophat_radius_px.value()),
            gaussian_lp_sigma=(
                None if self.gaussian_lp_auto.isChecked()
                else float(self.gaussian_lp_sigma.value())
            ),
            rl_iterations=int(self.rl_iterations.value()),
            rl_psf_sigma=float(self.rl_psf_sigma.value()),
            bm3d_sigma=(
                None if self.bm3d_auto.isChecked()
                else float(self.bm3d_sigma.value())
            ),
            apply_to_channels=selected or None,
        )

    def state(self) -> dict:
        return {
            "method": self.method.currentText(),
            "tophat_radius_px": int(self.tophat_radius_px.value()),
            "gaussian_lp_sigma": float(self.gaussian_lp_sigma.value()),
            "gaussian_lp_auto": self.gaussian_lp_auto.isChecked(),
            "rl_iterations": int(self.rl_iterations.value()),
            "rl_psf_sigma": float(self.rl_psf_sigma.value()),
            "bm3d_sigma": float(self.bm3d_sigma.value()),
            "bm3d_auto": self.bm3d_auto.isChecked(),
            "apply_to_channels": self.channels.selected_indices(),
        }

    def restore_state(self, s: dict) -> None:
        if not isinstance(s, dict):
            return
        self._loading = True
        try:
            if "method" in s:
                i = self.method.findText(str(s["method"]))
                if i >= 0:
                    self.method.setCurrentIndex(i)
            if "tophat_radius_px" in s:
                self.tophat_radius_px.setValue(int(s["tophat_radius_px"]))
            if "gaussian_lp_sigma" in s:
                self.gaussian_lp_sigma.setValue(float(s["gaussian_lp_sigma"]))
            if "gaussian_lp_auto" in s:
                self.gaussian_lp_auto.setChecked(bool(s["gaussian_lp_auto"]))
            if "rl_iterations" in s:
                self.rl_iterations.setValue(int(s["rl_iterations"]))
            if "rl_psf_sigma" in s:
                self.rl_psf_sigma.setValue(float(s["rl_psf_sigma"]))
            if "bm3d_sigma" in s:
                self.bm3d_sigma.setValue(float(s["bm3d_sigma"]))
            if "bm3d_auto" in s:
                self.bm3d_auto.setChecked(bool(s["bm3d_auto"]))
            if "apply_to_channels" in s:
                self.channels.set_selected_indices(
                    list(s["apply_to_channels"] or [])
                )
            self._refresh_param_visibility(self.method.currentText())
            self.gaussian_lp_sigma.setEnabled(
                not self.gaussian_lp_auto.isChecked()
            )
            self.bm3d_sigma.setEnabled(not self.bm3d_auto.isChecked())
        finally:
            self._loading = False


class FociDetectionPanel(QWidget):
    """Options for the Foci Detection stage.

    Multi-select list of detector keys (grouped by family) plus shared
    DetectorOpts knobs (sigma range, threshold, SNR floor) and a channel
    multi-select. The "Label foci…" button opens the FociLabelDialog when
    a Foci-Detection run has already produced parquet output.
    """

    optionsChanged = pyqtSignal()
    labelRequested = pyqtSignal()
    # Fires when the user drags any of the inline histogram threshold
    # lines. The host wires this to the live-preview controller's
    # ``apply_thresholds_only`` hot path so the scatter re-filters
    # without triggering a detector re-run.
    thresholdsChanged = pyqtSignal()
    # Fires when the user toggles the "Show foci on preview" checkbox.
    # Wired in MainWindow to LivePreviewPanel.set_foci_overlay_visible
    # so the scatter can flip on/off without re-detection.
    fociVisibilityChanged = pyqtSignal(bool)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._loading = False
        self._channel_names: list[str] = []
        self._phase_channel: int | None = None
        # Output directory + run id — populated by MainWindow when the
        # user picks them in the Input panel. Used by the inline "Save
        # filter" action; both default to ``None`` so the button stays
        # disabled until a destination is known.
        self._output_dir = None
        self._run_id = ""
        # Cached features DataFrame for the *current* FOV, pushed in by
        # the live-preview controller. Drives the inline histograms.
        self._foci_df: pd.DataFrame | None = None
        # Per-feature widgets created lazily in the histogram section.
        self._hist_plots: dict[str, pg.PlotWidget] = {}
        self._hist_lines: dict[str, pg.InfiniteLine] = {}
        self._hist_bars: dict[str, pg.BarGraphItem] = {}

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(tokens.S3)

        det_box = QGroupBox("Detector")
        det_layout = QVBoxLayout(det_box)
        det_layout.setContentsMargins(tokens.S4, tokens.S5, tokens.S4, tokens.S4)
        det_layout.setSpacing(tokens.S3)
        self.detectors = _DetectorKeySelect()
        det_layout.addWidget(self.detectors)
        cap = QLabel(
            "Pick one detector. `wavelet` is a strong default for dim signal; "
            "`dog` / `log` are faster baselines. Normalisation has already "
            "been applied upstream by the Fluorescent Normalisation stage."
        )
        cap.setObjectName("caption")
        cap.setStyleSheet(
            f"color: {tokens.active().text_subtle}; "
            f"font-size: {tokens.FS_CAPTION}px;"
        )
        cap.setWordWrap(True)
        det_layout.addWidget(cap)
        root.addWidget(det_box)

        opt_box = QGroupBox("Detector parameters (shared)")
        form = QFormLayout(opt_box)
        form.setContentsMargins(tokens.S4, tokens.S5, tokens.S4, tokens.S4)
        form.setHorizontalSpacing(tokens.S4)
        form.setVerticalSpacing(tokens.S3)
        root.addWidget(opt_box)

        defaults = DetectorOpts()

        self.min_sigma = QDoubleSpinBox()
        self.min_sigma.setRange(0.1, 20.0)
        self.min_sigma.setDecimals(2)
        self.min_sigma.setSingleStep(0.1)
        self.min_sigma.setValue(defaults.min_sigma)
        form.addRow("min σ (px):", self.min_sigma)

        self.max_sigma = QDoubleSpinBox()
        self.max_sigma.setRange(0.1, 50.0)
        self.max_sigma.setDecimals(2)
        self.max_sigma.setSingleStep(0.1)
        self.max_sigma.setValue(defaults.max_sigma)
        form.addRow("max σ (px):", self.max_sigma)

        self.threshold = QDoubleSpinBox()
        self.threshold.setRange(0.0, 1.0)
        self.threshold.setDecimals(4)
        self.threshold.setSingleStep(0.001)
        self.threshold.setValue(defaults.threshold)
        form.addRow("Threshold:", _with_helper(
            self.threshold,
            "Relative intensity threshold on the normalised image (0–1). "
            "Lower → more permissive (more false positives, easier to filter).",
        ))

        self.snr_min = QDoubleSpinBox()
        self.snr_min.setRange(0.0, 100.0)
        self.snr_min.setDecimals(2)
        self.snr_min.setSingleStep(0.1)
        self.snr_min.setValue(defaults.snr_min)
        form.addRow("SNR floor:", _with_helper(
            self.snr_min,
            "Discard candidates whose peak / local-bg-std falls below this. "
            "Permissive detection + downstream filtering is the recommended flow.",
        ))

        self.refine = QCheckBox("Sub-pixel Gaussian refinement")
        self.refine.setChecked(defaults.refine)
        form.addRow("", self.refine)

        ch_box = QGroupBox("Channels")
        ch_layout = QVBoxLayout(ch_box)
        ch_layout.setContentsMargins(tokens.S4, tokens.S5, tokens.S4, tokens.S4)
        self.channels = _ChannelIndexSelect()
        ch_layout.addWidget(self.channels)
        ch_cap = QLabel(
            "Empty → every channel that isn't phase or mask."
        )
        ch_cap.setObjectName("caption")
        ch_cap.setStyleSheet(
            f"color: {tokens.active().text_subtle}; "
            f"font-size: {tokens.FS_CAPTION}px;"
        )
        ch_cap.setWordWrap(True)
        ch_layout.addWidget(ch_cap)
        root.addWidget(ch_box)

        # Visibility toggle — lets the user flick the scatter on/off so
        # they can eyeball the underlying image without re-running.
        self.show_foci_check = QCheckBox("Show foci on preview")
        self.show_foci_check.setChecked(True)
        self.show_foci_check.setToolTip(
            "Hide the foci scatter overlay on the live preview without "
            "clearing the detection results. Useful for comparing the "
            "detection to the raw image."
        )
        self.show_foci_check.toggled.connect(self.fociVisibilityChanged.emit)
        root.addWidget(self.show_foci_check)

        # Apply row — re-detection only fires when the user clicks
        # Apply (or when a new FOV / sample is selected). Detector
        # parameter typing alone does NOT trigger the live preview.
        apply_row = QHBoxLayout()
        apply_row.setSpacing(tokens.S3)
        self.apply_status = QLabel("")
        self.apply_status.setObjectName("muted")
        apply_row.addWidget(self.apply_status, stretch=1)
        self.apply_btn = QPushButton("Apply")
        self.apply_btn.setObjectName("primary")
        self.apply_btn.setToolTip(
            "Re-run foci detection on the current FOV using the values "
            "above. Detector / channel / parameter edits don't take "
            "effect on the live preview until you click here."
        )
        self.apply_btn.setEnabled(False)
        self.apply_btn.clicked.connect(self._on_apply_clicked)
        apply_row.addWidget(self.apply_btn)
        root.addLayout(apply_row)

        # ─── Inline foci filter (histograms + draggable thresholds) ───
        # Lives in the panel so users can iterate on detector + threshold
        # together: foci detected on the current preview FOV are pushed
        # in here by the live-preview controller, the histograms re-bin,
        # and threshold drags filter the scatter overlay without
        # re-detection (see thresholdsChanged signal).
        self.filter_box = QGroupBox("Filter foci (live, by feature)")
        filter_layout = QVBoxLayout(self.filter_box)
        filter_layout.setContentsMargins(
            tokens.S4, tokens.S5, tokens.S4, tokens.S4,
        )
        filter_layout.setSpacing(tokens.S3)
        self.filter_status = QLabel(
            "Open the Foci Detection live preview to populate."
        )
        self.filter_status.setObjectName("muted")
        self.filter_status.setWordWrap(True)
        filter_layout.addWidget(self.filter_status)
        hist_grid_host = QWidget()
        self._hist_grid = QGridLayout(hist_grid_host)
        self._hist_grid.setContentsMargins(0, 0, 0, 0)
        self._hist_grid.setSpacing(tokens.S2)
        filter_layout.addWidget(hist_grid_host)
        self._build_histograms()

        # Save / reset row at the bottom of the filter section.
        save_row = QHBoxLayout()
        save_row.setSpacing(tokens.S3)
        self.reset_filter_btn = QPushButton("Reset thresholds")
        self.reset_filter_btn.clicked.connect(self._on_reset_thresholds)
        save_row.addWidget(self.reset_filter_btn)
        save_row.addStretch(1)
        self.save_filter_btn = QPushButton("Save filter…")
        self.save_filter_btn.setIcon(icons.icon("check", role="text"))
        self.save_filter_btn.setEnabled(False)
        self.save_filter_btn.setToolTip(
            "Write the threshold set to 05_foci_filters/ and apply it to "
            "every parquet in 04c_foci_detection/, landing the filtered "
            "subsets at 04d_foci_filtered/. Disabled until you've run "
            "the Foci Detection stage at least once."
        )
        self.save_filter_btn.clicked.connect(self._on_save_filter)
        save_row.addWidget(self.save_filter_btn)
        filter_layout.addLayout(save_row)
        root.addWidget(self.filter_box)

        # "Label foci…" button — enabled by the host once 04c exists.
        actions_row = QHBoxLayout()
        actions_row.setSpacing(tokens.S3)
        self.label_button = QPushButton("Label foci…")
        self.label_button.setIcon(icons.icon("label", role="text"))
        self.label_button.setEnabled(False)
        self.label_button.setToolTip(
            "Open the Foci Label dialog. Disabled until you've run the "
            "Foci Detection stage at least once."
        )
        self.label_button.clicked.connect(self.labelRequested.emit)
        actions_row.addWidget(self.label_button)
        actions_row.addStretch(1)
        root.addLayout(actions_row)

        root.addStretch(1)

        # Seed with the same default as FociDetectionOpts — done BEFORE
        # wiring the dirty-tracking signals below so the initial
        # selection doesn't light up Apply on launch.
        self.detectors.set_selected_key("wavelet")

        # Detector-options changes are NOT auto-applied — re-detection is
        # expensive (BM3D / wavelet on a 1k×1k FOV is sluggish). Track a
        # "dirty" flag instead and let the user click Apply to commit.
        # Each option widget bumps the flag; the Apply button consumes it.
        self._dirty_opts = False
        self.detectors._list.itemSelectionChanged.connect(
            self._mark_opts_dirty
        )
        for w in (self.min_sigma, self.max_sigma, self.threshold, self.snr_min):
            w.valueChanged.connect(self._mark_opts_dirty)
        self.refine.toggled.connect(self._mark_opts_dirty)
        self.channels._list.itemSelectionChanged.connect(
            self._mark_opts_dirty
        )

    def _emit_options_changed(self, *_args) -> None:
        if not self._loading:
            self.optionsChanged.emit()

    def _mark_opts_dirty(self, *_args) -> None:
        """Flag that the user has edited a detector option without
        applying. The Apply button enables and a small caption appears.
        ``_loading`` blocks this during restore so initial widget
        population doesn't make the panel look dirty on launch.
        """
        if self._loading:
            return
        self._dirty_opts = True
        self.apply_btn.setEnabled(True)
        self.apply_status.setText("Unapplied changes")

    def _on_apply_clicked(self) -> None:
        """Commit the pending detector / channel / parameter edits to
        the live preview by emitting the usual ``optionsChanged`` signal.
        """
        self._dirty_opts = False
        self.apply_btn.setEnabled(False)
        self.apply_status.setText("")
        # Same fan-in as the old auto-emit path: optionsChanged is what
        # the live-preview controller listens to.
        self.optionsChanged.emit()

    # ── external wiring ─────────────────────────────────────────────
    def set_channels(self, names: list[str]) -> None:
        self._channel_names = list(names)
        self._refresh_channel_list()

    def set_phase_channel(self, phase: int | None) -> None:
        self._phase_channel = phase if isinstance(phase, int) else None
        self._refresh_channel_list()

    def _refresh_channel_list(self) -> None:
        exclude: list[int] = []
        if isinstance(self._phase_channel, int):
            exclude.append(self._phase_channel)
        # Populating the channel list emits itemSelectionChanged on
        # both clear() and the subsequent re-selection. That's a
        # programmatic event, not a user edit — guard the dirty flag
        # so the Apply button doesn't light up on app launch / CZI swap.
        prev_loading = self._loading
        self._loading = True
        try:
            self.channels.set_channels(
                self._channel_names, exclude_indices=exclude,
            )
        finally:
            self._loading = prev_loading

    def set_label_button_enabled(self, enabled: bool) -> None:
        """Toggle the Label-foci + Save-filter buttons. Called by
        MainWindow when the output directory's 04c_foci_detection/
        folder changes state — both actions consume the same parquets,
        so they enable / disable in lockstep.
        """
        self.label_button.setEnabled(bool(enabled))
        self.save_filter_btn.setEnabled(bool(enabled))

    def set_output_context(self, output_dir, run_id: str = "") -> None:
        """Receive the run's output dir + run-id from MainWindow so the
        Save filter action knows where to write."""
        self._output_dir = output_dir
        self._run_id = run_id or ""

    # ── Inline filter histograms ────────────────────────────────────

    def _build_histograms(self) -> None:
        """Lay out one small log-y histogram per filter feature in a
        2-column grid. Each plot owns a draggable InfiniteLine that
        becomes the per-feature minimum threshold.
        """
        # Try theme-aware background; fall back to a dark grey.
        try:
            bg = tokens.active().bg
        except Exception:  # noqa: BLE001
            bg = "#222"
        cols = 2
        for idx, (col, label, logy) in enumerate(FILTER_FEATURES):
            row, gcol = divmod(idx, cols)
            plot = pg.PlotWidget(title=label)
            plot.setMinimumHeight(110)
            plot.setBackground(bg)
            plot.showGrid(x=True, y=True, alpha=0.25)
            if logy:
                plot.setLogMode(x=False, y=True)
            # Empty bars to start — populated when set_foci_features fires.
            bars = pg.BarGraphItem(
                x=[], height=[], width=1.0,
                brush=pg.mkBrush(80, 140, 220, 220),
                pen=pg.mkPen((0, 0, 0, 0)),
            )
            plot.addItem(bars)
            line = pg.InfiniteLine(
                pos=0.0,
                angle=90,
                movable=True,
                pen=pg.mkPen((230, 70, 70, 230), width=2,
                             style=Qt.PenStyle.DashLine),
                hoverPen=pg.mkPen((255, 120, 120, 240), width=2),
            )
            line.sigPositionChanged.connect(self._on_threshold_changed)
            plot.addItem(line)
            self._hist_plots[col] = plot
            self._hist_bars[col] = bars
            self._hist_lines[col] = line
            self._hist_grid.addWidget(plot, row, gcol)

    def _on_threshold_changed(self, *_args) -> None:
        if self._loading:
            return
        self._refresh_filter_status()
        self.thresholdsChanged.emit()

    def _on_reset_thresholds(self) -> None:
        self._loading = True
        try:
            for col, line in self._hist_lines.items():
                df = self._foci_df
                if df is not None and col in df.columns and not df.empty:
                    vals = df[col].to_numpy(dtype=np.float64, copy=False)
                    finite = vals[np.isfinite(vals)]
                    line.setValue(float(finite.min()) if finite.size else 0.0)
                else:
                    line.setValue(0.0)
        finally:
            self._loading = False
        self._refresh_filter_status()
        self.thresholdsChanged.emit()

    def thresholds(self) -> dict[str, float]:
        """Current minimum threshold per feature."""
        return {
            col: float(line.value())
            for col, line in self._hist_lines.items()
        }

    def foci_visible(self) -> bool:
        """Whether the live-preview scatter overlay should be drawn.
        Toggled by the "Show foci on preview" checkbox."""
        return bool(self.show_foci_check.isChecked())

    def set_foci_features(self, df: "pd.DataFrame | None") -> None:
        """Push the live-detected foci's features DataFrame in. Re-bins
        the histograms; threshold positions are preserved across FOVs so
        the user's choices stay sticky while they navigate.
        """
        if df is not None and isinstance(df, pd.DataFrame) and not df.empty:
            self._foci_df = df
        else:
            self._foci_df = None
        if self._foci_df is None:
            self._clear_histograms()
            self.filter_status.setText(
                "No foci on the current FOV — adjust detector options "
                "or pick a different FOV in the live preview."
            )
            return
        n_bins = 40
        for col, _label, _logy in FILTER_FEATURES:
            bars = self._hist_bars.get(col)
            if bars is None:
                continue
            if col not in self._foci_df.columns:
                bars.setOpts(x=[], height=[], width=1.0)
                continue
            vals = self._foci_df[col].to_numpy(dtype=np.float64, copy=False)
            finite = vals[np.isfinite(vals)]
            if finite.size == 0:
                bars.setOpts(x=[], height=[], width=1.0)
                continue
            lo = float(finite.min())
            hi = float(finite.max())
            if hi <= lo:
                hi = lo + 1.0
            counts, edges = np.histogram(
                finite, bins=n_bins, range=(lo, hi),
            )
            centers = 0.5 * (edges[:-1] + edges[1:])
            widths = edges[1:] - edges[:-1]
            bars.setOpts(
                x=centers,
                height=counts.astype(np.float64),
                width=widths,
            )
            # Clamp the threshold line to within the current data range
            # (otherwise it can be off-screen after switching FOVs).
            line = self._hist_lines.get(col)
            if line is not None and line.value() < lo:
                line.setValue(lo)
        self._refresh_filter_status()

    def _clear_histograms(self) -> None:
        for bars in self._hist_bars.values():
            bars.setOpts(x=[], height=[], width=1.0)

    def _refresh_filter_status(self) -> None:
        df = self._foci_df
        if df is None or df.empty:
            self.filter_status.setText(
                "No foci on the current FOV — adjust detector options "
                "or pick a different FOV in the live preview."
            )
            return
        thr = self.thresholds()
        n_pass = int(compute_pass_mask(df, thr).sum())
        n_total = int(len(df))
        pct = (100.0 * n_pass / n_total) if n_total else 0.0
        self.filter_status.setText(
            f"{n_pass} / {n_total} foci pass this FOV ({pct:.1f}%)"
        )

    def _on_save_filter(self) -> None:
        if self._output_dir is None:
            QMessageBox.warning(
                self, "No output directory",
                "Select an output directory in the Input panel first.",
            )
            return
        try:
            result = save_foci_filter(
                output_dir=self._output_dir,
                run_id=self._run_id,
                thresholds=self.thresholds(),
            )
        except Exception as e:  # noqa: BLE001
            QMessageBox.warning(
                self, "Save failed",
                f"Could not write filter outputs:\n{e}",
            )
            return
        if result.n_input == 0:
            QMessageBox.information(
                self, "Filter saved (JSON only)",
                "Wrote the thresholds to "
                f"{result.json_path.name}.\n\n"
                "No 04c_foci_detection/ parquets to apply the filter "
                "to yet — run the Foci Detection stage to produce the "
                "filtered parquets under 04d_foci_filtered/.",
            )
        else:
            QMessageBox.information(
                self, "Filter saved",
                f"Wrote {result.n_pass:,} / {result.n_input:,} foci "
                f"across {len(result.filtered_parquets)} well(s).\n\n"
                f"Filter JSON: {result.json_path.name}\n"
                f"Filtered parquets: 04d_foci_filtered/",
            )

    # ── opts / persistence ──────────────────────────────────────────
    def opts(self) -> FociDetectionOpts:
        det_opts = DetectorOpts(
            min_sigma=float(self.min_sigma.value()),
            max_sigma=float(self.max_sigma.value()),
            threshold=float(self.threshold.value()),
            snr_min=float(self.snr_min.value()),
            refine=self.refine.isChecked(),
        )
        selected = self.channels.selected_indices()
        # Single-detector now; keep the dataclass shape as ``list[str]`` so
        # the parquet schema (one row per focus, with a ``detector`` column)
        # doesn't have to special-case the v1 single-key path.
        key = self.detectors.selected_key()
        detector_keys = [key] if key else []
        return FociDetectionOpts(
            detector_keys=detector_keys,
            detector_opts=det_opts,
            apply_to_channels=selected or None,
        )

    def state(self) -> dict:
        return {
            "detector_key": self.detectors.selected_key(),
            "min_sigma": float(self.min_sigma.value()),
            "max_sigma": float(self.max_sigma.value()),
            "threshold": float(self.threshold.value()),
            "snr_min": float(self.snr_min.value()),
            "refine": self.refine.isChecked(),
            "apply_to_channels": self.channels.selected_indices(),
            "show_foci": bool(self.show_foci_check.isChecked()),
        }

    def restore_state(self, s: dict) -> None:
        if not isinstance(s, dict):
            return
        self._loading = True
        try:
            # Accept both ``detector_key`` (new) and ``detector_keys``
            # (legacy QSettings from before the single-select switch).
            # Fall back to ``wavelet`` if the saved key is no longer in
            # the visible list (e.g. saved during the bundled-normaliser
            # era when ``decon_bm3d_wavelet`` was the default).
            saved_key: str | None = None
            if "detector_key" in s:
                saved_key = s.get("detector_key") or None
            elif "detector_keys" in s:
                keys = list(s["detector_keys"] or [])
                saved_key = keys[0] if keys else None
            if saved_key is not None and saved_key not in self.detectors._key_rows:
                saved_key = "wavelet"
            if saved_key is not None or "detector_key" in s or "detector_keys" in s:
                self.detectors.set_selected_key(saved_key)
            if "min_sigma" in s:
                self.min_sigma.setValue(float(s["min_sigma"]))
            if "max_sigma" in s:
                self.max_sigma.setValue(float(s["max_sigma"]))
            if "threshold" in s:
                self.threshold.setValue(float(s["threshold"]))
            if "snr_min" in s:
                self.snr_min.setValue(float(s["snr_min"]))
            if "refine" in s:
                self.refine.setChecked(bool(s["refine"]))
            if "apply_to_channels" in s:
                self.channels.set_selected_indices(
                    list(s["apply_to_channels"] or [])
                )
            if "show_foci" in s:
                self.show_foci_check.setChecked(bool(s["show_foci"]))
        finally:
            self._loading = False
