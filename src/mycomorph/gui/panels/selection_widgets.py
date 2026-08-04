"""Reusable channel and detector selectors for imaging stage panels."""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QListWidget, QListWidgetItem, QVBoxLayout, QWidget

from mycomorph.core.foci.detectors import (
    BACTERIAL_SPECIFIC_KEYS,
    CLASSICAL_BASELINE_KEYS,
    DIM_SIGNAL_KEYS,
)


class ChannelIndexSelect(QWidget):
    """Multi-select fluorescence channels while preserving source indices."""

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
        self._index_at_row: list[int] = []
        self._pending_selection: list[int] | None = None

    def set_channels(
        self,
        names: list[str],
        exclude_indices: list[int] | None = None,
    ) -> None:
        self._channels = list(names)
        self._exclude = set(exclude_indices or [])
        previous = self.selected_indices() or self._pending_selection
        self._list.clear()
        self._index_at_row = []
        for index, name in enumerate(self._channels):
            if index in self._exclude:
                continue
            item = QListWidgetItem(f"{index}: {name}")
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsSelectable)
            self._list.addItem(item)
            self._index_at_row.append(index)
        if previous:
            self.set_selected_indices(list(previous))
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
        for original_index in indices:
            try:
                row = self._index_at_row.index(int(original_index))
            except ValueError:
                continue
            self._list.item(row).setSelected(True)


class DetectorKeySelect(QWidget):
    """Single-select a detector registry key, grouped by detector family."""

    _GROUPS: list[tuple[str, list[str]]] = [
        ("Classical baselines", CLASSICAL_BASELINE_KEYS),
        ("Dim signal", DIM_SIGNAL_KEYS),
        ("Bacterial-specific", BACTERIAL_SPECIFIC_KEYS),
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
            header_item = QListWidgetItem(f"— {header} —")
            header_item.setFlags(Qt.ItemFlag.NoItemFlags)
            header_item.setForeground(Qt.GlobalColor.gray)
            self._list.addItem(header_item)
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
