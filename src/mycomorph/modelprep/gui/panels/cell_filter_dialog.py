"""'Only show cells from…' — choose which genes / conditions / wells the
labeller may draw cells from.

One checkable list per metadata field, filled with the values actually present
in the HDF5 (with cell counts). Values ticked within a field are OR'd; fields
are AND'd; a field with nothing ticked allows any value. A live line shows how
many cells match and how many of those are still unlabelled.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ...core.cell_filter import FILTER_FIELDS, CellFilter, distinct_values
from ..ui import tokens

BLANK = "(blank)"
_SEARCH_MIN_VALUES = 8  # show a search box above lists longer than this


class CellFilterDialog(QDialog):
    """Edit a :class:`CellFilter` against one open HDF5."""

    def __init__(
        self,
        reader,
        labelled_uids: set[str],
        current: Optional[CellFilter] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Only show cells from…")
        self.setModal(True)
        self._reader = reader
        self._unlabelled = np.fromiter(
            (uid not in labelled_uids for uid in reader.cell_uids),
            dtype=bool, count=len(reader),
        )
        self._lists: dict[str, QListWidget] = {}
        self._boxes: dict[str, tuple[QGroupBox, str]] = {}  # key → (box, base title)
        self._build_ui(current or CellFilter())
        self._update_summary()

    # ── public ─────────────────────────────────────────────────────────

    def cell_filter(self) -> CellFilter:
        chosen = {}
        for key, lst in self._lists.items():
            chosen[key] = [lst.item(i).data(Qt.ItemDataRole.UserRole)
                           for i in range(lst.count())
                           if lst.item(i).checkState() == Qt.CheckState.Checked]
        return CellFilter.from_mapping(chosen)

    # ── UI ─────────────────────────────────────────────────────────────

    def _build_ui(self, current: CellFilter) -> None:
        v = QVBoxLayout(self)
        v.setSpacing(8)

        intro = QLabel(
            "Label only cells with the ticked values. Within a field, any ticked "
            "value matches; across fields, every field must match. A field with "
            "nothing ticked allows any value. Random sampling stays seeded, and the "
            "filter is recorded with the labelling session."
        )
        intro.setWordWrap(True)
        intro.setObjectName("muted")
        v.addWidget(intro)

        grid_host = QWidget()
        grid = QGridLayout(grid_host)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(8)
        constant: list[str] = []
        shown = 0
        for key, (dataset, name) in FILTER_FIELDS.items():
            pairs = distinct_values(self._reader, key)
            selected = set(current.values.get(key, ()))
            # A field with one value can't narrow anything — summarise it instead
            # (unless an existing filter already restricts it).
            if len(pairs) < 2 and not selected:
                if pairs:
                    constant.append(f"{name.lower()} = {pairs[0][0] or BLANK}")
                continue
            grid.addWidget(self._field_box(key, name, dataset, pairs, selected),
                           shown // 3, shown % 3)
            shown += 1
        v.addWidget(grid_host, stretch=1)

        if not shown:
            none = QLabel("Every cell in this file has the same gene, condition, "
                          "reporter, drug and TIFF — there is nothing to filter on.")
            none.setWordWrap(True)
            v.addWidget(none)
        if constant:
            note = QLabel("Same for every cell: " + " · ".join(constant))
            note.setWordWrap(True)
            note.setObjectName("muted")
            v.addWidget(note)

        self._summary = QLabel("")
        self._summary.setWordWrap(True)
        v.addWidget(self._summary)

        buttons = QHBoxLayout()
        all_btn = QPushButton("All cells")
        all_btn.setObjectName("ghost")
        all_btn.setToolTip("Untick everything (no filter).")
        all_btn.clicked.connect(self._clear_all)
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.reject)
        self._apply = QPushButton("Apply")
        self._apply.setObjectName("primary")
        self._apply.setDefault(True)
        self._apply.clicked.connect(self.accept)
        buttons.addWidget(all_btn)
        buttons.addStretch(1)
        buttons.addWidget(cancel)
        buttons.addWidget(self._apply)
        v.addLayout(buttons)

        self.resize(860, 520)

    def _field_box(self, key, name, dataset, pairs, selected) -> QGroupBox:
        title = f"{name} ({len(pairs)})"
        box = QGroupBox(title)
        self._boxes[key] = (box, title)
        box.setToolTip(f"HDF5 dataset '{dataset}'")
        bv = QVBoxLayout(box)
        bv.setContentsMargins(8, 14, 8, 8)
        bv.setSpacing(4)

        lst = QListWidget()
        lst.setMinimumHeight(160)
        for value, n in pairs:
            item = QListWidgetItem(f"{value or BLANK}   ({n:,})")
            item.setData(Qt.ItemDataRole.UserRole, value)
            item.setToolTip(value or "empty string")
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked if value in selected
                               else Qt.CheckState.Unchecked)
            lst.addItem(item)
        lst.itemChanged.connect(self._update_summary)
        self._lists[key] = lst

        if len(pairs) > _SEARCH_MIN_VALUES:
            search = QLineEdit()
            search.setPlaceholderText("Search…")
            search.setClearButtonEnabled(True)
            search.textChanged.connect(lambda text, l=lst: self._search(l, text))
            bv.addWidget(search)
        bv.addWidget(lst, stretch=1)

        clear = QPushButton("Clear")
        clear.setObjectName("ghost")
        clear.clicked.connect(lambda _=False, l=lst: self._set_all(l, False))
        bv.addWidget(clear, alignment=Qt.AlignmentFlag.AlignRight)
        return box

    def showEvent(self, event) -> None:
        super().showEvent(event)
        # Bring each list's first ticked value into view (long lists hide it
        # otherwise). Needs the lists laid out, hence here rather than in init.
        for lst in self._lists.values():
            first = next((lst.item(i) for i in range(lst.count())
                          if lst.item(i).checkState() == Qt.CheckState.Checked), None)
            if first is not None:
                lst.scrollToItem(first, QListWidget.ScrollHint.PositionAtTop)

    # ── behaviour ──────────────────────────────────────────────────────

    @staticmethod
    def _search(lst: QListWidget, text: str) -> None:
        needle = text.strip().lower()
        for i in range(lst.count()):
            item = lst.item(i)
            value = item.data(Qt.ItemDataRole.UserRole) or BLANK
            item.setHidden(bool(needle) and needle not in value.lower())

    def _set_all(self, lst: QListWidget, checked: bool) -> None:
        state = Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked
        lst.blockSignals(True)
        for i in range(lst.count()):
            lst.item(i).setCheckState(state)
        lst.blockSignals(False)
        self._update_summary()

    def _clear_all(self) -> None:
        for lst in self._lists.values():
            self._set_all(lst, False)

    def _update_summary(self, *_args) -> None:
        f = self.cell_filter()
        for key, (box, title) in self._boxes.items():
            n = len(f.values.get(key, ()))
            box.setTitle(f"{title} · {n} ticked" if n else title)
        mask = f.mask(self._reader)
        n_match = int(mask.sum())
        n_left = int((mask & self._unlabelled).sum())
        p = tokens.active()
        if n_match == 0:
            text = "No cells match — untick something, or clear a field."
            color = p.warning
        else:
            what = ("All cells" if f.is_empty()
                    else f"{n_match:,} of {len(self._reader):,} cells match")
            if f.is_empty():
                what += f": {n_match:,}"
            text = f"{what} · {n_left:,} still unlabelled"
            if not f.is_empty():
                text += f"   ({f.describe()})"
            color = p.text
        self._summary.setText(text)
        self._summary.setStyleSheet(f"color: {color}; font-weight: 600;")
        self._apply.setEnabled(n_match > 0)
