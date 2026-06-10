"""A wrapping flow layout (Qt's canonical FlowLayout, PyQt6 port).

Lays children left-to-right and wraps to the next line when it runs out of
width — so a long row of toolbar controls reflows onto multiple rows instead of
forcing a large minimum width on its window. ``minimumSize`` is the largest
single child, which is what lets the containing window narrow freely.
"""

from __future__ import annotations

from PyQt6.QtCore import QPoint, QRect, QSize, Qt
from PyQt6.QtWidgets import QLayout, QSizePolicy, QWidget


class FlowLayout(QLayout):
    def __init__(self, parent: QWidget | None = None,
                 margin: int = 0, hspacing: int = 6, vspacing: int = 6) -> None:
        super().__init__(parent)
        self._items: list = []
        self._hspace = hspacing
        self._vspace = vspacing
        self.setContentsMargins(margin, margin, margin, margin)

    # ── QLayout plumbing ────────────────────────────────────────────────
    def addItem(self, item) -> None:  # noqa: N802 (Qt API)
        self._items.append(item)

    def count(self) -> int:
        return len(self._items)

    def itemAt(self, index: int):  # noqa: N802
        if 0 <= index < len(self._items):
            return self._items[index]
        return None

    def takeAt(self, index: int):  # noqa: N802
        if 0 <= index < len(self._items):
            return self._items.pop(index)
        return None

    def expandingDirections(self):  # noqa: N802
        return Qt.Orientation(0)

    def hasHeightForWidth(self) -> bool:  # noqa: N802
        return True

    def heightForWidth(self, width: int) -> int:  # noqa: N802
        return self._do_layout(QRect(0, 0, width, 0), test_only=True)

    def setGeometry(self, rect: QRect) -> None:  # noqa: N802
        super().setGeometry(rect)
        self._do_layout(rect, test_only=False)

    def sizeHint(self) -> QSize:  # noqa: N802
        return self.minimumSize()

    def minimumSize(self) -> QSize:  # noqa: N802
        size = QSize()
        for item in self._items:
            size = size.expandedTo(item.minimumSize())
        margins = self.contentsMargins()
        size += QSize(margins.left() + margins.right(),
                      margins.top() + margins.bottom())
        return size

    # ── core ────────────────────────────────────────────────────────────
    def _do_layout(self, rect: QRect, test_only: bool) -> int:
        margins = self.contentsMargins()
        eff = rect.adjusted(margins.left(), margins.top(),
                            -margins.right(), -margins.bottom())
        x, y = eff.x(), eff.y()
        line_height = 0
        for item in self._items:
            hint = item.sizeHint()
            w, h = hint.width(), hint.height()
            next_x = x + w
            if next_x - eff.x() > eff.width() and line_height > 0:
                x = eff.x()
                y = y + line_height + self._vspace
                next_x = x + w
                line_height = 0
            if not test_only:
                item.setGeometry(QRect(QPoint(x, y), hint))
            x = next_x + self._hspace
            line_height = max(line_height, h)
        return y + line_height - rect.y() + margins.bottom()
