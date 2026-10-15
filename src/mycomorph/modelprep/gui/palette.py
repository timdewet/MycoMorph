"""Shared colour palette for ModelPrep GUI widgets.

Thin, theme-aware shim over the design tokens in :mod:`mycomorph.gui.ui.tokens`.
``COLORS`` and ``LABEL_COLORS`` behave like the dicts they replaced, but each
lookup resolves against the *currently active* palette, so widgets that read
them at paint time (cell thumbnails, the detail dialog) automatically track
light/dark theme switches. Lives in its own module so panels and dialogs can
share it without import cycles.
"""

from __future__ import annotations

from mycomorph.gui.ui import tokens


class _ActiveColors:
    """Maps the legacy ModelPrep colour keys onto live theme tokens."""

    def __getitem__(self, key: str) -> str:
        p = tokens.active()
        return {
            "border_strong": p.border_strong,
            "success":       p.success,
            "danger":        p.danger,
            "primary":       p.primary,
            "surface":       p.surface,
            "muted_text":    p.text_muted,
        }[key]


class _ActiveLabelColors:
    """Ring colour per label state, resolved against the active palette."""

    _KEYS = ("unlabeled", "good", "bad")

    def __getitem__(self, key: str) -> str:
        p = tokens.active()
        return {
            "unlabeled": p.border_strong,
            "good":      p.success,
            "bad":       p.danger,
        }[key]

    def __contains__(self, key: str) -> bool:
        return key in self._KEYS


COLORS = _ActiveColors()
LABEL_COLORS = _ActiveLabelColors()
