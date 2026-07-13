"""Cross-platform, deterministic names for user-labelled pipeline outputs."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from typing import Any

_INVALID_COMPONENT = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_WINDOWS_RESERVED = {
    "CON", "PRN", "AUX", "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}
_MAX_COMPONENT_LENGTH = 80
_MAX_STEM_LENGTH = 220


def safe_filename_component(value: Any) -> tuple[str, bool]:
    """Return ``(safe_value, changed)`` without altering already-safe labels."""
    original = str(value).strip()
    normalised = unicodedata.normalize("NFKC", original)
    safe = normalised.replace(" ", "_")
    safe = _INVALID_COMPONENT.sub("_", safe).rstrip(". ")
    if safe in {"", ".", ".."}:
        safe = "_"
    if safe.upper() in _WINDOWS_RESERVED:
        safe = f"_{safe}"
    if len(safe) > _MAX_COMPONENT_LENGTH:
        safe = safe[:_MAX_COMPONENT_LENGTH].rstrip(". ")
    return safe, safe != original.replace(" ", "_")


def build_output_filename(
    condition: Any,
    reporter: Any,
    mutant_or_drug: Any,
    replica: Any = "",
    *,
    stable_identifier: Any | None = None,
) -> str:
    """Build a safe, deterministic condition filename.

    Existing labels containing only portable filename characters retain their
    historical names. If sanitisation changes any component, a short digest of
    the original labels is appended so distinct labels cannot collapse onto the
    same filename after sanitisation.
    """
    original = [str(condition), str(reporter), str(mutant_or_drug)]
    if str(replica).strip():
        original.append(f"R{str(replica).strip()}")

    safe_parts: list[str] = []
    changed = False
    for part in original:
        safe, part_changed = safe_filename_component(part)
        safe_parts.append(safe)
        changed = changed or part_changed

    stem = "__".join(safe_parts)
    if changed:
        payload = json.dumps(original, ensure_ascii=False, separators=(",", ":"))
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:8]
        stem = f"{stem}__H{digest}"
    if stable_identifier is not None:
        raw_identifier = str(stable_identifier)
        identifier, identifier_changed = safe_filename_component(raw_identifier)
        if identifier_changed or len(identifier) > 32:
            suffix = hashlib.sha256(raw_identifier.encode("utf-8")).hexdigest()[:8]
            identifier = f"{identifier[:24]}_{suffix}"
        stem = f"{stem}__W{identifier}"
    if stem.upper() in _WINDOWS_RESERVED:
        stem = f"_{stem}"
    if len(stem) > _MAX_STEM_LENGTH:
        digest = hashlib.sha256(stem.encode("utf-8")).hexdigest()[:12]
        stem = f"{stem[:_MAX_STEM_LENGTH - 15]}__H{digest}"
    return f"{stem}.tif"
