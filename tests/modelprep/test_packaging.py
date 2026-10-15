"""Desktop-bundle inputs (review L1). Checks the repository files the
PyInstaller spec uses, not the installed package."""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def test_bundle_icons_exist():
    logo = REPO / "src" / "modelprep" / "gui" / "assets" / "logo"
    assert (logo / "logo.icns").read_bytes()[:4] == b"icns"
    assert (logo / "logo.ico").read_bytes()[:4] == b"\x00\x00\x01\x00"


def test_spec_builds_a_macos_app_with_licence_notice():
    spec = (REPO / "packaging" / "modelprep.spec").read_text(encoding="utf-8")
    assert "BUNDLE(" in spec and "logo.icns" in spec
    assert (REPO / "packaging" / "BUNDLE_LICENSE.txt").is_file()
    assert "BUNDLE_LICENSE.txt" in spec and "labelling_rubric.md" in spec
