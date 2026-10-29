"""Desktop-bundle inputs. Checks the repository files that
``packaging/mycomorph.spec`` uses, not a built bundle (CI builds and
smoke-launches the Windows and macOS bundles on pull requests)."""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PACKAGING = REPO / "packaging"
SPEC = (PACKAGING / "mycomorph.spec").read_text(encoding="utf-8")


def test_bundle_icons_exist():
    for logo in (REPO / "assets" / "logo",
                 REPO / "src" / "mycomorph" / "modelprep" / "gui" / "assets" / "logo"):
        assert (logo / "logo.icns").read_bytes()[:4] == b"icns"
        assert (logo / "logo.ico").read_bytes()[:4] == b"\x00\x00\x01\x00"


def test_spec_builds_a_macos_app_with_licence_notice():
    assert "BUNDLE(" in SPEC and "logo.icns" in SPEC
    assert (PACKAGING / "BUNDLE_LICENSE.txt").is_file()
    assert "BUNDLE_LICENSE.txt" in SPEC and "labelling_rubric.md" in SPEC


def test_spec_data_files_exist():
    """Every repository file the spec bundles is where the spec says."""
    sources = re.findall(r'datas\.append\(\(\s*"([^"]+)"', SPEC)
    assert "../docs/modelprep/labelling_rubric.md" in sources
    for src in sources:
        assert (PACKAGING / src).is_file(), src


def test_licence_notice_names_the_files_the_spec_bundles():
    notice = (PACKAGING / "BUNDLE_LICENSE.txt").read_text(encoding="utf-8")
    for dest in re.findall(r'_licence_file\("[^"]+"\), "([^"]+)"\)', SPEC):
        assert f"{dest}/LICENSE" in notice
    assert "GPL v3" in notice and "MIT" in notice


def test_make_icons_covers_both_logos():
    sys.path.insert(0, str(PACKAGING))
    try:
        import make_icons
    finally:
        sys.path.remove(str(PACKAGING))
    assert set(make_icons.LOGO_DIRS) == {"mycomorph", "modelprep"}
    for logo_dir in make_icons.LOGO_DIRS.values():
        assert (logo_dir / "logo.svg").is_file()


def test_modelprep_resources_in_a_frozen_bundle(tmp_path, monkeypatch):
    """In the bundle, ModelPrep's logo comes from its package data (the bundle
    root holds MycoMorph's logo) and the rubric from docs/modelprep/."""
    from mycomorph.modelprep.gui import _resources

    rubric = tmp_path / "docs" / "modelprep" / "labelling_rubric.md"
    rubric.parent.mkdir(parents=True)
    rubric.write_text("# rubric", encoding="utf-8")
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path), raising=False)

    assert _resources.resource_root() == Path(_resources.__file__).resolve().parent / "assets"
    assert (_resources.resource_root() / "logo" / "logo.svg").is_file()
    assert _resources.rubric_location() == str(rubric)
