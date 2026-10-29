"""Every per-user ModelPrep path goes through ``modelprep_home()``."""

from __future__ import annotations

from pathlib import Path


def test_modelprep_home_is_dot_modelprep_under_home(isolated_home):
    from mycomorph.modelprep.core.paths import modelprep_home

    # Kept as ~/.modelprep after the move into MycoMorph (plan D2).
    assert modelprep_home() == isolated_home / ".modelprep"


def test_user_folders_live_under_modelprep_home(isolated_home):
    from mycomorph.modelprep.core import cli
    from mycomorph.modelprep.core.paths import modelprep_home
    from mycomorph.modelprep.core.score_cache import score_cache_root
    from mycomorph.modelprep.gui.main_window import settings_file
    from mycomorph.modelprep.gui.trained_models import trained_models_root

    home = modelprep_home()
    assert trained_models_root() == home / "models"
    assert score_cache_root() == home / "score_cache"
    assert settings_file() == home / "settings.json"
    assert cli._default_output_dir().parent == home / "models"


def test_no_other_module_builds_the_folder_name():
    import mycomorph.modelprep as pkg

    root = Path(pkg.__file__).parent
    offenders = [
        str(p.relative_to(root))
        for p in root.rglob("*.py")
        if p.name != "paths.py" and '".modelprep"' in p.read_text(encoding="utf-8")
    ]
    assert offenders == [], f"build ~/.modelprep paths via modelprep_home(): {offenders}"
