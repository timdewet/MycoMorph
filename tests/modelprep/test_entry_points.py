"""ModelPrep keeps its ``modelprep`` / ``modelprep-cli`` commands inside
MycoMorph (decision D1). Checks the installed console-script metadata, so it
needs a (re)install after ``[project.scripts]`` changes."""

from __future__ import annotations

from importlib.metadata import entry_points

import pytest

EXPECTED = {
    "modelprep": "mycomorph.modelprep.gui.app:main",
    "modelprep-cli": "mycomorph.modelprep.core.cli:app",
}


def _console_script(name: str):
    matches = [ep for ep in entry_points(group="console_scripts") if ep.name == name]
    assert matches, f"console script {name!r} is not installed; reinstall MycoMorph"
    return matches[0]


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_console_script_resolves(name):
    ep = _console_script(name)
    assert ep.value == EXPECTED[name]
    assert callable(ep.load())


def test_modelprep_cli_runs():
    from typer.testing import CliRunner

    app = _console_script("modelprep-cli").load()
    res = CliRunner().invoke(app, ["--help"])
    assert res.exit_code == 0, res.output
    assert "train" in res.output


def test_python_m_modelprep_uses_the_gui_entry_point():
    import mycomorph.modelprep.__main__ as mp_main
    from mycomorph.modelprep.gui.app import main

    assert mp_main.main is main
