"""``mycomorph-cli label / review / train`` moved to ModelPrep (decision D8):
each now prints where it went and exits 2, whatever arguments it gets."""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from mycomorph.core.cli import _MOVED_TO_MODELPREP, app

runner = CliRunner()


@pytest.mark.parametrize("args", [
    ["label", "--input", "output_tiffs/", "--save-dir", "labeled_data/"],
    ["review", "-i", "output_tiffs/", "-m", "models/best_model.pth"],
    ["train", "--data", "labeled_data/", "--output", "models/", "--gpu"],
    ["train", "--help"],
    ["label"],
])
def test_old_command_points_to_modelprep(args):
    command = args[0]
    res = runner.invoke(app, args)
    assert res.exit_code == 2
    # .output holds what the user sees, stdout and stderr alike, on every
    # click version (.stderr is only captured separately from click 8.2).
    assert f"`mycomorph-cli {command}` has moved to ModelPrep" in res.output
    assert f"modelprep-cli {command} --h5" in res.output
    assert "all_crops.h5" in res.output


def test_cli_help_lists_them_as_moved():
    res = runner.invoke(app, ["--help"])
    assert res.exit_code == 0
    for command in ("label", "review", "train"):
        assert f"`modelprep-cli {command}`" in res.output


@pytest.mark.parametrize("command", sorted(_MOVED_TO_MODELPREP))
def test_suggested_modelprep_command_exists(command):
    """Keep the pointers true: every option they suggest is a real one."""
    from mycomorph.modelprep.core.cli import app as modelprep_app

    res = runner.invoke(modelprep_app, [command, "--help"])
    assert res.exit_code == 0, res.output
    suggested = _MOVED_TO_MODELPREP[command].split()
    assert suggested[:2] == ["modelprep-cli", command]
    for option in (word for word in suggested if word.startswith("--")):
        assert option in res.output
