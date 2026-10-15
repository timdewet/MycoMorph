"""``MycoMorph --modelprep`` opens ModelPrep's window on its own (decision D4):
the single desktop bundle launches the trainer this way."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from mycomorph.gui.app import _parse_args


def test_modelprep_flag_is_ours_and_qt_args_pass_through():
    args, qt_argv = _parse_args(["MycoMorph", "--modelprep", "-platform", "offscreen"])
    assert args.modelprep
    assert qt_argv == ["MycoMorph", "-platform", "offscreen"]

    args, qt_argv = _parse_args(["MycoMorph", "-psn_0_12345"])   # macOS Finder launch
    assert not args.modelprep
    assert qt_argv == ["MycoMorph", "-psn_0_12345"]

    # No prefix matching: an unrelated flag never opens ModelPrep.
    args, qt_argv = _parse_args(["MycoMorph", "--model"])
    assert not args.modelprep
    assert qt_argv == ["MycoMorph", "--model"]


def _smoke_launch(module: str, *flags: str, home: Path) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env.update(
        MYCOMORPH_SMOKE_TEST="1",
        QT_QPA_PLATFORM="offscreen",
        HOME=str(home),                          # never the user's ~/.modelprep
        USERPROFILE=str(home),
        XDG_CONFIG_HOME=str(home / ".config"),
        # MycoMorph's window embeds QtWebEngine; its sandbox can't start for
        # an unprivileged user on Ubuntu 24.04 CI runners.
        QTWEBENGINE_DISABLE_SANDBOX="1",
    )
    return subprocess.run(
        [sys.executable, "-m", module, *flags],
        env=env, capture_output=True, text=True, timeout=180,
    )


@pytest.mark.parametrize("module", ["mycomorph", "mycomorph.gui"])
def test_modelprep_flag_smoke_launch(module, tmp_path):
    """The bundle's entry script (``mycomorph/__main__.py``) and ``python -m
    mycomorph.gui`` open ModelPrep's window, honour MYCOMORPH_SMOKE_TEST and
    exit cleanly."""
    res = _smoke_launch(module, "--modelprep", home=tmp_path)
    assert res.returncode == 0, res.stderr[-2000:]
    assert "[mycomorph] smoke test: opened ModelPrep" in res.stdout


def test_plain_launch_still_opens_mycomorph(tmp_path):
    res = _smoke_launch("mycomorph", home=tmp_path)
    assert res.returncode == 0, res.stderr[-2000:]
    assert "[mycomorph] smoke test: opened MycoMorph" in res.stdout
