"""Software versions recorded in run outputs.

Methods sections and validation results need to say which MycoMorph version
produced an output. These tests cover the single package version, the
``software`` block written beside every provenance signature, and commit
detection for runs from a source checkout.
"""

from __future__ import annotations

import json
from pathlib import Path

import mycomorph
from mycomorph.core import provenance
from mycomorph.core.provenance import (
    SIGNATURE_FILE,
    artifact_signature_matches,
    artifact_signature_path,
    checkpoint_signature,
    prepare_checkpoint_dir,
    software_info,
    source_commit,
    write_artifact_signature,
)

SHA_A = "a" * 40
SHA_B = "b" * 40


def test_subpackages_report_the_package_version():
    import mycomorph.core
    import mycomorph.core.focus

    assert mycomorph.core.__version__ == mycomorph.__version__
    assert mycomorph.core.focus.__version__ == mycomorph.__version__


def test_software_info_reports_package_version_and_returns_a_copy():
    info = software_info()
    assert info["mycomorph"] == mycomorph.__version__
    assert "python" in info and "frozen_app" in info
    info["mycomorph"] = "tampered"
    assert software_info()["mycomorph"] == mycomorph.__version__


def _signature(tmp_path: Path) -> dict:
    source = tmp_path / "input.tif"
    source.write_bytes(b"pixels")
    return checkpoint_signature(stage="segment", input_path=source, options={"a": 1})


def test_artifact_sidecar_records_software_without_changing_reuse(tmp_path: Path):
    signature = _signature(tmp_path)
    before = dict(signature)
    output = tmp_path / "out.tif"

    write_artifact_signature(output, signature)

    assert signature == before, "the caller's signature must not be mutated"
    written = json.loads(artifact_signature_path(output).read_text())
    assert written["software"]["mycomorph"] == mycomorph.__version__
    assert written["digest"] == signature["digest"]
    assert artifact_signature_matches(output, signature) is True


def test_checkpoint_signature_file_records_software_and_stays_reusable(tmp_path: Path):
    signature = _signature(tmp_path)
    partial = tmp_path / "partial"

    assert prepare_checkpoint_dir(partial, signature) is False
    written = json.loads((partial / SIGNATURE_FILE).read_text())
    assert written["software"]["mycomorph"] == mycomorph.__version__
    assert prepare_checkpoint_dir(partial, signature) is True


def test_sidecars_written_by_an_older_version_stay_reusable(tmp_path: Path):
    signature = _signature(tmp_path)
    output = tmp_path / "out.tif"
    write_artifact_signature(output, signature)
    sidecar = artifact_signature_path(output)
    written = json.loads(sidecar.read_text())
    written["software"]["mycomorph"] = "0.0.1"
    sidecar.write_text(json.dumps(written))

    assert artifact_signature_matches(output, signature) is True


# ── source_commit ─────────────────────────────────────────────────────


def _repo(tmp_path: Path) -> tuple[Path, Path]:
    repo = tmp_path / "repo"
    package = repo / "src" / "mycomorph"
    package.mkdir(parents=True)
    (repo / "pyproject.toml").write_text("[project]\nname = 'mycomorph'\n")
    return repo, package


def test_source_commit_reads_branch_ref(tmp_path: Path):
    repo, package = _repo(tmp_path)
    git = repo / ".git"
    (git / "refs" / "heads").mkdir(parents=True)
    (git / "HEAD").write_text("ref: refs/heads/main\n")
    (git / "refs" / "heads" / "main").write_text(SHA_A + "\n")

    assert source_commit(package) == SHA_A


def test_source_commit_reads_detached_head(tmp_path: Path):
    repo, package = _repo(tmp_path)
    (repo / ".git").mkdir()
    (repo / ".git" / "HEAD").write_text(SHA_B + "\n")

    assert source_commit(package) == SHA_B


def test_source_commit_follows_worktree_to_packed_refs(tmp_path: Path):
    main_git = tmp_path / "main" / ".git"
    worktree_git = main_git / "worktrees" / "wt"
    worktree_git.mkdir(parents=True)
    (worktree_git / "HEAD").write_text("ref: refs/heads/feature\n")
    (worktree_git / "commondir").write_text("../..\n")
    (main_git / "packed-refs").write_text(
        "# pack-refs with: peeled fully-peeled sorted\n"
        f"{SHA_A} refs/heads/feature\n"
        f"^{SHA_B}\n"
    )
    repo, package = _repo(tmp_path)
    (repo / ".git").write_text(f"gitdir: {worktree_git}\n")

    assert source_commit(package) == SHA_A


def test_source_commit_ignores_installs_outside_a_source_layout(tmp_path: Path):
    # e.g. site-packages/mycomorph inside some other project's git repo
    project = tmp_path / "project"
    package = project / ".venv" / "lib" / "site-packages" / "mycomorph"
    package.mkdir(parents=True)
    (project / ".git").mkdir()
    (project / ".git" / "HEAD").write_text(SHA_A + "\n")

    assert source_commit(package) is None


def test_source_commit_without_git_is_none(tmp_path: Path):
    _repo_dir, package = _repo(tmp_path)
    assert source_commit(package) is None


def test_software_info_omits_commit_in_frozen_app(monkeypatch):
    import sys

    provenance._software_info_cached.cache_clear()
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    try:
        info = software_info()
        assert info["frozen_app"] is True
        assert "mycomorph_git_commit" not in info
    finally:
        monkeypatch.undo()
        provenance._software_info_cached.cache_clear()
