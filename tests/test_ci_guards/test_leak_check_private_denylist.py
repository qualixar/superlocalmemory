# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""scripts/check-no-internal-leaks.sh honours an optional private denylist."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "check-no-internal-leaks.sh"

pytestmark = pytest.mark.skipif(
    shutil.which("bash") is None or shutil.which("git") is None,
    reason="needs bash and git",
)


def _repo(tmp_path: Path, body: str) -> Path:
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    shutil.copy(SCRIPT, repo / "scripts" / SCRIPT.name)
    (repo / "notes.txt").write_text(body, encoding="utf-8")
    env = {**os.environ, "GIT_CONFIG_GLOBAL": os.devnull}
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True, env=env)
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True, env=env)
    return repo


def _run(repo: Path, denylist: Path | None) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if k != "SLM_LEAK_DENYLIST"}
    if denylist is not None:
        env["SLM_LEAK_DENYLIST"] = str(denylist)
    return subprocess.run(
        ["bash", str(repo / "scripts" / SCRIPT.name)],
        cwd=repo, env=env, capture_output=True, text=True,
    )


def test_denylist_pattern_in_tracked_file_fails(tmp_path):
    repo = _repo(tmp_path, "we shipped for Examplecorp last week\n")
    deny = tmp_path / "deny.txt"
    deny.write_text("# private names\n\nexample(corp|inc)\n", encoding="utf-8")
    result = _run(repo, deny)
    assert result.returncode == 1
    assert "private denylist" in result.stdout


def test_denylist_without_a_match_passes(tmp_path):
    repo = _repo(tmp_path, "nothing sensitive here\n")
    deny = tmp_path / "deny.txt"
    deny.write_text("examplecorp\n", encoding="utf-8")
    assert _run(repo, deny).returncode == 0


def test_without_denylist_behaviour_is_unchanged(tmp_path):
    repo = _repo(tmp_path, "we shipped for Examplecorp last week\n")
    assert _run(repo, None).returncode == 0
    missing = tmp_path / "does-not-exist.txt"
    assert _run(repo, missing).returncode == 0


def test_built_in_patterns_still_fire(tmp_path):
    repo = _repo(tmp_path, "the " + "Iron" + " Pattern lives here\n")
    assert _run(repo, None).returncode == 1
