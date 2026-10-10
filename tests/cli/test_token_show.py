# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""``slm token show``: the way to read the key the dashboard asks for.

With SLM_REQUIRE_CREDENTIALS=1 the dashboard no longer receives the install
token on its own; it asks the person to paste it. This command prints it.
"""

from __future__ import annotations

import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from tests._portable import child_env_base

REPO = Path(__file__).resolve().parents[2]


def _slm(tmp_path: Path, *argv: str) -> subprocess.CompletedProcess:
    env = {**os.environ, "PYTHONPATH": str(REPO / "src"), "SLM_DATA_DIR": str(tmp_path),
           **child_env_base(tmp_path / "home"), "SLM_SKIP_FIRST_USE": "1"}
    return subprocess.run([sys.executable, "-m", "superlocalmemory.cli.main", *argv],
                          env=env, capture_output=True, text=True, timeout=120)


def test_token_show_prints_the_install_token(tmp_path):
    (tmp_path / ".install_token").write_text("abc123tokenvalue\n", encoding="utf-8")

    r = _slm(tmp_path, "token", "show")

    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "abc123tokenvalue"


@pytest.mark.skipif(os.name == "nt", reason="POSIX file modes")
def test_token_show_tightens_a_readable_token_file(tmp_path):
    token = tmp_path / ".install_token"
    token.write_text("abc123tokenvalue\n", encoding="utf-8")
    token.chmod(0o644)

    r = _slm(tmp_path, "token", "show")

    assert r.returncode == 0, r.stderr
    assert stat.S_IMODE(token.stat().st_mode) == 0o600


def test_token_show_creates_the_token_when_there_is_none_yet(tmp_path):
    r = _slm(tmp_path, "token", "show")

    assert r.returncode == 0, r.stderr
    printed = r.stdout.strip()
    assert printed
    assert (tmp_path / ".install_token").read_text(encoding="utf-8").strip() == printed


def test_token_without_a_subcommand_says_how_to_use_it(tmp_path):
    r = _slm(tmp_path, "token")

    assert r.returncode != 0
    assert "slm token show" in (r.stdout + r.stderr)
