# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""``scripts/build_cli_skill.py``: --help and a typo never rewrite the skill; no temp folder is left."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "build_cli_skill.py"
SKILL = REPO / "plugin-src" / "skills" / "slm-cli" / "SKILL.md"


def _run(tmp_path: Path, *args: str) -> subprocess.CompletedProcess:
    temp = tmp_path / "tmp"
    temp.mkdir(exist_ok=True)
    env = {**os.environ, "TMPDIR": str(temp), "HOME": str(tmp_path / "home")}
    env.pop("SLM_DATA_DIR", None)
    return subprocess.run([sys.executable, str(SCRIPT), *args], capture_output=True, text=True,
                          timeout=300, env=env, cwd=str(tmp_path))


@pytest.mark.parametrize("args, code", [(["--help"], 0), (["-h"], 0), (["--chek"], 2)])
def test_help_or_a_typo_leaves_the_skill_untouched(tmp_path, args, code):
    before = (SKILL.read_bytes(), SKILL.stat().st_mtime_ns)
    proc = _run(tmp_path, *args)
    assert proc.returncode == code, proc.stderr
    assert "--check" in proc.stdout + proc.stderr
    assert (SKILL.read_bytes(), SKILL.stat().st_mtime_ns) == before


def test_check_leaves_no_temp_folder_behind(tmp_path):
    proc = _run(tmp_path, "--check")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert list((tmp_path / "tmp").iterdir()) == []
