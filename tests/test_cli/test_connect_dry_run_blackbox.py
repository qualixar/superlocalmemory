"""Fresh-home connect dry runs must be observational only."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
FORBIDDEN = (
    ".superlocalmemory",
    ".codex",
    ".claude",
    "AGENTS.md",
)


def _run_connect(home: Path, *args: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["HOME"] = str(home)
    env["PYTHONPATH"] = str(ROOT / "src")
    for name in (
        "SLM_DATA_DIR",
        "SL_MEMORY_PATH",
        "SLM_HOME",
        "CLAUDE_CONFIG_DIR",
        "CODEX_HOME",
    ):
        env.pop(name, None)
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "superlocalmemory.cli.main",
            "connect",
            *args,
            "--dry-run",
        ],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )


@pytest.mark.parametrize("args", [("codex",), tuple()])
def test_connect_dry_run_does_not_initialize_or_edit_fresh_home(
    tmp_path: Path, args: tuple[str, ...],
) -> None:
    home = tmp_path / "fresh-home"
    home.mkdir()

    result = _run_connect(home, *args)

    assert result.returncode == 0, result.stdout + result.stderr
    for relative in FORBIDDEN:
        assert not (home / relative).exists(), (
            f"dry run unexpectedly created {relative}\n"
            f"stdout={result.stdout}\nstderr={result.stderr}"
        )
