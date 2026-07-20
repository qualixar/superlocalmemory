# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later
"""Release contract: every root command advertised by ``slm --help`` is inspectable.

Help is deliberately exercised in a clean subprocess.  The CLI promises this
metadata path is activation-free, so a user can discover every command before
creating a data root, starting the daemon, or downloading a model.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


_ROOT_COMMANDS = (
    "init", "setup", "mode", "provider", "connect", "migrate", "db",
    "remember", "recall", "search", "forget", "delete", "update", "list",
    "status", "health", "trace", "doctor", "wrap", "mcp", "warmup",
    "dashboard", "serve", "restart", "profile", "hooks", "session-context",
    "session", "observe", "decay", "quantize", "consolidate", "soft-prompts",
    "reap", "adapters", "ingest", "config", "evolve", "disable", "enable",
    "clear-cache", "reconfigure", "benchmark", "evidence", "diagnostics",
    "rotate-token", "optimize", "cache", "compress", "proxy", "help-optimize",
)

_SRC = str(Path(__file__).resolve().parents[2] / "src")


@pytest.mark.parametrize("command", _ROOT_COMMANDS)
def test_every_advertised_root_command_has_activation_free_help(
    command: str, tmp_path: Path,
) -> None:
    """``slm <command> --help`` must not require a daemon or mutate SLM_HOME."""
    env = dict(os.environ)
    env["PYTHONPATH"] = _SRC + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["SLM_HOME"] = str(tmp_path / "must-not-be-created")

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from superlocalmemory.cli.main import main; import sys; "
            f"sys.argv=['slm', {command!r}, '--help']; main()",
        ],
        capture_output=True,
        text=True,
        timeout=15,
        env=env,
    )

    assert result.returncode == 0, result.stderr
    assert "usage: slm" in result.stdout.lower()
    assert not (tmp_path / "must-not-be-created").exists()
