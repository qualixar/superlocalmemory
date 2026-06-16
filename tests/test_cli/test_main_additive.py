# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later
"""Tests that existing subcommands are not broken by additive optimize changes."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

# Subprocesses spawned with sys.executable do NOT inherit pytest's
# `pythonpath = ["src"]` (pyproject.toml). Put src/ on the child's path so
# `from superlocalmemory...` imports work from a source checkout (no install).
_SRC = str(Path(__file__).resolve().parents[2] / "src")


def _env():
    e = dict(os.environ)
    e["PYTHONPATH"] = _SRC + (os.pathsep + e["PYTHONPATH"] if e.get("PYTHONPATH") else "")
    return e


def _slm_help(cmd):
    """Run ``slm <cmd>`` and return exit code + stdout."""
    from superlocalmemory.cli.main import main
    import io

    old_argv = sys.argv[:]
    old_stdout = sys.stdout
    sys.argv = ["slm", cmd]
    buf = io.StringIO()
    sys.stdout = buf
    try:
        main()
        rc = 0
    except SystemExit as e:
        rc = e.code if e.code is not None else 0
    finally:
        sys.stdout = old_stdout
        sys.argv = old_argv
    return rc, buf.getvalue()


def test_recall_help_works():
    """slm recall --help should work."""
    result = subprocess.run(
        [sys.executable, "-c", "from superlocalmemory.cli.main import main; import sys; sys.argv=['slm','recall','--help']; main()"],
        capture_output=True, text=True, timeout=15, env=_env(),
    )
    assert result.returncode == 0
    assert "query" in result.stdout.lower() or "recall" in result.stdout.lower()


def test_serve_help_works():
    result = subprocess.run(
        [sys.executable, "-c", "from superlocalmemory.cli.main import main; import sys; sys.argv=['slm','serve','--help']; main()"],
        capture_output=True, text=True, timeout=15, env=_env(),
    )
    assert result.returncode == 0


def test_list_help_works():
    result = subprocess.run(
        [sys.executable, "-c", "from superlocalmemory.cli.main import main; import sys; sys.argv=['slm','list','--help']; main()"],
        capture_output=True, text=True, timeout=15, env=_env(),
    )
    assert result.returncode == 0


def test_optimize_subcommand_parses():
    """slm optimize (no args) should print usage and exit 0."""
    result = subprocess.run(
        [sys.executable, "-c",
         "from superlocalmemory.cli.main import main; import sys; sys.argv=['slm','optimize']; main()"],
        capture_output=True, text=True, timeout=15, env=_env(),
    )
    assert result.returncode == 0
    assert "optimize" in result.stdout.lower() or "usage" in result.stdout.lower()


def test_cache_subcommand_parses():
    result = subprocess.run(
        [sys.executable, "-c",
         "from superlocalmemory.cli.main import main; import sys; sys.argv=['slm','cache']; main()"],
        capture_output=True, text=True, timeout=15, env=_env(),
    )
    assert result.returncode == 0


def test_compress_subcommand_parses():
    result = subprocess.run(
        [sys.executable, "-c",
         "from superlocalmemory.cli.main import main; import sys; sys.argv=['slm','compress']; main()"],
        capture_output=True, text=True, timeout=15, env=_env(),
    )
    assert result.returncode == 0

