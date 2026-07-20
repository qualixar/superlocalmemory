# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com

"""WP-07 cross-install tests — RED phase.

Tests for:
  - _ensure_initialized() creating dirs + config on fresh home (AC1)
  - idempotency: 2nd call leaves config mtime unchanged (AC2)
  - never-overwrite: custom config SHA256 identical after (AC3)
  - degrade gracefully on OSError / read-only home (AC4)
  - slm_home() resolves all 3 env aliases to same path (AC5)
  - stdout-silent: _ensure_initialized() never prints to stdout (CRIT-3)
  - slm init --auto non-interactive exits 0, config + sentinel exist (AC6)
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import sys
from pathlib import Path
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# AC1 — fresh home: dirs + config created
# ---------------------------------------------------------------------------


def test_ensure_initialized_creates_dirs_on_fresh_home(tmp_path, monkeypatch):
    """_ensure_initialized() on a fresh HOME creates the dir + config.json."""
    new_home = tmp_path / "slm_home"
    monkeypatch.setenv("SLM_DATA_DIR", str(new_home))
    monkeypatch.delenv("SL_MEMORY_PATH", raising=False)
    monkeypatch.delenv("SLM_HOME", raising=False)

    # Module must be imported AFTER env is set so slm_home() picks it up
    from superlocalmemory.cli._lazy_init import _ensure_initialized, slm_home

    assert not new_home.exists(), "precondition: home must not exist yet"

    _ensure_initialized()

    assert new_home.is_dir(), "home directory should have been created"
    config = new_home / "config.json"
    assert config.exists(), "config.json should have been created"

    # Validate it is parseable JSON and has a mode key
    data = json.loads(config.read_text())
    assert "mode" in data, "config.json must contain a 'mode' key"
    from superlocalmemory import __version__

    assert data["version"] == __version__, (
        "first-run config must identify the installed runtime, not a stale release"
    )


# ---------------------------------------------------------------------------
# AC2 — idempotent: 2nd call zero writes, mtime unchanged
# ---------------------------------------------------------------------------


def test_ensure_initialized_is_idempotent_no_rewrite(tmp_path, monkeypatch):
    """Second _ensure_initialized() call must not rewrite config.json (mtime unchanged)."""
    home = tmp_path / "slm_home"
    monkeypatch.setenv("SLM_DATA_DIR", str(home))
    monkeypatch.delenv("SL_MEMORY_PATH", raising=False)
    monkeypatch.delenv("SLM_HOME", raising=False)

    from superlocalmemory.cli._lazy_init import _ensure_initialized

    # First call — creates the file
    _ensure_initialized()

    config = home / "config.json"
    assert config.exists()
    mtime_after_first = config.stat().st_mtime_ns

    # Second call — must be a no-op
    _ensure_initialized()

    mtime_after_second = config.stat().st_mtime_ns
    assert mtime_after_first == mtime_after_second, (
        "config.json mtime changed on second call — not idempotent"
    )


# ---------------------------------------------------------------------------
# AC3 — never overwrite an existing custom config
# ---------------------------------------------------------------------------


def test_ensure_initialized_never_overwrites_user_config(tmp_path, monkeypatch):
    """If config.json already exists, _ensure_initialized() must not touch it."""
    home = tmp_path / "slm_home"
    home.mkdir(parents=True)
    monkeypatch.setenv("SLM_DATA_DIR", str(home))
    monkeypatch.delenv("SL_MEMORY_PATH", raising=False)
    monkeypatch.delenv("SLM_HOME", raising=False)

    custom_data = {"mode": "c", "custom_key": "do_not_touch", "llm": {"provider": "openai"}}
    config = home / "config.json"
    config.write_text(json.dumps(custom_data))

    sha_before = _sha256(config)

    from superlocalmemory.cli._lazy_init import _ensure_initialized

    _ensure_initialized()

    sha_after = _sha256(config)
    assert sha_before == sha_after, (
        f"config.json was modified! SHA256 before={sha_before} after={sha_after}"
    )


# ---------------------------------------------------------------------------
# AC4 — degrade silently on OSError (read-only home)
# ---------------------------------------------------------------------------


def test_ensure_initialized_degrades_on_oserror(tmp_path, monkeypatch):
    """_ensure_initialized() must NOT raise even if home is read-only."""
    # Point to a path inside a read-only parent so mkdir raises OSError
    ro_parent = tmp_path / "ro_parent"
    ro_parent.mkdir()
    # Make it read-only
    ro_parent.chmod(stat.S_IRUSR | stat.S_IXUSR)

    new_home = ro_parent / "slm_home"
    monkeypatch.setenv("SLM_DATA_DIR", str(new_home))
    monkeypatch.delenv("SL_MEMORY_PATH", raising=False)
    monkeypatch.delenv("SLM_HOME", raising=False)

    from superlocalmemory.cli._lazy_init import _ensure_initialized

    try:
        # Must not raise — AC4
        _ensure_initialized()
    finally:
        # Restore permissions so tmp_path cleanup works
        ro_parent.chmod(stat.S_IRWXU)


# ---------------------------------------------------------------------------
# AC5 — slm_home() resolves all 3 env aliases to the same path
# ---------------------------------------------------------------------------


def test_slm_home_aliases_resolve_equal(tmp_path, monkeypatch):
    """SLM_DATA_DIR, SL_MEMORY_PATH, SLM_HOME each map to the same resolved Path."""
    target = tmp_path / "my_slm"

    from superlocalmemory.cli._lazy_init import slm_home

    # SLM_DATA_DIR takes priority
    monkeypatch.setenv("SLM_DATA_DIR", str(target))
    monkeypatch.delenv("SL_MEMORY_PATH", raising=False)
    monkeypatch.delenv("SLM_HOME", raising=False)
    result_data_dir = slm_home()

    # SL_MEMORY_PATH (SLM_DATA_DIR absent)
    monkeypatch.delenv("SLM_DATA_DIR")
    monkeypatch.setenv("SL_MEMORY_PATH", str(target))
    monkeypatch.delenv("SLM_HOME", raising=False)
    result_sl_memory = slm_home()

    # SLM_HOME (both others absent)
    monkeypatch.delenv("SL_MEMORY_PATH")
    monkeypatch.setenv("SLM_HOME", str(target))
    result_slm_home = slm_home()

    assert result_data_dir == result_sl_memory == result_slm_home == target, (
        f"Alias resolution mismatch: {result_data_dir} / {result_sl_memory} / {result_slm_home}"
    )


# ---------------------------------------------------------------------------
# CRIT-3 — stdout-silent (mandatory)
# ---------------------------------------------------------------------------


def test_lazy_init_is_stdout_silent(tmp_path, monkeypatch, capsys):
    """_ensure_initialized() must produce ZERO stdout output."""
    home = tmp_path / "silent_test"
    monkeypatch.setenv("SLM_DATA_DIR", str(home))
    monkeypatch.delenv("SL_MEMORY_PATH", raising=False)
    monkeypatch.delenv("SLM_HOME", raising=False)

    from superlocalmemory.cli._lazy_init import _ensure_initialized

    _ensure_initialized()

    captured = capsys.readouterr()
    assert captured.out == "", (
        f"_ensure_initialized() printed to stdout — violates CRIT-3: {captured.out!r}"
    )


# ---------------------------------------------------------------------------
# AC6 — slm init --auto non-interactive exits 0, config + sentinel exist
# ---------------------------------------------------------------------------


def test_slm_init_auto_non_interactive(tmp_path, monkeypatch):
    """slm init --auto in non-interactive mode exits 0 and creates config + sentinel."""
    home = tmp_path / "auto_home"
    monkeypatch.setenv("SLM_DATA_DIR", str(home))
    monkeypatch.delenv("SL_MEMORY_PATH", raising=False)
    monkeypatch.delenv("SLM_HOME", raising=False)
    monkeypatch.setenv("SLM_NON_INTERACTIVE", "1")

    # Simulate argparse Namespace with args.auto=True, args.force=False, args.gate=False
    import argparse
    args = argparse.Namespace(command="init", auto=True, force=False, gate=False)

    from superlocalmemory.cli.commands import cmd_init

    # cmd_init should not raise and should exit 0 (no sys.exit for success in auto mode)
    try:
        cmd_init(args)
    except SystemExit as exc:
        assert exc.code == 0 or exc.code is None, (
            f"cmd_init --auto exited with non-zero code: {exc.code}"
        )

    # config.json must exist
    config = home / "config.json"
    assert config.exists(), "config.json must exist after slm init --auto"

    # .setup-complete sentinel must exist
    sentinel = home / ".setup-complete"
    assert sentinel.exists(), ".setup-complete sentinel must exist after slm init --auto"
