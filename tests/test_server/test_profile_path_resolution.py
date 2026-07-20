# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3

"""Regression test: dashboard and CLI must resolve the same SLM data dir.

Before the fix, ``server/routes/helpers.py`` and ``server/ui.py`` hardcoded
``MEMORY_DIR = Path.home() / ".superlocalmemory"``. The CLI, via
``SLMConfig.load()`` → ``cli/_lazy_init.py:slm_home()``, honours the
``SLM_DATA_DIR`` / ``SL_MEMORY_PATH`` / ``SLM_HOME`` env vars (and
``config.json:base_dir``). When those diverged, a profile created in the
dashboard landed in ``~/.superlocalmemory/memory.db`` while
``slm profile list`` read a different DB and reported the profile missing.

This test pins the invariant: the server-side path constants must resolve
to the env-overridden directory, not the hard default.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest


def _reload_helpers():
    """Reload the helpers module so module-level state picks up env vars.

    The path constants are lazy proxies (resolved on first access), but we
    reload to be defensive against any future eager binding.
    """
    import superlocalmemory.server.routes.helpers as helpers
    return importlib.reload(helpers)


def test_memory_dir_respects_slm_data_dir(tmp_path, monkeypatch):
    """MEMORY_DIR must point at $SLM_DATA_DIR, not ~/.superlocalmemory."""
    monkeypatch.setenv("SLM_DATA_DIR", str(tmp_path))
    helpers = _reload_helpers()
    assert str(helpers.MEMORY_DIR) == str(tmp_path), (
        f"MEMORY_DIR resolved to {helpers.MEMORY_DIR!s}, expected {tmp_path}. "
        "Dashboard profile writes would land in the wrong directory."
    )
    assert str(helpers.DB_PATH) == str(tmp_path / "memory.db")
    assert str(helpers.PROFILES_DIR) == str(tmp_path / "profiles")


def test_db_path_resolves_under_env_override(tmp_path, monkeypatch):
    """DB_PATH must resolve under the env-overridden home, matching the CLI."""
    monkeypatch.setenv("SLM_DATA_DIR", str(tmp_path))
    helpers = _reload_helpers()
    # The proxy must behave like a real Path for the operations the route
    # helpers perform: __fspath__ (sqlite3.connect(str(DB_PATH))), /, exists().
    assert os_fspath(helpers.DB_PATH) == str(tmp_path / "memory.db")
    derived = helpers.MEMORY_DIR / "profiles.json"
    assert isinstance(derived, Path)
    assert derived == tmp_path / "profiles.json"
    assert helpers.DB_PATH.exists() is False  # nothing created yet


def test_dashboard_and_cli_agree_on_profile_dir(tmp_path, monkeypatch):
    """End-to-end: a profile written via the dashboard path helpers must be
    visible to the CLI read path, because both resolve the same DB."""
    monkeypatch.setenv("SLM_DATA_DIR", str(tmp_path))
    helpers = _reload_helpers()

    # CLI side: DatabaseManager writes to config.db_path (env-resolved).
    from superlocalmemory.core.config import SLMConfig
    from superlocalmemory.storage.database import DatabaseManager
    from superlocalmemory.storage import schema

    config = SLMConfig.load()
    assert str(config.db_path) == str(tmp_path / "memory.db"), (
        "CLI config.db_path does not match env override — test premise broken"
    )
    db = DatabaseManager(config.db_path)
    db.initialize(schema)
    db.execute(
        "INSERT OR IGNORE INTO profiles (profile_id, name) VALUES (?, ?)",
        ("work", "work"),
    )

    # Dashboard side: route helpers read DB_PATH (must be the same file).
    assert str(helpers.DB_PATH) == str(config.db_path), (
        "Dashboard DB_PATH and CLI config.db_path diverge — profiles created "
        "in the dashboard would be invisible to `slm profile list`."
    )

    # Dashboard read path sees the CLI-written profile.
    merged = helpers.sync_profiles()
    ids = {p["profile_id"] for p in merged}
    assert "work" in ids, f"Dashboard sync_profiles() did not see 'work': {ids}"


def os_fspath(p) -> str:
    import os
    return os.fspath(p)
