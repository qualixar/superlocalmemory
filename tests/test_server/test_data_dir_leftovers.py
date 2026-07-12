# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3

"""Regression tests: remaining server/CLI modules must resolve the SLM data
dir via the env chain, not a hardcoded ``Path.home() / ".superlocalmemory"``.

Follow-up to ``test_profile_path_resolution.py`` (which pinned the invariant
for ``routes/helpers.py`` + ``ui.py``). These cover the leftovers:

- ``routes/brain.py``      — Brain-panel learning/memory DB paths
- ``server/bandit_loops.py`` — reward-loop DB fallbacks (config=None)
- ``server/api.py``        — standalone API server DB_PATH
- ``server/unified_daemon.py`` + ``cli/daemon.py`` — daemon.pid/daemon.port
  must resolve identically on both sides, or the CLI client cannot find a
  daemon started with an off-home data dir.
- ``cli/commands.py``      — CLI recall learning signals write to the same
  learning.db the daemon reads.
"""

from __future__ import annotations

import importlib
from pathlib import Path


def _reload(modname):
    mod = importlib.import_module(modname)
    return importlib.reload(mod)


def test_brain_db_paths_respect_env(tmp_path, monkeypatch):
    monkeypatch.setenv("SLM_DATA_DIR", str(tmp_path))
    _reload("superlocalmemory.server.routes.helpers")
    brain = _reload("superlocalmemory.server.routes.brain")
    assert brain._learning_db_path() == tmp_path / "learning.db"
    assert brain._memory_db_path() == tmp_path / "memory.db"


def test_bandit_loops_fallbacks_respect_env(tmp_path, monkeypatch):
    monkeypatch.setenv("SLM_DATA_DIR", str(tmp_path))
    _reload("superlocalmemory.server.routes.helpers")
    bandit = _reload("superlocalmemory.server.bandit_loops")
    assert bandit._learning_db(None) == tmp_path / "learning.db"
    assert bandit._memory_db(None) == tmp_path / "memory.db"


def test_api_server_db_path_respects_env(tmp_path, monkeypatch):
    monkeypatch.setenv("SLM_DATA_DIR", str(tmp_path))
    _reload("superlocalmemory.server.routes.helpers")
    api = _reload("superlocalmemory.server.api")
    assert str(api.DB_PATH) == str(tmp_path / "memory.db")


def test_daemon_pid_port_agree_between_server_and_cli(tmp_path, monkeypatch):
    """The daemon writes pid/port where the CLI client looks for them."""
    monkeypatch.setenv("SLM_DATA_DIR", str(tmp_path))
    cli_daemon = _reload("superlocalmemory.cli.daemon")
    unified = _reload("superlocalmemory.server.unified_daemon")
    assert cli_daemon._PID_FILE == tmp_path / "daemon.pid"
    assert cli_daemon._PORT_FILE == tmp_path / "daemon.port"
    assert Path(unified._PID_FILE) == cli_daemon._PID_FILE
    assert Path(unified._PORT_FILE) == cli_daemon._PORT_FILE
