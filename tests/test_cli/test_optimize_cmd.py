# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later
"""Tests for ``slm optimize`` CLI handlers."""

import json
from argparse import Namespace
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _isolate_config(tmp_path, monkeypatch):
    """Redirect ConfigStore and CacheDB to tmp paths."""
    from superlocalmemory.optimize.config.store import ConfigStore
    from superlocalmemory.optimize.config.defaults import DEFAULT_OPTIMIZE_CONFIG

    _store = ConfigStore(config_path=tmp_path / "optimize.json")
    monkeypatch.setattr(
        "superlocalmemory.cli.optimize_cmd._get_store", lambda: _store
    )
    return _store


@pytest.fixture
def _isolate_db(tmp_path, monkeypatch):
    from superlocalmemory.optimize.storage.db import CacheDB

    _db = CacheDB(tmp_path / "llmcache.db")
    monkeypatch.setattr(
        "superlocalmemory.cli.optimize_cmd._get_cache_db", lambda: _db
    )
    return _db


def test_optimize_status_human(capsys):
    from superlocalmemory.cli.optimize_cmd import cmd_optimize_status

    cmd_optimize_status(Namespace(json=False))
    out = capsys.readouterr().out
    assert "Optimize:" in out
    assert "Cache:" in out
    assert "Compress:" in out


def test_optimize_status_json(capsys):
    from superlocalmemory.cli.optimize_cmd import cmd_optimize_status

    cmd_optimize_status(Namespace(json=True))
    out = capsys.readouterr().out
    data = json.loads(out)
    for key in ("status", "optimize_enabled", "cache_enabled", "compress_enabled", "config_version"):
        assert key in data, f"Missing key: {key}"


def test_optimize_on_writes_config(_isolate_config):
    from superlocalmemory.cli.optimize_cmd import cmd_optimize_on
    import dataclasses

    cmd_optimize_on(Namespace(json=False))
    cfg = _isolate_config.get()
    assert cfg.enabled is True
    assert cfg.cache_enabled is True
    assert cfg.compress_enabled is True


def test_optimize_off_writes_config(_isolate_config):
    from superlocalmemory.cli.optimize_cmd import cmd_optimize_off
    import dataclasses

    # Pre-enable
    cfg = dataclasses.replace(
        _isolate_config.get(),
        enabled=True, cache_enabled=True, semantic_enabled=True, compress_enabled=True,
    )
    _isolate_config.save(cfg)

    cmd_optimize_off(Namespace(json=False))
    final = _isolate_config.get()
    assert final.enabled is False
    assert final.cache_enabled is False
    assert final.semantic_enabled is False
    assert final.compress_enabled is False


def test_optimize_savings_invalid_since(_isolate_db):
    from superlocalmemory.cli.optimize_cmd import cmd_optimize_savings

    with pytest.raises(SystemExit) as exc:
        cmd_optimize_savings(Namespace(json=False, since=0, provider=None))
    assert exc.value.code == 1


def test_optimize_savings_no_crash(_isolate_db, capsys):
    from superlocalmemory.cli.optimize_cmd import cmd_optimize_savings

    cmd_optimize_savings(Namespace(json=False, since=7, provider=None))
    out = capsys.readouterr().out
    assert "Savings" in out


def test_optimize_savings_json(_isolate_db, capsys):
    from superlocalmemory.cli.optimize_cmd import cmd_optimize_savings

    cmd_optimize_savings(Namespace(json=True, since=7, provider=None))
    out = capsys.readouterr().out
    data = json.loads(out)
    assert "tokens_saved" in data
    assert "estimated_savings_usd" in data
    assert "pricing_date" in data


def test_optimize_config_version_increments(_isolate_config):
    v0 = _isolate_config.get().config_version
    from superlocalmemory.cli.optimize_cmd import cmd_optimize_on

    cmd_optimize_on(Namespace(json=False))
    v1 = _isolate_config.get().config_version
    assert v1 > v0
