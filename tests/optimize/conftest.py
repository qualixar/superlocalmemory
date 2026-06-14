"""Shared pytest fixtures for SLM v3.6 Optimize module tests."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterator

import pytest

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


@pytest.fixture
def tmp_cache_db(tmp_path: Path, monkeypatch) -> Iterator[object]:
    """CacheDB backed by a temp file. Torn down after each test."""
    from superlocalmemory.optimize.storage import db as _db_mod
    from superlocalmemory.optimize.storage.db import CacheDB
    # v3.6.12 (cache-3): isolate the AES key file to tmp — otherwise CacheDB
    # init reads/writes the user's real ~/.superlocalmemory/opt-key.bin, coupling
    # "isolated" tests to machine state (and masking key/salt bugs).
    monkeypatch.setattr(_db_mod, "_KEY_FILE", tmp_path / "opt-key.bin")
    db = CacheDB(tmp_path / "llmcache.db")
    yield db


@pytest.fixture
def tmp_config_store(tmp_path: Path) -> Iterator[object]:
    """ConfigStore backed by a temp file. No watchdog started."""
    from superlocalmemory.optimize.config.store import ConfigStore
    store = ConfigStore(config_path=tmp_path / "optimize.json")
    yield store
