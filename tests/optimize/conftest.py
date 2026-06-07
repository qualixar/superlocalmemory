"""Shared pytest fixtures for SLM v3.6 Optimize module tests."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterator

import pytest

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


@pytest.fixture
def tmp_cache_db(tmp_path: Path) -> Iterator[object]:
    """CacheDB backed by a temp file. Torn down after each test."""
    from superlocalmemory.optimize.storage.db import CacheDB
    db = CacheDB(tmp_path / "llmcache.db")
    yield db


@pytest.fixture
def tmp_config_store(tmp_path: Path) -> Iterator[object]:
    """ConfigStore backed by a temp file. No watchdog started."""
    from superlocalmemory.optimize.config.store import ConfigStore
    store = ConfigStore(config_path=tmp_path / "optimize.json")
    yield store
