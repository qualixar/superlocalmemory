"""Module-level accessor tests."""

from __future__ import annotations

import json
from pathlib import Path

from superlocalmemory.optimize.config import (
    get_optimize_config,
    _reset_config_store,
    _set_config_store,
)
from superlocalmemory.optimize.config.defaults import DEFAULT_OPTIMIZE_CONFIG


def setup_function(_func) -> None:
    _reset_config_store()


def teardown_function(_func) -> None:
    _reset_config_store()


def test_get_optimize_config_returns_default_when_no_store() -> None:
    cfg = get_optimize_config()
    assert cfg is DEFAULT_OPTIMIZE_CONFIG
    assert cfg.proxy_enabled is False


def test_set_store_makes_it_canonical(tmp_path: Path) -> None:
    from superlocalmemory.optimize.config.store import ConfigStore
    p = tmp_path / "optimize.json"
    p.write_text(json.dumps({"proxy_enabled": True}))
    store = ConfigStore(config_path=p)
    _set_config_store(store)
    cfg = get_optimize_config()
    assert cfg.proxy_enabled is True
