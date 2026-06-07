"""LLD-00 §10.6 — ConfigStore tests."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from superlocalmemory.optimize.config.store import ConfigStore
from superlocalmemory.optimize.config.schema import OptimizeConfig


def test_get_returns_default_when_no_file(tmp_config_store) -> None:
    cfg = tmp_config_store.get()
    assert isinstance(cfg, OptimizeConfig)
    # proxy_enabled defaults False; cache_enabled True by default
    assert cfg.proxy_enabled is False


def test_save_persists_to_disk(tmp_path: Path) -> None:
    store1 = ConfigStore(config_path=tmp_path / "optimize.json")
    cfg = OptimizeConfig.from_dict({"proxy_enabled": True})
    store1.save(cfg)

    store2 = ConfigStore(config_path=tmp_path / "optimize.json")
    assert store2.get().proxy_enabled is True


def test_save_bumps_config_version(tmp_config_store) -> None:
    cfg = tmp_config_store.get()
    v0 = cfg.config_version
    tmp_config_store.save(cfg)
    v1 = tmp_config_store.version()
    assert v1 > v0
    # Also verify the in-memory config reflects the new version
    assert tmp_config_store.get().config_version == v1


def test_hot_reload_picks_up_file_change(tmp_path: Path) -> None:
    store = ConfigStore(
        config_path=tmp_path / "optimize.json",
        poll_interval=0.2,
    )
    store.start_watchdog()
    new_cfg = OptimizeConfig.from_dict({"proxy_enabled": True})
    (tmp_path / "optimize.json").write_text(
        json.dumps(new_cfg.as_dict()), encoding="utf-8"
    )
    # Wait for at least 3 poll intervals
    time.sleep(0.7)
    reloaded = store.get()
    store.stop_watchdog()
    assert reloaded.proxy_enabled is True


def test_hot_reload_keeps_old_config_on_invalid_json(tmp_path: Path) -> None:
    cfg_path = tmp_path / "optimize.json"
    store = ConfigStore(config_path=cfg_path, poll_interval=0.2)
    store.save(OptimizeConfig.from_dict({"proxy_enabled": True}))
    store.start_watchdog()
    cfg_path.write_text("{invalid json", encoding="utf-8")
    time.sleep(0.7)
    result = store.get()
    store.stop_watchdog()
    assert result.proxy_enabled is True


def test_version_increments_on_hot_reload(tmp_path: Path) -> None:
    cfg_path = tmp_path / "optimize.json"
    store = ConfigStore(config_path=cfg_path, poll_interval=0.2)
    store.start_watchdog()
    v0 = store.version()
    new_cfg = OptimizeConfig.from_dict({"compress_enabled": True})
    cfg_path.write_text(
        json.dumps(new_cfg.as_dict()), encoding="utf-8"
    )
    time.sleep(0.7)
    v1 = store.version()
    store.stop_watchdog()
    assert v1 > v0


def test_save_raises_on_invalid_config(tmp_config_store) -> None:
    bad_cfg = OptimizeConfig.from_dict({"compress_mode": "explosive"})
    with pytest.raises(ValueError):
        tmp_config_store.save(bad_cfg)


def test_register_change_callback_fires_on_reload(tmp_path: Path) -> None:
    cfg_path = tmp_path / "optimize.json"
    store = ConfigStore(config_path=cfg_path, poll_interval=0.2)
    received: list[bool] = []
    store.register_change_callback(lambda cfg: received.append(cfg.proxy_enabled))
    store.start_watchdog()
    new_cfg = OptimizeConfig.from_dict({"proxy_enabled": True})
    cfg_path.write_text(
        json.dumps(new_cfg.as_dict()), encoding="utf-8"
    )
    time.sleep(0.7)
    store.stop_watchdog()
    assert True in received
