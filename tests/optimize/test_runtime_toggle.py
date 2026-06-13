# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Runtime independent cache/compress toggles (v3.6.10).

Proves cache_enabled and compress_enabled can be flipped at runtime — no
restart — via the ConfigStore change-callback → ProxyApp.reload_from_config →
HookChain rebuild path. Covers all four combinations (cache-only, compress-only,
both, neither).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from superlocalmemory.optimize.config.schema import OptimizeConfig
from superlocalmemory.optimize.config.store import ConfigStore
from superlocalmemory.optimize.proxy.server import ProxyApp


def _cfg(**kw) -> OptimizeConfig:
    base = dict(
        enabled=True, proxy_enabled=True,
        cache_enabled=False, compress_enabled=False, ttl_seconds=300,
    )
    base.update(kw)
    return OptimizeConfig.from_dict(base)


def _proxy(cfg: OptimizeConfig) -> ProxyApp:
    p = ProxyApp(config=cfg)
    # Build the initial chain the way startup() would (without the http client).
    p.reload_from_config(cfg)
    return p


# ---- ConfigStore.save fires callbacks immediately --------------------------

def test_save_fires_change_callbacks(tmp_path: Path) -> None:
    store = ConfigStore(config_path=tmp_path / "optimize.json")
    seen: list[OptimizeConfig] = []
    store.register_change_callback(lambda c: seen.append(c))
    store.save(_cfg(cache_enabled=True))
    assert len(seen) == 1
    assert seen[0].cache_enabled is True


def test_save_callback_exception_does_not_break_save(tmp_path: Path) -> None:
    store = ConfigStore(config_path=tmp_path / "optimize.json")
    store.register_change_callback(lambda c: (_ for _ in ()).throw(RuntimeError("boom")))
    # Must not raise — save is the source of truth; callbacks are best-effort.
    store.save(_cfg(cache_enabled=True))
    assert store.get().cache_enabled is True


# ---- ProxyApp.reload_from_config rebuilds the hook chain --------------------

@pytest.mark.parametrize("cache_on,compress_on", [
    (False, False),  # neither
    (True, False),   # cache-only
    (False, True),   # compress-only
    (True, True),    # both
])
def test_reload_independent_toggles(cache_on: bool, compress_on: bool) -> None:
    p = _proxy(_cfg())
    p.reload_from_config(_cfg(cache_enabled=cache_on, compress_enabled=compress_on))
    assert (p.hooks.cache is not None) == cache_on
    assert (p.hooks.compress is not None) == compress_on


def test_toggle_cache_off_then_on_at_runtime() -> None:
    p = _proxy(_cfg(cache_enabled=True))
    assert p.hooks.cache is not None
    p.reload_from_config(_cfg(cache_enabled=False))
    assert p.hooks.cache is None            # turned OFF live
    p.reload_from_config(_cfg(cache_enabled=True))
    assert p.hooks.cache is not None        # turned back ON live


# ---- End-to-end: store.save → callback → proxy hooks swap -------------------

def test_store_save_drives_proxy_reload(tmp_path: Path) -> None:
    store = ConfigStore(config_path=tmp_path / "optimize.json")
    p = _proxy(_cfg(cache_enabled=False, compress_enabled=False))
    store.register_change_callback(p.reload_from_config)

    assert p.hooks.cache is None and p.hooks.compress is None
    # UI flips cache ON only
    store.save(_cfg(cache_enabled=True, compress_enabled=False))
    assert p.hooks.cache is not None
    assert p.hooks.compress is None
    # UI flips to compress ON only
    store.save(_cfg(cache_enabled=False, compress_enabled=True))
    assert p.hooks.cache is None
    assert p.hooks.compress is not None
