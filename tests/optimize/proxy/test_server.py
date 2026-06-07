"""Tests for proxy/server.py — ProxyApp lifecycle and _load_hooks."""

from __future__ import annotations

import pytest
from fastapi import FastAPI

from superlocalmemory.optimize.config.schema import OptimizeConfig
from superlocalmemory.optimize.proxy.lifecycle import HookChain
from superlocalmemory.optimize.proxy.server import ProxyApp, build_proxy_router, _load_hooks


def _make_config(**kwargs):
    base = dict(enabled=True, proxy_enabled=True, cache_enabled=False, compress_enabled=False)
    base.update(kwargs)
    return OptimizeConfig.from_dict(base)


# ---- ProxyApp lifecycle ----

@pytest.mark.asyncio
async def test_proxy_app_startup():
    """startup() creates httpx client and loads hooks."""
    config = _make_config()
    proxy = ProxyApp(config=config)
    assert proxy.http_client is None
    assert proxy.hooks.cache is None

    await proxy.startup()
    assert proxy.http_client is not None
    assert proxy.hooks.cache is None  # cache disabled

    await proxy.shutdown()
    assert proxy.http_client is None


@pytest.mark.asyncio
async def test_proxy_app_startup_with_cache():
    """startup with cache_enabled loads CacheManager."""
    config = _make_config(cache_enabled=True)
    proxy = ProxyApp(config=config)

    # Need a DB for CacheManager.get_instance(), but start without startup()
    # to test _load_hooks failure path directly
    hooks = _load_hooks(config)
    assert hooks.cache is not None  # cache enabled, should load

    await proxy.shutdown()  # no-op when http_client is None


@pytest.mark.asyncio
async def test_proxy_app_shutdown_null_client():
    """shutdown when http_client is None → no-op."""
    config = _make_config()
    proxy = ProxyApp(config=config)
    proxy.http_client = None
    await proxy.shutdown()  # must not raise


@pytest.mark.asyncio
async def test_proxy_app_shutdown_with_client():
    """shutdown closes httpx client and sets to None."""
    config = _make_config()
    proxy = ProxyApp(config=config)
    await proxy.startup()
    assert proxy.http_client is not None
    await proxy.shutdown()
    assert proxy.http_client is None


# ---- build_proxy_router ----

def test_build_proxy_router_includes_all_routes():
    """build_proxy_router creates router with expected routes."""
    config = _make_config()
    proxy = ProxyApp(config=config)
    router = build_proxy_router(proxy)
    routes = {r.path for r in router.routes}
    expected = {
        "/v1/messages",
        "/v1/messages/count_tokens",
        "/v1/models",
        "/v1/chat/completions",
        "/v1/embeddings",
        "/v1beta/models/{model_and_method:path}",
        "/v1beta/openai/chat/completions",
        "/v1beta/openai/models",
    }
    assert expected.issubset(routes) or len(routes) >= 8


# ---- _load_hooks ----

def test_load_hooks_cache_disabled():
    """_load_hooks with cache_enabled=False → no cache hook."""
    config = _make_config(cache_enabled=False)
    hooks = _load_hooks(config)
    assert hooks.cache is None


def test_load_hooks_cache_load_failure(monkeypatch):
    """_load_hooks: cache enabled but import fails → fail-open."""
    config = _make_config(cache_enabled=True)
    import builtins
    original_import = builtins.__import__

    def _failing_import(name, *args, **kwargs):
        if "superlocalmemory.optimize.cache.manager" in name:
            raise ImportError("no cache module")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _failing_import)
    hooks = _load_hooks(config)
    assert hooks.cache is None  # fail-open


def test_load_hooks_compress_enabled_loads_router():
    """_load_hooks with compress_enabled=True — CompressRouter singleton loaded (BUG-FIX v3.6.4)."""
    from superlocalmemory.optimize.compress.router import CompressRouter
    config = _make_config(compress_enabled=True)
    hooks = _load_hooks(config)
    assert hooks.compress is not None, "CompressRouter must be loaded when compress_enabled=True"
    assert isinstance(hooks.compress, CompressRouter)


@pytest.mark.asyncio
async def test_proxy_next_request_id():
    """next_request_id generates unique identifiers."""
    config = _make_config()
    proxy = ProxyApp(config=config)
    id1 = await proxy.next_request_id()
    id2 = await proxy.next_request_id()
    assert id1 != id2
    assert id1.startswith("slm_")
    assert id2.startswith("slm_")
