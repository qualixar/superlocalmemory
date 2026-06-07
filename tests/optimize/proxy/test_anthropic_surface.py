"""LLD-01 §8 — Anthropic surface tests (no respx dep, use in-test httpx mock)."""

from __future__ import annotations

import json
import time
from typing import Any

import httpx
import pytest
from fastapi import FastAPI

from superlocalmemory.optimize.config.schema import OptimizeConfig
from superlocalmemory.optimize.proxy.lifecycle import (
    CachedResponse,
    HookChain,
    ProviderResponse,
    ProxyRequest,
)
from superlocalmemory.optimize.proxy.server import ProxyApp, build_proxy_router


def _make_config(**kwargs: Any) -> OptimizeConfig:
    base = dict(
        enabled=True, proxy_enabled=True,
        cache_enabled=False, compress_enabled=False,
        ttl_seconds=300,
    )
    base.update(kwargs)
    return OptimizeConfig.from_dict(base)


def _anthropic_headers() -> dict:
    return {
        "x-api-key": "sk-ant-test",
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }


def _sample_req() -> dict:
    return {
        "model": "claude-sonnet-4-6",
        "max_tokens": 100,
        "messages": [{"role": "user", "content": "hi"}],
    }


def _sample_resp() -> dict:
    return {
        "id": "msg_01",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": "hello"}],
        "model": "claude-sonnet-4-6",
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }


class _MockTransport(httpx.AsyncBaseTransport):
    """Minimal in-test mock for httpx that records requests and returns canned responses."""

    def __init__(self, handler):
        self._handler = handler
        self.requests: list[httpx.Request] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return await self._handler(request)


def _make_app(config: OptimizeConfig, hooks: HookChain) -> tuple[FastAPI, ProxyApp, _MockTransport]:
    transport = _MockTransport(handler=_default_handler())
    proxy = ProxyApp(config=config)
    proxy.hooks = hooks
    proxy.http_client = httpx.AsyncClient(transport=transport)
    router = build_proxy_router(proxy)
    app = FastAPI()
    app.include_router(router, prefix="")
    return app, proxy, transport


def _default_handler():
    async def _handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_sample_resp())
    return _handler


@pytest.mark.asyncio
async def test_fail_open_cache_killed() -> None:
    """Kill cache hook mid-flight — request must still succeed."""

    class BrokenCacheHook:
        def check(self, ctx):
            raise RuntimeError("cache down")
        def store(self, ctx, resp):
            raise RuntimeError("cache down")
        def on_hit(self, ctx, resp, tokens_saved):
            raise RuntimeError("cache down")
        def on_miss(self, ctx):
            raise RuntimeError("cache down")

    config = _make_config(cache_enabled=True)
    app, proxy, _t = _make_app(config, HookChain(cache=BrokenCacheHook(), compress=None))

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.post("/v1/messages", json=_sample_req(), headers=_anthropic_headers())

    assert resp.status_code == 200
    assert resp.json()["type"] == "message"

    await proxy.http_client.aclose()


@pytest.mark.asyncio
async def test_non_200_not_cached() -> None:
    """401 must reach client AND must not be stored."""
    stored_calls: list = []

    class TrackingCacheHook:
        def check(self, ctx):
            return CachedResponse(hit=False, data=None, cache_key="k1", ttl_seconds=300)
        def store(self, ctx, resp):
            stored_calls.append(True)
        def on_hit(self, ctx, resp, ts):
            pass
        def on_miss(self, ctx):
            pass

    async def _handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"type": "error", "error": {"type": "authentication_error"}})

    config = _make_config(cache_enabled=True)
    proxy = ProxyApp(config=config)
    proxy.hooks = HookChain(cache=TrackingCacheHook(), compress=None)
    proxy.http_client = httpx.AsyncClient(transport=_MockTransport(handler=_handler))
    app = FastAPI()
    app.include_router(build_proxy_router(proxy), prefix="")

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.post("/v1/messages", json=_sample_req(), headers=_anthropic_headers())

    assert resp.status_code == 401
    assert stored_calls == []

    await proxy.http_client.aclose()


@pytest.mark.asyncio
async def test_empty_cache_key_skips_store() -> None:
    """cache_key='' must skip store() (fixes C-19)."""
    store_called: list = []

    class EmptyKeyHook:
        def check(self, ctx):
            return CachedResponse(hit=False, data=None, cache_key="", ttl_seconds=0)
        def store(self, ctx, resp):
            store_called.append(True)
        def on_hit(self, ctx, resp, ts):
            pass
        def on_miss(self, ctx):
            pass

    config = _make_config(cache_enabled=True)
    app, proxy, _t = _make_app(config, HookChain(cache=EmptyKeyHook(), compress=None))

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.post("/v1/messages", json=_sample_req(), headers=_anthropic_headers())

    assert resp.status_code == 200
    assert store_called == []

    await proxy.http_client.aclose()


@pytest.mark.asyncio
async def test_api_key_redacted_on_proxy_request() -> None:
    """CWE-532 guard: x-api-key must appear as [REDACTED] on ProxyRequest.headers."""
    captured: dict = {}

    class HeaderCapturingHook:
        def check(self, req):
            captured.update(req.headers)
            return CachedResponse(hit=False, data=None, cache_key="", ttl_seconds=0)
        def store(self, req, resp):
            pass
        def on_hit(self, req, resp, ts):
            pass
        def on_miss(self, req):
            pass

    config = _make_config(cache_enabled=True)
    app, proxy, _t = _make_app(config, HookChain(cache=HeaderCapturingHook(), compress=None))

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        await client.post(
            "/v1/messages", json=_sample_req(),
            headers={"x-api-key": "sk-ant-real-secret-12345",
                     "anthropic-version": "2023-06-01", "content-type": "application/json"},
        )

    assert captured.get("x-api-key") == "[REDACTED]"
    assert "sk-ant-real-secret" not in str(captured)

    await proxy.http_client.aclose()


@pytest.mark.asyncio
async def test_v1_models_passthrough() -> None:
    """GET /v1/models reaches upstream (fixes C-05)."""
    models_resp = {
        "data": [
            {"id": "claude-sonnet-4-6", "type": "model", "display_name": "Claude Sonnet 4.6",
             "created_at": "2025-11-01T00:00:00Z"},
        ],
        "has_more": False,
    }

    async def _handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=models_resp)

    config = _make_config()
    proxy = ProxyApp(config=config)
    proxy.hooks = HookChain.empty()
    proxy.http_client = httpx.AsyncClient(transport=_MockTransport(handler=_handler))
    app = FastAPI()
    app.include_router(build_proxy_router(proxy), prefix="")

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.get("/v1/models", headers=_anthropic_headers())

    assert resp.status_code == 200
    assert "data" in resp.json()

    await proxy.http_client.aclose()


@pytest.mark.asyncio
async def test_stream_forward_null_client_returns_502() -> None:
    """RA-C-02: streaming + http_client=None → 502 (not 200 with error body)."""
    config = _make_config()
    proxy = ProxyApp(config=config)
    proxy.http_client = None  # simulate missing startup
    app = FastAPI()
    app.include_router(build_proxy_router(proxy), prefix="")

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.post(
            "/v1/messages",
            json={**_sample_req(), "stream": True},
            headers=_anthropic_headers(),
        )

    assert resp.status_code == 502


@pytest.mark.asyncio
async def test_tool_use_id_preserved() -> None:
    """tool_use.id must pass through unchanged. Cache is skipped (tools present)."""
    tool_use_id = "toolu_01A09q90qw90lq917835lq9"
    upstream_body = {
        "id": "msg_abc", "type": "message", "role": "assistant",
        "content": [{"type": "tool_use", "id": tool_use_id, "name": "bash", "input": {"command": "ls"}}],
        "stop_reason": "tool_use", "model": "claude-sonnet-4-6",
        "usage": {"input_tokens": 100, "output_tokens": 50},
    }

    async def _handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=upstream_body)

    config = _make_config()
    app, proxy, _t = _make_app(config, HookChain.empty())
    proxy.http_client = httpx.AsyncClient(transport=_MockTransport(handler=_handler))

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.post(
            "/v1/messages",
            json={**_sample_req(), "tools": [{"name": "bash", "description": "run bash",
                                              "input_schema": {"type": "object"}}]},
            headers=_anthropic_headers(),
        )

    assert resp.status_code == 200
    content = resp.json()["content"]
    assert content[0]["id"] == tool_use_id

    await proxy.http_client.aclose()
