"""LLD-01 §5.5 — OpenAI surface tests."""

from __future__ import annotations

import json
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
from superlocalmemory.optimize.proxy._helpers import _MockTransport


def _openai_headers() -> dict:
    return {
        "authorization": "Bearer sk-openai-test",
        "content-type": "application/json",
    }


def _sample_req() -> dict:
    return {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "hi"}],
    }


def _sample_resp() -> dict:
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "hello"},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }


def _make_app(config: OptimizeConfig, hooks: HookChain, handler=None) -> tuple[FastAPI, ProxyApp, _MockTransport]:
    transport = _MockTransport(handler=handler or (lambda r: httpx.Response(200, json=_sample_resp())))
    proxy = ProxyApp(config=config)
    proxy.hooks = hooks
    proxy.http_client = httpx.AsyncClient(transport=transport)
    app = FastAPI()
    app.include_router(build_proxy_router(proxy), prefix="")
    return app, proxy, transport


@pytest.mark.asyncio
async def test_chat_completions_passthrough() -> None:
    config = OptimizeConfig.from_dict({"cache_enabled": False})
    app, proxy, _t = _make_app(config, HookChain.empty())

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.post("/v1/chat/completions", json=_sample_req(), headers=_openai_headers())

    assert resp.status_code == 200
    body = resp.json()
    assert body["id"] == "chatcmpl-1"
    await proxy.http_client.aclose()


@pytest.mark.asyncio
async def test_chat_completions_cache_miss_then_hit() -> None:
    """First call: upstream; second call: cache hit (no upstream)."""
    upstream_calls: list = []
    _cache: dict = {}

    async def handler(request):
        upstream_calls.append(1)
        return httpx.Response(200, json=_sample_resp())

    class TrackHook:
        def check(self, ctx):
            if "k1" in _cache:
                return CachedResponse(hit=True, data=_cache["k1"], cache_key="k1", ttl_seconds=300)
            return CachedResponse(hit=False, data=None, cache_key="k1", ttl_seconds=300)
        def store(self, ctx, resp):
            _cache["k1"] = b'{"id": "chatcmpl-1", "object": "chat.completion", "choices": [{"index": 0, "message": {"role": "assistant", "content": "hello"}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 10, "completion_tokens": 5}}'
        def on_hit(self, ctx, resp, ts):
            pass
        def on_miss(self, ctx):
            pass

    config = OptimizeConfig.from_dict({"cache_enabled": True})
    app, proxy, _t = _make_app(config, HookChain(cache=TrackHook(), compress=None), handler=handler)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        await client.post("/v1/chat/completions", json=_sample_req(), headers=_openai_headers())
        await client.post("/v1/chat/completions", json=_sample_req(), headers=_openai_headers())

    assert len(upstream_calls) == 1
    await proxy.http_client.aclose()


@pytest.mark.asyncio
async def test_chat_completions_tool_calls_skipped() -> None:
    """tools= present → skip cache, pass-through every time."""
    upstream_calls: list = []

    def handler(request):
        upstream_calls.append(1)
        return httpx.Response(200, json=_sample_resp())

    config = OptimizeConfig.from_dict({"cache_enabled": True})
    app, proxy, _t = _make_app(config, HookChain.empty(), handler=handler)

    body = {**_sample_req(), "tools": [{"name": "bash"}]}

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        await client.post("/v1/chat/completions", json=body, headers=_openai_headers())
        await client.post("/v1/chat/completions", json=body, headers=_openai_headers())

    assert len(upstream_calls) == 2
    await proxy.http_client.aclose()


@pytest.mark.asyncio
async def test_chat_completions_streaming_passthrough() -> None:
    config = OptimizeConfig.from_dict({"cache_enabled": True})

    def handler(request):
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=b"data: hello\n\n",
        )

    app, proxy, _t = _make_app(config, HookChain.empty(), handler=handler)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        body = {**_sample_req(), "stream": True}
        async with client.stream("POST", "/v1/chat/completions", json=body, headers=_openai_headers()) as resp:
            assert resp.status_code == 200
            chunks = []
            async for c in resp.aiter_bytes():
                if c:
                    chunks.append(c)

    assert b"hello" in b"".join(chunks)
    await proxy.http_client.aclose()


@pytest.mark.asyncio
async def test_chat_completions_empty_key_skips_store() -> None:
    stored: list = []

    class EmptyKeyHook:
        def check(self, ctx):
            return CachedResponse(hit=False, data=None, cache_key="", ttl_seconds=0)
        def store(self, ctx, resp):
            stored.append(True)
        def on_hit(self, ctx, resp, ts):
            pass
        def on_miss(self, ctx):
            pass

    config = OptimizeConfig.from_dict({"cache_enabled": True})
    app, proxy, _t = _make_app(config, HookChain(cache=EmptyKeyHook(), compress=None))

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.post("/v1/chat/completions", json=_sample_req(), headers=_openai_headers())

    assert resp.status_code == 200
    assert stored == []
    await proxy.http_client.aclose()


@pytest.mark.asyncio
async def test_embeddings_passthrough() -> None:
    config = OptimizeConfig.from_dict({})

    def handler(request):
        return httpx.Response(200, json={"object": "list", "data": [{"embedding": [0.1, 0.2]}]})

    app, proxy, _t = _make_app(config, HookChain.empty(), handler=handler)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.post(
            "/v1/embeddings",
            json={"model": "text-embedding-3-small", "input": "hello"},
            headers=_openai_headers(),
        )

    assert resp.status_code == 200
    body = resp.json()
    assert body["object"] == "list"
    await proxy.http_client.aclose()


@pytest.mark.asyncio
async def test_chat_completions_invalid_json() -> None:
    """Invalid JSON body → raw forward to upstream."""
    upstream_calls: list = []

    def handler(request):
        upstream_calls.append(1)
        return httpx.Response(200, json=_sample_resp())

    config = OptimizeConfig.from_dict({})
    app, proxy, _t = _make_app(config, HookChain.empty(), handler=handler)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.post(
            "/v1/chat/completions",
            content=b"not-valid-json",
            headers={**_openai_headers(), "content-type": "application/json"},
        )

    assert resp.status_code == 200
    assert len(upstream_calls) == 1
    await proxy.http_client.aclose()


@pytest.mark.asyncio
async def test_chat_completions_non_2xx_not_cached() -> None:
    """Non-200 from upstream must not trigger cache.store()."""
    stored: list = []

    class TrackHook:
        def check(self, ctx):
            return CachedResponse(hit=False, data=None, cache_key="k1", ttl_seconds=300)
        def store(self, ctx, resp):
            stored.append(True)
        def on_hit(self, ctx, resp, ts):
            pass
        def on_miss(self, ctx):
            pass

    def handler(request):
        return httpx.Response(401, json={"error": {"message": "auth"}})

    config = OptimizeConfig.from_dict({"cache_enabled": True})
    app, proxy, _t = _make_app(config, HookChain(cache=TrackHook(), compress=None), handler=handler)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.post("/v1/chat/completions", json=_sample_req(), headers=_openai_headers())

    assert resp.status_code == 401
    assert stored == []
    await proxy.http_client.aclose()


@pytest.mark.asyncio
async def test_embeddings_invalid_json() -> None:
    """Invalid JSON body for embeddings → raw forward."""
    def handler(request):
        return httpx.Response(200, json={"object": "list", "data": []})

    config = OptimizeConfig.from_dict({})
    app, proxy, _t = _make_app(config, HookChain.empty(), handler=handler)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.post(
            "/v1/embeddings",
            content=b"bad json",
            headers={**_openai_headers(), "content-type": "application/json"},
        )

    assert resp.status_code == 200
    await proxy.http_client.aclose()
