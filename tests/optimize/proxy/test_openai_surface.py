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
from superlocalmemory.optimize.proxy.openai_surface import (
    _parse_openai_sse_to_json,
    _openai_sse_from_cached_json,
)


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


# ============================================================
# Unit tests for _parse_openai_sse_to_json (v3.6.3)
# ============================================================

def _make_openai_sse(content: str = "Hello!", completion_id: str = "chatcmpl-abc", model: str = "gpt-4o") -> bytes:
    """Build a minimal valid OpenAI SSE stream with a text response."""
    chunks = [
        f'data: {{"id":"{completion_id}","object":"chat.completion.chunk","created":1234,"model":"{model}","choices":[{{"index":0,"delta":{{"role":"assistant","content":""}},"finish_reason":null}}]}}\n\n',
        f'data: {{"id":"{completion_id}","object":"chat.completion.chunk","created":1234,"model":"{model}","choices":[{{"index":0,"delta":{{"content":"{content}"}},"finish_reason":null}}]}}\n\n',
        f'data: {{"id":"{completion_id}","object":"chat.completion.chunk","created":1234,"model":"{model}","choices":[{{"index":0,"delta":{{}},"finish_reason":"stop"}}]}}\n\n',
        "data: [DONE]\n\n",
    ]
    return "".join(chunks).encode()


def test_parse_openai_sse_to_json_basic() -> None:
    """Valid complete stream → parsed chat.completion JSON."""
    sse = _make_openai_sse("Hello world")
    result = _parse_openai_sse_to_json(sse)
    assert result is not None
    data = json.loads(result)
    assert data["object"] == "chat.completion"
    assert data["id"] == "chatcmpl-abc"
    assert data["choices"][0]["message"]["content"] == "Hello world"
    assert data["choices"][0]["finish_reason"] == "stop"


def test_parse_openai_sse_to_json_missing_done_returns_none() -> None:
    """Incomplete stream (no [DONE]) → None."""
    sse = (
        b'data: {"id":"chatcmpl-x","object":"chat.completion.chunk","created":1,"model":"gpt-4o",'
        b'"choices":[{"index":0,"delta":{"content":"hi"},"finish_reason":null}]}\n\n'
    )
    assert _parse_openai_sse_to_json(sse) is None


def test_parse_openai_sse_to_json_tool_calls_cached() -> None:
    """Stream with tool_calls → valid JSON with tool_calls preserved (BUG-FIX v3.6.4).

    Previously returned None for tool_call responses, meaning OpenAI-compatible
    tool-using clients (Codex CLI, Antigravity) were NEVER cached. Now the
    assembled JSON includes the tool_calls array so _openai_sse_from_cached_json
    can replay them correctly.
    """
    sse = (
        b'data: {"id":"chatcmpl-t","object":"chat.completion.chunk","created":1,"model":"gpt-4o",'
        b'"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call1","type":"function",'
        b'"function":{"name":"bash","arguments":"{\\"command\\":"}}]},"finish_reason":null}]}\n\n'
        b'data: {"id":"chatcmpl-t","object":"chat.completion.chunk","created":1,"model":"gpt-4o",'
        b'"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":" \\"ls -la\\"}"}}]},'
        b'"finish_reason":"tool_calls"}]}\n\n'
        b'data: [DONE]\n\n'
    )
    result = _parse_openai_sse_to_json(sse)
    assert result is not None, "tool_call responses must now be cached (v3.6.4 fix)"
    parsed = json.loads(result)
    assert parsed["object"] == "chat.completion"
    assert parsed["id"] == "chatcmpl-t"
    choices = parsed["choices"]
    assert len(choices) == 1
    msg = choices[0]["message"]
    assert msg["content"] is None
    assert "tool_calls" in msg
    assert len(msg["tool_calls"]) == 1
    assert msg["tool_calls"][0]["function"]["name"] == "bash"
    assert "ls -la" in msg["tool_calls"][0]["function"]["arguments"]


def test_parse_openai_sse_to_json_empty_bytes_returns_none() -> None:
    assert _parse_openai_sse_to_json(b"") is None


def test_parse_openai_sse_to_json_usage_captured() -> None:
    """Usage block in last chunk is captured."""
    sse = (
        b'data: {"id":"chatcmpl-u","object":"chat.completion.chunk","created":1,"model":"gpt-4o",'
        b'"choices":[{"index":0,"delta":{"content":"ok"},"finish_reason":"stop"}],'
        b'"usage":{"prompt_tokens":5,"completion_tokens":2,"total_tokens":7}}\n\n'
        b'data: [DONE]\n\n'
    )
    result = _parse_openai_sse_to_json(sse)
    assert result is not None
    data = json.loads(result)
    assert data["usage"]["prompt_tokens"] == 5
    assert data["usage"]["completion_tokens"] == 2


# ============================================================
# Unit tests for _openai_sse_from_cached_json (v3.6.3)
# ============================================================

def test_openai_sse_from_cached_json_roundtrip() -> None:
    """Parse SSE → JSON → back to SSE StreamingResponse (no crash)."""
    sse_in = _make_openai_sse("Roundtrip works")
    cached = _parse_openai_sse_to_json(sse_in)
    assert cached is not None
    resp = _openai_sse_from_cached_json(cached)
    assert resp is not None
    assert resp.media_type == "text/event-stream"


def test_openai_sse_from_cached_json_bad_json_returns_none() -> None:
    assert _openai_sse_from_cached_json(b"not json") is None


def test_openai_sse_from_cached_json_wrong_object_returns_none() -> None:
    data = json.dumps({"object": "embedding", "data": []}).encode()
    assert _openai_sse_from_cached_json(data) is None


@pytest.mark.asyncio
async def test_streaming_cache_miss_store_hit_cycle() -> None:
    """Streaming: call 1 hits upstream and stores; call 2 is served from cache as SSE."""
    upstream_calls: list = []
    _cache: dict[str, bytes] = {}

    sse_body = _make_openai_sse("Cached SSE response")

    def handler(request):
        upstream_calls.append(1)
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=sse_body,
        )

    class TrackHook:
        def check(self, ctx):
            if "k1" in _cache:
                return CachedResponse(hit=True, data=_cache["k1"], cache_key="k1", ttl_seconds=300)
            return CachedResponse(hit=False, data=None, cache_key="k1", ttl_seconds=300)
        def store(self, ctx, resp):
            _cache["k1"] = resp.body_bytes
        def on_hit(self, ctx, resp, ts):
            pass
        def on_miss(self, ctx):
            pass

    config = OptimizeConfig.from_dict({"cache_enabled": True})
    app, proxy, _t = _make_app(config, HookChain(cache=TrackHook(), compress=None), handler=handler)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        body = {**_sample_req(), "stream": True}
        # Call 1 — upstream
        async with client.stream("POST", "/v1/chat/completions", json=body, headers=_openai_headers()) as r1:
            chunks1 = [c async for c in r1.aiter_bytes() if c]

        # Call 2 — should hit cache (no new upstream call)
        async with client.stream("POST", "/v1/chat/completions", json=body, headers=_openai_headers()) as r2:
            chunks2 = [c async for c in r2.aiter_bytes() if c]

    assert len(upstream_calls) == 1, "Second streaming call must be served from cache"
    assert b"[DONE]" in b"".join(chunks2), "Cache-replayed SSE must end with [DONE]"
    assert b"Cached SSE response" in b"".join(chunks2)

    await proxy.http_client.aclose()
