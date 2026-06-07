"""More _helpers coverage — _fail_open_forward, _stream_forward, _safe_cache_*."""

from __future__ import annotations

from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from fastapi.requests import Request
from fastapi.responses import Response

from superlocalmemory.optimize.proxy._helpers import (
    _fail_open_forward,
    _filter_response_headers,
    _safe_cache_check,
    _safe_cache_hit_callbacks,
    _safe_cache_store,
    _stream_forward,
)
from superlocalmemory.optimize.proxy.lifecycle import (
    CachedResponse,
    HookChain,
    ProviderResponse,
    ProxyRequest,
)


# ---- _filter_response_headers ----

def test_filter_response_headers_handles_case_insensitive() -> None:
    h = _filter_response_headers({
        "Content-Type": "application/json",
        "Connection": "keep-alive",
        "X-Request-ID": "abc",
    })
    assert "Content-Type" in h
    assert "Connection" not in h
    assert "X-Request-ID" in h


# ---- _safe_cache_* fail-open ----

@pytest.mark.asyncio
async def test_safe_cache_check_returns_miss_on_none() -> None:
    """check() returns None → safe miss (NOT raised)."""
    class Hook:
        def check(self, ctx):
            return None
    hooks = HookChain(cache=Hook(), compress=None)
    ctx = ProxyRequest(
        provider="anthropic", method="POST", path="/v1/messages",
        headers={}, body={}, body_bytes=b"", request_id="r1",
        stream=False, has_tools=False,
    )
    result = await _safe_cache_check(hooks, ctx)
    assert result.hit is False
    assert result.cache_key == ""


@pytest.mark.asyncio
async def test_safe_cache_check_swallows_exception() -> None:
    class Hook:
        def check(self, ctx):
            raise RuntimeError("boom")
    hooks = HookChain(cache=Hook(), compress=None)
    ctx = ProxyRequest(
        provider="anthropic", method="POST", path="/v1/messages",
        headers={}, body={}, body_bytes=b"", request_id="r1",
        stream=False, has_tools=False,
    )
    result = await _safe_cache_check(hooks, ctx)
    assert result.hit is False


@pytest.mark.asyncio
async def test_safe_cache_store_swallows_exception() -> None:
    class Hook:
        def store(self, ctx, resp):
            raise RuntimeError("boom")
        def on_miss(self, ctx):
            raise RuntimeError("boom")
    hooks = HookChain(cache=Hook(), compress=None)
    ctx = ProxyRequest(
        provider="anthropic", method="POST", path="/v1/messages",
        headers={}, body={}, body_bytes=b"", request_id="r1",
        stream=False, has_tools=False,
    )
    resp = ProviderResponse(
        modified=False, body={}, body_bytes=b"x",
        tokens_before=0, tokens_after=0, strategy="none",
    )
    # Must not raise
    await _safe_cache_store(hooks, ctx, resp)


@pytest.mark.asyncio
async def test_safe_cache_hit_callbacks_swallows_exception() -> None:
    class Hook:
        def on_hit(self, ctx, resp, tokens_saved):
            raise RuntimeError("boom")
    hooks = HookChain(cache=Hook(), compress=None)
    ctx = ProxyRequest(
        provider="anthropic", method="POST", path="/v1/messages",
        headers={}, body={}, body_bytes=b"", request_id="r1",
        stream=False, has_tools=False,
    )
    await _safe_cache_hit_callbacks(hooks, ctx, b"{}", 0)


# ---- _fail_open_forward ----

class _NullClientProxy:
    http_client = None


@pytest.mark.asyncio
async def test_fail_open_forward_returns_502_when_http_client_is_none() -> None:
    """http_client=None → synthetic 502."""
    proxy = _NullClientProxy()
    app = FastAPI()

    @app.post("/test")
    async def _route(request: Request) -> Response:
        return await _fail_open_forward(proxy, request, "http://example.com")

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.post("/test", json={})
    assert resp.status_code == 502


class _BrokenClientProxy:
    """httpx client that raises on every request."""

    def __init__(self):
        from superlocalmemory.optimize.proxy._helpers import _MockTransport
        self.http_client = httpx.AsyncClient(transport=_MockTransport(
            handler=lambda r: (_ for _ in ()).throw(RuntimeError("boom"))
        ))


@pytest.mark.asyncio
async def test_fail_open_forward_handles_upstream_exception() -> None:
    """Upstream exception → 502 (no propagate)."""
    proxy = _BrokenClientProxy()
    app = FastAPI()

    @app.post("/test")
    async def _route(request: Request) -> Response:
        return await _fail_open_forward(proxy, request, "http://example.com")

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.post("/test", json={})
    assert resp.status_code == 502
    await proxy.http_client.aclose()


# ---- _stream_forward ----

@pytest.mark.asyncio
async def test_stream_forward_returns_502_when_http_client_is_none() -> None:
    proxy = _NullClientProxy()
    result = await _stream_forward(
        proxy, "slm_1_000001", {}, b"", "http://example.com",
    )
    assert isinstance(result, Response)
    assert result.status_code == 502


@pytest.mark.asyncio
async def test_stream_forward_yields_chunks() -> None:
    """Streaming passes through chunks without buffering."""
    from superlocalmemory.optimize.proxy._helpers import _MockTransport

    async def handler(request):
        async def gen():
            yield b"chunk1"
            yield b"chunk2"
        # httpx.Response doesn't accept a generator directly. Instead, return
        # a Response with pre-buffered content; the stream-forward generator
        # path is tested via the response content.aiter_bytes() (MockTransport
        # buffers). For end-to-end streaming semantics, see the
        # anthropic_surface SSE test.
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=b"chunk1chunk2",
        )

    proxy = _NullClientProxy()
    proxy.http_client = httpx.AsyncClient(transport=_MockTransport(handler=handler))

    result = await _stream_forward(
        proxy, "slm_1_000001", {}, b"", "http://example.com",
    )
    from fastapi.responses import StreamingResponse
    assert isinstance(result, StreamingResponse)

    chunks = []
    async for c in result.body_iterator:
        if isinstance(c, bytes):
            chunks.append(c)
        else:
            chunks.append(c.encode())

    body = b"".join(chunks)
    assert b"chunk1" in body
    assert b"chunk2" in body
    await proxy.http_client.aclose()
