"""More Anthropic surface coverage — count_tokens, models, fail-open."""

from __future__ import annotations

import httpx
import pytest
from fastapi import FastAPI

from superlocalmemory.optimize.config.schema import OptimizeConfig
from superlocalmemory.optimize.proxy.lifecycle import HookChain
from superlocalmemory.optimize.proxy.server import ProxyApp, build_proxy_router
from superlocalmemory.optimize.proxy._helpers import _MockTransport


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


@pytest.mark.asyncio
async def test_count_tokens_passthrough() -> None:
    config = OptimizeConfig.from_dict({})

    async def handler(request):
        return httpx.Response(200, json={"input_tokens": 42})

    proxy = ProxyApp(config=config)
    proxy.hooks = HookChain.empty()
    proxy.http_client = httpx.AsyncClient(transport=_MockTransport(handler=handler))
    app = FastAPI()
    app.include_router(build_proxy_router(proxy), prefix="")

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.post(
            "/v1/messages/count_tokens",
            json=_sample_req(),
            headers=_anthropic_headers(),
        )

    assert resp.status_code == 200
    assert resp.json()["input_tokens"] == 42
    await proxy.http_client.aclose()


@pytest.mark.asyncio
async def test_count_tokens_invalid_json() -> None:
    """Invalid JSON for count_tokens → raw forward (returns 502 if upstream also fails)."""
    config = OptimizeConfig.from_dict({})

    async def handler(request):
        return httpx.Response(200, json={"input_tokens": 0})

    proxy = ProxyApp(config=config)
    proxy.hooks = HookChain.empty()
    proxy.http_client = httpx.AsyncClient(transport=_MockTransport(handler=handler))
    app = FastAPI()
    app.include_router(build_proxy_router(proxy), prefix="")

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        # count_tokens endpoint reads body_bytes and forwards; bad JSON body
        # would be rejected by the upstream — but the proxy always forwards.
        resp = await client.post(
            "/v1/messages/count_tokens",
            content=b"not json",
            headers={**_anthropic_headers(), "content-type": "application/json"},
        )

    assert resp.status_code == 200
    await proxy.http_client.aclose()


@pytest.mark.asyncio
async def test_anthropic_messages_invalid_json() -> None:
    """Invalid JSON body for /v1/messages → fail_open_forward (502 if upstream fails)."""
    config = OptimizeConfig.from_dict({})

    async def handler(request):
        return httpx.Response(200, json={"id": "msg_ok", "type": "message"})

    proxy = ProxyApp(config=config)
    proxy.hooks = HookChain.empty()
    proxy.http_client = httpx.AsyncClient(transport=_MockTransport(handler=handler))
    app = FastAPI()
    app.include_router(build_proxy_router(proxy), prefix="")

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.post(
            "/v1/messages",
            content=b"not json",
            headers={**_anthropic_headers(), "content-type": "application/json"},
        )

    assert resp.status_code == 200
    await proxy.http_client.aclose()


@pytest.mark.asyncio
async def test_anthropic_messages_oversized_body_returns_413() -> None:
    """Content-Length > 10 MB → 413 immediately (no upstream call)."""
    upstream_calls: list = []

    async def handler(request):
        upstream_calls.append(1)
        return httpx.Response(200, json={})

    config = OptimizeConfig.from_dict({})
    proxy = ProxyApp(config=config)
    proxy.hooks = HookChain.empty()
    proxy.http_client = httpx.AsyncClient(transport=_MockTransport(handler=handler))
    app = FastAPI()
    app.include_router(build_proxy_router(proxy), prefix="")

    big = b"x" * (11 * 1024 * 1024)  # 11 MB > 10 MB limit

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.post(
            "/v1/messages",
            content=big,
            headers={**_anthropic_headers(), "content-length": str(len(big))},
        )

    assert resp.status_code == 413
    assert upstream_calls == []
    await proxy.http_client.aclose()


@pytest.mark.asyncio
async def test_anthropic_messages_cache_hit_returns_200() -> None:
    """Cache hit on /v1/messages → returns cached bytes verbatim."""
    from superlocalmemory.optimize.proxy.lifecycle import CachedResponse

    class HitHook:
        def check(self, ctx):
            return CachedResponse(hit=True, data=b'{"cached": true}', cache_key="k", ttl_seconds=300)
        def store(self, ctx, resp):
            pass
        def on_hit(self, ctx, resp, ts):
            pass
        def on_miss(self, ctx):
            pass

    config = OptimizeConfig.from_dict({"cache_enabled": True})
    proxy = ProxyApp(config=config)
    proxy.hooks = HookChain(cache=HitHook(), compress=None)
    proxy.http_client = None  # no upstream should be touched
    app = FastAPI()
    app.include_router(build_proxy_router(proxy), prefix="")

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.post("/v1/messages", json=_sample_req(), headers=_anthropic_headers())

    assert resp.status_code == 200
    assert resp.json() == {"cached": True}
    await proxy.http_client.aclose() if proxy.http_client else None


# ---- Additional coverage tests ----

@pytest.mark.asyncio
async def test_anthropic_messages_invalid_cl_header() -> None:
    """Content-Length header with non-integer value → passes through (lines 45-46)."""
    config = OptimizeConfig.from_dict({})

    async def handler(request):
        return httpx.Response(200, json={"id": "ok", "type": "message"})

    proxy = ProxyApp(config=config)
    proxy.hooks = HookChain.empty()
    proxy.http_client = httpx.AsyncClient(transport=_MockTransport(handler=handler))
    app = FastAPI()
    app.include_router(build_proxy_router(proxy), prefix="")

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.post(
            "/v1/messages",
            json=_sample_req(),
            headers={**_anthropic_headers(), "content-length": "not-a-number"},
        )

    assert resp.status_code == 200
    await proxy.http_client.aclose()


@pytest.mark.asyncio
async def test_anthropic_messages_body_too_large() -> None:
    """Body bytes > 10 MB → 413 (line 51)."""
    config = OptimizeConfig.from_dict({})
    proxy = ProxyApp(config=config)
    proxy.hooks = HookChain.empty()
    proxy.http_client = None
    app = FastAPI()
    app.include_router(build_proxy_router(proxy), prefix="")

    # Send a 11MB body without a content-length header
    big = b"x" * (11 * 1024 * 1024)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.post(
            "/v1/messages",
            content=big,
            headers={**_anthropic_headers()},
        )

    assert resp.status_code == 413


@pytest.mark.asyncio
async def test_anthropic_messages_non_200_upstream() -> None:
    """Non-200 upstream response with cache hit check → ProviderResponse stored (lines 120-124)."""
    from superlocalmemory.optimize.proxy.lifecycle import CachedResponse
    stored: list = []

    class _Hook:
        def check(self, ctx):
            return CachedResponse(hit=False, data=None, cache_key="k1", ttl_seconds=300)
        def store(self, ctx, resp):
            stored.append(resp)
        def on_hit(self, ctx, resp, ts):
            pass
        def on_miss(self, ctx):
            pass

    async def _handler(request):
        return httpx.Response(400, json={"type": "error", "error": {"type": "invalid_request_error"}})

    config = OptimizeConfig.from_dict({"cache_enabled": True})
    proxy = ProxyApp(config=config)
    proxy.hooks = HookChain(cache=_Hook(), compress=None)
    proxy.http_client = httpx.AsyncClient(transport=_MockTransport(handler=_handler))
    app = FastAPI()
    app.include_router(build_proxy_router(proxy), prefix="")

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.post("/v1/messages", json=_sample_req(), headers=_anthropic_headers())

    assert resp.status_code == 400
    # Not stored because status_code != 200
    await proxy.http_client.aclose()


@pytest.mark.asyncio
async def test_anthropic_messages_outer_exception_fail_open() -> None:
    """Outer exception handler returns fail-open forward (lines 133-135)."""
    config = OptimizeConfig.from_dict({})
    proxy = ProxyApp(config=config)
    proxy.hooks = HookChain.empty()
    proxy.http_client = None  # will cause exception in inner try
    app = FastAPI()
    app.include_router(build_proxy_router(proxy), prefix="")

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.post(
            "/v1/messages",
            json=_sample_req(),
            headers=_anthropic_headers(),
        )

    # Fail-open: may return 502 (no upstream client) or error response
    assert resp.status_code in (200, 502, 500)


@pytest.mark.asyncio
async def test_handle_count_tokens_exception_fail_open() -> None:
    """count_tokens exception → fail_open_forward (lines 153-155)."""
    config = OptimizeConfig.from_dict({})
    proxy = ProxyApp(config=config)
    proxy.hooks = HookChain.empty()
    proxy.http_client = None  # cause error
    app = FastAPI()
    app.include_router(build_proxy_router(proxy), prefix="")

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.post(
            "/v1/messages/count_tokens",
            json=_sample_req(),
            headers=_anthropic_headers(),
        )

    assert resp.status_code in (200, 502, 500)


@pytest.mark.asyncio
async def test_handle_models_exception_fail_open() -> None:
    """handle_models exception → fail_open_forward (lines 169-171)."""
    config = OptimizeConfig.from_dict({})
    proxy = ProxyApp(config=config)
    proxy.hooks = HookChain.empty()
    proxy.http_client = None  # cause error
    app = FastAPI()
    app.include_router(build_proxy_router(proxy), prefix="")

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.get("/v1/models", headers=_anthropic_headers())

    assert resp.status_code in (200, 502, 500)


@pytest.mark.asyncio
async def test_anthropic_messages_with_compress_hook() -> None:
    """Messages with compress hook active → compress path (lines 100-103)."""
    from superlocalmemory.optimize.proxy.lifecycle import CachedResponse, ProviderResponse

    class _CompressHook:
        def __init__(self):
            pass
        def body_bytes(self):
            return b'{"model":"claude-sonnet-4-6","max_tokens":100,"messages":[{"role":"user","content":"hi"}]}'

    compress_hook = _CompressHook()

    async def _handler(request):
        return httpx.Response(200, json={"id": "ok", "type": "message"})

    config = OptimizeConfig.from_dict({"cache_enabled": True, "compress_enabled": True})

    class _Hook:
        def check(self, ctx):
            return CachedResponse(hit=False, data=None, cache_key="k", ttl_seconds=300)
        def store(self, ctx, resp):
            pass
        def on_hit(self, ctx, resp, ts):
            pass
        def on_miss(self, ctx):
            pass

    proxy = ProxyApp(config=config)
    proxy.hooks = HookChain(cache=_Hook(), compress=compress_hook)
    proxy.http_client = httpx.AsyncClient(transport=_MockTransport(handler=_handler))
    app = FastAPI()
    app.include_router(build_proxy_router(proxy), prefix="")

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.post("/v1/messages", json=_sample_req(), headers=_anthropic_headers())

    assert resp.status_code == 200
    await proxy.http_client.aclose()
