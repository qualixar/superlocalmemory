"""More Gemini surface coverage — gemini native + openai-compat."""

from __future__ import annotations

import httpx
import pytest
from fastapi import FastAPI

from superlocalmemory.optimize.config.schema import OptimizeConfig
from superlocalmemory.optimize.proxy.lifecycle import HookChain
from superlocalmemory.optimize.proxy.server import ProxyApp, build_proxy_router
from superlocalmemory.optimize.proxy._helpers import _MockTransport


@pytest.mark.asyncio
async def test_gemini_native_streaming_injects_alt_sse() -> None:
    """streamGenerateContent → alt=sse injected, query param filtered."""
    captured_url: list = []

    async def handler(request):
        captured_url.append(str(request.url))
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=b"data: ok\n\n",
        )

    config = OptimizeConfig.from_dict({})
    proxy = ProxyApp(config=config)
    proxy.hooks = HookChain.empty()
    proxy.http_client = httpx.AsyncClient(transport=_MockTransport(handler=handler))
    app = FastAPI()
    app.include_router(build_proxy_router(proxy), prefix="")

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        # Inject attacker query param "key" which must be dropped
        resp = await client.post(
            "/v1beta/models/gemini-3.5-flash:streamGenerateContent?key=ATTACKER&pagesize=10",
            json={"contents": [{"role": "user", "parts": [{"text": "hi"}]}]},
            headers={"x-goog-api-key": "real-key", "content-type": "application/json"},
        )

    assert resp.status_code == 200
    assert len(captured_url) == 1
    # alt=sse must be in the upstream URL
    assert "alt=sse" in captured_url[0]
    # "key=" must NOT be in the upstream URL (CWE-918: query injection guard)
    assert "key=" not in captured_url[0]
    await proxy.http_client.aclose()


@pytest.mark.asyncio
async def test_gemini_openai_compat_passthrough() -> None:
    """POST /v1beta/openai/chat/completions mirrors path to upstream."""
    config = OptimizeConfig.from_dict({})

    async def handler(request):
        return httpx.Response(200, json={"choices": []})

    proxy = ProxyApp(config=config)
    proxy.hooks = HookChain.empty()
    proxy.http_client = httpx.AsyncClient(transport=_MockTransport(handler=handler))
    app = FastAPI()
    app.include_router(build_proxy_router(proxy), prefix="")

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.post(
            "/v1beta/openai/chat/completions",
            json={"model": "gemini-3.5-flash", "messages": []},
            headers={"authorization": "Bearer test", "content-type": "application/json"},
        )

    assert resp.status_code == 200
    await proxy.http_client.aclose()


@pytest.mark.asyncio
async def test_gemini_openai_compat_models_get() -> None:
    """GET /v1beta/openai/models passthrough."""
    config = OptimizeConfig.from_dict({})

    async def handler(request):
        return httpx.Response(200, json={"data": []})

    proxy = ProxyApp(config=config)
    proxy.hooks = HookChain.empty()
    proxy.http_client = httpx.AsyncClient(transport=_MockTransport(handler=handler))
    app = FastAPI()
    app.include_router(build_proxy_router(proxy), prefix="")

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.get("/v1beta/openai/models")

    assert resp.status_code == 200
    await proxy.http_client.aclose()


@pytest.mark.asyncio
async def test_gemini_native_count_tokens() -> None:
    """countTokens is a valid Gemini method."""
    config = OptimizeConfig.from_dict({})

    async def handler(request):
        return httpx.Response(200, json={"totalTokens": 42})

    proxy = ProxyApp(config=config)
    proxy.hooks = HookChain.empty()
    proxy.http_client = httpx.AsyncClient(transport=_MockTransport(handler=handler))
    app = FastAPI()
    app.include_router(build_proxy_router(proxy), prefix="")

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.post(
            "/v1beta/models/gemini-3.5-flash:countTokens",
            json={"contents": [{"role": "user", "parts": [{"text": "hi"}]}]},
            headers={"x-goog-api-key": "k", "content-type": "application/json"},
        )

    assert resp.status_code == 200
    assert resp.json()["totalTokens"] == 42
    await proxy.http_client.aclose()


@pytest.mark.asyncio
async def test_gemini_native_fail_open_on_upstream_error() -> None:
    """Upstream exception → 502 from fail_open_forward."""
    from superlocalmemory.optimize.proxy._helpers import _MockTransport

    async def handler(request):
        raise RuntimeError("boom")

    config = OptimizeConfig.from_dict({})
    proxy = ProxyApp(config=config)
    proxy.hooks = HookChain.empty()
    proxy.http_client = httpx.AsyncClient(transport=_MockTransport(handler=handler))
    app = FastAPI()
    app.include_router(build_proxy_router(proxy), prefix="")

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.post(
            "/v1beta/models/gemini-3.5-flash:countTokens",
            json={"contents": [{"role": "user", "parts": [{"text": "hi"}]}]},
            headers={"x-goog-api-key": "k", "content-type": "application/json"},
        )

    assert resp.status_code == 502
    await proxy.http_client.aclose()
