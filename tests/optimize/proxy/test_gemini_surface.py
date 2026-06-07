"""LLD-01 §5.6 — Gemini surface tests (SSRF guard)."""

from __future__ import annotations

import httpx
import pytest
from fastapi import FastAPI

from superlocalmemory.optimize.config.schema import OptimizeConfig
from superlocalmemory.optimize.proxy.lifecycle import HookChain
from superlocalmemory.optimize.proxy.server import ProxyApp, build_proxy_router
from superlocalmemory.optimize.proxy.gemini_surface import _validate_gemini_path


def test_validate_gemini_path_valid() -> None:
    assert _validate_gemini_path("models/gemini-3.5-flash:generateContent") is True
    assert _validate_gemini_path("models/gemini-3.5-flash:streamGenerateContent") is True
    assert _validate_gemini_path("models/gemini-3.5-flash:countTokens") is True


def test_validate_gemini_path_rejects_traversal() -> None:
    assert _validate_gemini_path("models/../../etc/passwd:generateContent") is False


def test_validate_gemini_path_rejects_unknown_method() -> None:
    assert _validate_gemini_path("models/gemini-flash:UNKNOWN_METHOD") is False


def test_validate_gemini_path_rejects_overlong() -> None:
    assert _validate_gemini_path("models/" + "A" * 200 + ":generateContent") is False


def test_validate_gemini_path_rejects_path_traversal() -> None:
    assert _validate_gemini_path("../internal:generateContent") is False


@pytest.mark.asyncio
async def test_gemini_ssrf_path_rejected() -> None:
    """Malicious paths must be rejected with HTTP 400."""
    from superlocalmemory.optimize.proxy._helpers import _MockTransport  # noqa: F401
    config = OptimizeConfig.from_dict({})
    proxy = ProxyApp(config=config)
    proxy.hooks = HookChain.empty()

    async def _handler(request):
        return httpx.Response(200, json={"candidates": []})

    proxy.http_client = httpx.AsyncClient(transport=_MockTransport(handler=_handler))
    app = FastAPI()
    app.include_router(build_proxy_router(proxy), prefix="")

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        # These paths all REACH the route (FastAPI :path converter allows them)
        # but must be rejected by the SSRF guard with 400.
        for path in [
            "gemini-flash:UNKNOWN_METHOD",
            "models/gemini-flash:UNKNOWN_METHOD",
            "models/" + "A" * 200 + ":generateContent",
            "models/gemini-flash:EXEC",  # not in allowlist
        ]:
            resp = await client.post(
                f"/v1beta/models/{path}",
                json={"contents": [{"role": "user", "parts": [{"text": "hi"}]}]},
                headers={"x-goog-api-key": "test-key", "content-type": "application/json"},
            )
            assert resp.status_code == 400, (
                f"Expected 400 for malicious path {path!r}, got {resp.status_code}"
            )

    await proxy.http_client.aclose()


@pytest.mark.asyncio
async def test_gemini_valid_path_passes() -> None:
    """Valid Gemini path must reach upstream."""
    from superlocalmemory.optimize.proxy._helpers import _MockTransport  # noqa: F401
    config = OptimizeConfig.from_dict({})
    proxy = ProxyApp(config=config)
    proxy.hooks = HookChain.empty()

    async def _handler(request):
        return httpx.Response(200, json={"candidates": [{"content": {}}]})

    proxy.http_client = httpx.AsyncClient(transport=_MockTransport(handler=_handler))
    app = FastAPI()
    app.include_router(build_proxy_router(proxy), prefix="")

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.post(
            "/v1beta/models/gemini-3.5-flash:generateContent",
            json={"contents": [{"role": "user", "parts": [{"text": "hi"}]}]},
            headers={"x-goog-api-key": "test-key", "content-type": "application/json"},
        )

    assert resp.status_code == 200
    await proxy.http_client.aclose()
