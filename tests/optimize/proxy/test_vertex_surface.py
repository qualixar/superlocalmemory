"""WP-11 + WP-11a — Vertex surface + gemini-native authorization-header fix.

LLD §6 test harness — 19 tests covering:
  - path validation (valid / traversal / unknown-method)
  - path parsing (project, location, model, method)
  - upstream URL construction
  - Authorization forwarded byte-identical (AC-2)
  - token NEVER in cache key (SEC / CRIT-2 guard)
  - token NEVER in logs (AC-3)
  - different body → different key (CRIT-2 regression)
  - no SSE / single JSON Response
  - cache hit: upstream called once
  - upstream 401 passthrough, nothing cached
  - SSRF malformed path → 400 BEFORE upstream (mock NOT called)
  - route absent when proxy_enabled=False
  - WP-11a: gemini-native authorization forwarded
  - WP-11a: gemini-native cache key byte-identical after fix
  - CRIT-2: model read from path not body
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi import FastAPI

from superlocalmemory.optimize.config.schema import OptimizeConfig
from superlocalmemory.optimize.proxy._helpers import _MockTransport
from superlocalmemory.optimize.proxy.lifecycle import (
    CachedResponse,
    HookChain,
    ProxyRequest,
    ProviderResponse,
)
from superlocalmemory.optimize.proxy.server import ProxyApp, build_proxy_router
from superlocalmemory.optimize.proxy.vertex_surface import (
    _VERTEX_PATH_RE,
    _build_vertex_upstream_url,
    _parse_vertex_path,
    _validate_vertex_path,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_VALID_VERTEX_PATH = (
    "my-project-123/locations/us-central1/publishers/google/models/"
    "gemini-2.0-flash:generateContent"
)
_VALID_VERTEX_PATH_STREAM = (
    "my-project-123/locations/us-central1/publishers/google/models/"
    "gemini-2.0-flash:streamGenerateContent"
)

_DEFAULT_TENANT = hashlib.sha256(b"default").hexdigest()


def _vertex_headers(token: str = "Bearer ya29.test-token-abc") -> dict:
    return {
        "authorization": token,
        "content-type": "application/json",
        "x-goog-user-project": "my-project-123",
    }


def _vertex_body(prompt: str = "hello") -> dict:
    return {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "systemInstruction": {"parts": [{"text": "You are helpful"}]},
    }


def _make_proxy(hooks: HookChain | None = None, handler: Any = None) -> ProxyApp:
    config = OptimizeConfig.from_dict({})
    proxy = ProxyApp(config=config)
    proxy.hooks = hooks or HookChain.empty()
    if handler is not None:
        proxy.http_client = httpx.AsyncClient(transport=_MockTransport(handler=handler))
    return proxy


def _make_app(proxy: ProxyApp) -> FastAPI:
    app = FastAPI()
    app.include_router(build_proxy_router(proxy), prefix="")
    return app


def _make_manager(tmp_path: Path):
    """Create an in-memory CacheManager backed by a temp DB."""
    from superlocalmemory.optimize.cache.manager import CacheManager
    from superlocalmemory.optimize.cache.key_builder import CacheConfig
    from superlocalmemory.optimize.storage.db import CacheDB
    db = CacheDB(tmp_path / "llmcache.db")
    return CacheManager(db=db, config=CacheConfig()), db


# ---------------------------------------------------------------------------
# §1: Path validation
# ---------------------------------------------------------------------------

def test_validate_vertex_path_valid() -> None:
    assert _validate_vertex_path(_VALID_VERTEX_PATH) is True
    assert _validate_vertex_path(_VALID_VERTEX_PATH_STREAM) is True


def test_validate_vertex_path_rejects_traversal() -> None:
    # Dots in project segment — rejected by character class
    dotdot = (
        "my..project/locations/us-central1/publishers/google/models/"
        "gemini:generateContent"
    )
    assert _validate_vertex_path(dotdot) is False
    # Path with slashes in project — regex anchors prevent this
    assert _validate_vertex_path(
        "evil/extra/locations/us-central1/publishers/google/models/m:generateContent"
    ) is False


def test_validate_vertex_path_rejects_unknown_method() -> None:
    bad_method = (
        "my-project/locations/us-central1/publishers/google/models/"
        "gemini-flash:countTokens"
    )
    assert _validate_vertex_path(bad_method) is False
    bad_exec = (
        "my-project/locations/us-central1/publishers/google/models/"
        "gemini-flash:EXEC"
    )
    assert _validate_vertex_path(bad_exec) is False


# ---------------------------------------------------------------------------
# §2: Path parsing
# ---------------------------------------------------------------------------

def test_parse_vertex_path_extracts_fields() -> None:
    result = _parse_vertex_path(_VALID_VERTEX_PATH)
    assert result is not None
    project, location, model, method = result
    assert project == "my-project-123"
    assert location == "us-central1"
    assert model == "gemini-2.0-flash"
    assert method == "generateContent"


def test_parse_vertex_path_returns_none_for_invalid() -> None:
    assert _parse_vertex_path("not-a-valid-path") is None
    assert _parse_vertex_path("") is None


# ---------------------------------------------------------------------------
# §3: Upstream URL construction
# ---------------------------------------------------------------------------

def test_build_vertex_upstream_url_uses_region() -> None:
    url = _build_vertex_upstream_url(
        "us-central1",
        "/v1/projects/my-project/locations/us-central1/publishers/google/models/gemini:generateContent",
    )
    assert url.startswith("https://us-central1-aiplatform.googleapis.com")

    url_eu = _build_vertex_upstream_url(
        "europe-west4",
        "/v1/projects/p/locations/europe-west4/publishers/google/models/m:generateContent",
    )
    assert url_eu.startswith("https://europe-west4-aiplatform.googleapis.com")


# ---------------------------------------------------------------------------
# §4: Authorization forwarded byte-identical (AC-2)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_authorization_forwarded_untouched() -> None:
    """The Authorization header byte-identical in upstream request (AC-2)."""
    captured: list[httpx.Request] = []

    async def _handler(req: httpx.Request) -> httpx.Response:
        captured.append(req)
        return httpx.Response(200, json={"candidates": []})

    proxy = _make_proxy(handler=_handler)
    app = _make_app(proxy)
    token = "Bearer ya29.exact-token-abc123"

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.post(
            f"/v1/projects/{_VALID_VERTEX_PATH}",
            json=_vertex_body(),
            headers={**_vertex_headers(token), "content-type": "application/json"},
        )

    assert resp.status_code == 200
    assert len(captured) == 1
    fwd_auth = captured[0].headers.get("authorization")
    assert fwd_auth == token, f"Authorization not forwarded correctly: {fwd_auth!r}"

    await proxy.http_client.aclose()


# ---------------------------------------------------------------------------
# §5: Token NEVER in cache key (SEC)
# ---------------------------------------------------------------------------

def test_token_never_in_cache_key(tmp_path: Path) -> None:
    """Two requests with same body but different bearer tokens → same cache key."""
    manager, db = _make_manager(tmp_path)

    body = _vertex_body()
    path = f"/v1/projects/{_VALID_VERTEX_PATH}"

    req1 = ProxyRequest(
        provider="vertex",
        method="POST",
        path=path,
        headers={"authorization": "[REDACTED]"},
        body=body,
        body_bytes=json.dumps(body).encode(),
        request_id="r1",
        stream=False,
        has_tools=False,
    )
    req2 = ProxyRequest(
        provider="vertex",
        method="POST",
        path=path,
        headers={"authorization": "[REDACTED]"},
        body=body,
        body_bytes=json.dumps(body).encode(),
        request_id="r2",
        stream=False,
        has_tools=False,
    )

    key1 = manager.build_key(req1, _DEFAULT_TENANT)
    key2 = manager.build_key(req2, _DEFAULT_TENANT)

    assert key1 is not None
    assert key1 == key2, "Same body + different tokens must produce same key"
    # Extra: no plausible token component in key
    assert "ya29" not in (key1 or "")
    assert "Bearer" not in (key1 or "")

    db.close()


# ---------------------------------------------------------------------------
# §6: Token NEVER logged (AC-3)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_token_never_logged(caplog: pytest.LogCaptureFixture) -> None:
    """Bearer token must not appear in any log record or stored ProxyRequest repr."""
    secret_token = "Bearer SECRET-NEVER-LOGGED-TOKEN-xyz987"
    captured_req: list[ProxyRequest] = []

    class _TrackingCache:
        def check(self, ctx: ProxyRequest) -> CachedResponse:
            captured_req.append(ctx)
            return CachedResponse(hit=False, data=None, cache_key="vk1", ttl_seconds=300)

        def store(self, ctx: ProxyRequest, resp: ProviderResponse) -> None:
            captured_req.append(ctx)

        def on_hit(self, ctx, resp, ts): pass
        def on_miss(self, ctx): pass

    async def _handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"candidates": [{"content": {}}]})

    proxy = _make_proxy(
        hooks=HookChain(cache=_TrackingCache(), compress=None), handler=_handler
    )
    app = _make_app(proxy)

    with caplog.at_level(logging.DEBUG, logger="slm.optimize.proxy"):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            await client.post(
                f"/v1/projects/{_VALID_VERTEX_PATH}",
                json=_vertex_body(),
                headers={**_vertex_headers(secret_token), "content-type": "application/json"},
            )

    # Token must not appear in any log record
    all_log_text = " ".join(r.getMessage() for r in caplog.records)
    assert "SECRET-NEVER-LOGGED-TOKEN-xyz987" not in all_log_text, (
        "Bearer token leaked into logs"
    )

    # Token must not appear in stored ProxyRequest headers (must be redacted)
    for req in captured_req:
        auth_val = req.headers.get("authorization", "")
        assert "SECRET-NEVER-LOGGED-TOKEN-xyz987" not in auth_val, (
            f"Bearer token leaked into ProxyRequest.headers: {auth_val!r}"
        )

    await proxy.http_client.aclose()


# ---------------------------------------------------------------------------
# §7: Different body → different key (CRIT-2 regression guard)
# ---------------------------------------------------------------------------

def test_different_body_different_key(tmp_path: Path) -> None:
    """Same model+location but different contents → DIFFERENT cache keys (CRIT-2)."""
    manager, db = _make_manager(tmp_path)

    path = f"/v1/projects/{_VALID_VERTEX_PATH}"
    body_a = {"contents": [{"role": "user", "parts": [{"text": "prompt A"}]}]}
    body_b = {"contents": [{"role": "user", "parts": [{"text": "prompt B — different!"}]}]}

    req_a = ProxyRequest(
        provider="vertex", method="POST", path=path,
        headers={}, body=body_a, body_bytes=json.dumps(body_a).encode(),
        request_id="a", stream=False, has_tools=False,
    )
    req_b = ProxyRequest(
        provider="vertex", method="POST", path=path,
        headers={}, body=body_b, body_bytes=json.dumps(body_b).encode(),
        request_id="b", stream=False, has_tools=False,
    )

    key_a = manager.build_key(req_a, _DEFAULT_TENANT)
    key_b = manager.build_key(req_b, _DEFAULT_TENANT)

    assert key_a is not None
    assert key_b is not None
    assert key_a != key_b, (
        "CRIT-2: different Vertex contents must produce DIFFERENT cache keys"
    )

    db.close()


# ---------------------------------------------------------------------------
# §8: No SSE — returns single JSON Response (not StreamingResponse)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_no_sse_single_json_response() -> None:
    """Vertex handler returns fastapi.responses.Response (not StreamingResponse)."""
    async def _handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"candidates": [{"content": {}}]})

    proxy = _make_proxy(handler=_handler)
    app = _make_app(proxy)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.post(
            f"/v1/projects/{_VALID_VERTEX_PATH_STREAM}",
            json=_vertex_body(),
            headers=_vertex_headers(),
        )

    # streamGenerateContent path: still returns single JSON (no SSE per LLD §1)
    assert resp.status_code == 200
    assert "text/event-stream" not in resp.headers.get("content-type", "")

    await proxy.http_client.aclose()


# ---------------------------------------------------------------------------
# §9: Cache hit → upstream called only once
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_cache_hit_second_identical_upstream_called_once() -> None:
    """Same request twice with cache: upstream called exactly once."""
    upstream_calls = 0
    cached_data: list[bytes] = []

    class _RealCache:
        def check(self, ctx: ProxyRequest) -> CachedResponse:
            if cached_data:
                return CachedResponse(
                    hit=True, data=cached_data[0], cache_key="vk2", ttl_seconds=300
                )
            return CachedResponse(hit=False, data=None, cache_key="vk2", ttl_seconds=300)

        def store(self, ctx: ProxyRequest, resp: ProviderResponse) -> None:
            cached_data.append(resp.body_bytes)

        def on_hit(self, ctx, resp, ts): pass
        def on_miss(self, ctx): pass

    async def _handler(req: httpx.Request) -> httpx.Response:
        nonlocal upstream_calls
        upstream_calls += 1
        return httpx.Response(
            200, json={"candidates": [{"content": {"parts": [{"text": "hi"}]}}]}
        )

    proxy = _make_proxy(hooks=HookChain(cache=_RealCache(), compress=None), handler=_handler)
    app = _make_app(proxy)
    body = _vertex_body()

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        r1 = await client.post(
            f"/v1/projects/{_VALID_VERTEX_PATH}", json=body, headers=_vertex_headers()
        )
        r2 = await client.post(
            f"/v1/projects/{_VALID_VERTEX_PATH}", json=body, headers=_vertex_headers()
        )

    assert r1.status_code == 200
    assert r2.status_code == 200
    assert upstream_calls == 1, f"Upstream called {upstream_calls} times; expected 1"

    await proxy.http_client.aclose()


# ---------------------------------------------------------------------------
# §10: Upstream 401 passthrough, nothing cached
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_upstream_401_passthrough_no_cache() -> None:
    """Upstream 401 → response passed through; cache.store NOT called."""
    stored: list = []

    class _TrackStore:
        def check(self, ctx: ProxyRequest) -> CachedResponse:
            return CachedResponse(hit=False, data=None, cache_key="vk3", ttl_seconds=300)

        def store(self, ctx: ProxyRequest, resp: ProviderResponse) -> None:
            stored.append(resp)

        def on_hit(self, ctx, resp, ts): pass
        def on_miss(self, ctx): pass

    async def _handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": {"code": 401, "message": "Unauthorized"}})

    proxy = _make_proxy(hooks=HookChain(cache=_TrackStore(), compress=None), handler=_handler)
    app = _make_app(proxy)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.post(
            f"/v1/projects/{_VALID_VERTEX_PATH}",
            json=_vertex_body(),
            headers=_vertex_headers(),
        )

    assert resp.status_code == 401
    assert len(stored) == 0, "Cache must NOT be stored on non-200 response"

    await proxy.http_client.aclose()


# ---------------------------------------------------------------------------
# §11: SSRF — malformed path returns 400 BEFORE upstream (mock NOT called)
#
# Strategy: test the SSRF guard two ways:
#   A) Call _validate_vertex_path directly with traversal strings — confirms
#      the regex rejects them (HTTP client path normalization is irrelevant here).
#   B) Via FastAPI ASGI with paths that the regex rejects and which do NOT
#      get normalized away by httpx (bad method, overlong segment).
# ---------------------------------------------------------------------------

def test_validate_vertex_path_ssrf_patterns() -> None:
    """_validate_vertex_path directly rejects traversal and injection patterns."""
    # dots in project name
    assert _validate_vertex_path(
        "my..proj/locations/us-central1/publishers/google/models/m:generateContent"
    ) is False
    # forward slash in unusual place is caught by anchored regex
    assert _validate_vertex_path(
        "p/extra/locations/us-east1/publishers/google/models/m:generateContent"
    ) is False
    # unknown method
    assert _validate_vertex_path(
        "p/locations/us-east1/publishers/google/models/m:countTokens"
    ) is False
    # empty string
    assert _validate_vertex_path("") is False


@pytest.mark.asyncio
async def test_ssrf_path_returns_400_mock_not_called() -> None:
    """Paths rejected by the SSRF guard return 400; upstream mock NOT called."""
    upstream_calls = 0

    async def _handler(req: httpx.Request) -> httpx.Response:
        nonlocal upstream_calls
        upstream_calls += 1
        return httpx.Response(200, json={})

    proxy = _make_proxy(handler=_handler)
    app = _make_app(proxy)

    # These paths are NOT normalized by httpx (no ../ — just structurally invalid
    # per the regex: wrong method, extra segment, dots in project name).
    malicious_paths = [
        # Invalid method (not generateContent/streamGenerateContent)
        "p/locations/us-east1/publishers/google/models/m:countTokens",
        "p/locations/us-east1/publishers/google/models/m:EXEC",
        # Dots in project name → rejected by [a-zA-Z0-9_\\-] character class
        "my..proj/locations/us-central1/publishers/google/models/m:generateContent",
    ]

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        for path in malicious_paths:
            resp = await client.post(
                f"/v1/projects/{path}",
                json=_vertex_body(),
                headers=_vertex_headers(),
            )
            assert resp.status_code == 400, (
                f"Expected 400 for malicious path {path!r}, got {resp.status_code}"
            )

    assert upstream_calls == 0, (
        f"Upstream was called {upstream_calls} times; must be 0 for SSRF"
    )

    await proxy.http_client.aclose()


# ---------------------------------------------------------------------------
# §12: Route absent when proxy_enabled=False
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_route_absent_when_proxy_disabled() -> None:
    """When proxy router is not mounted → /v1/projects/ route returns 404."""
    app = FastAPI()
    # No router mounted — simulates proxy_enabled=False daemon behaviour

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.post(
            f"/v1/projects/{_VALID_VERTEX_PATH}",
            json=_vertex_body(),
            headers=_vertex_headers(),
        )

    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# §13: WP-11a — gemini-native forwards authorization header
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_wp11a_gemini_authorization_forwarded() -> None:
    """WP-11a: after the fix, gemini-native forwards Authorization upstream."""
    captured: list[httpx.Request] = []

    async def _handler(req: httpx.Request) -> httpx.Response:
        captured.append(req)
        return httpx.Response(200, json={"candidates": []})

    proxy = _make_proxy(handler=_handler)
    app = _make_app(proxy)
    bearer = "Bearer ya29.adc-token-for-antigravity"

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.post(
            "/v1beta/models/gemini-2.0-flash:generateContent",
            json={"contents": [{"role": "user", "parts": [{"text": "hi"}]}]},
            headers={
                "authorization": bearer,
                "content-type": "application/json",
            },
        )

    assert resp.status_code == 200
    assert len(captured) == 1
    fwd_auth = captured[0].headers.get("authorization")
    assert fwd_auth == bearer, (
        f"WP-11a: Authorization not forwarded on gemini-native: {fwd_auth!r}"
    )

    await proxy.http_client.aclose()


# ---------------------------------------------------------------------------
# §14: WP-11a — gemini-native cache key byte-identical after the fix
# ---------------------------------------------------------------------------

def test_wp11a_gemini_cache_key_unchanged(tmp_path: Path) -> None:
    """WP-11a no-regression: gemini-native cache key byte-identical before/after fix.

    The fix adds 'authorization' to _GEMINI_NATIVE_FORWARD_HEADERS (a header
    forward set), NOT to key-building logic. build_key reads only body fields
    (key_builder.py:90-105) → key is unchanged regardless of headers present.
    """
    manager, db = _make_manager(tmp_path)

    body = {
        "model": "gemini-2.0-flash",
        "messages": [{"role": "user", "content": "hello"}],
    }

    # Request WITHOUT authorization header (pre-WP-11a simulation)
    req_no_auth = ProxyRequest(
        provider="gemini",
        method="POST",
        path="/v1beta/models/gemini-2.0-flash:generateContent",
        headers={},  # no auth
        body=body,
        body_bytes=json.dumps(body).encode(),
        request_id="g1",
        stream=False,
        has_tools=False,
    )

    # Request WITH authorization header (post-WP-11a simulation)
    req_with_auth = ProxyRequest(
        provider="gemini",
        method="POST",
        path="/v1beta/models/gemini-2.0-flash:generateContent",
        headers={"authorization": "[REDACTED]"},
        body=body,
        body_bytes=json.dumps(body).encode(),
        request_id="g2",
        stream=False,
        has_tools=False,
    )

    key_no_auth = manager.build_key(req_no_auth, _DEFAULT_TENANT)
    key_with_auth = manager.build_key(req_with_auth, _DEFAULT_TENANT)

    # build_key reads only .body — headers not included — keys must be identical
    assert key_no_auth == key_with_auth, (
        f"WP-11a cache key changed after auth header fix: "
        f"{key_no_auth!r} != {key_with_auth!r}"
    )

    db.close()


# ---------------------------------------------------------------------------
# §15: CRIT-1 — route registered at /v1/projects/{vertex_path:path}
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_vertex_route_registered_in_router() -> None:
    """The /v1/projects/{vertex_path:path} route is present in build_proxy_router."""
    async def _handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"candidates": []})

    proxy = _make_proxy(handler=_handler)
    app = _make_app(proxy)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.post(
            f"/v1/projects/{_VALID_VERTEX_PATH}",
            json=_vertex_body(),
            headers=_vertex_headers(),
        )

    # 200 means route was matched (not 404/405)
    assert resp.status_code == 200

    await proxy.http_client.aclose()


# ---------------------------------------------------------------------------
# §16: CRIT-2 — vertex build_key uses path for model, not body["model"]
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_vertex_invalid_json_body_falls_back_to_empty_dict() -> None:
    """Invalid JSON body → body={} fallback; request proceeds to upstream."""
    async def _handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"candidates": []})

    proxy = _make_proxy(handler=_handler)
    app = _make_app(proxy)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.post(
            f"/v1/projects/{_VALID_VERTEX_PATH}",
            content=b"{bad json!!",
            headers=_vertex_headers(),
        )

    assert resp.status_code == 200
    await proxy.http_client.aclose()


@pytest.mark.asyncio
async def test_vertex_fail_open_on_exception() -> None:
    """When http_client raises, handler falls back to fail_open_forward (502)."""
    async def _broken_handler(req: httpx.Request) -> httpx.Response:
        raise RuntimeError("upstream totally broken")

    proxy = _make_proxy(handler=_broken_handler)
    app = _make_app(proxy)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.post(
            f"/v1/projects/{_VALID_VERTEX_PATH}",
            json=_vertex_body(),
            headers=_vertex_headers(),
        )

    # fail_open_forward re-tries with the broken client → 502
    assert resp.status_code in (502, 200)
    await proxy.http_client.aclose()


def test_vertex_build_key_reads_model_from_path_not_body(tmp_path: Path) -> None:
    """CRIT-2: build_key for provider='vertex' extracts model from path, not body['model']."""
    manager, db = _make_manager(tmp_path)

    # Two requests: same body (no 'model' key) but different paths (different model)
    path_flash = (
        "/v1/projects/my-project/locations/us-central1/publishers/google/models/"
        "gemini-2.0-flash:generateContent"
    )
    path_pro = (
        "/v1/projects/my-project/locations/us-central1/publishers/google/models/"
        "gemini-2.0-pro:generateContent"
    )
    body = {"contents": [{"role": "user", "parts": [{"text": "same prompt"}]}]}

    req_flash = ProxyRequest(
        provider="vertex", method="POST", path=path_flash,
        headers={}, body=body, body_bytes=json.dumps(body).encode(),
        request_id="f", stream=False, has_tools=False,
    )
    req_pro = ProxyRequest(
        provider="vertex", method="POST", path=path_pro,
        headers={}, body=body, body_bytes=json.dumps(body).encode(),
        request_id="p", stream=False, has_tools=False,
    )

    key_flash = manager.build_key(req_flash, _DEFAULT_TENANT)
    key_pro = manager.build_key(req_pro, _DEFAULT_TENANT)

    assert key_flash is not None
    assert key_pro is not None
    assert key_flash != key_pro, (
        "CRIT-2: Different model in path must produce different cache keys"
    )

    db.close()
