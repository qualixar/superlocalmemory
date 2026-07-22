"""test_proxy_tenant_isolation.py — Regression tests for cross-tenant cache disclosure.

WP-D: HIGH security fix — proxy CacheHook must derive tenant_id from the raw
inbound credential, NOT collapse all requests to a single _DEFAULT_TENANT_HASH.

Test matrix:
  TC-1  two different API keys, same prompt → different tenant_ids → cache keys differ
        → store(A) + check(B) = MISS (no cross-tenant disclosure)
  TC-2  same API key twice, same prompt → same tenant_id → cache keys same → HIT
  TC-3  no credential present → cache is NOT called (skip, no store/check)
  TC-4  _derive_tenant_id helper returns 64-char hex SHA-256
  TC-5  check()/store() accept optional tenant_id; default preserves backward compat

All tests are pure unit-level — no network, no real upstream, no FastAPI app.
CacheManager is exercised via an in-memory SQLite CacheDB (tmp_cache_db fixture).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterator

import pytest

# ---------------------------------------------------------------------------
# Minimal fixtures (cannot use tests/optimize/conftest.py from a different dir;
# replicate the one fixture we need inline so the new test is self-contained).
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_cache_db(tmp_path: Path, monkeypatch) -> Iterator[object]:
    from superlocalmemory.optimize.storage import db as _db_mod
    from superlocalmemory.optimize.storage.db import CacheDB

    monkeypatch.setattr(_db_mod, "_KEY_FILE", tmp_path / "opt-key.bin")
    db = CacheDB(tmp_path / "llmcache.db")
    yield db


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PROMPT = {"model": "claude-sonnet-4-5", "messages": [{"role": "user", "content": "Hello"}]}
_RESP_DICT = {
    "id": "msg_01",
    "type": "message",
    "role": "assistant",
    "content": [{"type": "text", "text": "Hi"}],
    "model": "claude-sonnet-4-5",
    "stop_reason": "end_turn",
    "usage": {"input_tokens": 5, "output_tokens": 3},
}


def _sha256_hex(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _make_proxy_request(provider: str = "anthropic") -> object:
    """Build a ProxyRequest with a body matching _PROMPT (headers already redacted)."""
    from superlocalmemory.optimize.proxy.lifecycle import ProxyRequest

    return ProxyRequest(
        provider=provider,
        method="POST",
        path="/v1/messages",
        headers={"x-api-key": "[REDACTED]", "content-type": "application/json"},
        body=_PROMPT,
        body_bytes=json.dumps(_PROMPT).encode(),
        request_id="req-1",
        stream=False,
        has_tools=False,
    )


def _make_provider_response() -> object:
    from superlocalmemory.optimize.proxy.lifecycle import ProviderResponse

    return ProviderResponse(
        modified=False,
        body={},
        body_bytes=json.dumps(_RESP_DICT).encode(),
        tokens_before=0,
        tokens_after=0,
        strategy="none",
    )


# ---------------------------------------------------------------------------
# TC-1: Two different API keys, same prompt → tenant_ids differ → cache MISS
# ---------------------------------------------------------------------------


def test_tc1_different_keys_produce_different_tenant_ids() -> None:
    """Core isolation property: sha256(anthropic:key_A) != sha256(anthropic:key_B)."""
    from superlocalmemory.optimize.proxy._helpers import _derive_tenant_id

    tid_a = _derive_tenant_id("anthropic", "sk-ant-aaaa")
    tid_b = _derive_tenant_id("anthropic", "sk-ant-bbbb")

    assert tid_a != tid_b, "Different keys must produce different tenant_ids"
    assert len(tid_a) == 64, "tenant_id must be 64-char hex"
    assert len(tid_b) == 64


def test_tc1_store_a_then_check_b_is_miss(tmp_cache_db) -> None:
    """store(key=A) then check(key=B) with same prompt must return a cache MISS."""
    from superlocalmemory.optimize.cache.manager import CacheManager
    from superlocalmemory.optimize.proxy._helpers import _derive_tenant_id

    cm = CacheManager(tmp_cache_db)
    req = _make_proxy_request()
    prov_resp = _make_provider_response()

    tid_a = _derive_tenant_id("anthropic", "sk-ant-aaaa")
    tid_b = _derive_tenant_id("anthropic", "sk-ant-bbbb")

    # Store with tenant A
    cm.set(req, prov_resp, tenant_id=tid_a)

    # Check with tenant B — must be a MISS, not a cross-tenant hit
    result = cm.get(req, tenant_id=tid_b)
    assert result is not None, "get() must return a CachedResponse (miss DTO)"
    assert result.hit is False, (
        "SECURITY REGRESSION: cross-tenant cache hit — "
        "user B received user A's private completion"
    )


# ---------------------------------------------------------------------------
# TC-2: Same API key, same prompt → cache HIT
# ---------------------------------------------------------------------------


def test_tc2_same_key_same_prompt_is_hit(tmp_cache_db) -> None:
    """Cache HIT when same credential + same prompt (functional correctness)."""
    from superlocalmemory.optimize.cache.manager import CacheManager
    from superlocalmemory.optimize.proxy._helpers import _derive_tenant_id

    cm = CacheManager(tmp_cache_db)
    req = _make_proxy_request()
    prov_resp = _make_provider_response()

    tid = _derive_tenant_id("anthropic", "sk-ant-aaaa")

    cm.set(req, prov_resp, tenant_id=tid)
    result = cm.get(req, tenant_id=tid)

    assert result is not None
    assert result.hit is True, "Same tenant + same prompt must be a cache HIT"


# ---------------------------------------------------------------------------
# TC-3: No credential → cache is NOT called (skip-caching path)
# ---------------------------------------------------------------------------


def test_tc3_no_credential_derive_returns_none() -> None:
    """_derive_tenant_id returns None when no raw credential is present."""
    from superlocalmemory.optimize.proxy._helpers import _derive_tenant_id

    assert _derive_tenant_id("anthropic", None) is None
    assert _derive_tenant_id("anthropic", "") is None
    assert _derive_tenant_id("openai", "") is None


def test_tc3_check_skips_when_no_tenant(tmp_cache_db) -> None:
    """CacheManager.check() must return None (skip) when tenant_id is None."""
    from superlocalmemory.optimize.cache.manager import CacheManager
    from superlocalmemory.optimize.proxy.lifecycle import ProxyRequest

    cm = CacheManager(tmp_cache_db)
    req = _make_proxy_request()

    # Simulate calling check() with explicit None tenant — must not touch the cache.
    # The surface handler is responsible for this, but we verify the manager
    # honours tenant_id=None by returning None (skip).
    result = cm.check(req, tenant_id=None)
    assert result is None, (
        "check() with tenant_id=None must return None (do not cache unauthenticated requests)"
    )


def test_tc3_store_skips_when_no_tenant(tmp_cache_db) -> None:
    """CacheManager.store() must be a no-op when tenant_id is None."""
    from superlocalmemory.optimize.cache.manager import CacheManager

    cm = CacheManager(tmp_cache_db)
    req = _make_proxy_request()
    prov_resp = _make_provider_response()

    # This must not raise and must not write anything.
    cm.store(req, prov_resp, tenant_id=None)
    # Verify nothing was stored: subsequent check under any tenant must miss.
    some_tid = _sha256_hex("anthropic:sk-any")
    result = cm.get(req, tenant_id=some_tid)
    assert result is not None
    assert result.hit is False


# ---------------------------------------------------------------------------
# TC-4: _derive_tenant_id output format
# ---------------------------------------------------------------------------


def test_tc4_derive_tenant_id_is_64char_hex() -> None:
    from superlocalmemory.optimize.proxy._helpers import _derive_tenant_id

    tid = _derive_tenant_id("anthropic", "sk-ant-test1234")
    assert isinstance(tid, str)
    assert len(tid) == 64
    # Must be lowercase hex
    assert all(c in "0123456789abcdef" for c in tid)


def test_tc4_derive_tenant_id_matches_manual_sha256(monkeypatch) -> None:
    """Output must match sha256(profile:provider:raw_key).hexdigest().

    I-5: the active memory profile is folded into the tenant key so two
    profiles sharing an API key never share cache. Pin the profile so the
    hash is deterministic.
    """
    from superlocalmemory.optimize.proxy import _helpers
    from superlocalmemory.optimize.proxy._helpers import _derive_tenant_id

    monkeypatch.setattr(_helpers, "_active_profile_for_cache", lambda: "work")
    raw_key = "sk-ant-api01-xyzzy"
    provider = "anthropic"
    expected = hashlib.sha256(f"work:{provider}:{raw_key}".encode()).hexdigest()
    assert _derive_tenant_id(provider, raw_key) == expected


def test_tc4_same_key_different_profiles_differ(monkeypatch) -> None:
    """I-5 core property: same API key under two profiles → different tenants."""
    from superlocalmemory.optimize.proxy import _helpers
    from superlocalmemory.optimize.proxy._helpers import _derive_tenant_id

    key = "sk-ant-shared"
    monkeypatch.setattr(_helpers, "_active_profile_for_cache", lambda: "work")
    tid_work = _derive_tenant_id("anthropic", key)
    monkeypatch.setattr(_helpers, "_active_profile_for_cache", lambda: "home")
    tid_home = _derive_tenant_id("anthropic", key)
    assert tid_work != tid_home, (
        "same key under different profiles must produce different tenant_ids "
        "(no cross-profile cache reuse)"
    )


def test_tc4_different_providers_same_key_differ() -> None:
    """Provider is folded into the hash — anthropic:KEY != openai:KEY."""
    from superlocalmemory.optimize.proxy._helpers import _derive_tenant_id

    key = "sk-shared-key"
    assert _derive_tenant_id("anthropic", key) != _derive_tenant_id("openai", key)


# ---------------------------------------------------------------------------
# TC-5: check()/store() optional tenant_id backward compatibility
# ---------------------------------------------------------------------------


def test_tc5_check_signature_accepts_optional_tenant_id(tmp_cache_db) -> None:
    """check() and store() must accept an optional tenant_id kwarg."""
    from superlocalmemory.optimize.cache.manager import CacheManager, _DEFAULT_TENANT_HASH

    cm = CacheManager(tmp_cache_db)
    req = _make_proxy_request()

    # Old callers that don't pass tenant_id must still work (backward compat).
    # Default should use _DEFAULT_TENANT_HASH — we verify it does NOT raise.
    result = cm.check(req)  # no tenant_id arg — backward compat call
    # Must return something (None or CachedResponse) without error.
    assert result is None or hasattr(result, "hit")


def test_tc5_store_signature_accepts_optional_tenant_id(tmp_cache_db) -> None:
    """store() without tenant_id must not raise (backward compat)."""
    from superlocalmemory.optimize.cache.manager import CacheManager

    cm = CacheManager(tmp_cache_db)
    req = _make_proxy_request()
    prov_resp = _make_provider_response()

    # Must not raise
    cm.store(req, prov_resp)  # no tenant_id arg


# ---------------------------------------------------------------------------
# TC-6: Anthropic surface derives tenant from raw header (integration)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tc6_anthropic_surface_extracts_raw_key(tmp_cache_db, monkeypatch) -> None:
    """handle_messages must derive tenant from x-api-key BEFORE redaction."""
    from superlocalmemory.optimize.proxy import _helpers as _proxy_helpers
    monkeypatch.setattr(
        _proxy_helpers, "_active_profile_for_cache", lambda: "default")
    import json

    import httpx
    from fastapi import FastAPI

    from superlocalmemory.optimize.config.schema import OptimizeConfig
    from superlocalmemory.optimize.proxy.lifecycle import HookChain, CachedResponse, ProviderResponse
    from superlocalmemory.optimize.proxy.server import ProxyApp, build_proxy_router

    check_tenant_ids: list[str | None] = []
    store_tenant_ids: list[str | None] = []

    class TenantCapturingManager:
        """Replaces CacheManager; records tenant_ids passed to check/store."""

        def check(self, req, tenant_id=None):
            check_tenant_ids.append(tenant_id)
            return CachedResponse(hit=False, data=None, cache_key="some-key", ttl_seconds=0)

        def store(self, req, resp, tenant_id=None):
            store_tenant_ids.append(tenant_id)

        def on_hit(self, req, resp, tokens_saved):
            pass

        def on_miss(self, req):
            pass

    resp_body = json.dumps({
        "id": "msg_02", "type": "message", "role": "assistant",
        "content": [{"type": "text", "text": "ok"}],
        "model": "claude-sonnet-4-5", "stop_reason": "end_turn",
        "usage": {"input_tokens": 5, "output_tokens": 2},
    }).encode()

    async def _run():
        async def _upstream_handler(req: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=resp_body,
                                  headers={"content-type": "application/json"})

        class _T(httpx.AsyncBaseTransport):
            async def handle_async_request(self, req):
                return await _upstream_handler(req)

        config = OptimizeConfig.from_dict({
            "enabled": True, "proxy_enabled": True,
            "cache_enabled": True, "compress_enabled": False,
            "ttl_seconds": 300,
        })
        proxy = ProxyApp(config=config)
        capturing_manager = TenantCapturingManager()
        proxy.hooks = HookChain(cache=capturing_manager, compress=None)
        proxy.http_client = httpx.AsyncClient(transport=_T())
        app = FastAPI()
        app.include_router(build_proxy_router(proxy), prefix="")

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            await client.post(
                "/v1/messages",
                json={"model": "claude-sonnet-4-5", "max_tokens": 10,
                      "messages": [{"role": "user", "content": "hi"}]},
                headers={
                    "x-api-key": "sk-ant-secret-key",
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
            )

        await proxy.http_client.aclose()

    await _run()

    expected_tid = hashlib.sha256("default:anthropic:sk-ant-secret-key".encode()).hexdigest()
    assert check_tenant_ids, "check() must have been called"
    assert check_tenant_ids[0] == expected_tid, (
        f"check() received wrong tenant_id: {check_tenant_ids[0]!r}, "
        f"expected {expected_tid!r}"
    )
