"""LLD-01 §5.1 — lifecycle DTO and hook protocol tests."""

from __future__ import annotations

import pytest

from superlocalmemory.optimize.proxy.lifecycle import (
    CachedResponse,
    CacheHook,
    CompressHook,
    HookChain,
    ProviderResponse,
    ProxyRequest,
    ensure_proxy_running,
    proxy_port,
)


def test_proxy_port_is_8765() -> None:
    assert proxy_port() == 8765


def test_ensure_proxy_running_returns_bool() -> None:
    result = ensure_proxy_running()
    assert isinstance(result, bool)


def test_proxy_request_repr_excludes_body() -> None:
    """body content must NOT appear in repr (PII guard)."""
    req = ProxyRequest(
        provider="anthropic", method="POST", path="/v1/messages",
        headers={"x-api-key": "[REDACTED]"},
        body={"secret": "sensitive-content-must-not-leak-in-repr-1234567890"},
        body_bytes=b"", request_id="slm_1", stream=False, has_tools=False,
    )
    r = repr(req)
    assert "sensitive-content" not in r
    assert "secret" not in r


def test_cached_response_default() -> None:
    r = CachedResponse(hit=False, data=None, cache_key="", ttl_seconds=0)
    assert r.hit is False
    assert r.data is None


def test_provider_response_modified_false() -> None:
    r = ProviderResponse(
        modified=False, body={}, body_bytes=b"",
        tokens_before=0, tokens_after=0, strategy="none",
    )
    assert r.modified is False


def test_hook_chain_empty() -> None:
    chain = HookChain.empty()
    assert chain.cache is None
    assert chain.compress is None


def test_cachehook_protocol_runtime_checkable() -> None:
    """A class implementing the protocol methods should be considered a subclass."""

    class _Good:
        def check(self, req):
            return None
        def store(self, req, resp):
            pass
        def on_hit(self, req, resp, tokens_saved):
            pass
        def on_miss(self, req):
            pass

    assert isinstance(_Good(), CacheHook)


def test_compresshook_protocol_runtime_checkable() -> None:
    class _Good:
        def compress(self, req):
            return req
        def on_compress(self, before_tokens, after_tokens, lossy):
            pass

    assert isinstance(_Good(), CompressHook)
