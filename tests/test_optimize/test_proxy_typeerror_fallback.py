"""Stage-9 R1 regression: a tenant-aware CacheHook that raises an INTERNAL
TypeError must fail-open to a cache MISS — never be silently retried on the
shared (tenant-less) namespace, which would re-open cross-tenant disclosure.
"""
import asyncio

from superlocalmemory.optimize.proxy._helpers import (
    _HOOK_TENANT_SUPPORT,
    _safe_cache_check,
    _safe_cache_store,
    _accepts_tenant_id,
)
from superlocalmemory.optimize.proxy.lifecycle import (
    CachedResponse,
    HookChain,
    ProxyRequest,
)


def _ctx():
    return ProxyRequest(
        provider="anthropic", method="POST", path="/v1/messages", headers={},
        body={}, body_bytes=b"{}", request_id="r", stream=False, has_tools=False,
    )


class _NoopCompress:
    def compress(self, ctx):
        return ctx

    def on_compress(self, b, a, c):
        pass


class _TenantHookInternalTypeError:
    """New-style (tenant_id param) hook that raises TypeError INTERNALLY."""

    def check(self, ctx, tenant_id=None):
        if tenant_id is not None:
            _ = [1, 2, 3][tenant_id]  # internal TypeError (str index)
        return CachedResponse(hit=True, data=b"SHARED_SECRET", cache_key="k", ttl_seconds=60)

    def store(self, ctx, resp, tenant_id=None):
        if tenant_id is not None:
            _ = [1, 2, 3][tenant_id]

    def on_hit(self, *a):
        pass

    def on_miss(self, ctx):
        pass


class _KwargsHookLeaks:
    """**kwargs hook that leaks data when called without tenant_id."""

    def check(self, ctx, **kwargs):
        if "tenant_id" in kwargs:
            raise TypeError("internal bug")
        return CachedResponse(hit=True, data=b"LEAK", cache_key="k", ttl_seconds=60)

    def store(self, ctx, resp, **kwargs):
        pass

    def on_hit(self, *a):
        pass

    def on_miss(self, ctx):
        pass


class _TrueLegacyHook:
    """Genuine legacy hook: no tenant_id param, no **kwargs."""

    def check(self, ctx):
        return CachedResponse(hit=True, data=b"legacy", cache_key="k", ttl_seconds=60)

    def store(self, ctx, resp):
        pass

    def on_hit(self, *a):
        pass

    def on_miss(self, ctx):
        pass


def test_internal_typeerror_fails_open_to_miss_not_shared_namespace():
    hooks = HookChain(cache=_TenantHookInternalTypeError(), compress=_NoopCompress())
    r = asyncio.run(_safe_cache_check(hooks, _ctx(), tenant_id="t" * 64))
    assert r.hit is False and r.data is None  # NOT b"SHARED_SECRET"


def test_kwargs_hook_never_downgraded_to_tenantless():
    hooks = HookChain(cache=_KwargsHookLeaks(), compress=_NoopCompress())
    r = asyncio.run(_safe_cache_check(hooks, _ctx(), tenant_id="t" * 64))
    assert r.hit is False  # NOT b"LEAK"


def test_true_legacy_hook_still_callable():
    hooks = HookChain(cache=_TrueLegacyHook(), compress=_NoopCompress())
    r = asyncio.run(_safe_cache_check(hooks, _ctx(), tenant_id="t" * 64))
    assert r.hit is True and r.data == b"legacy"  # backward compat preserved


def test_store_internal_typeerror_does_not_retry_tenantless():
    calls = []

    class _StoreHook:
        def store(self, ctx, resp, tenant_id=None):
            calls.append(tenant_id)
            if tenant_id is not None:
                _ = [1][tenant_id]  # internal TypeError

        def check(self, ctx, tenant_id=None):
            return None

        def on_hit(self, *a):
            pass

        def on_miss(self, ctx):
            pass

    hooks = HookChain(cache=_StoreHook(), compress=_NoopCompress())
    asyncio.run(_safe_cache_store(hooks, _ctx(), None, tenant_id="t" * 64))
    # Must be called exactly once (with tenant_id) — never retried tenant-less.
    assert calls == ["t" * 64]


def test_accepts_tenant_id_detection():
    _HOOK_TENANT_SUPPORT.clear()
    assert _accepts_tenant_id(_TenantHookInternalTypeError().check) is True
    assert _accepts_tenant_id(_KwargsHookLeaks().check) is True   # **kwargs
    assert _accepts_tenant_id(_TrueLegacyHook().check) is False
    assert all(not isinstance(key, int) for key in _HOOK_TENANT_SUPPORT)
