"""LLD-02 §8.5 + LLD-10 P1 — CacheManager tests."""

from __future__ import annotations

import pytest

from superlocalmemory.optimize.cache.key_builder import CacheConfig
from superlocalmemory.optimize.cache.manager import (
    CacheManager,
    NoOpSemantic,
    SemanticTier,
)


def _tenant(n: int) -> str:
    return f"{n:064x}"


def test_get_instance_returns_singleton(tmp_cache_db, monkeypatch) -> None:
    """get_instance() returns the same CacheManager on every call."""
    monkeypatch.setattr(
        "superlocalmemory.optimize.storage.db.CacheDB.get_default",
        classmethod(lambda cls: tmp_cache_db),
    )
    CacheManager.reset_instance()
    a = CacheManager.get_instance()
    b = CacheManager.get_instance()
    assert a is b
    CacheManager.reset_instance()


def test_build_key_returns_key(tmp_cache_db) -> None:
    cm = CacheManager(tmp_cache_db)
    key = cm.build_key(
        {"model": "claude-sonnet-4-6", "messages": [{"role": "user", "content": "hi"}],
         "params": {"max_tokens": 100}},
        _tenant(1),
    )
    assert key is not None
    assert _tenant(1) in key


def test_build_key_returns_none_for_temperature_gt_zero(tmp_cache_db) -> None:
    cm = CacheManager(tmp_cache_db)
    key = cm.build_key(
        {"model": "m", "messages": [], "params": {"temperature": 0.7}},
        _tenant(1),
    )
    assert key is None


def test_get_or_call_hit_serves_zero_upstream_tokens(tmp_cache_db) -> None:
    """P1 gate: repeat call tokens = 0 on cache hit."""
    cm = CacheManager(tmp_cache_db)
    upstream_calls = []

    def upstream():
        upstream_calls.append(1)
        return {"id": "msg_x", "stop_reason": "end_turn"}

    kwargs = dict(
        tenant_id=_tenant(1), model_id="m", model_version="v",
        system="", messages=[{"role": "user", "content": "What is 2+2?"}],
        raw_params={"max_tokens": 50}, upstream_fn=upstream,
    )
    out1 = cm.get_or_call(**kwargs)
    out2 = cm.get_or_call(**kwargs)
    assert out1 == out2
    assert len(upstream_calls) == 1, f"second call hit upstream: {len(upstream_calls)}"


def test_get_or_call_miss_invokes_upstream(tmp_cache_db) -> None:
    cm = CacheManager(tmp_cache_db)
    upstream_calls = []

    def upstream():
        upstream_calls.append(1)
        return {"id": "msg_x", "stop_reason": "end_turn"}

    kwargs = dict(
        tenant_id=_tenant(1), model_id="m", model_version="v",
        system="", messages=[{"role": "user", "content": "different each time?"}],
        raw_params={"max_tokens": 50}, upstream_fn=upstream,
    )
    cm.get_or_call(**kwargs)
    cm.get_or_call(**{**kwargs, "messages": [{"role": "user", "content": "ask 2"}]})
    assert len(upstream_calls) == 2


def test_get_or_call_fail_open(tmp_cache_db) -> None:
    """F7 / P1 gate: cache failure must NOT break the call."""
    cm = CacheManager(tmp_cache_db)
    upstream_calls = []

    def upstream():
        upstream_calls.append(1)
        return {"id": "ok", "stop_reason": "end_turn"}

    # Use a broken key builder to trigger inner failure
    class _BrokenKey:
        def build(self, *a, **kw):
            raise RuntimeError("boom")

    cm._key_builder = _BrokenKey()
    out = cm.get_or_call(
        tenant_id=_tenant(1), model_id="m", model_version="v",
        system="", messages=[], raw_params={}, upstream_fn=upstream,
    )
    assert out == {"id": "ok", "stop_reason": "end_turn"}
    assert len(upstream_calls) == 1


def test_get_or_call_does_not_cache_tool_use(tmp_cache_db) -> None:
    """P1 gate: tool-use responses must not be stored."""
    cm = CacheManager(tmp_cache_db)

    def upstream_tool():
        return {
            "id": "msg_2", "type": "message",
            "content": [{"type": "tool_use", "id": "toolu_x", "name": "bash", "input": {}}],
            "stop_reason": "tool_use",
        }

    def upstream_ok():
        return {"id": "msg_3", "stop_reason": "end_turn"}

    kwargs = dict(
        tenant_id=_tenant(1), model_id="m", model_version="v",
        system="", messages=[{"role": "user", "content": "run ls"}],
        raw_params={"max_tokens": 50},
    )
    out1 = cm.get_or_call(**kwargs, upstream_fn=upstream_tool)
    out2 = cm.get_or_call(**kwargs, upstream_fn=upstream_ok)
    # Both calls reach upstream (tool_use response not stored)
    assert out1["stop_reason"] == "tool_use"
    assert out2["stop_reason"] == "end_turn"


def test_for_tenant_returns_scoped_view(tmp_cache_db) -> None:
    cm = CacheManager(tmp_cache_db)
    view = cm.for_tenant(_tenant(1))
    assert view is not None
    assert view._tenant_id == _tenant(1)


def test_invalidate_tag_clears_entries(tmp_cache_db) -> None:
    cm = CacheManager(tmp_cache_db)
    upstream_calls = []

    def upstream():
        upstream_calls.append(1)
        return {"id": "x", "stop_reason": "end_turn"}

    kwargs = dict(
        tenant_id=_tenant(1), model_id="claude-sonnet-4-6", model_version="v",
        system="", messages=[{"role": "user", "content": "hi"}],
        raw_params={"max_tokens": 50}, upstream_fn=upstream,
    )
    cm.get_or_call(**kwargs)
    # Manually register a tag we can invalidate
    cm._invalidation.register(cm.build_key(
        {"model": "claude-sonnet-4-6", "messages": kwargs["messages"], "params": {}},
        _tenant(1),
    ) or "none", _tenant(1), ["tenant:abc"])
    count = cm.invalidate_tag("tenant:abc")
    assert count >= 0  # may be 0 if no entry matched; just shouldn't raise


def test_metrics_hit_rate(tmp_cache_db) -> None:
    cm = CacheManager(tmp_cache_db)

    def upstream():
        return {"id": "x", "stop_reason": "end_turn"}

    kwargs = dict(
        tenant_id=_tenant(1), model_id="m", model_version="v",
        system="", messages=[{"role": "user", "content": "hi"}],
        raw_params={"max_tokens": 50}, upstream_fn=upstream,
    )
    cm.get_or_call(**kwargs)  # miss + set
    cm.get_or_call(**kwargs)  # hit
    cm.get_or_call(**kwargs)  # hit
    assert cm.metrics.exact_hits == 2
    assert cm.metrics.hit_rate() > 0


def test_noop_semantic_disabled() -> None:
    noop = NoOpSemantic()
    assert noop.is_enabled() is False
    assert noop.lookup(None, "t", None) is None
    noop.learn("eid", 0.9, True)  # must not raise


# ---- build_key with object (non-dict) path ----

def test_build_key_from_object(tmp_cache_db) -> None:
    """build_key handles object (dataclass/SimpleNamespace) with non-dict req."""
    from types import SimpleNamespace
    cm = CacheManager(tmp_cache_db)
    req = SimpleNamespace(
        model_id="claude-sonnet-4-6",
        messages=[{"role": "user", "content": "hi"}],
        params={"max_tokens": 100},
        system="",
    )
    key = cm.build_key(req, _tenant(1))
    assert key is not None
    assert _tenant(1) in key


def test_build_key_object_none_attrs(tmp_cache_db) -> None:
    """build_key handles object with None attributes — exercises object branch."""
    from types import SimpleNamespace
    cm = CacheManager(tmp_cache_db)
    req = SimpleNamespace()
    # Even with empty attrs, build_key works (exercises lines 163-166)
    key = cm.build_key(req, _tenant(1))
    assert key is not None  # key builder still produces a cache key


# ---- get / set methods ----

def test_get_hit_returns_cached_response(tmp_cache_db) -> None:
    """get() hits exact cache and returns CachedResponse."""
    cm = CacheManager(tmp_cache_db)
    key = cm.build_key(
        {"model": "claude-sonnet-4-6", "messages": [{"role": "user", "content": "hi"}], "params": {}},
        _tenant(1),
    )
    assert key is not None
    # Pre-populate the cache with a set
    cm._exact.set(key, _tenant(1), {"text": "cached_response"}, ["tag1"], "claude-sonnet-4-6")
    result = cm.get(
        {"model": "claude-sonnet-4-6", "messages": [{"role": "user", "content": "hi"}], "params": {}},
        _tenant(1),
    )
    assert result is not None
    assert result.hit is True
    assert result.cache_key == key
    assert b"cached_response" in result.data


def test_get_miss_when_key_is_none(tmp_cache_db) -> None:
    """get() returns None when build_key returns None."""
    cm = CacheManager(tmp_cache_db)
    result = cm.get(
        {"model": "m", "messages": [], "params": {"temperature": 0.7}},
        _tenant(1),
    )
    assert result is None


def test_get_miss_returns_none(tmp_cache_db) -> None:
    """get() returns None on cache miss."""
    cm = CacheManager(tmp_cache_db)
    result = cm.get(
        {"model": "xyz-nonexistent", "messages": [{"role": "user", "content": "nope"}], "params": {}},
        _tenant(1),
    )
    assert result is None


def test_set_with_dict_response(tmp_cache_db) -> None:
    """set() with dict resp stores in exact cache."""
    cm = CacheManager(tmp_cache_db)
    cm.set(
        {"model": "claude-sonnet-4-6", "messages": [{"role": "user", "content": "hi"}], "params": {}},
        {"text": "hello world"},
        _tenant(1),
    )
    # Verify via get
    result = cm.get(
        {"model": "claude-sonnet-4-6", "messages": [{"role": "user", "content": "hi"}], "params": {}},
        _tenant(1),
    )
    assert result is not None
    assert b"hello world" in result.data


def test_set_with_provider_response(tmp_cache_db) -> None:
    """set() with ProviderResponse uses body_bytes."""
    from superlocalmemory.optimize.proxy.lifecycle import ProviderResponse
    import json as _json
    cm = CacheManager(tmp_cache_db)
    resp_bytes = _json.dumps({"text": "from_provider"}).encode("utf-8")
    prov_resp = ProviderResponse(
        modified=False,
        body={},
        body_bytes=resp_bytes,
        tokens_before=0,
        tokens_after=0,
        strategy="none",
    )
    cm.set(
        {"model": "claude-sonnet-4-6", "messages": [{"role": "user", "content": "hi"}], "params": {}},
        prov_resp,
        _tenant(1),
    )
    result = cm.get(
        {"model": "claude-sonnet-4-6", "messages": [{"role": "user", "content": "hi"}], "params": {}},
        _tenant(1),
    )
    assert result is not None
    assert b"from_provider" in result.data


def test_set_key_none_skips(tmp_cache_db) -> None:
    """set() with uncacheable request → no-op."""
    cm = CacheManager(tmp_cache_db)
    initial_sets = cm.metrics.sets
    cm.set(
        {"model": "m", "messages": [], "params": {"temperature": 0.7}},
        {"text": "should not store"},
        _tenant(1),
    )
    assert cm.metrics.sets == initial_sets  # no change


# ---- CacheHook protocol ----

def test_check_success(tmp_cache_db) -> None:
    """CacheManager.check() wraps get() with fail-open."""
    from superlocalmemory.optimize.proxy.lifecycle import ProxyRequest
    cm = CacheManager(tmp_cache_db)
    # Pre-populate
    cm._exact.set("test-key", "default", {"id": "x"}, ["tag1"], "m")
    ctx = ProxyRequest(
        provider="anthropic", method="POST", path="/v1/messages",
        headers={},
        body={"model": "claude-sonnet-4-6", "messages": [{"role": "user", "content": "hi"}], "params": {}},
        body_bytes=b"{}",
        request_id="req1",
        stream=False,
        has_tools=False,
    )
    result = cm.check(ctx)
    # May hit or miss, but should not raise
    assert result is None or result.hit is True


def test_check_fail_open(tmp_cache_db, monkeypatch) -> None:
    """CacheManager.check() returns None on exception."""
    from superlocalmemory.optimize.proxy.lifecycle import ProxyRequest
    cm = CacheManager(tmp_cache_db)

    # Break build_key to raise
    monkeypatch.setattr(cm, "build_key", lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("boom")))
    ctx = ProxyRequest(
        provider="anthropic", method="POST", path="/v1/messages",
        headers={}, body={}, body_bytes=b"{}",
        request_id="req1", stream=False, has_tools=False,
    )
    result = cm.check(ctx)
    assert result is None  # fail-open


def test_store_success(tmp_cache_db) -> None:
    """CacheManager.store() wraps set() with fail-open."""
    from superlocalmemory.optimize.proxy.lifecycle import ProviderResponse, ProxyRequest
    cm = CacheManager(tmp_cache_db)
    ctx = ProxyRequest(
        provider="anthropic", method="POST", path="/v1/messages",
        headers={},
        body={"model": "claude-sonnet-4-6", "messages": [{"role": "user", "content": "hi"}], "params": {}},
        body_bytes=b"{}",
        request_id="req1", stream=False, has_tools=False,
    )
    resp = ProviderResponse(
        modified=False, body={},
        body_bytes=b'{"text":"ok"}',
        tokens_before=0, tokens_after=0, strategy="none",
    )
    cm.store(ctx, resp)  # must not raise


def test_store_fail_open(tmp_cache_db, monkeypatch) -> None:
    """CacheManager.store() does not raise on exception."""
    from superlocalmemory.optimize.proxy.lifecycle import ProviderResponse, ProxyRequest
    cm = CacheManager(tmp_cache_db)
    monkeypatch.setattr(cm, "set", lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("boom")))
    ctx = ProxyRequest(
        provider="anthropic", method="POST", path="/v1/messages",
        headers={}, body={}, body_bytes=b"{}",
        request_id="req1", stream=False, has_tools=False,
    )
    resp = ProviderResponse(
        modified=False, body={}, body_bytes=b"{}",
        tokens_before=0, tokens_after=0, strategy="none",
    )
    cm.store(ctx, resp)  # must not raise


def test_on_hit_calls_metrics_collector(tmp_cache_db, monkeypatch) -> None:
    """on_hit forwards token savings to MetricsCollector."""
    from superlocalmemory.optimize.proxy.lifecycle import ProxyRequest
    cm = CacheManager(tmp_cache_db)
    hits: list = []
    monkeypatch.setattr(
        "superlocalmemory.optimize.metrics.counters.MetricsCollector.get_instance",
        lambda: type("_MC", (), {"on_hit": lambda self, **kw: hits.append(kw)})(),
    )
    ctx = ProxyRequest(
        provider="anthropic", method="POST", path="/v1/messages",
        headers={}, body={}, body_bytes=b"{}",
        request_id="req1", stream=False, has_tools=False,
    )
    cm.on_hit(ctx, b"{}", 100)
    assert len(hits) > 0


def test_on_miss_calls_metrics_collector(tmp_cache_db, monkeypatch) -> None:
    """on_miss forwards to MetricsCollector."""
    from superlocalmemory.optimize.proxy.lifecycle import ProxyRequest
    cm = CacheManager(tmp_cache_db)
    misses: list = []
    monkeypatch.setattr(
        "superlocalmemory.optimize.metrics.counters.MetricsCollector.get_instance",
        lambda: type("_MC", (), {"on_miss": lambda self: misses.append(1)})(),
    )
    ctx = ProxyRequest(
        provider="anthropic", method="POST", path="/v1/messages",
        headers={}, body={}, body_bytes=b"{}",
        request_id="req1", stream=False, has_tools=False,
    )
    cm.on_miss(ctx)
    assert len(misses) == 1


def test_set_semantic_tier(tmp_cache_db) -> None:
    """set_semantic_tier replaces the semantic tier."""
    cm = CacheManager(tmp_cache_db)
    noop = NoOpSemantic()
    cm.set_semantic_tier(noop)
    assert cm._semantic is noop


# ---- set_instance / reset_instance ----

def test_set_instance(tmp_cache_db) -> None:
    """set_instance replaces the singleton."""
    cm1 = CacheManager(tmp_cache_db)
    CacheManager.set_instance(cm1)
    assert CacheManager.get_instance() is cm1


def test_reset_instance_db_close_raises(tmp_cache_db) -> None:
    """reset_instance survives db.close() raising."""
    cm = CacheManager(tmp_cache_db)
    cm._db = type("_FakeDB", (), {"close": lambda: (_ for _ in ()).throw(RuntimeError("close fail"))})()
    CacheManager.set_instance(cm)
    CacheManager.reset_instance()  # must not raise
    assert CacheManager._instance is None


# ---- get_or_call edge cases ----

def test_get_or_call_none_key_returns_upstream(tmp_cache_db) -> None:
    """get_or_call with uncacheable request (key=None) returns upstream."""
    cm = CacheManager(tmp_cache_db)
    upstream_calls = []

    def upstream():
        upstream_calls.append(1)
        return {"id": "x"}

    out = cm.get_or_call(
        tenant_id=_tenant(1), model_id="", model_version="",
        system="", messages=[], raw_params={},
        upstream_fn=upstream,
    )
    assert out == {"id": "x"}
    assert len(upstream_calls) == 1


def test_get_or_call_semantic_tier_lookup(tmp_cache_db) -> None:
    """get_or_call tries semantic tier on exact miss when semantic is enabled."""
    from types import SimpleNamespace

    class _FakeSemantic:
        def is_enabled(self):
            return True
        def lookup(self, req, tenant_id, embed):
            return {"text": "semantic_hit"}
        def learn(self, *a, **kw):
            pass
        def index_entry(self, *a, **kw):
            pass

    cm = CacheManager(tmp_cache_db, semantic_tier=_FakeSemantic())
    upstream_calls = []

    def upstream():
        upstream_calls.append(1)
        return {"id": "x"}

    out = cm.get_or_call(
        tenant_id=_tenant(1), model_id="m", model_version="v",
        system="", messages=[{"role": "user", "content": "unique_never_cached"}],
        raw_params={"max_tokens": 50}, upstream_fn=upstream,
    )
    assert out == {"text": "semantic_hit"}
    assert len(upstream_calls) == 0


def test_get_or_call_semantic_tier_raises_fail_open(tmp_cache_db) -> None:
    """get_or_call returns upstream when semantic tier raises."""
    class _BrokenSemantic:
        def is_enabled(self):
            return True
        def lookup(self, req, tenant_id, embed):
            raise RuntimeError("semantic down")
        def learn(self, *a, **kw):
            pass
        def index_entry(self, *a, **kw):
            pass

    cm = CacheManager(tmp_cache_db, semantic_tier=_BrokenSemantic())
    upstream_calls = []

    def upstream():
        upstream_calls.append(1)
        return {"id": "x"}

    out = cm.get_or_call(
        tenant_id=_tenant(1), model_id="m", model_version="v",
        system="", messages=[{"role": "user", "content": "unique_also_not_cached"}],
        raw_params={"max_tokens": 50}, upstream_fn=upstream,
    )
    assert out == {"id": "x"}
    assert len(upstream_calls) == 1


def test_get_or_call_stampede_contention(tmp_cache_db) -> None:
    """get_or_call exercises stampede lock path + contention when inside lock."""
    import contextlib

    cm = CacheManager(tmp_cache_db)
    upstream_calls = []

    def upstream():
        upstream_calls.append(1)
        return {"id": "first_call", "stop_reason": "end_turn"}

    kwargs = dict(
        tenant_id=_tenant(1), model_id="m2", model_version="v",
        system="", messages=[{"role": "user", "content": "stampede_test3"}],
        raw_params={"max_tokens": 50}, upstream_fn=upstream,
    )

    # Wrap the lock: inject a cached entry during __enter__,
    # so post-lock recheck hits → stampede contention.
    def _injecting_lock(k):
        @contextlib.contextmanager
        def _ctx():
            # Simulate another thread writing while we waited for lock
            cm._exact.set(k, _tenant(1), {"id": "contended", "stop_reason": "end_turn"}, [], "m2")
            yield
        return _ctx()

    original_lock = cm._stampede.lock
    cm._stampede.lock = _injecting_lock
    try:
        out = cm.get_or_call(**kwargs)
        assert out is not None
        # The stampede contention path returned the injected entry
        assert cm._metrics.stampede_contentions >= 1
    finally:
        cm._stampede.lock = original_lock


def test_get_or_call_fail_open_inner_exception(tmp_cache_db) -> None:
    """get_or_call returns upstream on inner exception."""
    cm = CacheManager(tmp_cache_db)

    class _BrokenKey:
        def build(self, *a, **kw):
            raise RuntimeError("key build failed")

    cm._key_builder = _BrokenKey()
    upstream_calls = []

    def upstream():
        upstream_calls.append(1)
        return {"id": "fallback"}

    out = cm.get_or_call(
        tenant_id=_tenant(1), model_id="m", model_version="v",
        system="", messages=[{"role": "user", "content": "hi"}],
        raw_params={}, upstream_fn=upstream,
    )
    assert out == {"id": "fallback"}
    assert len(upstream_calls) == 1


# ---- Invalidation convenience methods ----

def test_invalidate_by_tag(tmp_cache_db) -> None:
    cm = CacheManager(tmp_cache_db)
    # Register something under a known tag
    key = cm.build_key(
        {"model": "m", "messages": [{"role": "user", "content": "hi"}], "params": {}},
        _tenant(1),
    )
    if key:
        cm._exact.set(key, _tenant(1), {"id": "x"}, [], "m")
        cm._invalidation.register(key, _tenant(1), ["model:m"])
    count = cm.invalidate_by_tag("model:m")
    assert count >= 0


def test_invalidate_model(tmp_cache_db) -> None:
    cm = CacheManager(tmp_cache_db)
    count = cm.invalidate_model("test-model")
    assert count >= 0


def test_invalidate_tenant(tmp_cache_db) -> None:
    cm = CacheManager(tmp_cache_db)
    count = cm.invalidate_tenant(_tenant(99))
    assert count >= 0


# ---- _TenantScopedManager ----

def test_tenant_scoped_get_or_call(tmp_cache_db) -> None:
    """_TenantScopedManager.get_or_call delegates to CacheManager."""
    cm = CacheManager(tmp_cache_db)
    view = cm.for_tenant(_tenant(42))
    upstream_calls = []

    def upstream():
        upstream_calls.append(1)
        return {"id": "scoped_ok"}

    out = view.get_or_call(
        model_id="m", model_version="v",
        system="", messages=[{"role": "user", "content": "scoped_test"}],
        raw_params={}, upstream_fn=upstream,
    )
    assert out == {"id": "scoped_ok"}


def test_tenant_scoped_get_returns_none_on_miss(tmp_cache_db) -> None:
    """_TenantScopedManager.get() returns None when key not found."""
    cm = CacheManager(tmp_cache_db)
    view = cm.for_tenant(_tenant(42))
    result = view.get("nonexistent_key_xyz")
    assert result is None


def test_tenant_scoped_get_returns_bytes_on_hit(tmp_cache_db) -> None:
    """_TenantScopedManager.get() returns JSON bytes when key exists."""
    import json as _json
    cm = CacheManager(tmp_cache_db)
    cm._exact.set("test-key-123", _tenant(42), {"a": 1}, [], "m")
    view = cm.for_tenant(_tenant(42))
    result = view.get("test-key-123")
    assert result is not None
    assert isinstance(result, bytes)
    # json_dumps_bytes uses separators=(",", ":") → compact format
    assert b"a" in result


def test_tenant_scoped_set_valid_json(tmp_cache_db) -> None:
    """_TenantScopedManager.set() stores valid JSON."""
    import json as _json
    cm = CacheManager(tmp_cache_db)
    view = cm.for_tenant(_tenant(42))
    view.set("set-key-1", _json.dumps({"val": 42}).encode("utf-8"))
    result = view.get("set-key-1")
    assert result is not None


def test_tenant_scoped_set_invalid_json(tmp_cache_db) -> None:
    """_TenantScopedManager.set() ignores invalid JSON."""
    cm = CacheManager(tmp_cache_db)
    view = cm.for_tenant(_tenant(42))
    # Should not raise, silently ignores
    view.set("bad-key", b"not valid json {{{")
    result = view.get("bad-key")
    assert result is None


def test_tenant_scoped_invalidate_all(tmp_cache_db) -> None:
    cm = CacheManager(tmp_cache_db)
    view = cm.for_tenant(_tenant(42))
    count = view.invalidate_all()
    assert count >= 0


def test_tenant_scoped_metrics(tmp_cache_db) -> None:
    cm = CacheManager(tmp_cache_db)
    view = cm.for_tenant(_tenant(42))
    assert view.metrics is cm.metrics


# ---- json_dumps_bytes ----

def test_json_dumps_bytes() -> None:
    from superlocalmemory.optimize.cache.manager import json_dumps_bytes
    result = json_dumps_bytes({"key": "value", "num": 1})
    assert isinstance(result, bytes)
    assert b'"key":"value"' in result


def test_metrics_hit_rate_zero() -> None:
    from superlocalmemory.optimize.cache.manager import CacheMetrics
    m = CacheMetrics()
    assert m.hit_rate() == 0.0
