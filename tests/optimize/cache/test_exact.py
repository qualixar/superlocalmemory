"""LLD-02 §8.2 + LLD-10 P1 — ExactCache tests."""

from __future__ import annotations

import time

import pytest

from superlocalmemory.optimize.cache.exact import ExactCache
from superlocalmemory.optimize.cache.key_builder import CacheConfig


def _tenant(n: int) -> str:
    return f"{n:064x}"


def test_set_and_get(tmp_cache_db) -> None:
    ec = ExactCache(tmp_cache_db)
    payload = {"id": "msg_1", "type": "message", "stop_reason": "end_turn"}
    ok = ec.set("k1", _tenant(1), payload, tags=["session:abc"], model="claude-sonnet-4-6", ttl=300)
    assert ok is True
    out = ec.get("k1", _tenant(1))
    assert out == payload


def test_tool_call_response_not_cached(tmp_cache_db) -> None:
    """F2 / P1 gate: stop_reason='tool_use' must not be cached."""
    ec = ExactCache(tmp_cache_db)
    payload = {
        "id": "msg_2", "type": "message",
        "content": [{"type": "tool_use", "id": "toolu_x", "name": "bash", "input": {}}],
        "stop_reason": "tool_use",
    }
    assert ec.set("k1", _tenant(1), payload, tags=[], model="m") is False
    assert ec.get("k1", _tenant(1)) is None


def test_tool_calls_openai_not_cached(tmp_cache_db) -> None:
    payload = {
        "choices": [{
            "message": {"role": "assistant", "tool_calls": [{"id": "call_x"}]},
            "finish_reason": "tool_calls",
        }],
    }
    ec = ExactCache(tmp_cache_db)
    assert ec.set("k1", _tenant(1), payload, tags=[], model="m") is False


def test_ttl_expiry_serves_miss(tmp_cache_db) -> None:
    """F3 / P1 gate: TTL expiry serves a miss."""
    ec = ExactCache(tmp_cache_db)
    payload = {"id": "x", "stop_reason": "end_turn"}
    assert ec.set("k1", _tenant(1), payload, tags=[], model="m", ttl=1) is True
    time.sleep(1.1)
    assert ec.get("k1", _tenant(1)) is None


def test_non_cacheable_response_skipped(tmp_cache_db) -> None:
    ec = ExactCache(tmp_cache_db)
    # finish_reason=length must not be cached (A-18 fix)
    payload = {"id": "x", "finish_reason": "length"}
    assert ec.set("k1", _tenant(1), payload, tags=[], model="m") is False


def test_tenant_isolation(tmp_cache_db) -> None:
    """F4 / P1 gate: same key, different tenants → separate entries."""
    ec = ExactCache(tmp_cache_db)
    payload = {"id": "x", "stop_reason": "end_turn"}
    ec.set("k1", _tenant(1), payload, tags=[], model="m", ttl=300)
    # get for tenant 2 must miss (cache_key includes tenant_id in the full path,
    # but in our key builder we pass raw key — here the key already encodes
    # isolation via the SQLite (cache_key, tenant_id) PK).
    assert ec.get("k1", _tenant(1)) == payload
    assert ec.get("k1", _tenant(2)) is None


def test_delete_removes_entry(tmp_cache_db) -> None:
    ec = ExactCache(tmp_cache_db)
    payload = {"id": "x", "stop_reason": "end_turn"}
    ec.set("k1", _tenant(1), payload, tags=[], model="m", ttl=300)
    ec.delete("k1", _tenant(1))
    assert ec.get("k1", _tenant(1)) is None


def test_is_cacheable_response_helper() -> None:
    from superlocalmemory.optimize.cache.exact import _is_cacheable_response
    assert _is_cacheable_response({"stop_reason": "end_turn"}) is True
    assert _is_cacheable_response({"stop_reason": "tool_use"}) is False
    assert _is_cacheable_response({"choices": [{"finish_reason": "stop"}]}) is True
    assert _is_cacheable_response({"choices": [{"finish_reason": "length"}]}) is False
    assert _is_cacheable_response({"content": [{"type": "text"}]}) is True
    assert _is_cacheable_response({"content": [{"type": "tool_use"}]}) is False
