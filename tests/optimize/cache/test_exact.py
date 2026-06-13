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


def test_c05_debug_log_on_tool_use_skip(caplog) -> None:
    """C-05: _is_cacheable_response must emit a debug log when skipping tool_use response."""
    import logging
    from superlocalmemory.optimize.cache.exact import _is_cacheable_response

    with caplog.at_level(logging.DEBUG, logger="superlocalmemory.optimize.cache.exact"):
        result = _is_cacheable_response({"stop_reason": "tool_use"})

    assert result is False
    assert any("tool_use" in rec.message or "finish_reason" in rec.message
               for rec in caplog.records), (
        f"C-05: no debug log emitted for tool_use skip; records={[r.message for r in caplog.records]}"
    )


def test_c09_no_python_ttl_recheck(tmp_cache_db) -> None:
    """C-09: ExactCache.get() must not call db.delete() on TTL-expired rows.

    The SQL in db.get() already filters ttl_expires > now(), so returned rows
    are never expired. The Python-side recheck is dead code and was removed.
    """
    from unittest.mock import patch
    ec = ExactCache(tmp_cache_db)
    payload = {"id": "x", "stop_reason": "end_turn"}
    ec.set("k-c09", _tenant(1), payload, tags=[], model="m", ttl=300)

    delete_calls = []
    original_delete = tmp_cache_db.delete
    with patch.object(tmp_cache_db, "delete", side_effect=lambda *a: delete_calls.append(a) or original_delete(*a)):
        out = ec.get("k-c09", _tenant(1))

    assert out == payload, "get() must return the cached value"
    assert len(delete_calls) == 0, (
        f"C-09: get() must not call db.delete() on a live entry; got {len(delete_calls)} call(s)"
    )


def test_c07_tags_passed_to_db_set(tmp_cache_db) -> None:
    """C-07: ExactCache.set() must thread actual tags to db.set (not hardcoded [])."""
    from unittest.mock import patch, MagicMock

    ec = ExactCache(tmp_cache_db)
    tags = ["model:claude-3", "tenant:abc123"]
    payload = {"id": "x", "stop_reason": "end_turn"}

    calls = []
    original_set = tmp_cache_db.set
    def capturing_set(**kwargs):
        calls.append(kwargs)
        return original_set(**kwargs)

    with patch.object(tmp_cache_db, "set", side_effect=capturing_set):
        ec.set("k-c07", "a" * 64, payload, tags=tags, model="m", ttl=300)

    assert len(calls) == 1
    assert calls[0]["tags"] == tags, (
        f"C-07 regression: tags passed to db.set was {calls[0].get('tags')!r}, expected {tags!r}"
    )
