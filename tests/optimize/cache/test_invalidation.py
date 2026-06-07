"""LLD-02 §8.4 — InvalidationEngine tests."""

from __future__ import annotations

import pytest

from superlocalmemory.optimize.cache.invalidation import InvalidationEngine


def _tenant(n: int) -> str:
    return f"{n:064x}"


def test_register_and_invalidate(tmp_cache_db) -> None:
    inv = InvalidationEngine(tmp_cache_db)
    tmp_cache_db.set("k1", _tenant(1), b"v", model="m", ttl_expires=None, tags=[])
    inv.register("k1", _tenant(1), ["session:abc"])
    count = inv.invalidate_tag("session:abc")
    assert count == 1
    assert tmp_cache_db.get("k1", _tenant(1)) is None


def test_invalidate_model_uses_model_tag(tmp_cache_db) -> None:
    inv = InvalidationEngine(tmp_cache_db)
    tmp_cache_db.set("k1", _tenant(1), b"v", model="claude-sonnet-4-6", ttl_expires=None, tags=[])
    inv.register("k1", _tenant(1), ["model:claude-sonnet-4-6"])
    count = inv.invalidate_model("claude-sonnet-4-6")
    assert count == 1


def test_invalidate_tenant_uses_tenant_tag(tmp_cache_db) -> None:
    inv = InvalidationEngine(tmp_cache_db)
    tmp_cache_db.set("k1", _tenant(1), b"v", model="m", ttl_expires=None, tags=[])
    inv.register("k1", _tenant(1), ["tenant:abc"])
    count = inv.invalidate_tenant("abc")
    assert count == 1


def test_invalidate_key(tmp_cache_db) -> None:
    inv = InvalidationEngine(tmp_cache_db)
    tmp_cache_db.set("k1", _tenant(1), b"v", model="m", ttl_expires=None, tags=[])
    inv.invalidate_key("k1", _tenant(1))
    assert tmp_cache_db.get("k1", _tenant(1)) is None


def test_get_tags_for_key(tmp_cache_db) -> None:
    inv = InvalidationEngine(tmp_cache_db)
    tmp_cache_db.set("k1", _tenant(1), b"v", model="m", ttl_expires=None, tags=["a", "b"])
    tags = inv.get_tags_for_key("k1", _tenant(1))
    assert "a" in tags
    assert "b" in tags


def test_get_tags_for_missing_key(tmp_cache_db) -> None:
    inv = InvalidationEngine(tmp_cache_db)
    tags = inv.get_tags_for_key("nonexistent", _tenant(1))
    assert tags == []
