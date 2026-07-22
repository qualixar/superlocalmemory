"""Tests for ccr.py — CCRStore byte-exact recovery."""
from __future__ import annotations

import time

import pytest

from superlocalmemory.optimize.compress.ccr import CCRStore


class _BrokenCacheDB:
    """Always raises on any CCR method."""
    def ccr_put(self, *args, **kwargs) -> None:
        raise RuntimeError("simulated DB failure")

    def ccr_get(self, *args, **kwargs) -> None:
        raise RuntimeError("simulated DB failure")


def test_ccr_byte_exact_recovery(tmp_cache_db) -> None:
    """CRITICAL: Retrieved original must be byte-exact copy of stored original."""
    store = CCRStore()
    store._db = tmp_cache_db

    original = b"This is the full original content with all details preserved."
    ccr_id = store.store(original)
    assert ccr_id, "CCR store returned empty ccr_id"

    retrieved = store.retrieve(ccr_id)
    assert retrieved == original, (
        f"Retrieved bytes != original bytes. "
        f"original={original!r} retrieved={retrieved!r}"
    )


def test_ccr_store_returns_empty_on_db_failure() -> None:
    store = CCRStore()
    store._db = _BrokenCacheDB()
    ccr_id = store.store(b"original")
    assert ccr_id == ""


def test_ccr_retrieve_returns_none_on_unknown_id() -> None:
    store = CCRStore()
    result = store.retrieve("00000000-0000-0000-0000-000000000000")
    assert result is None


def test_ccr_retrieve_returns_none_on_expired(tmp_cache_db) -> None:
    store = CCRStore()
    store._db = tmp_cache_db
    ccr_id = store.store(b"original", ttl_seconds=0)
    time.sleep(0.2)
    retrieved = store.retrieve(ccr_id)
    assert retrieved is None, f"Expected None for expired entry, got {retrieved!r}"


def test_ccr_update_compressed(tmp_cache_db) -> None:
    """ccr_update_compressed updates the compressed_hash for a stored entry."""
    store = CCRStore()
    store._db = tmp_cache_db
    ccr_id = store.store(b"original text")
    assert ccr_id

    # Should not raise
    store.update_compressed(ccr_id, b"compressed text")
    # Original still retrievable
    retrieved = store.retrieve(ccr_id)
    assert retrieved == b"original text"


def test_ccr_multiple_originals_independent(tmp_cache_db) -> None:
    store = CCRStore()
    store._db = tmp_cache_db
    id1 = store.store(b"first")
    id2 = store.store(b"second")

    assert id1 != id2
    assert store.retrieve(id1) == b"first"
    assert store.retrieve(id2) == b"second"


# ---- get_instance singleton ----

def test_ccr_get_instance_returns_same_object() -> None:
    """get_instance() singleton returns same instance across calls."""
    # Reset first
    CCRStore._instance = None
    a = CCRStore.get_instance()
    b = CCRStore.get_instance()
    assert a is b


# ---- update_compressed error path ----

def test_ccr_update_compressed_error_non_fatal() -> None:
    """update_compressed swallows errors from _get_db failing."""
    store = CCRStore()
    # _db is None; _get_db() tries to import CacheDB which will work but
    # we can force an error by making _get_db itself raise.
    original_get_db = store._get_db
    try:
        store._get_db = lambda: (_ for _ in ()).throw(RuntimeError("simulated"))
        # Must not raise
        store.update_compressed("any-id", b"data")
    finally:
        store._get_db = original_get_db


# ---- retrieve error path ----

def test_ccr_retrieve_db_error_returns_none() -> None:
    """retrieve returns None when db.ccr_get raises."""
    store = CCRStore()
    store._db = _BrokenCacheDB()
    result = store.retrieve("any-id")
    assert result is None


# ---- store with TTL ----

def test_ccr_store_with_ttl_expires_after_expiry(tmp_cache_db) -> None:
    """store with ttl_seconds properly sets expiration."""
    import time
    store = CCRStore()
    store._db = tmp_cache_db
    ccr_id = store.store(b"original text", ttl_seconds=0)
    assert ccr_id
    time.sleep(0.2)
    # Should be expired
    assert store.retrieve(ccr_id) is None
