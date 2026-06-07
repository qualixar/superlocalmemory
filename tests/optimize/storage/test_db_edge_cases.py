"""storage/db.py edge-case coverage — boost from 60% to 80%+.

Focuses on paths that were missing tests: vec_search, vec_delete,
get_all_vectors, boundary_*, centroid_*, ccr_*, get_by_id, sweep_expired,
delete_batch, clear_tenant, entry_count, db_size_bytes, get_default.
"""

from __future__ import annotations

import sqlite3
import struct
import time
import uuid
from pathlib import Path

import pytest


def _tenant(n: int = 1) -> str:
    return f"{n:064x}"


# ---- vec_* operations ----

def test_vec_add_and_search(tmp_cache_db) -> None:
    tmp_cache_db.set("k1", _tenant(), b"v", model="m", ttl_expires=None, tags=[])
    row = tmp_cache_db.get("k1", _tenant())
    assert row is not None
    eid = row.entry_id
    # 4-dim vector [1, 0, 0, 0]
    v1 = struct.pack("4f", 1.0, 0.0, 0.0, 0.0)
    v2 = struct.pack("4f", 0.0, 1.0, 0.0, 0.0)
    v3 = struct.pack("4f", 0.9, 0.1, 0.0, 0.0)  # close to v1

    tmp_cache_db.vec_add(eid, _tenant(), v1, meta={"dim": 4, "model": "test"})
    # Get entry_id of second entry
    tmp_cache_db.set("k2", _tenant(), b"v2", model="m", ttl_expires=None, tags=[])
    row2 = tmp_cache_db.get("k2", _tenant())
    tmp_cache_db.vec_add(row2.entry_id, _tenant(), v2, meta={"dim": 4, "model": "test"})
    tmp_cache_db.set("k3", _tenant(), b"v3", model="m", ttl_expires=None, tags=[])
    row3 = tmp_cache_db.get("k3", _tenant())
    tmp_cache_db.vec_add(row3.entry_id, _tenant(), v3, meta={"dim": 4, "model": "test"})

    # Search with v1 as query
    results = tmp_cache_db.vec_search(_tenant(), v1, top_k=2)
    assert len(results) == 2
    # First result should be the exact match (cos=1.0)
    assert results[0][0] == eid
    assert results[0][1] == pytest.approx(1.0, abs=1e-5)


def test_vec_search_empty_tenant(tmp_cache_db) -> None:
    """No vectors for tenant → empty list."""
    v = struct.pack("4f", 1.0, 0.0, 0.0, 0.0)
    results = tmp_cache_db.vec_search(_tenant(), v, top_k=10)
    assert results == []


def test_vec_delete(tmp_cache_db) -> None:
    tmp_cache_db.set("k1", _tenant(), b"v", model="m", ttl_expires=None, tags=[])
    row = tmp_cache_db.get("k1", _tenant())
    v = struct.pack("4f", 1.0, 0.0, 0.0, 0.0)
    tmp_cache_db.vec_add(row.entry_id, _tenant(), v, meta={"dim": 4, "model": "test"})
    tmp_cache_db.vec_delete(row.entry_id)
    assert tmp_cache_db.get_all_vectors(_tenant()) == []


def test_get_all_vectors(tmp_cache_db) -> None:
    tmp_cache_db.set("k1", _tenant(), b"v", model="m", ttl_expires=None, tags=[])
    row = tmp_cache_db.get("k1", _tenant())
    v = struct.pack("4f", 1.0, 2.0, 3.0, 4.0)
    tmp_cache_db.vec_add(row.entry_id, _tenant(), v, meta={"dim": 4, "model": "test"})
    all_vecs = tmp_cache_db.get_all_vectors(_tenant())
    assert len(all_vecs) == 1
    assert all_vecs[0][0] == row.entry_id
    assert all_vecs[0][1] == v


def test_vec_search_tenant_isolation(tmp_cache_db) -> None:
    """Vectors for tenant A must not appear in tenant B's search."""
    tmp_cache_db.set("k1", _tenant(1), b"v", model="m", ttl_expires=None, tags=[])
    row = tmp_cache_db.get("k1", _tenant(1))
    v = struct.pack("4f", 1.0, 0.0, 0.0, 0.0)
    tmp_cache_db.vec_add(row.entry_id, _tenant(1), v, meta={"dim": 4, "model": "test"})

    results = tmp_cache_db.vec_search(_tenant(2), v, top_k=10)
    assert results == []


# ---- get_by_id ----

def test_get_by_id_roundtrip(tmp_cache_db) -> None:
    value = b"secret-bytes"
    tmp_cache_db.set("k1", _tenant(), value, model="m", ttl_expires=None, tags=[])
    row = tmp_cache_db.get("k1", _tenant())
    assert row is not None
    fetched = tmp_cache_db.get_by_id(row.entry_id)
    assert fetched is not None
    assert fetched.value == value


def test_get_by_id_missing(tmp_cache_db) -> None:
    assert tmp_cache_db.get_by_id("nonexistent-id") is None


# ---- delete_batch ----

def test_delete_batch_empty(tmp_cache_db) -> None:
    """Empty list → 0, no-op."""
    assert tmp_cache_db.delete_batch([]) == 0


def test_delete_batch_multiple_keys(tmp_cache_db) -> None:
    tmp_cache_db.set("k1", _tenant(1), b"v1", model="m", ttl_expires=None, tags=[])
    tmp_cache_db.set("k2", _tenant(1), b"v2", model="m", ttl_expires=None, tags=[])
    tmp_cache_db.set("k3", _tenant(2), b"v3", model="m", ttl_expires=None, tags=[])

    deleted = tmp_cache_db.delete_batch([("k1", _tenant(1)), ("k3", _tenant(2))])
    assert deleted == 2
    assert tmp_cache_db.get("k1", _tenant(1)) is None
    assert tmp_cache_db.get("k2", _tenant(1)) is not None
    assert tmp_cache_db.get("k3", _tenant(2)) is None


# ---- entry_count + db_size_bytes ----

def test_entry_count(tmp_cache_db) -> None:
    assert tmp_cache_db.entry_count(_tenant()) == 0
    tmp_cache_db.set("k1", _tenant(), b"v1", model="m", ttl_expires=None, tags=[])
    tmp_cache_db.set("k2", _tenant(), b"v2", model="m", ttl_expires=None, tags=[])
    assert tmp_cache_db.entry_count(_tenant()) == 2


def test_entry_count_excludes_expired(tmp_cache_db) -> None:
    tmp_cache_db.set("k1", _tenant(), b"v", model="m",
                     ttl_expires=time.time() - 1, tags=[])
    tmp_cache_db.set("k2", _tenant(), b"v", model="m",
                     ttl_expires=time.time() + 9999, tags=[])
    assert tmp_cache_db.entry_count(_tenant()) == 1


def test_db_size_bytes(tmp_cache_db) -> None:
    size = tmp_cache_db.db_size_bytes()
    assert size > 0


# ---- clear_tenant ----

def test_clear_tenant_removes_all_entries_and_tags(tmp_cache_db) -> None:
    tmp_cache_db.set("k1", _tenant(), b"v", model="m", ttl_expires=None, tags=[])
    tmp_cache_db.set("k2", _tenant(), b"v", model="m", ttl_expires=None, tags=[])
    tmp_cache_db.tag_register("k1", _tenant(), ["session:abc"])
    tmp_cache_db.tag_register("k2", _tenant(), ["session:xyz"])
    cleared = tmp_cache_db.clear_tenant(_tenant())
    assert cleared == 2
    assert tmp_cache_db.entry_count(_tenant()) == 0
    assert tmp_cache_db.get_all_vectors(_tenant()) == []


# ---- boundary_* + centroid_* ----

def test_boundary_upsert_and_get(tmp_cache_db) -> None:
    from superlocalmemory.optimize.storage.db import BoundaryRow
    boundary = BoundaryRow(
        entry_id="eid-1", logistic_t=0.93, logistic_gamma=8.5, sample_count=10,
        updated_at=time.time(),
    )
    tmp_cache_db.boundary_upsert("eid-1", boundary)
    fetched = tmp_cache_db.boundary_get("eid-1")
    assert fetched is not None
    assert fetched.logistic_t == pytest.approx(0.93)
    assert fetched.logistic_gamma == pytest.approx(8.5)
    assert fetched.sample_count == 10


def test_boundary_get_missing(tmp_cache_db) -> None:
    assert tmp_cache_db.boundary_get("nonexistent") is None


def test_boundary_upsert_updates_existing(tmp_cache_db) -> None:
    from superlocalmemory.optimize.storage.db import BoundaryRow
    tmp_cache_db.boundary_upsert("eid-1", BoundaryRow(entry_id="eid-1", logistic_t=0.9))
    tmp_cache_db.boundary_upsert("eid-1", BoundaryRow(entry_id="eid-1", logistic_t=0.95))
    fetched = tmp_cache_db.boundary_get("eid-1")
    assert fetched.logistic_t == pytest.approx(0.95)


def test_centroid_get_missing(tmp_cache_db) -> None:
    assert tmp_cache_db.centroid_get("nonexistent") is None


def test_centroid_upsert_and_get(tmp_cache_db) -> None:
    blob = struct.pack("4f", 0.1, 0.2, 0.3, 0.4)
    tmp_cache_db.centroid_update(_tenant(), blob, n=100)
    fetched = tmp_cache_db.centroid_get(_tenant())
    assert fetched == blob


def test_centroid_upsert_updates_existing(tmp_cache_db) -> None:
    tmp_cache_db.centroid_update(_tenant(), struct.pack("4f", 1.0, 0, 0, 0), n=10)
    tmp_cache_db.centroid_update(_tenant(), struct.pack("4f", 0.5, 0.5, 0, 0), n=20)
    assert tmp_cache_db.centroid_get(_tenant()) == struct.pack("4f", 0.5, 0.5, 0, 0)


# ---- ccr_update_compressed ----

def test_ccr_update_compressed(tmp_cache_db) -> None:
    ccr_id = uuid.uuid4().hex
    tmp_cache_db.ccr_put(ccr_id, b"original-context-12345")
    # Update with new compressed bytes
    tmp_cache_db.ccr_update_compressed(ccr_id, b"new-compressed-bytes")
    # The CCR row still exists and can be retrieved
    assert tmp_cache_db.ccr_get(ccr_id) == b"original-context-12345"


def test_ccr_update_compressed_missing(tmp_cache_db) -> None:
    """Update on non-existent ccr_id → no-op (doesn't raise)."""
    tmp_cache_db.ccr_update_compressed("nonexistent", b"x")  # must not raise


# ---- ccr_get after TTL expiry ----

def test_ccr_get_after_ttl_expiry(tmp_cache_db) -> None:
    ccr_id = uuid.uuid4().hex
    tmp_cache_db.ccr_put(ccr_id, b"context")
    # Expire it
    tmp_cache_db._db.execute(
        "UPDATE llmcache_ccr_originals SET ttl_expires = ? WHERE ccr_id = ?",
        (time.time() - 1, ccr_id),
    )
    assert tmp_cache_db.ccr_get(ccr_id) is None


# ---- tag_keys ----

def test_tag_keys_returns_pairs(tmp_cache_db) -> None:
    tmp_cache_db.set("k1", _tenant(1), b"v", model="m", ttl_expires=None, tags=[])
    tmp_cache_db.set("k2", _tenant(2), b"v", model="m", ttl_expires=None, tags=[])
    tmp_cache_db.tag_register("k1", _tenant(1), ["session:abc"])
    tmp_cache_db.tag_register("k2", _tenant(2), ["session:abc"])
    pairs = tmp_cache_db.tag_keys("session:abc")
    assert len(pairs) == 2
    assert ("k1", _tenant(1)) in pairs
    assert ("k2", _tenant(2)) in pairs


def test_tag_keys_no_match(tmp_cache_db) -> None:
    assert tmp_cache_db.tag_keys("nonexistent-tag") == []


# ---- sweep_expired with multiple entries ----

def test_sweep_expired_count_is_correct(tmp_cache_db) -> None:
    past = time.time() - 1
    future = time.time() + 9999
    tmp_cache_db.set("e1", _tenant(), b"v", model="m", ttl_expires=past, tags=[])
    tmp_cache_db.set("e2", _tenant(), b"v", model="m", ttl_expires=past, tags=[])
    tmp_cache_db.set("e3", _tenant(), b"v", model="m", ttl_expires=future, tags=[])

    deleted = tmp_cache_db.sweep_expired(now=time.time())
    assert deleted == 2
    assert tmp_cache_db.get("e3", _tenant()) is not None


# ---- sweep_expired also cleans up tags + vectors ----

def test_sweep_expired_cascades_to_tags_and_vectors(tmp_cache_db) -> None:
    # Use a longer TTL so we can read the row before sweep
    future = time.time() + 9999
    tmp_cache_db.set("k1", _tenant(), b"v", model="m", ttl_expires=future, tags=[])
    row = tmp_cache_db.get("k1", _tenant())
    assert row is not None
    tmp_cache_db.tag_register("k1", _tenant(), ["session:abc"])
    tmp_cache_db.vec_add(row.entry_id, _tenant(),
                          struct.pack("4f", 1, 0, 0, 0),
                          meta={"dim": 4, "model": "test"})

    # Now expire it and sweep
    tmp_cache_db._db.execute(
        "UPDATE llmcache_entries SET ttl_expires = ? WHERE cache_key = ?",
        (time.time() - 1, "k1"),
    )
    tmp_cache_db.sweep_expired(now=time.time())

    # All should be cleaned up
    assert tmp_cache_db.get("k1", _tenant()) is None
    assert tmp_cache_db.tag_keys("session:abc") == []
    assert tmp_cache_db.get_all_vectors(_tenant()) == []


# ---- get_default singleton ----

def test_get_default_singleton(tmp_path: Path, monkeypatch) -> None:
    """get_default returns a singleton across calls."""
    from superlocalmemory.optimize.storage.db import CacheDB
    monkeypatch.setenv("HOME", str(tmp_path))
    CacheDB.reset_default()
    try:
        a = CacheDB.get_default()
        b = CacheDB.get_default()
        assert a is b
        # db_path property works
        assert a.db_path.endswith("llmcache.db")
    finally:
        CacheDB.reset_default()


# ---- schema drop_all_tables ----

def test_drop_all_tables_idempotent(tmp_path: Path) -> None:
    """drop_all_tables is safe to call twice."""
    from superlocalmemory.optimize.storage.schema import create_all_tables, drop_all_tables
    conn = sqlite3.connect(str(tmp_path / "x.db"))
    create_all_tables(conn)
    conn.commit()
    drop_all_tables(conn)
    conn.commit()
    drop_all_tables(conn)  # second call OK
    conn.close()


# ---- entry_exists ----

def test_entry_exists_true_and_false(tmp_cache_db) -> None:
    assert tmp_cache_db.entry_exists("k1", _tenant()) is False
    tmp_cache_db.set("k1", _tenant(), b"v", model="m", ttl_expires=None, tags=[])
    assert tmp_cache_db.entry_exists("k1", _tenant()) is True


# ---- __enter__/__exit__ context manager ----

def test_context_manager(tmp_path: Path) -> None:
    from superlocalmemory.optimize.storage.db import CacheDB
    db_path = tmp_path / "ctx.db"
    with CacheDB(db_path) as db:
        db.set("k1", _tenant(), b"v", model="m", ttl_expires=None, tags=[])
        assert db.get("k1", _tenant()) is not None
    # After exit, reopen and verify persisted
    db2 = CacheDB(db_path)
    assert db2.get("k1", _tenant()) is not None
    db2.close()


# ---- MetricsSnapshot unit tests (no DB needed) ----

def test_metrics_snapshot_hit_rate_zero_hits_and_misses() -> None:
    """hit_rate with hits=0, misses=0 returns 0.0."""
    from superlocalmemory.optimize.storage.db import MetricsSnapshot
    snap = MetricsSnapshot(hits=0, misses=0)
    assert snap.hit_rate == 0.0


def test_metrics_snapshot_hit_rate_zero_hits_some_misses() -> None:
    """hit_rate with hits=0, misses>0 returns 0.0."""
    from superlocalmemory.optimize.storage.db import MetricsSnapshot
    snap = MetricsSnapshot(hits=0, misses=5)
    assert snap.hit_rate == 0.0


def test_metrics_snapshot_avg_latency_zero_samples() -> None:
    """avg_latency_overhead_ms with 0 samples returns 0.0."""
    from superlocalmemory.optimize.storage.db import MetricsSnapshot
    snap = MetricsSnapshot(latency_overhead_ms_sum=500.0, latency_samples=0)
    assert snap.avg_latency_overhead_ms == 0.0


def test_metrics_snapshot_compression_ratio_zero_original() -> None:
    """compression_ratio with compress_bytes_original=0 returns 1.0."""
    from superlocalmemory.optimize.storage.db import MetricsSnapshot
    snap = MetricsSnapshot(compress_bytes_original=0, compress_bytes_after=500)
    assert snap.compression_ratio == 1.0


# ---- set_with_entry_id ----

def test_set_with_entry_id_stores_value(tmp_cache_db) -> None:
    """set_with_entry_id stores a pre-assigned entry_id that can be retrieved."""
    eid = uuid.uuid4().hex
    value = b"value-with-custom-entry-id"
    tmp_cache_db.set_with_entry_id(
        "custom-key", _tenant(), value, entry_id=eid
    )
    row = tmp_cache_db.get_by_id(eid)
    assert row is not None
    assert row.value == value
    assert row.cache_key == "custom-key"


def test_set_with_entry_id_with_ttl(tmp_cache_db) -> None:
    """set_with_entry_id with non-zero ttl_seconds sets expiration (checked via get)."""
    eid = uuid.uuid4().hex
    # Note: ttl_seconds=0 is treated as "no TTL" (Python truthiness check).
    # Use a very small positive value to set immediate expiry.
    tmp_cache_db.set_with_entry_id(
        "ttl-key", _tenant(), b"ttl-value",
        entry_id=eid, ttl_seconds=1,
    )
    time.sleep(1.5)
    # get() checks TTL, should return None after expiry
    row = tmp_cache_db.get("ttl-key", _tenant())
    assert row is None


# ---- get_entry_by_id (v2 method) ----

def test_get_entry_by_id_miss(tmp_cache_db) -> None:
    """get_entry_by_id returns None for non-existent ID."""
    assert tmp_cache_db.get_entry_by_id("nonexistent-eid") is None


def test_get_entry_by_id_roundtrip(tmp_cache_db) -> None:
    """get_entry_by_id returns decoded response dict."""
    import json as _json
    resp = {"text": "Hello world", "tokens": 42}
    value = _json.dumps(resp).encode("utf-8")
    tmp_cache_db.set("json-key", _tenant(), value, model="m", ttl_expires=None, tags=[])
    row = tmp_cache_db.get("json-key", _tenant())
    assert row is not None
    decoded = tmp_cache_db.get_entry_by_id(row.entry_id)
    assert decoded is not None
    assert decoded["text"] == "Hello world"
    assert decoded["tokens"] == 42


def test_get_entry_by_id_non_json_bytes(tmp_cache_db) -> None:
    """get_entry_by_id falls back to raw_bytes hex for non-JSON content."""
    raw_bytes = b'\x00\x01\x02\xff'
    tmp_cache_db.set("bin-key", _tenant(), raw_bytes, model="m", ttl_expires=None, tags=[])
    row = tmp_cache_db.get("bin-key", _tenant())
    assert row is not None
    decoded = tmp_cache_db.get_entry_by_id(row.entry_id)
    assert decoded is not None
    assert "raw_bytes" in decoded
    assert decoded["raw_bytes"] == raw_bytes.hex()


# ---- _row_to_cacherow with bad tag_json ----

def test_row_to_cacherow_bad_tag_json(tmp_cache_db) -> None:
    """_row_to_cacherow handles invalid JSON in tag_json gracefully."""
    tmp_cache_db.set("k1", _tenant(), b"v", model="m", ttl_expires=None, tags=["good"])
    # Manually corrupt the tag_json
    tmp_cache_db._db.execute(
        "UPDATE llmcache_entries SET tag_json = ? WHERE cache_key = ?",
        ("NOT VALID JSON{{{", "k1"),
    )
    row = tmp_cache_db.get("k1", _tenant())
    assert row is not None
    assert row.tags == []


# ---- tag_register with empty tags ----

def test_tag_register_empty_tags_is_noop(tmp_cache_db) -> None:
    """Passing empty list to tag_register returns immediately (line 568)."""
    # Must not raise
    tmp_cache_db.tag_register("k1", _tenant(), [])


# ---- error paths via broken DB (sqlite3.Error) ----

def _make_broken_db(tmp_cache_db):
    """Replace _db.execute with one that raises sqlite3.Error."""
    import sqlite3
    original = tmp_cache_db._db.execute

    def _broken(*args, **kwargs):
        raise sqlite3.Error("simulated DB failure")

    tmp_cache_db._db.execute = _broken
    return original


def test_set_fail_open_on_db_error(tmp_cache_db) -> None:
    """set() must not raise when DB is broken."""
    original = _make_broken_db(tmp_cache_db)
    try:
        # Must not raise
        tmp_cache_db.set("k1", _tenant(), b"v", model="m", ttl_expires=None, tags=[])
    finally:
        tmp_cache_db._db.execute = original


def test_set_with_entry_id_fail_open_on_db_error(tmp_cache_db) -> None:
    """set_with_entry_id() must not raise when DB is broken."""
    original = _make_broken_db(tmp_cache_db)
    try:
        tmp_cache_db.set_with_entry_id("k1", _tenant(), b"v", entry_id="eid1")
    finally:
        tmp_cache_db._db.execute = original


def test_get_fail_open_on_db_error(tmp_cache_db) -> None:
    """get() returns None when DB is broken."""
    original = _make_broken_db(tmp_cache_db)
    try:
        result = tmp_cache_db.get("k1", _tenant())
        assert result is None
    finally:
        tmp_cache_db._db.execute = original


def test_get_by_id_fail_open_on_db_error(tmp_cache_db) -> None:
    """get_by_id() returns None when DB is broken."""
    original = _make_broken_db(tmp_cache_db)
    try:
        result = tmp_cache_db.get_by_id("eid1")
        assert result is None
    finally:
        tmp_cache_db._db.execute = original


def test_delete_fail_open_on_db_error(tmp_cache_db) -> None:
    """delete() must not raise when DB is broken."""
    original = _make_broken_db(tmp_cache_db)
    try:
        tmp_cache_db.delete("k1", _tenant())
    finally:
        tmp_cache_db._db.execute = original


def test_delete_batch_fail_open_on_db_error(tmp_cache_db) -> None:
    """delete_batch() returns 0 when DB is broken."""
    original = _make_broken_db(tmp_cache_db)
    try:
        result = tmp_cache_db.delete_batch([("k1", _tenant())])
        assert result == 0
    finally:
        tmp_cache_db._db.execute = original


def test_sweep_expired_fail_open_on_db_error(tmp_cache_db) -> None:
    """sweep_expired() returns 0 when DB is broken."""
    original = _make_broken_db(tmp_cache_db)
    try:
        result = tmp_cache_db.sweep_expired(now=time.time())
        assert result == 0
    finally:
        tmp_cache_db._db.execute = original


def test_tag_register_fail_open_on_db_error(tmp_cache_db) -> None:
    """tag_register() swallows error when DB is broken."""
    original = _make_broken_db(tmp_cache_db)
    try:
        tmp_cache_db.tag_register("k1", _tenant(), ["tag1"])
    finally:
        tmp_cache_db._db.execute = original


def test_tag_keys_fail_open_on_db_error(tmp_cache_db) -> None:
    """tag_keys() returns [] when DB is broken."""
    original = _make_broken_db(tmp_cache_db)
    try:
        result = tmp_cache_db.tag_keys("tag1")
        assert result == []
    finally:
        tmp_cache_db._db.execute = original


def test_invalidate_by_tag_fail_open_on_db_error(tmp_cache_db) -> None:
    """invalidate_by_tag() returns 0 when DB is broken."""
    original = _make_broken_db(tmp_cache_db)
    try:
        result = tmp_cache_db.invalidate_by_tag("tag1")
        assert result == 0
    finally:
        tmp_cache_db._db.execute = original


def test_vec_add_fail_open_on_db_error(tmp_cache_db) -> None:
    """vec_add() swallows error when DB is broken."""
    import struct
    original = _make_broken_db(tmp_cache_db)
    try:
        v = struct.pack("4f", 1.0, 0.0, 0.0, 0.0)
        tmp_cache_db.vec_add("eid", _tenant(), v, meta={"dim": 4})
    finally:
        tmp_cache_db._db.execute = original


def test_vec_delete_fail_open_on_db_error(tmp_cache_db) -> None:
    """vec_delete() swallows error when DB is broken."""
    original = _make_broken_db(tmp_cache_db)
    try:
        tmp_cache_db.vec_delete("eid")
    finally:
        tmp_cache_db._db.execute = original


def test_vec_search_fail_open_on_db_error(tmp_cache_db) -> None:
    """vec_search() returns [] when DB is broken."""
    import struct
    original = _make_broken_db(tmp_cache_db)
    try:
        v = struct.pack("4f", 1.0, 0.0, 0.0, 0.0)
        result = tmp_cache_db.vec_search(_tenant(), v, top_k=5)
        assert result == []
    finally:
        tmp_cache_db._db.execute = original


def test_vec_search_import_error(tmp_cache_db) -> None:
    """vec_search returns [] when numpy is not available."""
    import struct
    import builtins

    v = struct.pack("4f", 1.0, 0.0, 0.0, 0.0)
    tmp_cache_db.set("k1", _tenant(), b"v", model="m", ttl_expires=None, tags=[])
    row = tmp_cache_db.get("k1", _tenant())
    tmp_cache_db.vec_add(row.entry_id, _tenant(), v, meta={"dim": 4})

    original_import = builtins.__import__

    def _block_numpy(name, *args, **kwargs):
        if name == "numpy":
            raise ImportError("No numpy")
        return original_import(name, *args, **kwargs)

    import sys
    # Remove numpy from cache to trigger ImportError
    saved_numpy = sys.modules.get("numpy")
    sys.modules["numpy"] = None  # Force import failure next time

    try:
        # Need to force re-import; _np import is inside vec_search
        builtins.__import__ = _block_numpy
        result = tmp_cache_db.vec_search(_tenant(), v, top_k=5)
        assert result == []
    finally:
        builtins.__import__ = original_import
        if saved_numpy is not None:
            sys.modules["numpy"] = saved_numpy
        elif "numpy" in sys.modules:
            del sys.modules["numpy"]


def test_vec_search_zero_norm(tmp_cache_db) -> None:
    """vec_search returns [] when query vector has zero norm."""
    import struct
    # Zero vector
    v_zero = struct.pack("4f", 0.0, 0.0, 0.0, 0.0)
    v_normal = struct.pack("4f", 1.0, 0.0, 0.0, 0.0)

    tmp_cache_db.set("k1", _tenant(), b"v", model="m", ttl_expires=None, tags=[])
    row = tmp_cache_db.get("k1", _tenant())
    tmp_cache_db.vec_add(row.entry_id, _tenant(), v_normal, meta={"dim": 4})

    result = tmp_cache_db.vec_search(_tenant(), v_zero, top_k=5)
    assert result == []


def test_boundary_get_fail_open_on_db_error(tmp_cache_db) -> None:
    """boundary_get() returns None when DB is broken."""
    original = _make_broken_db(tmp_cache_db)
    try:
        result = tmp_cache_db.boundary_get("eid")
        assert result is None
    finally:
        tmp_cache_db._db.execute = original


def test_boundary_upsert_fail_open_on_db_error(tmp_cache_db) -> None:
    """boundary_upsert() swallows error when DB is broken."""
    from superlocalmemory.optimize.storage.db import BoundaryRow
    original = _make_broken_db(tmp_cache_db)
    try:
        tmp_cache_db.boundary_upsert("eid", BoundaryRow(entry_id="eid"))
    finally:
        tmp_cache_db._db.execute = original


def test_centroid_get_fail_open_on_db_error(tmp_cache_db) -> None:
    """centroid_get() returns None when DB is broken."""
    original = _make_broken_db(tmp_cache_db)
    try:
        result = tmp_cache_db.centroid_get(_tenant())
        assert result is None
    finally:
        tmp_cache_db._db.execute = original


def test_centroid_update_fail_open_on_db_error(tmp_cache_db) -> None:
    """centroid_update() swallows error when DB is broken."""
    import struct
    original = _make_broken_db(tmp_cache_db)
    try:
        tmp_cache_db.centroid_update(_tenant(), struct.pack("4f", 1, 0, 0, 0), n=1)
    finally:
        tmp_cache_db._db.execute = original


def test_get_all_boundaries_fail_open_on_db_error(tmp_cache_db) -> None:
    """get_all_boundaries() returns [] when DB is broken."""
    original = _make_broken_db(tmp_cache_db)
    try:
        result = tmp_cache_db.get_all_boundaries()
        assert result == []
    finally:
        tmp_cache_db._db.execute = original


def test_delete_boundary_fail_open_on_db_error(tmp_cache_db) -> None:
    """delete_boundary() swallows error when DB is broken."""
    original = _make_broken_db(tmp_cache_db)
    try:
        tmp_cache_db.delete_boundary("eid")
    finally:
        tmp_cache_db._db.execute = original


def test_get_all_vectors_fail_open_on_db_error(tmp_cache_db) -> None:
    """get_all_vectors() returns [] when DB is broken."""
    original = _make_broken_db(tmp_cache_db)
    try:
        result = tmp_cache_db.get_all_vectors(_tenant())
        assert result == []
    finally:
        tmp_cache_db._db.execute = original


def test_ccr_put_fail_open_on_db_error(tmp_cache_db) -> None:
    """ccr_put() swallows error when DB is broken."""
    original = _make_broken_db(tmp_cache_db)
    try:
        tmp_cache_db.ccr_put("ccr-1", b"data")
    finally:
        tmp_cache_db._db.execute = original


def test_ccr_get_fail_open_on_db_error(tmp_cache_db) -> None:
    """ccr_get() returns None when DB is broken."""
    original = _make_broken_db(tmp_cache_db)
    try:
        result = tmp_cache_db.ccr_get("ccr-1")
        assert result is None
    finally:
        tmp_cache_db._db.execute = original


def test_ccr_update_compressed_fail_open_on_db_error(tmp_cache_db) -> None:
    """ccr_update_compressed() swallows error when DB is broken."""
    original = _make_broken_db(tmp_cache_db)
    try:
        tmp_cache_db.ccr_update_compressed("ccr-1", b"data")
    finally:
        tmp_cache_db._db.execute = original


def test_metrics_load_fail_open_on_db_error(tmp_cache_db) -> None:
    """metrics_load() returns empty snapshot when DB is broken."""
    from superlocalmemory.optimize.storage.db import MetricsSnapshot
    original = _make_broken_db(tmp_cache_db)
    try:
        result = tmp_cache_db.metrics_load()
        assert isinstance(result, MetricsSnapshot)
        assert result.hits == 0
    finally:
        tmp_cache_db._db.execute = original


def test_metrics_flush_fail_open_on_db_error(tmp_cache_db) -> None:
    """metrics_flush() swallows error when DB is broken."""
    from superlocalmemory.optimize.storage.db import MetricsSnapshot
    original = _make_broken_db(tmp_cache_db)
    try:
        snap = MetricsSnapshot(hits=1)
        tmp_cache_db.metrics_flush(snap)
    finally:
        tmp_cache_db._db.execute = original


def test_entry_exists_fail_open_on_db_error(tmp_cache_db) -> None:
    """entry_exists() returns False when DB is broken."""
    original = _make_broken_db(tmp_cache_db)
    try:
        result = tmp_cache_db.entry_exists("k1", _tenant())
        assert result is False
    finally:
        tmp_cache_db._db.execute = original


def test_clear_tenant_fail_open_on_db_error(tmp_cache_db) -> None:
    """clear_tenant() returns 0 when DB is broken."""
    original = _make_broken_db(tmp_cache_db)
    try:
        result = tmp_cache_db.clear_tenant(_tenant())
        assert result == 0
    finally:
        tmp_cache_db._db.execute = original


def test_entry_count_fail_open_on_db_error(tmp_cache_db) -> None:
    """entry_count() returns 0 when DB is broken."""
    original = _make_broken_db(tmp_cache_db)
    try:
        result = tmp_cache_db.entry_count(_tenant())
        assert result == 0
    finally:
        tmp_cache_db._db.execute = original


def test_db_size_bytes_fail_open_on_db_error(tmp_cache_db) -> None:
    """db_size_bytes() returns 0 when DB is broken."""
    original = _make_broken_db(tmp_cache_db)
    try:
        result = tmp_cache_db.db_size_bytes()
        assert result == 0
    finally:
        tmp_cache_db._db.execute = original


def test_get_entry_by_id_fail_open_on_db_error(tmp_cache_db) -> None:
    """get_entry_by_id() returns None when DB is broken."""
    original = _make_broken_db(tmp_cache_db)
    try:
        result = tmp_cache_db.get_entry_by_id("eid")
        assert result is None
    finally:
        tmp_cache_db._db.execute = original


# ---- delete_all_boundaries (alias) ----
# Note: delete_boundary already tested; get_all_boundaries tested above


# ---- _decrypt error path on get ----

def test_get_decrypt_fail(tmp_cache_db) -> None:
    """get() returns None when value_blob is too short to decrypt (triggers ValueError)."""
    import sqlite3 as _sq
    # 5-byte blob is shorter than AES_NONCE_BYTES (12), triggers ValueError
    conn = _sq.connect(str(tmp_cache_db.db_path))
    conn.execute(
        "INSERT INTO llmcache_entries "
        "(entry_id, cache_key, tenant_id, model, value_blob, compressed) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        ("corrupt-eid", "corrupt-key", _tenant(), "m", b"short", 0),
    )
    conn.commit()
    conn.close()
    result = tmp_cache_db.get("corrupt-key", _tenant())
    assert result is None


def test_get_by_id_decrypt_fail(tmp_cache_db) -> None:
    """get_by_id() returns None when value_blob cannot be decrypted (ValueError raised)."""
    import sqlite3 as _sq
    conn = _sq.connect(str(tmp_cache_db.db_path))
    conn.execute(
        "INSERT INTO llmcache_entries "
        "(entry_id, cache_key, tenant_id, model, value_blob, compressed) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        ("corrupt-eid-2", "unused-key", _tenant(), "m", b"short", 0),
    )
    conn.commit()
    conn.close()
    result = tmp_cache_db.get_by_id("corrupt-eid-2")
    assert result is None


def test_get_entry_by_id_decrypt_fail(tmp_cache_db) -> None:
    """get_entry_by_id() returns None when value_blob is too short to decrypt."""
    import sqlite3 as _sq
    conn = _sq.connect(str(tmp_cache_db.db_path))
    conn.execute(
        "INSERT INTO llmcache_entries "
        "(entry_id, cache_key, tenant_id, model, value_blob, compressed) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        ("corrupt-eid-3", "unused-key-2", _tenant(), "m", b"short", 0),
    )
    conn.commit()
    conn.close()
    result = tmp_cache_db.get_entry_by_id("corrupt-eid-3")
    assert result is None


# ---- reset_default with close error ----

def test_reset_default_close_error_non_fatal(tmp_path: Path, monkeypatch) -> None:
    """reset_default swallows error from close()."""
    from superlocalmemory.optimize.storage.db import CacheDB
    monkeypatch.setenv("HOME", str(tmp_path))
    CacheDB.reset_default()
    inst = CacheDB.get_default()
    # Force close to raise
    original_close = inst._db.close
    inst._db.close = lambda: (_ for _ in ()).throw(RuntimeError("close failed"))
    try:
        CacheDB.reset_default()
        assert CacheDB._default_instance is None
    finally:
        CacheDB.reset_default()


# ---- get_all_boundaries success ----

def test_get_all_boundaries_returns_all(tmp_cache_db) -> None:
    """get_all_boundaries returns all boundary records."""
    from superlocalmemory.optimize.storage.db import BoundaryRow
    tmp_cache_db.boundary_upsert("eid-a", BoundaryRow(entry_id="eid-a", sample_count=1))
    tmp_cache_db.boundary_upsert("eid-b", BoundaryRow(entry_id="eid-b", sample_count=2))
    all_b = tmp_cache_db.get_all_boundaries()
    assert len(all_b) == 2
    eids = {b["entry_id"] for b in all_b}
    assert eids == {"eid-a", "eid-b"}


# ---- delete_boundary success ----

def test_delete_boundary_removes_record(tmp_cache_db) -> None:
    """delete_boundary removes the boundary record."""
    from superlocalmemory.optimize.storage.db import BoundaryRow
    tmp_cache_db.boundary_upsert("eid-del", BoundaryRow(entry_id="eid-del"))
    assert tmp_cache_db.boundary_get("eid-del") is not None
    tmp_cache_db.delete_boundary("eid-del")
    assert tmp_cache_db.boundary_get("eid-del") is None


# ---- vec_search zero-norm stored vector ----

def test_vec_search_zero_norm_stored_vector_skipped(tmp_cache_db) -> None:
    """vec_search skips stored vectors with zero norm."""
    import struct
    v_zero = struct.pack("4f", 0.0, 0.0, 0.0, 0.0)
    v_query = struct.pack("4f", 1.0, 0.0, 0.0, 0.0)

    tmp_cache_db.set("k1", _tenant(), b"v", model="m", ttl_expires=None, tags=[])
    row = tmp_cache_db.get("k1", _tenant())
    tmp_cache_db.vec_add(row.entry_id, _tenant(), v_zero, meta={"dim": 4})

    result = tmp_cache_db.vec_search(_tenant(), v_query, top_k=5)
    assert result == []
