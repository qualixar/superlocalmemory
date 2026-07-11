"""LLD-00 §10.4 + LLD-10 — CacheDB tests."""

from __future__ import annotations

import os
import struct
import time
import uuid
from pathlib import Path

import pytest


# ---- isolation ----

def test_assert_no_memory_db_tables_passes(tmp_cache_db) -> None:
    # fixture construction IS the test
    pass


def test_isolation_wrong_db_raises(tmp_path: Path) -> None:
    """Opening CacheDB on memory.db must raise RuntimeError."""
    import sqlite3
    from superlocalmemory.optimize.storage.db import CacheDB
    # Create a memory.db by importing the memory schema; simpler: create a
    # forbidden table directly.
    mem_db = tmp_path / "memory.db"
    conn = sqlite3.connect(str(mem_db))
    conn.execute("CREATE TABLE atomic_facts (id TEXT PRIMARY KEY, content TEXT)")
    conn.execute("CREATE TABLE memories (id TEXT PRIMARY KEY, content TEXT)")
    conn.commit()
    conn.close()
    with pytest.raises(RuntimeError, match="ISOLATION VIOLATION"):
        CacheDB(mem_db)


# ---- file permissions (SEC-C-01) ----

def test_chmod_600_set(tmp_cache_db, tmp_path: Path) -> None:
    mode = stat_mode(tmp_path / "llmcache.db")
    assert mode is not None
    assert mode & 0o777 == 0o600, f"db perms not 600: {oct(mode & 0o777)}"


def stat_mode(path: Path):
    try:
        return path.stat().st_mode
    except OSError:
        return None


# ---- CRUD ----

def test_set_and_get(tmp_cache_db) -> None:
    value = b"hello world response"
    tmp_cache_db.set("sha256abc", "t1", value,
                     model="claude-sonnet-4-6", ttl_expires=None, tags=[])
    row = tmp_cache_db.get("sha256abc", "t1")
    assert row is not None
    assert row.value == value
    assert row.model == "claude-sonnet-4-6"


def test_get_increments_hit_count(tmp_cache_db) -> None:
    tmp_cache_db.set("k1", "t1", b"v", model="m", ttl_expires=None, tags=[])
    tmp_cache_db.get("k1", "t1")
    tmp_cache_db.get("k1", "t1")
    row = tmp_cache_db.get("k1", "t1")
    assert row is not None
    assert row.hit_count == 3


def test_get_tenant_isolation(tmp_cache_db) -> None:
    tmp_cache_db.set("k1", "tenantA", b"secret", model="m", ttl_expires=None, tags=[])
    row = tmp_cache_db.get("k1", "tenantB")
    assert row is None


def test_delete(tmp_cache_db) -> None:
    tmp_cache_db.set("k1", "t1", b"v", model="m", ttl_expires=None, tags=[])
    row = tmp_cache_db.get("k1", "t1")
    assert row is not None
    entry_id = row.entry_id
    vector_bytes = struct.pack("3f", 0.1, 0.2, 0.3)
    tmp_cache_db.vec_add(entry_id, "t1", vector_bytes, meta={"dim": 3, "model": "x"})
    tmp_cache_db.delete("k1", "t1")
    assert tmp_cache_db.get("k1", "t1") is None
    assert tmp_cache_db.get_all_vectors("t1") == []


def test_entry_exists_returns_false_on_miss(tmp_cache_db) -> None:
    assert tmp_cache_db.entry_exists("nonexistent", "t1") is False


# ---- TTL ----

def test_get_returns_none_after_ttl(tmp_cache_db) -> None:
    tmp_cache_db.set("ttlkey", "t1", b"data", model="m",
                     ttl_expires=time.time() + 1, tags=[])
    time.sleep(1.2)
    assert tmp_cache_db.get("ttlkey", "t1") is None


def test_sweep_expired_removes_expired_entries(tmp_cache_db) -> None:
    past = time.time() - 1
    future = time.time() + 9999
    tmp_cache_db.set("old", "t1", b"v", model="m", ttl_expires=past, tags=[])
    tmp_cache_db.set("new", "t1", b"v", model="m", ttl_expires=future, tags=[])
    deleted = tmp_cache_db.sweep_expired(now=time.time())
    assert deleted >= 1
    assert tmp_cache_db.get("old", "t1") is None
    assert tmp_cache_db.get("new", "t1") is not None


def test_sweep_expired_also_removes_ccr_originals(tmp_cache_db) -> None:
    ccr_id = uuid.uuid4().hex
    tmp_cache_db.ccr_put(ccr_id, b"context-bytes-here-12345")
    # Manually expire it
    tmp_cache_db._db.execute(
        "UPDATE llmcache_ccr_originals SET ttl_expires = ? WHERE ccr_id = ?",
        (time.time() - 1, ccr_id),
    )
    deleted = tmp_cache_db.sweep_expired(now=time.time())
    assert deleted >= 1
    assert tmp_cache_db.ccr_get(ccr_id) is None


# ---- Tag invalidation ----

def test_invalidate_by_tag_removes_matching_entries(tmp_cache_db) -> None:
    tmp_cache_db.set("k1", "t1", b"v", model="m", ttl_expires=None,
                     tags=["session:abc", "model:claude"])
    tmp_cache_db.set("k2", "t1", b"v", model="m", ttl_expires=None,
                     tags=["session:xyz"])
    tmp_cache_db.tag_register("k1", "t1", ["session:abc", "model:claude"])
    tmp_cache_db.tag_register("k2", "t1", ["session:xyz"])
    count = tmp_cache_db.invalidate_by_tag("session:abc")
    assert count == 1
    assert tmp_cache_db.get("k1", "t1") is None
    assert tmp_cache_db.get("k2", "t1") is not None


def test_invalidate_by_tag_does_not_cross_tenants(tmp_cache_db) -> None:
    tmp_cache_db.set("k1", "tenantA", b"v", model="m", ttl_expires=None,
                     tags=["session:abc"])
    tmp_cache_db.set("k2", "tenantB", b"v", model="m", ttl_expires=None,
                     tags=["session:abc"])
    tmp_cache_db.tag_register("k1", "tenantA", ["session:abc"])
    count = tmp_cache_db.invalidate_by_tag("session:abc")
    assert count == 1
    assert tmp_cache_db.get("k2", "tenantB") is not None


# ---- Metrics ----

def test_metrics_flush_load_roundtrip(tmp_cache_db) -> None:
    from superlocalmemory.optimize.storage.db import MetricsSnapshot
    snap = MetricsSnapshot(
        hits=10, misses=3, calls_skipped=2,
        tokens_saved_input=500, tokens_saved_output=200, tokens_saved_compress=100,
        evictions=1, latency_overhead_ms_sum=150.5, latency_samples=20,
        compress_runs=5, compress_bytes_original=10000, compress_bytes_after=6000,
        cache_size_bytes=204800, cache_entry_count=42,
        updated_at=1234567890.0,
    )
    tmp_cache_db.metrics_flush(snap)
    loaded = tmp_cache_db.metrics_load()
    assert loaded.hits == 10
    assert loaded.misses == 3
    assert loaded.compress_runs == 5
    assert loaded.compress_bytes_original == 10000
    assert loaded.compress_bytes_after == 6000
    assert loaded.cache_size_bytes == 204800
    assert loaded.cache_entry_count == 42
    assert abs(loaded.compression_ratio - 0.6) < 0.001


# ---- Encryption ----

def test_encryption_roundtrip(tmp_cache_db) -> None:
    """Stored blob must NOT contain plaintext bytes."""
    secret = b"SECRET_PASSWORD_PLAINTEXT_VALUE"
    tmp_cache_db.set("k1", "t1", secret, model="m", ttl_expires=None, tags=[])
    # Read raw DB bytes
    import sqlite3
    conn = sqlite3.connect(str(tmp_cache_db.db_path))
    rows = conn.execute(
        "SELECT value_blob FROM llmcache_entries WHERE cache_key = ? AND tenant_id = ?",
        ("k1", "t1"),
    ).fetchall()
    conn.close()
    assert rows
    raw = rows[0][0]
    assert secret not in raw, "plaintext leaked into raw BLOB"


def test_key_stability_across_reruns(tmp_path: Path) -> None:
    """Two CacheDB instances opened on the same path derive the same AES key."""
    from superlocalmemory.optimize.storage.db import CacheDB
    db1 = CacheDB(tmp_path / "llmcache.db")
    db1.set("k", "t1", b"v", model="m", ttl_expires=None, tags=[])
    db1.close()
    db2 = CacheDB(tmp_path / "llmcache.db")
    row = db2.get("k", "t1")
    assert row is not None
    assert row.value == b"v"
    db2.close()


def test_machine_id_falls_back_on_windows_without_os_uname(
    tmp_path: Path, monkeypatch,
) -> None:
    """Windows has no os.uname(); CacheDB must use the persisted uuid fallback."""
    import os as _os
    from superlocalmemory.optimize.storage.db import CacheDB

    monkeypatch.setattr(
        "superlocalmemory.optimize.storage.db.platform.system",
        lambda: "Windows",
    )
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.delattr(_os, "uname", raising=False)

    db = CacheDB.__new__(CacheDB)
    mid = db._get_machine_id()

    mid_file = tmp_path / ".superlocalmemory" / ".llmcache_key"
    assert mid
    assert mid_file.exists()
    assert mid_file.read_text(encoding="utf-8").strip() == mid


def test_ccr_put_get_roundtrip(tmp_cache_db) -> None:
    ccr_id = uuid.uuid4().hex
    original = b"verbatim user context that must be encrypted"
    tmp_cache_db.ccr_put(ccr_id, original)
    assert tmp_cache_db.ccr_get(ccr_id) == original


# ---- singleton ----

def test_get_default_returns_singleton(tmp_path: Path, monkeypatch) -> None:
    """get_default() returns a single instance across calls."""
    from superlocalmemory.optimize.storage.db import CacheDB
    # Patch home dir to a tmp dir to avoid polluting the user's llmcache.db
    monkeypatch.setenv("HOME", str(tmp_path))
    CacheDB.reset_default()
    a = CacheDB.get_default()
    b = CacheDB.get_default()
    assert a is b
    CacheDB.reset_default()


# ---- fail-open ----

def test_get_fail_open_on_corrupt_db(tmp_path: Path) -> None:
    """get() must return None (not raise) for an unopenable DB."""
    from superlocalmemory.optimize.storage.db import CacheDB
    db_path = tmp_path / "corrupt.db"
    db_path.write_bytes(b"not a sqlite file at all")
    # CacheDB init may raise or open; either way, get() must not raise.
    try:
        db = CacheDB(db_path)
        row = db.get("any", "t1")
        assert row is None
    except RuntimeError:
        # Acceptable: init refuses a corrupt file; proxy won't instantiate.
        pass


# ---- C-06: AES key persistence ----

def test_c06_aes_key_persisted_to_file(tmp_path: Path) -> None:
    """C-06: _get_or_persist_aes_key() must write 32 bytes to opt-key.bin on first open."""
    from superlocalmemory.optimize.storage.db import CacheDB, _KEY_FILE
    import importlib

    key_file = tmp_path / "opt-key.bin"

    # Patch _KEY_FILE to use tmp_path so we don't pollute the real ~/.superlocalmemory/
    import superlocalmemory.optimize.storage.db as db_mod
    original = db_mod._KEY_FILE
    db_mod._KEY_FILE = key_file
    try:
        db = CacheDB(tmp_path / "llmcache.db")
        assert key_file.exists(), "opt-key.bin must be created on first DB open"
        key_bytes = key_file.read_bytes()
        assert len(key_bytes) == 32, f"Persisted key must be 32 bytes, got {len(key_bytes)}"
        # Permissions must be 0600
        mode = oct(key_file.stat().st_mode & 0o777)
        assert mode == "0o600", f"opt-key.bin must be 0600, got {mode}"
    finally:
        db_mod._KEY_FILE = original


def test_c10_vec_add_persists_context_fp(tmp_cache_db) -> None:
    """C-10: vec_add must persist context_fp; get_all_vectors must return it."""
    import struct
    tmp_cache_db.set("k-c10", "t1", b"v", model="m", ttl_expires=None, tags=[])
    row = tmp_cache_db.get("k-c10", "t1")
    vec = struct.pack("3f", 0.5, 0.6, 0.7)
    ctx = "sha256:abc123"

    tmp_cache_db.vec_add(row.entry_id, "t1", vec, meta={"dim": 3, "model": "x", "context_fp": ctx})

    all_vecs = tmp_cache_db.get_all_vectors("t1")
    assert len(all_vecs) == 1
    entry_id_ret, blob_ret, ctx_fp_ret = all_vecs[0]
    assert entry_id_ret == row.entry_id
    assert blob_ret == vec
    assert ctx_fp_ret == ctx, f"C-10: context_fp must be persisted; got {ctx_fp_ret!r}"


def test_c10_vec_add_default_context_fp_is_empty(tmp_cache_db) -> None:
    """C-10: vec_add without context_fp in meta defaults to empty string."""
    import struct
    tmp_cache_db.set("k-c10b", "t1", b"v", model="m", ttl_expires=None, tags=[])
    row = tmp_cache_db.get("k-c10b", "t1")
    vec = struct.pack("3f", 0.1, 0.2, 0.3)

    tmp_cache_db.vec_add(row.entry_id, "t1", vec, meta={"dim": 3, "model": "x"})

    all_vecs = tmp_cache_db.get_all_vectors("t1")
    assert len(all_vecs) == 1
    _, _, ctx_fp_ret = all_vecs[0]
    assert ctx_fp_ret == "", f"C-10: missing context_fp must default to ''; got {ctx_fp_ret!r}"


def test_c06_second_open_uses_persisted_key(tmp_path: Path) -> None:
    """C-06: Second CacheDB open must use the persisted key (decrypt still works)."""
    from superlocalmemory.optimize.storage.db import CacheDB
    import superlocalmemory.optimize.storage.db as db_mod

    key_file = tmp_path / "opt-key.bin"
    db_path = tmp_path / "llmcache.db"
    original = db_mod._KEY_FILE
    db_mod._KEY_FILE = key_file
    try:
        # First open — derives and persists key
        db1 = CacheDB(db_path)
        db1.set("k1", "a" * 64, b"hello world", model="m", ttl_expires=None, tags=[])
        db1.close()

        # Second open — must load persisted key and decrypt successfully
        db2 = CacheDB(db_path)
        row = db2.get("k1", "a" * 64)
        assert row is not None, "C-06: second open must read entries written by first open"
    finally:
        db_mod._KEY_FILE = original
