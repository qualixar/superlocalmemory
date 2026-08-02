# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

from __future__ import annotations

import time

from superlocalmemory.core.transactions import OperationContext
from superlocalmemory.core.transactions.concrete_owners import VectorOwner
from superlocalmemory.core.transactions.erasure import (
    tombstone_memory_id,
    write_tombstones,
)


def _erase_db(tmp_path):
    import sqlite3

    from superlocalmemory.storage import schema
    from superlocalmemory.storage.database import DatabaseManager
    from superlocalmemory.storage.migrations import (
        M033_projection_transactions,
        M034_obligation_integrity,
        M035_erasure_receipts,
        M036_vector_row_map,
    )

    db_path = tmp_path / "memory.db"
    conn = sqlite3.connect(db_path)
    try:
        M033_projection_transactions.apply(conn)
        M034_obligation_integrity.apply(conn)
        M035_erasure_receipts.apply(conn)
        M036_vector_row_map.apply(conn)
        conn.commit()
    finally:
        conn.close()

    db = DatabaseManager(db_path)
    db.initialize(schema)
    db.execute(
        "INSERT OR IGNORE INTO profiles (profile_id, name) VALUES (?, ?)",
        ("default", "default"),
    )
    return db


def _ctx(fact_id: str) -> OperationContext:
    return OperationContext(
        operation_id="erase-op",
        profile_id="default",
        subject_id=fact_id,
        fact_ids=(fact_id,),
    )


# ── F2: tombstone provenance fail-closed ────────────────────────────────────


def test_write_tombstones_failclosed_on_divergent_memory_id(tmp_path):
    db = _erase_db(tmp_path)
    assert write_tombstones(db, "default", ("f1",), "e1", time.time(), "mA") is True
    assert tombstone_memory_id(db, "default", "f1") == "mA"

    # A conflicting non-null memory_id must fail closed, not warn-and-retain.
    assert write_tombstones(db, "default", ("f1",), "e2", time.time(), "mB") is False
    # Original provenance is retained (never overwritten).
    assert tombstone_memory_id(db, "default", "f1") == "mA"

    # The same memory_id is an idempotent success.
    assert write_tombstones(db, "default", ("f1",), "e3", time.time(), "mA") is True


def test_write_tombstones_backfills_null_memory_id(tmp_path):
    db = _erase_db(tmp_path)
    assert write_tombstones(db, "default", ("f1",), "e1", time.time(), None) is True
    assert tombstone_memory_id(db, "default", "f1") is None
    # A later non-null provenance repairs the NULL (not a conflict).
    assert write_tombstones(db, "default", ("f1",), "e2", time.time(), "mA") is True
    assert tombstone_memory_id(db, "default", "f1") == "mA"


# ── F1: vector residue detection when backend is down / read errors ─────────


class _DownStoreWithMap:
    available = False

    def __init__(self, residue: set[str]) -> None:
        self._residue = set(residue)

    def raw_vector_present(self, fact_id: str) -> bool:
        return fact_id in self._residue


class _ReadErrorStore:
    available = False

    def raw_vector_present(self, fact_id: str) -> bool:
        raise RuntimeError("vector backend read error")


def test_vector_residue_detected_when_store_unavailable(tmp_path):
    db = _erase_db(tmp_path)
    owner = VectorOwner(db, vector_store=_DownStoreWithMap({"f1"}))
    proof = owner.prove_erased(_ctx("f1"))
    assert proof.erased is False
    assert "f1" in proof.detail.get("residue", [])


def test_vector_no_false_block_when_never_vectorized(tmp_path):
    db = _erase_db(tmp_path)
    owner = VectorOwner(db, vector_store=_DownStoreWithMap(set()))
    proof = owner.prove_erased(_ctx("f1"))
    assert proof.erased is True


def test_vector_read_error_fails_closed(tmp_path):
    db = _erase_db(tmp_path)
    owner = VectorOwner(db, vector_store=_ReadErrorStore())
    proof = owner.prove_erased(_ctx("f1"))
    assert proof.erased is False


class _ContainsErrAnn:
    def contains(self, fact_id: str) -> bool:
        raise RuntimeError("ann read error")


def test_vector_ann_read_error_fails_closed(tmp_path):
    db = _erase_db(tmp_path)
    owner = VectorOwner(
        db, vector_store=_DownStoreWithMap(set()), ann_index=_ContainsErrAnn(),
    )
    proof = owner.prove_erased(_ctx("f1"))
    assert proof.erased is False


# ── Phase 3b: erasure fence + materializer reconciliation hardening ─────────


def test_erasure_fence_mark_is_clear_ttl():
    from superlocalmemory.storage import erasure_fence

    erasure_fence.clear_erasing("p1", "f1")
    assert erasure_fence.is_erasing("p1", "f1") is False
    erasure_fence.mark_erasing("p1", "f1")
    assert erasure_fence.is_erasing("p1", "f1") is True
    # Profile-scoped: a different profile is unaffected.
    assert erasure_fence.is_erasing("p2", "f1") is False
    erasure_fence.clear_erasing("p1", "f1")
    assert erasure_fence.is_erasing("p1", "f1") is False


def test_vector_residue_present_failclosed_on_read_error():
    import sqlite3

    from superlocalmemory.core.store_pipeline import _vector_residue_present

    class _ResidueRaisingDB:
        def execute(self, sql, params=()):
            if "sqlite_master" in sql.lower():
                return [{"1": 1}]  # table exists
            raise sqlite3.OperationalError("residue read boom")

    # Table exists but is unreadable -> uncertain -> fail closed (residue present).
    assert _vector_residue_present(_ResidueRaisingDB(), "f1") is True


def test_vector_residue_absent_when_tables_missing():
    from superlocalmemory.core.store_pipeline import _vector_residue_present

    class _NoTableDB:
        def execute(self, sql, params=()):
            return []  # sqlite_master -> no such table

    # No vector tables at all -> no false residue alarm.
    assert _vector_residue_present(_NoTableDB(), "f1") is False


def test_drop_resurrected_uses_fence_when_tombstone_unreadable():
    import sqlite3

    from superlocalmemory.core.store_pipeline import _drop_resurrected_facts
    from superlocalmemory.storage import erasure_fence

    class _RaisingTombstoneDB:
        def __init__(self):
            self.deleted: list[str] = []

        def execute(self, sql, params=()):
            s = sql.lower()
            if "projection_tombstones" in s:
                raise sqlite3.OperationalError("tombstone read boom")
            return []  # sqlite_master residue tables absent

        def delete_fact(self, fid, profile_id=None):
            self.deleted.append(fid)

        def delete_bm25_tokens_for_fact(self, fid):
            pass

    db = _RaisingTombstoneDB()
    erasure_fence.mark_erasing("p1", "ferase")
    try:
        survivors = _drop_resurrected_facts(db, "p1", ["ferase"], None, None, None)
    finally:
        erasure_fence.clear_erasing("p1", "ferase")

    # Fence confirmed erasure intent despite the unreadable tombstone -> cleaned.
    assert "ferase" in db.deleted
    assert "ferase" not in survivors


def test_drop_resurrected_defers_when_no_fence_and_unreadable():
    import sqlite3

    from superlocalmemory.core.store_pipeline import _drop_resurrected_facts

    class _RaisingTombstoneDB:
        def __init__(self):
            self.deleted: list[str] = []

        def execute(self, sql, params=()):
            if "projection_tombstones" in sql.lower():
                raise sqlite3.OperationalError("tombstone read boom")
            return []

        def delete_fact(self, fid, profile_id=None):
            self.deleted.append(fid)

        def delete_bm25_tokens_for_fact(self, fid):
            pass

    db = _RaisingTombstoneDB()
    survivors = _drop_resurrected_facts(db, "p1", ["fkeep"], None, None, None)
    # No fence, unreadable tombstone: neither deleted nor promoted to survivor.
    assert db.deleted == []
    assert "fkeep" not in survivors


# ── Phase 3b: vec0 orphan GC ────────────────────────────────────────────────


def test_orphan_rowids_computes_unmapped_set():
    import sqlite3

    from superlocalmemory.retrieval.vector_store import VectorStore

    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE fact_embeddings (rowid INTEGER PRIMARY KEY, profile_id TEXT)")
    conn.execute("CREATE TABLE embedding_metadata (vec_rowid INTEGER, profile_id TEXT)")
    conn.execute("CREATE TABLE vector_row_map (vec_rowid INTEGER, profile_id TEXT)")
    conn.executemany(
        "INSERT INTO fact_embeddings (rowid, profile_id) VALUES (?, ?)",
        [(1, "p1"), (2, "p1"), (3, "p1"), (4, "p2")],
    )
    conn.execute("INSERT INTO embedding_metadata (vec_rowid, profile_id) VALUES (1, 'p1')")
    conn.execute("INSERT INTO vector_row_map (vec_rowid, profile_id) VALUES (2, 'p1')")
    conn.commit()
    # rowid 1 mapped via metadata, 2 via map, 3 orphan(p1), 4 orphan(p2).
    assert VectorStore._orphan_rowids(conn, "p1") == {3}
    assert VectorStore._orphan_rowids(conn, "p2") == {4}
    assert VectorStore._orphan_rowids(conn, None) == {3, 4}
    conn.close()


def test_vec0_delete_fallback_and_gc(tmp_path):
    import pytest

    from superlocalmemory.retrieval.vector_store import VectorStore, VectorStoreConfig

    vs = VectorStore(tmp_path / "vec.db", VectorStoreConfig(dimension=4))
    if not vs.available:
        pytest.skip("sqlite-vec extension unavailable in this environment")

    assert vs.upsert("f1", "p1", [0.1, 0.2, 0.3, 0.4]) is True
    assert vs.upsert("f2", "p1", [0.5, 0.6, 0.7, 0.8]) is True

    # Metadata-less orphan: drop f2's metadata, keep map + raw vec0 row.
    with vs._managed_connection() as c:
        c.execute("DELETE FROM embedding_metadata WHERE fact_id='f2'")
        c.commit()
    assert vs.raw_vector_present("f2") is True
    assert vs.delete("f2") is True  # map fallback resolves the rowid
    assert vs.raw_vector_present("f2") is False

    # Truly unmapped orphan: drop both metadata and map for f3.
    assert vs.upsert("f3", "p1", [0.9, 0.1, 0.2, 0.3]) is True
    with vs._managed_connection() as c:
        c.execute("DELETE FROM embedding_metadata WHERE fact_id='f3'")
        c.execute("DELETE FROM vector_row_map WHERE fact_id='f3'")
        c.commit()
    assert vs.gc_orphaned_vectors("p1") == 1
    assert vs.gc_orphaned_vectors("p1") == 0

    # A fully-mapped fact survives GC and stays searchable.
    hits = vs.search([0.1, 0.2, 0.3, 0.4], top_k=1, profile_id="p1")
    assert any(fid == "f1" for fid, _ in hits)
