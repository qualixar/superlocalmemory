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
