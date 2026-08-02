# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

from __future__ import annotations

from superlocalmemory.core.transactions import OperationContext
from superlocalmemory.core.transactions.concrete_owners import (
    Bm25Owner,
    VectorOwner,
)


def _fresh_db(tmp_path):
    from superlocalmemory.storage import schema
    from superlocalmemory.storage.database import DatabaseManager

    db = DatabaseManager(tmp_path / "memory.db")
    db.initialize(schema)
    db.execute(
        "INSERT OR IGNORE INTO profiles (profile_id, name) VALUES (?, ?)",
        ("default", "default"),
    )
    return db


class _NoOpVectorStore:
    def __init__(self, indexed: set[str]) -> None:
        self.available = True
        self._indexed = set(indexed)
        self.delete_calls: list[str] = []

    def delete(self, fact_id: str) -> bool:
        self.delete_calls.append(fact_id)
        return True

    def indexed_fact_ids(self, profile_id: str) -> set[str]:
        return set(self._indexed)

    def raw_vector_present(self, fact_id: str) -> bool:
        return fact_id in self._indexed


def _ctx(fact_id: str) -> OperationContext:
    return OperationContext(
        operation_id="erase-op",
        profile_id="default",
        subject_id=fact_id,
        fact_ids=(fact_id,),
    )


def test_bm25_erase_proof_detects_residue_after_failed_delete(tmp_path):
    db = _fresh_db(tmp_path)
    db.store_bm25_tokens("f1", "default", ["alpha", "beta"])
    db.delete_bm25_tokens_for_fact = lambda *_a, **_k: None

    proof = Bm25Owner(db).erase(_ctx("f1"))

    remaining = db.execute(
        "SELECT COUNT(*) AS c FROM bm25_tokens WHERE fact_id = ?", ("f1",)
    )
    assert dict(remaining[0])["c"] == 1
    assert proof.erased is False


def test_bm25_erase_proof_true_when_store_clean(tmp_path):
    db = _fresh_db(tmp_path)
    db.store_bm25_tokens("f1", "default", ["alpha", "beta"])

    proof = Bm25Owner(db).erase(_ctx("f1"))

    remaining = db.execute(
        "SELECT COUNT(*) AS c FROM bm25_tokens WHERE fact_id = ?", ("f1",)
    )
    assert dict(remaining[0])["c"] == 0
    assert proof.erased is True


def test_vector_erase_proof_detects_residue_when_backend_noop(tmp_path):
    db = _fresh_db(tmp_path)
    store = _NoOpVectorStore(indexed={"f1"})

    proof = VectorOwner(db, vector_store=store).erase(_ctx("f1"))

    assert store.delete_calls == ["f1"]
    assert "f1" in store.indexed_fact_ids("default")
    assert proof.erased is False


def test_vector_erase_proof_true_when_backend_removes(tmp_path):
    db = _fresh_db(tmp_path)

    class _RealDeleteStore(_NoOpVectorStore):
        def delete(self, fact_id: str) -> bool:
            self.delete_calls.append(fact_id)
            self._indexed.discard(fact_id)
            return True

    store = _RealDeleteStore(indexed={"f1"})

    proof = VectorOwner(db, vector_store=store).erase(_ctx("f1"))

    assert "f1" not in store.indexed_fact_ids("default")
    assert proof.erased is True
