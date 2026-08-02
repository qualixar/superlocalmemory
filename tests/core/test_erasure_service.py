# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

from __future__ import annotations

from superlocalmemory.core.transactions import (
    ErasureService,
    ErasureState,
    OperationContext,
    fetch_receipt,
    is_tombstoned,
    verify_receipt,
)
from superlocalmemory.core.transactions.concrete_owners import (
    Bm25Owner,
    TemporalOwner,
    VectorOwner,
)


def _erase_db(tmp_path, *, with_receipts: bool = True):
    import sqlite3

    from superlocalmemory.storage import schema
    from superlocalmemory.storage.database import DatabaseManager
    from superlocalmemory.storage.migrations import (
        M033_projection_transactions,
        M034_obligation_integrity,
        M035_erasure_receipts,
    )

    db_path = tmp_path / "memory.db"
    conn = sqlite3.connect(db_path)
    try:
        M033_projection_transactions.apply(conn)
        M034_obligation_integrity.apply(conn)
        if with_receipts:
            M035_erasure_receipts.apply(conn)
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


class _NoOpVectorStore:
    def __init__(self, indexed: set[str]) -> None:
        self.available = True
        self._indexed = set(indexed)

    def delete(self, fact_id: str) -> bool:
        return True

    def indexed_fact_ids(self, profile_id: str) -> set[str]:
        return set(self._indexed)

    def raw_vector_present(self, fact_id: str) -> bool:
        return fact_id in self._indexed


class _RealVectorStore(_NoOpVectorStore):
    def delete(self, fact_id: str) -> bool:
        self._indexed.discard(fact_id)
        return True


def _service(db, vector_store, recorder):
    owners = {
        "bm25": Bm25Owner(db),
        "temporal": TemporalOwner(db),
        "vector": VectorOwner(db, vector_store=vector_store),
    }
    return ErasureService(owners, audit_logger=recorder.append)


def _ctx(fact_id: str) -> OperationContext:
    return OperationContext(
        operation_id="erase-1",
        profile_id="default",
        subject_id="entity-x",
        fact_ids=(fact_id,),
    )


def test_erasure_complete_when_all_owners_prove_absence(tmp_path):
    db = _erase_db(tmp_path)
    db.store_bm25_tokens("f1", "default", ["alpha", "beta"])
    events: list = []

    receipt = _service(db, _RealVectorStore({"f1"}), events).erase(
        db, _ctx("f1"), subject_type="entity", subject_id="entity-x",
        requested_by="dpo",
    )

    assert receipt.state == ErasureState.COMPLETE
    assert receipt.all_erased is True
    assert receipt.persisted is True
    remaining = db.execute(
        "SELECT COUNT(*) AS c FROM bm25_tokens WHERE fact_id = ?", ("f1",)
    )
    assert dict(remaining[0])["c"] == 0
    assert len(events) == 1
    assert events[0]["state"] == ErasureState.COMPLETE


def test_erasure_failclosed_when_backend_leaves_residue(tmp_path):
    db = _erase_db(tmp_path)
    db.store_bm25_tokens("f1", "default", ["alpha"])
    events: list = []

    receipt = _service(db, _NoOpVectorStore({"f1"}), events).erase(
        db, _ctx("f1"), subject_type="entity", subject_id="entity-x",
    )

    assert receipt.state == ErasureState.FAILED
    assert receipt.all_erased is False
    vector_proof = {p.owner: p.erased for p in receipt.proofs}
    assert vector_proof["vector"] is False
    assert vector_proof["bm25"] is True
    with db.raw_connection() as conn:
        rows = conn.execute(
            "SELECT owner, state FROM projection_obligations "
            "WHERE operation_id = ? AND kind = 'erase'",
            ("erase-1",),
        ).fetchall()
    states = {r[0]: r[1] for r in rows}
    assert states["vector"] == "failed"
    assert states["bm25"] == "erased"


def test_receipt_is_persisted_and_reverifiable(tmp_path):
    db = _erase_db(tmp_path)
    db.store_bm25_tokens("f1", "default", ["alpha"])
    events: list = []

    _service(db, _RealVectorStore({"f1"}), events).erase(
        db, _ctx("f1"), subject_type="fact", subject_id="f1", requested_by="dpo",
    )

    with db.raw_connection() as conn:
        stored = fetch_receipt(conn, "erase-1")
        assert stored is not None
        assert stored.state == ErasureState.COMPLETE
        assert verify_receipt(conn, "erase-1") is True
        conn.execute(
            "UPDATE erasure_receipts SET state = 'FAILED' WHERE erasure_id = ?",
            ("erase-1",),
        )
        conn.commit()
        assert verify_receipt(conn, "erase-1") is False


def test_tombstone_written_for_target_facts(tmp_path):
    db = _erase_db(tmp_path)
    db.store_bm25_tokens("f1", "default", ["alpha"])
    events: list = []

    _service(db, _RealVectorStore({"f1"}), events).erase(
        db, _ctx("f1"), subject_type="entity", subject_id="entity-x",
    )

    with db.raw_connection() as conn:
        assert is_tombstoned(conn, "default", "f1") is True
        assert is_tombstoned(conn, "default", "other") is False


def test_tombstone_written_even_when_erasure_failed(tmp_path):
    db = _erase_db(tmp_path)
    db.store_bm25_tokens("f1", "default", ["alpha"])
    events: list = []

    receipt = _service(db, _NoOpVectorStore({"f1"}), events).erase(
        db, _ctx("f1"), subject_type="entity", subject_id="entity-x",
    )

    assert receipt.state == ErasureState.FAILED
    with db.raw_connection() as conn:
        assert is_tombstoned(conn, "default", "f1") is True


def test_apply_skips_tombstoned_fact(tmp_path):
    db = _erase_db(tmp_path)
    with db.raw_connection() as conn:
        conn.execute(
            "INSERT INTO projection_tombstones "
            "(profile_id, fact_id, erasure_id, created_at) VALUES "
            "('default', 'f1', 'e1', 0)"
        )
        conn.commit()
    owner = Bm25Owner(db)
    ctx = OperationContext(
        operation_id="op", profile_id="default", subject_id="f1", fact_ids=("f1",),
    )
    assert owner._required(ctx) == set()
    healed: list = []
    owner._heal = lambda c, fid: healed.append(fid) or True
    owner.apply(ctx)
    assert healed == []


def test_apply_heals_non_tombstoned_fact(tmp_path):
    db = _erase_db(tmp_path)
    owner = Bm25Owner(db)
    ctx = OperationContext(
        operation_id="op", profile_id="default", subject_id="f2", fact_ids=("f2",),
    )
    assert owner._required(ctx) == {"f2"}
    healed: list = []
    owner._heal = lambda c, fid: healed.append(fid) or True
    owner.apply(ctx)
    assert healed == ["f2"]


def test_required_failopen_without_tombstone_table(tmp_path):
    db = _erase_db(tmp_path, with_receipts=False)
    owner = Bm25Owner(db)
    ctx = OperationContext(
        operation_id="op", profile_id="default", subject_id="f1", fact_ids=("f1",),
    )
    assert owner._required(ctx) == {"f1"}


def test_tombstone_written_before_owner_work_survives_exception(tmp_path):
    db = _erase_db(tmp_path)
    db.store_bm25_tokens("f1", "default", ["alpha"])

    class _BoomVector(_RealVectorStore):
        def delete(self, fact_id):
            raise RuntimeError("backend down")

        def indexed_fact_ids(self, profile_id):
            raise RuntimeError("backend down")

    events: list = []
    receipt = _service(db, _BoomVector({"f1"}), events).erase(
        db, _ctx("f1"), subject_type="entity", subject_id="entity-x",
    )

    assert receipt.state == ErasureState.FAILED
    proofs = {p.owner: p.erased for p in receipt.proofs}
    assert proofs["vector"] is False
    with db.raw_connection() as conn:
        assert is_tombstoned(conn, "default", "f1") is True


def test_erasure_failopen_on_receipts_when_schema_absent(tmp_path):
    db = _erase_db(tmp_path, with_receipts=False)
    db.store_bm25_tokens("f1", "default", ["alpha"])
    events: list = []

    receipt = _service(db, _RealVectorStore({"f1"}), events).erase(
        db, _ctx("f1"), subject_type="entity", subject_id="entity-x",
    )

    assert receipt.state == ErasureState.COMPLETE
    assert receipt.persisted is False
    remaining = db.execute(
        "SELECT COUNT(*) AS c FROM bm25_tokens WHERE fact_id = ?", ("f1",)
    )
    assert dict(remaining[0])["c"] == 0
    assert len(events) == 1
