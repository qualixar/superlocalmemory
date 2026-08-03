# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

from __future__ import annotations

from superlocalmemory.compliance.gdpr import GDPRCompliance


class _FakeStore:
    def __init__(self, indexed):
        self.available = True
        self._idx = set(indexed)
        self.deleted = []

    def indexed_fact_ids(self, profile_id):
        return set(self._idx)

    def delete(self, fact_id):
        self.deleted.append(fact_id)
        self._idx.discard(fact_id)
        return True


class _FakeAnn:
    def __init__(self):
        self.removed = []

    def remove(self, fact_id):
        self.removed.append(fact_id)


class _FakeEngine:
    def __init__(self, store, ann):
        self._vector_store = store
        self._ann_index = ann


def test_purge_removes_profile_vectors_from_store_and_ann():
    store = _FakeStore({"f1", "f2"})
    ann = _FakeAnn()
    gdpr = GDPRCompliance(db=None, engine=_FakeEngine(store, ann))

    # _purge_vector_and_ann now returns (purged_count, failure_count)
    purged, failures = gdpr._purge_vector_and_ann("p1")

    assert purged == 2
    assert set(store.deleted) == {"f1", "f2"}
    assert set(ann.removed) == {"f1", "f2"}
    assert store.indexed_fact_ids("p1") == set()


def test_purge_is_noop_without_engine():
    gdpr = GDPRCompliance(db=None)
    purged, failures = gdpr._purge_vector_and_ann("p1")
    assert purged == 0
    assert failures == 0


def test_purge_is_noop_when_store_unavailable():
    store = _FakeStore({"f1"})
    store.available = False
    gdpr = GDPRCompliance(db=None, engine=_FakeEngine(store, _FakeAnn()))
    purged, failures = gdpr._purge_vector_and_ann("p1")
    assert purged == 0
    assert store.deleted == []


class _ResidueDB:
    """Minimal DB exposing raw vector residue counts for the erasure honesty
    checks. Returns [] for the atomic_facts enumeration and residue counts for
    the vector projection tables."""

    def execute(self, sql, params=()):
        s = sql.lower()
        if "from atomic_facts" in s:
            return []
        if "vector_row_map" in s:
            return [{"c": 3, "fact_id": "f1"}] if "count" in s else [{"fact_id": "f1"}]
        return [{"c": 0}]


def test_purge_surfaces_failures_when_store_unavailable_with_residue():
    store = _FakeStore({"f1"})
    store.available = False
    gdpr = GDPRCompliance(db=_ResidueDB(), engine=_FakeEngine(store, _FakeAnn()))
    purged, failures = gdpr._purge_vector_and_ann("p1")
    assert purged == 0
    # Un-purgeable raw vectors are surfaced so a receipt cannot claim COMPLETE.
    assert failures == 3


def test_fact_vector_residue_counts_remaining_map_rows():
    gdpr = GDPRCompliance(db=_ResidueDB())
    assert gdpr._fact_vector_residue("p1", ["f1"]) == 1
    assert gdpr._fact_vector_residue("p1", []) == 0


def test_entity_receipt_marks_failed_when_residue_remains(tmp_path):
    """ErasureService.finalize with BM25 residue remaining must produce a FAILED
    receipt with real per-owner proofs (not the dead helper's proofs:[]).

    Retargeted from _write_entity_erasure_receipt (removed — wrote proofs:[] +
    unkeyed SHA, no production caller) to ErasureService.finalize.
    """
    import sqlite3
    import uuid

    from superlocalmemory.core.transactions.concrete_owners import (
        build_erasure_service_for_db,
    )
    from superlocalmemory.core.transactions.erasure import fetch_receipt, verify_receipt
    from superlocalmemory.core.transactions.owners import OperationContext
    from superlocalmemory.storage import schema
    from superlocalmemory.storage.database import DatabaseManager
    from superlocalmemory.storage.migrations import (
        M033_projection_transactions,
        M034_obligation_integrity,
        M035_erasure_receipts,
    )

    db_path = tmp_path / "receipts.db"
    raw = sqlite3.connect(db_path)
    M033_projection_transactions.apply(raw)
    M034_obligation_integrity.apply(raw)
    M035_erasure_receipts.apply(raw)
    raw.commit()
    raw.close()

    db = DatabaseManager(db_path)
    db.initialize(schema)
    db.execute(
        "INSERT OR IGNORE INTO profiles (profile_id, name) VALUES ('p1', 'p1')"
    )
    db.execute(
        "INSERT INTO memories (memory_id, profile_id, content) VALUES ('m1', 'p1', 'x')"
    )
    db.execute(
        "INSERT INTO atomic_facts (fact_id, memory_id, profile_id, content) "
        "VALUES ('f1', 'm1', 'p1', 'entity fact')"
    )
    # Write BM25 tokens so BM25 owner detects residue when finalize() is called
    # without remove() first — simulates a wipe where BM25 data was left behind.
    db.store_bm25_tokens("f1", "p1", ["residue"])

    erasure_id = uuid.uuid4().hex
    ctx = OperationContext(
        operation_id=erasure_id,
        profile_id="p1",
        subject_id="Acme",
        fact_ids=("f1",),
    )
    svc = build_erasure_service_for_db(db, engine=None)
    # finalize() without remove() → BM25 owner finds residue → FAILED receipt
    receipt = svc.finalize(
        db, ctx,
        subject_type="entity",
        subject_id="Acme",
        requested_by="gdpr",
        requested_at=0.0,
    )

    assert not receipt.all_erased, "receipt must be FAILED when residue remains"

    with db.raw_connection() as conn:
        rows = conn.execute(
            "SELECT erasure_id, state, all_erased, owner_evidence_json "
            "FROM erasure_receipts"
        ).fetchall()

    assert len(rows) == 1, "exactly one receipt must be persisted"
    assert rows[0][1] == "FAILED", f"state must be FAILED, got {rows[0][1]}"
    assert rows[0][2] == 0, "all_erased must be 0"

    # Real per-owner proofs — not proofs:[] like the dead helper wrote.
    proofs = __import__("json").loads(rows[0][3]).get("proofs", [])
    assert len(proofs) > 0, (
        f"receipt must have real per-owner proofs; got: {rows[0][3]}"
    )

    # Tamper-evidence: verify_receipt must pass for the unmodified row.
    with db.raw_connection() as conn:
        assert fetch_receipt(conn, rows[0][0]) is not None
        assert verify_receipt(conn, rows[0][0]) is True
