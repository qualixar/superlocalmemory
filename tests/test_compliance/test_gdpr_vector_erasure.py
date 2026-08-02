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
    import sqlite3

    from superlocalmemory.core.transactions.erasure import fetch_receipt
    from superlocalmemory.storage.migrations import M035_erasure_receipts

    db_path = tmp_path / "receipts.db"
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    M035_erasure_receipts.apply(conn)
    conn.commit()

    class _ReceiptDB:
        def raw_connection(self):
            from contextlib import contextmanager

            @contextmanager
            def _cm():
                yield conn
            return _cm()

    gdpr = GDPRCompliance(db=_ReceiptDB())
    ok = gdpr._write_entity_erasure_receipt(
        "p1", "Acme", ["f1"], 0.0, all_erased=False,
    )
    assert ok is True
    rows = conn.execute(
        "SELECT erasure_id, state, all_erased FROM erasure_receipts"
    ).fetchall()
    assert len(rows) == 1
    assert rows[0]["state"] == "FAILED"
    assert rows[0]["all_erased"] == 0
    # The persisted receipt is still tamper-evident.
    assert fetch_receipt(conn, rows[0]["erasure_id"]) is not None
    conn.close()
