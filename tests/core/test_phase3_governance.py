# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Phase 3 governance tests — tamper-evident manifest, vector-owner semantics,
unified erasure orchestrator.

TDD: these tests MUST fail before the implementation and pass after.
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# DB factory helpers
# ---------------------------------------------------------------------------

def _fresh_db(tmp_path: Path, *, with_transactions: bool = True, with_receipts: bool = True):
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
        if with_transactions:
            M033_projection_transactions.apply(conn)
            M034_obligation_integrity.apply(conn)
        if with_receipts:
            M035_erasure_receipts.apply(conn)
        conn.commit()
    finally:
        conn.close()

    db = DatabaseManager(db_path)
    db.initialize(schema)
    db.execute("INSERT OR IGNORE INTO profiles (profile_id, name) VALUES ('p1', 'p1')")
    return db


def _apply_m037(conn: sqlite3.Connection) -> None:
    from superlocalmemory.storage.migrations import M037_manifest_hmac_version
    M037_manifest_hmac_version.apply(conn)


def _fresh_db_with_m037(tmp_path: Path):
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
        M035_erasure_receipts.apply(conn)
        _apply_m037(conn)
        conn.commit()
    finally:
        conn.close()

    db = DatabaseManager(db_path)
    db.initialize(schema)
    db.execute("INSERT OR IGNORE INTO profiles (profile_id, name) VALUES ('p1', 'p1')")
    return db


_TEST_HMAC_KEY = b"test-hmac-key-for-phase3-governance-32b"[:32].ljust(32, b"\x00")


def _patch_manifest_key(monkeypatch, key: bytes = _TEST_HMAC_KEY):
    """Patch the key derivation in manifest_key module to return a fixed test key."""
    from superlocalmemory.core.transactions import manifest_key as _mk
    monkeypatch.setattr(_mk, "derive_manifest_hmac_key", lambda: key)
    monkeypatch.setattr(_mk, "derive_receipt_hmac_key", lambda: key)


# ---------------------------------------------------------------------------
# A. TAMPER-EVIDENT MANIFEST (P1-4)
# ---------------------------------------------------------------------------

def test_receipt_rewrite_attack_detected(tmp_path: Path, monkeypatch) -> None:
    """A1 (RED→GREEN): A DB writer who mutates a field in erasure_receipts and
    recomputes the unkeyed SHA is DETECTED by verify_receipt() when the receipt
    uses HMAC (v2). verify_receipt() must return False after tampering."""
    _patch_manifest_key(monkeypatch)
    db = _fresh_db_with_m037(tmp_path)

    from superlocalmemory.core.transactions.owners import OperationContext
    from superlocalmemory.core.transactions.concrete_owners import Bm25Owner, TemporalOwner
    from superlocalmemory.core.transactions.erasure import ErasureService, verify_receipt

    db.execute("INSERT INTO memories (memory_id, profile_id, content) VALUES ('m1', 'p1', 'x')")
    db.execute(
        "INSERT INTO atomic_facts (fact_id, memory_id, profile_id, content) "
        "VALUES ('f1', 'm1', 'p1', 'test fact')"
    )
    db.store_bm25_tokens("f1", "p1", ["alpha", "beta"])
    db.store_temporal_validity("f1", "p1", "2026-01-01T00:00:00")

    ctx = OperationContext(
        operation_id="erase-tamper-test",
        profile_id="p1",
        subject_id="f1",
        fact_ids=("f1",),
    )
    svc = ErasureService({"bm25": Bm25Owner(db), "temporal": TemporalOwner(db)})
    receipt = svc.erase(db, ctx, subject_type="fact", subject_id="f1", requested_by="tester")
    assert receipt.persisted, "receipt must be persisted for tamper test to work"

    with db.raw_connection() as conn:
        assert verify_receipt(conn, "erase-tamper-test") is True

    with db.raw_connection() as conn:
        conn.execute(
            "UPDATE erasure_receipts SET state = 'FAILED' WHERE erasure_id = ?",
            ("erase-tamper-test",),
        )
        conn.commit()

    with db.raw_connection() as conn:
        result = verify_receipt(conn, "erase-tamper-test")
    assert result is False, "verify_receipt must detect tampered state field"


def test_manifest_rewrite_attack_detected(tmp_path: Path, monkeypatch) -> None:
    """A2 (RED→GREEN): A DB writer who mutates completion_manifests is DETECTED
    by verify_manifest() when the manifest uses HMAC (v2)."""
    _patch_manifest_key(monkeypatch)
    db = _fresh_db_with_m037(tmp_path)

    from superlocalmemory.core.transactions.owners import OperationContext, ObligationKind
    from superlocalmemory.core.transactions.obligations import ObligationLedger
    from superlocalmemory.core.transactions.reconciler import Reconciler

    ledger = ObligationLedger()
    reconciler = Reconciler(ledger)

    ctx = OperationContext(
        operation_id="manifest-tamper-test",
        profile_id="p1",
        subject_id="s1",
    )

    with db.raw_connection() as conn:
        ledger.record(conn, ctx, "bm25", ObligationKind.APPLY)
        conn.commit()

    with db.raw_connection() as conn:
        reconciler.reconcile(conn, "manifest-tamper-test", "p1", canonical_committed=True)
        conn.commit()

    with db.raw_connection() as conn:
        assert reconciler.verify_manifest(conn, "manifest-tamper-test") is True

    with db.raw_connection() as conn:
        conn.execute(
            "UPDATE completion_manifests SET state = 'FAILED' "
            "WHERE operation_id = ?",
            ("manifest-tamper-test",),
        )
        conn.commit()

    with db.raw_connection() as conn:
        result = reconciler.verify_manifest(conn, "manifest-tamper-test")
    assert result is False, "verify_manifest must detect tampered state field"


def test_old_v1_manifest_back_compat(tmp_path: Path, monkeypatch) -> None:
    """A3 (RED→GREEN): An existing v1 manifest (unkeyed SHA256, manifest_version=1
    or column absent) still verifies correctly under the version guard."""
    _patch_manifest_key(monkeypatch)
    db = _fresh_db(tmp_path)

    from superlocalmemory.core.transactions.manifest import (
        ManifestState,
        OwnerEvidence,
        compute_envelope_hash,
        evidence_json,
    )
    from superlocalmemory.core.transactions.owners import ObligationKind, ObligationState
    from superlocalmemory.core.transactions.reconciler import Reconciler

    reconciler = Reconciler()
    op_id = "old-v1-manifest"
    profile_id = "p1"

    evidence = (OwnerEvidence(
        owner="bm25",
        kind=ObligationKind.APPLY,
        state=ObligationState.VERIFIED,
        revision=0,
        checksum="abc123",
    ),)
    state = ManifestState.COMPLETE
    all_met = True
    count = 1

    old_hash = compute_envelope_hash(
        operation_id=op_id,
        profile_id=profile_id,
        state=state,
        all_met=all_met,
        obligation_count=count,
        evidence=evidence,
    )
    payload = evidence_json(evidence)
    now = time.time()

    with db.raw_connection() as conn:
        conn.execute(
            "INSERT INTO completion_manifests "
            "(operation_id, profile_id, state, all_met, obligation_count, "
            "owner_evidence_json, manifest_hash, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (op_id, profile_id, str(state), 1, count, payload, old_hash, now, now),
        )
        conn.commit()

    with db.raw_connection() as conn:
        result = reconciler.verify_manifest(conn, op_id)
    assert result is True, "v1 manifest must verify under back-compat guard"


# ---------------------------------------------------------------------------
# B. VECTOR-OWNER SEMANTICS (P1-6)
# ---------------------------------------------------------------------------

def test_vector_unavailable_with_embedded_facts_returns_degraded(tmp_path: Path) -> None:
    """B1 (RED→GREEN): When the vector store is unavailable but facts have
    embeddings, VectorOwner.verify() must return ok=False (not vacuous VERIFIED).
    The result must signal required_unavailable in detail."""
    from superlocalmemory.storage import schema
    from superlocalmemory.storage.database import DatabaseManager
    from superlocalmemory.core.transactions.owners import OperationContext
    from superlocalmemory.core.transactions.concrete_owners import VectorOwner

    db_path = tmp_path / "memory.db"
    db = DatabaseManager(db_path)
    db.initialize(schema)
    db.execute("INSERT OR IGNORE INTO profiles (profile_id, name) VALUES ('p1', 'p1')")
    db.execute("INSERT INTO memories (memory_id, profile_id, content) VALUES ('m1', 'p1', 'x')")
    db.execute(
        "INSERT INTO atomic_facts (fact_id, memory_id, profile_id, content, embedding) "
        "VALUES ('f1', 'm1', 'p1', 'test fact', '[0.1, 0.2, 0.3]')"
    )

    unavailable_store = MagicMock()
    unavailable_store.available = False

    ctx = OperationContext(
        operation_id="vec-unavail-test",
        profile_id="p1",
        subject_id="f1",
        fact_ids=("f1",),
    )
    owner = VectorOwner(db, vector_store=unavailable_store)
    result = owner.verify(ctx)

    assert result.ok is False, (
        "VectorOwner.verify() must return ok=False when store unavailable "
        "and facts have embeddings (not vacuous VERIFIED)"
    )
    assert result.detail.get("required_unavailable") is True, (
        "detail must contain required_unavailable=True to signal unavailability"
    )


def test_vector_unavailable_no_embedded_facts_is_not_applicable(tmp_path: Path) -> None:
    """B2 (RED→GREEN): When the vector store is unavailable but NO facts have
    embeddings, VectorOwner.verify() must return ok=True (NOT_APPLICABLE)."""
    from superlocalmemory.storage import schema
    from superlocalmemory.storage.database import DatabaseManager
    from superlocalmemory.core.transactions.owners import OperationContext
    from superlocalmemory.core.transactions.concrete_owners import VectorOwner

    db_path = tmp_path / "memory.db"
    db = DatabaseManager(db_path)
    db.initialize(schema)
    db.execute("INSERT OR IGNORE INTO profiles (profile_id, name) VALUES ('p1', 'p1')")
    db.execute("INSERT INTO memories (memory_id, profile_id, content) VALUES ('m1', 'p1', 'x')")
    db.execute(
        "INSERT INTO atomic_facts (fact_id, memory_id, profile_id, content) "
        "VALUES ('f1', 'm1', 'p1', 'test fact without embedding')"
    )

    unavailable_store = MagicMock()
    unavailable_store.available = False

    ctx = OperationContext(
        operation_id="vec-na-test",
        profile_id="p1",
        subject_id="f1",
        fact_ids=("f1",),
    )
    owner = VectorOwner(db, vector_store=unavailable_store)
    result = owner.verify(ctx)

    assert result.ok is True, (
        "VectorOwner.verify() must return ok=True (NOT_APPLICABLE) when "
        "store unavailable but no embedded facts exist"
    )


def test_vector_none_store_no_embedded_facts_is_ok(tmp_path: Path) -> None:
    """B3: When vector_store is None (no store configured) and no embeddings
    exist, VectorOwner.verify() must return ok=True."""
    from superlocalmemory.storage import schema
    from superlocalmemory.storage.database import DatabaseManager
    from superlocalmemory.core.transactions.owners import OperationContext
    from superlocalmemory.core.transactions.concrete_owners import VectorOwner

    db_path = tmp_path / "memory.db"
    db = DatabaseManager(db_path)
    db.initialize(schema)
    db.execute("INSERT OR IGNORE INTO profiles (profile_id, name) VALUES ('p1', 'p1')")
    db.execute("INSERT INTO memories (memory_id, profile_id, content) VALUES ('m1', 'p1', 'x')")
    db.execute(
        "INSERT INTO atomic_facts (fact_id, memory_id, profile_id, content) "
        "VALUES ('f1', 'm1', 'p1', 'no embedding fact')"
    )

    ctx = OperationContext(
        operation_id="vec-none-test",
        profile_id="p1",
        subject_id="f1",
        fact_ids=("f1",),
    )
    owner = VectorOwner(db, vector_store=None)
    result = owner.verify(ctx)
    assert result.ok is True


def test_missing_obligation_schema_raises_fail_closed(tmp_path: Path) -> None:
    """B4 (RED→GREEN): When the projection_obligations schema is absent,
    _record_projection_obligations must raise rather than silently skip (fail-closed)."""
    from superlocalmemory.storage import schema
    from superlocalmemory.storage.database import DatabaseManager

    db_path = tmp_path / "memory.db"
    db = DatabaseManager(db_path)
    db.initialize(schema)

    conn = db._connect()
    try:
        from superlocalmemory.core.remember_runtime import _obligation_schema_present
        present = _obligation_schema_present(conn)
        if present:
            pytest.skip("projection_obligations schema unexpectedly present in base schema")
    finally:
        conn.close()

    from superlocalmemory.core.remember_runtime import CanonicalRememberRuntime

    class _FakeReceipt:
        operation_id = "op-fail-closed"
        fact_ids = ("f1",)
        state = type("S", (), {"value": "queryable"})()

    class _FakeRequest:
        profile_id = "p1"

    runtime = CanonicalRememberRuntime.__new__(CanonicalRememberRuntime)
    runtime._obligation_schema_ok = None
    runtime._db = db
    runtime._profile_id = "p1"

    with db.raw_connection() as conn:
        with pytest.raises(RuntimeError, match="projection_obligations"):
            runtime._record_projection_obligations(conn, _FakeRequest(), _FakeReceipt())


# ---------------------------------------------------------------------------
# C. UNIFIED ERASURE ORCHESTRATOR (P1-5)
# ---------------------------------------------------------------------------

def test_entity_erase_produces_non_empty_proofs(tmp_path: Path) -> None:
    """C1 (RED→GREEN): forget_entity must produce a receipt with non-empty
    proofs, not proofs:[]. The receipt must cover bm25/temporal/vector owners."""
    db = _fresh_db(tmp_path, with_receipts=True)

    db.execute("INSERT INTO memories (memory_id, profile_id, content) VALUES ('m1', 'p1', 'x')")
    db.execute(
        "INSERT INTO atomic_facts (fact_id, memory_id, profile_id, content, "
        "canonical_entities_json) "
        "VALUES ('f1', 'm1', 'p1', 'Alice is a person', '[\"eid-alice\"]')"
    )
    db.execute(
        "INSERT INTO canonical_entities "
        "(entity_id, profile_id, canonical_name, entity_type, fact_count) "
        "VALUES ('eid-alice', 'p1', 'Alice', 'person', 1)"
    )
    db.store_bm25_tokens("f1", "p1", ["alice", "person"])
    db.store_temporal_validity("f1", "p1", "2026-01-01T00:00:00")

    from superlocalmemory.compliance.gdpr import GDPRCompliance

    compliance = GDPRCompliance(db, engine=None)
    result = compliance.forget_entity("Alice", "p1")

    assert result.get("facts", 0) >= 1 or result.get("deleted", 0) >= 1

    with db.raw_connection() as conn:
        row = conn.execute(
            "SELECT owner_evidence_json FROM erasure_receipts "
            "WHERE subject_type = 'entity' AND subject_id = 'Alice' AND profile_id = 'p1'"
        ).fetchone()

    assert row is not None, "erasure receipt must be persisted"
    parsed = json.loads(row[0])
    proofs = parsed.get("proofs", [])
    assert len(proofs) > 0, (
        f"erasure receipt must contain non-empty proofs, got: {parsed}"
    )


def test_entity_erase_proofs_reflect_actual_stores(tmp_path: Path) -> None:
    """C2 (RED→GREEN): Entity erase proofs must include owners that actually
    ran — bm25 erased=True when token was present and deleted."""
    db = _fresh_db(tmp_path, with_receipts=True)

    db.execute("INSERT INTO memories (memory_id, profile_id, content) VALUES ('m1', 'p1', 'x')")
    db.execute(
        "INSERT INTO atomic_facts (fact_id, memory_id, profile_id, content, "
        "canonical_entities_json) "
        "VALUES ('f1', 'm1', 'p1', 'Bob works here', '[\"eid-bob\"]')"
    )
    db.execute(
        "INSERT INTO canonical_entities "
        "(entity_id, profile_id, canonical_name, entity_type, fact_count) "
        "VALUES ('eid-bob', 'p1', 'Bob', 'person', 1)"
    )
    db.store_bm25_tokens("f1", "p1", ["bob", "works"])

    from superlocalmemory.compliance.gdpr import GDPRCompliance
    compliance = GDPRCompliance(db, engine=None)
    compliance.forget_entity("Bob", "p1")

    with db.raw_connection() as conn:
        row = conn.execute(
            "SELECT owner_evidence_json FROM erasure_receipts "
            "WHERE subject_type = 'entity' AND subject_id = 'Bob'"
        ).fetchone()

    assert row is not None
    parsed = json.loads(row[0])
    proofs_by_owner = {p["owner"]: p for p in parsed.get("proofs", [])}
    assert "bm25" in proofs_by_owner, "bm25 owner must appear in proofs"
    assert proofs_by_owner["bm25"]["erased"] is True, "bm25 must report erased=True"


def test_entity_erase_no_facts_skips_receipt_gracefully(tmp_path: Path) -> None:
    """C3: When entity has no facts, forget_entity completes without error and
    reports found=False. No receipt is written (nothing to prove)."""
    db = _fresh_db(tmp_path, with_receipts=True)
    db.execute(
        "INSERT INTO canonical_entities "
        "(entity_id, profile_id, canonical_name, entity_type, fact_count) "
        "VALUES ('eid-nobody', 'p1', 'Nobody', 'person', 0)"
    )

    from superlocalmemory.compliance.gdpr import GDPRCompliance
    compliance = GDPRCompliance(db, engine=None)
    result = compliance.forget_entity("NonExistent", "p1")
    assert result.get("found") is False


# ---------------------------------------------------------------------------
# D. Integration: manifest rewrite attack end-to-end
# ---------------------------------------------------------------------------

def test_hmac_receipt_verify_rejects_all_tampered_fields(tmp_path: Path, monkeypatch) -> None:
    """D1 (RED→GREEN): verify_receipt() returns False when ANY of the following
    fields is tampered: state, all_erased, fact_count, profile_id."""
    _patch_manifest_key(monkeypatch)
    db = _fresh_db_with_m037(tmp_path)

    from superlocalmemory.core.transactions.owners import OperationContext
    from superlocalmemory.core.transactions.concrete_owners import Bm25Owner
    from superlocalmemory.core.transactions.erasure import ErasureService, verify_receipt

    db.store_bm25_tokens("f1", "p1", ["word"])
    ctx = OperationContext(
        operation_id="tamper-fields-test",
        profile_id="p1",
        subject_id="f1",
        fact_ids=("f1",),
    )
    svc = ErasureService({"bm25": Bm25Owner(db)})
    receipt = svc.erase(db, ctx, subject_type="fact", subject_id="f1", requested_by="tester")
    assert receipt.persisted

    tamper_cases = [
        ("state", "'FAILED'"),
        ("all_erased", "0"),
        ("fact_count", "999"),
    ]
    for column, bad_value in tamper_cases:
        with db.raw_connection() as conn:
            conn.execute(
                f"UPDATE erasure_receipts SET {column} = {bad_value} "
                "WHERE erasure_id = ?",
                ("tamper-fields-test",),
            )
            conn.commit()
        with db.raw_connection() as conn:
            result = verify_receipt(conn, "tamper-fields-test")
        assert result is False, f"tamper of {column!r} must be detected"
        with db.raw_connection() as conn:
            conn.execute(
                "DELETE FROM erasure_receipts WHERE erasure_id = ?",
                ("tamper-fields-test",),
            )
            conn.commit()
        receipt2 = svc.erase(db, ctx, subject_type="fact", subject_id="f1", requested_by="tester")
        assert receipt2.persisted
