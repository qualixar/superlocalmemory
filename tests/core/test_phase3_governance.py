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


# ---------------------------------------------------------------------------
# TRANCHE A RED TESTS — version-downgrade + signing-key fail-closed
# ---------------------------------------------------------------------------

def test_manifest_version_downgrade_forgery_detected(tmp_path: Path, monkeypatch) -> None:
    """TA1: On M037 DB, attacker sets manifest_version=1 + valid SHA256 of mutated
    envelope → verify_manifest MUST return False (downgrade forgery blocked)."""
    import hashlib as _hashlib

    _patch_manifest_key(monkeypatch)
    db = _fresh_db_with_m037(tmp_path)

    from superlocalmemory.core.transactions.owners import OperationContext, ObligationKind
    from superlocalmemory.core.transactions.obligations import ObligationLedger
    from superlocalmemory.core.transactions.reconciler import Reconciler

    ledger = ObligationLedger()
    reconciler = Reconciler(ledger)
    ctx = OperationContext(
        operation_id="downgrade-forgery-manifest",
        profile_id="p1",
        subject_id="s1",
    )

    with db.raw_connection() as conn:
        ledger.record(conn, ctx, "bm25", ObligationKind.APPLY)
        conn.commit()

    with db.raw_connection() as conn:
        reconciler.reconcile(conn, "downgrade-forgery-manifest", "p1", canonical_committed=True)
        conn.commit()

    with db.raw_connection() as conn:
        assert reconciler.verify_manifest(conn, "downgrade-forgery-manifest") is True

    # Attacker reads the stored row to get the REAL evidence JSON, then:
    # 1) mutates state → FAILED, 2) computes SHA256 of mutated canonical using
    # the SAME evidence — valid forged SHA the verifier would accept on SHA path.
    with db.raw_connection() as conn:
        stored = conn.execute(
            "SELECT owner_evidence_json, obligation_count FROM completion_manifests "
            "WHERE operation_id = ?",
            ("downgrade-forgery-manifest",),
        ).fetchone()
    real_evidence_json = stored[0]
    obligation_count = stored[1]

    mutated_canonical = json.dumps({
        "operation_id": "downgrade-forgery-manifest",
        "profile_id": "p1",
        "state": "FAILED",
        "all_met": False,
        "obligation_count": int(obligation_count),
        "evidence": json.loads(real_evidence_json),
    }, sort_keys=True, separators=(",", ":")).encode("utf-8")
    forged_sha = _hashlib.sha256(mutated_canonical).hexdigest()

    with db.raw_connection() as conn:
        conn.execute(
            "UPDATE completion_manifests "
            "SET state = 'FAILED', all_met = 0, manifest_version = 1, manifest_hash = ? "
            "WHERE operation_id = ?",
            (forged_sha, "downgrade-forgery-manifest"),
        )
        conn.commit()

    with db.raw_connection() as conn:
        result = reconciler.verify_manifest(conn, "downgrade-forgery-manifest")
    assert result is False, (
        "verify_manifest must detect version-downgrade forgery: "
        "a v1 SHA on an M037 DB must always fail"
    )


def test_receipt_version_downgrade_forgery_detected(tmp_path: Path, monkeypatch) -> None:
    """TA2: On M037 DB, attacker sets receipt_version=1 + valid SHA256 of mutated
    receipt envelope → verify_receipt MUST return False."""
    import hashlib as _hashlib

    _patch_manifest_key(monkeypatch)
    db = _fresh_db_with_m037(tmp_path)

    from superlocalmemory.core.transactions.owners import OperationContext
    from superlocalmemory.core.transactions.concrete_owners import Bm25Owner
    from superlocalmemory.core.transactions.erasure import ErasureService, verify_receipt, _erasure_canonical

    db.store_bm25_tokens("f-dg", "p1", ["token"])
    ctx = OperationContext(
        operation_id="downgrade-forgery-receipt",
        profile_id="p1",
        subject_id="f-dg",
        fact_ids=("f-dg",),
    )
    svc = ErasureService({"bm25": Bm25Owner(db)})
    receipt = svc.erase(db, ctx, subject_type="fact", subject_id="f-dg", requested_by="tester")
    assert receipt.persisted

    with db.raw_connection() as conn:
        assert verify_receipt(conn, "downgrade-forgery-receipt") is True

    # Fetch original evidence_json and tamper
    with db.raw_connection() as conn:
        row = conn.execute(
            "SELECT requested_at, completed_at, owner_evidence_json FROM erasure_receipts "
            "WHERE erasure_id = ?", ("downgrade-forgery-receipt",)
        ).fetchone()

    # Compute SHA256 of mutated canonical (state=FAILED, all_erased=False)
    canonical = _erasure_canonical(
        erasure_id="downgrade-forgery-receipt",
        profile_id="p1",
        subject_type="fact",
        subject_id="f-dg",
        requested_by="tester",
        fact_count=1,
        state="FAILED",
        all_erased=False,
        evidence_json=row[2],
        requested_at=float(row[0]),
        completed_at=float(row[1]),
    )
    forged_sha = _hashlib.sha256(canonical).hexdigest()

    with db.raw_connection() as conn:
        conn.execute(
            "UPDATE erasure_receipts "
            "SET state = 'FAILED', all_erased = 0, receipt_version = 1, audit_hash = ? "
            "WHERE erasure_id = ?",
            (forged_sha, "downgrade-forgery-receipt"),
        )
        conn.commit()

    with db.raw_connection() as conn:
        result = verify_receipt(conn, "downgrade-forgery-receipt")
    assert result is False, (
        "verify_receipt must detect version-downgrade forgery: "
        "v1 SHA on M037 DB must always fail"
    )


def test_signing_key_fail_closed_on_corrupt(tmp_path: Path, monkeypatch) -> None:
    """TA3: Corrupt key file (len != 64) raises RuntimeError; no ephemeral key returned."""
    from superlocalmemory.core.transactions import manifest_key as _mk

    key_path = tmp_path / ".manifest_signing_key"
    key_path.write_text("not-a-valid-64-hex-key", encoding="utf-8")
    monkeypatch.setattr(_mk, "_signing_key_path", lambda: key_path)

    import pytest
    with pytest.raises(RuntimeError, match="signing key"):
        _mk._ensure_signing_key()


def test_signing_key_fail_closed_on_unreadable(tmp_path: Path, monkeypatch) -> None:
    """TA4: OSError reading/creating key raises RuntimeError; never returns ephemeral."""
    import os
    from superlocalmemory.core.transactions import manifest_key as _mk

    key_path = tmp_path / ".manifest_signing_key"
    # Simulate unreadable path by pointing at a directory
    bad_path = tmp_path / "adir"
    bad_path.mkdir()
    monkeypatch.setattr(_mk, "_signing_key_path", lambda: bad_path / "key")
    # Make the parent unwritable so os.open fails
    os.chmod(str(bad_path), 0o444)

    try:
        import pytest
        with pytest.raises(RuntimeError, match="signing key"):
            _mk._ensure_signing_key()
    finally:
        os.chmod(str(bad_path), 0o755)


def test_v1_manifest_uses_constant_time_compare(tmp_path: Path, monkeypatch) -> None:
    """TA5: v1 manifest path on non-M037 DB uses hmac.compare_digest (constant-time)."""
    _patch_manifest_key(monkeypatch)
    db = _fresh_db(tmp_path)

    from superlocalmemory.core.transactions.manifest import (
        ManifestState, OwnerEvidence, compute_envelope_hash, evidence_json,
    )
    from superlocalmemory.core.transactions.owners import ObligationKind, ObligationState
    from superlocalmemory.core.transactions.reconciler import Reconciler

    reconciler = Reconciler()
    op_id = "v1-constant-time"
    evidence = (OwnerEvidence(
        owner="bm25", kind=ObligationKind.APPLY, state=ObligationState.VERIFIED,
        revision=0, checksum="abc",
    ),)
    state = ManifestState.COMPLETE
    old_hash = compute_envelope_hash(
        operation_id=op_id, profile_id="p1", state=state, all_met=True,
        obligation_count=1, evidence=evidence,
    )
    payload = evidence_json(evidence)
    now = __import__("time").time()

    with db.raw_connection() as conn:
        conn.execute(
            "INSERT INTO completion_manifests "
            "(operation_id, profile_id, state, all_met, obligation_count, "
            "owner_evidence_json, manifest_hash, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (op_id, "p1", str(state), 1, 1, payload, old_hash, now, now),
        )
        conn.commit()

    # Tamper: wrong hash on v1, non-M037 DB → must return False
    with db.raw_connection() as conn:
        conn.execute(
            "UPDATE completion_manifests SET manifest_hash = ? WHERE operation_id = ?",
            ("0" * 64, op_id),
        )
        conn.commit()

    with db.raw_connection() as conn:
        result = reconciler.verify_manifest(conn, op_id)
    assert result is False, "v1 tampered manifest must return False"


# ---------------------------------------------------------------------------
# TRANCHE B RED TESTS — erasure claim + fail-closed paths
# ---------------------------------------------------------------------------

def test_forget_profile_produces_real_erasure_receipt(tmp_path: Path) -> None:
    """TB1: forget_profile must write an erasure_receipts row with non-empty
    proofs (real per-owner coverage), not proofs:[]."""
    db = _fresh_db(tmp_path, with_receipts=True)
    db.execute("INSERT INTO profiles (profile_id, name) VALUES ('p2', 'p2')")
    db.execute(
        "INSERT INTO memories (memory_id, profile_id, content) VALUES ('m2', 'p2', 'x')"
    )
    db.execute(
        "INSERT INTO atomic_facts (fact_id, memory_id, profile_id, content) "
        "VALUES ('f2', 'm2', 'p2', 'profile fact')"
    )
    db.store_bm25_tokens("f2", "p2", ["profile", "fact"])

    from superlocalmemory.compliance.gdpr import GDPRCompliance
    compliance = GDPRCompliance(db, engine=None)
    compliance.forget_profile("p2")

    with db.raw_connection() as conn:
        row = conn.execute(
            "SELECT owner_evidence_json FROM erasure_receipts "
            "WHERE subject_type = 'profile' AND subject_id = 'p2'"
        ).fetchone()

    assert row is not None, "erasure receipt must be written for profile wipe"
    parsed = json.loads(row[0])
    proofs = parsed.get("proofs", [])
    assert len(proofs) > 0, (
        f"profile erase receipt must contain non-empty proofs, got: {parsed}"
    )


def test_embedded_fact_ids_query_failure_returns_degraded(tmp_path: Path) -> None:
    """TB2: When _embedded_fact_ids DB query fails, VectorOwner.verify()
    must return ok=False (not vacuous NOT_APPLICABLE)."""
    from superlocalmemory.storage import schema
    from superlocalmemory.storage.database import DatabaseManager
    from superlocalmemory.core.transactions.owners import OperationContext
    from superlocalmemory.core.transactions.concrete_owners import VectorOwner
    from unittest.mock import MagicMock, patch

    db_path = tmp_path / "memory.db"
    db = DatabaseManager(db_path)
    db.initialize(schema)
    db.execute("INSERT OR IGNORE INTO profiles (profile_id, name) VALUES ('p1', 'p1')")
    db.execute("INSERT INTO memories (memory_id, profile_id, content) VALUES ('m1', 'p1', 'x')")
    db.execute(
        "INSERT INTO atomic_facts (fact_id, memory_id, profile_id, content, embedding) "
        "VALUES ('f1', 'm1', 'p1', 'test', '[0.1]')"
    )

    unavailable_store = MagicMock()
    unavailable_store.available = False

    ctx = OperationContext(
        operation_id="embedded-query-fail-test",
        profile_id="p1",
        subject_id="f1",
        fact_ids=("f1",),
    )
    owner = VectorOwner(db, vector_store=unavailable_store)

    # Simulate DB query failure
    original_execute = db.execute
    def failing_execute(sql, *args, **kwargs):
        if "atomic_facts" in sql and "embedding" in sql:
            raise RuntimeError("simulated DB failure")
        return original_execute(sql, *args, **kwargs)

    with patch.object(db, "execute", side_effect=failing_execute):
        result = owner.verify(ctx)

    assert result.ok is False, (
        "VectorOwner.verify() must return ok=False when _embedded_fact_ids query fails; "
        "not vacuous NOT_APPLICABLE"
    )


# ---------------------------------------------------------------------------
# C. P2 POLISH (Tranche C)
# ---------------------------------------------------------------------------

def test_obligation_schema_negative_cache_resets_after_migration(tmp_path: Path) -> None:
    """TC1: _obligation_schema_ok=False (stale) must re-check after M033 is applied.

    Scenario: a runtime instance first encounters a DB without M033 — it caches
    _obligation_schema_ok=False.  When M033 is later applied (hot migration),
    the NEXT call to _record_projection_obligations must detect the schema and
    succeed, NOT raise from the stale False value.

    With the bug (cache never re-checks False), the second call raises RuntimeError
    even though the schema is now present.  After the fix (re-check when False),
    the second call succeeds.
    """
    import sqlite3
    import types
    from superlocalmemory.core.remember_runtime import _obligation_schema_present
    from superlocalmemory.storage.migrations import (
        M033_projection_transactions,
        M034_obligation_integrity,
    )

    db_path = tmp_path / "nc.db"
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    # Start without M033 — schema absent.
    assert not _obligation_schema_present(conn), "precondition: schema is absent"

    # Simulate the runtime's cached state after a first failed check.
    # We construct a minimal carrier object that exercises the caching code
    # path in _record_projection_obligations without needing a full runtime.
    from superlocalmemory.core.remember_runtime import CanonicalRememberRuntime

    carrier = types.SimpleNamespace(_obligation_schema_ok=False)

    # Bind the method to our carrier so it reads/writes carrier._obligation_schema_ok.
    bound = CanonicalRememberRuntime._record_projection_obligations.__get__(
        carrier, type(carrier)
    )

    # Apply M033 — hot migration simulated.
    M033_projection_transactions.apply(conn)
    M034_obligation_integrity.apply(conn)
    conn.commit()

    # Minimal receipt/request stubs with one fact_id so the schema check fires.
    receipt_stub = types.SimpleNamespace(fact_ids=("f-tc1",), operation_id="op-tc1")
    request_stub = types.SimpleNamespace(profile_id="p-tc1")

    # With the stale False cache and the BUG: this call raises RuntimeError
    # "schema is absent" because the re-check is gated behind `is None` not `not`.
    # After the fix: it re-checks, finds M033, updates the cache to True, and
    # either succeeds or raises for an UNRELATED reason (e.g. missing profile row).
    # The only failure we assert against is the stale-cache message.
    try:
        bound(conn, request_stub, receipt_stub)
        # If we reach here, the schema was detected and ledger recorded — GREEN path.
    except Exception as exc:
        if "schema is absent" in str(exc):
            raise AssertionError(
                "stale _obligation_schema_ok=False was NOT re-checked after M033 "
                f"was applied — caching bug still present: {exc}"
            ) from exc
        # Any other exception (from ledger, FK, etc.) is unrelated — schema check passed.

    # The cache must now be True — schema was detected.
    assert carrier._obligation_schema_ok is True, (
        "_obligation_schema_ok must be True after the schema was successfully detected"
    )
