# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Erase redrive: back-off, entity/profile erasures, and fail-closed vector proof.

Companion to ``test_erase_redrive.py``. Each test here drives the real
background pass, the real erasure service or the real vector store.
"""

from __future__ import annotations

import json
import time

import pytest


def _store(engine, text: str) -> str:
    from superlocalmemory.core.engine_ingestion import (
        canonical_store,
        local_trusted_actor_id,
    )

    operation = canonical_store(
        engine,
        text,
        source_type="python-api",
        trusted_actor_id=local_trusted_actor_id("python-api"),
        require_complete=True,
        return_receipt=True,
    )
    return list(operation.final_fact_ids)[0]


def _redrive(engine, passes: int = 1) -> None:
    from superlocalmemory.server.unified_daemon import (
        _reconcile_pending_projections,
    )

    for _ in range(passes):
        _reconcile_pending_projections(engine, force=True)


def _rows(engine, operation_id: str) -> dict[str, dict]:
    rows = engine._db.execute(
        "SELECT owner, state, attempts, verify_attempts, detail, updated_at "
        "FROM projection_obligations WHERE operation_id = ? AND kind = 'erase'",
        (operation_id,),
    )
    return {dict(r)["owner"]: dict(r) for r in rows}


def _seed_unconfirmed(engine, op_id: str, subject: str, *, verify_attempts: int = 0,
                      age_s: float = 0.0) -> None:
    now = time.time() - age_s
    with engine._db.raw_connection() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO projection_tombstones "
            "(profile_id, fact_id, erasure_id, created_at) VALUES (?, ?, ?, ?)",
            (engine._profile_id, subject, op_id, now),
        )
        for owner in ("bm25", "temporal", "vector"):
            conn.execute(
                "INSERT INTO projection_obligations "
                "(operation_id, profile_id, owner, kind, subject_id, state, attempts, "
                "verify_attempts, created_at, updated_at) "
                "VALUES (?, ?, ?, 'erase', ?, 'failed', 1, ?, ?, ?)",
                (op_id, engine._profile_id, owner, subject, verify_attempts, now, now),
            )


def _age(engine, op_id: str, seconds: float) -> None:
    with engine._db.raw_connection() as conn:
        conn.execute(
            "UPDATE projection_obligations SET updated_at = updated_at - ? "
            "WHERE operation_id = ?",
            (seconds, op_id),
        )


def _exhausted(engine) -> int:
    from superlocalmemory.core.ops_remediation import get_failure_counts

    return get_failure_counts(engine._db.db_path)["exhausted_obligations"]


# ---------------------------------------------------------------------------
# 1. Back-off: an unconfirmed deletion is not rewritten every 30 s forever
# ---------------------------------------------------------------------------


def test_recheck_delay_is_free_then_doubles_then_caps():
    from superlocalmemory.core.transactions.erase_redrive import (
        MAX_REPROOFS,
        RECHECK_BASE_S,
        RECHECK_CAP_S,
        recheck_delay_s,
    )

    assert [recheck_delay_s(n) for n in range(MAX_REPROOFS)] == [0.0] * MAX_REPROOFS
    assert recheck_delay_s(MAX_REPROOFS) == RECHECK_BASE_S
    assert recheck_delay_s(MAX_REPROOFS + 1) == 2 * RECHECK_BASE_S
    assert recheck_delay_s(MAX_REPROOFS + 3) == 8 * RECHECK_BASE_S
    assert recheck_delay_s(MAX_REPROOFS + 50) == RECHECK_CAP_S
    assert recheck_delay_s(10_000) == RECHECK_CAP_S


def test_unconfirmed_deletion_backs_off_but_stays_reported(engine_with_mock_deps):
    from superlocalmemory.core.transactions.erase_redrive import (
        MAX_REPROOFS,
        RECHECK_BASE_S,
    )

    engine = engine_with_mock_deps
    fact_id = _store(engine, "Mateo leads the Madrid data team since 2023-02-14")
    _seed_unconfirmed(engine, "erase-backoff", fact_id)

    _redrive(engine, passes=MAX_REPROOFS + 5)

    assert {r["verify_attempts"] for r in _rows(engine, "erase-backoff").values()} == {
        MAX_REPROOFS,
    }
    assert _exhausted(engine) == 1

    _age(engine, "erase-backoff", RECHECK_BASE_S + 1)
    _redrive(engine, passes=3)

    assert {r["verify_attempts"] for r in _rows(engine, "erase-backoff").values()} == {
        MAX_REPROOFS + 1,
    }
    assert _exhausted(engine) == 1


def test_capped_recheck_runs_once_the_cap_has_passed(engine_with_mock_deps):
    from superlocalmemory.core.transactions.erase_redrive import RECHECK_CAP_S

    engine = engine_with_mock_deps
    fact_id = _store(engine, "Mateo leads the Madrid data team since 2023-02-14")
    _seed_unconfirmed(engine, "erase-capped", fact_id, verify_attempts=40,
                      age_s=RECHECK_CAP_S - 60)

    _redrive(engine)
    assert {r["verify_attempts"] for r in _rows(engine, "erase-capped").values()} == {40}

    _age(engine, "erase-capped", 61)
    _redrive(engine)
    assert {r["verify_attempts"] for r in _rows(engine, "erase-capped").values()} == {41}


def test_backed_off_deletion_still_closes_on_reconcile(engine_with_mock_deps):
    """Back-off delays the background check only; Reconcile always re-proves."""
    from superlocalmemory.core.ops_remediation import resolve_operation

    engine = engine_with_mock_deps
    fact_id = _store(engine, "Mateo leads the Madrid data team since 2023-02-14")
    _seed_unconfirmed(engine, "erase-manual", fact_id, verify_attempts=30)
    # A later delete of the same memory succeeds and removes every trace of it.
    from superlocalmemory.core.engine_ingestion import local_trusted_actor_id
    from superlocalmemory.core.mutations import delete_fact_authorized

    later = delete_fact_authorized(
        engine, fact_id,
        trusted_actor_id=local_trusted_actor_id("dashboard"),
        source_agent_id="dashboard",
    )
    assert later["ok"] is True, later

    _redrive(engine)
    assert {r["state"] for r in _rows(engine, "erase-manual").values()} == {"failed"}

    result = resolve_operation(engine._db.db_path, engine, "erase-manual", "force_reconcile")
    assert result["success"] is True, result
    assert {r["state"] for r in _rows(engine, "erase-manual").values()} == {"erased"}


# ---------------------------------------------------------------------------
# 2. Entity and profile erasures are re-proven from their receipt's fact list
# ---------------------------------------------------------------------------


def _tag_entity(engine, fact_ids: list[str], name: str) -> None:
    with engine._db.raw_connection() as conn:
        conn.execute(
            "INSERT INTO canonical_entities "
            "(entity_id, profile_id, canonical_name, entity_type, fact_count) "
            "VALUES ('eid-acme', ?, ?, 'organization', ?)",
            (engine._profile_id, name, len(fact_ids)),
        )
        for fid in fact_ids:
            conn.execute(
                "UPDATE atomic_facts SET canonical_entities_json = '[\"eid-acme\"]' "
                "WHERE fact_id = ?",
                (fid,),
            )


def _entity_erasure_id(engine, name: str) -> str:
    from superlocalmemory.compliance.erasure_outcome import subject_ref

    # The receipt names the erasure by a reference, never by the erased name.
    rows = engine._db.execute(
        "SELECT erasure_id FROM erasure_receipts WHERE subject_type = 'entity' "
        "AND subject_id = ?",
        (subject_ref("entity", name),),
    )
    return dict(rows[0])["erasure_id"]


def test_entity_erasure_heals_from_its_receipt(engine_with_mock_deps, monkeypatch):
    from superlocalmemory.compliance.gdpr import GDPRCompliance
    from superlocalmemory.core.transactions import concrete_owners

    engine = engine_with_mock_deps
    first = _store(engine, "Acme Corp signed the Lisbon lease on 2024-03-02")
    second = _store(engine, "Acme Corp hired Ingrid as Oslo lead on 2024-05-20")
    _tag_entity(engine, [first, second], "Kestrel Programme")

    def _locked(self, context, fact_id):
        raise OSError("bm25 index is locked")

    with monkeypatch.context() as patched:
        patched.setattr(concrete_owners.Bm25Owner, "_remove", _locked)
        GDPRCompliance(engine._db, engine=engine).forget_entity(
            "Kestrel Programme", engine._profile_id,
        )
    erasure_id = _entity_erasure_id(engine, "Kestrel Programme")
    assert _rows(engine, erasure_id)["bm25"]["state"] == "failed"

    # The memories are gone; clear whatever bm25 residue the failed purge left.
    for fid in (first, second):
        engine._db.delete_bm25_tokens_for_fact(fid)
    _redrive(engine)

    assert {r["state"] for r in _rows(engine, erasure_id).values()} == {"erased"}
    assert _exhausted(engine) == 0


def test_entity_erasure_is_not_closed_while_a_listed_memory_remains(
    engine_with_mock_deps, monkeypatch,
):
    from superlocalmemory.compliance.gdpr import GDPRCompliance
    from superlocalmemory.core.transactions import concrete_owners

    engine = engine_with_mock_deps
    first = _store(engine, "Acme Corp signed the Lisbon lease on 2024-03-02")
    second = _store(engine, "Acme Corp hired Ingrid as Oslo lead on 2024-05-20")
    _tag_entity(engine, [first, second], "Kestrel Programme")

    def _locked(self, context, fact_id):
        raise OSError("bm25 index is locked")

    with monkeypatch.context() as patched:
        patched.setattr(concrete_owners.Bm25Owner, "_remove", _locked)
        GDPRCompliance(engine._db, engine=engine).forget_entity(
            "Kestrel Programme", engine._profile_id,
        )
    erasure_id = _entity_erasure_id(engine, "Kestrel Programme")
    # 4.1.22: deleting a fact now removes its token row in the same transaction
    # (storage/fact_dependents.py), so residue no longer survives on its own.
    # Plant it for the second memory: the redrive must still refuse to close.
    engine._db.store_bm25_tokens(second, engine._profile_id, ["acme", "ingrid"])

    # 4.2.8: the redrive finishes a removal left undone, so the index must still
    # refuse the removal for the erasure to stay open.
    with monkeypatch.context() as patched:
        patched.setattr(concrete_owners.Bm25Owner, "_remove", _locked)
        _redrive(engine)

    bm25 = _rows(engine, erasure_id)["bm25"]
    assert bm25["state"] == "failed", bm25
    assert "residue" in bm25["detail"] and "OSError" in bm25["detail"], bm25


def _write_receipt_without_fact_list(engine, erasure_id: str, subject: str) -> None:
    from superlocalmemory.core.transactions.erasure import compute_erasure_hmac

    evidence = json.dumps({"proofs": []}, sort_keys=True, separators=(",", ":"))
    fields = dict(
        erasure_id=erasure_id, profile_id=engine._profile_id, subject_type="entity",
        subject_id=subject, requested_by="gdpr", fact_count=0, state="FAILED",
        all_erased=False, evidence_json=evidence, requested_at=1.0, completed_at=2.0,
    )
    with engine._db.raw_connection() as conn:
        conn.execute(
            "INSERT INTO erasure_receipts (erasure_id, profile_id, subject_type, "
            "subject_id, requested_by, fact_count, state, all_erased, "
            "owner_evidence_json, audit_hash, receipt_version, requested_at, "
            "completed_at) VALUES (?, ?, 'entity', ?, 'gdpr', 0, 'FAILED', 0, ?, ?, 2, "
            "1.0, 2.0)",
            (erasure_id, engine._profile_id, subject, evidence,
             compute_erasure_hmac(**fields)),
        )


def test_receipt_without_a_fact_list_is_reported_plainly(engine_with_mock_deps):
    from superlocalmemory.core.ops_remediation import resolve_operation
    from superlocalmemory.core.transactions.erasure import verify_receipt

    from superlocalmemory.core.ops_remediation import list_failed_operations

    engine = engine_with_mock_deps
    _seed_unconfirmed(engine, "erase-no-list", "Acme Corp", verify_attempts=9)
    _write_receipt_without_fact_list(engine, "erase-no-list", "Acme Corp")
    with engine._db.raw_connection() as conn:
        assert verify_receipt(conn, "erase-no-list") is True

    result = resolve_operation(engine._db.db_path, engine, "erase-no-list", "force_reconcile")

    expected = "its erasure receipt does not list which memories it covered"
    assert result["success"] is False
    assert expected in result["reason"], result
    assert all(expected in r["detail"] for r in _rows(engine, "erase-no-list").values())
    # Reported, with that reason, by the same list the CLI and dashboard read.
    entries = list_failed_operations(engine._db.db_path)["exhausted_obligations"]
    entry = next(e for e in entries if e["operation_id"] == "erase-no-list")
    assert entry["kind"] == "erase"
    assert expected in entry["error"], entry


def test_tampered_receipt_is_not_trusted(engine_with_mock_deps):
    engine = engine_with_mock_deps
    _seed_unconfirmed(engine, "erase-tampered", "Acme Corp")
    _write_receipt_without_fact_list(engine, "erase-tampered", "Acme Corp")
    with engine._db.raw_connection() as conn:
        conn.execute(
            "UPDATE erasure_receipts SET owner_evidence_json = ? WHERE erasure_id = ?",
            (json.dumps({"fact_ids": [], "proofs": []}), "erase-tampered"),
        )

    _redrive(engine)

    for row in _rows(engine, "erase-tampered").values():
        assert row["state"] == "failed", row
        assert "failed its integrity check" in row["detail"], row


# ---------------------------------------------------------------------------
# 3. An unreadable vector store is never proof that a vector is gone
# ---------------------------------------------------------------------------


def _unreadable_store(tmp_path):
    from superlocalmemory.retrieval.vector_store import VectorStore, VectorStoreConfig

    not_a_database = tmp_path / "vectors-unreadable"
    not_a_database.mkdir()
    store = VectorStore(tmp_path / "vectors.db", VectorStoreConfig())
    store._available = False
    store._db_path = not_a_database
    return store


def test_raw_vector_present_raises_when_the_store_cannot_be_read(tmp_path):
    from superlocalmemory.retrieval.vector_presence import VectorPresenceUnknown

    store = _unreadable_store(tmp_path)

    with pytest.raises(VectorPresenceUnknown):
        store.raw_vector_present("f1")


def test_raw_vector_present_available_path_also_fails_closed(tmp_path, monkeypatch):
    from superlocalmemory.retrieval.vector_presence import VectorPresenceUnknown

    store = _unreadable_store(tmp_path)
    store._available = True

    with pytest.raises(VectorPresenceUnknown):
        store.raw_vector_present("f1")


def test_unreadable_vector_store_does_not_prove_erasure(engine_with_mock_deps, tmp_path):
    from superlocalmemory.core.transactions import OperationContext
    from superlocalmemory.core.transactions.concrete_owners import build_erasure_service

    engine = engine_with_mock_deps
    engine._vector_store = _unreadable_store(tmp_path)
    context = OperationContext(
        operation_id="erase-vector-unknown", profile_id=engine._profile_id,
        subject_id="gone-fact", fact_ids=("gone-fact",),
    )

    proof = build_erasure_service(engine).prove_erased(context, "vector")

    assert proof.erased is False
    assert proof.residue == ("gone-fact",)


# ---------------------------------------------------------------------------
# 4. The CLI shows the kind, the explanation and the reason
# ---------------------------------------------------------------------------


def test_cli_ops_list_explains_an_unconfirmed_deletion(monkeypatch, capsys):
    from argparse import Namespace

    from superlocalmemory.cli import ops_cmd

    payload = {
        "dead_letter": [], "degraded_manifests": [], "total": 1,
        "exhausted_obligations": [{
            "category": "exhausted_obligation", "operation_id": "erase-1",
            "kind": "erase", "attempts": 1, "profile_id": "default",
            "error": "deletion not confirmed: the memory is still stored",
            "what_happened": "A deletion could not be confirmed.",
        }],
    }
    monkeypatch.setattr(ops_cmd, "_daemon_get", lambda path: payload)

    ops_cmd._cmd_ops_list(Namespace(profile=None, json=False))

    out = capsys.readouterr().out
    assert "kind=erase" in out
    assert "A deletion could not be confirmed." in out
    assert "deletion not confirmed: the memory is still stored" in out
