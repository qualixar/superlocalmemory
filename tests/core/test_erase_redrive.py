# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""The background redrive must never turn a finished deletion into a failure.

These drive the real delete path (``delete_fact_authorized``) and then the real
30 s background pass, rather than seeding ledger rows by hand, so they fail for
the reason a user saw: a successful dashboard DELETE that ``slm ops status``
reported as DEGRADED a few minutes later.
"""

from __future__ import annotations

import pytest

_PASSES = 12  # more than the ten attempts that exhaust an obligation


def _store(engine, text: str):
    from superlocalmemory.core.engine_ingestion import (
        canonical_store,
        local_trusted_actor_id,
    )

    return canonical_store(
        engine,
        text,
        source_type="python-api",
        trusted_actor_id=local_trusted_actor_id("python-api"),
        require_complete=True,
        return_receipt=True,
    )


def _delete(engine, fact_id: str) -> dict:
    from superlocalmemory.core.engine_ingestion import local_trusted_actor_id
    from superlocalmemory.core.mutations import delete_fact_authorized

    return delete_fact_authorized(
        engine, fact_id,
        trusted_actor_id=local_trusted_actor_id("dashboard"),
        source_agent_id="dashboard",
    )


def _redrive(engine, passes: int = _PASSES) -> None:
    from superlocalmemory.server.unified_daemon import (
        _reconcile_pending_projections,
    )

    for _ in range(passes):
        _reconcile_pending_projections(engine, force=True)


def _erase_rows(engine, operation_id: str) -> dict[str, dict]:
    rows = engine._db.execute(
        "SELECT owner, kind, state, attempts, detail FROM projection_obligations "
        "WHERE operation_id = ? AND kind = 'erase'",
        (operation_id,),
    )
    return {dict(r)["owner"]: dict(r) for r in rows}


@pytest.fixture
def deleted_fact(engine_with_mock_deps):
    engine = engine_with_mock_deps
    operation = _store(
        engine, "Priya joined the Berlin platform team on 2025-01-10 as staff engineer",
    )
    fact_id = list(operation.final_fact_ids)[0]
    result = _delete(engine, fact_id)
    assert result["ok"] is True and result["erasure_verified"] is True, result
    return engine, fact_id, result["erasure_id"]


def test_successful_delete_stays_erased_through_background_redrive(deleted_fact):
    engine, _fact_id, erasure_id = deleted_fact
    before = _erase_rows(engine, erasure_id)
    assert set(before) == {"bm25", "media", "temporal", "vector"}
    assert all(row["state"] == "erased" for row in before.values()), before

    _redrive(engine)

    after = _erase_rows(engine, erasure_id)
    for owner, row in after.items():
        assert row["state"] == "erased", f"{owner}: {row}"
        assert row["attempts"] == before[owner]["attempts"], f"{owner}: {row}"


def test_successful_delete_never_reports_degraded(deleted_fact):
    from superlocalmemory.core.ops_remediation import (
        get_failure_counts,
        list_failed_operations,
    )

    engine, _fact_id, erasure_id = deleted_fact
    _redrive(engine)

    counts = get_failure_counts(engine._db.db_path)
    assert counts["exhausted_obligations"] == 0, counts
    assert counts["degraded_operations"] == 0, counts
    listed = list_failed_operations(engine._db.db_path)
    assert erasure_id not in {e["operation_id"] for e in listed["exhausted_obligations"]}


# ---------------------------------------------------------------------------
# Helpers for the review findings below
# ---------------------------------------------------------------------------


def _record_verified_apply(engine, operation) -> None:
    """An ingestion whose projections are all verified but whose manifest is missing.

    That is the state a crash between the last obligation mark and the manifest
    write leaves behind; only the missing-manifest feed can find it.
    """
    from superlocalmemory.core.transactions import ObligationKind, OperationContext
    from superlocalmemory.core.transactions.concrete_owners import (
        REQUIRED_ADMISSION_OWNERS,
        build_transaction_service,
    )

    context = OperationContext(
        operation_id=operation.operation_id,
        profile_id=engine._profile_id,
        subject_id=operation.operation_id,
        fact_ids=tuple(operation.final_fact_ids),
    )
    service = build_transaction_service(engine)
    with engine._db.raw_connection() as conn:
        service.record(
            conn, context, owners=REQUIRED_ADMISSION_OWNERS, kind=ObligationKind.APPLY,
        )
        conn.execute(
            "UPDATE projection_obligations SET state = 'verified' "
            "WHERE operation_id = ?",
            (operation.operation_id,),
        )


def _seed_erase_rows(
    engine, op_id: str, subject: str, *, state: str, attempts: int = 0,
    profile_id: str | None = None, tombstone: bool = False,
) -> None:
    import time as _time

    pid = profile_id or engine._profile_id
    now = _time.time()
    with engine._db.raw_connection() as conn:
        if tombstone:
            conn.execute(
                "INSERT INTO projection_tombstones "
                "(profile_id, fact_id, erasure_id, created_at) VALUES (?, ?, ?, ?)",
                (pid, subject, op_id, now),
            )
        for owner in ("bm25", "temporal", "vector"):
            conn.execute(
                "INSERT INTO projection_obligations "
                "(operation_id, profile_id, owner, kind, subject_id, state, "
                "attempts, created_at, updated_at) "
                "VALUES (?, ?, ?, 'erase', ?, ?, ?, ?, ?)",
                (op_id, pid, owner, subject, state, attempts, now, now),
            )


def _manifest(engine, operation_id: str) -> dict | None:
    rows = engine._db.execute(
        "SELECT state, all_met FROM completion_manifests WHERE operation_id = ?",
        (operation_id,),
    )
    return dict(rows[0]) if rows else None


def _apply_old_bug(engine, erasure_id: str) -> None:
    """Leave the ledger exactly as 4.1.17-4.1.20 did after ten background passes."""
    from superlocalmemory.core.transactions.reconciler import Reconciler

    with engine._db.raw_connection() as conn:
        conn.execute(
            "UPDATE projection_obligations SET state = 'failed', attempts = 10, "
            "detail = '{\"error\":\"canonical record missing\",\"phase\":\"orphan\"}' "
            "WHERE operation_id = ? AND kind = 'erase'",
            (erasure_id,),
        )
        Reconciler().reconcile(
            conn, erasure_id, engine._profile_id, canonical_committed=False,
        )


def _failing_bm25_remove(monkeypatch) -> None:
    from superlocalmemory.core.transactions import concrete_owners

    def _boom(self, context, fact_id):
        raise OSError("bm25 index is locked")

    monkeypatch.setattr(concrete_owners.Bm25Owner, "_remove", _boom)


# ---------------------------------------------------------------------------
# Finding 1: finished operations must leave the missing-manifest feed
# ---------------------------------------------------------------------------


def test_verified_ingestion_missing_its_manifest_still_gets_one(engine_with_mock_deps):
    engine = engine_with_mock_deps
    operation = _store(engine, "The Lisbon office opened on 2024-06-01 with twelve staff")
    _record_verified_apply(engine, operation)
    assert _manifest(engine, operation.operation_id) is None

    _redrive(engine, passes=1)

    assert _manifest(engine, operation.operation_id) is not None


def test_finished_deletions_do_not_starve_ingestion_manifests(deleted_fact):
    from superlocalmemory.server.unified_daemon import (
        _reconcile_pending_projections,
    )

    engine, _fact_id, _erasure_id = deleted_fact
    # A finished erasure whose id sorts first, as any of hundreds would.
    _seed_erase_rows(engine, "0000-finished-erasure", "gone-fact", state="erased")
    operation = _store(engine, "The Lisbon office opened on 2024-06-01 with twelve staff")
    _record_verified_apply(engine, operation)

    for _ in range(3):
        _reconcile_pending_projections(engine, limit=1, force=True)

    assert _manifest(engine, operation.operation_id) is not None


def test_closing_an_erasure_writes_no_ingestion_manifest(deleted_fact):
    """Manifests describe ingestions; an erasure's record is its receipt."""
    engine, _fact_id, erasure_id = deleted_fact
    with engine._db.raw_connection() as conn:
        conn.execute(
            "UPDATE projection_obligations SET state = 'pending' "
            "WHERE operation_id = ?",
            (erasure_id,),
        )

    _redrive(engine, passes=1)

    assert all(r["state"] == "erased" for r in _erase_rows(engine, erasure_id).values())
    assert _manifest(engine, erasure_id) is None


# ---------------------------------------------------------------------------
# Finding 2: stores upgraded from 4.1.17-4.1.20 are already DEGRADED
# ---------------------------------------------------------------------------


def test_upgrade_heals_erasures_exhausted_by_the_old_redrive(deleted_fact):
    from superlocalmemory.core.ops_remediation import get_failure_counts
    from superlocalmemory.core.transactions.reconciler import Reconciler

    engine, fact_id, erasure_id = deleted_fact
    _apply_old_bug(engine, erasure_id)
    assert get_failure_counts(engine._db.db_path)["exhausted_obligations"] == 1

    _redrive(engine, passes=1)

    for owner, row in _erase_rows(engine, erasure_id).items():
        assert row["state"] == "erased", f"{owner}: {row}"
    assert get_failure_counts(engine._db.db_path)["exhausted_obligations"] == 0
    # The manifest the old code wrote said FAILED; it now tells the truth.
    assert _manifest(engine, erasure_id) == {"state": "COMPLETE", "all_met": 1}
    with engine._db.raw_connection() as conn:
        assert Reconciler().verify_manifest(conn, erasure_id) is True
    # Healing only closes ledger rows; it never brings the memory back.
    assert not engine._db.execute(
        "SELECT 1 FROM atomic_facts WHERE fact_id = ?", (fact_id,),
    )
    assert not engine._db.execute(
        "SELECT 1 FROM bm25_tokens WHERE fact_id = ?", (fact_id,),
    )


def test_upgrade_heal_never_closes_an_erasure_whose_memory_is_still_stored(
    engine_with_mock_deps,
):
    from superlocalmemory.core.ops_remediation import get_failure_counts

    engine = engine_with_mock_deps
    operation = _store(engine, "Mateo leads the Madrid data team since 2023-02-14")
    fact_id = list(operation.final_fact_ids)[0]
    _seed_erase_rows(
        engine, "erase-still-stored", fact_id, state="failed", attempts=10,
        tombstone=True,
    )

    _redrive(engine, passes=2)

    for owner, row in _erase_rows(engine, "erase-still-stored").items():
        assert row["state"] == "failed", f"{owner}: {row}"
    assert get_failure_counts(engine._db.db_path)["exhausted_obligations"] == 1


# ---------------------------------------------------------------------------
# Finding 3: an erasure that cannot be confirmed must be visible, not silent
# ---------------------------------------------------------------------------


def test_unconfirmed_deletion_is_reported_then_clears_once_deleted(
    engine_with_mock_deps, monkeypatch,
):
    from superlocalmemory.core.ops_remediation import (
        get_failure_counts,
        list_failed_operations,
    )

    engine = engine_with_mock_deps
    operation = _store(engine, "Ingrid manages the Oslo support desk since 2022-09-05")
    fact_id = list(operation.final_fact_ids)[0]
    with monkeypatch.context() as patched:
        _failing_bm25_remove(patched)
        failed = _delete(engine, fact_id)
    assert failed["ok"] is False
    first_id = failed["erasure_id"]

    _redrive(engine)

    assert get_failure_counts(engine._db.db_path)["exhausted_obligations"] == 1
    entries = list_failed_operations(engine._db.db_path)["exhausted_obligations"]
    entry = next(e for e in entries if e["operation_id"] == first_id)
    assert entry["kind"] == "erase"
    assert "deletion" in entry["what_happened"].lower()

    retried = _delete(engine, fact_id)
    assert retried["ok"] is True, retried
    # Reported erasures are re-checked with back-off; let the first wait pass.
    with engine._db.raw_connection() as conn:
        conn.execute(
            "UPDATE projection_obligations SET updated_at = updated_at - 31 "
            "WHERE operation_id = ?",
            (first_id,),
        )
    _redrive(engine, passes=1)

    assert all(r["state"] == "erased" for r in _erase_rows(engine, first_id).values())
    assert get_failure_counts(engine._db.db_path)["exhausted_obligations"] == 0


def test_reconcile_action_reproves_a_deletion(deleted_fact):
    from superlocalmemory.core.ops_remediation import resolve_operation

    engine, _fact_id, erasure_id = deleted_fact
    _apply_old_bug(engine, erasure_id)

    result = resolve_operation(
        engine._db.db_path, engine, erasure_id, "force_reconcile",
    )

    assert result["success"] is True, result
    assert all(r["state"] == "erased" for r in _erase_rows(engine, erasure_id).values())


def test_reconcile_action_on_a_finished_deletion_says_so(deleted_fact):
    from superlocalmemory.core.ops_remediation import resolve_operation

    engine, _fact_id, erasure_id = deleted_fact

    result = resolve_operation(
        engine._db.db_path, engine, erasure_id, "force_reconcile",
    )

    assert result["success"] is True, result


def test_reconcile_action_explains_an_unconfirmed_deletion(engine_with_mock_deps):
    from superlocalmemory.core.ops_remediation import resolve_operation

    engine = engine_with_mock_deps
    operation = _store(engine, "Mateo leads the Madrid data team since 2023-02-14")
    fact_id = list(operation.final_fact_ids)[0]
    _seed_erase_rows(engine, "erase-unconfirmed", fact_id, state="failed", tombstone=True)

    result = resolve_operation(
        engine._db.db_path, engine, "erase-unconfirmed", "force_reconcile",
    )

    assert result["success"] is False
    assert "still stored" in result["reason"], result


# ---------------------------------------------------------------------------
# Finding 4: concurrency with a delete that is still in flight
# ---------------------------------------------------------------------------


def test_redrive_inside_a_live_delete_never_undoes_its_proof(
    engine_with_mock_deps, monkeypatch,
):
    """Run the background pass between the tombstone write and the owner purge."""
    from superlocalmemory.core.transactions import concrete_owners

    engine = engine_with_mock_deps
    operation = _store(engine, "Aiko runs the Osaka release train since 2021-11-30")
    fact_id = list(operation.final_fact_ids)[0]
    original = concrete_owners.Bm25Owner.erase

    def _erase_with_redrive(self, context):
        _redrive(engine, passes=1)
        return original(self, context)

    monkeypatch.setattr(concrete_owners.Bm25Owner, "erase", _erase_with_redrive)
    result = _delete(engine, fact_id)
    monkeypatch.setattr(concrete_owners.Bm25Owner, "erase", original)

    assert result["ok"] is True and result["erasure_verified"] is True, result
    _redrive(engine, passes=1)
    for owner, row in _erase_rows(engine, result["erasure_id"]).items():
        assert row["state"] == "erased", f"{owner}: {row}"


def test_unconfirmed_mark_never_overwrites_a_concurrent_close(
    engine_with_mock_deps, monkeypatch,
):
    """The erasure service closes an owner while the redrive is still proving it."""
    from superlocalmemory.core.transactions import erase_redrive

    engine = engine_with_mock_deps
    operation = _store(engine, "Aiko runs the Osaka release train since 2021-11-30")
    fact_id = list(operation.final_fact_ids)[0]
    _seed_erase_rows(engine, "erase-racing", fact_id, state="pending", tombstone=True)
    original = erase_redrive._unproven_reason

    def _close_concurrently(*args, **kwargs):
        reason = original(*args, **kwargs)
        with engine._db.raw_connection() as conn:
            conn.execute(
                "UPDATE projection_obligations SET state = 'erased', "
                "updated_at = updated_at + 1 "
                "WHERE operation_id = 'erase-racing' AND owner = 'bm25'",
            )
        return reason

    monkeypatch.setattr(erase_redrive, "_unproven_reason", _close_concurrently)
    _redrive(engine, passes=1)

    rows = _erase_rows(engine, "erase-racing")
    assert rows["bm25"]["state"] == "erased", rows["bm25"]
    assert rows["temporal"]["state"] == "failed", rows["temporal"]


# ---------------------------------------------------------------------------
# Guards: read-only proof, atomic close, profile scope, mixed operations
# ---------------------------------------------------------------------------


def test_prove_erased_issues_no_writes(deleted_fact):
    from superlocalmemory.core.transactions import OperationContext
    from superlocalmemory.core.transactions.concrete_owners import (
        build_erasure_service,
    )

    engine, fact_id, erasure_id = deleted_fact
    service = build_erasure_service(engine)
    context = OperationContext(
        operation_id=erasure_id, profile_id=engine._profile_id,
        subject_id=fact_id, fact_ids=(fact_id,),
    )
    statements: list[str] = []
    original = engine._db.execute

    def _spy(sql, params=()):
        statements.append(sql.strip().split()[0].upper())
        return original(sql, params)

    engine._db.execute = _spy
    try:
        before = _erase_rows(engine, erasure_id)
        proofs = [service.prove_erased(context, owner) for owner in before]
    finally:
        engine._db.execute = original

    assert all(p.erased for p in proofs), proofs
    assert statements and set(statements) == {"SELECT"}, statements
    assert _erase_rows(engine, erasure_id) == before


def test_a_failure_while_closing_changes_nothing(deleted_fact, monkeypatch):
    """Closing the obligations and re-sealing the manifest commit together or not at all."""
    from superlocalmemory.core.transactions import reconciler

    engine, _fact_id, erasure_id = deleted_fact
    _apply_old_bug(engine, erasure_id)
    with engine._db.raw_connection() as conn:
        conn.execute(
            "UPDATE projection_obligations SET attempts = 1 WHERE operation_id = ?",
            (erasure_id,),
        )
    before_rows = _erase_rows(engine, erasure_id)
    before_manifest = _manifest(engine, erasure_id)

    def _crash(self, *args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(reconciler.Reconciler, "reconcile", _crash)
    _redrive(engine, passes=1)

    assert _erase_rows(engine, erasure_id) == before_rows
    assert _manifest(engine, erasure_id) == before_manifest


def test_a_failure_between_owner_closes_changes_nothing(deleted_fact, monkeypatch):
    """No half-closed erasure: the owners of one subject close together."""
    from superlocalmemory.core.transactions import obligations

    engine, _fact_id, erasure_id = deleted_fact
    with engine._db.raw_connection() as conn:
        conn.execute(
            "UPDATE projection_obligations SET state = 'pending' WHERE operation_id = ?",
            (erasure_id,),
        )
    before_rows = _erase_rows(engine, erasure_id)
    original = obligations.ObligationLedger.mark
    calls: list[str] = []

    def _crash_on_second(self, conn, operation_id, owner, *args, **kwargs):
        calls.append(owner)
        if len(calls) == 2:
            raise OSError("disk full")
        return original(self, conn, operation_id, owner, *args, **kwargs)

    monkeypatch.setattr(obligations.ObligationLedger, "mark", _crash_on_second)
    _redrive(engine, passes=1)

    assert len(calls) == 2, calls
    assert _erase_rows(engine, erasure_id) == before_rows


def test_an_erasure_of_a_profile_that_no_longer_exists_is_left_alone(engine_with_mock_deps):
    """Every existing profile's erasures are re-proven
    (test_erase_redrive_every_profile.py); a profile that is gone has none to
    prove them in."""
    engine = engine_with_mock_deps
    _seed_erase_rows(
        engine, "erase-other-profile", "their-fact", state="failed",
        profile_id="someone-else", tombstone=True,
    )

    _redrive(engine)

    rows = engine._db.execute(
        "SELECT state, attempts, verify_attempts, detail FROM projection_obligations "
        "WHERE operation_id = 'erase-other-profile'",
    )
    assert [dict(r) for r in rows] == [
        {"state": "failed", "attempts": 0, "verify_attempts": 0, "detail": None},
    ] * 3


def test_operation_with_both_kinds_closes_erase_and_reconciles_apply(deleted_fact):
    import json

    engine, deleted_id, _erasure_id = deleted_fact
    operation = _store(engine, "The Lisbon office opened on 2024-06-01 with twelve staff")
    _record_verified_apply(engine, operation)
    with engine._db.raw_connection() as conn:
        conn.execute(
            "UPDATE projection_obligations SET state = 'pending' WHERE operation_id = ?",
            (operation.operation_id,),
        )
    _seed_erase_rows(engine, operation.operation_id, deleted_id, state="pending")

    _redrive(engine, passes=1)

    rows = engine._db.execute(
        "SELECT kind, state, detail FROM projection_obligations WHERE operation_id = ?",
        (operation.operation_id,),
    )
    by_kind: dict[str, list[dict]] = {}
    for row in rows:
        by_kind.setdefault(dict(row)["kind"], []).append(dict(row))
    assert all(r["state"] == "erased" for r in by_kind["erase"]), by_kind["erase"]
    assert all("orphan" not in (r["detail"] or "") for r in by_kind["apply"])
    manifest = engine._db.execute(
        "SELECT owner_evidence_json FROM completion_manifests WHERE operation_id = ?",
        (operation.operation_id,),
    )
    evidence = json.loads(dict(manifest[0])["owner_evidence_json"])
    assert {e["state"] for e in evidence if e["kind"] == "erase"} == {"erased"}
