# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

from __future__ import annotations

import sqlite3

import pytest

from superlocalmemory.core.transactions import ManifestState


def _install_ledger(path, *, with_m033: bool) -> None:
    from superlocalmemory.storage.migrations import (
        M018_ingestion_operations,
        M032_write_coordinator_admission,
        M033_projection_transactions,
        M034_obligation_integrity,
    )

    conn = sqlite3.connect(path)
    try:
        M018_ingestion_operations.apply(conn)
        M032_write_coordinator_admission.apply(conn)
        if with_m033:
            M033_projection_transactions.apply(conn)
            M034_obligation_integrity.apply(conn)
        conn.commit()
    finally:
        conn.close()


def _build_runtime(tmp_path, *, with_m033: bool):
    from superlocalmemory.core.remember_runtime import CanonicalRememberRuntime
    from superlocalmemory.storage import schema
    from superlocalmemory.storage.database import DatabaseManager

    db_path = tmp_path / "memory.db"
    _install_ledger(db_path, with_m033=with_m033)
    db = DatabaseManager(db_path)
    db.initialize(schema)

    def writer(request, operation_id):
        db.execute(
            "INSERT INTO runtime_probe(operation_id, content) VALUES (?, ?)",
            (operation_id, request.content),
        )
        return ["fact-1", "fact-2"]

    db.execute("CREATE TABLE runtime_probe(operation_id TEXT, content TEXT)")
    runtime = CanonicalRememberRuntime(
        db=db,
        profile_id="default",
        writer=writer,
        journal_path=tmp_path / "admission_journal.db",
    )
    return runtime, db


def _remember(runtime):
    from superlocalmemory.storage.admission_journal import Actor, RememberRequest

    actor = Actor("actor", frozenset({"default"}), frozenset({"personal"}))
    request = RememberRequest(
        content="the quarterly board meeting is on the fifteenth of March",
        profile_id="default",
        source_type="http",
        idempotency_key="spine-int-1",
        trusted_actor_id="actor",
    )
    return runtime.remember(request, actor, deadline_ms=1_500).payload


def test_admission_records_obligations_atomically(tmp_path) -> None:
    runtime, db = _build_runtime(tmp_path, with_m033=True)
    runtime.start()
    try:
        receipt = _remember(runtime)
    finally:
        runtime.stop()

    operation_id = receipt["operation_id"]
    receipts = db.execute(
        "SELECT operation_id FROM write_commits WHERE operation_id = ?",
        (operation_id,),
    )
    assert receipts, "canonical receipt must be durable"

    rows = db.execute(
        "SELECT owner, kind, state FROM projection_obligations "
        "WHERE operation_id = ?",
        (operation_id,),
    )
    owners = {dict(r)["owner"] for r in rows}
    assert owners == {"bm25", "temporal", "vector"}
    assert all(dict(r)["state"] == "pending" for r in rows)
    assert all(dict(r)["kind"] == "apply" for r in rows)


def test_admission_failopen_without_ledger_table(tmp_path) -> None:
    runtime, db = _build_runtime(tmp_path, with_m033=False)
    runtime.start()
    try:
        receipt = _remember(runtime)
    finally:
        runtime.stop()

    assert receipt["status"] == "queryable"
    assert receipt["fact_ids"] == ["fact-1", "fact-2"]
    tables = db.execute(
        "SELECT name FROM sqlite_master WHERE name = 'projection_obligations'"
    )
    assert not tables


@pytest.fixture
def stored_engine(engine_with_mock_deps):
    from superlocalmemory.core.engine_ingestion import (
        canonical_store,
        local_trusted_actor_id,
    )

    engine = engine_with_mock_deps
    operation = canonical_store(
        engine,
        "Priya joined the platform team in Berlin on 2025-01-10 as staff engineer",
        source_type="python-api",
        trusted_actor_id=local_trusted_actor_id("python-api"),
        require_complete=True,
        return_receipt=True,
    )
    return engine, operation


def _manifest_row(engine, operation_id):
    rows = engine._db.execute(
        "SELECT state, all_met, obligation_count, manifest_hash "
        "FROM completion_manifests WHERE operation_id = ?",
        (operation_id,),
    )
    return dict(rows[0]) if rows else None


def test_reconcile_produces_manifest(stored_engine) -> None:
    from superlocalmemory.server.unified_daemon import (
        _reconcile_projection_manifest,
    )

    engine, operation = stored_engine
    fact_ids = list(operation.final_fact_ids)
    assert fact_ids

    _reconcile_projection_manifest(
        engine, operation.operation_id, engine._profile_id, fact_ids,
    )

    manifest = _manifest_row(engine, operation.operation_id)
    assert manifest is not None
    assert manifest["state"] == ManifestState.COMPLETE.value, manifest["state"]
    assert manifest["all_met"] == 1
    assert manifest["obligation_count"] == 3
    assert len(manifest["manifest_hash"]) == 64


def test_deleted_projection_is_self_healed(stored_engine) -> None:
    from superlocalmemory.server.unified_daemon import (
        _reconcile_projection_manifest,
    )

    engine, operation = stored_engine
    fact_ids = list(operation.final_fact_ids)
    for fact_id in fact_ids:
        engine._db.delete_bm25_tokens_for_fact(fact_id)
    before = engine._db.execute("SELECT COUNT(*) c FROM bm25_tokens")
    assert dict(before[0])["c"] == 0

    _reconcile_projection_manifest(
        engine, operation.operation_id, engine._profile_id, fact_ids,
    )

    manifest = _manifest_row(engine, operation.operation_id)
    assert manifest is not None
    assert manifest["state"] == ManifestState.COMPLETE.value, manifest["state"]
    after = engine._db.execute("SELECT COUNT(*) c FROM bm25_tokens")
    assert dict(after[0])["c"] >= 1
    rows = engine._db.execute(
        "SELECT state FROM projection_obligations "
        "WHERE operation_id = ? AND owner = 'bm25'",
        (operation.operation_id,),
    )
    assert dict(rows[0])["state"] == "verified"


def test_redrive_reconciles_orphaned_obligations(stored_engine) -> None:
    from superlocalmemory.core.transactions import ObligationKind, OperationContext
    from superlocalmemory.core.transactions.concrete_owners import (
        REQUIRED_ADMISSION_OWNERS,
        build_transaction_service,
    )
    from superlocalmemory.server.unified_daemon import (
        _reconcile_pending_projections,
    )

    engine, operation = stored_engine
    context = OperationContext(
        operation_id=operation.operation_id,
        profile_id=engine._profile_id,
        subject_id=operation.operation_id,
        fact_ids=tuple(operation.final_fact_ids),
    )
    service = build_transaction_service(engine)
    with engine._db.raw_connection() as conn:
        service.record(
            conn, context, owners=REQUIRED_ADMISSION_OWNERS,
            kind=ObligationKind.APPLY,
        )
    assert _manifest_row(engine, operation.operation_id) is None

    reconciled = _reconcile_pending_projections(engine, force=True)
    assert reconciled >= 1
    manifest = _manifest_row(engine, operation.operation_id)
    assert manifest is not None
    assert manifest["state"] == ManifestState.COMPLETE.value, manifest["state"]


def test_manifest_is_reverifiable(stored_engine) -> None:
    from superlocalmemory.core.transactions import Reconciler
    from superlocalmemory.server.unified_daemon import (
        _reconcile_projection_manifest,
    )

    engine, operation = stored_engine
    fact_ids = list(operation.final_fact_ids)
    _reconcile_projection_manifest(
        engine, operation.operation_id, engine._profile_id, fact_ids,
    )
    reconciler = Reconciler()
    with engine._db.raw_connection() as conn:
        assert reconciler.verify_manifest(conn, operation.operation_id) is True
        conn.execute(
            "UPDATE completion_manifests SET state = 'FAILED' "
            "WHERE operation_id = ?",
            (operation.operation_id,),
        )
        assert reconciler.verify_manifest(conn, operation.operation_id) is False
