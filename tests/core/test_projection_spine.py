# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path

import pytest

from superlocalmemory.core.transactions import (
    ManifestState,
    MemoryTransactionService,
    ObligationKind,
    ObligationLedger,
    ObligationState,
    OperationContext,
    OwnerErasureProof,
    OwnerHealth,
    OwnerResult,
    Reconciler,
)
from superlocalmemory.storage.migrations import (
    M003_migration_log as m003,
)
from superlocalmemory.storage.migrations import (
    M033_projection_transactions as m033,
)
from superlocalmemory.storage.migrations import (
    M034_obligation_integrity as m034,
)


@pytest.fixture
def conn(tmp_path: Path) -> sqlite3.Connection:
    path = tmp_path / "memory.db"
    connection = sqlite3.connect(path, isolation_level=None)
    connection.executescript(m003.DDL)
    m033.apply(connection)
    m034.apply(connection)
    connection.executescript(
        "CREATE TABLE read_model ("
        "operation_id TEXT NOT NULL, owner TEXT NOT NULL, subject_id TEXT NOT NULL, "
        "PRIMARY KEY (operation_id, owner))"
    )
    return connection


class _ReadModelOwner:
    def __init__(self, name: str, connection: sqlite3.Connection) -> None:
        self._name = name
        self._conn = connection

    @property
    def name(self) -> str:
        return self._name

    def apply(self, context: OperationContext) -> OwnerResult:
        self._conn.execute(
            "INSERT OR REPLACE INTO read_model (operation_id, owner, subject_id) "
            "VALUES (?, ?, ?)",
            (context.operation_id, self._name, context.subject_id),
        )
        return OwnerResult(owner=self._name, ok=True, checksum=self._checksum(context))

    def verify(self, context: OperationContext) -> OwnerResult:
        row = self._conn.execute(
            "SELECT subject_id FROM read_model WHERE operation_id = ? AND owner = ?",
            (context.operation_id, self._name),
        ).fetchone()
        ok = row is not None and row[0] == context.subject_id
        return OwnerResult(
            owner=self._name,
            ok=ok,
            checksum=self._checksum(context) if ok else None,
            detail={} if ok else {"error": "projection row absent"},
        )

    def compensate(self, context: OperationContext) -> OwnerResult:
        self._conn.execute(
            "DELETE FROM read_model WHERE operation_id = ? AND owner = ?",
            (context.operation_id, self._name),
        )
        return OwnerResult(owner=self._name, ok=True)

    def erase(self, context: OperationContext) -> OwnerErasureProof:
        self._conn.execute(
            "DELETE FROM read_model WHERE operation_id = ? AND owner = ?",
            (context.operation_id, self._name),
        )
        remaining = self._conn.execute(
            "SELECT COUNT(*) FROM read_model WHERE operation_id = ? AND owner = ?",
            (context.operation_id, self._name),
        ).fetchone()[0]
        return OwnerErasureProof(
            owner=self._name,
            erased=remaining == 0,
            checksum=hashlib.sha256(b"erased").hexdigest(),
        )

    def health(self) -> OwnerHealth:
        return OwnerHealth(owner=self._name, healthy=True)

    def _checksum(self, context: OperationContext) -> str:
        return hashlib.sha256(
            f"{self._name}:{context.subject_id}".encode("utf-8")
        ).hexdigest()


class _ExplodingOwner:
    def __init__(self, name: str) -> None:
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def apply(self, context: OperationContext) -> OwnerResult:
        raise RuntimeError("projection backend unavailable")

    def verify(self, context: OperationContext) -> OwnerResult:
        return OwnerResult(owner=self._name, ok=False)

    def compensate(self, context: OperationContext) -> OwnerResult:
        return OwnerResult(owner=self._name, ok=True)

    def erase(self, context: OperationContext) -> OwnerErasureProof:
        return OwnerErasureProof(owner=self._name, erased=True)

    def health(self) -> OwnerHealth:
        return OwnerHealth(owner=self._name, healthy=False, detail="backend down")


def _ctx(operation_id: str = "op-1", subject: str = "fact-1") -> OperationContext:
    return OperationContext(
        operation_id=operation_id,
        profile_id="profile-a",
        subject_id=subject,
    )


def test_operation_context_requires_identifiers() -> None:
    with pytest.raises(ValueError):
        OperationContext(operation_id="", profile_id="p", subject_id="s")
    with pytest.raises(ValueError):
        OperationContext(operation_id="o", profile_id="", subject_id="s")
    with pytest.raises(ValueError):
        OperationContext(operation_id="o", profile_id="p", subject_id="")


def test_ledger_record_is_idempotent(conn: sqlite3.Connection) -> None:
    ledger = ObligationLedger()
    ctx = _ctx()
    ledger.record(conn, ctx, "vector", ObligationKind.APPLY)
    ledger.record(conn, ctx, "vector", ObligationKind.APPLY)
    obligations = ledger.fetch(conn, ctx.operation_id)
    assert len(obligations) == 1
    assert obligations[0].state is ObligationState.PENDING


def test_ledger_mark_transitions_state(conn: sqlite3.Connection) -> None:
    ledger = ObligationLedger()
    ctx = _ctx()
    ledger.record(conn, ctx, "vector", ObligationKind.APPLY)
    ledger.mark(
        conn, ctx.operation_id, "vector", ObligationKind.APPLY,
        ObligationState.VERIFIED, checksum="abc", bump_attempts=True,
    )
    obligation = ledger.fetch(conn, ctx.operation_id)[0]
    assert obligation.state is ObligationState.VERIFIED
    assert obligation.checksum == "abc"
    assert obligation.attempts == 1


def test_reconcile_complete_when_all_verified(conn: sqlite3.Connection) -> None:
    service = MemoryTransactionService({
        "vector": _ReadModelOwner("vector", conn),
        "bm25": _ReadModelOwner("bm25", conn),
    })
    ctx = _ctx()
    service.record(conn, ctx)
    manifest = service.run(conn, ctx)
    assert manifest.state is ManifestState.COMPLETE
    assert manifest.all_met is True
    assert manifest.obligation_count == 2
    assert len(manifest.manifest_hash) == 64


def test_injected_projection_failure_degrades_manifest(conn: sqlite3.Connection) -> None:
    service = MemoryTransactionService({
        "vector": _ReadModelOwner("vector", conn),
        "graph": _ExplodingOwner("graph"),
    })
    ctx = _ctx()
    service.record(conn, ctx)
    manifest = service.run(conn, ctx)
    assert manifest.state is ManifestState.DEGRADED
    assert manifest.all_met is False
    ledger = ObligationLedger()
    by_owner = {o.owner: o for o in ledger.fetch(conn, ctx.operation_id)}
    assert by_owner["vector"].state is ObligationState.VERIFIED
    assert by_owner["graph"].state is ObligationState.FAILED
    row = conn.execute(
        "SELECT subject_id FROM read_model WHERE owner = 'vector'"
    ).fetchone()
    assert row is not None and row[0] == ctx.subject_id


def test_reconcile_failed_when_canonical_not_committed(conn: sqlite3.Connection) -> None:
    service = MemoryTransactionService({"vector": _ReadModelOwner("vector", conn)})
    ctx = _ctx()
    service.record(conn, ctx)
    service.apply(conn, ctx)
    manifest = service.reconcile(
        conn, ctx.operation_id, ctx.profile_id, canonical_committed=False,
    )
    assert manifest.state is ManifestState.FAILED
    assert manifest.all_met is False


def test_apply_is_idempotent_replay(conn: sqlite3.Connection) -> None:
    service = MemoryTransactionService({"vector": _ReadModelOwner("vector", conn)})
    ctx = _ctx()
    service.record(conn, ctx)
    first = service.run(conn, ctx)
    second = service.run(conn, ctx)
    assert first.state is ManifestState.COMPLETE
    assert second.state is ManifestState.COMPLETE
    assert first.manifest_hash == second.manifest_hash
    ledger = ObligationLedger()
    obligation = ledger.fetch(conn, ctx.operation_id)[0]
    assert obligation.attempts == 1


def test_manifest_record_tampering_is_detected(conn: sqlite3.Connection) -> None:
    service = MemoryTransactionService({"vector": _ReadModelOwner("vector", conn)})
    ctx = _ctx()
    service.record(conn, ctx)
    service.run(conn, ctx)
    assert service.verify_manifest(conn, ctx.operation_id) is True
    conn.execute(
        "UPDATE completion_manifests SET profile_id = 'evil' WHERE operation_id = ?",
        (ctx.operation_id,),
    )
    assert service.verify_manifest(conn, ctx.operation_id) is False


def test_manifest_state_tampering_is_detected(conn: sqlite3.Connection) -> None:
    service = MemoryTransactionService({"vector": _ReadModelOwner("vector", conn)})
    ctx = _ctx()
    service.record(conn, ctx)
    service.run(conn, ctx)
    conn.execute(
        "UPDATE completion_manifests SET state = 'FAILED' WHERE operation_id = ?",
        (ctx.operation_id,),
    )
    assert service.verify_manifest(conn, ctx.operation_id) is False


def test_projection_drift_is_detected_on_reverify(conn: sqlite3.Connection) -> None:
    owner = _ReadModelOwner("vector", conn)
    service = MemoryTransactionService({"vector": owner})
    ctx = _ctx()
    service.record(conn, ctx)
    first = service.run(conn, ctx)
    assert first.state is ManifestState.COMPLETE
    conn.execute("DELETE FROM read_model WHERE owner = 'vector'")
    conn.execute("DELETE FROM read_model WHERE owner = 'vector'")

    class _NoHealOwner(_ReadModelOwner):
        def apply(self, context: OperationContext) -> OwnerResult:
            return OwnerResult(owner="vector", ok=False, detail={"error": "no heal"})

    service_no_heal = MemoryTransactionService({"vector": _NoHealOwner("vector", conn)})
    second = service_no_heal.run(conn, ctx)
    assert second.state is ManifestState.DEGRADED


def test_compensate_marks_compensated(conn: sqlite3.Connection) -> None:
    owner = _ReadModelOwner("vector", conn)
    service = MemoryTransactionService({"vector": owner})
    ctx = _ctx()
    service.record(conn, ctx)
    service.run(conn, ctx)
    service.compensate(conn, ctx, "vector")
    ledger = ObligationLedger()
    obligation = ledger.fetch(conn, ctx.operation_id)[0]
    assert obligation.state is ObligationState.COMPENSATED
    assert conn.execute("SELECT COUNT(*) FROM read_model").fetchone()[0] == 0


def test_erase_marks_erased(conn: sqlite3.Connection) -> None:
    owner = _ReadModelOwner("vector", conn)
    service = MemoryTransactionService({"vector": owner})
    ctx = _ctx()
    owner.apply(ctx)
    service.record(conn, ctx, kind=ObligationKind.ERASE)
    service.erase(conn, ctx)
    manifest = service.reconcile(conn, ctx.operation_id, ctx.profile_id)
    assert manifest.state is ManifestState.COMPLETE
    ledger = ObligationLedger()
    obligation = ledger.fetch(conn, ctx.operation_id)[0]
    assert obligation.state is ObligationState.ERASED
    assert conn.execute("SELECT COUNT(*) FROM read_model").fetchone()[0] == 0


def test_reconciler_standalone_degraded(conn: sqlite3.Connection) -> None:
    ledger = ObligationLedger()
    reconciler = Reconciler(ledger)
    ctx = _ctx()
    ledger.record(conn, ctx, "vector", ObligationKind.APPLY)
    ledger.mark(
        conn, ctx.operation_id, "vector", ObligationKind.APPLY,
        ObligationState.FAILED,
    )
    manifest = reconciler.reconcile(conn, ctx.operation_id, ctx.profile_id)
    assert manifest.state is ManifestState.DEGRADED
    fetched = reconciler.fetch_manifest(conn, ctx.operation_id)
    assert fetched is not None
    assert fetched.manifest_hash == manifest.manifest_hash


def test_verified_projection_drift_is_not_reblessed(conn: sqlite3.Connection) -> None:
    class _DriftingOwner:
        def __init__(self) -> None:
            self._name = "vector"
            self._checksum = "baseline"

        @property
        def name(self) -> str:
            return self._name

        def apply(self, context: OperationContext) -> OwnerResult:
            return OwnerResult(owner="vector", ok=False, detail={"error": "no heal"})

        def verify(self, context: OperationContext) -> OwnerResult:
            return OwnerResult(owner="vector", ok=True, checksum=self._checksum)

        def compensate(self, context: OperationContext) -> OwnerResult:
            return OwnerResult(owner="vector", ok=True)

        def erase(self, context: OperationContext) -> OwnerErasureProof:
            return OwnerErasureProof(owner="vector", erased=True)

        def health(self) -> OwnerHealth:
            return OwnerHealth(owner="vector", healthy=True)

    owner = _DriftingOwner()
    service = MemoryTransactionService({"vector": owner})
    ctx = _ctx()
    service.record(conn, ctx)
    first = service.run(conn, ctx)
    assert first.state is ManifestState.COMPLETE
    owner._checksum = "mutated"
    second = service.run(conn, ctx)
    assert second.state is ManifestState.DEGRADED


def test_conflicting_replay_raises(conn: sqlite3.Connection) -> None:
    from superlocalmemory.core.transactions import ObligationConflictError

    ledger = ObligationLedger()
    ctx = OperationContext(
        operation_id="op-x", profile_id="profile-a", subject_id="op-x",
        fact_ids=("f1",),
    )
    ledger.record(conn, ctx, "vector", ObligationKind.APPLY)
    conflicting = OperationContext(
        operation_id="op-x", profile_id="profile-b", subject_id="op-x",
        fact_ids=("f2",),
    )
    with pytest.raises(ObligationConflictError):
        ledger.record(conn, conflicting, "vector", ObligationKind.APPLY)


def test_dual_record_different_factset_is_idempotent(conn: sqlite3.Connection) -> None:
    ledger = ObligationLedger()
    queryable = OperationContext(
        operation_id="op-d", profile_id="p", subject_id="op-d",
        fact_ids=("f1", "f2"),
    )
    ledger.record(conn, queryable, "vector", ObligationKind.APPLY)
    final = OperationContext(
        operation_id="op-d", profile_id="p", subject_id="op-d",
        fact_ids=("f1",),
    )
    ledger.record(conn, final, "vector", ObligationKind.APPLY)
    obligations = ledger.fetch(conn, "op-d")
    assert len(obligations) == 1


def test_empty_obligations_reconcile_failed(conn: sqlite3.Connection) -> None:
    reconciler = Reconciler()
    manifest = reconciler.reconcile(conn, "ghost-op", "profile-a")
    assert manifest.state is ManifestState.FAILED
    assert manifest.all_met is False
    assert manifest.obligation_count == 0


def _bm25_fixture() -> sqlite3.Connection:
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    c.executescript(
        "CREATE TABLE bm25_tokens (fact_id TEXT, profile_id TEXT, tokens TEXT);"
        "CREATE TABLE atomic_facts "
        "(fact_id TEXT, profile_id TEXT, content TEXT, embedding TEXT);"
    )
    return c


def _bm25_insert(c: sqlite3.Connection, fact_id: str, content: str, tokens) -> None:
    import json

    c.execute(
        "INSERT INTO atomic_facts VALUES (?, 'p', ?, NULL)", (fact_id, content),
    )
    c.execute(
        "INSERT INTO bm25_tokens VALUES (?, 'p', ?)", (fact_id, json.dumps(tokens)),
    )


def test_bm25_content_integrity_rejects_corrupted_tokens() -> None:
    from superlocalmemory.core.transactions.concrete_owners import Bm25Owner
    from superlocalmemory.retrieval.bm25_channel import tokenize

    c = _bm25_fixture()
    _bm25_insert(c, "f1", "the quarterly report", ["corrupted", "tokens"])
    _bm25_insert(c, "f2", "annual budget review", tokenize("annual budget review"))
    owner = Bm25Owner(c)
    result = owner.verify(OperationContext(
        operation_id="op", profile_id="p", subject_id="op", fact_ids=("f1", "f2"),
    ))
    assert result.ok is False
    assert result.detail["missing"] == ["f1"]


def test_owner_checksum_binds_fact_scope() -> None:
    from superlocalmemory.core.transactions.concrete_owners import Bm25Owner
    from superlocalmemory.retrieval.bm25_channel import tokenize

    c = _bm25_fixture()
    _bm25_insert(c, "f1", "the quarterly report", tokenize("the quarterly report"))
    _bm25_insert(c, "f2", "annual budget review", tokenize("annual budget review"))
    owner = Bm25Owner(c)
    full = owner.verify(OperationContext(
        operation_id="op", profile_id="p", subject_id="op", fact_ids=("f1", "f2"),
    ))
    subset = owner.verify(OperationContext(
        operation_id="op", profile_id="p", subject_id="op", fact_ids=("f1",),
    ))
    assert full.ok is True and subset.ok is True
    assert full.checksum != subset.checksum


def test_unregistered_owner_obligation_fails(conn: sqlite3.Connection) -> None:
    service = MemoryTransactionService({"vector": _ReadModelOwner("vector", conn)})
    ctx = _ctx()
    ledger = ObligationLedger()
    ledger.record(conn, ctx, "vector", ObligationKind.APPLY)
    ledger.record(conn, ctx, "phantom", ObligationKind.APPLY)
    manifest = service.run(conn, ctx)
    assert manifest.state is ManifestState.DEGRADED
    by_owner = {o.owner: o for o in ledger.fetch(conn, ctx.operation_id)}
    assert by_owner["phantom"].state is ObligationState.FAILED
    assert by_owner["vector"].state is ObligationState.VERIFIED
