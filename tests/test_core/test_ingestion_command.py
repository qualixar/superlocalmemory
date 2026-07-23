# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later

"""Canonical ingestion command contracts for V3.7 Wave 3."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pytest

from superlocalmemory.core.ingestion_command import (
    IdempotencyConflict,
    IngestionCommand,
    IngestionOperationRepository,
    IngestionRequest,
    IngestionState,
    InvalidStateTransition,
    LeaseLost,
    MaterializationResult,
    OperationInProgress,
)
from superlocalmemory.storage import schema
from superlocalmemory.storage.database import DatabaseManager
from superlocalmemory.storage.migrations import M018_ingestion_operations


@pytest.fixture
def db(tmp_path):
    manager = DatabaseManager(tmp_path / "memory.db")
    manager.initialize(schema)
    with manager.raw_connection() as conn:
        M018_ingestion_operations.apply(conn)
    return manager


@pytest.fixture
def ingestion_request() -> IngestionRequest:
    return IngestionRequest(
        content="Alice leads the reliability program.",
        profile_id="work",
        source_type="mcp",
        idempotency_key="session-7:turn-12",
        metadata={"source_agent_id": "caller-claimed"},
        scope="shared",
        shared_with=("reviewer",),
        trusted_actor_id="capability:agent-42",
        session_id="session-7",
        session_date="2026-07-14",
        speaker="Varun",
        role="user",
    )


def test_submit_persists_immutable_raw_evidence_and_queryable_receipt(
    db, ingestion_request,
) -> None:
    request = ingestion_request
    writes: list[str] = []

    def write_queryable(req: IngestionRequest, operation_id: str) -> list[str]:
        writes.append(operation_id)
        return ["fact-fast-1"]

    command = IngestionCommand(
        IngestionOperationRepository(db),
        write_queryable=write_queryable,
        materialize=lambda *_: ["fact-final-1"],
    )

    receipt = command.submit(request)
    stored = command.repository.get(receipt.operation_id)

    assert receipt.state is IngestionState.QUERYABLE
    assert receipt.fact_ids == ("fact-fast-1",)
    assert stored.raw_content == request.content
    assert stored.source_hash == request.source_hash
    assert stored.trusted_actor_id == "capability:agent-42"
    assert stored.metadata == {"source_agent_id": "caller-claimed"}
    assert stored.session_date == "2026-07-14"
    assert stored.speaker == "Varun"
    assert stored.role == "user"
    assert writes == [receipt.operation_id]


def test_same_idempotency_key_reuses_receipt_without_duplicate_write(
    db, ingestion_request,
) -> None:
    request = ingestion_request
    writes = 0

    def write_queryable(*_):
        nonlocal writes
        writes += 1
        return ["fact-fast-1"]

    command = IngestionCommand(
        IngestionOperationRepository(db),
        write_queryable=write_queryable,
        materialize=lambda *_: ["fact-final-1"],
    )

    first = command.submit(request)
    second = command.submit(request)

    assert second.operation_id == first.operation_id
    assert second.fact_ids == first.fact_ids
    assert writes == 1


def test_same_idempotency_key_with_changed_evidence_fails_closed(
    db, ingestion_request,
) -> None:
    request = ingestion_request
    command = IngestionCommand(
        IngestionOperationRepository(db),
        write_queryable=lambda *_: ["fact-fast-1"],
        materialize=lambda *_: ["fact-final-1"],
    )
    command.submit(request)

    with pytest.raises(IdempotencyConflict):
        command.submit(replace(request, content="Conflicting evidence"))


def test_queryable_projection_and_raw_operation_roll_back_together(
    db, ingestion_request,
) -> None:
    request = ingestion_request
    db.execute("CREATE TABLE projection_probe (value TEXT NOT NULL)")

    def broken_writer(*_) -> list[str]:
        db.execute("INSERT INTO projection_probe(value) VALUES ('partial')")
        raise RuntimeError("projection failed")

    command = IngestionCommand(
        IngestionOperationRepository(db),
        write_queryable=broken_writer,
        materialize=lambda *_: [],
    )

    with pytest.raises(RuntimeError, match="projection failed"):
        command.submit(request)

    assert db.execute("SELECT * FROM projection_probe") == []
    assert command.repository.list_operations() == []


def test_empty_queryable_projection_is_not_reported_as_queryable(
    db, ingestion_request,
) -> None:
    command = IngestionCommand(
        IngestionOperationRepository(db),
        write_queryable=lambda *_: [],
        materialize=lambda *_: [],
    )

    with pytest.raises(RuntimeError, match="no queryable facts"):
        command.submit(ingestion_request)

    assert command.repository.list_operations() == []


def test_materialization_records_complete_state_and_derivation(
    db, ingestion_request,
) -> None:
    request = ingestion_request
    command = IngestionCommand(
        IngestionOperationRepository(db),
        write_queryable=lambda *_: ["fact-fast-1"],
        materialize=lambda operation: ["fact-fast-1", "fact-derived-2"],
        derivation_version="v3.7-ingestion-1",
    )
    receipt = command.submit(request)

    completed = command.materialize(receipt.operation_id)

    assert completed.state is IngestionState.COMPLETE
    assert completed.queryable_fact_ids == ("fact-fast-1",)
    assert completed.final_fact_ids == ("fact-fast-1", "fact-derived-2")
    assert completed.derivation_version == "v3.7-ingestion-1"
    assert completed.attempt_count == 1


def test_failure_is_inspectable_and_retry_does_not_repeat_queryable_write(
    db, ingestion_request,
) -> None:
    request = ingestion_request
    queryable_writes = 0
    attempts = 0

    def write_queryable(*_):
        nonlocal queryable_writes
        queryable_writes += 1
        return ["fact-fast-1"]

    def materialize(_operation):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("temporary model failure")
        return ["fact-fast-1"]

    command = IngestionCommand(
        IngestionOperationRepository(db),
        write_queryable=write_queryable,
        materialize=materialize,
    )
    receipt = command.submit(request)

    failed = command.materialize(receipt.operation_id)
    assert failed.state is IngestionState.FAILED
    assert failed.last_error == "temporary model failure"
    assert failed.raw_content == request.content

    completed = command.retry(receipt.operation_id)
    assert completed.state is IngestionState.COMPLETE
    assert completed.attempt_count == 2
    assert queryable_writes == 1


def test_partial_materialization_checkpoint_survives_with_failed_state(
    db, ingestion_request,
) -> None:
    db.execute("CREATE TABLE materialization_probe (value TEXT NOT NULL)")

    def broken_materializer(_operation):
        db.execute("INSERT INTO materialization_probe(value) VALUES ('partial')")
        raise RuntimeError("derivation failed")

    command = IngestionCommand(
        IngestionOperationRepository(db),
        write_queryable=lambda *_: ["fact-fast-1"],
        materialize=broken_materializer,
    )
    receipt = command.submit(ingestion_request)

    failed = command.materialize(receipt.operation_id)

    assert failed.state is IngestionState.FAILED
    assert [
        row["value"] for row in db.execute("SELECT * FROM materialization_probe")
    ] == ["partial"]


def test_materialization_does_not_hold_writer_lock_during_model_work(
    db, ingestion_request,
) -> None:
    db.execute("CREATE TABLE materialization_probe (value TEXT NOT NULL)")
    entered_model_work = threading.Event()
    release_model_work = threading.Event()

    def slow_materializer(_operation):
        db.execute("INSERT INTO materialization_probe(value) VALUES ('checkpoint')")
        entered_model_work.set()
        assert release_model_work.wait(timeout=5)
        return ["fact-fast-1"]

    command = IngestionCommand(
        IngestionOperationRepository(db),
        write_queryable=lambda *_: ["fact-fast-1"],
        materialize=slow_materializer,
    )
    receipt = command.submit(ingestion_request)

    with ThreadPoolExecutor(max_workers=2) as executor:
        materialization = executor.submit(command.materialize, receipt.operation_id)
        assert entered_model_work.wait(timeout=5)
        competing_write = executor.submit(
            db.execute,
            "INSERT INTO materialization_probe(value) VALUES ('interactive')",
        )
        try:
            competing_write.result(timeout=1)
        finally:
            release_model_work.set()
        assert materialization.result(timeout=5).state is IngestionState.COMPLETE

    assert {
        row["value"] for row in db.execute("SELECT value FROM materialization_probe")
    } == {"checkpoint", "interactive"}


def test_long_materializer_renews_owner_bound_lease(
    db, ingestion_request, monkeypatch,
) -> None:
    renewed = threading.Event()
    command = IngestionCommand(
        IngestionOperationRepository(db),
        write_queryable=lambda *_: ["fact-fast-1"],
        materialize=lambda *_: (
            ["fact-fast-1"] if renewed.wait(timeout=2) else []
        ),
        lease_seconds=1,
    )
    actual_renew = command.repository.renew_enriching_lease

    def record_renewal(*args, **kwargs):
        result = actual_renew(*args, **kwargs)
        renewed.set()
        return result

    monkeypatch.setattr(
        command.repository, "renew_enriching_lease", record_renewal,
    )
    receipt = command.submit(ingestion_request)

    completed = command.materialize(receipt.operation_id)

    assert completed.state is IngestionState.COMPLETE
    assert renewed.is_set()


def test_lost_materialization_lease_aborts_before_checkpoint(
    db, ingestion_request, monkeypatch,
) -> None:
    renewal_attempted = threading.Event()

    def materialize(_operation):
        assert renewal_attempted.wait(timeout=2)
        return ["fact-fast-1"]

    command = IngestionCommand(
        IngestionOperationRepository(db),
        write_queryable=lambda *_: ["fact-fast-1"],
        materialize=materialize,
        lease_seconds=1,
    )

    def lose_lease(*_args, **_kwargs):
        renewal_attempted.set()
        return False

    monkeypatch.setattr(
        command.repository, "renew_enriching_lease", lose_lease,
    )
    receipt = command.submit(ingestion_request)

    with pytest.raises(LeaseLost, match=receipt.operation_id):
        command.materialize(receipt.operation_id)

    stranded = command.repository.get(receipt.operation_id)
    assert stranded.final_fact_ids == ()
    assert stranded.state is IngestionState.ENRICHING


def test_incomplete_declared_derivation_cannot_be_marked_complete(
    db, ingestion_request,
) -> None:
    command = IngestionCommand(
        IngestionOperationRepository(db),
        write_queryable=lambda *_: ["fact-fast-1"],
        materialize=lambda *_: MaterializationResult(
            fact_ids=("fact-fast-1",),
            derivation_state={
                "relational": True,
                "provenance": False,
            },
        ),
    )
    receipt = command.submit(ingestion_request)

    failed = command.materialize(receipt.operation_id)

    assert failed.state is IngestionState.FAILED
    assert "provenance" in failed.last_error
    assert failed.final_fact_ids == ("fact-fast-1",)
    assert failed.derivation_state == {
        "relational": True,
        "provenance": False,
    }


def test_external_projection_retry_resumes_checkpoint_without_rederiving(
    db, ingestion_request,
) -> None:
    materializer_calls = 0
    projector_calls = 0

    def materialize(_operation):
        nonlocal materializer_calls
        materializer_calls += 1
        return MaterializationResult(
            fact_ids=("fact-fast-1", "fact-derived-2"),
            derivation_state={
                "relational": True,
                "provenance": True,
            },
        )

    def project(_operation):
        nonlocal projector_calls
        projector_calls += 1
        if projector_calls == 1:
            raise RuntimeError("vector backend busy")
        return {
            "ann": True,
            "vector": True,
            "bm25": True,
        }

    command = IngestionCommand(
        IngestionOperationRepository(db),
        write_queryable=lambda *_: ["fact-fast-1"],
        materialize=materialize,
        project=project,
    )
    receipt = command.submit(ingestion_request)

    failed = command.materialize(receipt.operation_id)

    assert failed.state is IngestionState.FAILED
    assert failed.final_fact_ids == ("fact-fast-1", "fact-derived-2")
    assert failed.derivation_state == {
        "relational": True,
        "provenance": True,
    }
    assert failed.last_error == "vector backend busy"

    completed = command.retry(receipt.operation_id)

    assert completed.state is IngestionState.COMPLETE
    assert completed.derivation_state == {
        "relational": True,
        "provenance": True,
        "ann": True,
        "vector": True,
        "bm25": True,
    }
    assert materializer_calls == 1
    assert projector_calls == 2


def test_expired_lease_recovers_crash_before_derivation_checkpoint(
    db, ingestion_request,
) -> None:
    crashed = IngestionCommand(
        IngestionOperationRepository(db),
        write_queryable=lambda *_: ["fact-fast-1"],
        materialize=lambda *_: (_ for _ in ()).throw(SystemExit("worker died")),
        lease_seconds=60,
    )
    receipt = crashed.submit(ingestion_request)

    with pytest.raises(SystemExit, match="worker died"):
        crashed.materialize(receipt.operation_id)

    stranded = crashed.repository.get(receipt.operation_id)
    assert stranded.state is IngestionState.ENRICHING
    assert stranded.final_fact_ids == ()
    assert stranded.lease_owner
    with pytest.raises(OperationInProgress):
        IngestionCommand(
            IngestionOperationRepository(db),
            write_queryable=lambda *_: ["unused"],
            materialize=lambda *_: ["fact-fast-1"],
        ).materialize(receipt.operation_id)

    db.execute(
        "UPDATE ingestion_operations SET lease_expires_at=0 "
        "WHERE operation_id=?",
        (receipt.operation_id,),
    )
    restarted = IngestionCommand(
        IngestionOperationRepository(db),
        write_queryable=lambda *_: ["unused"],
        materialize=lambda *_: ["fact-fast-1"],
    )

    completed = restarted.materialize(receipt.operation_id)

    assert completed.state is IngestionState.COMPLETE
    assert completed.attempt_count == 2
    assert completed.lease_owner == ""
    assert completed.lease_expires_at == 0


def test_concurrent_materialize_same_operation_coalesces(
    db, ingestion_request,
) -> None:
    entered = threading.Event()
    release = threading.Event()
    calls = 0

    def materialize(_operation):
        nonlocal calls
        calls += 1
        entered.set()
        assert release.wait(timeout=5)
        return ["fact-fast-1"]

    command = IngestionCommand(
        IngestionOperationRepository(db),
        write_queryable=lambda *_: ["fact-fast-1"],
        materialize=materialize,
    )
    receipt = command.submit(ingestion_request)

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(command.materialize, receipt.operation_id)
        assert entered.wait(timeout=5)
        second = executor.submit(command.materialize, receipt.operation_id)
        release.set()
        results = [first.result(timeout=5), second.result(timeout=5)]

    assert [result.state for result in results] == [
        IngestionState.COMPLETE,
        IngestionState.COMPLETE,
    ]
    assert calls == 1
    assert results[0].operation_id == results[1].operation_id


def test_illegal_state_transition_is_rejected(db, ingestion_request) -> None:
    request = ingestion_request
    repository = IngestionOperationRepository(db)
    operation = repository.create(request)

    with pytest.raises(InvalidStateTransition):
        repository.transition(
            operation.operation_id,
            expected=IngestionState.RAW,
            target=IngestionState.COMPLETE,
            final_fact_ids=("fact-1",),
        )


def test_materializable_queue_respects_failed_retry_backoff(
    db, ingestion_request,
) -> None:
    command = IngestionCommand(
        IngestionOperationRepository(db),
        write_queryable=lambda *_: ["fact-fast-1"],
        materialize=lambda *_: (_ for _ in ()).throw(RuntimeError("retry me")),
    )
    queryable = command.submit(ingestion_request)
    second = command.submit(
        replace(ingestion_request, idempotency_key="second-operation")
    )
    failed = command.materialize(second.operation_id)

    queued = command.repository.list_materializable(limit=10)

    assert [item.operation_id for item in queued] == [queryable.operation_id]
    assert failed.next_retry_at > 0

    db.execute(
        "UPDATE ingestion_operations SET next_retry_at=0 WHERE operation_id=?",
        (failed.operation_id,),
    )
    due = command.repository.list_materializable(limit=10)
    assert [item.operation_id for item in due] == [
        queryable.operation_id,
        failed.operation_id,
    ]


def test_materializable_queue_stops_automatic_retry_after_budget(
    db, ingestion_request,
) -> None:
    command = IngestionCommand(
        IngestionOperationRepository(db),
        write_queryable=lambda *_: ["fact-fast-1"],
        materialize=lambda *_: (_ for _ in ()).throw(RuntimeError("deterministic failure")),
    )
    receipt = command.submit(ingestion_request)
    failed = command.materialize(receipt.operation_id)
    db.execute(
        "UPDATE ingestion_operations SET attempt_count=10, next_retry_at=0 "
        "WHERE operation_id=?",
        (failed.operation_id,),
    )

    assert command.repository.list_materializable(limit=10) == []

    # Circuit breaking automatic work must not destroy the operator escape
    # hatch: an explicit retry remains possible after configuration repair.
    command._materializer = lambda *_: ["fact-fast-1"]
    completed = command.retry(failed.operation_id)
    assert completed.state is IngestionState.COMPLETE


def test_materializable_queue_can_defer_fresh_queryable_receipts(
    db, ingestion_request,
) -> None:
    command = IngestionCommand(
        IngestionOperationRepository(db),
        write_queryable=lambda *_: ["fact-fast-1"],
        materialize=lambda *_: ["fact-final-1"],
    )
    queryable = command.submit(ingestion_request)

    assert command.repository.list_materializable(
        limit=10, min_queryable_age_seconds=1.0,
    ) == []

    db.execute(
        "UPDATE ingestion_operations SET "
        "created_at=strftime('%Y-%m-%dT%H:%M:%fZ', 'now', '-2 seconds') "
        "WHERE operation_id=?",
        (queryable.operation_id,),
    )
    due = command.repository.list_materializable(
        limit=10, min_queryable_age_seconds=1.0,
    )
    assert [item.operation_id for item in due] == [queryable.operation_id]
