# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Regression contracts for issue #77's phantom enriching operations."""

from __future__ import annotations

import time

import pytest

from superlocalmemory.core.ingestion_command import (
    _MAX_AUTOMATIC_MATERIALIZATION_ATTEMPTS,
    IngestionOperationRepository,
    IngestionRequest,
    IngestionState,
)
from superlocalmemory.storage import schema
from superlocalmemory.storage.database import DatabaseManager
from superlocalmemory.storage.migrations import (
    M018_ingestion_operations,
    M031_dead_letter_operations,
)


@pytest.fixture
def repository(tmp_path) -> IngestionOperationRepository:
    db = DatabaseManager(tmp_path / "memory.db")
    db.initialize(schema)
    with db.raw_connection() as conn:
        M018_ingestion_operations.apply(conn)
        M031_dead_letter_operations.apply(conn)
    return IngestionOperationRepository(db)


def _enriching_operation(
    repository: IngestionOperationRepository,
    *,
    key: str,
    lease_expires_at: float,
    attempts: int,
) -> str:
    request = IngestionRequest(
        content=f"Queryable evidence for {key}.",
        profile_id="default",
        source_type="mcp",
        idempotency_key=key,
    )
    operation = repository.create(request)
    repository.db.execute(
        "UPDATE ingestion_operations "
        "SET state='enriching', queryable_fact_ids_json='[\"fact-visible\"]', "
        "lease_owner='dead-worker', lease_expires_at=?, attempt_count=?, "
        "last_error='worker interrupted' WHERE operation_id=?",
        (lease_expires_at, attempts, operation.operation_id),
    )
    return operation.operation_id


def test_386_reaper_terminalizes_expired_operation_at_attempt_cap(repository) -> None:
    now = time.time()
    operation_id = _enriching_operation(
        repository,
        key="386:stuck",
        lease_expires_at=now - 60,
        attempts=_MAX_AUTOMATIC_MATERIALIZATION_ATTEMPTS,
    )

    assert repository.reap_stuck_enriching(now=now) == [operation_id]

    operation = repository.get(operation_id)
    assert operation.state is IngestionState.FAILED
    assert operation.lease_owner == ""
    assert operation.lease_expires_at == 0
    assert operation.next_retry_at > now + 60 * 60 * 24 * 365
    assert operation.queryable_fact_ids == ("fact-visible",)
    dead_letters = repository.db.execute(
        "SELECT original_op_id, attempt_count, error "
        "FROM dead_letter_operations WHERE original_op_id=?",
        (operation_id,),
    )
    assert len(dead_letters) == 1
    assert dead_letters[0]["attempt_count"] == _MAX_AUTOMATIC_MATERIALIZATION_ATTEMPTS
    assert dead_letters[0]["error"] == "worker interrupted"

    assert repository.reap_stuck_enriching(now=now) == []
    assert repository.db.execute(
        "SELECT COUNT(*) AS count FROM dead_letter_operations "
        "WHERE original_op_id=?",
        (operation_id,),
    )[0]["count"] == 1


def test_386_real_daemon_materializer_pass_invokes_reaper_and_preserves_fact(
    engine_with_mock_deps,
) -> None:
    """Issue #77 is guarded at the loop wiring, not only repository level."""
    from superlocalmemory.server.unified_daemon import _materialize_ingestion_one_pass
    from superlocalmemory.storage.models import AtomicFact, MemoryRecord

    engine = engine_with_mock_deps
    with engine._db.raw_connection() as conn:
        M018_ingestion_operations.apply(conn)
        M031_dead_letter_operations.apply(conn)
    repository = IngestionOperationRepository(engine._db)
    memory_id = "reaper-loop-memory"
    fact_id = "reaper-loop-visible-fact"
    engine._db.store_memory(MemoryRecord(
        memory_id=memory_id,
        profile_id=engine._profile_id,
        content="The queryable fact survives a dead materializer lease.",
    ))
    engine._db.store_fact(AtomicFact(
        fact_id=fact_id,
        memory_id=memory_id,
        profile_id=engine._profile_id,
        content="The queryable fact survives a dead materializer lease.",
    ))
    operation = repository.create(IngestionRequest(
        content="The queryable fact survives a dead materializer lease.",
        profile_id=engine._profile_id,
        source_type="mcp",
        idempotency_key="386:real-materializer-pass-reaper",
    ))
    engine._db.execute(
        "UPDATE ingestion_operations SET state='enriching', "
        "queryable_fact_ids_json=?, lease_owner='dead-daemon', "
        "lease_expires_at=?, attempt_count=?, last_error='daemon terminated' "
        "WHERE operation_id=?",
        (
            f'["{fact_id}"]',
            time.time() - 60,
            _MAX_AUTOMATIC_MATERIALIZATION_ATTEMPTS,
            operation.operation_id,
        ),
    )

    assert _materialize_ingestion_one_pass(engine, limit=1) == (0, 0)
    reaped = repository.get(operation.operation_id)
    assert reaped.state is IngestionState.FAILED
    assert reaped.queryable_fact_ids == (fact_id,)
    assert reaped.next_retry_at > time.time() + 60 * 60 * 24 * 365
    assert engine._db.get_fact(fact_id, engine._profile_id) is not None
    dead_letters = engine._db.execute(
        "SELECT attempt_count FROM dead_letter_operations WHERE original_op_id=?",
        (operation.operation_id,),
    )
    assert len(dead_letters) == 1
    assert (
        dead_letters[0]["attempt_count"]
        == _MAX_AUTOMATIC_MATERIALIZATION_ATTEMPTS
    )


def test_386_reaper_spares_live_and_below_cap_operations(repository) -> None:
    now = time.time()
    live = _enriching_operation(
        repository,
        key="386:live",
        lease_expires_at=now + 60,
        attempts=_MAX_AUTOMATIC_MATERIALIZATION_ATTEMPTS,
    )
    retryable = _enriching_operation(
        repository,
        key="386:retryable",
        lease_expires_at=now - 60,
        attempts=_MAX_AUTOMATIC_MATERIALIZATION_ATTEMPTS - 1,
    )

    assert repository.reap_stuck_enriching(now=now) == []
    assert repository.get(live).state is IngestionState.ENRICHING
    assert repository.get(retryable).state is IngestionState.ENRICHING


def test_386_reaper_terminalizes_when_dead_letter_migration_is_absent(tmp_path) -> None:
    db = DatabaseManager(tmp_path / "memory.db")
    db.initialize(schema)
    with db.raw_connection() as conn:
        M018_ingestion_operations.apply(conn)
    repository = IngestionOperationRepository(db)
    now = time.time()
    operation_id = _enriching_operation(
        repository,
        key="386:no-dlq-table",
        lease_expires_at=now - 60,
        attempts=_MAX_AUTOMATIC_MATERIALIZATION_ATTEMPTS,
    )

    assert repository.reap_stuck_enriching(now=now) == [operation_id]
    assert repository.get(operation_id).state is IngestionState.FAILED
