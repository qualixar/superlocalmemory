# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Boot-sweep contracts for issue #131's wedged enrichment leases.

A killed daemon leaves rows stuck in ``enriching``. Recovery must never
depend on worker warmth: the sweep terminalizes exhausted-expired rows
and leaves under-attempt rows reclaimable, and the materializer pass
reaps even with a cold embedder.
"""
from __future__ import annotations

import time
from types import SimpleNamespace

import pytest

from superlocalmemory.core.ingestion_command import (
    _MAX_AUTOMATIC_MATERIALIZATION_ATTEMPTS,
    IngestionOperationRepository,
    IngestionRequest,
)
from superlocalmemory.server.unified_daemon import (
    _materialize_ingestion_one_pass,
    _reap_stuck_ingestion,
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


def _stuck_operation(
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


def _row(repository: IngestionOperationRepository, operation_id: str) -> dict:
    rows = repository.db.execute(
        "SELECT state, lease_owner FROM ingestion_operations WHERE operation_id=?",
        (operation_id,),
    )
    return dict(rows[0])


def test_boot_sweep_terminalizes_exhausted_expired_row(repository) -> None:
    now = time.time()
    operation_id = _stuck_operation(
        repository,
        key="131:exhausted",
        lease_expires_at=now - 60,
        attempts=_MAX_AUTOMATIC_MATERIALIZATION_ATTEMPTS,
    )

    assert _reap_stuck_ingestion(repository.db) == [operation_id]
    assert _row(repository, operation_id)["state"] == "failed"
    dead = repository.db.execute(
        "SELECT original_op_id FROM dead_letter_operations WHERE original_op_id=?",
        (operation_id,),
    )
    assert [dict(r)["original_op_id"] for r in dead] == [operation_id]


def test_boot_sweep_leaves_reclaimable_row_for_the_pass(repository) -> None:
    now = time.time()
    operation_id = _stuck_operation(
        repository,
        key="131:reclaimable",
        lease_expires_at=now - 60,
        attempts=3,
    )

    assert _reap_stuck_ingestion(repository.db) == []
    assert _row(repository, operation_id)["state"] == "enriching"
    due = [
        op.operation_id
        for op in repository.list_materializable(limit=50)
    ]
    assert operation_id in due


def test_materializer_pass_reaps_with_cold_embedder(repository) -> None:
    now = time.time()
    operation_id = _stuck_operation(
        repository,
        key="131:cold",
        lease_expires_at=now - 60,
        attempts=_MAX_AUTOMATIC_MATERIALIZATION_ATTEMPTS,
    )
    cold_engine = SimpleNamespace(
        _db=repository.db,
        _embedder=SimpleNamespace(is_warm=False),
    )

    assert _materialize_ingestion_one_pass(cold_engine) == (0, 0)
    assert _row(repository, operation_id)["state"] == "failed"


def test_boot_sweep_never_raises_without_ingestion_tables(tmp_path) -> None:
    db = DatabaseManager(tmp_path / "fresh.db")
    assert _reap_stuck_ingestion(db) == []


def test_double_reap_is_idempotent(repository) -> None:
    """Audit: the boot sweep racing the first materializer pass is benign —
    CAS terminalization means the second reap finds nothing to do."""
    now = time.time()
    operation_id = _stuck_operation(
        repository,
        key="131:double-reap",
        lease_expires_at=now - 60,
        attempts=_MAX_AUTOMATIC_MATERIALIZATION_ATTEMPTS,
    )
    assert _reap_stuck_ingestion(repository.db) == [operation_id]
    assert _reap_stuck_ingestion(repository.db) == []
