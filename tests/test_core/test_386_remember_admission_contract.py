# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Remember admission contract: immediate visibility plus idempotency."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from superlocalmemory.core.ingestion_command import (
    IngestionCommand,
    IngestionOperationRepository,
    IngestionRequest,
)
from superlocalmemory.storage import schema
from superlocalmemory.storage.database import DatabaseManager
from superlocalmemory.storage.migrations import M018_ingestion_operations


def test_386_parallel_same_idempotency_key_is_one_immediately_queryable_admission(tmp_path) -> None:
    """Concurrent identical remembers return one receipt and one visible fact."""
    db = DatabaseManager(tmp_path / "memory.db")
    db.initialize(schema)
    with db.raw_connection() as conn:
        M018_ingestion_operations.apply(conn)
        conn.execute(
            "CREATE TABLE admission_visibility (fact_id TEXT PRIMARY KEY, content TEXT NOT NULL)"
        )

    writes: list[str] = []

    def write_queryable(request: IngestionRequest, operation_id: str) -> list[str]:
        fact_id = f"fact-{operation_id}"
        db.execute(
            "INSERT INTO admission_visibility(fact_id, content) VALUES (?, ?)",
            (fact_id, request.content),
        )
        writes.append(operation_id)
        return [fact_id]

    command = IngestionCommand(
        IngestionOperationRepository(db),
        write_queryable=write_queryable,
        materialize=lambda operation: list(operation.queryable_fact_ids),
    )
    request = IngestionRequest(
        content="The admission receipt must be immediately queryable.",
        profile_id="default",
        source_type="mcp",
        idempotency_key="386:concurrent-admission",
    )

    with ThreadPoolExecutor(max_workers=8) as pool:
        receipts = list(pool.map(lambda _: command.submit(request), range(24)))

    operation_ids = {receipt.operation_id for receipt in receipts}
    fact_ids = {receipt.fact_ids for receipt in receipts}
    assert len(operation_ids) == 1
    assert len(fact_ids) == 1
    assert len(writes) == 1
    assert db.execute("SELECT COUNT(*) AS count FROM admission_visibility")[0]["count"] == 1
    assert db.execute("SELECT content FROM admission_visibility")[0]["content"] == request.content
