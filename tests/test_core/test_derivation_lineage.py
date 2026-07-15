"""Durable derivation lineage captures exact spans and honest unknowns."""

from __future__ import annotations

from pathlib import Path

from superlocalmemory.core.ingestion_command import (
    IngestionCommand,
    IngestionOperationRepository,
    IngestionRequest,
)
from superlocalmemory.storage import schema
from superlocalmemory.storage.database import DatabaseManager
from superlocalmemory.storage.migrations import (
    M018_ingestion_operations,
    M019_derivation_lineage,
)


def _db(path: Path) -> DatabaseManager:
    db = DatabaseManager(path)
    db.initialize(schema)
    with db.raw_connection() as conn:
        M018_ingestion_operations.apply(conn)
        M019_derivation_lineage.apply(conn)
    return db


def test_canonical_checkpoint_records_exact_and_unresolved_fact_spans(
    tmp_path: Path,
) -> None:
    db = _db(tmp_path / "lineage.db")

    def write_queryable(request: IngestionRequest, _operation_id: str) -> list[str]:
        db.execute(
            "INSERT INTO memories "
            "(memory_id,profile_id,content,session_id,speaker,role) "
            "VALUES ('m1',?,?, '', '', 'user')",
            (request.profile_id, request.content),
        )
        for fact_id, content in (
            ("f-exact", "Alpha uses SQLite"),
            ("f-paraphrase", "Beta relies on PostgreSQL"),
        ):
            db.execute(
                "INSERT INTO atomic_facts "
                "(fact_id,memory_id,profile_id,content,fact_type) "
                "VALUES (?,'m1',?,?,'semantic')",
                (fact_id, request.profile_id, content),
            )
        return ["f-exact", "f-paraphrase"]

    command = IngestionCommand(
        IngestionOperationRepository(db),
        write_queryable=write_queryable,
        materialize=lambda operation: operation.queryable_fact_ids,
        derivation_version="test-derivation-v7",
    )
    receipt = command.submit(
        IngestionRequest(
            content="Alpha uses SQLite. Beta uses Postgres.",
            profile_id="default",
            source_type="test",
            idempotency_key="lineage-1",
        )
    )
    completed = command.materialize(receipt.operation_id)
    assert completed.state.value == "complete"

    rows = {
        row["object_id"]: dict(row)
        for row in db.execute(
            "SELECT * FROM derivation_lineage WHERE object_type='fact'"
        )
    }
    exact = rows["f-exact"]
    assert exact["derivation_version"] == "test-derivation-v7"
    assert exact["source_status"] == "exact"
    assert exact["source_start"] == 0
    assert exact["source_end"] == len("Alpha uses SQLite")
    assert len(exact["source_text_sha256"]) == 64

    unresolved = rows["f-paraphrase"]
    assert unresolved["source_status"] == "unresolved"
    assert unresolved["source_start"] is None
    assert unresolved["source_end"] is None
    assert unresolved["source_text_sha256"] == ""
    assert unresolved["unresolved_reason"] == "no_exact_span_in_raw_source"


def test_lineage_migration_is_idempotent_and_runner_registered(tmp_path: Path) -> None:
    import sqlite3

    from superlocalmemory.storage.migration_runner import MIGRATIONS

    conn = sqlite3.connect(tmp_path / "migration.db")
    try:
        M019_derivation_lineage.apply(conn)
        M019_derivation_lineage.apply(conn)
        assert M019_derivation_lineage.verify(conn) is True
    finally:
        conn.close()
    names = [migration.name for migration in MIGRATIONS]
    assert M019_derivation_lineage.NAME in names
    assert names.index(M019_derivation_lineage.NAME) > names.index(
        M018_ingestion_operations.NAME
    )
