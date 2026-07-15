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


def test_checkpoint_versions_summaries_edges_scenes_and_lexical_index(
    tmp_path: Path,
) -> None:
    db = _db(tmp_path / "derived-lineage.db")

    def write(request: IngestionRequest, _operation_id: str) -> list[str]:
        db.execute(
            "INSERT INTO memories (memory_id,profile_id,content) VALUES ('m1',?,?)",
            (request.profile_id, request.content),
        )
        db.execute(
            "INSERT INTO atomic_facts "
            "(fact_id,memory_id,profile_id,content,fact_type) "
            "VALUES ('f1','m1',?,'Alpha uses SQLite','semantic')",
            (request.profile_id,),
        )
        return ["f1"]

    def materialize(operation):
        db.execute(
            "INSERT INTO canonical_entities "
            "(entity_id,profile_id,canonical_name) VALUES ('e1',?,'Alpha')",
            (operation.profile_id,),
        )
        db.execute(
            "INSERT INTO entity_profiles "
            "(profile_entry_id,entity_id,profile_id,knowledge_summary,fact_ids_json) "
            "VALUES ('ep1','e1',?,'Alpha database profile','[\"f1\"]')",
            (operation.profile_id,),
        )
        db.execute(
            "INSERT INTO memory_scenes "
            "(scene_id,profile_id,theme,fact_ids_json) "
            "VALUES ('scene1',?,'database','[\"f1\"]')",
            (operation.profile_id,),
        )
        db.execute(
            "INSERT INTO graph_edges "
            "(edge_id,profile_id,source_id,target_id,edge_type) "
            "VALUES ('edge1',?,'f1','e1','entity')",
            (operation.profile_id,),
        )
        db.execute(
            "INSERT INTO bm25_tokens (fact_id,profile_id,tokens) "
            "VALUES ('f1',?,'[\"alpha\",\"uses\",\"sqlite\"]')",
            (operation.profile_id,),
        )
        return operation.queryable_fact_ids

    command = IngestionCommand(
        IngestionOperationRepository(db),
        write_queryable=write,
        materialize=materialize,
        derivation_version="test-derived-v2",
    )
    receipt = command.submit(IngestionRequest(
        content="Alpha uses SQLite", profile_id="default", source_type="test",
        idempotency_key="derived-1",
    ))
    command.materialize(receipt.operation_id)

    rows = {
        (row["object_type"], row["object_id"]): dict(row)
        for row in db.execute("SELECT * FROM derivation_lineage")
    }
    assert rows[("entity_summary", "ep1")]["source_status"] == "derived_from_facts"
    assert rows[("memory_scene", "scene1")]["source_status"] == "derived_from_facts"
    assert rows[("graph_edge", "edge1")]["source_status"] == "derived_from_facts"
    index = rows[("index_bm25", "f1")]
    assert index["source_status"] == "exact"
    assert index["source_start"] == 0
    assert {row["derivation_version"] for row in rows.values()} == {"test-derived-v2"}
