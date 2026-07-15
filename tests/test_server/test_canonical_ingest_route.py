# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later

"""Adapter ingestion must share the durable canonical write contract."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI
from fastapi.testclient import TestClient

from superlocalmemory.core.ingestion_command import IngestionOperationRepository
from superlocalmemory.server.routes.ingest import router
from superlocalmemory.storage.migrations import M018_ingestion_operations


def _client(engine) -> TestClient:
    with engine._db.raw_connection() as conn:
        M018_ingestion_operations.apply(conn)
    app = FastAPI()
    app.state.engine = engine
    app.include_router(router)
    return TestClient(app)


def test_ingest_preserves_adapter_source_and_deduplicates_by_operation(
    engine_with_mock_deps,
) -> None:
    client = _client(engine_with_mock_deps)
    payload = {
        "content": "Gmail message says Dana approved the production recovery plan.",
        "source_type": "gmail",
        "dedup_key": "message-1",
        "metadata": {"mailbox": "reliability"},
    }

    first = client.post("/ingest", json=payload)
    second = client.post("/ingest", json=payload)

    assert first.status_code == 200
    assert first.json()["ingested"] is True
    assert second.status_code == 200
    assert second.json() == {
        "ingested": False,
        "reason": "already_ingested",
        "operation_id": first.json()["operation_id"],
        "fact_ids": first.json()["fact_ids"],
    }
    operations = IngestionOperationRepository(
        engine_with_mock_deps._db,
    ).list_operations()
    assert len(operations) == 1
    assert operations[0].source_type == "gmail"
    assert operations[0].idempotency_key == "message-1"
    assert operations[0].metadata == {"mailbox": "reliability"}


def test_concurrent_ingest_same_key_creates_one_operation_and_one_fact_set(
    engine_with_mock_deps,
) -> None:
    client = _client(engine_with_mock_deps)
    payload = {
        "content": (
            "Calendar event records Dana's approved production recovery "
            "review for the platform team."
        ),
        "source_type": "calendar",
        "dedup_key": "event-42",
        "metadata": {"calendar": "reliability"},
    }

    with ThreadPoolExecutor(max_workers=2) as executor:
        responses = list(executor.map(
            lambda _: client.post("/ingest", json=payload),
            range(2),
        ))

    assert [response.status_code for response in responses] == [200, 200]
    bodies = [response.json() for response in responses]
    assert sorted(body["ingested"] for body in bodies) == [False, True]
    assert len({body["operation_id"] for body in bodies}) == 1
    assert len({tuple(body["fact_ids"]) for body in bodies}) == 1
    operations = IngestionOperationRepository(
        engine_with_mock_deps._db,
    ).list_operations()
    assert len(operations) == 1
    final_ids = operations[0].final_fact_ids
    assert final_ids
    facts = engine_with_mock_deps._db.get_facts_by_ids(
        list(final_ids),
        engine_with_mock_deps._profile_id,
    )
    assert len(facts) == len(final_ids)
