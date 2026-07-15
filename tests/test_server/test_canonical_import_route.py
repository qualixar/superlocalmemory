# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later

"""Import is a canonical, idempotent ingestion entrypoint."""

from __future__ import annotations

import json
from unittest.mock import patch

from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from superlocalmemory.core.ingestion_command import IngestionOperationRepository
from superlocalmemory.server.routes.data_io import router
from superlocalmemory.storage.migrations import M018_ingestion_operations


def _client(engine) -> TestClient:
    with engine._db.raw_connection() as conn:
        M018_ingestion_operations.apply(conn)
    app = FastAPI()
    app.state.engine = engine
    app.include_router(router)
    return TestClient(app)


def test_replaying_same_import_file_reuses_canonical_operations(
    engine_with_mock_deps,
) -> None:
    client = _client(engine_with_mock_deps)
    payload = json.dumps({
        "memories": [{
            "content": "Dana approved the July production recovery plan.",
            "session_id": "import-session",
            "project_name": "reliability",
            "category": "decision",
            "tags": "approved",
        }],
    }).encode()

    first = client.post(
        "/api/import",
        files={"file": ("memories.json", payload, "application/json")},
    )
    second = client.post(
        "/api/import",
        files={"file": ("memories.json", payload, "application/json")},
    )

    assert first.status_code == 200
    assert first.json()["imported_count"] == 1
    assert first.json()["skipped_count"] == 0
    assert second.status_code == 200
    assert second.json()["imported_count"] == 0
    assert second.json()["skipped_count"] == 1
    operations = IngestionOperationRepository(
        engine_with_mock_deps._db,
    ).list_operations()
    assert len(operations) == 1
    assert operations[0].source_type == "http-import"
    assert operations[0].session_id == "import-session"


def test_import_fails_closed_when_canonical_engine_is_unavailable(
    engine_with_mock_deps,
) -> None:
    client = _client(engine_with_mock_deps)
    payload = json.dumps([{"content": "must not bypass canonical ingestion"}]).encode()

    with patch(
        "superlocalmemory.server.routes.data_io.require_engine",
        side_effect=HTTPException(503, detail="engine unavailable"),
    ):
        response = client.post(
            "/api/import",
            files={"file": ("memories.json", payload, "application/json")},
        )

    assert response.status_code == 503
    facts = engine_with_mock_deps._db.get_all_facts(
        engine_with_mock_deps._profile_id,
    )
    assert facts == []
