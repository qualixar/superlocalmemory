# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""A store with a damaged value in every tolerated JSON column still reads.

``storage/json_columns.TOLERATED_COLUMNS`` lists the JSON text columns whose
BLOB values may hold bytes that are not UTF-8. One of those used to fail the
save that read it; the readers that decoded them strictly failed a recall, a
dashboard listing or an export the same way. This test damages every one of
those columns on a store the engine wrote, and reads it through the doors a
person uses: recall, the memory and cluster listings, a memory and a fact
detail, the full export and the GDPR export.
"""

from __future__ import annotations

import gzip
import json
import sqlite3
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from superlocalmemory.core.engine_ingestion import (
    build_engine_ingestion_command,
    canonical_store,
    local_trusted_actor_id,
)
from superlocalmemory.core.ingestion_command import IngestionState
from superlocalmemory.server.profile_runtime import bind_profile_runtime
from superlocalmemory.server.routes import compliance, data_io, helpers, memories
from superlocalmemory.server.unified_daemon import create_app
from superlocalmemory.storage.json_columns import TOLERATED_COLUMNS

NOT_TEXT = bytes.fromhex("61628163")
TEXTS = [
    "Alice met Bob in Tokyo to discuss the Postgres billing migration on Friday.",
    "Alice and Bob reviewed the Postgres billing schema in Tokyo again.",
    "Bob told Alice the Postgres billing cutover in Tokyo moves to Monday.",
]


@pytest.fixture()
def damaged_store(engine_with_mock_deps):
    engine = engine_with_mock_deps
    for text in TEXTS:
        receipt = canonical_store(
            engine, text, source_type="python-api",
            trusted_actor_id=local_trusted_actor_id("python-api"),
            metadata={"project": "demo"}, require_complete=False, return_receipt=True)
        done = build_engine_ingestion_command(engine).materialize(receipt.operation_id)
        assert done.state is IngestionState.COMPLETE, done.last_error
    for table, column in TOLERATED_COLUMNS:
        hit = engine._db.execute(f"UPDATE {table} SET {column} = ? RETURNING rowid", (NOT_TEXT,))
        assert hit, f"no {table} rows to damage for {column}"
    app = create_app()
    app.state.engine = engine
    app.state.config = engine._config
    bind_profile_runtime(app.state, engine, engine._config)

    def connect():
        return sqlite3.connect(f"file:{engine._db.db_path}?mode=ro", uri=True)

    with patch.object(helpers, "DB_PATH", engine._db.db_path), \
            patch.object(data_io, "get_db_connection", connect), \
            patch.object(data_io, "get_active_profile", lambda: "default"), \
            patch.object(compliance, "get_active_profile", lambda: "default"), \
            patch.object(memories, "get_active_profile", lambda: "default"):
        yield TestClient(app, raise_server_exceptions=True), engine


def test_recall_succeeds_on_a_store_with_a_damaged_value_in_every_column(damaged_store) -> None:
    _http, engine = damaged_store
    response = engine.recall("Alice Bob Tokyo Postgres billing")
    assert response.results, "recall returned nothing from a store that holds three facts"


@pytest.mark.parametrize("path", [
    "/api/memories",
    "/api/clusters",
    "/api/graph",
    "/api/export?format=json",
    "/api/compliance/gdpr/export",
])
def test_the_listings_and_exports_succeed(damaged_store, path) -> None:
    http, _engine = damaged_store
    reply = http.get(path)
    assert reply.status_code == 200, f"{path}: {reply.status_code} {reply.text[:300]}"
    body = reply.content
    if reply.headers.get("content-type") == "application/gzip":
        body = gzip.decompress(body)
    json.loads(body)  # a whole document, not a torn one


def test_a_memory_and_a_fact_open(damaged_store) -> None:
    http, engine = damaged_store
    memory_id = dict(engine._db.execute("SELECT memory_id FROM memories LIMIT 1")[0])["memory_id"]
    fact_id = dict(engine._db.execute("SELECT fact_id FROM atomic_facts LIMIT 1")[0])["fact_id"]
    for path in (f"/api/memories/{memory_id}/detail", f"/api/memories/{memory_id}/facts",
                 f"/api/facts/{fact_id}"):
        reply = http.get(path)
        assert reply.status_code == 200, f"{path}: {reply.status_code} {reply.text[:300]}"
