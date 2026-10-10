# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""A data export says which memories came from a picture, a document page or a
connected folder (text only, no files, no paths), and an import still saves text."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from superlocalmemory.server.routes import data_io
from superlocalmemory.storage import schema as real_schema
from superlocalmemory.storage.database import DatabaseManager

_MARKERS = {
    "m-pic": {"type": "media", "media_id": "a" * 32, "origin": "tool"},
    "m-page": {"type": "document", "document_id": "d" * 32, "page": 3, "part": 2},
    "m-folder": {"type": "folder", "source_id": "s1", "relpath": "private/diary.md", "version": "v9"},
    "m-folder-pic": {"type": "folder", "source_id": "s1", "media_id": "b" * 32,
                     "relpath": "photos/me.jpg"},
}


@pytest.fixture()
def db_path(tmp_path: Path) -> Path:
    mgr = DatabaseManager(tmp_path / "memory.db")
    mgr.initialize(real_schema)
    mgr.execute("INSERT OR IGNORE INTO profiles (profile_id, name) VALUES ('default', 'default')")
    rows = {**_MARKERS, "m-plain": None}
    for memory_id, marker in rows.items():
        meta = json.dumps({"_slm_source": marker} if marker else {})
        mgr.execute("INSERT INTO memories (memory_id, profile_id, content, metadata_json) "
                    "VALUES (?, 'default', ?, ?)", (memory_id, f"text of {memory_id}", meta))
        mgr.execute("INSERT INTO atomic_facts (fact_id, memory_id, profile_id, content) "
                    "VALUES (?, ?, 'default', ?)", (f"f-{memory_id}", memory_id, f"fact of {memory_id}"))
    return tmp_path / "memory.db"


def _export(db_path: Path) -> dict:
    app = FastAPI()
    app.include_router(data_io.router)

    def connect():
        return sqlite3.connect(str(db_path))

    with patch.object(data_io, "get_db_connection", connect), \
            patch.object(data_io, "get_active_profile", lambda: "default"), \
            patch("superlocalmemory.server.write_identity.require_http_mutation_actor",
                  lambda *a, **k: "test"):
        reply = TestClient(app).get("/api/export?format=json")
    assert reply.status_code == 200
    return {m["fact_id"]: m for m in reply.json()["memories"]}


def test_media_document_and_folder_memories_are_marked(db_path: Path) -> None:
    out = _export(db_path)
    assert out["f-m-pic"]["source"] == {"type": "media", "media_id": "a" * 32}
    assert out["f-m-page"]["source"] == {"type": "document_page", "document_id": "d" * 32,
                                         "page": 3, "part": 2}
    assert out["f-m-folder"]["source"] == {"type": "folder", "source_id": "s1"}
    assert out["f-m-folder-pic"]["source"]["type"] == "folder"


def test_ordinary_memories_have_no_source_field(db_path: Path) -> None:
    assert "source" not in _export(db_path)["f-m-plain"]


def test_marker_carries_no_paths(db_path: Path) -> None:
    text = json.dumps(_export(db_path))
    for leak in ("diary.md", "me.jpg", "relpath", "v9"):
        assert leak not in text


def test_import_of_a_marked_record_saves_plain_text(engine_with_mock_deps) -> None:
    from superlocalmemory.core.ingestion_command import IngestionOperationRepository
    from superlocalmemory.storage.migrations import M018_ingestion_operations

    engine = engine_with_mock_deps
    with engine._db.raw_connection() as conn:
        M018_ingestion_operations.apply(conn)
    app = FastAPI()
    app.state.engine = engine

    @app.middleware("http")
    async def _actor(request, call_next):
        request.state.authenticated_actor = "authenticated:test-import"
        return await call_next(request)

    app.include_router(data_io.router)
    payload = json.dumps({"memories": [{
        "content": "Text that once came from a picture.",
        "source": {"type": "media", "media_id": "a" * 32}}]}).encode()
    reply = TestClient(app).post("/api/import", files={"file": ("m.json", payload, "application/json")})
    assert reply.status_code == 200 and reply.json()["imported_count"] == 1
    rows = engine._db.execute("SELECT metadata_json, content FROM memories")
    saved = [dict(r) for r in rows]
    assert saved and all("_slm_source" not in r["metadata_json"] for r in saved)
