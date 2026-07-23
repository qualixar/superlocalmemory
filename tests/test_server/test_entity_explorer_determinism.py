"""Entity Explorer count/list determinism under multi-project summaries."""

from __future__ import annotations

import sqlite3
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from superlocalmemory.server.routes import entity
from superlocalmemory.storage.migrations import (
    M030_entity_explorer_indexes as m030,
)


def _client(tmp_path, monkeypatch) -> TestClient:
    db_path = tmp_path / "memory.db"
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE canonical_entities (
            entity_id TEXT PRIMARY KEY,
            canonical_name TEXT,
            entity_type TEXT,
            fact_count INTEGER,
            first_seen TEXT,
            last_seen TEXT,
            profile_id TEXT
        );
        CREATE TABLE entity_profiles (
            entity_id TEXT,
            profile_id TEXT,
            project_name TEXT,
            knowledge_summary TEXT,
            compiled_truth TEXT,
            compilation_confidence REAL,
            last_compiled_at TEXT
        );
        INSERT INTO canonical_entities VALUES
            ('e2', 'Beta', 'person', 5, '', '', 'work'),
            ('e1', 'Alpha', 'person', 5, '', '', 'work'),
            ('e3', 'Gamma', 'person', 5, '', '', 'work');
        INSERT INTO entity_profiles VALUES
            ('e1', 'work', 'z-project', 'older summary', '', 0.6,
             '2026-01-01'),
            ('e1', 'work', 'a-project', 'latest summary', 'truth', 0.9,
             '2026-02-01'),
            ('e2', 'work', 'only', 'beta summary', '', 0.7, '2026-01-01'),
            ('e3', 'work', 'only', 'gamma summary', '', 0.7, '2026-01-01');
        """
    )
    conn.commit()
    conn.close()

    monkeypatch.setattr(entity, "_require_read", lambda request, profile: None)
    app = FastAPI()
    app.state.engine = SimpleNamespace(
        _config=SimpleNamespace(db_path=db_path)
    )
    app.include_router(entity.router)
    return TestClient(app)


def test_entity_list_is_unique_and_stable_across_tied_pages(
    tmp_path, monkeypatch
):
    client = _client(tmp_path, monkeypatch)

    first = client.get("/api/entity/list?profile=work&limit=2&offset=0")
    second = client.get("/api/entity/list?profile=work&limit=2&offset=2")

    assert first.status_code == second.status_code == 200
    assert first.json()["total"] == second.json()["total"] == 3
    assert [row["entity_id"] for row in first.json()["entities"]] == ["e1", "e2"]
    assert [row["entity_id"] for row in second.json()["entities"]] == ["e3"]
    assert first.json()["entities"][0]["summary_preview"] == "latest summary"
    assert first.json()["entities"][0]["has_compiled_truth"] is True


def test_entity_search_matches_any_project_without_duplicate_count(
    tmp_path, monkeypatch
):
    client = _client(tmp_path, monkeypatch)

    response = client.get("/api/entity/list?profile=work&search=older")

    assert response.status_code == 200
    assert response.json()["total"] == 1
    assert [row["entity_id"] for row in response.json()["entities"]] == ["e1"]


def test_entity_page_query_uses_page_first_composite_indexes(tmp_path):
    db_path = tmp_path / "memory.db"
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE canonical_entities (
            entity_id TEXT PRIMARY KEY,
            canonical_name TEXT,
            entity_type TEXT,
            fact_count INTEGER,
            first_seen TEXT,
            last_seen TEXT,
            profile_id TEXT
        );
        CREATE TABLE entity_profiles (
            entity_id TEXT,
            profile_id TEXT,
            project_name TEXT,
            knowledge_summary TEXT,
            compiled_truth TEXT,
            compilation_confidence REAL,
            last_compiled_at TEXT
        );
        """
    )
    m030.apply(conn)

    plan = conn.execute(
        "EXPLAIN QUERY PLAN " + entity._list_entities_sql("ce.profile_id = ?"),
        ("work", 25, 0, "work"),
    ).fetchall()
    plan_text = "\n".join(str(row[3]) for row in plan)

    assert "idx_entities_profile_fact_count_id" in plan_text
    assert "idx_entity_profiles_profile_entity_rank" in plan_text
    assert plan_text.index("MATERIALIZE page_entities") < plan_text.index(
        "MATERIALIZE ranked_profiles"
    )
