"""Security and runtime-truth contracts for dashboard control-plane routes."""

from __future__ import annotations

import inspect
import json
import sqlite3
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from superlocalmemory.server import unified_daemon
from superlocalmemory.server.routes import behavioral, entity, evolution, learning


def _entity_client(tmp_path, monkeypatch):
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
            last_compiled_at TEXT,
            timeline TEXT,
            fact_ids_json TEXT
        );
        INSERT INTO canonical_entities VALUES
            ('e1', 'Ada', 'person', 1, '', '', 'work');
        INSERT INTO entity_profiles VALUES
            ('e1', 'work', '', 'summary', 'truth', 0.9, '', '[]', '[]');
        """
    )
    conn.commit()
    conn.close()

    app = FastAPI()
    app.state.engine = SimpleNamespace(
        _config=SimpleNamespace(db_path=db_path),
    )
    app.include_router(entity.router)
    return TestClient(app)


def test_entity_reads_authorize_the_requested_profile(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(entity, "_require_read", lambda request, profile: seen.append(profile))
    client = _entity_client(tmp_path, monkeypatch)

    listed = client.get("/api/entity/list?profile=work")
    detailed = client.get("/api/entity/Ada?profile=work")

    assert listed.status_code == 200
    assert detailed.status_code == 200
    assert seen == ["work", "work"]


def test_entity_recompile_requires_manage_for_requested_profile(
    tmp_path, monkeypatch
):
    seen = []
    monkeypatch.setattr(
        entity, "_require_manage", lambda request, profile: seen.append(profile)
    )
    client = _entity_client(tmp_path, monkeypatch)
    monkeypatch.setattr(
        "superlocalmemory.learning.entity_compiler.EntityCompiler.compile_entity",
        lambda self, profile, project, entity_id, name: {"entity_name": name},
    )

    response = client.post("/api/entity/Ada/recompile?profile=work")

    assert response.status_code == 200
    assert seen == ["work"]


def test_blocking_entity_handlers_are_sync():
    assert not inspect.iscoroutinefunction(entity.list_entities)
    assert not inspect.iscoroutinefunction(entity.get_entity)
    assert not inspect.iscoroutinefunction(entity.recompile_entity)


def _evolution_client(monkeypatch, tmp_path):
    monkeypatch.setattr(evolution, "MEMORY_DIR", tmp_path)
    app = FastAPI()
    app.include_router(evolution.router)
    return TestClient(app)


def test_evolution_reads_require_read(monkeypatch, tmp_path):
    seen = []
    monkeypatch.setattr(evolution, "_require_read", lambda request: seen.append(request))
    monkeypatch.setattr(
        "superlocalmemory.evolution.evolution_store.EvolutionStore.get_stats",
        lambda self, profile: {},
    )
    monkeypatch.setattr(
        "superlocalmemory.evolution.evolution_store.EvolutionStore.get_recent",
        lambda self, profile, limit: [],
    )
    conn = sqlite3.connect(tmp_path / "memory.db")
    conn.execute(
        """
        CREATE TABLE skill_evolution_log (
            id TEXT, skill_name TEXT, parent_skill_id TEXT,
            evolution_type TEXT, trigger_type TEXT, generation INTEGER,
            status TEXT, mutation_summary TEXT, blind_verified INTEGER,
            created_at TEXT, completed_at TEXT, profile_id TEXT
        )
        """
    )
    conn.close()
    client = _evolution_client(monkeypatch, tmp_path)

    assert client.get("/api/evolution/status").status_code == 200
    assert client.get("/api/evolution/lineage").status_code == 200
    assert len(seen) == 2


def test_enable_preserves_selected_backend(monkeypatch, tmp_path):
    monkeypatch.setattr(evolution, "_require_manage", lambda request: None)
    (tmp_path / "config.json").write_text(
        json.dumps({"evolution": {"enabled": False, "backend": "ollama"}})
    )
    client = _evolution_client(monkeypatch, tmp_path)

    response = client.post("/api/evolution/enable")

    assert response.status_code == 200
    saved = json.loads((tmp_path / "config.json").read_text())
    assert saved["evolution"] == {"enabled": True, "backend": "ollama"}


def test_blocking_evolution_handlers_are_sync():
    for handler in (
        evolution.evolution_status,
        evolution.evolution_enable,
        evolution.evolution_disable,
        evolution.evolution_run,
        evolution.evolution_config,
        evolution.evolution_lineage,
    ):
        assert not inspect.iscoroutinefunction(handler)


def test_blocking_learning_and_behavioral_status_handlers_are_sync():
    for handler in (
        learning.ranker_phase,
        learning.learning_status,
        behavioral.behavioral_status,
    ):
        assert not inspect.iscoroutinefunction(handler)


@pytest.mark.parametrize(
    "path",
    [
        "/api/learning/status",
        "/api/learning/ranker_phase",
        "/api/behavioral/status",
        "/api/behavioral/assertions",
        "/api/behavioral/tool-events",
        "/api/behavioral/soft-prompts",
        "/api/patterns",
        "/api/feedback/stats",
        "/api/stats",
        "/api/timeline",
    ],
)
def test_learning_and_behavioral_reads_are_sensitive(path):
    assert unified_daemon._is_sensitive_dashboard_read("GET", path)
    assert not unified_daemon._is_sensitive_dashboard_read("POST", path)


def test_company_mode_rejects_unauthenticated_sensitive_read():
    class _Rbac:
        @staticmethod
        def user_count():
            return 1

        @staticmethod
        def require_login():
            return True

    request = SimpleNamespace(headers={}, cookies={})
    response = unified_daemon._rbac_read_gate(
        request,
        SimpleNamespace(rbac=_Rbac()),
    )

    assert response.status_code == 401
