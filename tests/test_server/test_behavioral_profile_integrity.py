"""Behavioral dashboard profile-isolation and outcome-integrity contracts."""

from __future__ import annotations

import json
import sqlite3

from fastapi import FastAPI
from fastapi.testclient import TestClient

from superlocalmemory.server.routes import behavioral


def _client(tmp_path, monkeypatch, *, active_profile: str = "alpha") -> TestClient:
    monkeypatch.setattr(behavioral, "MEMORY_DIR", tmp_path)
    monkeypatch.setattr(
        behavioral, "get_active_profile", lambda: active_profile
    )
    monkeypatch.setattr(behavioral, "_require_write", lambda request: None)
    monkeypatch.setattr(behavioral, "_require_read", lambda request: None)
    app = FastAPI()
    app.include_router(behavioral.router)
    return TestClient(app)


def _create_behavioral_db(path) -> None:
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE atomic_facts (
            fact_id TEXT PRIMARY KEY,
            profile_id TEXT NOT NULL
        );
        CREATE TABLE action_outcomes (
            outcome_id TEXT PRIMARY KEY,
            profile_id TEXT NOT NULL,
            query TEXT NOT NULL,
            fact_ids_json TEXT NOT NULL,
            outcome TEXT NOT NULL,
            context_json TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            reward REAL,
            settled INTEGER,
            settled_at TEXT
        );
        CREATE TABLE soft_prompt_templates (
            prompt_id TEXT PRIMARY KEY,
            profile_id TEXT NOT NULL,
            category TEXT NOT NULL,
            content TEXT NOT NULL,
            confidence REAL NOT NULL,
            effectiveness REAL NOT NULL,
            token_count INTEGER NOT NULL,
            active INTEGER NOT NULL,
            version INTEGER NOT NULL,
            created_at TEXT NOT NULL
        );
        INSERT INTO atomic_facts VALUES ('alpha-fact', 'alpha');
        INSERT INTO atomic_facts VALUES ('beta-fact', 'beta');
        INSERT INTO soft_prompt_templates VALUES
            ('alpha-prompt', 'alpha', 'identity', 'alpha only',
             0.9, 0.8, 2, 1, 1, '2026-01-01'),
            ('beta-prompt', 'beta', 'identity', 'beta only',
             0.9, 0.8, 2, 1, 1, '2026-01-01');
        """
    )
    conn.commit()
    conn.close()


def test_soft_prompts_only_returns_active_profile_to_viewer(
    tmp_path, monkeypatch
):
    _create_behavioral_db(tmp_path / "memory.db")
    client = _client(tmp_path, monkeypatch)
    authorized = []
    monkeypatch.setattr(
        behavioral,
        "_require_read",
        lambda request: authorized.append(behavioral.get_active_profile()),
    )

    response = client.get("/api/behavioral/soft-prompts")

    assert response.status_code == 200
    assert authorized == ["alpha"]
    assert [row["prompt_id"] for row in response.json()["prompts"]] == [
        "alpha-prompt"
    ]


def test_report_outcome_deduplicates_bounded_profile_fact_ids(
    tmp_path, monkeypatch
):
    _create_behavioral_db(tmp_path / "memory.db")
    client = _client(tmp_path, monkeypatch)

    response = client.post(
        "/api/behavioral/report-outcome",
        json={
            "memory_ids": ["alpha-fact", "alpha-fact"],
            "outcome": "success",
            "action_type": "answer",
        },
    )

    assert response.status_code == 200
    assert response.json()["success"] is True
    conn = sqlite3.connect(tmp_path / "memory.db")
    stored = conn.execute(
        "SELECT fact_ids_json FROM action_outcomes"
    ).fetchone()[0]
    conn.close()
    assert json.loads(stored) == ["alpha-fact"]


def test_report_outcome_rejects_cross_profile_fact_before_commit(
    tmp_path, monkeypatch
):
    _create_behavioral_db(tmp_path / "memory.db")
    client = _client(tmp_path, monkeypatch)

    response = client.post(
        "/api/behavioral/report-outcome",
        json={"memory_ids": ["alpha-fact", "beta-fact"], "outcome": "failure"},
    )

    assert response.status_code == 422
    conn = sqlite3.connect(tmp_path / "memory.db")
    assert conn.execute("SELECT COUNT(*) FROM action_outcomes").fetchone()[0] == 0
    conn.close()


def test_report_outcome_rejects_unbounded_or_extra_payload(
    tmp_path, monkeypatch
):
    _create_behavioral_db(tmp_path / "memory.db")
    client = _client(tmp_path, monkeypatch)

    too_many = client.post(
        "/api/behavioral/report-outcome",
        json={
            "memory_ids": [f"fact-{index}" for index in range(101)],
            "outcome": "partial",
        },
    )
    extra = client.post(
        "/api/behavioral/report-outcome",
        json={
            "memory_ids": ["alpha-fact"],
            "outcome": "partial",
            "unexpected": True,
        },
    )

    assert too_many.status_code == 422
    assert extra.status_code == 422
