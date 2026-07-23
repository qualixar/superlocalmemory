"""Truthfulness tests for profile-scoped Brain telemetry."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.requests import Request


def _learning_request(repair_status: dict | None = None) -> Request:
    app = SimpleNamespace(
        state=SimpleNamespace(
            source_quality_repair_status=repair_status,
        ),
    )
    return Request({"type": "http", "app": app, "headers": []})


def _seed_memory_db(path: Path) -> None:
    connection = sqlite3.connect(path)
    try:
        connection.execute(
            "CREATE TABLE action_outcomes ("
            "profile_id TEXT, outcome TEXT, context_json TEXT, "
            "timestamp TEXT, reward REAL, settled INTEGER, settled_at TEXT)",
        )
        connection.executemany(
            "INSERT INTO action_outcomes VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    "default", "success", '{"action_type":"deploy"}',
                    "2026-07-23T10:00:00+00:00", 1.0, 1,
                    "2026-07-23T10:00:00+00:00",
                ),
                (
                    "other", "failure", '{"action_type":"other"}',
                    "2026-07-23T11:00:00+00:00", 0.0, 1,
                    "2026-07-23T11:00:00+00:00",
                ),
            ],
        )
        connection.commit()
    finally:
        connection.close()


def test_behavioral_status_reads_profile_scoped_memory_outcomes(
    monkeypatch, tmp_path: Path,
) -> None:
    """Dashboard reports actual action_outcomes, not a wrong-database zero."""
    from superlocalmemory.server.routes import behavioral

    _seed_memory_db(tmp_path / "memory.db")
    monkeypatch.setattr(behavioral, "MEMORY_DIR", tmp_path)
    monkeypatch.setattr(behavioral, "get_active_profile", lambda: "default")

    status = behavioral.behavioral_status()

    assert status["total_outcomes"] == 1
    assert status["outcome_breakdown"] == {
        "success": 1, "failure": 0, "partial": 0,
    }
    assert status["recent_outcomes"] == [{
        "outcome": "success",
        "action_type": "deploy",
        "timestamp": "2026-07-23T10:00:00+00:00",
        "source": "memory.db:action_outcomes",
    }]
    assert status["outcomes_source"] == "memory.db:action_outcomes"
    assert status["outcomes_are_finalized"] is True
    assert status["outcomes_provenance"] == "explicit_reports_or_finalized_signals"


def test_behavioral_status_separates_settled_rewards_from_explicit_outcomes(
    monkeypatch, tmp_path: Path,
) -> None:
    """Numeric settled labels remain visible without being called successes."""
    from superlocalmemory.server.routes import behavioral

    memory_db = tmp_path / "memory.db"
    connection = sqlite3.connect(memory_db)
    try:
        connection.execute(
            "CREATE TABLE action_outcomes ("
            "profile_id TEXT, outcome TEXT, context_json TEXT, "
            "timestamp TEXT, reward REAL, settled INTEGER, settled_at TEXT)",
        )
        connection.executemany(
            "INSERT INTO action_outcomes VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    "default", "settled", "{}", "2026-07-20T10:00:00Z",
                    0.9, 1, "2026-07-20T10:00:00Z",
                ),
                (
                    "default", "settled", "{}", "2026-07-20T11:00:00Z",
                    0.5, 1, "2026-07-20T11:00:00Z",
                ),
                (
                    "default", "settled", "{}", "2026-07-21T10:00:00Z",
                    0.1, 1, "2026-07-21T10:00:00Z",
                ),
                (
                    "default", "success", '{"action_type":"ship"}',
                    "2026-07-22T10:00:00Z", 1.0, 1,
                    "2026-07-22T10:00:00Z",
                ),
                (
                    "other", "settled", "{}", "2026-07-23T10:00:00Z",
                    1.0, 1, "2026-07-23T10:00:00Z",
                ),
            ],
        )
        connection.commit()
    finally:
        connection.close()

    monkeypatch.setattr(behavioral, "MEMORY_DIR", tmp_path)
    monkeypatch.setattr(behavioral, "get_active_profile", lambda: "default")

    status = behavioral.behavioral_status()

    assert status["total_outcomes"] == 1
    assert status["outcome_breakdown"] == {
        "success": 1, "failure": 0, "partial": 0,
    }
    assert status["reward_telemetry"] == {
        "count": 4,
        "average": 0.625,
        "distribution": {"positive": 2, "neutral": 1, "negative": 1},
        "timeline": [
            {"date": "2026-07-20", "count": 2, "average": 0.7},
            {"date": "2026-07-21", "count": 1, "average": 0.1},
            {"date": "2026-07-22", "count": 1, "average": 1.0},
        ],
        "source": "memory.db:action_outcomes.reward",
        "window_days": 182,
    }
    assert status["recent_outcomes"] == [{
        "outcome": "success",
        "action_type": "ship",
        "timestamp": "2026-07-22T10:00:00Z",
        "source": "memory.db:action_outcomes",
    }]


def test_behavioral_status_survives_pre_reward_schema(
    monkeypatch, tmp_path: Path,
) -> None:
    """Upgraded legacy users get empty reward telemetry, not a 500."""
    from superlocalmemory.server.routes import behavioral

    connection = sqlite3.connect(tmp_path / "memory.db")
    try:
        connection.execute(
            "CREATE TABLE action_outcomes ("
            "profile_id TEXT, outcome TEXT, context_json TEXT, timestamp TEXT)",
        )
        connection.execute(
            "INSERT INTO action_outcomes VALUES (?, ?, ?, ?)",
            ("default", "partial", "{}", "2026-07-20T10:00:00Z"),
        )
        connection.commit()
    finally:
        connection.close()
    monkeypatch.setattr(behavioral, "MEMORY_DIR", tmp_path)
    monkeypatch.setattr(behavioral, "get_active_profile", lambda: "default")

    status = behavioral.behavioral_status()

    assert status["total_outcomes"] == 1
    assert status["reward_telemetry"]["count"] == 0


def test_behavioral_status_detects_transferred_from_metadata(
    monkeypatch, tmp_path: Path,
) -> None:
    from superlocalmemory.server.routes import behavioral

    class Store:
        def __init__(self, _path: str) -> None:
            pass

        def get_patterns(self, profile_id: str) -> list[dict]:
            assert profile_id == "default"
            return [
                {
                    "pattern_key": "python",
                    "metadata": {"transferred_from": "work"},
                },
                {"pattern_key": "local", "metadata": {}},
            ]

    monkeypatch.setattr(behavioral, "MEMORY_DIR", tmp_path)
    monkeypatch.setattr(behavioral, "get_active_profile", lambda: "default")
    monkeypatch.setattr(behavioral, "BehavioralPatternStore", Store)

    status = behavioral.behavioral_status()

    assert status["cross_project_transfers"] == 1
    assert status["cross_project_patterns"] == [{
        "pattern_key": "python",
        "metadata": {"transferred_from": "work"},
    }]


def test_tool_events_rejects_unbounded_negative_limit(
    monkeypatch, tmp_path: Path,
) -> None:
    from superlocalmemory.server.routes import behavioral

    monkeypatch.setattr(behavioral, "MEMORY_DIR", tmp_path)
    app = FastAPI()
    app.include_router(behavioral.router)
    client = TestClient(app)

    assert client.get("/api/behavioral/tool-events?limit=-1").status_code == 422
    assert client.get("/api/behavioral/tool-events?limit=1001").status_code == 422
    assert client.get("/api/behavioral/assertions?limit=-1").status_code == 422
    assert client.get(
        "/api/behavioral/assertions?min_confidence=1.1",
    ).status_code == 422


def test_explicit_outcome_feeds_fact_provenance_source_quality(
    monkeypatch, tmp_path: Path,
) -> None:
    from superlocalmemory.learning.source_quality import SourceQualityScorer
    from superlocalmemory.server.routes import behavioral

    connection = sqlite3.connect(tmp_path / "memory.db")
    try:
        connection.executescript(
            """
            CREATE TABLE action_outcomes (
                outcome_id TEXT PRIMARY KEY, profile_id TEXT, query TEXT,
                fact_ids_json TEXT, outcome TEXT, context_json TEXT,
                timestamp TEXT, reward REAL, settled INTEGER,
                settled_at TEXT
            );
                CREATE TABLE provenance (
                    profile_id TEXT, fact_id TEXT, source_type TEXT,
                    source_id TEXT, created_by TEXT
                );
                CREATE TABLE atomic_facts (
                    fact_id TEXT PRIMARY KEY, profile_id TEXT
                );
                INSERT INTO atomic_facts VALUES ('f1', 'default');
                INSERT INTO provenance VALUES
                    ('default', 'f1', 'mcp', 'claude-code', '');
            """
        )
        connection.commit()
    finally:
        connection.close()
    monkeypatch.setattr(behavioral, "MEMORY_DIR", tmp_path)
    monkeypatch.setattr(behavioral, "get_active_profile", lambda: "default")

    result = behavioral.report_outcome(_learning_request(), {
        "memory_ids": ["f1"],
        "outcome": "success",
        "action_type": "ship",
    })

    assert result["success"] is True
    scorer = SourceQualityScorer(tmp_path / "learning.db")
    assert scorer.get_quality("default", "mcp:claude-code") > 0.5


def test_learning_status_reads_persisted_source_quality_only(
    monkeypatch, tmp_path: Path,
) -> None:
    """Source scores are real posterior values, never feedback-derived guesses."""
    from superlocalmemory.server.routes import learning

    learning_db = tmp_path / "learning.db"
    connection = sqlite3.connect(learning_db)
    try:
        connection.execute(
            "CREATE TABLE source_quality ("
            "profile_id TEXT, source_id TEXT, alpha REAL, beta REAL, updated_at TEXT)",
        )
        connection.executemany(
            "INSERT INTO source_quality VALUES (?, ?, ?, ?, ?)",
            [
                ("default", "manual", 4.0, 1.0, "2026-07-23T10:00:00+00:00"),
                ("other", "excluded", 1.0, 4.0, "2026-07-23T10:00:00+00:00"),
            ],
        )
        connection.commit()
    finally:
        connection.close()

    monkeypatch.setattr(learning, "LEARNING_DB", learning_db)
    monkeypatch.setattr(learning, "get_active_profile", lambda: "default")
    monkeypatch.setattr(learning, "_get_feedback", lambda: None)
    monkeypatch.setattr(learning, "_get_engagement", lambda: None)

    status = learning.learning_status(_learning_request({
        "state": "running",
        "source": "startup_background_repair",
        "batch_size": 25,
        "profiles": ["default", "other"],
        "completed_profiles": ["other"],
        "batches_completed": 3,
        "scanned": 75,
        "observations": 12,
        "last_error": None,
    }))

    assert status["source_scores"] == {"manual": 0.8}
    assert status["source_scores_source"] == "learning.db:source_quality"
    assert status["source_scores_are_posterior"] is True
    assert status["stats"]["tracked_sources"] == 1
    assert status["telemetry_status"]["source_quality"] == "available"
    assert status["telemetry_status"]["source_quality_repair"] == "running"
    assert status["source_quality_repair"] == {
        "state": "running",
        "source": "startup_background_repair",
        "batch_size": 25,
        "profiles_total": 2,
        "profiles_complete": 1,
        "batches_completed": 3,
        "scanned": 75,
        "observations": 12,
        "last_error": None,
    }


def test_learning_status_uses_verified_ranker_and_persisted_model_rows(
    monkeypatch, tmp_path: Path,
) -> None:
    """Signal volume alone must not claim an active or trained ML model."""
    from superlocalmemory.server.routes import learning

    learning_db = tmp_path / "learning.db"
    connection = sqlite3.connect(learning_db)
    try:
        connection.execute(
            "CREATE TABLE learning_model_state ("
            "id INTEGER PRIMARY KEY, profile_id TEXT, is_active INTEGER)",
        )
        connection.executemany(
            "INSERT INTO learning_model_state VALUES (?, ?, ?)",
            [
                (1, "default", 0),
                (2, "default", 0),
                (3, "other", 1),
            ],
        )
        connection.execute(
            "CREATE TABLE source_quality ("
            "profile_id TEXT, source_id TEXT, alpha REAL, beta REAL, "
            "updated_at TEXT)",
        )
        connection.executemany(
            "INSERT INTO source_quality VALUES (?, ?, ?, ?, ?)",
            [
                ("default", "manual", 4.0, 1.0, "2026-07-23T10:00:00Z"),
                ("default", "codex", 3.0, 2.0, "2026-07-23T11:00:00Z"),
                ("other", "excluded", 5.0, 1.0, "2026-07-23T12:00:00Z"),
            ],
        )
        connection.commit()
    finally:
        connection.close()

    verified_phase = {
        "phase": 2,
        "key": "rule_based",
        "label": "Contextual bandit",
        "model_active": False,
        "signals": 2_814,
        "gates": {
            "rule_based_min_signals": 50,
            "ml_model_min_signals": 200,
            "ml_model_requires_verified_active_model": True,
        },
        "status": "available",
    }
    monkeypatch.setattr(learning, "LEARNING_DB", learning_db)
    monkeypatch.setattr(learning, "get_active_profile", lambda: "default")
    monkeypatch.setattr(learning, "_compute_ranker_phase", lambda _profile: verified_phase)
    monkeypatch.setattr(learning, "_get_feedback", lambda: None)
    monkeypatch.setattr(learning, "_get_engagement", lambda: None)

    status = learning.learning_status(_learning_request())

    assert status["ranking_phase"] == "rule_based"
    assert status["ranker_phase"] == verified_phase
    assert status["ranking_phase_gates"] == verified_phase["gates"]
    assert status["stats"]["models_trained"] == 2
    assert status["stats"]["models_active_verified"] == 0
    assert status["stats"]["tracked_sources"] == 2
    assert status["telemetry_status"]["model_state"] == "available"


def test_source_quality_missing_table_is_honest_empty(tmp_path: Path) -> None:
    """A pre-source-quality database is empty, not internally failed."""
    from superlocalmemory.server.routes.learning import _load_source_quality_state

    learning_db = tmp_path / "learning.db"
    sqlite3.connect(learning_db).close()

    state = _load_source_quality_state("default", learning_db)

    assert state == {
        "scores": {},
        "tracked_sources": 0,
        "status": "missing_table",
    }


def test_learning_pattern_fields_use_one_explicit_schema(
    monkeypatch, tmp_path: Path,
) -> None:
    """Tech and workflow items expose the same explicit field names."""
    from superlocalmemory.learning import behavioral
    from superlocalmemory.server.routes import learning

    class _PatternStore:
        def __init__(self, _db_path: str) -> None:
            pass

        def get_patterns(self, *, profile_id: str) -> list[dict]:
            assert profile_id == "default"
            return [
                {
                    "pattern_type": "tech_preference",
                    "pattern_key": "python",
                    "metadata": {"value": "Python"},
                    "confidence": 0.9,
                    "evidence_count": 12,
                },
                {
                    "pattern_type": "workflow",
                    "pattern_key": "release_gate",
                    "metadata": {"value": "test before release"},
                    "confidence": 0.8,
                    "evidence_count": 7,
                },
            ]

    learning_db = tmp_path / "learning.db"
    sqlite3.connect(learning_db).close()
    monkeypatch.setattr(behavioral, "BehavioralPatternStore", _PatternStore)
    monkeypatch.setattr(learning, "LEARNING_DB", learning_db)
    monkeypatch.setattr(learning, "get_active_profile", lambda: "default")
    monkeypatch.setattr(
        learning,
        "_compute_ranker_phase",
        lambda _profile: {
            "phase": 1,
            "key": "baseline",
            "label": "Cold start (cross-encoder only)",
            "model_active": False,
            "signals": 0,
            "gates": {
                "rule_based_min_signals": 50,
                "ml_model_min_signals": 200,
                "ml_model_requires_verified_active_model": True,
            },
            "status": "available",
        },
    )
    monkeypatch.setattr(learning, "_get_feedback", lambda: None)
    monkeypatch.setattr(learning, "_get_engagement", lambda: None)

    status = learning.learning_status(_learning_request())

    assert status["tech_preferences"][0] == {
        "type": "tech_preference",
        "key": "python",
        "value": "Python",
        "confidence": 0.9,
        "evidence_count": 12,
        "evidence": 12,
    }
    assert status["workflow_patterns"][0] == {
        "type": "workflow",
        "key": "release_gate",
        "value": "test before release",
        "confidence": 0.8,
        "evidence_count": 7,
    }


def test_brain_outcome_preview_reads_memory_database(
    monkeypatch, tmp_path: Path,
) -> None:
    """The unified Brain route counts the table where outcomes are stored."""
    from superlocalmemory.server.routes import brain

    memory_db = tmp_path / "memory.db"
    _seed_memory_db(memory_db)
    monkeypatch.setattr(brain, "_memory_db_path", lambda: memory_db)

    preview = brain._compute_action_outcomes_preview("default")

    assert preview == {
        "action_outcomes_rows": 1,
        "source": "memory.db:action_outcomes",
        "is_real": True,
    }
