"""M029 behavioral-history composite index and query-plan contracts."""

from __future__ import annotations

import sqlite3

from superlocalmemory.storage.migrations import M029_behavioral_history_indexes


def _plan(conn: sqlite3.Connection, sql: str, params=()) -> str:
    rows = conn.execute(f"EXPLAIN QUERY PLAN {sql}", params).fetchall()
    return " ".join(str(cell) for row in rows for cell in row)


def test_m029_adds_profile_scoped_behavioral_history_indexes():
    conn = sqlite3.connect(":memory:")
    conn.executescript(
        """
        CREATE TABLE action_outcomes (
            outcome_id TEXT PRIMARY KEY,
            profile_id TEXT,
            outcome TEXT,
            timestamp TEXT,
            fact_ids_json TEXT,
            settled INTEGER,
            settled_at TEXT,
            reward REAL
        );
        CREATE TABLE behavioral_assertions (
            id TEXT PRIMARY KEY,
            profile_id TEXT,
            confidence REAL
        );
        CREATE TABLE tool_events (
            id TEXT PRIMARY KEY,
            profile_id TEXT,
            tool_name TEXT,
            created_at TEXT
        );
        CREATE TABLE soft_prompt_templates (
            prompt_id TEXT PRIMARY KEY,
            profile_id TEXT,
            active INTEGER,
            category TEXT
        );
        """
    )

    M029_behavioral_history_indexes.apply(conn)

    assert M029_behavioral_history_indexes.verify(conn)
    assert "idx_outcomes_profile_outcome_time" in _plan(
        conn,
        "SELECT outcome, timestamp FROM action_outcomes "
        "WHERE profile_id=? AND outcome=? ORDER BY timestamp DESC LIMIT 20",
        ("work", "success"),
    )
    assert "idx_outcomes_profile_settled_time" in _plan(
        conn,
        "SELECT reward FROM action_outcomes "
        "WHERE profile_id=? AND settled=1 ORDER BY settled_at DESC LIMIT 20",
        ("work",),
    )
    source_quality_plan = _plan(
        conn,
        "SELECT outcome_id, fact_ids_json, reward, "
        "COALESCE(settled_at, '') AS settled_key "
        "FROM action_outcomes WHERE profile_id = ? "
        "AND settled = 1 AND reward IS NOT NULL "
        "AND typeof(reward) IN ('integer', 'real') "
        "AND (COALESCE(settled_at, '') > ? OR "
        "(COALESCE(settled_at, '') = ? AND outcome_id > ?)) "
        "ORDER BY COALESCE(settled_at, '') ASC, outcome_id ASC LIMIT ?",
        ("work", "", "", "", 250),
    )
    assert "idx_outcomes_profile_settled_cursor" in source_quality_plan
    assert "USE TEMP B-TREE" not in source_quality_plan
    assert "idx_assertions_profile_confidence" in _plan(
        conn,
        "SELECT id FROM behavioral_assertions "
        "WHERE profile_id=? ORDER BY confidence DESC LIMIT 20",
        ("work",),
    )
    assert "idx_tool_events_profile_created" in _plan(
        conn,
        "SELECT id FROM tool_events "
        "WHERE profile_id=? ORDER BY created_at DESC LIMIT 20",
        ("work",),
    )
    assert "idx_soft_prompts_profile_active_category" in _plan(
        conn,
        "SELECT prompt_id FROM soft_prompt_templates "
        "WHERE profile_id=? AND active=1 ORDER BY category, prompt_id",
        ("work",),
    )


def test_m029_supports_partial_runtime_schema_without_losing_outcome_indexes():
    conn = sqlite3.connect(":memory:")
    conn.execute(
        "CREATE TABLE action_outcomes ("
        "profile_id TEXT, outcome TEXT, timestamp TEXT, "
        "settled INTEGER, settled_at TEXT)"
    )

    M029_behavioral_history_indexes.apply(conn)

    assert M029_behavioral_history_indexes.verify(conn)
