# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later

"""M029 — profile-scoped indexes for large behavioral histories.

Dashboard reads always filter behavioral and outcome history by profile before
sorting or aggregating. These composite indexes keep those operations bounded
to one tenant as the shared database grows.
"""

from __future__ import annotations

import sqlite3

NAME = "M029_behavioral_history_indexes"
DB_TARGET = "memory"

DDL = """
CREATE INDEX IF NOT EXISTS idx_outcomes_profile_outcome_time
    ON action_outcomes(profile_id, outcome, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_outcomes_profile_settled_time
    ON action_outcomes(profile_id, settled, settled_at DESC);
CREATE INDEX IF NOT EXISTS idx_outcomes_profile_settled_cursor
    ON action_outcomes(
        profile_id, settled, COALESCE(settled_at, ''), outcome_id
    );
CREATE INDEX IF NOT EXISTS idx_assertions_profile_confidence
    ON behavioral_assertions(profile_id, confidence DESC);
CREATE INDEX IF NOT EXISTS idx_tool_events_profile_created
    ON tool_events(profile_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_soft_prompts_profile_active_category
    ON soft_prompt_templates(profile_id, active, category, prompt_id);
"""

_INDEXES: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    (
        "idx_outcomes_profile_outcome_time",
        "action_outcomes",
        ("profile_id", "outcome", "timestamp"),
    ),
    (
        "idx_outcomes_profile_settled_time",
        "action_outcomes",
        ("profile_id", "settled", "settled_at"),
    ),
    (
        "idx_outcomes_profile_settled_cursor",
        "action_outcomes",
        ("profile_id", "settled", "settled_at", "outcome_id"),
    ),
    (
        "idx_assertions_profile_confidence",
        "behavioral_assertions",
        ("profile_id", "confidence"),
    ),
    (
        "idx_tool_events_profile_created",
        "tool_events",
        ("profile_id", "created_at"),
    ),
    (
        "idx_soft_prompts_profile_active_category",
        "soft_prompt_templates",
        ("profile_id", "active", "category", "prompt_id"),
    ),
)


def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {
        str(row[1])
        for row in conn.execute(f"PRAGMA table_info({table})").fetchall()
    }


def _index_sql(name: str) -> str:
    statements = {
        "idx_outcomes_profile_outcome_time": (
            "CREATE INDEX IF NOT EXISTS idx_outcomes_profile_outcome_time "
            "ON action_outcomes(profile_id, outcome, timestamp DESC)"
        ),
        "idx_outcomes_profile_settled_time": (
            "CREATE INDEX IF NOT EXISTS idx_outcomes_profile_settled_time "
            "ON action_outcomes(profile_id, settled, settled_at DESC)"
        ),
        "idx_outcomes_profile_settled_cursor": (
            "CREATE INDEX IF NOT EXISTS idx_outcomes_profile_settled_cursor "
            "ON action_outcomes("
            "profile_id, settled, COALESCE(settled_at, ''), outcome_id)"
        ),
        "idx_assertions_profile_confidence": (
            "CREATE INDEX IF NOT EXISTS idx_assertions_profile_confidence "
            "ON behavioral_assertions(profile_id, confidence DESC)"
        ),
        "idx_tool_events_profile_created": (
            "CREATE INDEX IF NOT EXISTS idx_tool_events_profile_created "
            "ON tool_events(profile_id, created_at DESC)"
        ),
        "idx_soft_prompts_profile_active_category": (
            "CREATE INDEX IF NOT EXISTS "
            "idx_soft_prompts_profile_active_category "
            "ON soft_prompt_templates(profile_id, active, category, prompt_id)"
        ),
    }
    return statements[name]


def apply(conn: sqlite3.Connection) -> None:
    """Add each index supported by the installed runtime schema.

    Optional v3.2/v3.4.7 tables also declare these indexes in their owning
    schema modules, so a partial bootstrap cannot permanently miss an index
    when those tables are created later.
    """
    for name, table, required_columns in _INDEXES:
        if set(required_columns) <= _columns(conn, table):
            conn.execute(_index_sql(name))


def verify(conn: sqlite3.Connection) -> bool:
    """Require each index whose backing table and columns are present."""
    names = {
        str(row[0])
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index'"
        ).fetchall()
    }
    return all(
        name in names
        for name, table, required_columns in _INDEXES
        if set(required_columns) <= _columns(conn, table)
    )


def repair(conn: sqlite3.Connection) -> None:
    """Create only indexes supported by the current optional schema."""
    apply(conn)
