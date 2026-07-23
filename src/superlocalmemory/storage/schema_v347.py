# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""SuperLocalMemory V3.4.7 "The Learning Brain" — Schema Extensions.

New tables for two-way learning:
  - tool_events: Passive tool usage telemetry
  - behavioral_assertions: Learned patterns with confidence evolution

Design rules (inherited):
  - CREATE IF NOT EXISTS for idempotency
  - profile_id where applicable
  - Never ALTER existing column types

Part of Qualixar | Author: Varun Pratap Bhardwaj
"""

from __future__ import annotations

import logging
import sqlite3

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# DDL — Tool Events (passive telemetry for learning)
# ---------------------------------------------------------------------------

_TOOL_EVENTS_DDL = """
CREATE TABLE IF NOT EXISTS tool_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    profile_id TEXT DEFAULT 'default',
    project_path TEXT DEFAULT '',
    tool_name TEXT NOT NULL,
    event_type TEXT NOT NULL DEFAULT 'invoke',
    input_summary TEXT DEFAULT '',
    output_summary TEXT DEFAULT '',
    duration_ms INTEGER DEFAULT 0,
    metadata TEXT DEFAULT '{}',
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_tool_events_tool
    ON tool_events(tool_name, created_at);
CREATE INDEX IF NOT EXISTS idx_tool_events_session
    ON tool_events(session_id);
CREATE INDEX IF NOT EXISTS idx_tool_events_project
    ON tool_events(project_path);
CREATE INDEX IF NOT EXISTS idx_tool_events_profile_created
    ON tool_events(profile_id, created_at DESC);
"""

# ---------------------------------------------------------------------------
# DDL — Behavioral Assertions (learned patterns with confidence)
# ---------------------------------------------------------------------------

_ASSERTIONS_DDL = """
CREATE TABLE IF NOT EXISTS behavioral_assertions (
    id TEXT PRIMARY KEY,
    profile_id TEXT NOT NULL DEFAULT 'default',
    project_path TEXT DEFAULT '',
    trigger_condition TEXT NOT NULL,
    action TEXT NOT NULL,
    category TEXT DEFAULT 'workflow',
    confidence REAL DEFAULT 0.3,
    evidence_fact_ids TEXT DEFAULT '[]',
    evidence_count INTEGER DEFAULT 1,
    reinforcement_count INTEGER DEFAULT 0,
    contradiction_count INTEGER DEFAULT 0,
    last_reinforced_at TEXT,
    last_contradicted_at TEXT,
    source TEXT DEFAULT 'auto',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_assertions_confidence
    ON behavioral_assertions(confidence DESC);
CREATE INDEX IF NOT EXISTS idx_assertions_project
    ON behavioral_assertions(project_path, profile_id);
CREATE INDEX IF NOT EXISTS idx_assertions_category
    ON behavioral_assertions(category);
CREATE INDEX IF NOT EXISTS idx_assertions_profile_confidence
    ON behavioral_assertions(profile_id, confidence DESC);
"""

# ---------------------------------------------------------------------------
# Migration runner
# ---------------------------------------------------------------------------

def apply_v347_schema(db_path: str | sqlite3.Connection) -> dict:
    """Apply all v3.4.7 schema changes. Idempotent — safe to call multiple times."""
    result = {"applied": [], "errors": []}

    if isinstance(db_path, sqlite3.Connection):
        conn = db_path
        own_connection = False
    else:
        conn = sqlite3.connect(str(db_path))
        own_connection = True

    try:
        # Tool events table
        try:
            conn.executescript(_TOOL_EVENTS_DDL)
            result["applied"].append("tool_events table + indexes")
        except sqlite3.OperationalError as e:
            result["errors"].append(f"tool_events: {e}")

        # Behavioral assertions table
        try:
            conn.executescript(_ASSERTIONS_DDL)
            result["applied"].append("behavioral_assertions table + indexes")
        except sqlite3.OperationalError as e:
            result["errors"].append(f"behavioral_assertions: {e}")

        # Mark version
        try:
            conn.execute(
                "INSERT OR IGNORE INTO schema_version (version, applied_at) VALUES (?, ?)",
                ("3.4.7", __import__("datetime").datetime.now().isoformat()),
            )
        except sqlite3.OperationalError:
            pass  # schema_version table might not exist yet

        conn.commit()

        if result["applied"]:
            logger.info("Schema v3.4.7 applied: %s", ", ".join(result["applied"]))

    except Exception as e:
        result["errors"].append(f"fatal: {e}")
        logger.error("Schema v3.4.7 migration failed: %s", e)
    finally:
        if own_connection:
            conn.close()

    return result
