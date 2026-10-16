# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""A memory.db with just the tables the hook event path touches."""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

from superlocalmemory.storage.schema_v347 import _TOOL_EVENTS_DDL

_OUTCOMES_DDL = """
CREATE TABLE IF NOT EXISTS action_outcomes (
    outcome_id TEXT PRIMARY KEY, profile_id TEXT NOT NULL DEFAULT 'default',
    query TEXT NOT NULL DEFAULT '', fact_ids_json TEXT NOT NULL DEFAULT '[]',
    outcome TEXT NOT NULL DEFAULT '', context_json TEXT NOT NULL DEFAULT '{}',
    timestamp TEXT NOT NULL DEFAULT (datetime('now')), reward REAL,
    settled INTEGER NOT NULL DEFAULT 0, settled_at TEXT, recall_query_id TEXT);
CREATE TABLE IF NOT EXISTS pending_outcomes (
    outcome_id TEXT PRIMARY KEY, profile_id TEXT NOT NULL, session_id TEXT NOT NULL,
    recall_query_id TEXT NOT NULL, fact_ids_json TEXT NOT NULL,
    query_text_hash TEXT NOT NULL, created_at_ms INTEGER NOT NULL,
    expires_at_ms INTEGER NOT NULL, signals_json TEXT NOT NULL DEFAULT '{}',
    status TEXT NOT NULL DEFAULT 'pending');
"""


def bootstrap(db: Path) -> Path:
    conn = sqlite3.connect(db)
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.executescript(_TOOL_EVENTS_DDL + _OUTCOMES_DDL)
        conn.commit()
    finally:
        conn.close()
    return db


def seed_pending(db: Path, *, outcome_id: str, session_id: str, fact_ids: list[str],
                 signals: dict | None = None, expires_in_ms: int = 60_000,
                 profile_id: str = "default") -> None:
    now = int(time.time() * 1000)
    conn = sqlite3.connect(db)
    try:
        conn.execute(
            "INSERT INTO pending_outcomes (outcome_id, profile_id, session_id, "
            "recall_query_id, fact_ids_json, query_text_hash, created_at_ms, "
            "expires_at_ms, signals_json, status) VALUES (?,?,?,?,?,?,?,?,?, 'pending')",
            (outcome_id, profile_id, session_id, "q-1", json.dumps(fact_ids), "0" * 64,
             now, now + expires_in_ms, json.dumps(signals or {})),
        )
        conn.commit()
    finally:
        conn.close()


def query(db: Path, sql: str, params: tuple = ()) -> list[dict]:
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in conn.execute(sql, params)]
    finally:
        conn.close()
