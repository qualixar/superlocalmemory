# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Pending memory store — zero-loss async remember (Option C).

Problem: v3.3.20 async `slm remember` spawns a detached subprocess with
stderr=DEVNULL. If the embedding worker crashes, the user's data is silently
lost — they see "Queued for background processing" but the memory never stores.

Solution: Store-first, embed-later (Netflix pattern).
  1. INSERT raw content into pending_memories (synchronous, 0.1s, no engine init)
  2. Spawn background subprocess to process (extract facts, embed, build graph)
  3. If background crashes, content survives in pending table
  4. Next engine.initialize() auto-retries pending items

Uses a separate `pending.db` file — never touches memory.db directly.
Stdlib only — no SLM imports (must be fast).

Part of Qualixar | Author: Varun Pratap Bhardwaj
License: AGPL-3.0-or-later
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path


def _default_dir() -> Path:
    """Resolve the pending queue inside the canonical process namespace."""
    from superlocalmemory.infra.data_root import canonical_data_root
    return canonical_data_root()


_PENDING_DB = "pending.db"
_MAX_RETRIES = 3
_STUCK_DAYS = 7
_MAX_RETRY_DELAY_SECONDS = 3600

_SCHEMA = """
CREATE TABLE IF NOT EXISTS pending_memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    content TEXT NOT NULL,
    tags TEXT DEFAULT '',
    metadata TEXT DEFAULT '{}',
    created_at TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    error TEXT DEFAULT NULL,
    retry_count INTEGER DEFAULT 0,
    next_retry_at REAL DEFAULT 0
);
"""


def _get_db(base_dir: Path | None = None) -> sqlite3.Connection:
    """Open pending.db with WAL mode. Creates if needed."""
    d = base_dir or _default_dir()
    d.mkdir(parents=True, exist_ok=True)
    db_path = d / _PENDING_DB
    conn = sqlite3.connect(str(db_path), timeout=5)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(_SCHEMA)
    columns = {
        row[1]
        for row in conn.execute("PRAGMA table_info(pending_memories)").fetchall()
    }
    if "next_retry_at" not in columns:
        conn.execute(
            "ALTER TABLE pending_memories ADD COLUMN "
            "next_retry_at REAL DEFAULT 0"
        )
        conn.commit()
    # Pre-V3.7 rows were terminally hidden after three failures. Restore them
    # to the retry queue; M018 makes replay idempotent and no raw evidence may
    # remain stranded solely because an older version exhausted its counter.
    conn.execute(
        "UPDATE pending_memories SET status='pending', next_retry_at=0 "
        "WHERE status='failed'"
    )
    conn.commit()
    return conn


def store_pending(
    content: str,
    tags: str = "",
    metadata: dict | None = None,
    base_dir: Path | None = None,
) -> int:
    """Store content in pending table. Returns the row ID.

    This is intentionally FAST — no engine init, no embedding, no model loading.
    Just a raw SQLite INSERT (~0.1s).
    """
    conn = _get_db(base_dir)
    try:
        cur = conn.execute(
            "INSERT INTO pending_memories (content, tags, metadata, created_at, status) "
            "VALUES (?, ?, ?, ?, 'pending')",
            (content, tags, json.dumps(metadata or {}), time.strftime("%Y-%m-%dT%H:%M:%S")),
        )
        conn.commit()
        return cur.lastrowid or 0
    finally:
        conn.close()


def get_pending(base_dir: Path | None = None, limit: int = 50) -> list[dict]:
    """Get unprocessed pending memories."""
    conn = _get_db(base_dir)
    try:
        rows = conn.execute(
            "SELECT id, content, tags, metadata, created_at, retry_count "
            "FROM pending_memories WHERE status = 'pending' "
            "AND COALESCE(next_retry_at, 0) <= ? "
            "ORDER BY id ASC LIMIT ?",
            (time.time(), limit),
        ).fetchall()
        return [
            {"id": r[0], "content": r[1], "tags": r[2], "metadata": r[3],
             "created_at": r[4], "retry_count": r[5]}
            for r in rows
        ]
    finally:
        conn.close()


def mark_done(row_id: int, base_dir: Path | None = None) -> None:
    """Mark a pending memory as successfully processed."""
    conn = _get_db(base_dir)
    try:
        conn.execute(
            "UPDATE pending_memories SET status = 'done' WHERE id = ?",
            (row_id,),
        )
        conn.commit()
    finally:
        conn.close()


def mark_failed(row_id: int, error: str, base_dir: Path | None = None) -> None:
    """Mark a pending memory as failed with error message.

    Unprocessed evidence is never deleted or terminally hidden. Failures stay
    pending with bounded exponential backoff; M018 idempotency makes repeated
    replay safe once the canonical operation has been created.
    """
    conn = _get_db(base_dir)
    try:
        row = conn.execute(
            "SELECT retry_count FROM pending_memories WHERE id=?",
            (row_id,),
        ).fetchone()
        if row is None:
            return
        next_count = int(row[0] or 0) + 1
        delay = 0 if next_count == 1 else min(
            2 ** min(next_count - 1, 12),
            _MAX_RETRY_DELAY_SECONDS,
        )
        conn.execute(
            "UPDATE pending_memories SET error = ?, "
            "retry_count = retry_count + 1, "
            "status = 'pending', next_retry_at = ? "
            "WHERE id = ?",
            (error, time.time() + delay, row_id),
        )
        conn.commit()
    finally:
        conn.close()


def pending_count(base_dir: Path | None = None) -> int:
    """Count unprocessed pending memories."""
    d = base_dir or _default_dir()
    db_path = d / _PENDING_DB
    if not db_path.exists():
        return 0
    conn = sqlite3.connect(str(db_path), timeout=5)
    try:
        return conn.execute(
            "SELECT COUNT(*) FROM pending_memories WHERE status = 'pending'"
        ).fetchone()[0]
    except sqlite3.OperationalError:
        return 0
    finally:
        conn.close()


def cleanup_done(days: int = 7, base_dir: Path | None = None) -> int:
    """Remove processed entries older than N days."""
    conn = _get_db(base_dir)
    try:
        cur = conn.execute(
            "DELETE FROM pending_memories WHERE status = 'done' "
            "AND created_at < datetime('now', ?)",
            (f"-{days} days",),
        )
        conn.commit()
        return cur.rowcount
    finally:
        conn.close()


def cleanup_stale(base_dir: Path | None = None) -> dict[str, int]:
    """Sweep stale rows from pending.db. Runs periodically from the daemon.

    Deletes only completed receipts. Pending and failed raw evidence is retained
    regardless of age so maintenance can never become a data-loss path.
    """
    conn = _get_db(base_dir)
    try:
        done = conn.execute(
            "DELETE FROM pending_memories WHERE status = 'done' "
            "AND created_at < datetime('now', ?)",
            (f"-{_STUCK_DAYS} days",),
        ).rowcount
        retained = conn.execute(
            "SELECT COUNT(*) FROM pending_memories "
            "WHERE status != 'done'"
        ).fetchone()[0]
        conn.commit()
        return {
            "done": done,
            "failed_over_retries": 0,
            "stuck_pending": 0,
            "hard_cap_expired": 0,
            "retained_unprocessed": int(retained),
            "total": done,
        }
    finally:
        conn.close()
