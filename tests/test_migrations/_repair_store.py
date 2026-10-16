# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""A temporary store as 4.2.7 left it, for the 4.2.8 automatic data repairs.

The store is fully migrated, then the repair under test is struck from the
migration log, so the next ``apply_deferred`` is exactly the first start of
4.2.8 on that store: no command, just the daemon starting.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from superlocalmemory.storage import schema
from superlocalmemory.storage.database import DatabaseManager
from superlocalmemory.storage.migration_runner import apply_all, apply_deferred


def migrated_store(tmp_path: Path) -> tuple[Path, Path]:
    memory, learning = tmp_path / "memory.db", tmp_path / "learning.db"
    manager = DatabaseManager(memory)
    manager.initialize(schema)
    manager.close()
    apply_all(learning, memory)
    apply_deferred(learning, memory)
    return memory, learning


def forget_migration(memory: Path, name: str) -> None:
    """Make the store look like one that has not run ``name`` yet."""
    conn = sqlite3.connect(memory)
    conn.execute("DELETE FROM migration_log WHERE name = ?", (name,))
    conn.commit()
    conn.close()


def add_fact(conn: sqlite3.Connection, fact_id: str, content: str, *,
             created_at: str = "2026-05-10T07:14:22+00:00", source: str = "raw text",
             session_date: str | None = None, profile: str = "default", **cols) -> None:
    conn.execute("INSERT OR IGNORE INTO memories (memory_id, profile_id, content, session_date, "
                 "created_at) VALUES (?, ?, ?, ?, ?)",
                 (f"m-{fact_id}", profile, source, session_date, created_at))
    names = ["fact_id", "memory_id", "profile_id", "content", "created_at", *cols]
    values = [fact_id, f"m-{fact_id}", profile, content, created_at, *cols.values()]
    conn.execute(f"INSERT INTO atomic_facts ({','.join(names)}) "  # noqa: S608
                 f"VALUES ({','.join('?' * len(names))})", values)


def add_event(conn: sqlite3.Connection, event_id: str, fact_id: str, content: str, *,
              referenced: str | None = None, observed: str | None = None) -> None:
    conn.execute("INSERT INTO temporal_events (event_id, profile_id, entity_id, fact_id, "
                 "observation_date, referenced_date, description) "
                 "VALUES (?, 'default', 'e1', ?, ?, ?, ?)",
                 (event_id, fact_id, observed, referenced, content))


def first_start(memory: Path, learning: Path) -> dict:
    return apply_deferred(learning, memory)


def fts_ids(conn: sqlite3.Connection, phrase: str) -> set[str]:
    return {r[0] for r in conn.execute(
        "SELECT fact_id FROM atomic_facts_fts WHERE atomic_facts_fts MATCH ?",
        (f'"{phrase}"',))}
