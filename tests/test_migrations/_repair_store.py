# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""A temporary store as 4.2.7 left it, for the 4.2.8 automatic data repairs.

The store is fully migrated, then the repair under test is struck from the
migration log, so the next ``apply_deferred`` is exactly the first start of
4.2.8 on that store: no command, just the daemon starting.
"""

from __future__ import annotations

import json
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
             created_at: str = "2026-05-10T07:14:22+00:00", source: str | None = "raw text",
             session_date: str | None = None, profile: str = "default", **cols) -> None:
    if source is not None:  # None: the source memory is gone
        conn.execute("INSERT OR IGNORE INTO memories (memory_id, profile_id, content, "
                     "session_date, created_at) VALUES (?, ?, ?, ?, ?)",
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


def add_ingestion(conn: sqlite3.Connection, fact_id: str, raw: str, *,
                  linked_by: str = "lineage") -> None:
    """An ingestion record holding the raw text ``fact_id`` was extracted from,
    linked by derivation lineage or by the operation's own fact list."""
    op = f"op-{fact_id}"
    facts = json.dumps([fact_id]) if linked_by == "fact_list" else "[]"
    conn.execute("INSERT INTO ingestion_operations (operation_id, profile_id, source_type, "
                 "idempotency_key, source_hash, raw_content, final_fact_ids_json) "
                 "VALUES (?, 'default', 'store', ?, 'h', ?, ?)", (op, op, raw, facts))
    if linked_by == "lineage":
        conn.execute("INSERT INTO derivation_lineage (lineage_id, profile_id, object_type, "
                     "object_id, operation_id, source_status) "
                     "VALUES (?, 'default', 'fact', ?, ?, 'exact')", (f"l-{fact_id}", fact_id, op))


def first_start(memory: Path, learning: Path) -> dict:
    return apply_deferred(learning, memory)


def fts_ids(conn: sqlite3.Connection, phrase: str) -> set[str]:
    return {r[0] for r in conn.execute(
        "SELECT fact_id FROM atomic_facts_fts WHERE atomic_facts_fts MATCH ?",
        (f'"{phrase}"',))}
