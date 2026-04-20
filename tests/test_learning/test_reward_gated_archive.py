# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory v3.4.22 — Track B.2 (LLD-12 §4)

"""Defence-in-depth tests for reward-gated Ebbinghaus archive flow.

The manifest's three archive-related tests live in ``test_hnsw_dedup``
(``test_reward_gated_archive_*``). This file adds two extra coverage
tests for edge cases the LLD-12 §4 contract declares explicitly.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest


def _bootstrap(path: Path) -> None:
    with sqlite3.connect(path) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.executescript(
            """
            CREATE TABLE atomic_facts (
                fact_id            TEXT PRIMARY KEY,
                profile_id         TEXT NOT NULL,
                content            TEXT NOT NULL,
                canonical_entities_json TEXT NOT NULL DEFAULT '[]',
                embedding          TEXT,
                confidence         REAL NOT NULL DEFAULT 1.0,
                importance         REAL NOT NULL DEFAULT 0.5,
                created_at         TEXT NOT NULL DEFAULT (datetime('now')),
                archive_status     TEXT DEFAULT 'live',
                archive_reason     TEXT,
                merged_into        TEXT,
                retrieval_prior    REAL DEFAULT 0.0
            );
            CREATE TABLE memory_archive (
                archive_id    TEXT PRIMARY KEY,
                fact_id       TEXT NOT NULL,
                profile_id    TEXT NOT NULL,
                payload_json  TEXT NOT NULL,
                archived_at   TEXT NOT NULL,
                reason        TEXT NOT NULL
            );
            CREATE TABLE action_outcomes (
                outcome_id       TEXT PRIMARY KEY,
                profile_id       TEXT NOT NULL DEFAULT 'default',
                fact_ids_json    TEXT NOT NULL DEFAULT '[]',
                reward           REAL,
                settled          INTEGER NOT NULL DEFAULT 0,
                settled_at       TEXT
            );
            """
        )


@pytest.fixture
def mem_db(tmp_path: Path) -> Path:
    p = tmp_path / "memory.db"
    _bootstrap(p)
    return p


def test_archive_never_deletes_from_atomic_facts(mem_db: Path) -> None:
    """LLD-12 §1 — archive must UPDATE status, not DELETE row."""
    from superlocalmemory.learning.hnsw_dedup import run_reward_gated_archive

    with sqlite3.connect(mem_db) as conn:
        conn.execute(
            "INSERT INTO atomic_facts (fact_id, profile_id, content) "
            "VALUES ('cold', 'p1', 'stale fact')"
        )

    # Install a SQLite authorizer that explodes on DELETE FROM atomic_facts.
    conn = sqlite3.connect(mem_db)

    def _authorizer(code, arg1, arg2, arg3, arg4):
        if code == sqlite3.SQLITE_DELETE and arg1 == "atomic_facts":
            return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK

    conn.set_authorizer(_authorizer)
    conn.close()

    run_reward_gated_archive(mem_db, "p1", candidate_fact_ids=["cold"])

    conn = sqlite3.connect(mem_db)
    row = conn.execute(
        "SELECT archive_status, archive_reason FROM atomic_facts "
        "WHERE fact_id='cold'"
    ).fetchone()
    archive_row = conn.execute(
        "SELECT COUNT(*) FROM memory_archive WHERE fact_id='cold'"
    ).fetchone()[0]
    conn.close()

    assert row is not None, "row must still exist (never deleted)"
    assert row[0] == "archived"
    assert row[1]
    assert archive_row == 1


def test_archive_writes_payload_preserving_snapshot(mem_db: Path) -> None:
    """Archive row MUST include a payload snapshot that could rehydrate."""
    from superlocalmemory.learning.hnsw_dedup import run_reward_gated_archive

    with sqlite3.connect(mem_db) as conn:
        conn.execute(
            "INSERT INTO atomic_facts "
            "(fact_id, profile_id, content, canonical_entities_json) "
            "VALUES ('x', 'p1', 'payload text', ?)",
            (json.dumps(["ent_1", "ent_2"]),)
        )

    run_reward_gated_archive(mem_db, "p1", candidate_fact_ids=["x"])

    conn = sqlite3.connect(mem_db)
    raw = conn.execute(
        "SELECT payload_json FROM memory_archive WHERE fact_id='x'"
    ).fetchone()
    conn.close()
    assert raw is not None
    blob = json.loads(raw[0])
    assert blob.get("content") == "payload text"
    assert "ent_1" in json.dumps(blob)
