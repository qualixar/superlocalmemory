# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory v3.4.22 — Track B.2 (LLD-12)

"""Placeholder companion test for ``memory_merge.apply_merges`` + ``unmerge``.

The canonical manifest tests live in ``test_hnsw_dedup.py`` (see
``test_merge_*``). This file adds small defence-in-depth coverage for
the merge log writer's behaviour under duplicate candidates.
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
            CREATE TABLE memory_merge_log (
                merge_id          TEXT PRIMARY KEY,
                profile_id        TEXT NOT NULL,
                canonical_fact_id TEXT NOT NULL,
                merged_fact_id    TEXT NOT NULL,
                cosine_sim        REAL,
                entity_jaccard    REAL,
                merged_at         TEXT NOT NULL,
                reversible        INTEGER DEFAULT 1
            );
            """
        )


@pytest.fixture
def mem_db(tmp_path: Path) -> Path:
    p = tmp_path / "memory.db"
    _bootstrap(p)
    with sqlite3.connect(p) as conn:
        conn.execute(
            "INSERT INTO atomic_facts (fact_id, profile_id, content) "
            "VALUES ('a', 'p1', 'left'), ('b', 'p1', 'right')"
        )
    return p


def test_apply_merges_idempotent_on_already_merged(mem_db: Path) -> None:
    from superlocalmemory.learning.memory_merge import apply_merges

    candidates = [("a", "b", 0.99, 0.95)]
    first = apply_merges(mem_db, candidates, profile_id="p1")
    second = apply_merges(mem_db, candidates, profile_id="p1")
    assert first == 1
    assert second == 0  # skipped: 'b' already merged


def test_unmerge_unknown_id_returns_false(mem_db: Path) -> None:
    from superlocalmemory.learning.memory_merge import unmerge

    assert unmerge(mem_db, "no-such-merge") is False
