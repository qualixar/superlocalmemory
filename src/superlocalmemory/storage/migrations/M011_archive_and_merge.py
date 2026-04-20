# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory v3.4.22 — LLD-00 §1.4 + LLD-12

"""M011 — archive + merge log (memory.db, DEFERRED).

LLD-12 real consolidation uses hnswlib to find near-duplicate atomic
facts, then archives or merges them reversibly. This migration:

- Extends ``atomic_facts`` with four lifecycle columns:
    * ``archive_status`` — 'live' | 'archived' | 'merged'
    * ``archive_reason`` — tag explaining why (cosine_dup, reward_gate, ...)
    * ``merged_into`` — fact_id of the canonical survivor when merged
    * ``retrieval_prior`` — reward-derived boost factor used by ranker
- Creates ``memory_archive`` — payload-preserving archive table.
  Consolidation NEVER deletes from atomic_facts; it only flips status
  and writes a payload snapshot here.
- Creates ``memory_merge_log`` — merge decisions are reversible via
  ``slm memory unmerge <merge_id>``. Records the canonical + duplicate
  fact_ids plus the cosine + jaccard scores that drove the merge.

Deferred like M006 — ``atomic_facts`` is bootstrapped at engine init,
not by the migration runner. DEFERRED_MIGRATIONS runs after
``MemoryEngine.initialize()``.
"""

from __future__ import annotations

import sqlite3

NAME = "M011_archive_and_merge"
DB_TARGET = "memory"

_REQUIRED_ATOMIC = frozenset({
    "archive_status", "archive_reason", "merged_into", "retrieval_prior",
})
_REQUIRED_TABLES = frozenset({"memory_archive", "memory_merge_log"})


def verify(conn: sqlite3.Connection) -> bool:
    try:
        atomic_cols = {
            r[1]
            for r in conn.execute(
                "PRAGMA table_info(atomic_facts)"
            ).fetchall()
        }
        names = {
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
    except sqlite3.Error:
        return False
    return _REQUIRED_ATOMIC.issubset(atomic_cols) and _REQUIRED_TABLES.issubset(names)


DDL = """
ALTER TABLE atomic_facts ADD COLUMN archive_status TEXT DEFAULT 'live';
ALTER TABLE atomic_facts ADD COLUMN archive_reason TEXT;
ALTER TABLE atomic_facts ADD COLUMN merged_into TEXT;
ALTER TABLE atomic_facts ADD COLUMN retrieval_prior REAL DEFAULT 0.0;

CREATE TABLE IF NOT EXISTS memory_archive (
    archive_id    TEXT PRIMARY KEY,
    fact_id       TEXT NOT NULL,
    profile_id    TEXT NOT NULL,
    payload_json  TEXT NOT NULL,
    archived_at   TEXT NOT NULL,
    reason        TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_archive_profile
    ON memory_archive(profile_id, archived_at);

CREATE TABLE IF NOT EXISTS memory_merge_log (
    merge_id          TEXT PRIMARY KEY,
    profile_id        TEXT NOT NULL,
    canonical_fact_id TEXT NOT NULL,
    merged_fact_id    TEXT NOT NULL,
    cosine_sim        REAL,
    entity_jaccard    REAL,
    merged_at         TEXT NOT NULL,
    reversible        INTEGER DEFAULT 1
);
CREATE INDEX IF NOT EXISTS idx_merge_profile ON memory_merge_log(profile_id);
"""
