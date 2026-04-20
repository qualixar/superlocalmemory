# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory v3.4.22

"""M012 — shadow_observations table (learning.db).

Persists the paired NDCG@10 observations accumulated by
``ShadowTest`` so a daemon restart during an in-flight shadow test
does NOT throw away the 6-hour observation window. Without this
table the candidate_id was re-attached on restart (Stage 9 W4
H-ARC-01) but the observation list restarted from zero.

Schema (additive only; no existing tables touched):

    shadow_observations(
      id            INTEGER PRIMARY KEY AUTOINCREMENT,
      profile_id    TEXT NOT NULL,
      candidate_id  INTEGER NOT NULL,
      query_id      TEXT NOT NULL,
      arm           TEXT NOT NULL CHECK (arm IN ('active','candidate')),
      ndcg_at_10    REAL NOT NULL,
      recorded_at   TEXT NOT NULL,
      UNIQUE(candidate_id, query_id, arm)
    )

The ``UNIQUE(candidate_id, query_id, arm)`` constraint means a
query_id is scored at most once per arm per candidate — if the
daemon crashes mid-write and restarts, re-ingesting the same
observation is an INSERT-OR-IGNORE no-op.

Author: Varun Pratap Bhardwaj / Qualixar
"""

from __future__ import annotations

import sqlite3

NAME = "M012_shadow_observations"
DB_TARGET = "learning"

_REQUIRED_TABLES = frozenset({"shadow_observations"})


def verify(conn: sqlite3.Connection) -> bool:
    try:
        names = {
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
    except sqlite3.Error:
        return False
    return _REQUIRED_TABLES.issubset(names)


DDL = """
CREATE TABLE IF NOT EXISTS shadow_observations (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    profile_id    TEXT NOT NULL,
    candidate_id  INTEGER NOT NULL,
    query_id      TEXT NOT NULL,
    arm           TEXT NOT NULL,
    ndcg_at_10    REAL NOT NULL,
    recorded_at   TEXT NOT NULL,
    UNIQUE(candidate_id, query_id, arm)
);
CREATE INDEX IF NOT EXISTS idx_shadow_obs_candidate
    ON shadow_observations(candidate_id, recorded_at);
CREATE INDEX IF NOT EXISTS idx_shadow_obs_profile
    ON shadow_observations(profile_id, recorded_at);
"""
