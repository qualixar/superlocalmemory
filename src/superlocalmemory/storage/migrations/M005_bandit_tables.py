# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory v3.4.22 — LLD-07 §3.5

"""M005 — Thompson-sampling bandit tables (learning.db, LLD-03).

Two tables:
  - ``bandit_arms`` — per (profile, stratum, arm) Beta-distribution state.
    ``WITHOUT ROWID`` since primary key is a natural composite.
  - ``bandit_plays`` — individual play events with delayed-reward settlement.

All indexes are ``IF NOT EXISTS`` so reapply is a no-op.
"""

from __future__ import annotations

import sqlite3

NAME = "M005_bandit_tables"
DB_TARGET = "learning"


def verify(conn: sqlite3.Connection) -> bool:
    """Return True if bandit tables exist with expected columns."""
    try:
        arms_cols = {r[1] for r in conn.execute(
            "PRAGMA table_info(bandit_arms)"
        ).fetchall()}
        plays_cols = {r[1] for r in conn.execute(
            "PRAGMA table_info(bandit_plays)"
        ).fetchall()}
    except sqlite3.Error:
        return False
    return (
        {"profile_id", "stratum", "arm_id", "alpha",
         "beta", "plays"} <= arms_cols
        and {"play_id", "profile_id", "query_id", "stratum",
             "arm_id", "played_at"} <= plays_cols
    )

DDL = """
CREATE TABLE IF NOT EXISTS bandit_arms (
    profile_id      TEXT NOT NULL,
    stratum         TEXT NOT NULL,
    arm_id          TEXT NOT NULL,
    alpha           REAL NOT NULL DEFAULT 1.0,
    beta            REAL NOT NULL DEFAULT 1.0,
    plays           INTEGER NOT NULL DEFAULT 0,
    last_played_at  TEXT,
    PRIMARY KEY (profile_id, stratum, arm_id)
) WITHOUT ROWID;

CREATE INDEX IF NOT EXISTS idx_bandit_profile_strat
    ON bandit_arms(profile_id, stratum);

CREATE TABLE IF NOT EXISTS bandit_plays (
    play_id          INTEGER PRIMARY KEY AUTOINCREMENT,
    profile_id       TEXT NOT NULL,
    query_id         TEXT NOT NULL,
    stratum          TEXT NOT NULL,
    arm_id           TEXT NOT NULL,
    played_at        TEXT NOT NULL,
    reward           REAL,
    settled_at       TEXT,
    settlement_type  TEXT
);

CREATE INDEX IF NOT EXISTS idx_plays_query
    ON bandit_plays(query_id);
CREATE INDEX IF NOT EXISTS idx_plays_unsettled
    ON bandit_plays(profile_id, played_at)
    WHERE settled_at IS NULL;
CREATE INDEX IF NOT EXISTS idx_plays_retention
    ON bandit_plays(settled_at);
"""
