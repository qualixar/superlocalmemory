# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory v3.4.22 — LLD-00 §1.6 + LLD-11

"""M010 — evolution_config + evolution_llm_cost_log tables (learning.db).

LLD-11 skill evolution is opt-in by default (MASTER-PLAN D3). This
migration creates the two tables the evolution subsystem needs:

- ``evolution_config`` — per-profile feature flag + LLM backend choice.
  The default row is created lazily by the installer; this migration
  only creates the table so the installer's INSERT can land.
- ``evolution_llm_cost_log`` — every LLM call the evolution cycle makes
  is logged here so the cost-accounting widget on the dashboard can
  report tokens + USD spend.

Target DB: learning.db. Additive only.

SEC-L2 — ``cost_usd`` is IEEE-754 REAL (double). This loses precision
when summing thousands of sub-cent rows (rounding drift < 0.5 ¢ per
10k rows in practice). The schema is additive and therefore locked
for v3.4.22 — dashboards MUST compute aggregate cost as
``SUM(cost_usd)`` with explicit ``ROUND(x, 4)`` at display time, and
MUST NOT branch on sub-cent equality. A follow-on migration is
scheduled to switch the column to INTEGER millicents (see FINAL
board). The ``cost_usd >= 0`` non-negativity invariant is enforced
application-side by ``evolution.llm_dispatch._log_cost``.
"""

from __future__ import annotations

import sqlite3

NAME = "M010_evolution_config"
DB_TARGET = "learning"

_REQUIRED_TABLES = frozenset({"evolution_config", "evolution_llm_cost_log"})


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
CREATE TABLE IF NOT EXISTS evolution_config (
    profile_id        TEXT PRIMARY KEY,
    enabled           INTEGER NOT NULL DEFAULT 0,
    llm_backend       TEXT NOT NULL DEFAULT 'haiku',
    llm_model         TEXT NOT NULL DEFAULT 'claude-haiku-4-5',
    last_cycle_at     TEXT,
    cycles_this_week  INTEGER NOT NULL DEFAULT 0,
    disabled_until    TEXT
);
CREATE TABLE IF NOT EXISTS evolution_llm_cost_log (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    profile_id    TEXT NOT NULL,
    ts            TEXT NOT NULL,
    model         TEXT NOT NULL,
    tokens_in     INTEGER NOT NULL DEFAULT 0,
    tokens_out    INTEGER NOT NULL DEFAULT 0,
    cost_usd      REAL NOT NULL DEFAULT 0.0,
    cycle_id      TEXT
);
CREATE INDEX IF NOT EXISTS idx_cost_profile_ts
    ON evolution_llm_cost_log(profile_id, ts);
"""
