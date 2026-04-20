# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory v3.4.22 — LLD-07 §3.6

"""M006 — action_outcomes reward + settlement columns (memory.db).

Extends ``action_outcomes`` with the four columns the learning trainer
needs in order to swap off the position proxy onto the real reward
label:

  * ``reward REAL`` — numeric reward (usually in [-1, 1]).
  * ``settled INTEGER DEFAULT 0`` — 1 when the outcome has been settled.
  * ``settled_at TEXT`` — ISO-8601 timestamp of settlement.
  * ``recall_query_id TEXT`` — links the outcome back to the recall that
    produced the candidate facts (so the trainer can join against
    ``learning_signals.query_id``).

The gate in ``learning.database.fetch_training_examples``
(``_migration_applied("M006_action_outcomes_reward")``) already falls
back to the position proxy when this migration hasn't completed, so a
skipped/failed apply never crashes the trainer.

Deferred: this migration is NOT in ``MIGRATIONS``. It ships via
``DEFERRED_MIGRATIONS`` and runs from the daemon lifespan immediately
after ``MemoryEngine.initialize()`` has bootstrapped the
``action_outcomes`` table. Idempotent — ``verify(conn)`` returns True
once all four columns are present, so reapply is a no-op.
"""

from __future__ import annotations

import sqlite3

NAME = "M006_action_outcomes_reward"
# action_outcomes lives in memory.db (see storage.schema).
DB_TARGET = "memory"

_REQUIRED_COLS = frozenset({"reward", "settled", "settled_at", "recall_query_id"})


def verify(conn: sqlite3.Connection) -> bool:
    """Return True once every M006 column is present on ``action_outcomes``.

    Lets the migration runner detect "already applied" state when a retry
    would otherwise hit a duplicate-column error mid-script.
    """
    try:
        cols = {
            r[1]
            for r in conn.execute("PRAGMA table_info(action_outcomes)").fetchall()
        }
    except sqlite3.Error:
        return False
    return _REQUIRED_COLS.issubset(cols)


# DDL. Runs inside an explicit transaction opened by the migration runner.
# SQLite doesn't support ``ALTER TABLE IF EXISTS`` — if the table is missing
# this raises sqlite3.OperationalError which the runner catches and records
# as ``failed``; the gate in ``fetch_training_examples`` keeps the trainer
# on the position proxy in that case, so the daemon is still healthy.
DDL = """
ALTER TABLE action_outcomes ADD COLUMN reward REAL;
ALTER TABLE action_outcomes ADD COLUMN settled INTEGER NOT NULL DEFAULT 0;
ALTER TABLE action_outcomes ADD COLUMN settled_at TEXT;
ALTER TABLE action_outcomes ADD COLUMN recall_query_id TEXT;

CREATE INDEX IF NOT EXISTS idx_action_outcomes_settled_reward
    ON action_outcomes(settled, settled_at)
    WHERE reward IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_action_outcomes_recall_query
    ON action_outcomes(recall_query_id)
    WHERE recall_query_id IS NOT NULL;
"""
