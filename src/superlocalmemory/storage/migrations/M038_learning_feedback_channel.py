# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Ported into the SuperLocalMemory V4 migration sequence.

"""M038 — add the ``channel`` column to ``learning_feedback``.

``pattern_miner._mine_channel_and_coretrieval`` has always executed::

    SELECT channel, COUNT(*) AS cnt, AVG(signal_value) AS avg_signal
    FROM learning_feedback GROUP BY channel

but ``channel`` was never defined on the table. Every database therefore
raised ``sqlite3.OperationalError: no such column: channel``. That error was
caught by the miner's outer ``except Exception`` and logged at DEBUG, so the
failure was invisible — and because the channel query runs FIRST inside that
try block, it also aborted the co-retrieval mining below it. One missing
column silently disabled two pattern types (issue #102, "Patterns learned
remains 0" despite a restored backup: restoring rows cannot fix a schema gap).

Additive only — ``ALTER TABLE ADD COLUMN`` with a default. No data loss and no
type changes; existing rows get ``'unknown'``, which groups cleanly rather than
being dropped by the miner's ``GROUP BY``.

``learning_feedback`` is bootstrapped at runtime by
``learning.feedback.FeedbackCollector._ensure_schema`` rather than by a
migration, so the DDL below CREATEs it (without ``channel``) when absent
before the ALTER. That keeps all three states correct:

  - table missing            -> CREATE, then ALTER adds ``channel``
  - table present, no column -> CREATE is a no-op, ALTER adds ``channel``
  - table already migrated   -> ``verify()`` returns True, runner skips entirely
"""

from __future__ import annotations

import sqlite3

NAME = "M038_learning_feedback_channel"
DB_TARGET = "learning"


def verify(conn: sqlite3.Connection) -> bool:
    """Return True if ``learning_feedback.channel`` already exists."""
    try:
        cols = {
            row[1] for row in
            conn.execute("PRAGMA table_info(learning_feedback)").fetchall()
        }
    except sqlite3.Error:
        return False
    # An empty set means the table does not exist yet — not migrated.
    return bool(cols) and "channel" in cols


DDL = """
BEGIN IMMEDIATE;

CREATE TABLE IF NOT EXISTS learning_feedback (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    profile_id   TEXT    NOT NULL,
    fact_id      TEXT    NOT NULL,
    signal_type  TEXT    NOT NULL,
    signal_value REAL    NOT NULL,
    query_hash   TEXT,
    created_at   TEXT    NOT NULL,
    metadata     TEXT
);

ALTER TABLE learning_feedback ADD COLUMN channel TEXT DEFAULT 'unknown';

CREATE INDEX IF NOT EXISTS idx_feedback_profile
    ON learning_feedback (profile_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_feedback_channel
    ON learning_feedback (profile_id, channel);

COMMIT;
"""
