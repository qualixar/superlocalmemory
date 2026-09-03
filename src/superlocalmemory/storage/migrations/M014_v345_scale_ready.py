# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory v3.4.5 — Scale-Ready

"""M014 — v3.4.5 Scale-Ready schema extensions (memory.db).

Adds:
  - atomic_facts.access_count_30d: rolling 30-day access window
  - idx_graph_edges_source_id / idx_graph_edges_target_id: bulk import perf (F-20)

Idempotent: ALTER TABLE ADD COLUMN with DEFAULT, CREATE INDEX IF NOT EXISTS.
Verify checks for access_count_30d column presence on atomic_facts.
"""

from __future__ import annotations

import sqlite3

NAME = "M014_v345_scale_ready"
DB_TARGET = "memory"

_REQUIRED_COLS = frozenset({"access_count_30d"})


def verify(conn: sqlite3.Connection) -> bool:
    try:
        cols = {
            r[1]
            for r in conn.execute(
                "PRAGMA table_info(atomic_facts)"
            ).fetchall()
        }
    except sqlite3.Error:
        return False
    return _REQUIRED_COLS.issubset(cols)


DDL = """
ALTER TABLE atomic_facts ADD COLUMN access_count_30d INTEGER DEFAULT 0;

CREATE INDEX IF NOT EXISTS idx_graph_edges_source_id
    ON graph_edges(source_id);
CREATE INDEX IF NOT EXISTS idx_graph_edges_target_id
    ON graph_edges(target_id);
"""


def repair(conn: sqlite3.Connection) -> None:
    """Re-apply missing end-state idempotently (4.1.14 #133)."""
    from superlocalmemory.storage.migrations._repair_util import repair_ddl
    repair_ddl(conn, DDL)
