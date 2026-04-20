# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory v3.4.21

"""M013 — bi-temporal columns on ``atomic_facts`` (memory.db, deferred).

Adds two NULLable columns that let SLM capture fact validity windows
WITHOUT changing the retrieval path today. Wiring the retrieval
consumer is a later-cycle item; what we ship in v3.4.21 is the
data-capture surface so existing user memories start accumulating
temporal metadata on every new fact.

Columns (both nullable, no default so existing rows stay NULL):

    valid_from     TEXT  — ISO-8601 instant from which the fact is
                           considered valid. NULL ⇒ "valid since
                           creation" (the current default semantics).
    valid_until    TEXT  — ISO-8601 instant after which the fact is
                           considered superseded. NULL ⇒ still valid.

Deferred like M006 and M011 because ``atomic_facts`` is bootstrapped
at engine init, not at migration time. Daemon lifespan calls
``apply_deferred`` right after engine init so these columns materialise
on first boot after upgrade.

Author: Varun Pratap Bhardwaj / Qualixar
"""

from __future__ import annotations

import sqlite3

NAME = "M013_bi_temporal_columns"
DB_TARGET = "memory"

_REQUIRED_COLS = frozenset({"valid_from", "valid_until"})


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
ALTER TABLE atomic_facts ADD COLUMN valid_from TEXT;
ALTER TABLE atomic_facts ADD COLUMN valid_until TEXT;
"""
