# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""M079: each fact's origin class and taint, computed and kept (4.2.30, WS-5, M-15).

``fact_origin`` (M074) keeps the raw signals of how a fact was admitted. This
table keeps what ``trust/origin_class.py`` and ``trust/fact_trust.py`` make of
them: the fact's own class, its taint (the lowest class among the fact and
everything it was derived from) and ``basis``, the name of the mapping row that
decided the class, so a class can always be explained and a change to the
mapping can find the rows it affects. No memory text.

Written at admission when the flag ``trust.taint`` is on or in shadow, and by a
bounded background batch in the maintenance cycle; a read never writes. Erasing
a fact erases its row (I-11); a profile fold moves it with the fact.

CHECKs hold invariants only (non-empty), never a list of classes, which SQLite
could not extend after release (W0-29).
"""

from __future__ import annotations

import sqlite3

NAME = "M079_fact_trust"
DB_TARGET = "memory"
#: Additive: a new table an older build never reads.
DOWNGRADE_FLOOR = 51

DDL = """
CREATE TABLE IF NOT EXISTS fact_trust (
    fact_id      TEXT PRIMARY KEY,
    profile_id   TEXT NOT NULL,
    origin_class TEXT NOT NULL CHECK (length(origin_class) > 0),
    taint_class  TEXT NOT NULL CHECK (length(taint_class) > 0),
    basis        TEXT NOT NULL CHECK (length(basis) > 0),
    computed_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_fact_trust_taint ON fact_trust (profile_id, taint_class)
"""

_FACT_TRIGGER = """
CREATE TRIGGER IF NOT EXISTS trg_fact_trust_fact_erased
AFTER DELETE ON atomic_facts
BEGIN
    DELETE FROM fact_trust WHERE fact_id = OLD.fact_id;
END
"""


def _has_facts(conn: sqlite3.Connection) -> bool:
    return bool(conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='atomic_facts'").fetchone())


def apply(conn: sqlite3.Connection) -> None:
    for statement in DDL.split(";"):
        if statement.strip():
            conn.execute(statement)
    if _has_facts(conn):
        conn.execute(_FACT_TRIGGER)


def verify(conn: sqlite3.Connection) -> bool:
    names = {r[0] for r in conn.execute("SELECT name FROM sqlite_master")}
    if not {"fact_trust", "idx_fact_trust_taint"} <= names:
        return False
    return not _has_facts(conn) or "trg_fact_trust_fact_erased" in names


def repair(conn: sqlite3.Connection) -> None:
    apply(conn)
