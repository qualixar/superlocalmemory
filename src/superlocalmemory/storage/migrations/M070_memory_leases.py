# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""M070: memory leases, so a memory knows when the file it came from changed.

A fact derived from a file, a folder source, a git object or a URL holds a
lease on that source: its locator and the fingerprint it had when the fact was
made (sha256 + size + mtime for files, the blob id for git, ETag or
Last-Modified for URLs). Recall checks the leases of the facts it is about to
deliver (a ``stat`` call), and a nightly sweep checks the rest. A changed or
missing source marks the lease and lowers the fact; it never deletes anything
on its own (``truth/leases.py``).

One additive table. A trigger removes a fact's leases when the fact is erased.
"""

from __future__ import annotations

import sqlite3

NAME = "M070_memory_leases"
DB_TARGET = "memory"
#: Additive (one new table and one trigger that writes only to it).
DOWNGRADE_FLOOR = 51

DDL = """
CREATE TABLE IF NOT EXISTS memory_leases (
    fact_id      TEXT NOT NULL,
    profile_id   TEXT NOT NULL,
    source_kind  TEXT NOT NULL,
    locator      TEXT NOT NULL,
    fingerprint  TEXT NOT NULL,
    size_bytes   INTEGER,
    mtime_ns     INTEGER,
    state        TEXT NOT NULL DEFAULT 'fresh',
    leased_at    TEXT NOT NULL,
    checked_at   TEXT NOT NULL,
    changed_at   TEXT,
    PRIMARY KEY (fact_id, source_kind, locator)
);
CREATE INDEX IF NOT EXISTS idx_memory_leases_profile_state
    ON memory_leases (profile_id, state, checked_at);
CREATE INDEX IF NOT EXISTS idx_memory_leases_locator ON memory_leases (locator)
"""

_FACT_TRIGGER = """
CREATE TRIGGER IF NOT EXISTS trg_lease_fact_erased
AFTER DELETE ON atomic_facts
BEGIN
    DELETE FROM memory_leases WHERE fact_id = OLD.fact_id;
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
    if not {"memory_leases", "idx_memory_leases_profile_state"} <= names:
        return False
    return not _has_facts(conn) or "trg_lease_fact_erased" in names


def repair(conn: sqlite3.Connection) -> None:
    apply(conn)
