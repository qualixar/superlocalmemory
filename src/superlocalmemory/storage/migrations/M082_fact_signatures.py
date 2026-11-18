# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""M082: signed facts (4.2.30, WS-5, D-16): one HMAC per fact, kept beside the fact.

``trust/signatures.py`` signs ``(profile, fact id, sha256(content), created_at)`` with a
key that lives in the operating system's keychain and never in this file. This table
keeps the signature, which key made it (``key_id``, a digest of the key, never the key)
and when. No memory text.

``profile_id`` is where the fact is now; ``signed_profile_id`` is the profile the
signature covers. They differ only after a profile fold moved the fact (the fold's MOVE
changes ``profile_id`` and nothing else): the daemon then verifies the old signature over
``signed_profile_id`` and, if it holds, signs again for the new profile. A fold therefore
never turns a good signature into a "modified" one, and never launders a bad one.

Erasing a fact erases its row (I-11). CHECKs hold invariants only (non-empty), never a
list of keys or states, which SQLite could not extend after release (W0-29).
"""

from __future__ import annotations

import sqlite3

NAME = "M082_fact_signatures"
DB_TARGET = "memory"
#: Additive: a new table an older build never reads.
DOWNGRADE_FLOOR = 51

DDL = """
CREATE TABLE IF NOT EXISTS fact_signatures (
    fact_id           TEXT PRIMARY KEY,
    profile_id        TEXT NOT NULL,
    signed_profile_id TEXT NOT NULL,
    sig               TEXT NOT NULL CHECK (length(sig) > 0),
    key_id            TEXT NOT NULL CHECK (length(key_id) > 0),
    signed_at         TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_fact_signatures_key ON fact_signatures (key_id);
CREATE INDEX IF NOT EXISTS idx_fact_signatures_moved ON fact_signatures (profile_id, signed_profile_id)
"""

_FACT_TRIGGER = """
CREATE TRIGGER IF NOT EXISTS trg_fact_signatures_fact_erased
AFTER DELETE ON atomic_facts
BEGIN
    DELETE FROM fact_signatures WHERE fact_id = OLD.fact_id;
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
    if not {"fact_signatures", "idx_fact_signatures_key", "idx_fact_signatures_moved"} <= names:
        return False
    return not _has_facts(conn) or "trg_fact_signatures_fact_erased" in names


def repair(conn: sqlite3.Connection) -> None:
    apply(conn)
