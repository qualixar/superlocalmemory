# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""M069: slots, so SLM knows which memory is the current value of something.

A slot is "the value of relation R for subject S in scope X": the release
cadence of project SLM, the voice model Alex prefers. Each fact bound to a
slot is one immutable revision. A single-valued slot has one head; a new
revision either supersedes it (through the correction ledger, so it can be
rolled back exactly), joins it as compatible, repeats it (duplicate), or opens a
dispute that shows both sides until someone resolves it.

Three additive tables (``truth/slot_store.py`` owns them). Ids, hashes and
controlled codes only: a revision points at its fact and never copies its
text. Triggers keep erasure complete: deleting a fact removes its revisions,
and removing a revision removes the disputes that name it and empties the head
it held, so no slot can point at a memory that is gone.
"""

from __future__ import annotations

import sqlite3

NAME = "M069_memory_slots"
DB_TARGET = "memory"
#: Additive (three new tables and their triggers; an older build never reads
#: them, and the triggers write only to them), so going back to 4.1.18 stays
#: possible.
DOWNGRADE_FLOOR = 51

DDL = """
CREATE TABLE IF NOT EXISTS memory_slots (
    slot_id          TEXT PRIMARY KEY,
    profile_id       TEXT NOT NULL,
    scope            TEXT NOT NULL,
    subject_key      TEXT NOT NULL,
    relation_key     TEXT NOT NULL,
    qualifier        TEXT NOT NULL DEFAULT '',
    cardinality      TEXT NOT NULL DEFAULT 'single'
                     CHECK (cardinality IN ('single', 'plural')),
    key_source       TEXT NOT NULL
                     CHECK (key_source IN ('explicit', 'pattern', 'pair', 'judge')),
    head_revision_id TEXT,
    state            TEXT NOT NULL DEFAULT 'empty'
                     CHECK (state IN ('current', 'disputed', 'empty')),
    version          INTEGER NOT NULL DEFAULT 0 CHECK (version >= 0),
    created_at       TEXT NOT NULL,
    updated_at       TEXT NOT NULL,
    UNIQUE (profile_id, scope, subject_key, relation_key, qualifier)
);
CREATE TABLE IF NOT EXISTS slot_revisions (
    revision_id  TEXT PRIMARY KEY,
    slot_id      TEXT NOT NULL,
    profile_id   TEXT NOT NULL,
    fact_id      TEXT NOT NULL,
    value_hash   TEXT NOT NULL,
    state        TEXT NOT NULL
                 CHECK (state IN ('head', 'superseded', 'compatible', 'duplicate',
                                  'disputed', 'historical')),
    superseded_by TEXT,
    decided_by   TEXT NOT NULL,
    reason_code  TEXT NOT NULL,
    case_id      TEXT,
    receipt_id   TEXT,
    valid_from   TEXT,
    recorded_at  TEXT NOT NULL,
    UNIQUE (slot_id, fact_id)
);
CREATE INDEX IF NOT EXISTS idx_slot_revisions_fact ON slot_revisions (fact_id);
CREATE INDEX IF NOT EXISTS idx_slot_revisions_slot ON slot_revisions (slot_id, state);
CREATE TABLE IF NOT EXISTS slot_disputes (
    dispute_id   TEXT PRIMARY KEY,
    slot_id      TEXT NOT NULL,
    profile_id   TEXT NOT NULL,
    revision_a   TEXT NOT NULL,
    revision_b   TEXT NOT NULL,
    state        TEXT NOT NULL DEFAULT 'open'
                 CHECK (state IN ('open', 'resolved', 'dismissed')),
    resolution   TEXT CHECK (resolution IS NULL OR resolution IN ('a', 'b', 'both', 'neither')),
    reason_code  TEXT NOT NULL,
    opened_by    TEXT NOT NULL,
    opened_at    TEXT NOT NULL,
    resolved_by  TEXT,
    resolved_at  TEXT
);
CREATE UNIQUE INDEX IF NOT EXISTS uq_slot_disputes_open_pair
    ON slot_disputes (slot_id, revision_a, revision_b) WHERE state = 'open';
CREATE INDEX IF NOT EXISTS idx_slot_disputes_profile_state
    ON slot_disputes (profile_id, state);
CREATE TRIGGER IF NOT EXISTS trg_slot_revision_removed
AFTER DELETE ON slot_revisions
BEGIN
    DELETE FROM slot_disputes
     WHERE revision_a = OLD.revision_id OR revision_b = OLD.revision_id;
    UPDATE memory_slots
       SET head_revision_id = NULL, state = 'empty', version = version + 1
     WHERE head_revision_id = OLD.revision_id;
END;
"""

_FACT_TRIGGER = """
CREATE TRIGGER IF NOT EXISTS trg_slot_fact_erased
AFTER DELETE ON atomic_facts
BEGIN
    DELETE FROM slot_revisions WHERE fact_id = OLD.fact_id;
END
"""

_OBJECTS = ("memory_slots", "slot_revisions", "slot_disputes", "uq_slot_disputes_open_pair",
            "trg_slot_revision_removed")


def _has_facts(conn: sqlite3.Connection) -> bool:
    return bool(conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='atomic_facts'").fetchone())


def _statements(script: str) -> list[str]:
    """Split on ';' outside the trigger body (which holds its own semicolons)."""
    head, trigger = script.split("CREATE TRIGGER", 1)
    parts = [s.strip() for s in head.split(";") if s.strip()]
    return [*parts, "CREATE TRIGGER " + trigger.strip().rstrip(";")]


def apply(conn: sqlite3.Connection) -> None:
    for statement in _statements(DDL):
        conn.execute(statement)
    if _has_facts(conn):  # the erase trigger needs the table it watches
        conn.execute(_FACT_TRIGGER)


def verify(conn: sqlite3.Connection) -> bool:
    names = {r[0] for r in conn.execute("SELECT name FROM sqlite_master")}
    if not set(_OBJECTS) <= names:
        return False
    return not _has_facts(conn) or "trg_slot_fact_erased" in names


def repair(conn: sqlite3.Connection) -> None:
    apply(conn)
