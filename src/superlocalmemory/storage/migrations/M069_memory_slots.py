# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""M069: slots, so SLM knows which memory is the current value of something.

A slot is "the value of relation R for subject S in scope X": the release
cadence of project SLM, the voice model Alex prefers. Each fact bound to a
slot is one immutable revision. A single-valued slot has one head; a new
revision either supersedes it (through the correction ledger, so it can be
rolled back exactly), joins it as compatible, repeats it (duplicate), or opens a
dispute that shows both sides until someone resolves it.

Three additive tables (``truth/slot_store.py`` owns them) and the slot pass's
queue. Ids, hashes and controlled codes only: a revision points at its fact and
never copies its text.

Erasure leaves a slot as if the erased fact had never been saved (Fable B2):

* erasing any revision removes the disputes that name it, clears pointers to
  it (``superseded_by``) and ``case_id``s whose correction case is gone, then
  recomputes the slot's state (``empty`` / ``disputed`` / ``current``) and
  bumps ``version`` and ``updated_at``;
* erasing the head rebuilds the slot: every remaining revision whose fact is
  still live (disputed, duplicate, compatible, historical) is removed and
  queued in ``slot_pass_queue`` to be bound again, oldest first, by the slot
  pass; superseded revisions stay as history (their facts were retired by the
  ledger, and erasure never restores them). No value is promoted by the
  trigger: promotion is a memory change, and re-binding applies the same rules
  as a first save;
* a slot left with no revisions is deleted with its key aliases, since keys
  are derived from the erased text (I-11).

Slot and revision ids are opaque (R-12): a slot is found by its natural key
or by an alias in ``slot_key_alias``, so re-keying a slot (``name:`` to
``entity:<id>`` subjects) orphans nothing.

``slot_pass_queue`` also holds facts whose binding failed, retried with
backoff (R-19), so no fact is skipped silently. Amended in place before any
release (blueprint D-29).
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
    resolution   TEXT CHECK (resolution IS NULL OR resolution IN
                 ('a', 'b', 'both', 'neither', 'same', 'unrelated')),
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
CREATE INDEX IF NOT EXISTS idx_slot_disputes_slot ON slot_disputes (slot_id);
CREATE TABLE IF NOT EXISTS slot_key_alias (
    profile_id   TEXT NOT NULL,
    scope        TEXT NOT NULL,
    subject_key  TEXT NOT NULL,
    relation_key TEXT NOT NULL,
    qualifier    TEXT NOT NULL DEFAULT '',
    slot_id      TEXT NOT NULL,
    reason_code  TEXT NOT NULL,
    aliased_by   TEXT NOT NULL,
    aliased_at   TEXT NOT NULL,
    PRIMARY KEY (profile_id, scope, subject_key, relation_key, qualifier)
);
CREATE INDEX IF NOT EXISTS idx_slot_key_alias_slot ON slot_key_alias (slot_id);
CREATE TABLE IF NOT EXISTS slot_pass_queue (
    fact_id    TEXT PRIMARY KEY,
    reason     TEXT NOT NULL CHECK (reason IN ('rebind', 'retry')),
    attempts   INTEGER NOT NULL DEFAULT 0 CHECK (attempts >= 0),
    next_at    TEXT NOT NULL,
    last_error TEXT,
    queued_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_slot_pass_queue_due ON slot_pass_queue (next_at)
"""

_NOW_SQL = "strftime('%Y-%m-%dT%H:%M:%f+00:00', 'now')"

def _revision_trigger(with_ledger: bool) -> str:
    """The revision-removed trigger; the ``case_id`` clean-up needs the ledger table."""
    clear_cases = (
        "    UPDATE slot_revisions SET case_id = NULL\n"
        "     WHERE slot_id = OLD.slot_id AND case_id IS NOT NULL\n"
        "       AND NOT EXISTS (SELECT 1 FROM correction_cases c\n"
        "                        WHERE c.case_id = slot_revisions.case_id);\n"
    ) if with_ledger else ""
    return f"""
CREATE TRIGGER trg_slot_revision_removed
AFTER DELETE ON slot_revisions
BEGIN
    UPDATE memory_slots SET head_revision_id = NULL WHERE head_revision_id = OLD.revision_id;
    INSERT OR REPLACE INTO slot_pass_queue (fact_id, reason, attempts, next_at, queued_at)
        SELECT r.fact_id, 'rebind', 0, {_NOW_SQL}, {_NOW_SQL} FROM slot_revisions r
         WHERE OLD.state = 'head' AND r.slot_id = OLD.slot_id AND r.state != 'superseded';
    DELETE FROM slot_revisions
     WHERE OLD.state = 'head' AND slot_id = OLD.slot_id AND state != 'superseded';
    -- Rows deleted just above do not fire this trigger (recursive_triggers is off),
    -- so clean up whatever now dangles in the slot, not only what named OLD.
    DELETE FROM slot_disputes
     WHERE revision_a = OLD.revision_id OR revision_b = OLD.revision_id
        OR (slot_id = OLD.slot_id
            AND (NOT EXISTS (SELECT 1 FROM slot_revisions r WHERE r.revision_id = revision_a)
                 OR NOT EXISTS (SELECT 1 FROM slot_revisions r WHERE r.revision_id = revision_b)));
    UPDATE slot_revisions SET superseded_by = NULL
     WHERE superseded_by = OLD.revision_id
        OR (slot_id = OLD.slot_id AND superseded_by IS NOT NULL
            AND NOT EXISTS (SELECT 1 FROM slot_revisions r
                             WHERE r.revision_id = slot_revisions.superseded_by));
{clear_cases}    UPDATE memory_slots
       SET state = CASE
               WHEN head_revision_id IS NULL THEN 'empty'
               WHEN EXISTS (SELECT 1 FROM slot_disputes d
                             WHERE d.slot_id = memory_slots.slot_id AND d.state = 'open')
                    THEN 'disputed'
               ELSE 'current' END,
           version = version + 1,
           updated_at = {_NOW_SQL}
     WHERE slot_id = OLD.slot_id;
    DELETE FROM slot_key_alias
     WHERE slot_id = OLD.slot_id
       AND NOT EXISTS (SELECT 1 FROM slot_revisions r WHERE r.slot_id = OLD.slot_id);
    DELETE FROM memory_slots
     WHERE slot_id = OLD.slot_id
       AND NOT EXISTS (SELECT 1 FROM slot_revisions r WHERE r.slot_id = OLD.slot_id);
END
"""


_FACT_TRIGGER = """
CREATE TRIGGER IF NOT EXISTS trg_slot_fact_erased
AFTER DELETE ON atomic_facts
BEGIN
    DELETE FROM slot_revisions WHERE fact_id = OLD.fact_id;
    DELETE FROM slot_pass_queue WHERE fact_id = OLD.fact_id;
END
"""

_OBJECTS = ("memory_slots", "slot_revisions", "slot_disputes", "uq_slot_disputes_open_pair",
            "slot_key_alias", "slot_pass_queue", "trg_slot_revision_removed")
#: A phrase only the current trigger body has: a store copy that ran an earlier
#: draft of this migration (D-29) fails ``verify`` and is repaired.
_TRIGGER_MARK = "DELETE FROM slot_key_alias"


def _has_table(conn: sqlite3.Connection, name: str) -> bool:
    return bool(conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)).fetchone())


def _has_facts(conn: sqlite3.Connection) -> bool:
    return _has_table(conn, "atomic_facts")


def apply(conn: sqlite3.Connection) -> None:
    for statement in (s.strip() for s in DDL.split(";")):
        if statement:
            conn.execute(statement)
    # Triggers hold no data: always (re)create the revision trigger, so a copy
    # that ran an earlier draft gets the current body.
    conn.execute("DROP TRIGGER IF EXISTS trg_slot_revision_removed")
    conn.execute(_revision_trigger(_has_table(conn, "correction_cases")))
    if _has_facts(conn):  # the erase trigger needs the table it watches
        conn.execute("DROP TRIGGER IF EXISTS trg_slot_fact_erased")
        conn.execute(_FACT_TRIGGER)


def verify(conn: sqlite3.Connection) -> bool:
    rows = {r[0]: r[1] or "" for r in conn.execute("SELECT name, sql FROM sqlite_master")}
    if not set(_OBJECTS) <= set(rows):
        return False
    if _TRIGGER_MARK not in rows["trg_slot_revision_removed"]:
        return False
    if _has_facts(conn) and "slot_pass_queue" not in rows.get("trg_slot_fact_erased", ""):
        return False
    return True


def repair(conn: sqlite3.Connection) -> None:
    apply(conn)
