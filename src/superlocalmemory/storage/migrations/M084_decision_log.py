# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""M084: the decision log, one row for each decision a caller made (4.2.30, D-49, I-10).

SuperLocalMemory is a memory system for agents. Any caller (an agent through MCP, the
command line or HTTP, a script, or a person on the dashboard) may release a memory the
instruction firewall holds, settle a dispute, give a blind audit label, and apply or roll
back a set of proposed changes. Nothing here asks who the caller "really" is. Instead every
such decision is written down: **who** (an agent id, a client name or ``dashboard``),
**through which channel** (``mcp``, ``cli``, ``http`` or ``dashboard``), **when**, what it
was about (a kind and an id) and, if someone took it back, who undid it and when. The
dashboard card "Recent decisions by your agents" reads this table, and each row can be undone
through the same doors that made it (a promotion is withdrawn, a set is rolled back, a
dispute is reopened, a label is retracted).

The table holds ids, words for a person to read and a small JSON of what is needed to undo
the decision (a class name, a list of wishes). Never memory text. Erasing a fact erases the
decisions about it (I-11). CHECKs hold invariants only, never a list of kinds or channels,
which SQLite could not extend after release (W0-29).

**Grandfathering (D-49).** ``apply`` also does one data step, once, at the upgrade: a memory
saved before 4.2.30 has no origin row (``unknown``). If it was already a pin, a standing rule
or a standing decision, it keeps being one: ``fact_trust`` gets a row with origin ``unknown``
and basis ``confirmed:grandfathered:pre-4.2.30``, so nothing the user relied on silently
disappears when the instruction firewall is turned on. Only new promotions follow the rule. The
step never runs again: ``repair`` restores the schema only, because replaying it later would
grandfather a memory that something marked as a rule after the upgrade.
"""

from __future__ import annotations

import sqlite3

NAME = "M084_decision_log"
DB_TARGET = "memory"
#: Additive: a new table an older build never reads.
DOWNGRADE_FLOOR = 51

DDL = """
CREATE TABLE IF NOT EXISTS decision_log (
    decision_id      TEXT PRIMARY KEY,
    profile_id       TEXT NOT NULL,
    kind             TEXT NOT NULL CHECK (length(kind) > 0),
    subject_id       TEXT NOT NULL CHECK (length(subject_id) > 0),
    action           TEXT NOT NULL CHECK (length(action) > 0),
    actor            TEXT NOT NULL CHECK (length(actor) > 0),
    channel          TEXT NOT NULL CHECK (length(channel) > 0),
    summary          TEXT NOT NULL DEFAULT '',
    detail_json      TEXT NOT NULL DEFAULT '{}',
    decided_at       TEXT NOT NULL,
    reverted_at      TEXT,
    reverted_by      TEXT,
    reverted_channel TEXT,
    CHECK ((reverted_at IS NULL) = (reverted_by IS NULL)
           AND (reverted_at IS NULL) = (reverted_channel IS NULL))
);
CREATE INDEX IF NOT EXISTS idx_decision_log_recent ON decision_log (profile_id, decided_at);
CREATE INDEX IF NOT EXISTS idx_decision_log_subject ON decision_log (kind, subject_id)
"""

#: I-11: erasing a fact erases the decisions about it (a promotion names the fact).
_FACT_TRIGGER = """
CREATE TRIGGER IF NOT EXISTS trg_decision_log_fact_erased
AFTER DELETE ON atomic_facts
BEGIN
    DELETE FROM decision_log WHERE kind = 'promotion' AND subject_id = OLD.fact_id;
END
"""


def _has_facts(conn: sqlite3.Connection) -> bool:
    return bool(conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='atomic_facts'").fetchone())


GRANDFATHERED = "confirmed:grandfathered:pre-4.2.30"

_GRANDFATHER = """
INSERT OR IGNORE INTO fact_trust (fact_id, profile_id, origin_class, taint_class, basis, computed_at)
SELECT f.fact_id, f.profile_id, 'unknown', 'user_confirmed', ?, ?
  FROM atomic_facts f
 WHERE NOT EXISTS (SELECT 1 FROM fact_origin o WHERE o.fact_id = f.fact_id)
   AND ({marked})
"""


def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}


def _tables(conn: sqlite3.Connection) -> set[str]:
    return {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}


def grandfather(conn: sqlite3.Connection) -> int:
    """Record that memories already pinned or marked rule/decision before 4.2.30 stay so."""
    from datetime import UTC, datetime

    if not {"atomic_facts", "fact_origin", "fact_trust"} <= _tables(conn):
        return 0
    have = _columns(conn, "atomic_facts")
    marks = []
    if "pinned" in have:
        marks.append("f.pinned = 1")
    if {"memory_kind", "memory_kind_source"} <= have:
        marks.append("(f.memory_kind IN ('rule', 'decision') "
                     "AND f.memory_kind_source IN ('user', 'caller'))")
    if not marks:
        return 0
    cursor = conn.execute(_GRANDFATHER.format(marked=" OR ".join(marks)),
                          (GRANDFATHERED, datetime.now(UTC).isoformat()))
    return max(0, cursor.rowcount)


def repair(conn: sqlite3.Connection) -> None:
    """Schema only: the grandfathering is a one-time step at the upgrade (see the text)."""
    _schema(conn)


def _schema(conn: sqlite3.Connection) -> None:
    for statement in DDL.split(";"):
        if statement.strip():
            conn.execute(statement)
    if _has_facts(conn):
        conn.execute(_FACT_TRIGGER)


def apply(conn: sqlite3.Connection) -> None:
    _schema(conn)
    grandfather(conn)


def verify(conn: sqlite3.Connection) -> bool:
    names = {r[0] for r in conn.execute("SELECT name FROM sqlite_master")}
    if not {"decision_log", "idx_decision_log_recent", "idx_decision_log_subject"} <= names:
        return False
    return not _has_facts(conn) or "trg_decision_log_fact_erased" in names

