# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""M081: the instruction firewall's approval record (4.2.30, WS-5, D-37, I-10, I-11).

A memory whose taint is below ``system`` (a tool's output, a web page, another agent's
message, an import, a model's text, an agent's own words, or anything saved before 4.2.30)
cannot become a pin, a standing rule or a core block, and is not shown in the session-start
sections, until a person approves it (``trust/firewall.py``).

This table records the two things nothing else can: what was *asked for* and refused or
withheld (``wanted``: ``pin``, ``rule``, ``decision``, ``core`` or ``session_start``, with who
asked and through which door), and the person's answer (who, when, through which door:
``mcp``, ``cli``, ``http`` or ``dashboard``; any caller may promote, D-49). The promotion itself raises the fact to
``user_confirmed`` in ``fact_trust`` (``fact_trust.mark_confirmed``); this row is the
review record, and the waiting list. No memory text is stored here.

CHECKs hold invariants only (a decided row says who, when and how), never a list of states
or wishes, which SQLite could not extend after release (W0-29). Erasing a fact erases its
rows (I-11); a profile fold moves them with the fact.
"""

from __future__ import annotations

import sqlite3

NAME = "M081_trust_approvals"
DB_TARGET = "memory"
#: Additive: a new table an older build never reads.
DOWNGRADE_FLOOR = 51

DDL = """
CREATE TABLE IF NOT EXISTS trust_approvals (
    approval_id       TEXT PRIMARY KEY,
    fact_id           TEXT NOT NULL,
    profile_id        TEXT NOT NULL,
    wanted            TEXT NOT NULL CHECK (length(wanted) > 0),
    state             TEXT NOT NULL CHECK (length(state) > 0),
    requested_by      TEXT NOT NULL DEFAULT '',
    requested_channel TEXT NOT NULL DEFAULT '',
    requested_at      TEXT NOT NULL,
    decided_by        TEXT,
    decided_channel   TEXT,
    decided_at        TEXT,
    UNIQUE (fact_id, wanted),
    CHECK (state <> 'approved' OR (decided_by IS NOT NULL AND decided_channel IS NOT NULL
                                   AND decided_at IS NOT NULL))
);
CREATE INDEX IF NOT EXISTS idx_trust_approvals_waiting
    ON trust_approvals (profile_id, state, requested_at)
"""

_FACT_TRIGGER = """
CREATE TRIGGER IF NOT EXISTS trg_trust_approvals_fact_erased
AFTER DELETE ON atomic_facts
BEGIN
    DELETE FROM trust_approvals WHERE fact_id = OLD.fact_id;
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
    if not {"trust_approvals", "idx_trust_approvals_waiting"} <= names:
        return False
    return not _has_facts(conn) or "trg_trust_approvals_fact_erased" in names


def repair(conn: sqlite3.Connection) -> None:
    apply(conn)
