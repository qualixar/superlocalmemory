# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""M075: injection receipts with propensities (4.2.30, D-14, D-22, D-28, D-33, D-34).

One ``injection_deliveries`` row each time SLM pushes memories into an agent's
context (session start today; every path by W3), and one ``injection_receipts``
row per candidate fact of that delivery: its role, rank, score, whether it was
eligible for randomised withholding (and why not), its logged inclusion
probability, whether it was included, and whether it reached the text the agent
got. The design is ``one-out-v1`` (D-33): in a share epsilon of deliveries one
eligible candidate, drawn uniformly, is left out; the two keyed draws that
decided it are on the delivery row, with the id of the key that made them, so
every decision can be checked later. No memory text and no query text (the
query is a keyed digest). The causal estimates of WS-3 (M-12) read only these rows.

CHECKs hold invariants only (ranges, 0/1, cross-column rules), never lists of
names, which SQLite could not extend after release (W0-29). Erasing a fact
erases its receipts (I-11); a delivery row holds no fact id and stays.
"""

from __future__ import annotations

import sqlite3

NAME = "M075_injection_receipts"
DB_TARGET = "memory"
#: Additive: new tables an older build never reads.
DOWNGRADE_FLOOR = 51

DDL = """
CREATE TABLE IF NOT EXISTS injection_deliveries (
    delivery_id    TEXT PRIMARY KEY,
    profile_id     TEXT NOT NULL,
    session_id     TEXT,
    client         TEXT,
    path           TEXT NOT NULL,
    design         TEXT NOT NULL,
    holdout_state  TEXT NOT NULL,
    epsilon        REAL NOT NULL CHECK (epsilon >= 0 AND epsilon <= 0.10),
    eligible_count INTEGER NOT NULL CHECK (eligible_count >= 0),
    withhold_draw  REAL CHECK (withhold_draw IS NULL OR (withhold_draw >= 0 AND withhold_draw < 1)),
    pick_draw      REAL CHECK (pick_draw IS NULL OR (pick_draw >= 0 AND pick_draw < 1)),
    key_id         TEXT,
    query_digest   TEXT,
    candidates     INTEGER NOT NULL,
    delivered_at   TEXT NOT NULL,
    CHECK ((withhold_draw IS NULL) = (pick_draw IS NULL)),
    CHECK (withhold_draw IS NULL OR key_id IS NOT NULL)
);
CREATE INDEX IF NOT EXISTS idx_injection_deliveries_session
    ON injection_deliveries (profile_id, session_id, delivered_at);
CREATE TABLE IF NOT EXISTS injection_receipts (
    delivery_id TEXT NOT NULL,
    fact_id     TEXT NOT NULL,
    profile_id  TEXT NOT NULL,
    role        TEXT NOT NULL,
    position    INTEGER NOT NULL,
    score       REAL,
    eligible    INTEGER NOT NULL CHECK (eligible IN (0, 1)),
    exclusion   TEXT,
    propensity  REAL NOT NULL CHECK (propensity > 0 AND propensity <= 1),
    included    INTEGER NOT NULL CHECK (included IN (0, 1)),
    delivered   INTEGER NOT NULL CHECK (delivered IN (0, 1)),
    PRIMARY KEY (delivery_id, fact_id),
    CHECK ((eligible = 1) = (exclusion IS NULL)),
    CHECK (eligible = 1 OR (propensity = 1 AND included = 1)),
    CHECK (included = 1 OR delivered = 0)
);
CREATE INDEX IF NOT EXISTS idx_injection_receipts_fact ON injection_receipts (fact_id)
"""

_FACT_TRIGGER = """
CREATE TRIGGER IF NOT EXISTS trg_injection_receipts_fact_erased
AFTER DELETE ON atomic_facts
BEGIN
    DELETE FROM injection_receipts WHERE fact_id = OLD.fact_id;
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
    if not {"injection_deliveries", "injection_receipts"} <= names:
        return False
    return not _has_facts(conn) or "trg_injection_receipts_fact_erased" in names


def repair(conn: sqlite3.Connection) -> None:
    apply(conn)
