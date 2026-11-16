# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""M077: judge certificates as stored records (I-14, R-26, M-07).

The acting gate (``core/judge_calibration.ActingGate``) decides from blind
audit labels whether a judge's verdict may change memory. Until now that
decision was recomputed each time and left no trace. A certificate is now a
record: its key (profile, question and version, tier, model, verdict,
population, batching regime), the method, alpha, delta, the labels it rests on
(n, errors, a hash of the label set), the threshold, and when it was issued and
voided, with why (``labels``: new labels no longer support it; ``drift``: the
judge's recent scores no longer look like the audited ones, M-07).

At most one open certificate per key. Counts and ids only, never memory text,
so erasing a memory has nothing to erase here; the labels it rests on are
erased with their receipts (I-11) and the next check voids it.
"""

from __future__ import annotations

import sqlite3

NAME = "M077_judge_certificates"
DB_TARGET = "memory"
#: Additive: a new table an older build never reads.
DOWNGRADE_FLOOR = 51

DDL = """
CREATE TABLE IF NOT EXISTS judge_certificates (
    cert_id          TEXT PRIMARY KEY,
    profile_id       TEXT NOT NULL,
    question_id      TEXT NOT NULL,
    question_version TEXT NOT NULL,
    tier             TEXT NOT NULL,
    model            TEXT NOT NULL,
    verdict          TEXT NOT NULL,
    population       TEXT NOT NULL,
    regime           TEXT NOT NULL,
    method           TEXT NOT NULL,
    alpha            REAL NOT NULL,
    delta            REAL NOT NULL,
    n                INTEGER NOT NULL CHECK (n >= 0),
    errors           INTEGER NOT NULL CHECK (errors >= 0 AND errors <= n),
    threshold        REAL NOT NULL,
    label_set_hash   TEXT NOT NULL,
    issued_at        TEXT NOT NULL,
    voided_at        TEXT,
    void_reason      TEXT,
    CHECK ((voided_at IS NULL) = (void_reason IS NULL))
);
CREATE UNIQUE INDEX IF NOT EXISTS uq_judge_certificates_open
    ON judge_certificates (profile_id, question_id, question_version, tier, model, verdict,
                           population, regime) WHERE voided_at IS NULL;
CREATE INDEX IF NOT EXISTS idx_judge_certificates_profile
    ON judge_certificates (profile_id, issued_at)
"""


def apply(conn: sqlite3.Connection) -> None:
    for statement in DDL.split(";"):
        if statement.strip():
            conn.execute(statement)


def verify(conn: sqlite3.Connection) -> bool:
    names = {r[0] for r in conn.execute("SELECT name FROM sqlite_master")}
    return {"judge_certificates", "uq_judge_certificates_open"} <= names


def repair(conn: sqlite3.Connection) -> None:
    apply(conn)
