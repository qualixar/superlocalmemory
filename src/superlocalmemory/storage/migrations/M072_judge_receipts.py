# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""M072: a receipt for every judge decision, and the labels that calibrate it.

The judge ladder (``core/judge_ladder.py``) records each answer it gets from a
tier: which typed question, which tier and model, the verdict and its
confidence, the threshold it was held to, the calibration it came from, and a
hash of what the judge was shown. Never the text itself. A decision can then
be audited ("why was this memory retired?") and replayed, and every receipt a
person later confirms or overrides becomes a label for calibrating that
question on that person's own store.

Two additive tables.
"""

from __future__ import annotations

import sqlite3

NAME = "M072_judge_receipts"
DB_TARGET = "memory"
#: Additive (two new tables no older build reads).
DOWNGRADE_FLOOR = 51

DDL = """
CREATE TABLE IF NOT EXISTS judge_receipts (
    receipt_id        TEXT PRIMARY KEY,
    profile_id        TEXT NOT NULL,
    question_id       TEXT NOT NULL,
    question_version  TEXT NOT NULL,
    subject_kind      TEXT NOT NULL,
    subject_id        TEXT NOT NULL,
    tier              TEXT NOT NULL,
    model             TEXT NOT NULL,
    verdict           TEXT NOT NULL,
    confidence        REAL NOT NULL CHECK (confidence >= 0.0 AND confidence <= 1.0),
    threshold         REAL NOT NULL,
    accepted          INTEGER NOT NULL CHECK (accepted IN (0, 1)),
    probabilities_json TEXT NOT NULL DEFAULT '{}',
    calibration_id    TEXT NOT NULL,
    inputs_hash       TEXT NOT NULL,
    decided_at        TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_judge_receipts_subject
    ON judge_receipts (subject_kind, subject_id);
CREATE INDEX IF NOT EXISTS idx_judge_receipts_question
    ON judge_receipts (profile_id, question_id, decided_at);
CREATE TABLE IF NOT EXISTS judge_labels (
    receipt_id  TEXT NOT NULL REFERENCES judge_receipts (receipt_id) ON DELETE CASCADE,
    label       TEXT NOT NULL,
    source      TEXT NOT NULL CHECK (source IN ('review', 'rollback', 'feedback')),
    labelled_at TEXT NOT NULL,
    PRIMARY KEY (receipt_id, source)
)
"""


def apply(conn: sqlite3.Connection) -> None:
    for statement in DDL.split(";"):
        if statement.strip():
            conn.execute(statement)


def verify(conn: sqlite3.Connection) -> bool:
    names = {r[0] for r in conn.execute("SELECT name FROM sqlite_master")}
    return {"judge_receipts", "judge_labels", "idx_judge_receipts_subject"} <= names


def repair(conn: sqlite3.Connection) -> None:
    apply(conn)
