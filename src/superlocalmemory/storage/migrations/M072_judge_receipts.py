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

Two additive tables and an erase trigger. Schema v2 (Fable review, 15 Nov):
population, the judged fact ids, tier index, raw score, batch id / size /
position / peers hash (batch dependence is measured, not assumed away), the
flag state at decision time, the action taken and its reference (so a decision
made in shadow is acted on later instead of stranded), and audit selection with
its probability (only blind audit labels feed certificates). A tier that was
asked and failed as a whole writes ``tier_failed`` with ``failure_kind`` (R-35);
the CHECK keeps the two in step. A deterministic tier (rules) has no score:
its ``confidence`` and ``raw_score`` are NULL rather than a made-up 1.0 (R-36).
A hosted ask's receipts carry ``consent_ref``, the consent record it relied on
(I-15, R-42); that one is checked by
``slm doctor --truth``, not a CHECK, so one bad row never voids a batch. ``inputs_hash`` is
an HMAC under a local key (``core/receipt_key.py``), so it identifies what was
judged without confirming a guess of erased text. Labels: many per receipt
(one per source and labeller), with ``blind``, ``labeller_kind`` (``person`` or ``agent``) and
``labeller_model`` (D-49: an agent's label counts toward a certificate cell only when its model
differs from the judged tier's model; a person's always counts; labels written before the
columns existed are a person's). Amended in place before any
release (blueprint D-29); only disposable store copies ran the first version.
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
    population        TEXT NOT NULL DEFAULT 'unknown',
    fact_ids_json     TEXT NOT NULL DEFAULT '[]',
    tier              TEXT NOT NULL,
    tier_index        INTEGER NOT NULL DEFAULT 0,
    model             TEXT NOT NULL,
    verdict           TEXT NOT NULL,
    confidence        REAL CHECK (confidence IS NULL OR (confidence >= 0.0 AND confidence <= 1.0)),
    raw_score         REAL,
    threshold         REAL NOT NULL,
    accepted          INTEGER NOT NULL CHECK (accepted IN (0, 1)),
    probabilities_json TEXT NOT NULL DEFAULT '{}',
    calibration_id    TEXT NOT NULL,
    inputs_hash       TEXT NOT NULL,
    batch_id          TEXT,
    batch_size        INTEGER,
    batch_position    INTEGER,
    peers_hash        TEXT,
    flag_state        TEXT NOT NULL DEFAULT 'unknown',
    action_taken      TEXT,
    action_ref        TEXT,
    audit_selected    INTEGER NOT NULL DEFAULT 0 CHECK (audit_selected IN (0, 1)),
    audit_probability REAL,
    inbox_selected    INTEGER NOT NULL DEFAULT 0 CHECK (inbox_selected IN (0, 1)),
    decided_at        TEXT NOT NULL,
    failure_kind      TEXT,
    consent_ref       TEXT,
    batching_regime   TEXT NOT NULL DEFAULT 'unknown',
    CHECK ((verdict = 'tier_failed') = (failure_kind IS NOT NULL))
);
CREATE INDEX IF NOT EXISTS idx_judge_receipts_subject
    ON judge_receipts (subject_kind, subject_id);
CREATE INDEX IF NOT EXISTS idx_judge_receipts_question
    ON judge_receipts (profile_id, question_id, decided_at);
CREATE INDEX IF NOT EXISTS idx_judge_receipts_inputs
    ON judge_receipts (inputs_hash, question_version);
CREATE TABLE IF NOT EXISTS judge_labels (
    receipt_id  TEXT NOT NULL REFERENCES judge_receipts (receipt_id) ON DELETE CASCADE,
    label       TEXT NOT NULL,
    source      TEXT NOT NULL,
    labeller    TEXT NOT NULL DEFAULT '',
    blind       INTEGER NOT NULL DEFAULT 0 CHECK (blind IN (0, 1)),
    labelled_at TEXT NOT NULL,
    labeller_kind  TEXT NOT NULL DEFAULT 'person',
    labeller_model TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (receipt_id, source, labeller)
);
CREATE TABLE IF NOT EXISTS judge_receipt_facts (
    receipt_id TEXT NOT NULL,
    fact_id    TEXT NOT NULL,
    PRIMARY KEY (receipt_id, fact_id)
);
CREATE INDEX IF NOT EXISTS idx_judge_receipt_facts_fact ON judge_receipt_facts (fact_id)
"""

#: Erasing a fact erases every receipt (and so every label) that judged it (I-11),
#: through the indexed link table: one lookup per erased fact, not a scan of every
#: receipt (W0-09; ``LIKE`` also read ``_`` in an id as a wildcard).
_FACT_TRIGGER = """
CREATE TRIGGER trg_judge_receipts_fact_erased
AFTER DELETE ON atomic_facts
BEGIN
    DELETE FROM judge_labels WHERE receipt_id IN (
        SELECT receipt_id FROM judge_receipt_facts WHERE fact_id = OLD.fact_id);
    DELETE FROM judge_receipts WHERE receipt_id IN (
        SELECT receipt_id FROM judge_receipt_facts WHERE fact_id = OLD.fact_id);
    DELETE FROM judge_receipt_facts WHERE receipt_id IN (
        SELECT receipt_id FROM judge_receipt_facts WHERE fact_id = OLD.fact_id);
END
"""


def _has_facts(conn: sqlite3.Connection) -> bool:
    return bool(conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='atomic_facts'").fetchone())


_LABEL_COLUMNS = (("labeller_kind", "TEXT NOT NULL DEFAULT 'person'"),
                  ("labeller_model", "TEXT NOT NULL DEFAULT ''"))


def _add_label_columns(conn: sqlite3.Connection) -> None:
    """A store that ran the first version of this migration gains the D-49 columns."""
    have = {r[1] for r in conn.execute("PRAGMA table_info(judge_labels)")}
    for name, spec in _LABEL_COLUMNS:
        if name not in have:
            conn.execute(f"ALTER TABLE judge_labels ADD COLUMN {name} {spec}")


def apply(conn: sqlite3.Connection) -> None:
    for statement in DDL.split(";"):
        if statement.strip():
            conn.execute(statement)
    _add_label_columns(conn)
    if _has_facts(conn):  # triggers hold no data: always the current body
        conn.execute("DROP TRIGGER IF EXISTS trg_judge_receipts_fact_erased")
        conn.execute(_FACT_TRIGGER)


def verify(conn: sqlite3.Connection) -> bool:
    names = {r[0] for r in conn.execute("SELECT name FROM sqlite_master")}
    if not {"judge_receipts", "judge_labels", "idx_judge_receipts_inputs",
            "judge_receipt_facts", "idx_judge_receipt_facts_fact"} <= names:
        return False
    have = {r[1] for r in conn.execute("PRAGMA table_info(judge_labels)")}
    if not {"labeller_kind", "labeller_model"} <= have:
        return False
    return not _has_facts(conn) or "trg_judge_receipts_fact_erased" in names


def repair(conn: sqlite3.Connection) -> None:
    apply(conn)
