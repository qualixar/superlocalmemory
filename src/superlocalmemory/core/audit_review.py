# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""The audit queue a person answers: blind labels that let a judge earn a certificate (W2, R-16).

``core/audit_sampler`` draws which judge receipts get audited, at random and
before anyone knows the answer. This module is what a person sees and how the
answer is written:

* one item per *pair* (``inputs_hash``), whichever tiers were asked about it:
  the label is the truth about the pair, so it is written to every drawn
  receipt of that pair at once;
* the item shows the question in plain words and the two memories in time
  order (earlier valid time first, as the dispute pass orders them), and never
  the judge's verdict, model or score: a labeller who sees them is not blind;
* the answer is one of the question's own options or "I can't tell"
  (``cannot_tell``, which never counts for or against a judge).

Audits are for people on this machine: the CLI asks only at an interactive
terminal and the HTTP routes refuse remote callers. An agent that answers in a
person's place would be grading its own kind, so no MCP tool writes an audit
label. (A local process can still reach the HTTP route with the install token;
the labeller and channel are recorded so such labels can be told apart.)
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Sequence

from superlocalmemory.core.audit_sampler import record_audit_label
from superlocalmemory.core.judge_calibration import CANNOT_TELL
from superlocalmemory.core.judge_questions import QUESTIONS

CHANNELS = ("cli", "dashboard")
TEXT_CHARS = 1200
MAX_ITEMS = 50

#: The question and its options as a person reads them (same labels as the judge's).
PLAIN: dict[str, dict] = {
    "SLOT_RELATION": {
        "question": "Memory A was saved before memory B. What does B do to A?",
        "options": (
            ("replaces", "B gives a new value for the same thing; A is no longer true"),
            ("same", "B says the same as A, maybe in other words"),
            ("both_true", "Both can be true at the same time"),
            ("separate_events", "They record different events or moments; both stay true"),
            ("unrelated", "They are about different things"),
            (CANNOT_TELL, "I can't tell"),
        ),
    },
}

_PENDING = """
SELECT r.inputs_hash, MIN(r.question_id), MIN(r.fact_ids_json), MIN(r.decided_at)
  FROM judge_receipts r
 WHERE r.profile_id = ? AND (r.audit_selected = 1 OR r.inbox_selected = 1)
   AND r.inputs_hash IS NOT NULL
   AND NOT EXISTS (SELECT 1 FROM judge_labels l
                    WHERE l.receipt_id = r.receipt_id AND l.source = 'audit')
 GROUP BY r.inputs_hash
 ORDER BY MIN(r.decided_at), r.inputs_hash
 LIMIT ?
"""


def _texts(conn: sqlite3.Connection, fact_ids: Sequence[str]) -> list[dict]:
    marks = ",".join("?" * len(fact_ids))
    rows = conn.execute(
        "SELECT f.fact_id, f.content, COALESCE(v.valid_from, f.valid_from, f.created_at) "
        "FROM atomic_facts f LEFT JOIN fact_temporal_validity v ON v.fact_id = f.fact_id "
        f"WHERE f.fact_id IN ({marks})", tuple(fact_ids)).fetchall()
    rows.sort(key=lambda r: r[2] or "")  # earlier valid time first: A, then B
    return [{"name": "AB"[i] if i < 2 else str(i + 1), "fact_id": f,
             "content": (c or "")[:TEXT_CHARS], "said_at": w} for i, (f, c, w) in enumerate(rows)]


def pending(conn: sqlite3.Connection, profile_id: str, limit: int = 10) -> list[dict]:
    """Pairs drawn for audit and not yet answered, oldest first, without any verdict."""
    try:
        rows = conn.execute(_PENDING, (profile_id, max(1, min(int(limit), MAX_ITEMS)))).fetchall()
    except sqlite3.OperationalError:  # a store before M072
        return []
    out = []
    for inputs_hash, question_id, fact_ids_json, _when in rows:
        plain = PLAIN.get(question_id)
        if plain is None or question_id not in QUESTIONS:
            continue  # no wording a person can answer yet
        texts = _texts(conn, json.loads(fact_ids_json or "[]"))
        if len(texts) < 2:
            continue  # a memory it judged is gone; nothing left to ask about
        out.append({"item_id": inputs_hash, "question_id": question_id,
                    "question": plain["question"],
                    "options": [{"label": k, "text": v} for k, v in plain["options"]],
                    "memories": texts})
    return out


def pending_count(conn: sqlite3.Connection, profile_id: str) -> int:
    try:
        return int(conn.execute(
            "SELECT COUNT(DISTINCT inputs_hash) FROM judge_receipts r WHERE r.profile_id = ? "
            "AND (r.audit_selected = 1 OR r.inbox_selected = 1) AND r.inputs_hash IS NOT NULL "
            "AND NOT EXISTS (SELECT 1 FROM judge_labels l WHERE l.receipt_id = r.receipt_id "
            "AND l.source = 'audit')", (profile_id,)).fetchone()[0])
    except sqlite3.OperationalError:
        return 0


def answer(conn: sqlite3.Connection, *, profile_id: str, item_id: str, label: str,
           channel: str, actor: str = "") -> int:
    """Write the person's label to every drawn, unlabelled receipt of the pair.

    Returns how many receipts it labelled. Raises ``LookupError`` when the item
    is not waiting in this profile and ``ValueError`` for an answer that is not
    one of the question's options.
    """
    if channel not in CHANNELS:
        raise ValueError(f"channel must be one of {CHANNELS}")
    receipts = [r[0] for r in conn.execute(
        "SELECT r.receipt_id FROM judge_receipts r WHERE r.profile_id = ? AND r.inputs_hash = ? "
        "AND (r.audit_selected = 1 OR r.inbox_selected = 1) AND NOT EXISTS (SELECT 1 FROM "
        "judge_labels l WHERE l.receipt_id = r.receipt_id AND l.source = 'audit') "
        "ORDER BY r.receipt_id", (profile_id, item_id))]
    if not receipts:
        raise LookupError(f"no audit item {item_id} waiting")
    labeller = f"person:{channel}" + (f":{actor[:80]}" if actor else "")
    conn.execute("SAVEPOINT audit")
    try:
        for receipt_id in receipts:  # the question's own options are checked per receipt
            record_audit_label(conn, receipt_id, label, labeller=labeller, blind=True)
    except Exception:
        conn.execute("ROLLBACK TO audit")
        conn.execute("RELEASE audit")
        raise
    conn.execute("RELEASE audit")
    return len(receipts)


__all__ = ["CHANNELS", "PLAIN", "answer", "pending", "pending_count"]
