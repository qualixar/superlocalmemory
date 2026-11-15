# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Which judge receipts get a blind audit label, and the only way to write one.

Certificates (``judge_calibration.ActingGate``) learn a judge's error rate
from audit labels. That rate is only honest if the audited receipts are a
random sample of what the judge decided, drawn before anyone knows the answer
and regardless of the judge's score (blueprint M-04). So:

* every receipt with a real answer is a candidate, whatever its score and
  whether or not routing accepted it (the certificate grid reaches 0.50);
  misses (``no_answer``) and failed asks (``tier_failed``) never are;
* a candidate is selected iff a keyed hash of its receipt id falls below one
  constant probability ``p``. Deterministic and reproducible (the same key and
  id always give the same draw), independent of the label, which does not exist
  yet, and of the score. The draw and ``p`` are stored on the receipt;
* the audit queue is a query: selected receipts with no audit label yet;
* ``record_audit_label`` refuses a receipt that was not selected. An audit
  label on a hand-picked receipt would let anyone certify a judge by labelling
  its easy cases, so the gate reads only selected receipts as well.

Labels hold a label from the question's own criteria; never memory text.
"""

from __future__ import annotations

import hashlib
import hmac
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime

from superlocalmemory.core.judge_questions import QUESTIONS

#: Verdicts that are not answers: never audited.
NOT_ANSWERS = frozenset({"no_answer", "tier_failed", "abstain"})


def draw(key: bytes | None, receipt_id: str, *, domain: str = "audit") -> float:
    """A uniform number in [0, 1) fixed by the key, the domain and the id."""
    msg = domain.encode("utf-8") + b":" + receipt_id.encode("utf-8")
    digest = (hmac.new(key, msg, hashlib.sha256).digest() if key is not None
              else hashlib.sha256(msg).digest())
    return int.from_bytes(digest[:8], "big") / 2 ** 64


def select(key: bytes | None, receipt_id: str, verdict: str,
           probability: float | None) -> tuple[bool, float | None]:
    """(selected, logged probability). No probability: no sampling on this path."""
    if probability is None or verdict in NOT_ANSWERS:
        return False, None
    if not 0.0 <= probability <= 1.0:
        raise ValueError(f"audit probability out of range: {probability}")
    return draw(key, receipt_id) < probability, float(probability)


def inbox_select(key: bytes | None, inputs_hash: str, probability: float | None) -> bool:
    """The inbox sample (W0-05): one keyed draw per *pair*, whatever tiers answered
    it, so every pair that reaches a tier is in the sample with the same chance."""
    if probability is None:
        return False
    return draw(key, inputs_hash, domain="inbox") < probability


@dataclass(frozen=True)
class AuditItem:
    receipt_id: str
    profile_id: str
    question_id: str
    subject_kind: str
    subject_id: str
    fact_ids_json: str
    audit_probability: float


_QUEUE = """
SELECT r.receipt_id, r.profile_id, r.question_id, r.subject_kind, r.subject_id,
       r.fact_ids_json, r.audit_probability
  FROM judge_receipts r
 WHERE r.profile_id = ? AND (r.audit_selected = 1 OR r.inbox_selected = 1)
   AND NOT EXISTS (SELECT 1 FROM judge_labels l
                    WHERE l.receipt_id = r.receipt_id AND l.source = 'audit')
 ORDER BY r.decided_at, r.receipt_id
 LIMIT ?
"""


def queue(conn: sqlite3.Connection, profile_id: str, limit: int = 50) -> list[AuditItem]:
    """Selected receipts still waiting for an audit label, oldest first.

    The verdict, model and confidence are deliberately left out: a labeller
    who sees them is no longer blind.
    """
    return [AuditItem(*row) for row in conn.execute(_QUEUE, (profile_id, limit))]


class NotSelectedError(ValueError):
    """The receipt was not drawn for audit; an audit label on it would bias certificates."""


def record_audit_label(conn: sqlite3.Connection, receipt_id: str, label: str, *,
                       labeller: str, blind: bool = True) -> None:
    """Write one audit label. ``blind`` False is stored but never certifies."""
    from superlocalmemory.core.judge_calibration import CANNOT_TELL

    row = conn.execute("SELECT question_id, audit_selected, inbox_selected FROM judge_receipts "
                       "WHERE receipt_id = ?", (receipt_id,)).fetchone()
    if row is None:
        raise LookupError(f"no receipt {receipt_id}")
    if row[1] != 1 and row[2] != 1:
        raise NotSelectedError(f"receipt {receipt_id} was not selected for audit")
    question = QUESTIONS.get(str(row[0]))
    if question is None or (label not in question.criteria and label != CANNOT_TELL):
        raise ValueError(f"label {label!r} is not an answer to {row[0]}")
    if not labeller:
        raise ValueError("an audit label names its labeller")
    conn.execute(
        "INSERT INTO judge_labels (receipt_id, label, source, labeller, blind, labelled_at) "
        "VALUES (?, ?, 'audit', ?, ?, ?)",
        (receipt_id, label, labeller, int(bool(blind)), datetime.now(UTC).isoformat()))


__all__ = ["AuditItem", "NOT_ANSWERS", "NotSelectedError", "draw", "queue",
           "record_audit_label", "select"]
