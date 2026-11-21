# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""The audit queue any caller answers: blind labels that let a judge earn a certificate (W2, R-16, D-49).

``core/audit_sampler`` draws which judge receipts get audited, at random and
before anyone knows the answer. This module is what a labeller sees and how the
answer is written:

* one item per *pair* (``inputs_hash``), whichever tiers were asked about it:
  the label is the truth about the pair, so it is written to every drawn
  receipt of that pair at once;
* the item shows the question in plain words and the two memories in time
  order (earlier valid time first, as the dispute pass orders them), and never
  the judge's verdict, model or score: a labeller who sees them is not blind;
* the answer is one of the question's own options or "I can't tell"
  (``cannot_tell``, which never counts for or against a judge).

**Any caller may label** (D-49): an agent through MCP, the command line or HTTP, a script, or
a person on the dashboard. The label row records the labeller (``actor@channel``), its kind
(``person`` for the dashboard's door, otherwise ``agent``) and, for an agent, its own model id.
**A label counts toward a certificate cell only if the labeller's model differs from the judged
tier's model** (a person's label always counts; an agent that gives no model id does not count),
so a judge never grades itself. The labeller is not shown the judge's verdict, model or score:
blindness is unchanged, and an agent that names its model is simply not offered pairs that only
its own model judged. One label per receipt per labeller; ``retract`` takes a label back
(recorded in the decision log), and the receipt is waiting again.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence

from superlocalmemory.core import decisions
from superlocalmemory.core.audit_sampler import record_audit_label
from superlocalmemory.core.judge_calibration import CANNOT_TELL, COUNTING_LABEL
from superlocalmemory.core.judge_questions import QUESTIONS
from superlocalmemory.storage.json_columns import load_json_column

CHANNELS = decisions.CHANNELS
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
    # W2: the same time order as the judge's question (``a`` is the older memory).
    "DUPLICATE": {
        "question": "Memory A was saved before memory B. Do they hold the same information?",
        "options": (
            ("duplicate", "Yes: keeping either one loses nothing"),
            ("a_contains_b", "A holds everything in B and more"),
            ("b_contains_a", "B holds everything in A and more"),
            ("different", "Each holds something the other does not"),
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
                    WHERE l.receipt_id = r.receipt_id AND l.source = 'audit'
                      AND {counting})
   AND (? = '' OR r.model <> ?)
 GROUP BY r.inputs_hash
 ORDER BY MIN(r.decided_at), r.inputs_hash
 LIMIT ?
""".replace("{counting}", COUNTING_LABEL)


def _texts(conn: sqlite3.Connection, fact_ids: Sequence[str]) -> list[dict]:
    marks = ",".join("?" * len(fact_ids))
    rows = conn.execute(
        "SELECT f.fact_id, f.content, COALESCE(v.valid_from, f.valid_from, f.created_at) "
        "FROM atomic_facts f LEFT JOIN fact_temporal_validity v ON v.fact_id = f.fact_id "
        f"WHERE f.fact_id IN ({marks})", tuple(fact_ids)).fetchall()
    rows.sort(key=lambda r: r[2] or "")  # earlier valid time first: A, then B
    return [{"name": "AB"[i] if i < 2 else str(i + 1), "fact_id": f,
             "content": (c or "")[:TEXT_CHARS], "said_at": w} for i, (f, c, w) in enumerate(rows)]


#: How many waiting pairs are ranked by value before the first ``limit`` are shown.
POOL = 500


def _model(raw: str) -> str:
    return " ".join(str(raw or "").split())[:80]


def pending(conn: sqlite3.Connection, profile_id: str, limit: int = 10,
            labeller_model: str = "") -> list[dict]:
    """Pairs drawn for audit and not yet answered, without any verdict, the most
    valuable first (``core/audit_value``: answers that move a certificate, then risk).

    ``labeller_model`` (an agent's own model id) leaves out pairs that only that same model
    judged: its label could not count, so it is not offered the work.
    """
    from superlocalmemory.core import audit_value

    limit = max(1, min(int(limit), MAX_ITEMS))
    mine = _model(labeller_model)
    try:
        rows = conn.execute(_PENDING, (profile_id, mine, mine, POOL)).fetchall()
    except sqlite3.OperationalError:  # a store before M072
        return []
    out = []
    for inputs_hash, question_id, fact_ids_json, _when in audit_value.order(conn, profile_id, rows):
        if len(out) >= limit:
            break
        plain = PLAIN.get(question_id)
        if plain is None or question_id not in QUESTIONS:
            continue  # no wording a person can answer yet
        texts = _texts(conn, load_json_column(fact_ids_json, [], column="audit_questions.fact_ids_json"))
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
            "AND l.source = 'audit' AND " + COUNTING_LABEL + ")", (profile_id,)).fetchone()[0])
    except sqlite3.OperationalError:
        return 0


def answer(conn: sqlite3.Connection, *, profile_id: str, item_id: str, label: str,
           channel: str, actor: str = "", kind: str = "agent", model: str = "") -> dict:
    """Write one blind label to every drawn receipt of the pair that this labeller has not
    labelled and that has no counting label yet.

    Returns ``{"labelled": n, "counted": k}``: how many receipts it labelled and for how many
    the label counts toward a certificate cell (D-49: always for a person; for an agent only
    where its model, ``model``, is known and differs from the judged tier's). Raises
    ``LookupError`` when the item is not waiting for this labeller in this profile and
    ``ValueError`` for an answer that is not one of the question's options.
    """
    if channel not in CHANNELS:
        raise ValueError(f"channel must be one of {CHANNELS}")
    who = decisions.Decider(decisions.clean_actor(actor), channel)
    labeller = who.ref
    receipts = [r[0] for r in conn.execute(
        "SELECT r.receipt_id FROM judge_receipts r WHERE r.profile_id = ? AND r.inputs_hash = ? "
        "AND (r.audit_selected = 1 OR r.inbox_selected = 1) AND NOT EXISTS (SELECT 1 FROM "
        "judge_labels l WHERE l.receipt_id = r.receipt_id AND l.source = 'audit' AND "
        f"({COUNTING_LABEL} OR l.labeller = ?)) ORDER BY r.receipt_id",
        (profile_id, item_id, labeller))]
    if not receipts:
        raise LookupError(f"no audit item {item_id} waiting")
    conn.execute("SAVEPOINT audit")
    counted = 0
    try:
        for receipt_id in receipts:  # the question's own options are checked per receipt
            counted += bool(record_audit_label(conn, receipt_id, label, labeller=labeller,
                                               blind=True, kind=kind, model=model))
    except Exception:
        conn.execute("ROLLBACK TO audit")
        conn.execute("RELEASE audit")
        raise
    conn.execute("RELEASE audit")
    decisions.log(conn, profile_id=profile_id, kind=decisions.LABEL, subject_id=item_id,
                  action="label", decider=who,
                  summary="Gave a blind audit label" + ("" if counted else " (does not count)"),
                  detail={"receipts": receipts, "labeller": labeller, "kind": kind,
                          "model": _model(model), "counted": counted})
    return {"labelled": len(receipts), "counted": counted}


def retract(conn: sqlite3.Connection, *, profile_id: str, item_id: str, channel: str,
            actor: str = "") -> dict:
    """Take back the labels the newest un-retracted label decision for ``item_id`` wrote: the
    receipts are waiting again and no certificate rests on them. Recorded like any decision.
    ``LookupError`` when there is no label of this pair to retract."""
    if channel not in CHANNELS:
        raise ValueError(f"channel must be one of {CHANNELS}")
    who = decisions.Decider(decisions.clean_actor(actor), channel)
    before = decisions.open_decision(conn, profile_id, decisions.LABEL, item_id)
    if before is None:
        raise LookupError(f"no label of item {item_id} to retract")
    detail = before["detail"]
    receipts = list(detail.get("receipts") or [])
    marks = ",".join("?" * len(receipts)) or "''"
    removed = conn.execute(
        f"DELETE FROM judge_labels WHERE source = 'audit' AND labeller = ? AND receipt_id IN ({marks})",
        (detail.get("labeller", ""), *receipts)).rowcount
    decisions.mark_undone(conn, before["decision_id"], who)
    decisions.log(conn, profile_id=profile_id, kind=decisions.LABEL, subject_id=item_id,
                  action="retract", decider=who, summary="Retracted a blind audit label",
                  detail={"removed": removed})
    return {"retracted": removed}


__all__ = ["CHANNELS", "PLAIN", "answer", "pending", "pending_count", "retract"]
