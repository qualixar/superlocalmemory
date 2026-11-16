# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Applied corrections from outside the slot layer, each on its own private slot (D-40).

Before 4.2.30 the correction ledger applied cases ("this memory replaces that
one") that no slot knew about: 340 on the maintainer's store, most with no
slot key. A slot's history is a timeline of values, so each such case gets a
private slot of its own, ``legacy / case / <case_id>``: the predecessor as a
superseded revision whose value ended where the successor's began, the
successor as the head, the case id on the successor's revision. A chain (A
replaced by B, then B by C) stays one timeline: a case whose predecessor is the
current value of a legacy slot continues that slot. No meaning is guessed, only
the ledger's own links; a later alias (R-12) may fold such a slot into a real one.

The same holds for every applied case the ledger records later from outside
the slot layer (``remember(replaces=…)``, a review), so the timeline stays
whole. Cases the slot layer made itself (``slot_supersede``) already have
their slot. A case whose successor starts before its predecessor in valid time
is left out and counted (``legacy_inverted``): it would break I-05, and M071
measured none on the copy. Only the truth layer's own tables are written, so
shadow runs it too.
"""

from __future__ import annotations

import sqlite3
from collections import Counter

from superlocalmemory.truth import slot_store
from superlocalmemory.truth.slot_keys import SlotKey, value_hash
from superlocalmemory.truth.slot_resolver import REASON as SLOT_SUPERSEDE
from superlocalmemory.truth.validity import BEFORE, order

SUBJECT, RELATION = "legacy", "case"
KEY_SOURCE = "legacy_case"
DECIDED_BY = "ledger"
REASON = "legacy_applied_case"

_TODO = """
SELECT c.case_id, c.profile_id, c.scope, c.predecessor_fact_id, c.successor_fact_id,
       COALESCE(vp.valid_from, p.valid_from, p.created_at), p.content,
       COALESCE(vs.valid_from, s.valid_from, s.created_at), s.content
  FROM correction_cases c
  JOIN atomic_facts p ON p.fact_id = c.predecessor_fact_id
  JOIN atomic_facts s ON s.fact_id = c.successor_fact_id
  LEFT JOIN fact_temporal_validity vp ON vp.fact_id = p.fact_id
  LEFT JOIN fact_temporal_validity vs ON vs.fact_id = s.fact_id
 WHERE c.status = 'applied' AND c.reason_code != ?
   AND NOT EXISTS (SELECT 1 FROM memory_slots m
                    WHERE m.profile_id = c.profile_id AND m.scope = c.scope
                      AND m.subject_key = ? AND m.relation_key = ? AND m.qualifier = c.case_id)
   AND NOT EXISTS (SELECT 1 FROM slot_revisions r WHERE r.case_id = c.case_id)
 ORDER BY c.applied_at, c.case_id
 LIMIT ?
"""


#: The legacy slot whose current value is this case's predecessor, if any.
_CHAIN = """
SELECT s.slot_id, s.version, s.head_revision_id
  FROM memory_slots s
  JOIN slot_revisions r ON r.revision_id = s.head_revision_id
 WHERE s.key_source = 'legacy_case' AND s.profile_id = ? AND s.scope = ? AND r.fact_id = ?
"""


def _key(case_id: str, content: str) -> SlotKey:
    return SlotKey(subject_key=SUBJECT, relation_key=RELATION, qualifier=case_id,
                   cardinality="single", key_source=KEY_SOURCE, confidence=1.0,
                   value_hash=value_hash(content or ""))


def load(conn: sqlite3.Connection, limit: int = 200) -> dict[str, int]:
    """Put up to ``limit`` applied cases on their own slots. Returns counts."""
    counts: Counter[str] = Counter()
    try:
        rows = conn.execute(_TODO, (SLOT_SUPERSEDE, SUBJECT, RELATION, int(limit))).fetchall()
    except sqlite3.OperationalError:  # no ledger or no slot tables: nothing to load
        return {}
    for case_id, profile, scope, pred, succ, pred_from, pred_text, succ_from, succ_text in rows:
        if order(succ_from, pred_from) == BEFORE:
            counts["legacy_inverted"] += 1
            continue
        now = slot_store._now()
        head = slot_store._new_id("rev_")
        chain = conn.execute(_CHAIN, (profile, scope, pred)).fetchone()
        if chain is not None:
            # A replaced B, then B was replaced by C: one timeline, by the ledger's
            # own links (fact identity), so no meaning is guessed.
            slot_id, version, old = chain[0], int(chain[1]), chain[2]
            conn.execute("UPDATE slot_revisions SET state = 'superseded', superseded_by = ? "
                         "WHERE revision_id = ?", (head, old))
            counts["legacy_chained"] += 1
        else:
            slot_id, _head, version = slot_store._ensure_slot(
                conn, profile, scope, _key(case_id, succ_text), now)
            slot_store._insert_revision(
                conn, revision_id=slot_store._new_id("rev_"), slot_id=slot_id,
                profile_id=profile, fact_id=pred, key=_key(case_id, pred_text),
                state="superseded", reason=REASON, decided_by=DECIDED_BY,
                valid_from=pred_from, now=now, superseded_by=head)
        slot_store._insert_revision(
            conn, revision_id=head, slot_id=slot_id, profile_id=profile, fact_id=succ,
            key=_key(case_id, succ_text), state="head", reason=REASON, decided_by=DECIDED_BY,
            valid_from=succ_from, now=now)
        slot_store.record_case(conn, head, case_id)
        slot_store._move_head(conn, slot_id, version, head, "current", now)
        counts["legacy_loaded"] += 1
    return dict(counts)


__all__ = ["KEY_SOURCE", "REASON", "load"]
