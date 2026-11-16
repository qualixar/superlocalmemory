# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Keeping slots in step with the ledger and the fact lifecycle (R-09, Fable B8).

Nothing outside ``truth/`` writes the slot tables, and it should stay that way:
six paths take a fact out of current memory (``replaces=``, forget, archive,
merge, quarantine, ``update_memory``'s correction) and a seventh will come. So
instead of a hook in each, the slot pass reconciles from what the store says:

* **a head that left current memory** is handled as an erased head is
  (``trg_slot_revision_removed``, M069): the slot is emptied and re-formed from
  its live revisions, whose facts are queued to bind again; the head's own
  revision stays, as ``gone``, so the history keeps it. A merged head's
  survivor is queued too, so the value lives on where the merge put it;
* **a challenger that left current memory** has its open disputes dismissed:
  there is no longer a question to ask about it;
* **a rolled-back retirement** (the ledger restored the predecessor) puts the
  slot back the way it was before the decision and opens a dispute that only a
  person settles (``rolled_back``: no judge answers it, D-36's rule included),
  and records the rollback against the receipt that acted. That label is not
  blind and not an audit, so it never certifies anything (the gate counts blind
  audit labels only); it is the drift signal M-07 reads.
"""

from __future__ import annotations

import sqlite3
from collections import Counter
from datetime import UTC, datetime

from superlocalmemory.truth import slot_store

GONE = "gone"
#: A dispute only a person settles, and the label a rollback writes.
ROLLED_BACK = "rolled_back"
ROLLBACK_SOURCE = "rollback"

_NOT_LIVE = """(f.fact_id IS NULL OR f.quarantined = 1 OR f.lifecycle = 'archived'
    OR f.merged_into IS NOT NULL OR v.system_expired_at IS NOT NULL)"""

_DEAD_HEADS = f"""
SELECT s.slot_id, s.version, r.revision_id, f.merged_into
  FROM memory_slots s
  JOIN slot_revisions r ON r.revision_id = s.head_revision_id
  LEFT JOIN atomic_facts f ON f.fact_id = r.fact_id
  LEFT JOIN fact_temporal_validity v ON v.fact_id = r.fact_id
 WHERE {_NOT_LIVE}
 LIMIT ?
"""

_DEAD_CHALLENGERS = f"""
SELECT d.dispute_id
  FROM slot_disputes d
  JOIN slot_revisions r ON r.revision_id = d.revision_b
  LEFT JOIN atomic_facts f ON f.fact_id = r.fact_id
  LEFT JOIN fact_temporal_validity v ON v.fact_id = r.fact_id
 WHERE d.state = 'open' AND {_NOT_LIVE}
 LIMIT ?
"""

#: The newer revision of a truth-layer retirement the ledger has since rolled back,
#: still the slot's head, with the revision it had superseded.
_ROLLED_BACK = """
SELECT n.revision_id, n.slot_id, n.profile_id, n.receipt_id, p.revision_id, s.version
  FROM slot_revisions n
  JOIN memory_slots s ON s.slot_id = n.slot_id AND s.head_revision_id = n.revision_id
  JOIN correction_cases c ON c.case_id = n.case_id
  JOIN slot_revisions p ON p.slot_id = n.slot_id AND p.superseded_by = n.revision_id
 WHERE c.status = 'rolled_back'
 LIMIT ?
"""

_QUEUE = ("INSERT OR REPLACE INTO slot_pass_queue (fact_id, reason, attempts, next_at, queued_at) "
          "VALUES (?, 'rebind', 0, ?, ?)")


def _now() -> str:
    return datetime.now(UTC).isoformat()


def repair(conn: sqlite3.Connection, *, limit: int = 200) -> Counter[str]:
    """One bounded reconciliation; each slot in its own savepoint."""
    counts: Counter[str] = Counter()
    for row in conn.execute(_DEAD_HEADS, (limit,)).fetchall():
        _one(conn, counts, "head_gone", _vacate, row)
    for (dispute_id,) in conn.execute(_DEAD_CHALLENGERS, (limit,)).fetchall():
        slot_store.dismiss_dispute(conn, dispute_id, by="lifecycle:gone")
        counts["challenger_gone"] += 1
    for row in conn.execute(_ROLLED_BACK, (limit,)).fetchall():
        _one(conn, counts, "rolled_back", _restore, row)
    return counts


def _one(conn, counts, name, fn, row) -> None:
    conn.execute("SAVEPOINT slot_repair")
    try:
        fn(conn, row)
    except slot_store.SlotConflictError:
        conn.execute("ROLLBACK TO slot_repair")
        counts[f"{name}_conflict"] += 1
    else:
        counts[name] += 1
    finally:
        conn.execute("RELEASE slot_repair")


def _vacate(conn: sqlite3.Connection, row: tuple) -> None:
    slot_id, version, head, merged_into = row
    now = _now()
    slot_store._move_head(conn, slot_id, version, None, "empty", now)
    conn.execute("UPDATE slot_revisions SET state = ? WHERE revision_id = ?", (GONE, head))
    for (dispute_id,) in conn.execute(
            "SELECT dispute_id FROM slot_disputes WHERE state = 'open' AND slot_id = ?",
            (slot_id,)).fetchall():
        slot_store.dismiss_dispute(conn, dispute_id, by="lifecycle:head_gone")
    survivors = [r[0] for r in conn.execute(
        "SELECT fact_id FROM slot_revisions WHERE slot_id = ? "
        "AND state NOT IN ('superseded', 'unbound', ?)", (slot_id, GONE))]
    conn.execute("DELETE FROM slot_revisions WHERE slot_id = ? "
                 "AND state NOT IN ('superseded', 'unbound', ?)", (slot_id, GONE))
    for fact_id in survivors + ([merged_into] if merged_into else []):
        conn.execute(_QUEUE, (fact_id, now, now))


def _restore(conn: sqlite3.Connection, row: tuple) -> None:
    newer, slot_id, profile_id, receipt_id, older, version = row
    now = _now()
    slot_store._move_head(conn, slot_id, version, older, "disputed", now)
    conn.execute("UPDATE slot_revisions SET state = 'head', superseded_by = NULL "
                 "WHERE revision_id = ?", (older,))
    conn.execute("UPDATE slot_revisions SET state = 'disputed' WHERE revision_id = ?", (newer,))
    slot_store.open_dispute(conn, slot_id=slot_id, profile_id=profile_id, revision_a=older,
                            revision_b=newer, reason_code=ROLLED_BACK,
                            opened_by="ledger:rollback")
    if receipt_id:
        conn.execute("INSERT OR IGNORE INTO judge_labels (receipt_id, label, source, labeller, "
                     "blind, labelled_at) VALUES (?, ?, ?, 'ledger', 0, ?)",
                     (receipt_id, ROLLED_BACK, ROLLBACK_SOURCE, now))


__all__ = ["GONE", "ROLLBACK_SOURCE", "ROLLED_BACK", "repair"]
