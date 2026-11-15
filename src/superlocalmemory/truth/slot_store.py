# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Slots, revisions and disputes: the decisions, recorded (tables from M069).

``bind`` takes a fact with its slot key and decides, deterministically:

* **head**: the slot had no current value; this fact is it.
* **duplicate**: it states the head's value again.
* **compatible**: a plural slot ("tools Alex uses"); both stay current.
* **historical**: it is older than the head (a backfill of the past); it joins
  the history behind the head and replaces nothing.
* **supersede**: a newer value for a single-valued slot with a confident key.
  The head moves (compare-and-swap on ``memory_slots.version``) and the old
  revision is marked superseded. Retiring the old *fact* is a separate step
  through the correction ledger (``slot_resolver``), so it stays reviewable
  and exactly reversible; ``case_id`` records it once done.
* **dispute**: a newer value, but the key is not confident enough to replace
  anything. Both sides stay, a dispute is opened, and recall can show both.

Every function takes the caller's connection and never commits: the caller's
transaction owns the write. Ids and hashes only, never memory text.
"""

from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime

from superlocalmemory.truth.slot_keys import SlotKey
from superlocalmemory.truth.validity import earlier, normalise

#: A key below this confidence opens a dispute instead of superseding.
SUPERSEDE_CONFIDENCE = 0.75

HEAD, DUPLICATE, COMPATIBLE, HISTORICAL, SUPERSEDE, DISPUTE = (
    "head", "duplicate", "compatible", "historical", "supersede", "dispute")


class SlotConflictError(RuntimeError):
    """The slot changed between read and write; the caller retries."""


@dataclass(frozen=True)
class Binding:
    outcome: str
    slot_id: str
    revision_id: str
    #: The fact the new one replaces (supersede) or argues with (dispute).
    other_fact_id: str | None = None
    other_revision_id: str | None = None
    dispute_id: str | None = None


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _digest(*parts: str) -> str:
    return hashlib.sha256("\0".join(parts).encode("utf-8")).hexdigest()[:32]


def slot_id_for(profile_id: str, scope: str, key: SlotKey) -> str:
    return "slot_" + _digest(profile_id, scope, key.subject_key, key.relation_key, key.qualifier)


def revision_id_for(slot_id: str, fact_id: str) -> str:
    return "rev_" + _digest(slot_id, fact_id)


def _ensure_slot(conn: sqlite3.Connection, slot_id: str, profile_id: str, scope: str,
                 key: SlotKey, now: str) -> tuple[str | None, int]:
    conn.execute(
        "INSERT OR IGNORE INTO memory_slots (slot_id, profile_id, scope, subject_key, "
        "relation_key, qualifier, cardinality, key_source, state, version, created_at, "
        "updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'empty', 0, ?, ?)",
        (slot_id, profile_id, scope, key.subject_key, key.relation_key, key.qualifier,
         key.cardinality, key.key_source, now, now))
    row = conn.execute("SELECT head_revision_id, version FROM memory_slots WHERE slot_id = ?",
                       (slot_id,)).fetchone()
    return row[0], int(row[1])


def _head(conn: sqlite3.Connection, revision_id: str | None) -> tuple | None:
    if revision_id is None:
        return None
    return conn.execute(
        "SELECT revision_id, fact_id, value_hash, valid_from, recorded_at FROM slot_revisions "
        "WHERE revision_id = ?", (revision_id,)).fetchone()


def _move_head(conn: sqlite3.Connection, slot_id: str, version: int, head: str | None,
               state: str, now: str) -> None:
    cursor = conn.execute(
        "UPDATE memory_slots SET head_revision_id = ?, state = ?, version = version + 1, "
        "updated_at = ? WHERE slot_id = ? AND version = ?", (head, state, now, slot_id, version))
    if cursor.rowcount != 1:
        raise SlotConflictError(f"slot {slot_id} changed while it was being decided")


def _insert_revision(conn: sqlite3.Connection, *, revision_id: str, slot_id: str,
                     profile_id: str, fact_id: str, key: SlotKey, state: str,
                     reason: str, decided_by: str, valid_from: str | None, now: str,
                     superseded_by: str | None = None) -> None:
    conn.execute(
        "INSERT INTO slot_revisions (revision_id, slot_id, profile_id, fact_id, value_hash, "
        "state, superseded_by, decided_by, reason_code, valid_from, recorded_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (revision_id, slot_id, profile_id, fact_id, key.value_hash, state, superseded_by,
         decided_by, reason, valid_from, now))


def _decide(key: SlotKey, head: tuple | None, valid_from: str | None) -> str:
    if head is None:
        return HEAD
    if key.value_hash == head[2]:
        return DUPLICATE
    if key.cardinality == "plural":
        return COMPATIBLE
    if earlier(valid_from, head[3] or head[4]):
        return HISTORICAL
    return SUPERSEDE if key.confidence >= SUPERSEDE_CONFIDENCE else DISPUTE


def bind(conn: sqlite3.Connection, *, profile_id: str, scope: str, fact_id: str, key: SlotKey,
         valid_from: str | None = None, decided_by: str = "deterministic") -> Binding:
    """Record ``fact_id`` as a revision of its slot and decide what it does."""
    now = _now()
    slot_id = slot_id_for(profile_id, scope, key)
    revision_id = revision_id_for(slot_id, fact_id)
    existing = conn.execute("SELECT state FROM slot_revisions WHERE revision_id = ?",
                            (revision_id,)).fetchone()
    if existing is not None:  # the same fact bound twice: the first decision stands
        return Binding(outcome=str(existing[0]), slot_id=slot_id, revision_id=revision_id)
    head_id, version = _ensure_slot(conn, slot_id, profile_id, scope, key, now)
    head = _head(conn, head_id)
    when = normalise(valid_from)
    outcome = _decide(key, head, when)
    common = {"revision_id": revision_id, "slot_id": slot_id, "profile_id": profile_id,
              "fact_id": fact_id, "key": key, "decided_by": decided_by, "valid_from": when,
              "now": now}
    if outcome == HEAD:
        _insert_revision(conn, state="head", reason="first_value", **common)
        _move_head(conn, slot_id, version, revision_id, "current", now)
        return Binding(HEAD, slot_id, revision_id)
    if outcome in (DUPLICATE, COMPATIBLE):
        _insert_revision(conn, state=outcome, reason=f"{outcome}_value", **common)
        return Binding(outcome, slot_id, revision_id, head[1], head[0])
    if outcome == HISTORICAL:
        _insert_revision(conn, state="historical", reason="older_than_head",
                         superseded_by=head[0], **common)
        return Binding(HISTORICAL, slot_id, revision_id, head[1], head[0])
    if outcome == SUPERSEDE:
        _insert_revision(conn, state="head", reason="newer_value", **common)
        _move_head(conn, slot_id, version, revision_id, "current", now)
        conn.execute("UPDATE slot_revisions SET state = 'superseded', superseded_by = ? "
                     "WHERE revision_id = ?", (revision_id, head[0]))
        return Binding(SUPERSEDE, slot_id, revision_id, head[1], head[0])
    _insert_revision(conn, state="disputed", reason="uncertain_key", **common)
    dispute_id = open_dispute(conn, slot_id=slot_id, profile_id=profile_id,
                              revision_a=head[0], revision_b=revision_id,
                              reason_code="uncertain_key", opened_by=decided_by)
    _move_head(conn, slot_id, version, head[0], "disputed", now)
    return Binding(DISPUTE, slot_id, revision_id, head[1], head[0], dispute_id)


def open_dispute(conn: sqlite3.Connection, *, slot_id: str, profile_id: str, revision_a: str,
                 revision_b: str, reason_code: str, opened_by: str) -> str:
    dispute_id = "dsp_" + _digest(slot_id, revision_a, revision_b, _now())
    conn.execute(
        "INSERT INTO slot_disputes (dispute_id, slot_id, profile_id, revision_a, revision_b, "
        "reason_code, opened_by, opened_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (dispute_id, slot_id, profile_id, revision_a, revision_b, reason_code, opened_by,
         _now()))
    return dispute_id


def record_case(conn: sqlite3.Connection, revision_id: str, case_id: str) -> None:
    """Note the correction case that retired the fact this revision superseded."""
    conn.execute("UPDATE slot_revisions SET case_id = ? WHERE revision_id = ?",
                 (case_id, revision_id))


RESOLUTIONS = ("a", "b", "both", "neither")


def resolve_dispute(conn: sqlite3.Connection, dispute_id: str, resolution: str, *,
                    resolved_by: str) -> tuple[str, ...]:
    """Settle an open dispute. Returns the fact ids the caller must retire.

    ``a`` keeps the earlier value, ``b`` takes the newer one, ``both`` keeps
    both as compatible values, ``neither`` says neither is true now: both stay
    on record as history and the slot has no current value until a new memory
    arrives. Retiring facts goes through the correction ledger, by the caller.
    """
    if resolution not in RESOLUTIONS:
        raise ValueError(f"resolution must be one of {RESOLUTIONS}")
    row = conn.execute("SELECT slot_id, revision_a, revision_b FROM slot_disputes "
                       "WHERE dispute_id = ? AND state = 'open'", (dispute_id,)).fetchone()
    if row is None:
        raise LookupError(f"no open dispute {dispute_id}")
    slot_id, rev_a, rev_b = row
    now = _now()
    version = int(conn.execute("SELECT version FROM memory_slots WHERE slot_id = ?",
                               (slot_id,)).fetchone()[0])
    winner, loser = {"a": (rev_a, rev_b), "b": (rev_b, rev_a)}.get(resolution, (None, None))
    retire: tuple[str, ...] = ()
    if winner is not None:
        conn.execute("UPDATE slot_revisions SET state = 'head', superseded_by = NULL "
                     "WHERE revision_id = ?", (winner,))
        conn.execute("UPDATE slot_revisions SET state = 'superseded', superseded_by = ? "
                     "WHERE revision_id = ?", (winner, loser))
        _move_head(conn, slot_id, version, winner, "current", now)
        retire = (conn.execute("SELECT fact_id FROM slot_revisions WHERE revision_id = ?",
                               (loser,)).fetchone()[0],)
    elif resolution == "both":
        conn.execute("UPDATE slot_revisions SET state = 'compatible' WHERE revision_id = ?",
                     (rev_b,))
        _move_head(conn, slot_id, version, rev_a, "current", now)
    else:
        conn.execute("UPDATE slot_revisions SET state = 'historical', superseded_by = NULL "
                     "WHERE revision_id IN (?, ?)", (rev_a, rev_b))
        _move_head(conn, slot_id, version, None, "empty", now)
        retire = tuple(r[0] for r in conn.execute(
            "SELECT fact_id FROM slot_revisions WHERE revision_id IN (?, ?) ORDER BY fact_id",
            (rev_a, rev_b)))
    conn.execute("UPDATE slot_disputes SET state = 'resolved', resolution = ?, resolved_by = ?, "
                 "resolved_at = ? WHERE dispute_id = ?", (resolution, resolved_by, now,
                                                         dispute_id))
    return retire


__all__ = ["Binding", "RESOLUTIONS", "resolve_dispute", "COMPATIBLE", "DISPUTE", "DUPLICATE", "HEAD", "HISTORICAL",
           "SUPERSEDE", "SUPERSEDE_CONFIDENCE", "SlotConflictError", "bind", "open_dispute",
           "record_case", "revision_id_for", "slot_id_for"]
