# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Slots, revisions and disputes: the decisions, recorded (tables from M069).

``bind`` takes a fact with its slot key and decides, deterministically:

* **head**: the slot had no current value; this fact is it.
* **duplicate**: it states the head's value again.
* **compatible**: a plural slot ("tools Alex uses"); both stay current.
* **unbound**: a judge or person said the key was wrong; it stays on record so
  the fact is never bound to this key again, and is no value of the slot.
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

Slot and revision ids are opaque surrogates (R-12), never derived from a key: a
slot is found by its natural key ``(profile, scope, subject, relation,
qualifier)`` or by an alias of it (``slot_key_alias``). ``rekey_slot`` moves a
slot to a new key -- the move from ``name:`` to ``entity:<id>`` subjects --
and keeps the old key as an alias, so nothing that names the slot, its
revisions or its disputes is orphaned, and a fact still keyed the old way
lands in the same slot.
"""

from __future__ import annotations

import sqlite3
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime

from superlocalmemory.truth.slot_keys import EXPLICIT, EXPLICIT_UNVALUED, SlotKey, is_stated
from superlocalmemory.truth.validity import BEFORE, INCOMPARABLE, order, span

HEAD, DUPLICATE, COMPATIBLE, HISTORICAL, SUPERSEDE, DISPUTE = (
    "head", "duplicate", "compatible", "historical", "supersede", "dispute")
#: Why a dispute was opened. ``explicit_value_changed`` is the one reason the
#: rule tier ``explicit-v1`` answers (D-36).
EXPLICIT_VALUE_CHANGED = "explicit_value_changed"
INCOMPARABLE_TIME = "incomparable_time"
#: A value older in valid time than the head (R-10): it is history, but making it
#: history retires a fact, so it is a dispute, never a silent label. Its sides
#: are asked in time order (the newcomer is the older text) and ``replaces``
#: settles it for the head. ``explicit_older_value`` when both values were stated.
OLDER_THAN_HEAD = "older_than_head"
EXPLICIT_OLDER_VALUE = "explicit_older_value"
OLDER_REASONS = frozenset({OLDER_THAN_HEAD, EXPLICIT_OLDER_VALUE})


class SlotConflictError(RuntimeError):
    """The slot changed between read and write; the caller retries."""


class SlotKeyTakenError(ValueError):
    """The new key already names another slot: that is a merge, not a re-key."""


class StaleDisputeError(LookupError):
    """The dispute no longer names the slot's current value; it was dismissed."""


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


def _new_id(prefix: str) -> str:
    return prefix + uuid.uuid4().hex


_NATURAL = "profile_id = ? AND scope = ? AND subject_key = ? AND relation_key = ? AND qualifier = ?"


def _natural(profile_id: str, scope: str, key: SlotKey) -> tuple[str, ...]:
    return (profile_id, scope, key.subject_key, key.relation_key, key.qualifier)


def find_slot(conn: sqlite3.Connection, profile_id: str, scope: str, key: SlotKey) -> str | None:
    """The slot this key names, directly or through an alias; None if there is none."""
    args = _natural(profile_id, scope, key)
    row = conn.execute(f"SELECT slot_id FROM memory_slots WHERE {_NATURAL}", args).fetchone()  # noqa: S608
    if row is None:
        row = conn.execute(f"SELECT slot_id FROM slot_key_alias WHERE {_NATURAL}",  # noqa: S608
                           args).fetchone()
    return str(row[0]) if row else None


def _ensure_slot(conn: sqlite3.Connection, profile_id: str, scope: str, key: SlotKey,
                 now: str) -> tuple[str, str | None, int]:
    slot_id = find_slot(conn, profile_id, scope, key)
    if slot_id is None:
        slot_id = _new_id("slot_")
        conn.execute(
            "INSERT INTO memory_slots (slot_id, profile_id, scope, subject_key, "
            "relation_key, qualifier, cardinality, key_source, state, version, created_at, "
            "updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'empty', 0, ?, ?)",
            (slot_id, profile_id, scope, key.subject_key, key.relation_key, key.qualifier,
             key.cardinality, key.key_source, now, now))
    row = conn.execute("SELECT head_revision_id, version FROM memory_slots WHERE slot_id = ?",
                       (slot_id,)).fetchone()
    return slot_id, row[0], int(row[1])


def rekey_slot(conn: sqlite3.Connection, slot_id: str, new_key: SlotKey, *, by: str,
               reason_code: str) -> None:
    """Give ``slot_id`` a new natural key; its old key stays an alias of it.

    Ids do not change, so revisions, disputes and receipts keep pointing at the
    slot. Raises ``SlotKeyTakenError`` when the new key already names a
    different slot (two slots being the same is a merge, decided separately).
    """
    row = conn.execute("SELECT profile_id, scope, subject_key, relation_key, qualifier "
                       "FROM memory_slots WHERE slot_id = ?", (slot_id,)).fetchone()
    if row is None:
        raise LookupError(f"no slot {slot_id}")
    profile_id, scope = str(row[0]), str(row[1])
    holder = find_slot(conn, profile_id, scope, new_key)
    if holder is not None and holder != slot_id:
        raise SlotKeyTakenError(f"key already names slot {holder}")
    now = _now()
    conn.execute(f"DELETE FROM slot_key_alias WHERE {_NATURAL}",  # noqa: S608
                 _natural(profile_id, scope, new_key))  # the new key is no longer an alias
    if tuple(row[2:]) != (new_key.subject_key, new_key.relation_key, new_key.qualifier):
        conn.execute(
            "INSERT OR REPLACE INTO slot_key_alias (profile_id, scope, subject_key, relation_key, "
            "qualifier, slot_id, reason_code, aliased_by, aliased_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", (*row, slot_id, reason_code, by, now))
    conn.execute(
        "UPDATE memory_slots SET subject_key = ?, relation_key = ?, qualifier = ?, "
        "version = version + 1, updated_at = ? WHERE slot_id = ?",
        (new_key.subject_key, new_key.relation_key, new_key.qualifier, now, slot_id))


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
    when = order(valid_from, head[3] or head[4])
    if when == BEFORE:
        return HISTORICAL
    if when == INCOMPARABLE:
        return DISPUTE  # cannot tell which value is newer: a person or a judge decides
    # No key moves a head on its own (I-10: a head move is a memory change). A
    # later value on an explicit key is a ``replaces`` question for the rule tier
    # ``explicit-v1``, which acts only on its own certificate (D-36); every other
    # new value is a question for a person or a certified judge (D-10).
    return DISPUTE


def _dispute_reason(key: SlotKey, when: str | None, head: tuple) -> str:
    if order(when, head[3] or head[4]) == INCOMPARABLE:
        return INCOMPARABLE_TIME
    if key.key_source == EXPLICIT and is_stated(key.value_hash) and is_stated(head[2]):
        return EXPLICIT_VALUE_CHANGED  # a stated value against a stated value only
    return key.key_source if key.key_source == EXPLICIT_UNVALUED else "pattern_key"


def bind(conn: sqlite3.Connection, *, profile_id: str, scope: str, fact_id: str, key: SlotKey,
         valid_from: str | None = None, decided_by: str = "deterministic") -> Binding:
    """Record ``fact_id`` as a revision of its slot and decide what it does."""
    now = _now()
    slot_id, head_id, version = _ensure_slot(conn, profile_id, scope, key, now)
    existing = conn.execute("SELECT revision_id, state FROM slot_revisions "
                            "WHERE slot_id = ? AND fact_id = ?", (slot_id, fact_id)).fetchone()
    if existing is not None:  # the same fact bound twice: the first decision stands
        return Binding(outcome=str(existing[1]), slot_id=slot_id, revision_id=str(existing[0]))
    revision_id = _new_id("rev_")
    head = _head(conn, head_id)
    # Kept as written: a bare date must stay a date (an interval), not become midnight UTC.
    when = valid_from.strip() if isinstance(valid_from, str) and valid_from.strip() else None
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
    if outcome == HISTORICAL:  # R-10: history retires a fact, so it is a question
        stated = key.key_source == EXPLICIT and is_stated(key.value_hash) and is_stated(head[2])
        reason = EXPLICIT_OLDER_VALUE if stated else OLDER_THAN_HEAD
    else:
        reason = _dispute_reason(key, when, head)
    _insert_revision(conn, state="disputed", reason=reason, **common)
    dispute_id = open_dispute(conn, slot_id=slot_id, profile_id=profile_id,
                              revision_a=head[0], revision_b=revision_id,
                              reason_code=reason, opened_by=decided_by)
    _move_head(conn, slot_id, version, head[0], "disputed", now)
    return Binding(DISPUTE, slot_id, revision_id, head[1], head[0], dispute_id)


def open_dispute(conn: sqlite3.Connection, *, slot_id: str, profile_id: str, revision_a: str,
                 revision_b: str, reason_code: str, opened_by: str) -> str:
    dispute_id = _new_id("dsp_")
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


RESOLUTIONS = ("a", "b", "both", "same", "unrelated")


def resolve_dispute(conn: sqlite3.Connection, dispute_id: str, resolution: str, *,
                    resolved_by: str) -> tuple[str, ...]:
    """Settle an open dispute. Returns the fact ids the caller must retire.

    ``a`` keeps the earlier value, ``b`` takes the newer one, ``both`` keeps
    both as compatible values, ``same`` marks the newer one a duplicate of the
    head, and ``unrelated`` says the key was wrong: the newer memory leaves the
    slot (its revision stays, ``unbound``, reason ``keying_error``, so the slot
    pass never binds it to this key again) and stays current. Retiring facts
    goes through the correction ledger, by the caller. (R-22 removed a
    ``neither`` resolution: no question produced it and no caller could retire
    two facts without a successor.)
    """
    if resolution not in RESOLUTIONS:
        raise ValueError(f"resolution must be one of {RESOLUTIONS}")
    row = conn.execute("SELECT slot_id, revision_a, revision_b, reason_code FROM slot_disputes "
                       "WHERE dispute_id = ? AND state = 'open'", (dispute_id,)).fetchone()
    if row is None:
        raise LookupError(f"no open dispute {dispute_id}")
    slot_id, rev_a, rev_b, reason_code = row
    now = _now()
    head, version = conn.execute("SELECT head_revision_id, version FROM memory_slots "
                                 "WHERE slot_id = ?", (slot_id,)).fetchone()
    if head != rev_a:
        # The slot moved on since this dispute opened: deciding it now would act on
        # a value that is no longer current. Nothing is written here, so the caller
        # can roll back anything it did first and then ``dismiss_dispute``.
        raise StaleDisputeError(f"dispute {dispute_id} no longer names the current value")
    version = int(version)
    winner, loser = {"a": (rev_a, rev_b), "b": (rev_b, rev_a)}.get(resolution, (None, None))
    retire: tuple[str, ...] = ()
    if winner is not None:
        conn.execute("UPDATE slot_revisions SET state = 'head', superseded_by = NULL "
                     "WHERE revision_id = ?", (winner,))
        # An older value keeping the head ends where the next value in time begins,
        # which may be a value the head itself superseded (I-05: no overlaps).
        after = (_successor_in_time(conn, slot_id, loser) or winner
                 if reason_code in OLDER_REASONS and winner == rev_a else winner)
        conn.execute("UPDATE slot_revisions SET state = 'superseded', superseded_by = ? "
                     "WHERE revision_id = ?", (after, loser))
        _move_head(conn, slot_id, version, winner, "current", now)
        retire = (conn.execute("SELECT fact_id FROM slot_revisions WHERE revision_id = ?",
                               (loser,)).fetchone()[0],)
    elif resolution in ("both", "same"):
        state = "duplicate" if resolution == "same" else "compatible"
        conn.execute("UPDATE slot_revisions SET state = ? WHERE revision_id = ?",
                     (state, rev_b))
        _move_head(conn, slot_id, version, rev_a, "current", now)
    else:  # unrelated: the key was wrong (R-22)
        conn.execute("UPDATE slot_revisions SET state = 'unbound', superseded_by = NULL, "
                     "reason_code = 'keying_error' WHERE revision_id = ?", (rev_b,))
        _move_head(conn, slot_id, version, rev_a, "current", now)
    conn.execute("UPDATE slot_disputes SET state = 'resolved', resolution = ?, resolved_by = ?, "
                 "resolved_at = ? WHERE dispute_id = ?", (resolution, resolved_by, now,
                                                         dispute_id))
    _reanchor(conn, slot_id, dispute_id, now)
    return retire


def _when(row: tuple) -> str | None:
    return row[1] or row[2]


def _successor_in_time(conn: sqlite3.Connection, slot_id: str, revision_id: str) -> str | None:
    """The slot's value that began first after ``revision_id``'s value did."""
    mine = conn.execute("SELECT revision_id, valid_from, recorded_at FROM slot_revisions "
                        "WHERE revision_id = ?", (revision_id,)).fetchone()
    later = [r for r in conn.execute(
        "SELECT revision_id, valid_from, recorded_at FROM slot_revisions WHERE slot_id = ? "
        "AND revision_id != ? AND state IN ('head', 'superseded')", (slot_id, revision_id))
        if order(_when(mine), _when(r)) == BEFORE]
    if not later:
        return None
    return min(later, key=lambda r: span(_when(r))[0])[0]


def successor_fact(conn: sqlite3.Connection, dispute_id: str, fact_id: str) -> str | None:
    """The fact whose value followed ``fact_id``'s in the dispute's slot (for the ledger)."""
    row = conn.execute(
        "SELECT w.fact_id FROM slot_revisions r JOIN slot_revisions w "
        "ON w.revision_id = r.superseded_by WHERE r.fact_id = ? AND r.slot_id = "
        "(SELECT slot_id FROM slot_disputes WHERE dispute_id = ?)", (fact_id, dispute_id)).fetchone()
    return row[0] if row else None


def dismiss_dispute(conn: sqlite3.Connection, dispute_id: str, *, by: str) -> None:
    conn.execute("UPDATE slot_disputes SET state = 'dismissed', resolved_by = ?, resolved_at = ? "
                 "WHERE dispute_id = ? AND state = 'open'", (by, _now(), dispute_id))


def _reanchor(conn: sqlite3.Connection, slot_id: str, settled: str, now: str) -> None:
    """Re-open the slot's other open disputes against its head, as NEW disputes.

    A dispute is a question about one pair. When the head moves, (old head, C)
    is no longer the question; (new head, C) is. Keeping the id would let a
    verdict already given for the old pair stand for the new one, so the old
    dispute is dismissed and a new one opened (judged afresh: its pair, and so
    its keyed inputs hash, differs).
    """
    head = conn.execute("SELECT head_revision_id FROM memory_slots WHERE slot_id = ?",
                        (slot_id,)).fetchone()[0]
    others = conn.execute("SELECT dispute_id, profile_id, revision_a, revision_b, reason_code "
                          "FROM slot_disputes WHERE slot_id = ? AND state = 'open' "
                          "AND dispute_id != ?", (slot_id, settled)).fetchall()
    for dispute_id, profile_id, rev_a, rev_b, reason in others:
        if rev_a == head:
            continue
        conn.execute("UPDATE slot_disputes SET state = 'dismissed', resolved_by = 'reanchor', "
                     "resolved_at = ? WHERE dispute_id = ?", (now, dispute_id))
        if head is not None and rev_b != head:
            open_dispute(conn, slot_id=slot_id, profile_id=profile_id, revision_a=head,
                         revision_b=rev_b, reason_code=reason, opened_by="reanchor")
    state = "disputed" if conn.execute(
        "SELECT 1 FROM slot_disputes WHERE slot_id = ? AND state = 'open'", (slot_id,)
    ).fetchone() else ("current" if head is not None else "empty")
    conn.execute("UPDATE memory_slots SET state = ? WHERE slot_id = ?", (state, slot_id))


__all__ = ["Binding", "COMPATIBLE", "DISPUTE", "DUPLICATE", "HEAD", "HISTORICAL", "RESOLUTIONS",
           "EXPLICIT_OLDER_VALUE", "EXPLICIT_VALUE_CHANGED", "INCOMPARABLE_TIME", "OLDER_REASONS",
           "OLDER_THAN_HEAD", "SUPERSEDE", "SlotConflictError",
           "SlotKeyTakenError", "StaleDisputeError", "bind",
           "dismiss_dispute", "find_slot", "open_dispute", "record_case", "rekey_slot",
           "resolve_dispute", "successor_fact"]
