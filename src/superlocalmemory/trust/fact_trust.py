# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Each fact's origin class and taint, computed from its origin signals (M-15, WS-5).

* **Class** of a fact: ``origin_class.classify`` over the ``fact_origin`` row
  (M074). A fact with no such row (saved before 4.2.30) is ``unknown``.
* **Taint** of a fact: the lattice join (the lowest class) over the fact and
  everything it was derived from. A fact is derived from ``parent_fact_id``
  (a correction's predecessor) and, when it is a derived fact or its parent
  memory is not its own, from the source facts of ``parent_memory_id`` (the
  facts of that memory that were not themselves derived). Siblings that merely
  share a memory do not taint each other: a model's rewrite of a user's
  sentence does not make the sentence itself model-written. The walk is bounded
  in depth and in facts looked at, and cycle-safe.

Persisted in ``fact_trust`` (M079) with the mapping row that decided the class
(``basis``): at admission (``trust.taint`` on or in shadow), and by
:func:`run_batch` in the maintenance cycle. Reading never writes:
:func:`origin_of` and :func:`for_results` only select, and compute from the raw
signals what no row yet holds. ``mark_confirmed`` is the one door by which a
person's approval raises a memory to ``user_confirmed`` (package B calls it).

Ids and class names only; no memory text is read or logged.
"""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from superlocalmemory.trust import origin_class as oc

logger = logging.getLogger(__name__)

#: Bounds of the parent walk (routing only: they cap work, they decide nothing).
MAX_DEPTH = 6
MAX_FACTS = 60
MAX_PARENT_FACTS = 20
BATCH = 500
CONFIRMED = "confirmed"

Query = Callable[[str, tuple], Sequence[Sequence[Any]]]

_ORIGIN = ("SELECT f.memory_id, o.channel, o.transform, o.source_type, o.agent_id, o.speaker, "
           "o.role, o.parent_memory_id, o.parent_fact_id, o.agent_id_source, f.created_at "
           "FROM atomic_facts f LEFT JOIN fact_origin o ON o.fact_id = f.fact_id "
           "WHERE f.fact_id = ?")
_MEMORY_SOURCES = ("SELECT f.fact_id FROM atomic_facts f JOIN fact_origin o ON "
                   "o.fact_id = f.fact_id WHERE f.memory_id = ? AND f.fact_id != ? "
                   "AND o.channel != 'derived' ORDER BY f.rowid LIMIT ?")


@dataclass(frozen=True)
class Trust:
    origin_class: str
    taint_class: str
    basis: str

    def as_dict(self) -> dict[str, str]:
        """What a result or a person is shown: both classes in words and in ids."""
        return {"class": self.origin_class, "taint": self.taint_class,
                "label": oc.label(self.origin_class), "taint_label": oc.label(self.taint_class),
                "basis": self.basis}


#: ``fact_origin`` (M074) is written for every save from the moment it exists on a store, so a
#: fact created before it was applied here was saved by a release before 4.2.30.
_ORIGIN_MIGRATION = "M074_fact_origin"
_CUTOFF_SQL = "SELECT applied_at FROM migration_log WHERE name = ?"
#: Rows are only appended, so the first by rowid is the earliest recorded (no scan, no index).
_FIRST_ORIGIN_SQL = "SELECT recorded_at FROM fact_origin ORDER BY rowid LIMIT 1"


def legacy_cutoff(query: Query) -> str | None:
    """When this store started recording origins (ISO UTC), or ``None`` if it cannot be read.

    ``None`` grandfathers nothing: an unreadable log never makes a fact trusted (fails closed).
    """
    try:
        rows = query(_CUTOFF_SQL, (_ORIGIN_MIGRATION,))
    except Exception:  # noqa: BLE001 - no migration_log: nothing is grandfathered
        return None
    value = str(rows[0][0]).replace(" ", "T", 1) if rows and rows[0][0] else None
    if value is None:
        return None
    try:  # the log's time moves on when a migration is repaired; the first origin row does not
        first = query(_FIRST_ORIGIN_SQL, ())
    except Exception:  # noqa: BLE001
        first = []
    if first and first[0][0]:
        value = min(value, str(first[0][0]).replace(" ", "T", 1))
    return value


def is_legacy(query: Query, created_at: object) -> bool:
    """True for a fact created before this store started recording origins."""
    cutoff = legacy_cutoff(query)
    if not (cutoff and created_at):
        return False
    return str(created_at).replace(" ", "T", 1) < cutoff.replace(" ", "T", 1)


def _node(query: Query, fact_id: str) -> tuple[Trust, list[str]] | None:
    """The fact's own class (taint not yet joined) and its parent fact ids; ``None`` if erased."""
    rows = query(_ORIGIN, (fact_id,))
    if not rows:
        return None
    (memory_id, channel, transform, source_type, agent_id, speaker, role, p_mem, p_fact,
     id_source, created_at) = rows[0]
    if channel is None and is_legacy(query, created_at):  # saved before this store took 4.2.30
        own = oc.Classified(oc.LEGACY, f"{oc.VERSION}:before_4230")
    elif channel is None:  # no origin row after the upgrade: nothing vouches for it
        own = oc.Classified(oc.UNKNOWN, f"{oc.VERSION}:no_origin_row")
    else:
        own = oc.classify(oc.Signals(channel=channel, transform=transform or "unknown",
                                     source_type=source_type, agent_id=agent_id,
                                     speaker=speaker, role=role, agent_id_source=id_source))
    parents: list[str] = [p_fact] if p_fact else []
    if p_mem and (channel == "derived" or p_mem != memory_id):
        parents += [r[0] for r in query(_MEMORY_SOURCES, (p_mem, fact_id, MAX_PARENT_FACTS))]
    return Trust(own.origin_class, own.origin_class, own.basis), parents


def compute(query: Query, fact_id: str) -> Trust | None:
    """Class, taint and basis of ``fact_id`` from its signals; ``None`` if it does not exist."""
    start = _node(query, fact_id)
    if start is None:
        return None
    own, frontier = start
    taint, seen = own.origin_class, {fact_id}
    for _depth in range(MAX_DEPTH):
        nxt: list[str] = []
        for parent in frontier:
            if parent in seen or len(seen) >= MAX_FACTS:
                continue
            seen.add(parent)
            node = _node(query, parent)
            if node is None:  # erased: nothing left to inherit from
                continue
            taint = oc.join(taint, node[0].origin_class)
            nxt += node[1]
        if not nxt:
            break
        frontier = nxt
    return Trust(own.origin_class, taint, own.basis)


def on_conn(conn: sqlite3.Connection) -> Query:
    return lambda sql, params: conn.execute(sql, params).fetchall()


# -- persistence ----------------------------------------------------------------------------------

_UPSERT = ("INSERT INTO fact_trust (fact_id, profile_id, origin_class, taint_class, basis, "
           "computed_at) VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT(fact_id) DO UPDATE SET "
           "origin_class = excluded.origin_class, taint_class = excluded.taint_class, "
           "basis = excluded.basis, computed_at = excluded.computed_at "
           f"WHERE fact_trust.basis NOT LIKE '{CONFIRMED}:%'")


def _now() -> str:
    return datetime.now(UTC).isoformat()


def store(run: Callable[[str, tuple], Any], fact_id: str, profile_id: str, trust: Trust) -> None:
    """Write (or refresh) one fact's row. A person's confirmation is never overwritten."""
    run(_UPSERT, (fact_id, profile_id, trust.origin_class, trust.taint_class, trust.basis, _now()))


def mark_confirmed(conn: sqlite3.Connection, fact_id: str, *, by: str, channel: str) -> Trust:
    """A person approved this memory: class and taint ``user_confirmed``.

    ``basis`` names who and through which door (``confirmed:<channel>:<by>``) and
    the batch never recomputes it. Raises ``LookupError`` for a fact that does not exist.
    """
    row = conn.execute("SELECT profile_id FROM atomic_facts WHERE fact_id = ?",
                       (fact_id,)).fetchone()
    if row is None:
        raise LookupError(f"no fact {fact_id}")
    trust = Trust(oc.USER_CONFIRMED, oc.USER_CONFIRMED, f"{CONFIRMED}:{channel}:{by[:80]}")
    conn.execute(
        "INSERT INTO fact_trust (fact_id, profile_id, origin_class, taint_class, basis, "
        "computed_at) VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT(fact_id) DO UPDATE SET "
        "origin_class = excluded.origin_class, taint_class = excluded.taint_class, "
        "basis = excluded.basis, computed_at = excluded.computed_at",
        (fact_id, row[0], trust.origin_class, trust.taint_class, trust.basis, _now()))
    return trust


def admit(db: Any, fact: Any) -> None:
    """At admission: class and taint for the fact just written. Never raises.

    Called by ``storage/fact_origin.record`` beside the origin row, in the same
    transaction. Only while ``trust.taint`` is on or in shadow; otherwise nothing
    is done and a save costs what it did before.
    """
    try:
        if not _observing():
            return
        if not getattr(db, "_fact_trust_ready", False):
            if not db.execute("SELECT 1 FROM sqlite_master WHERE type='table' "
                              "AND name='fact_trust'", ()):
                return  # the deferred M079 has not run yet
            db._fact_trust_ready = True
        trust = compute(lambda sql, params: db.execute(sql, params), fact.fact_id)
        if trust is not None:
            store(lambda sql, params: db.execute(sql, params), fact.fact_id, fact.profile_id, trust)
    except Exception as exc:  # noqa: BLE001 - class is derived evidence, never a lost save
        logger.warning("fact trust not recorded for %s (%s)", getattr(fact, "fact_id", "?"),
                       type(exc).__name__)


def _observing() -> bool:
    from superlocalmemory.core.feature_flags import FeatureFlags
    from superlocalmemory.infra.data_root import canonical_data_root

    return FeatureFlags(canonical_data_root()).observing("trust.taint")


# -- the background batch -------------------------------------------------------------------------

_MISSING = ("SELECT f.rowid, f.fact_id, f.profile_id FROM atomic_facts f "
            "LEFT JOIN fact_trust t ON t.fact_id = f.fact_id "
            f"WHERE t.fact_id IS NULL OR (t.basis NOT LIKE '{oc.VERSION}:%' "
            f"AND t.basis NOT LIKE '{CONFIRMED}:%') ORDER BY f.rowid LIMIT ?")


def run_batch(db: Any, *, limit: int = BATCH) -> int:
    """Compute up to ``limit`` facts that have no current row. Returns how many were written.

    Resumable by construction: a fact with a row of the current mapping version
    is never selected again, so a stop loses nothing. Only while ``trust.taint``
    is on or in shadow (the caller checks).
    """
    written = 0
    with db.raw_connection() as conn:
        try:
            rows = conn.execute(_MISSING, (int(limit),)).fetchall()
        except sqlite3.OperationalError:  # a store before M079
            return 0
        query = on_conn(conn)
        for _rid, fact_id, profile_id in rows:
            trust = compute(query, fact_id)
            if trust is None:
                continue
            store(conn.execute, fact_id, profile_id, trust)
            written += 1
    return written


# -- reads (never write) --------------------------------------------------------------------------

def origin_of(conn: sqlite3.Connection, fact_id: str) -> dict[str, str] | None:
    """Where one fact came from: class, taint, their words and the mapping row (``slm why``).

    The stored row when there is one, else computed from the raw signals right
    now (nothing is written). ``None`` for a fact that does not exist.
    """
    try:
        row = conn.execute("SELECT origin_class, taint_class, basis FROM fact_trust "
                           "WHERE fact_id = ?", (fact_id,)).fetchone()
    except sqlite3.OperationalError:
        row = None
    if row is not None:
        return Trust(*row).as_dict()
    trust = _safe_compute(conn, fact_id)
    return trust.as_dict() if trust is not None else None


def _safe_compute(conn: sqlite3.Connection, fact_id: str) -> Trust | None:
    """``compute`` for a read: a store too old to hold the tables has nothing to say."""
    try:
        return compute(on_conn(conn), fact_id)
    except sqlite3.OperationalError:
        return None


#: Facts a single recall may compute live for (the rest simply carry no origin yet).
LIVE_LIMIT = 10


def for_results(conn: sqlite3.Connection, fact_ids: Iterable[str]) -> dict[str, dict[str, str]]:
    """``{fact_id: origin}`` for the facts of one recall: one batched select, then a
    bounded live computation for those with no row yet. Read-only."""
    ids = [f for f in dict.fromkeys(fact_ids) if f]
    if not ids:
        return {}
    marks = ",".join("?" * len(ids))
    try:
        rows = conn.execute("SELECT fact_id, origin_class, taint_class, basis FROM fact_trust "
                            f"WHERE fact_id IN ({marks})", ids).fetchall()
    except sqlite3.OperationalError:
        rows = []
    out = {r[0]: Trust(r[1], r[2], r[3]).as_dict() for r in rows}
    for fact_id in [i for i in ids if i not in out][:LIVE_LIMIT]:
        trust = _safe_compute(conn, fact_id)
        if trust is not None:
            out[fact_id] = trust.as_dict()
    return out


# -- the maintenance step -------------------------------------------------------------------------

def maintenance_step(db: Any, *, max_batches: int = 5) -> int:
    """One maintenance cycle's share: bounded batches, only while the flag observes."""
    if not _observing():
        return 0
    total = 0
    for _ in range(max_batches):
        done = run_batch(db)
        total += done
        if done < BATCH:
            break
    return total


__all__ = ["BATCH", "CONFIRMED", "LIVE_LIMIT", "MAX_DEPTH", "Trust", "admit", "compute",
           "for_results", "maintenance_step", "mark_confirmed", "on_conn", "origin_of",
           "run_batch", "store"]
