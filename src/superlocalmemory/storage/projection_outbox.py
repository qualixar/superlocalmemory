# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""The queue that makes a cross-engine projection write impossible to lose.

The graph lives in CozoDB (RocksDB underneath) and the vectors in LanceDB. Both
are separate storage engines from SQLite, so no transaction can span them and
there is no two-phase commit to reach for. The obvious sequence — commit the
fact, then write the projection — has a window in the middle. A process killed
inside that window leaves a memory that exists in SQLite, is absent from the
graph, and is therefore unrecallable, with nothing anywhere recording that it
happened.

This module is the transactional-outbox side of the answer. A row naming the
fact is written **inside the same SQLite transaction as the fact itself**, so
the intent to project is exactly as durable as the fact. A worker later claims
the row, applies it to Cozo and Lance, and only then deletes it. Killed at any
point, the row is still there and the work is retried.

WHAT THE ROW CARRIES, AND WHY IT IS SO THIN
-------------------------------------------
Only a fact id, a profile, and an intent. Not the entities, not the edges, not
the embedding. The drain re-reads the fact's current state from SQLite at the
moment it projects.

That choice buys idempotency for free. Replaying a row can only ever write the
present truth, so a retry after a crash, a double delivery, and an intent that
was superseded three times while queued all converge on the same correct
projection. Serialising a snapshot into the row instead would mean replaying
stale state over fresh state — the classic outbox bug.

WHY THE PRIMARY KEY IS THE FACT ID
----------------------------------
It coalesces. Ten writes to one fact are one unit of work, because the drain
re-derives everything anyway. It also puts a hard ceiling on the table:
**at most one row per fact, so the queue can never exceed the store.** That is
what makes it safe to queue unconditionally on a machine where no projection
has been built yet — the backlog is bounded by the fact count and is precisely
the set of facts a future migration needs to project.

LATEST INTENT WINS
------------------
``op`` is replaced on conflict rather than appended, so a delete queued after an
upsert leaves one row saying ``delete``. A queue that kept both would depend on
draining them in order to avoid resurrecting a forgotten memory; this one cannot
get that wrong because there is only ever one intent to read.

``revision`` IS A CONCURRENCY TOKEN, NOT A COUNTER
--------------------------------------------------
The drain claims a row, spends time in Cozo, then clears it. A write landing in
that gap must not be discarded. So every enqueue bumps ``revision``, and the
clear is conditional on the revision the drain claimed. If it moved, the delete
matches nothing and the new intent stays queued. A timestamp would not do:
two enqueues inside one clock tick would be indistinguishable.

FAILURE POLICY
--------------
``enqueue`` is allowed to raise, and the caller's transaction then rolls back
the fact along with it. That is deliberate: this is a durability mechanism, and
a durability mechanism that silently degrades to best-effort is the defect it
exists to remove. The one expected absence — a store old enough not to have the
table — is answered by :func:`is_available` rather than by an exception, so an
un-migrated store keeps working and simply queues nothing.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from typing import Any

#: The table. Deliberately un-indexed beyond its primary key: the drain reads
#: it in (attempts, revision) order, and no secondary index serves that, while
#: every index costs another B-tree write on a path that runs on every store.
#: The table is bounded by the fact count and its steady-state depth is zero, so
#: a scan of it is a scan of nothing.
#:
#: Applied by ``schema.create_all_tables``, which the engine runs on every init
#: — so an upgraded store gains the queue on first open with no migration and no
#: manual step. There is deliberately NO migration for it: a migration can be
#: skipped, deferred or fail, and if this table is absent then facts silently
#: stop being projected. The invariant must not be contingent on a pass that can
#: not run.
DDL = """
CREATE TABLE IF NOT EXISTS projection_outbox (
    fact_id     TEXT PRIMARY KEY,
    profile_id  TEXT NOT NULL,
    op          TEXT NOT NULL DEFAULT 'upsert',
    revision    INTEGER NOT NULL DEFAULT 1,
    enqueued_at TEXT NOT NULL,
    attempts    INTEGER NOT NULL DEFAULT 0,
    last_error  TEXT
);
"""

TABLE = "projection_outbox"

#: The depth query, as a constant, because two surfaces need it against two
#: different kinds of connection and a second copy is how two surfaces come to
#: disagree about one number.
DEPTH_SQL = "SELECT COUNT(*) FROM projection_outbox"

OP_UPSERT = "upsert"
OP_DELETE = "delete"

#: The fact columns a projection is derived from. An update touching none of
#: these changes nothing Cozo or Lance holds, so it must not queue work.
#:
#: ``access_count`` is the one that matters here: recall bumps it on every hit,
#: so queueing on any update at all would put one row per returned memory per
#: recall onto the drain worker — a self-inflicted denial of service for a
#: column neither projection stores.
PROJECTED_FACT_COLUMNS = frozenset({
    "embedding",
    "lifecycle",
    "canonical_entities_json",
    "canonical_entities",
    "profile_id",
    "scope",
})

_AVAILABILITY_ATTR = "_projection_outbox_available"

#: Queues a fact whenever an UPDATE changes what its projection should hold.
#:
#: ``DatabaseManager.update_fact`` queues on ``PROJECTED_FACT_COLUMNS``, but most
#: visibility changes are plain ``UPDATE atomic_facts`` statements elsewhere
#: (archive, merge, quarantine, lifecycle), and none of them queued. An archived
#: fact kept its edges, bridge and vector in Cozo and LanceDB until something
#: else happened to re-project it, and the copy drifted from the store (#296).
#: A trigger covers every such path, present and future, in the writer's own
#: transaction.
CHANGE_TRIGGER = "trg_projection_fact_changed"

#: Visibility (``archive_status``, ``quarantined``, ``merged_into``) and what the
#: projection is built from. Not ``embedding``: an embedding switch renames the
#: vector columns and SQLite rewrites a trigger that names a renamed column (see
#: storage/hub_baseline_trigger.py); vector changes are queued where they happen.
CHANGE_TRIGGER_COLUMNS = (
    "archive_status", "quarantined", "merged_into", "lifecycle",
    "canonical_entities_json", "profile_id", "scope",
)


def change_trigger_ddl(columns: list[str]) -> str:
    """The trigger watching ``columns`` (all present on ``atomic_facts``)."""
    changed = " OR ".join(f"OLD.{c} IS NOT NEW.{c}" for c in columns)
    return (
        f"CREATE TRIGGER {CHANGE_TRIGGER} AFTER UPDATE OF {', '.join(columns)} "
        f"ON atomic_facts WHEN {changed} BEGIN "
        "INSERT INTO projection_outbox "
        "(fact_id, profile_id, op, revision, enqueued_at, attempts, last_error) "
        "VALUES (NEW.fact_id, COALESCE(NEW.profile_id, 'default'), 'upsert', 1, "
        "strftime('%Y-%m-%dT%H:%M:%fZ', 'now'), 0, NULL) "
        "ON CONFLICT(fact_id) DO UPDATE SET op = 'upsert', "
        "profile_id = excluded.profile_id, revision = projection_outbox.revision + 1, "
        "enqueued_at = excluded.enqueued_at, attempts = 0, last_error = NULL; END"
    )


def install_change_trigger(conn: sqlite3.Connection) -> bool:
    """(Re-)create the change trigger over the watched columns this store has.

    False when there is nothing to install on (no queue, or no fact table).
    Re-created every time so a column a later migration adds is watched too.
    The caller owns the transaction.
    """
    names = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' "
        "AND name IN ('atomic_facts', 'projection_outbox')")}
    if names != {"atomic_facts", "projection_outbox"}:
        return False
    present = {r[1] for r in conn.execute("PRAGMA table_info(atomic_facts)")}
    columns = [c for c in CHANGE_TRIGGER_COLUMNS if c in present]
    if not columns:
        return False
    conn.execute(f"DROP TRIGGER IF EXISTS {CHANGE_TRIGGER}")
    conn.execute(change_trigger_ddl(columns))
    return True


def change_trigger_installed(conn: sqlite3.Connection) -> bool:
    """True when the store has the change trigger."""
    return conn.execute("SELECT 1 FROM sqlite_master WHERE type = 'trigger' AND name = ?",
                        (CHANGE_TRIGGER,)).fetchone() is not None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Availability
# ---------------------------------------------------------------------------

def is_available(db: Any) -> bool:
    """Whether this store has an outbox to queue into.

    Cached on the manager after the first look. The table is created at engine
    init before any fact can be stored, so it cannot appear or vanish part-way
    through a process in a way that matters; paying a ``sqlite_master`` lookup
    on every single write to prove that again would be the more expensive
    mistake.
    """
    cached = getattr(db, _AVAILABILITY_ATTR, None)
    if cached is not None:
        return bool(cached)
    try:
        rows = db.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (TABLE,),
        )
        present = bool(rows)
    except sqlite3.Error:
        present = False
    if not present:
        # The change trigger writes this table; without it every UPDATE of a fact
        # would fail. Nothing drops the table, but a write must never be the price.
        try:
            db.execute(f"DROP TRIGGER IF EXISTS {CHANGE_TRIGGER}")
        except sqlite3.Error:
            pass
    setattr(db, _AVAILABILITY_ATTR, present)
    return present


def forget_availability(db: Any) -> None:
    """Drop the cached answer, so the next call looks again.

    For the migration that creates the table on a manager already in use, and
    for tests that remove it to exercise the un-migrated path.
    """
    if hasattr(db, _AVAILABILITY_ATTR):
        delattr(db, _AVAILABILITY_ATTR)
    # Look now, not at the next enqueue: a fact write is prepared together with
    # the change trigger, so a trigger left without its table fails that write
    # before any enqueue could notice.
    is_available(db)


# ---------------------------------------------------------------------------
# Enqueue — runs inside the caller's transaction
# ---------------------------------------------------------------------------

def enqueue(db: Any, fact_id: str, profile_id: str, op: str = OP_UPSERT) -> None:
    """Queue one fact to be projected, in the caller's transaction.

    Silent no-op when there is no table (see :func:`is_available`) or when the
    fact id is empty — an empty id names nothing and would give the drain a row
    it can never resolve.
    """
    if not fact_id or not is_available(db):
        return
    db.execute(
        """INSERT INTO projection_outbox
               (fact_id, profile_id, op, revision, enqueued_at, attempts, last_error)
           VALUES (?, ?, ?, 1, ?, 0, NULL)
           ON CONFLICT(fact_id) DO UPDATE SET
               op          = excluded.op,
               profile_id  = excluded.profile_id,
               revision    = projection_outbox.revision + 1,
               enqueued_at = excluded.enqueued_at,
               attempts    = 0,
               last_error  = NULL""",
        (fact_id, profile_id or "default", op, _now()),
    )


def enqueue_many(
    db: Any, fact_ids: object, profile_id: str, op: str = OP_UPSERT,
) -> None:
    """Queue several facts. Duplicates in the input collapse to one row each."""
    if not is_available(db):
        return
    for fact_id in dict.fromkeys(fact_ids):
        enqueue(db, fact_id, profile_id, op)


def resolve_profile(db: Any, fact_id: str) -> str | None:
    """Which tenant a fact belongs to, for a caller that was not told.

    The drain never reads ``profile_id`` off the row — it re-reads the fact. The
    column exists for one reason: erasure finds a tenant's tables by looking for
    a ``profile_id`` column, so a row carrying the wrong tenant is a fact id
    that survives that tenant's erasure. Anyone tempted to drop the column as
    unused should read that sentence twice.

    Falls back to an existing row's tenant so a re-queue by a caller with less
    information cannot overwrite a correct answer with a guess.
    """
    for sql in (
        "SELECT profile_id FROM atomic_facts WHERE fact_id = ?",
        "SELECT profile_id FROM projection_outbox WHERE fact_id = ?",
    ):
        try:
            rows = db.execute(sql, (fact_id,))
        except sqlite3.Error:
            continue
        if rows:
            found = dict(rows[0]).get("profile_id")
            if found:
                return str(found)
    return None


def enqueue_for_fact(db: Any, fact_id: str, op: str = OP_UPSERT) -> None:
    """Queue a fact whose tenant the caller does not know."""
    if not fact_id or not is_available(db):
        return
    enqueue(db, fact_id, resolve_profile(db, fact_id) or "default", op)


# ---------------------------------------------------------------------------
# Drain
# ---------------------------------------------------------------------------

def claim_batch(db: Any, limit: int = 200) -> list[dict[str, Any]]:
    """The next rows to project, oldest intent first.

    Claiming does not mark or lock anything. The row stays visible and stays
    queued until :func:`resolve` clears it, so a worker that dies mid-batch
    costs a repeat of work that is idempotent by construction — which is
    cheaper and far easier to reason about than a lease that can expire while
    its holder is still alive.
    """
    if not is_available(db):
        return []
    rows = db.execute(
        "SELECT fact_id, profile_id, op, revision, attempts, enqueued_at "
        "FROM projection_outbox ORDER BY attempts, revision LIMIT ?",
        (int(limit),),
    )
    return [dict(r) for r in rows]


def resolve(db: Any, fact_id: str, revision: int) -> bool:
    """Clear a row the projections have accepted. Returns whether it cleared.

    ``False`` means the fact was written again while this projection was in
    flight, so a newer intent is queued and must not be discarded. The caller
    does not need to do anything about it — the next drain picks it up.
    """
    if not is_available(db):
        return False
    db.execute(
        "DELETE FROM projection_outbox WHERE fact_id = ? AND revision = ?",
        (fact_id, int(revision)),
    )
    remaining = db.execute(
        "SELECT 1 FROM projection_outbox WHERE fact_id = ? AND revision = ?",
        (fact_id, int(revision)),
    )
    return not remaining


def record_failure(db: Any, fact_id: str, error: str) -> int:
    """Count a failed attempt and keep the row. Returns the new attempt count.

    The row surviving is the whole point: a projection that cannot be written
    stays visible in :func:`depth` instead of disappearing into a debug log.
    """
    if not is_available(db):
        return 0
    db.execute(
        "UPDATE projection_outbox SET attempts = attempts + 1, last_error = ? "
        "WHERE fact_id = ?",
        (str(error)[:500], fact_id),
    )
    rows = db.execute(
        "SELECT attempts FROM projection_outbox WHERE fact_id = ?", (fact_id,),
    )
    return int(dict(rows[0])["attempts"]) if rows else 0


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

def depth(db: Any) -> int:
    """How many facts are queued and not yet projected.

    Zero is the healthy steady state. A number that does not fall is a
    projection that has stopped keeping up, which is the failure this whole
    mechanism exists to make visible rather than silent.
    """
    if not is_available(db):
        return 0
    try:
        rows = db.execute(f"{DEPTH_SQL} /* depth */")
    except sqlite3.Error:
        return 0
    return int(tuple(rows[0])[0]) if rows else 0


def stalled_count(db: Any, min_attempts: int = 3) -> int:
    """Queued facts that have already failed to project ``min_attempts`` times.

    Depth alone cannot tell a busy queue from a stuck one. This can: a row with
    attempts on it has been tried and refused, so any non-zero answer here is a
    defect with a fact id attached to it.
    """
    if not is_available(db):
        return 0
    try:
        rows = db.execute(
            "SELECT COUNT(*) AS n FROM projection_outbox WHERE attempts >= ?",
            (int(min_attempts),),
        )
    except sqlite3.Error:
        return 0
    return int(dict(rows[0])["n"]) if rows else 0


def health(db: Any) -> dict[str, Any]:
    """The outbox as a status surface reads it."""
    return {
        "available": is_available(db),
        "depth": depth(db),
        "stalled": stalled_count(db),
    }
