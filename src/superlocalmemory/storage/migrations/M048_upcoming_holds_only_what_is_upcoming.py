# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Re-read what is filed as a plan, and file the rest correctly.

M046 renamed the type used for planned events. Renaming is all it did: every
row that said ``temporal`` now says ``prospective``, and the contents were never
re-read. On a real store 869 rows carried that type and seven of them contained
any planning language — session summaries, records of finished work, lists of
commits. After the rename that same set carries a more confident name, and the
question a user actually asks — "what is coming up" — reads exactly that set.

So the rename needs a second half. This one re-reads each of those memories
under the rule that decides the question today and moves the ones that are not
plans to the type their wording supports.

WHY THIS ONLY DEMOTES

It never promotes. A memory filed as ordinary that turns out to read like a plan
is left alone, because the cost is asymmetric in the same way the classifier's
own tiers are: a plan filed as an ordinary memory is still found by every
retrieval channel, and an ordinary memory filed as a plan is the pollution this
exists to remove. Walking every fact in the store to look for promotions would
also cost a full table scan for the smaller half of the benefit.

WHY IT IS SAFE TO RE-RUN

The rule is a pure function of the text. Running it twice gives the same answer,
so a second pass moves nothing.
"""

from __future__ import annotations

import logging
import contextlib
import sqlite3
from typing import Any

logger = logging.getLogger(__name__)

NAME = "M048_upcoming_holds_only_what_is_upcoming"
DB_TARGET = "memory"

#: No schema change and no new value — every type written here already existed.
BREAKING_VERSION = 0

_TABLE = "atomic_facts"
_PROSPECTIVE = "prospective"

#: Rows per transaction.
_BATCH = 500

DDL = """
-- No schema change. apply() re-reads the content of every fact typed
-- 'prospective' and demotes the ones whose wording does not describe something
-- still ahead.
"""


def _resolve(content: str) -> str:
    """The type this text supports — and the question asked in the right order.

    The full classifier answers "which of the four is this", and it asks about
    opinion before it asks about plans. That is right for a new memory: "I think
    we should ship next week" is an opinion. It is WRONG here, because this
    migration is only deciding whether a row belongs under the plan type, and
    "we should deploy next Tuesday" IS something coming up. Asking the full
    classifier demoted real deadlines into opinions and they vanished from the
    list of what is upcoming — which is the exact harm this exists to repair.

    So: if it reads as a plan, it stays a plan. Only when it does not is the
    full classifier asked where it should go instead, and its answer is taken
    only if it is not "prospective".
    """
    from superlocalmemory.encoding.fact_extractor import _classify_sentence
    from superlocalmemory.encoding.prospective_markers import looks_prospective

    text = content or ""
    if looks_prospective(text):
        return _PROSPECTIVE
    resolved = _classify_sentence(text).value
    return "semantic" if resolved == _PROSPECTIVE else resolved


@contextlib.contextmanager
def _held(conn: sqlite3.Connection):
    """Yield the connection the caller already owns, and leave it open."""
    yield conn


def apply(
    conn: sqlite3.Connection | None = None,
    *,
    open_connection: Any = None,
) -> None:
    """Demote every wrongly-filed plan, in batches, resumably.

    Pass ``conn`` when the caller owns the connection for the whole pass -- the
    migration runner at startup, where nothing else is writing. Pass
    ``open_connection`` (a context manager factory, typically the database
    manager's ``raw_connection``) on a running store: each batch then takes and
    releases the process write lock, so a memory being saved waits one batch
    rather than the whole pass. The batching below cannot do that on its own,
    because holding the connection is what holds the lock.
    """
    if (conn is None) == (open_connection is None):
        raise ValueError("pass exactly one of conn or open_connection")
    acquire = (lambda: _held(conn)) if conn is not None else open_connection

    with acquire() as probe:
        existing = {r[1] for r in probe.execute(f"PRAGMA table_info({_TABLE})")}
    if "fact_type" not in existing or "content" not in existing:
        logger.info("M048: %s has no fact_type/content; nothing to re-read", _TABLE)
        return

    cursor = 0
    demoted = 0
    kept = 0
    while True:
        with acquire() as active:
            batch = active.execute(
                f"SELECT rowid, fact_id, content FROM {_TABLE} "
                f"WHERE rowid > ? AND fact_type = ? ORDER BY rowid LIMIT {_BATCH}",
                (cursor, _PROSPECTIVE),
            ).fetchall()
            if not batch:
                break
            cursor = batch[-1][0]

            moves: list[tuple[str, int]] = []
            for rowid, _fact_id, content in batch:
                resolved = _resolve(content)
                if resolved == _PROSPECTIVE:
                    kept += 1
                    continue
                moves.append((resolved, rowid))

            if moves:
                active.execute("BEGIN IMMEDIATE")
                try:
                    # Conditional on the row still being what was read, so a
                    # concurrent write is not clobbered.
                    changed = 0
                    for new_type, rowid in moves:
                        cur = active.execute(
                            f"UPDATE {_TABLE} SET fact_type = ? "
                            f"WHERE rowid = ? AND fact_type = '{_PROSPECTIVE}'",
                            (new_type, rowid),
                        )
                        # Count what the guard let through, not what was
                        # offered. A concurrent write can change the row
                        # underneath, and a log line that reports the intention
                        # as the outcome is how a receipt comes to overstate
                        # what happened.
                        changed += cur.rowcount if cur.rowcount and cur.rowcount > 0 else 0
                    active.commit()
                except Exception:
                    active.rollback()
                    raise
                demoted += changed

    logger.info(
        "M048: re-read %d memories filed as plans; %d were, %d moved",
        demoted + kept, kept, demoted,
    )


def verify(conn: sqlite3.Connection) -> bool:
    """Is the end state in place: would running this pass again change nothing?

    That is the question the runner asks a verify, and it is answerable exactly
    — this pass is a pure function of the text, so its end state is its own
    fixed point. No estimate, no threshold.

    An earlier version tried to tell "the pass never ran" from "the rule got
    sharper" by how much disagreed, and a share cannot carry that: a store with
    nine memories, six of them real plans, sits at 33% disagreement having never
    been touched at all. It reported success.

    Drift after a rule change makes this answer False, correctly — the end state
    under today's rule is genuinely not in place. It is not left to the runner
    to repair, because a completed migration is never replayed: the maintenance
    cycle re-runs the same pass, so the store converges on its own. See
    ``core/maintenance_scheduler``.
    """
    existing = {r[1] for r in conn.execute(f"PRAGMA table_info({_TABLE})")}
    if "fact_type" not in existing or "content" not in existing:
        return True

    rows = conn.execute(
        f"SELECT fact_id, content FROM {_TABLE} WHERE fact_type = ?",
        (_PROSPECTIVE,),
    ).fetchall()
    if not rows:
        return True

    drifted = [
        fact_id for fact_id, content in rows if _resolve(content) != _PROSPECTIVE
    ]
    if drifted:
        logger.info(
            "M048: %d of %d memories filed as plans no longer read as one; the "
            "maintenance cycle re-reads them", len(drifted), len(rows),
        )
        return False
    return True


def blocks_serving(conn: sqlite3.Connection) -> bool:
    """Should a daemon refuse to serve while this check does not hold? No.

    Nothing here is about schema. ``verify()`` reads ``content`` and compares
    today's reading of it against the ``fact_type`` already stored, so a False
    means some memories are filed as plans that no longer read as plans. Every
    table and column a query needs is present either way; the store answers
    normally, and at worst a handful of memories carry a stale label until the
    next maintenance pass re-reads them.

    The distinction matters because this is a standing guard over data that
    ordinary use re-violates by design: a plan whose date passes stops reading
    as upcoming, which is the rule working, not a fault. Treating that like a
    missing table let one drifted row answer 503 on every route for as long as
    the process lived — and because the readiness snapshot is taken once at
    startup, the background pass that repairs the data could not lift the
    refusal it caused. An outage produced by a quality check is worse than the
    thing the check is for. Same reasoning as ``M043.blocks_serving``.
    """
    return False


#: No runner repair for this migration, by design (4.1.14 #133).
#: M048's verify is a DATA check that re-fails by routine date rollover,
#: and its own contract says the maintenance cycle (not migration replay)
#: converges the drift. blocks_serving() is False, so drift is listed
#: without refusing writes. Re-running the demotion pass from the runner
#: would UPDATE fact rows on every drifted boot — schema self-heal must
#: never mutate user data.
REPAIR_NOT_APPLICABLE = (
    "data-quality drift converged by the maintenance cycle "
    "(core/maintenance_scheduler); blocks_serving is False so drift "
    "never refuses writes; runner replay would UPDATE user rows"
)
