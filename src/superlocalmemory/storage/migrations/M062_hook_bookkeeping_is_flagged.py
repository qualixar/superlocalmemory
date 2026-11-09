# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Flag the hooks' own bookkeeping that is already stored as if it were an answer.

WHAT WAS WRONG

``atomic_facts.non_answer`` keeps a stored stub out of recall's answers, and the
stub rule (``encoding/non_answer.is_stub``) only knew the bare
``session ended <date> <time>``. The Stop hook has written
``[<project>] session ended <date> <time> | files: ...`` since 4.1.21, the
Write/Edit hook writes ``File changed: <name>``, and the flag itself was set by
a background pass that had not reached most rows. Measured on one real store:
30 of 291 bare "session ended" rows carried the flag, and of 5,443 facts
written in 7 days about 1,056 were this bookkeeping. "What am I working on?"
reads the newest rows, so it answered from them.

New saves are now flagged by the save itself. This flags the rows already
there, by the same rule, so the store does not wait for each one to be saved
again.

WHY IT IS SAFE

It only ever sets ``non_answer`` from 0 to 1, only where the rule says the
WHOLE text is machine bookkeeping, and never on a row where ``non_answer_user``
is set: the user's own "this is an answer" (or "this is not") decision wins
over any automatic rule, here as everywhere else. The row is not edited,
deleted or hidden from lists and counts; recall's answer paths skip it, and
``slm`` lists show it flagged. Running it again changes nothing, because a
flagged row is not a candidate.

Rows longer than ``_MAX_STUB_CHARS`` are not read: no stub shape reaches that
length, and skipping them keeps a store of long documents from being scanned.
"""

from __future__ import annotations

import logging
import sqlite3

logger = logging.getLogger(__name__)

NAME = "M062_hook_bookkeeping_is_flagged"
DB_TARGET = "memory"
#: Additive in effect (one existing flag column set where it was 0); an older
#: build ignores or never reads it, so going back to 4.1.18 stays possible.
DOWNGRADE_FLOOR = 51

_TABLE = "atomic_facts"
#: Rows read per transaction. A stub is a few hundred characters at most.
_BATCH = 2000
_MAX_STUB_CHARS = 1000

# Kept so the catalogue and the DDL-drift hash have something to read. The
# runner prefers apply() when a module defines one, and apply() is what runs.
DDL = """
-- No schema change. apply() sets atomic_facts.non_answer = 1 on every row the
-- stub rule recognises and the user has not decided about.
"""


def _columns(conn: sqlite3.Connection) -> set[str]:
    try:
        return {row[1] for row in conn.execute(f"PRAGMA table_info({_TABLE})")}
    except sqlite3.Error:
        return set()


def apply(conn: sqlite3.Connection) -> None:
    """Flag every unflagged, undecided stub, in resumable batches."""
    flag_stubs(conn, "M062")


def flag_stubs(conn: sqlite3.Connection, label: str) -> int:
    """Flag every unflagged, undecided row the CURRENT stub rule recognises.

    Shared with later migrations that run it again after the rule learns a new
    shape (M068), so each one flags what was stored before it. Returns the
    number of rows flagged.
    """
    from superlocalmemory.encoding.non_answer import is_stub

    columns = _columns(conn)
    if not {"content", "non_answer"} <= columns:
        logger.debug("%s: no atomic_facts.non_answer column; nothing to flag", label)
        return 0
    undecided = " AND non_answer_user IS NULL" if "non_answer_user" in columns else ""
    cursor, read, flagged = 0, 0, 0
    while True:
        batch = conn.execute(
            f"SELECT rowid, content FROM {_TABLE} WHERE rowid > ? "
            f"AND COALESCE(non_answer, 0) = 0{undecided} "
            f"AND length(content) <= {_MAX_STUB_CHARS} ORDER BY rowid LIMIT {_BATCH}",
            (cursor,),
        ).fetchall()
        if not batch:
            break
        cursor = batch[-1][0]
        read += len(batch)
        stubs = [(row[0],) for row in batch if is_stub(row[1])]
        flagged += _flag(conn, stubs, undecided)
    logger.info("%s: read %d unflagged short facts; flagged %d as stubs", label, read, flagged)
    return flagged


def _flag(conn: sqlite3.Connection, rows: list[tuple[int]], undecided: str) -> int:
    if not rows:
        return 0
    own_transaction = not conn.in_transaction
    if own_transaction:
        conn.execute("BEGIN IMMEDIATE")
    try:
        # Guarded again on the write, so a decision made since the read stands.
        changed = conn.executemany(
            f"UPDATE {_TABLE} SET non_answer = 1 "
            f"WHERE rowid = ? AND COALESCE(non_answer, 0) = 0{undecided}", rows,
        ).rowcount
        if own_transaction:
            conn.execute("COMMIT")
    except Exception:
        if own_transaction:
            conn.execute("ROLLBACK")
        raise
    return max(changed, 0)


REPAIR_NOT_APPLICABLE = (
    "a one-time data repair whose rule is a pure function of each row's text; "
    "new stubs are flagged by the save that writes them, so there is no end "
    "state to drift from, and replaying it from the runner would UPDATE user "
    "rows on every boot for no reason"
)
