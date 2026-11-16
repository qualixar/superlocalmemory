# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Flag the reworded session ends the stub rule learned in 4.2.30 (R-31).

WHAT WAS WRONG

The extraction model rewords a hook's "session ended" line in many ways ("The
conversation ended at 22:48", "The session ended on October 8, 2026, at 18:47",
"2026-07-19 is the end date of <name>'s session"). The stub rule knew some of
them (M068); the rest were stored as answers and could be shown at a session
start as "relevant memories". Measured on one real store: 353 such rows.

WHAT IT DOES

Runs M062's pass again with the current rule, which now also recognises a
session end whose every word after "ended" is a date, a time or a weekday. A
rewording that says anything else ("ended with files X, Y", "ended in
AgentAssert", "ended after the e2e test", "ended last week") is real content
and stays an answer: 52 such rows on the same store.

WHY IT IS SAFE

It is M068's code, so the same guarantees: it only sets ``non_answer`` from 0
to 1, only where the whole text is a session stamp, never on a row the user
decided about (``non_answer_user`` set), and never edits, deletes or hides a
row from lists and counts. Running it again changes nothing.
"""

from __future__ import annotations

import sqlite3

from superlocalmemory.storage.migrations.M062_hook_bookkeeping_is_flagged import flag_stubs

NAME = "M076_session_end_stamps_are_flagged"
DB_TARGET = "memory"
#: Sets an existing flag column only; an older build ignores it.
DOWNGRADE_FLOOR = 51

DDL = """
-- No schema change. apply() sets atomic_facts.non_answer = 1 on every row the
-- current stub rule recognises and the user has not decided about.
"""


def apply(conn: sqlite3.Connection) -> None:
    """Flag every unflagged, undecided stub the current rule recognises."""
    flag_stubs(conn, "M076")


REPAIR_NOT_APPLICABLE = (
    "a one-time data repair whose rule is a pure function of each row's text; "
    "new stubs are flagged by the save that writes them, so there is no end "
    "state to drift from, and replaying it from the runner would UPDATE user "
    "rows on every boot for no reason"
)
