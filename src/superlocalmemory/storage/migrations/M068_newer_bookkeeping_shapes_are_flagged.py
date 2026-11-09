# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Flag the hook bookkeeping the stub rule learned to recognise after M062.

WHAT WAS WRONG

M062 (4.2.2) flagged the stored stubs the rule knew then, and new saves are
flagged as they are written. The rule has since learned more shapes: the
Codex/agent checkpoint without its "[agent]" prefix (4.2.14), "the session
ended <date> <time>" (4.2.15), and the Stop hook summary that adds only the
branch and the ``git diff --stat`` total, plus the copies the extractor
paraphrases from these lines ("The session ended on <stamp>", "Varun's session
ended at <stamp>", "<date> is the date when the session ended", "2 files
changed, 0 insertions(+)"). Rows of those shapes stored before each change were
never re-checked, and recall leaves a row out of its answers by the flag, not
the text. Measured on one real store: 2,030 such rows, every one a bare
timestamp or git total; some were shown at a session start as "relevant
memories".

WHAT IT DOES

Runs M062's pass again with the current rule, so every row stored before this
release is checked against every shape the rule now knows. A summary that
lists commits ("| recent: ...") or the agent's last action is real content and
stays an answer.

WHY IT IS SAFE

The same guarantees as M062, because it is M062's code: it only sets
``non_answer`` from 0 to 1, only where the WHOLE text is machine bookkeeping,
never on a row the user decided about (``non_answer_user`` set), and never
edits, deletes or hides a row from lists and counts. Running it again changes
nothing.
"""

from __future__ import annotations

import sqlite3

from superlocalmemory.storage.migrations.M062_hook_bookkeeping_is_flagged import flag_stubs

NAME = "M068_newer_bookkeeping_shapes_are_flagged"
DB_TARGET = "memory"
#: Sets an existing flag column only; an older build ignores it, so going back
#: to 4.1.18 stays possible.
DOWNGRADE_FLOOR = 51

DDL = """
-- No schema change. apply() sets atomic_facts.non_answer = 1 on every row the
-- current stub rule recognises and the user has not decided about.
"""


def apply(conn: sqlite3.Connection) -> None:
    """Flag every unflagged, undecided stub the current rule recognises."""
    flag_stubs(conn, "M068")


REPAIR_NOT_APPLICABLE = (
    "a one-time data repair whose rule is a pure function of each row's text; "
    "new stubs are flagged by the save that writes them, so there is no end "
    "state to drift from, and replaying it from the runner would UPDATE user "
    "rows on every boot for no reason"
)
