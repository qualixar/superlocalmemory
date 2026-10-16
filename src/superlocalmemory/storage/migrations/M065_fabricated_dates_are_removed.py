# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Made-up dates leave the memories they were written into, once, when 4.2.8 first starts.

WHAT WAS WRONG
Before 4.1.6 (24 Aug 2026) the extraction prompt explained relative dates with
the example "session date 2024-01-15". A model that did not know the date
echoed it back, and facts were saved as "The file X was modified on 2024-01-15"
with 2024-01-15 as their date. On one real store, 175 facts carried 2024-01-15
or 2026-01-15 in their words and about 500 date rows carried it as their time.

WHAT THIS DOES
``storage.fabricated_dates_repair``: for facts saved before 4.1.6 whose source
never named the day, the clause with the date goes from the text when the
sentence reads without it, and the date leaves the fact's date columns, its
temporal events and its validity row. No fact is removed. Facts saved after
4.1.6 are never changed. Every change is written to ``data_repair_undo`` in the
same transaction, so ``data_repair_undo.undo(conn, NAME)`` reverses it and
``slm doctor`` reports it.

WHY IT IS SAFE
One transaction: a crash leaves the store as it was and the runner retries at
the next start. Running it again finds nothing left to change.
"""

from __future__ import annotations

import logging
import sqlite3

logger = logging.getLogger(__name__)

NAME = "M065_fabricated_dates_are_removed"
DB_TARGET = "memory"
#: Data plus one new table (the undo record); an older build reads both as before.
DOWNGRADE_FLOOR = 51

DDL = """
-- apply() removes made-up dates and records each change in data_repair_undo.
"""


def apply(conn: sqlite3.Connection) -> None:
    from superlocalmemory.storage.fabricated_dates_repair import apply as repair

    if not conn.execute("SELECT 1 FROM sqlite_master WHERE name='atomic_facts'").fetchone():
        logger.debug("M065: no atomic_facts table yet; nothing to check")
        return
    result = repair(conn, NAME)
    logger.info("M065: made-up dates removed from %d fact(s) (%d sentence(s) rewritten, "
                "%d with date rows cleared); every change is in data_repair_undo",
                result.facts, result.text_changed, result.dates_removed)


REPAIR_NOT_APPLICABLE = (
    "A one-time data repair with no schema end state: replaying it would find "
    "nothing left to change, and its undo record is the way back."
)
