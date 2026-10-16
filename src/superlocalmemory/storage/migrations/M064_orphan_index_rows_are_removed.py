# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Index rows of a memory that is gone are removed, once, when 4.2.8 first starts.

WHAT WAS WRONG
A delete on 13 Oct 2026 failed partway (the vector step raised while an
embedding re-index was swapping the index) and left a ``vector_row_map`` row and
its vector for a memory that no longer existed. The metadata cleanup then took
only the ``embedding_metadata`` row. The vector stayed referenced by the map row,
so the sweep for unreferenced vectors never removed it, and ``slm doctor`` kept
counting a memory as missing from the search index.

WHY IT IS SAFE
A row goes only when its ``fact_id`` is absent from ``atomic_facts``, and a
vector only when no stored fact's mapping points at it
(``vector_residue.remove_orphan_index_rows``). Nothing of a stored memory is
read or changed. The routine maintenance pass now does the same, so this runs
once and the daemon keeps the index clean afterwards. Running it again removes
nothing. Without the vector extension the map rows still go and the routine
sweep takes the vectors.
"""

from __future__ import annotations

import logging
import sqlite3

logger = logging.getLogger(__name__)

NAME = "M064_orphan_index_rows_are_removed"
DB_TARGET = "memory"
#: Data only; an older build reads the smaller index exactly as it read the larger.
DOWNGRADE_FLOOR = 51

DDL = """
-- No schema change. apply() removes the index rows of memories that are gone.
"""


def apply(conn: sqlite3.Connection) -> None:
    """Remove orphan index rows in one transaction; log what went."""
    from superlocalmemory.storage.vector_residue import load_vec, remove_orphan_index_rows

    if not conn.execute("SELECT 1 FROM sqlite_master WHERE name='atomic_facts'").fetchone():
        logger.debug("M064: no atomic_facts table yet; nothing to check")
        return
    vec = load_vec(conn)
    removed = remove_orphan_index_rows(conn)
    logger.info("M064: removed %d vector(s), %d metadata and %d map row(s) of memories "
                "that no longer exist%s", removed.vectors, removed.metadata, removed.row_map,
                "" if vec else " (vector extension unavailable: vectors left to the routine sweep)")


REPAIR_NOT_APPLICABLE = (
    "A one-time data cleanup with no end state to drift from: the routine "
    "maintenance pass removes the same rows from now on, and replaying it from "
    "the runner would only repeat a removal that is already idempotent."
)
