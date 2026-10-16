# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Memories archived by the wrong clock come back, once, when 4.2.8 first starts.

WHAT WAS WRONG
With ``math.ebbinghaus_langevin_coupling_enabled`` on (default off), the
maintenance ELC step fed real elapsed time to an hours-tuned retention curve. A
memory three months old computed R ~ 0 and was archived, while the store-scaled
clock the decay pass uses (GitHub #136) called it warm. One contributor's store
lost 3,281 memories to the archive in a single 02:00 tick. PR #266 gave the ELC
step the store-scaled clock. Separately, the maintenance Langevin pass decided
tiers on the default forgetting settings instead of the person's own, and on
one real store archived 630 memories those settings keep. 4.2.8 makes every
tier-setting pass use one function with the person's settings
(``EbbinghausCurve.retention_and_zone``); this repairs what was archived.

WHAT THIS DOES
``storage.archive_repair``: an archived memory with no recorded decision behind
the archive (not the user's, not erasure, not a merge or supersede, not a
consolidation, not pinned or otherwise protected) gets its zone recomputed with
the decay pass's own function (``EbbinghausCurve.batch_compute_retention`` and
the user's inactivity policy, if one is set). Those it calls active, warm or
cold go back to that zone; the rest stay archived. Every change is written to
``data_repair_undo``; ``data_repair_undo.undo(conn, NAME)`` reverses it and
``slm doctor`` reports it. It runs for every store and is a no-op when nothing
qualifies.

WHY IT IS SAFE
One transaction; a crash leaves the store as it was and the runner retries.
Running it again finds nothing archived that the clock calls live.
"""

from __future__ import annotations

import logging
import sqlite3

logger = logging.getLogger(__name__)

NAME = "M066_wrongly_archived_memories_are_restored"
DB_TARGET = "memory"
#: Data only (the undo table is shared with M065); an older build reads both.
DOWNGRADE_FLOOR = 51

DDL = """
-- apply() restores wrongly archived memories, recording each change in data_repair_undo.
"""


def apply(conn: sqlite3.Connection) -> None:
    from superlocalmemory.storage.archive_repair import apply as repair

    if not conn.execute("SELECT 1 FROM sqlite_master WHERE name='atomic_facts'").fetchone():
        logger.debug("M066: no atomic_facts table yet; nothing to check")
        return
    result = repair(conn, NAME)
    logger.info("M066: %d archived memorie(s) checked, %d restored to their zone %s; "
                "every change is in data_repair_undo",
                result.checked, result.restored, result.by_zone)


REPAIR_NOT_APPLICABLE = (
    "A one-time data repair with no schema end state: replaying it would find "
    "nothing left to restore, and its undo record is the way back."
)
