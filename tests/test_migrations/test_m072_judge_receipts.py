# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
from __future__ import annotations

import sqlite3

from superlocalmemory.storage._migration_internals import _MODULES
from superlocalmemory.storage.migration_runner import DEFERRED_MIGRATIONS
from superlocalmemory.storage.migrations import M072_judge_receipts as M072


def test_registered_additive_idempotent_repairable() -> None:
    assert _MODULES[M072.NAME] is M072 and M072.DOWNGRADE_FLOOR == 51
    assert M072.NAME in {m.name for m in DEFERRED_MIGRATIONS}
    conn = sqlite3.connect(":memory:")
    assert not M072.verify(conn)
    M072.apply(conn)
    M072.apply(conn)
    assert M072.verify(conn)
    conn.execute("DROP INDEX idx_judge_receipts_subject")
    assert not M072.verify(conn)
    M072.repair(conn)
    assert M072.verify(conn)
