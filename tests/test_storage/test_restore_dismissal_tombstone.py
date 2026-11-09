# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""A restore keeps the dismissal tombstone of a pair the copy already held (issue #309).

A save that was pending in the copy can come back with the same operation and fact
ids, which makes the same deterministic case id. The dismissal row the person's
``slm corrections dismiss`` left is what refuses it, so the restore must carry that
row even when the copy has no open case to close.
"""

from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from superlocalmemory.storage import backup
from superlocalmemory.storage import upgrade_restore as ur

from ._upgrade_store import add_memory, current_store
from .test_restore_overtaken_cases import _propose, _restore, _rows


def _snapshot_with_a_pair_and_no_case(tmp_path: Path):
    learning_db, memory_db = current_store(tmp_path)
    add_memory(memory_db, "m1", ["Deploys go out on Tuesdays", "Deploys go out on Thursdays"],
               fact_ids=["f1", "f2"])
    backup._pre_migration_backup(learning_db, memory_db,
                                 backups_root=tmp_path / "pre-migration-snapshots")
    [point] = ur.list_restore_points(tmp_path)
    return learning_db, memory_db, point


def _overtake(memory_db: Path, fact_id: str, action: str) -> None:
    from superlocalmemory.core import overtaken_cases as ot

    with closing(sqlite3.connect(str(memory_db), isolation_level=None)) as conn:
        conn.row_factory = sqlite3.Row
        conn.execute("BEGIN IMMEDIATE")
        ot.overtake(conn, ot.cases_naming(conn, [fact_id]), user_action=action,
                    actor_id="cli", operation_id="op-d",
                    closed_reason="dismissed in bulk by the user")
        conn.execute("COMMIT")


def _repropose(memory_db: Path) -> None:
    _propose(memory_db, "late", "f1", "f2")


def test_a_dismissal_of_a_case_born_after_the_copy_still_refuses_the_same_proposal(
        tmp_path) -> None:
    from superlocalmemory.storage.correction_cases import CorrectionDismissedError

    learning_db, memory_db, point = _snapshot_with_a_pair_and_no_case(tmp_path)
    _propose(memory_db, "late", "f1", "f2")
    _overtake(memory_db, "f1", "dismissed")

    outcome = _restore(tmp_path, learning_db, memory_db, point)

    assert outcome.status == "restored", outcome.message
    assert outcome.applied["overtaken_carried"] == 1
    assert _rows(memory_db, "SELECT COUNT(*) FROM correction_cases") == [(0,)]   # no open case
    assert _rows(memory_db, "SELECT case_id, user_action, restored_at FROM "
                            "correction_cases_overtaken") == [("late", "dismissed", None)]
    with pytest.raises(CorrectionDismissedError):
        _repropose(memory_db)
    assert _rows(memory_db, "SELECT COUNT(*) FROM correction_cases") == [(0,)]


def test_only_a_dismissal_is_kept_not_other_overtaken_rows(tmp_path) -> None:
    learning_db, memory_db, point = _snapshot_with_a_pair_and_no_case(tmp_path)
    _propose(memory_db, "late", "f1", "f2")
    _overtake(memory_db, "f1", "delete")

    outcome = _restore(tmp_path, learning_db, memory_db, point)

    assert outcome.status == "restored", outcome.message
    assert outcome.applied["overtaken_carried"] == 0
    assert _rows(memory_db, "SELECT COUNT(*) FROM correction_cases_overtaken") == [(0,)]
