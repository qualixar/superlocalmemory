# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""A restore repeats only the deletions a person or an erasure asked for (L1-01).

The product records every forget and every erasure as a tombstone (and an
erasure receipt) before it deletes. A row that is missing from the live store
with no such record was LOST -- a bug, a bad merge, a sync client -- and the
copy is exactly what the person needs to get it back.
"""

from __future__ import annotations

import json
from contextlib import closing
from pathlib import Path

from superlocalmemory.storage import backup
from superlocalmemory.storage import upgrade_restore as ur
from superlocalmemory.storage._restore_delta import export_delta, open_compare

from ._upgrade_store import (
    add_memory,
    add_profile,
    current_store,
    delete_fact,
    delete_memory,
    fact_ids,
    lose_rows,
    memory_ids,
    record_erasure,
)


def _scenario(tmp_path: Path):
    learning_db, memory_db = current_store(tmp_path)
    add_memory(memory_db, "m1", ["Deploys go out on Tuesdays", "Use signed wheels"],
               fact_ids=["f1", "f2"])
    add_memory(memory_db, "m2", ["The office moved to Pune"], fact_ids=["f3"])
    backup._pre_migration_backup(learning_db, memory_db,
                                 backups_root=tmp_path / "pre-migration-snapshots")
    [point] = ur.list_restore_points(tmp_path)
    return learning_db, memory_db, point


def _restore(tmp_path, learning_db, memory_db, point):
    ur.request_restore(point.point_id, requested_by="t", data_root=tmp_path,
                       memory_db=memory_db)
    return ur.perform_pending_restore(tmp_path, memory_db, learning_db)


def test_rows_the_store_lost_come_back(tmp_path) -> None:
    learning_db, memory_db, point = _scenario(tmp_path)
    lose_rows(memory_db, fact_ids=["f1", "f2", "f3"], memory_ids=["m1", "m2"])

    preview = ur.preview_restore(point.point_id, data_root=tmp_path, memory_db=memory_db)
    outcome = _restore(tmp_path, learning_db, memory_db, point)

    assert (preview.deleted_facts, preview.deleted_memories) == (0, 0)
    assert (preview.returning_facts, preview.returning_memories) == (3, 2)
    assert any("no deletion was recorded" in w for w in preview.warnings), preview.warnings
    assert outcome.status == "restored", outcome.message
    assert outcome.applied["facts_deleted"] == 0
    assert fact_ids(memory_db) == {"f1", "f2", "f3"}
    assert memory_ids(memory_db) == {"m1", "m2"}


def test_a_forgotten_memory_stays_forgotten_while_a_lost_one_returns(tmp_path) -> None:
    learning_db, memory_db, point = _scenario(tmp_path)
    delete_memory(memory_db, "m2")                      # forgotten: f3 tombstoned
    lose_rows(memory_db, fact_ids=["f1"])               # lost: nothing recorded

    preview = ur.preview_restore(point.point_id, data_root=tmp_path, memory_db=memory_db)
    export = tmp_path / "export-for-test"
    with closing(open_compare(memory_db, point.memory_snapshot)) as conn:
        export_delta(conn, export)
    outcome = _restore(tmp_path, learning_db, memory_db, point)

    assert (preview.deleted_facts, preview.deleted_memories, preview.returning_facts) == (1, 1, 1)
    assert outcome.status == "restored", outcome.message
    assert fact_ids(memory_db) == {"f1", "f2"}
    assert memory_ids(memory_db) == {"m1"}
    deleted = [json.loads(line) for line in (export / "deleted.jsonl").read_text(encoding="utf-8").splitlines()]
    assert {d.get("fact_id") or d.get("memory_id") for d in deleted} == {"f3", "m2"}
    assert all(d["intent"] == "tombstone" for d in deleted), deleted


def test_one_forgotten_fact_does_not_take_its_lost_sibling(tmp_path) -> None:
    learning_db, memory_db, point = _scenario(tmp_path)
    delete_fact(memory_db, "f1")                        # forgotten
    lose_rows(memory_db, fact_ids=["f2"], memory_ids=["m1"])   # then the rest was lost

    outcome = _restore(tmp_path, learning_db, memory_db, point)

    assert outcome.status == "restored", outcome.message
    assert fact_ids(memory_db) == {"f2", "f3"}
    assert memory_ids(memory_db) == {"m1", "m2"}, "m1 still holds f2"


def test_an_erased_profile_is_not_counted_as_coming_back(tmp_path) -> None:
    learning_db, memory_db = current_store(tmp_path)
    add_profile(memory_db, "alice")
    add_memory(memory_db, "ma", ["Alice's address is 12 Elm St"], profile="alice",
               fact_ids=["fa1"])
    backup._pre_migration_backup(learning_db, memory_db,
                                 backups_root=tmp_path / "pre-migration-snapshots")
    [point] = ur.list_restore_points(tmp_path)
    lose_rows(memory_db, fact_ids=["fa1"], memory_ids=["ma"])   # an erasure's own delete
    record_erasure(memory_db, profile_id="alice", subject_type="profile",
                   subject_id="alice", fact_ids=[], memory_ids={})

    preview = ur.preview_restore(point.point_id, data_root=tmp_path, memory_db=memory_db)
    outcome = _restore(tmp_path, learning_db, memory_db, point)

    assert (preview.returning_facts, preview.returning_memories) == (0, 0)
    assert outcome.status == "restored" and fact_ids(memory_db, "alice") == set()
