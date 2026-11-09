# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""A restore carries the cases a user dismissed or overtook since the copy (issue #309).

The copy still holds those cases as ``proposed``. Without this, restoring it
brought every dismissed case back as an open one, and a fact the person deleted
after dismissing its case could not be deleted again on the copy (the open case
protected it), so it came back too.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from pathlib import Path

from superlocalmemory.storage import backup
from superlocalmemory.storage import upgrade_restore as ur
from superlocalmemory.storage._restore_delta import export_delta, open_compare

from ._upgrade_store import add_memory, current_store, delete_fact, fact_ids


def _propose(memory_db: Path, case_id: str, pred: str, succ: str,
             reason: str = "temporal_contradiction") -> None:
    from superlocalmemory.storage.correction_cases import CorrectionActor, propose_on_connection

    actor = CorrectionActor("canonical-writer", "host_attested", "canonical_writer")
    with closing(sqlite3.connect(str(memory_db), isolation_level=None)) as conn:
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("BEGIN IMMEDIATE")
        propose_on_connection(conn, case_id=case_id, profile_id="default", scope="personal",
                              predecessor_fact_id=pred, successor_fact_id=succ,
                              reason_code=reason, actor=actor, idempotency_key=f"k-{case_id}",
                              is_profile_active=lambda p: True, is_actor_trusted=lambda a: True)
        conn.execute("COMMIT")


def _dismiss(memory_db: Path, case_id: str, action: str = "dismissed") -> None:
    from superlocalmemory.core import overtaken_cases as ot

    with closing(sqlite3.connect(str(memory_db), isolation_level=None)) as conn:
        conn.row_factory = sqlite3.Row
        conn.execute("BEGIN IMMEDIATE")
        case = [c for c in ot.cases_naming(conn, ["f1", "f3", "f5"]) if c["case_id"] == case_id]
        assert ot.overtake(conn, case, user_action=action, actor_id="cli", operation_id="op-d",
                           closed_reason="dismissed in bulk by the user") == [case_id]
        conn.execute("COMMIT")


def _put_back(memory_db: Path, case_id: str) -> None:
    from superlocalmemory.core import overtaken_cases as ot

    with closing(sqlite3.connect(str(memory_db), isolation_level=None)) as conn:
        conn.row_factory = sqlite3.Row
        conn.execute("BEGIN IMMEDIATE")
        ot.restore(conn, case_id, actor_id="cli-person")
        conn.execute("COMMIT")


def _scenario(tmp_path: Path, dismissed_before: tuple[str, ...] = ()):
    learning_db, memory_db = current_store(tmp_path)
    add_memory(memory_db, "m1", ["Deploys go out on Tuesdays", "Deploys go out on Thursdays",
                                 "Use signed wheels", "Use unsigned wheels"],
               fact_ids=["f1", "f2", "f3", "f4"])
    add_memory(memory_db, "m2", ["The office moved to Pune", "The office is in Pune"],
               fact_ids=["f5", "f6"])
    _propose(memory_db, "c1", "f1", "f2")
    _propose(memory_db, "c2", "f3", "f4", "consolidation_update")
    _propose(memory_db, "c3", "f5", "f6", "consolidation_supersede")
    for case_id in dismissed_before:
        _dismiss(memory_db, case_id)
    backup._pre_migration_backup(learning_db, memory_db,
                                 backups_root=tmp_path / "pre-migration-snapshots")
    [point] = ur.list_restore_points(tmp_path)
    return learning_db, memory_db, point


def _restore(tmp_path, learning_db, memory_db, point):
    ur.request_restore(point.point_id, requested_by="t", data_root=tmp_path, memory_db=memory_db)
    return ur.perform_pending_restore(tmp_path, memory_db, learning_db)


def _rows(memory_db: Path, sql: str) -> list[tuple]:
    with closing(sqlite3.connect(f"file:{memory_db}?mode=ro", uri=True)) as conn:
        return conn.execute(sql).fetchall()


def test_a_dismissed_case_stays_dismissed_across_a_restore(tmp_path) -> None:
    learning_db, memory_db, point = _scenario(tmp_path)
    _dismiss(memory_db, "c1")
    _dismiss(memory_db, "c3")

    outcome = _restore(tmp_path, learning_db, memory_db, point)

    assert outcome.status == "restored", outcome.message
    assert _rows(memory_db, "SELECT case_id FROM correction_cases ORDER BY case_id") == [("c2",)]
    assert _rows(memory_db, "SELECT case_id, user_action, closed_reason, restored_at "
                            "FROM correction_cases_overtaken ORDER BY case_id") == [
        ("c1", "dismissed", "dismissed in bulk by the user", None),
        ("c3", "dismissed", "dismissed in bulk by the user", None)]
    assert _rows(memory_db, "SELECT COUNT(*) FROM correction_events WHERE case_id IN "
                            "('c1', 'c3')") == [(0,)]
    assert outcome.applied["overtaken_carried"] == 2


def test_the_carried_case_can_still_be_restored(tmp_path) -> None:
    from superlocalmemory.core import overtaken_cases as ot

    learning_db, memory_db, point = _scenario(tmp_path)
    _dismiss(memory_db, "c1")
    assert _restore(tmp_path, learning_db, memory_db, point).status == "restored"

    with closing(sqlite3.connect(str(memory_db), isolation_level=None)) as conn:
        conn.row_factory = sqlite3.Row
        conn.execute("BEGIN IMMEDIATE")
        assert ot.restore(conn, "c1") == {"restored": "c1", "events": 1}
        conn.execute("COMMIT")
    assert _rows(memory_db, "SELECT status FROM correction_cases WHERE case_id = 'c1'") == [
        ("proposed",)]


def test_a_fact_deleted_after_its_case_was_dismissed_stays_deleted(tmp_path) -> None:
    learning_db, memory_db, point = _scenario(tmp_path)
    _dismiss(memory_db, "c1")
    delete_fact(memory_db, "f1")          # allowed only because the case was closed

    outcome = _restore(tmp_path, learning_db, memory_db, point)

    assert outcome.status == "restored", outcome.message
    assert "f1" not in fact_ids(memory_db)
    assert outcome.applied["kept_protected"] == 0
    assert _rows(memory_db, "SELECT case_id FROM correction_cases WHERE case_id = 'c1'") == []


def test_a_case_born_and_dismissed_after_the_copy_is_not_invented(tmp_path) -> None:
    learning_db, memory_db, point = _scenario(tmp_path)
    add_memory(memory_db, "m3", ["a", "b"], fact_ids=["f7", "f8"])
    _propose(memory_db, "late", "f7", "f8")
    from superlocalmemory.core import overtaken_cases as ot

    with closing(sqlite3.connect(str(memory_db), isolation_level=None)) as conn:
        conn.row_factory = sqlite3.Row
        conn.execute("BEGIN IMMEDIATE")
        ot.overtake(conn, ot.cases_naming(conn, ["f7"]), user_action="dismissed",
                    actor_id="cli", operation_id="op")
        conn.execute("COMMIT")

    outcome = _restore(tmp_path, learning_db, memory_db, point)

    assert outcome.status == "restored", outcome.message
    assert outcome.applied["overtaken_carried"] == 0
    assert _rows(memory_db, "SELECT COUNT(*) FROM correction_cases_overtaken") == [(0,)]


def test_the_export_lists_new_overtaken_rows_only_in_deleted_jsonl(tmp_path) -> None:
    learning_db, memory_db, point = _scenario(tmp_path)
    _dismiss(memory_db, "c2")
    export = tmp_path / "export"
    with closing(open_compare(memory_db, point.memory_snapshot)) as conn:
        export_delta(conn, export)

    lines = [json.loads(x) for x in (export / "deleted.jsonl").read_text().splitlines()]
    [row] = [x for x in lines if x["type"] == "overtaken"]
    assert (row["case_id"], row["user_action"], row["profile_id"]) == ("c2", "dismissed", "default")
    assert "overtake_id" not in row and row["restored_at"] is None
    assert not (export / "overtaken.jsonl").exists()


# -- a case the person PUT BACK after the copy ---------------------------------------
#
# The delta's model is "carry forward what the person did after the copy" (a restore
# must not cost them anything), never "roll back to the copy". So putting a case back
# is carried like dismissing one is.


def test_a_case_dismissed_before_the_copy_and_put_back_after_it_is_open_after_a_restore(
        tmp_path) -> None:
    learning_db, memory_db, point = _scenario(tmp_path, dismissed_before=("c1",))
    _put_back(memory_db, "c1")
    [(restored_at,)] = _rows(memory_db, "SELECT restored_at FROM correction_cases_overtaken")

    outcome = _restore(tmp_path, learning_db, memory_db, point)

    assert outcome.status == "restored", outcome.message
    assert outcome.applied["overtaken_reopened"] == 1
    assert _rows(memory_db, "SELECT case_id, status FROM correction_cases ORDER BY case_id") == [
        ("c1", "proposed"), ("c2", "proposed"), ("c3", "proposed")]
    assert _rows(memory_db, "SELECT case_id, restored_at, restored_by FROM "
                            "correction_cases_overtaken") == [("c1", restored_at, "cli-person")]
    assert _rows(memory_db, "SELECT COUNT(*) FROM correction_events WHERE case_id = 'c1'") == [(1,)]


def test_a_case_dismissed_and_put_back_after_the_copy_is_open_as_in_the_copy(tmp_path) -> None:
    learning_db, memory_db, point = _scenario(tmp_path)
    _dismiss(memory_db, "c1")
    _put_back(memory_db, "c1")

    outcome = _restore(tmp_path, learning_db, memory_db, point)

    assert outcome.status == "restored", outcome.message
    assert outcome.applied["overtaken_carried"] == 0 and outcome.applied["overtaken_reopened"] == 0
    assert _rows(memory_db, "SELECT case_id FROM correction_cases ORDER BY case_id") == [
        ("c1",), ("c2",), ("c3",)]
    assert _rows(memory_db, "SELECT COUNT(*) FROM correction_cases_overtaken") == [(0,)]


def test_put_back_then_deleted_after_the_copy_stays_deleted(tmp_path) -> None:
    learning_db, memory_db, point = _scenario(tmp_path, dismissed_before=("c1",))
    _put_back(memory_db, "c1")
    _dismiss(memory_db, "c1", action="delete")     # the person's delete closes it again
    delete_fact(memory_db, "f1")

    outcome = _restore(tmp_path, learning_db, memory_db, point)

    assert outcome.status == "restored", outcome.message
    assert "f1" not in fact_ids(memory_db)
    assert outcome.applied["kept_protected"] == 0
    assert _rows(memory_db, "SELECT case_id FROM correction_cases ORDER BY case_id") == [
        ("c2",), ("c3",)]
    assert _rows(memory_db, "SELECT user_action, restored_at IS NULL FROM "
                            "correction_cases_overtaken ORDER BY overtake_id") == [
        ("dismissed", 0), ("delete", 1)]

