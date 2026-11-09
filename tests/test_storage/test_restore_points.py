# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Restore points: listed with what they hold, previewed without touching anything."""

from __future__ import annotations

import hashlib
import sqlite3
from contextlib import closing
from pathlib import Path

from superlocalmemory.storage import backup
from superlocalmemory.storage import upgrade_restore as ur

from ._upgrade_store import (
    add_memory,
    confirm_kind,
    current_store,
    delete_fact,
    delete_memory,
    file_digest,
)


def _db(path: Path) -> None:
    with closing(sqlite3.connect(path)) as conn:
        conn.execute("CREATE TABLE atomic_facts (fact_id TEXT PRIMARY KEY)")
        conn.execute("CREATE TABLE memories (memory_id TEXT PRIMARY KEY)")
        conn.commit()


def test_legacy_generation_listed_without_manifest(tmp_path) -> None:
    learning_db, memory_db = current_store(tmp_path)
    snaps = tmp_path / "pre-migration-snapshots"
    snaps.mkdir(exist_ok=True)
    _db(snaps / "memory-20260819-120000-pre-migration.db")          # first release naming
    _db(snaps / "learning-20260819-120000-pre-migration.db")
    _db(snaps / "memory-20260823-144350-pre-4.1.0.db")              # named for a version
    add_memory(memory_db, "m1", ["one fact"])
    backup._pre_migration_backup(learning_db, memory_db, backups_root=snaps)

    points = ur.list_restore_points(tmp_path)

    by_id = {p.point_id: p for p in points}
    assert "20260819-120000" in by_id and "20260823-144350-pre-4.1.0" in by_id
    first = by_id["20260819-120000"]
    assert first.legacy and first.sha256 is None and first.learning_snapshot is not None
    assert by_id["20260823-144350-pre-4.1.0"].learning_snapshot is None
    current = [p for p in points if not p.legacy]
    assert len(current) == 1 and current[0].facts == 1 and current[0].sha256
    assert points[0] is current[0], "newest first"
    assert all(isinstance(p.as_dict()["memory_snapshot"], str) for p in points)


def _store_with_history(tmp_path: Path):
    learning_db, memory_db = current_store(tmp_path)
    add_memory(memory_db, "m1", ["Deploys go out on Tuesdays", "Use signed wheels"],
               fact_ids=["f1", "f2"])
    add_memory(memory_db, "m2", ["The office moved to Pune"], fact_ids=["f3"])
    snaps = tmp_path / "pre-migration-snapshots"
    backup._pre_migration_backup(learning_db, memory_db, backups_root=snaps)
    [point] = ur.list_restore_points(tmp_path)
    return learning_db, memory_db, point


def test_preview_counts_added_deleted_and_kind_edits(tmp_path) -> None:
    _l, memory_db, point = _store_with_history(tmp_path)
    add_memory(memory_db, "m3", ["Standup moved to 10am"], fact_ids=["f4"])   # added later
    delete_fact(memory_db, "f3")                                               # deleted later
    delete_memory(memory_db, "m2")
    confirm_kind(memory_db, "f1", "rule")                                      # confirmed later

    preview = ur.preview_restore(point.point_id, data_root=tmp_path, memory_db=memory_db)

    assert preview.verified and preview.disk_ok and not preview.problems
    assert (preview.added_memories, preview.deleted_facts, preview.deleted_memories,
            preview.kind_edits) == (1, 1, 1, 1)
    assert (preview.facts_in_snapshot, preview.facts_now) == (3, 3)
    assert preview.as_dict()["point_id"] == point.point_id


def test_preview_changes_no_byte_of_the_live_store(tmp_path) -> None:
    _l, memory_db, point = _store_with_history(tmp_path)
    add_memory(memory_db, "m3", ["Standup moved to 10am"])
    watched = [memory_db, Path(f"{memory_db}-wal"), point.memory_snapshot]
    before = [file_digest(p) for p in watched]
    snapshot_dir = sorted(p.name for p in point.memory_snapshot.parent.iterdir())

    ur.preview_restore(point.point_id, data_root=tmp_path, memory_db=memory_db)

    after = [file_digest(p) for p in watched]
    assert after[0] == before[0] and after[2] == before[2]
    # A read-only reader of a WAL store may create an EMPTY log beside it when
    # none existed (SQLite does this); it never writes a frame into it.
    empty = hashlib.sha256(b"").hexdigest()
    assert after[1] == before[1] or (before[1] is None and after[1] == empty)
    assert sorted(p.name for p in point.memory_snapshot.parent.iterdir()) == snapshot_dir
    assert not (tmp_path / "restore-intent.json").exists()


def test_preview_of_a_damaged_copy_says_so_and_offers_nothing(tmp_path) -> None:
    _l, memory_db, point = _store_with_history(tmp_path)
    data = bytearray(point.memory_snapshot.read_bytes())
    data[len(data) // 2] ^= 0xFF
    point.memory_snapshot.write_bytes(bytes(data))

    preview = ur.preview_restore(point.point_id, data_root=tmp_path, memory_db=memory_db)

    assert not preview.verified and "changed after it was taken" in preview.problems[0]


def test_unknown_or_hostile_point_ids_are_refused(tmp_path) -> None:
    _l, memory_db, _point = _store_with_history(tmp_path)
    for bad in ("../../etc/passwd", "", "nope", "a/b"):
        try:
            ur.preview_restore(bad, data_root=tmp_path, memory_db=memory_db)
        except ur.RestoreRefusedError:
            continue
        raise AssertionError(f"{bad!r} was accepted")
