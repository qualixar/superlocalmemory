# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""An update copies only the databases it changes, and says so before it starts.

Seen in the 4.1.20 audit: the first start of 4.1.20 copied all of memory.db
(about 8 s per GB, before the daemon answered anything) although its one new
migration, M053, changes learning.db only -- and said nothing while it did.

And the guarantee that must survive the change: the copies of memory.db that a
person can go back to (restore points, and the copy taken before going back to
an older version) are never pushed out by copies of learning.db alone.
"""

# ruff: noqa: F811 (pytest fixtures are imported and redeclared as parameters)

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import pytest

from superlocalmemory.storage import backup
from superlocalmemory.storage._snapshot_retention import prune
from superlocalmemory.storage.upgrade_restore import list_restore_points, prepare_downgrade

from .test_one_start_keeps_one_safety_copy import (  # noqa: F401 — the fixture
    _clear_snapshots,
    _forget_migration,
    _generations,
    _start,
    store,
)

_LEARNING_ONLY = "M053_answer_check_history"           # eager, learning.db
_MEMORY_EAGER = "M004_cross_platform_sync_log"         # eager, memory.db
_MEMORY_DEFERRED = "M030_entity_explorer_indexes"      # deferred, memory.db


def _copies(snapshots: Path, stem: str) -> list[str]:
    if not snapshots.exists():
        return []
    return sorted(p.name for p in snapshots.iterdir()
                  if p.name.startswith(f"{stem}-") and p.name.endswith("-pre-migration.db"))


class TestOnlyWhatChangesIsCopied:
    def test_a_learning_only_update_does_not_copy_memory(self, store) -> None:
        learning_db, memory_db, snapshots = store
        _forget_migration(learning_db, _LEARNING_ONLY)
        eager, _deferred = _start(learning_db, memory_db)
        assert _LEARNING_ONLY in eager["applied"]
        assert eager["details"].get("_backup") == str(snapshots)
        assert len(_copies(snapshots, "learning")) == 1
        assert _copies(snapshots, "memory") == [], "memory.db was copied for nothing"

    def test_a_memory_update_still_copies_memory(self, store) -> None:
        learning_db, memory_db, snapshots = store
        _forget_migration(memory_db, _MEMORY_EAGER)
        _start(learning_db, memory_db)
        assert len(_copies(snapshots, "memory")) == 1
        assert _copies(snapshots, "learning") == []

    def test_one_copy_covers_the_deferred_pass_too(self, store) -> None:
        """Eager pass changes learning.db, deferred pass memory.db: still one copy,
        holding both, taken before anything changed."""
        learning_db, memory_db, snapshots = store
        _forget_migration(learning_db, _LEARNING_ONLY)
        _forget_migration(memory_db, _MEMORY_DEFERRED)
        eager, deferred = _start(learning_db, memory_db)
        assert _MEMORY_DEFERRED in deferred["applied"]
        assert len(_generations(snapshots)) == 1
        assert len(_copies(snapshots, "memory")) == 1
        assert len(_copies(snapshots, "learning")) == 1
        assert deferred["details"].get("_deferred_backup") == eager["details"]["_backup"]


class TestRestorePointsSurvive:
    def test_learning_only_updates_never_push_out_a_restore_point(self, store) -> None:
        learning_db, memory_db, snapshots = store
        _forget_migration(learning_db, _LEARNING_ONLY)
        _forget_migration(memory_db, _MEMORY_EAGER)
        _start(learning_db, memory_db)                  # a real upgrade: both copied
        points = [p.point_id for p in list_restore_points(memory_db.parent)]
        assert len(points) == 1
        for _ in range(3):                              # three learning-only updates
            time.sleep(0.002)
            _forget_migration(learning_db, _LEARNING_ONLY)
            _start(learning_db, memory_db)
        assert [p.point_id for p in list_restore_points(memory_db.parent)] == points
        assert len(_copies(snapshots, "learning")) == 3  # 2 newest + the point's own

    def test_learning_only_copies_never_count_against_restore_points(self, tmp_path) -> None:
        def make(stamp: str, *stems: str) -> None:
            for stem in stems:
                (tmp_path / f"{stem}-{stamp}-pre-migration.db").write_bytes(b"x")
        make("20261001-000000-000000", "memory", "learning")
        make("20261002-000000-000000", "memory", "learning")
        make("20261003-000000-000000", "learning")
        make("20261004-000000-000000", "learning")
        make("20261005-000000-000000", "learning")
        prune(tmp_path, keep=2)
        left = sorted(p.name for p in tmp_path.iterdir())
        assert "memory-20261001-000000-000000-pre-migration.db" in left
        assert "memory-20261002-000000-000000-pre-migration.db" in left
        assert "learning-20261003-000000-000000-pre-migration.db" not in left
        assert "learning-20261005-000000-000000-pre-migration.db" in left

    def test_going_back_still_copies_both_databases(self, store) -> None:
        learning_db, memory_db, snapshots = store
        report = prepare_downgrade(target_schema=51, requested_by="test",
                                   data_root=memory_db.parent, memory_db=memory_db,
                                   learning_db=learning_db)
        assert report.prepared and report.point_id
        assert len(_copies(snapshots, "memory")) == 1
        assert len(_copies(snapshots, "learning")) == 1


class TestTheCopyIsAnnounced:
    def test_size_and_time_are_said_before_the_copy_starts(self, store, monkeypatch,
                                                            caplog) -> None:
        learning_db, memory_db, snapshots = store
        _clear_snapshots(snapshots)
        said_first: list[bool] = []
        real = backup._backup_via_sqlite_api

        def copy(src, dest):
            said_first.append(any("safety copy" in r.getMessage() for r in caplog.records))
            return real(src, dest)
        monkeypatch.setattr(backup, "_backup_via_sqlite_api", copy)
        caplog.set_level(logging.INFO, logger="superlocalmemory.storage.backup")
        backup._pre_migration_backup(learning_db, memory_db, backups_root=snapshots,
                                     databases=frozenset({"learning"}))
        assert said_first == [True]
        message = next(r for r in caplog.records if "safety copy" in r.getMessage())
        text = message.getMessage()
        assert "learning.db" in text and "memory.db" not in text
        assert " MB" in text and ("under a second" in text or "about" in text)

    def test_a_long_copy_is_announced_as_a_warning(self, store, monkeypatch,
                                                    caplog) -> None:
        learning_db, memory_db, snapshots = store
        monkeypatch.setattr(backup, "_COPY_BYTES_PER_S", 1024)  # a very slow disk
        caplog.set_level(logging.INFO, logger="superlocalmemory.storage.backup")
        backup._pre_migration_backup(learning_db, memory_db, backups_root=snapshots)
        message = next(r for r in caplog.records if "safety copy" in r.getMessage())
        assert message.levelno == logging.WARNING
        assert "memory.db and learning.db" in message.getMessage()
        assert "about" in message.getMessage() and "seconds" in message.getMessage()

    def test_an_unknown_database_name_is_refused(self, store) -> None:
        learning_db, memory_db, snapshots = store
        with pytest.raises(ValueError):
            backup._pre_migration_backup(learning_db, memory_db, backups_root=snapshots,
                                         databases=frozenset({"mem"}))
        assert not os.listdir(snapshots)
