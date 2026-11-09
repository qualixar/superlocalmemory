# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""M054: saved views, one additive table in learning.db.

Three promises, each checked by running the real runner, not by reading code:

1. The upgrade copies learning.db only. memory.db is never copied for it.
2. ``slm db prepare-downgrade`` still works, and a 4.1.20 build then opens the
   store: the table is one 4.1.20 has never heard of and leaves alone.
3. Running it twice changes nothing, and its verify / repair hooks put a
   damaged table back.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

import superlocalmemory.storage._schema_version as sv
from superlocalmemory.storage import _snapshot_manifest as sm
from superlocalmemory.storage import migration_runner as mr
from superlocalmemory.storage import upgrade_restore as ur
from superlocalmemory.storage._downgrade import _blockers
from superlocalmemory.storage._migration_internals import _MODULES
from superlocalmemory.storage.migrations import M054_saved_views as M054
from tests.test_storage._upgrade_store import clear_snapshots, current_store


def _snapshot_files(root: Path) -> list[str]:
    snaps = root / "pre-migration-snapshots"
    if not snaps.is_dir():
        return []
    return sorted(p.name.split("-")[0] for p in snaps.iterdir()
                  if p.name.endswith("-pre-migration.db"))


def _forget(db: Path) -> None:
    """Make M054 pending again and drop its table: a 4.1.20 store."""
    with sqlite3.connect(db) as conn:
        conn.execute("DROP TABLE IF EXISTS saved_views")
        conn.execute("DELETE FROM migration_log WHERE name = ?", (M054.NAME,))


def _a_view(db: Path) -> None:
    with sqlite3.connect(db) as conn:
        conn.execute(
            "INSERT INTO saved_views (profile_id, view_id, name, name_key, query, "
            "created_at, updated_at) VALUES ('default', 'v1', 'Work', 'work', "
            "'what did I ship', '2026-10-05T00:00:00Z', '2026-10-05T00:00:00Z')")


def test_registered_eager_on_learning() -> None:
    m = next(m for m in mr.MIGRATIONS if m.name == M054.NAME)
    assert m.db_target == "learning" == M054.DB_TARGET
    assert _MODULES[M054.NAME] is M054
    assert not [d for d in mr.DEFERRED_MIGRATIONS if d.name == M054.NAME]
    assert sv.SUPPORTED_SCHEMA_VERSION >= 54  # 55 since M055 (learning attribution)
    assert M054.BREAKING_VERSION == 0


class TestTheUpgradeCopiesOnlyLearningDb:
    def test_a_4_1_20_store_upgrades_with_one_learning_copy(self, tmp_path) -> None:
        learning, memory = current_store(tmp_path)
        _forget(learning)
        result = mr.apply_all(learning, memory)
        assert M054.NAME in result["applied"] and result["failed"] == []
        assert _snapshot_files(tmp_path) == ["learning"], (
            "memory.db was copied for a change that only touches learning.db")
        with sqlite3.connect(learning) as conn:
            assert M054.verify(conn) is True
        with sqlite3.connect(memory) as conn:
            assert not conn.execute(
                "SELECT 1 FROM sqlite_master WHERE name = 'saved_views'").fetchone()


class TestGoingBackStillWorks:
    @pytest.fixture()
    def store(self, tmp_path, monkeypatch):
        learning, memory = current_store(tmp_path)
        (tmp_path / ".last_version").write_text("4.1.21", encoding="utf-8")
        monkeypatch.setattr(sm, "package_version", lambda: "4.1.21")
        monkeypatch.setattr(sv, "_detect_all_installs", lambda: [])
        _a_view(learning)
        return tmp_path, learning, memory

    def test_the_floor_permits_every_older_release_since_4_1_18(self, store) -> None:
        _root, learning, memory = store
        for target in (53, 52, 51):
            assert not [p for p in _blockers(learning, memory, target) if M054.NAME in p]

    def test_prepare_downgrade_names_4_1_20_and_4_1_20_opens_the_store(
            self, store, monkeypatch) -> None:
        root, learning, memory = store
        report = ur.prepare_downgrade(requested_by="test", data_root=root,
                                      memory_db=memory, learning_db=learning)
        assert report.prepared and report.target_version == "4.1.20"
        # What a 4.1.20 build does at start: check the stamp against its own
        # ceiling, then run its own catalogue, which has no M054.
        monkeypatch.setattr(sv, "SUPPORTED_SCHEMA_VERSION", 53)
        sv.check_version_or_raise(learning)
        sv.check_version_or_raise(memory)
        older = [m for m in mr.MIGRATIONS if m.name != M054.NAME]
        monkeypatch.setattr(mr, "MIGRATIONS", older)
        result = mr.apply_all(learning, memory)
        assert result["failed"] == [], result
        with sqlite3.connect(learning) as conn:      # left alone, not dropped
            assert conn.execute("SELECT name FROM saved_views").fetchall() == [("Work",)]


class TestIdempotentAndRepairable:
    def test_running_twice_is_a_no_op(self, tmp_path) -> None:
        learning, memory = current_store(tmp_path)
        _a_view(learning)
        again = mr.apply_all(learning, memory)
        assert M054.NAME not in again["applied"] and again["failed"] == []
        with sqlite3.connect(learning) as conn:
            conn.executescript(M054.DDL)             # the DDL itself is re-runnable
            assert conn.execute("SELECT COUNT(*) FROM saved_views").fetchone() == (1,)
        assert _snapshot_files(tmp_path) == [], "a no-op start copied a database"

    def test_repair_restores_a_dropped_index(self, tmp_path) -> None:
        learning, _memory = current_store(tmp_path)
        with sqlite3.connect(learning) as conn:
            conn.execute("DROP INDEX idx_saved_views_profile_name")
            assert M054.verify(conn) is False
            M054.repair(conn)
            assert M054.verify(conn) is True

    def test_the_runner_repairs_a_dropped_table_on_its_own(self, tmp_path) -> None:
        learning, memory = current_store(tmp_path)
        with sqlite3.connect(learning) as conn:
            conn.execute("DROP TABLE saved_views")
        result = mr.apply_all(learning, memory)
        assert result["failed"] == [], result
        with sqlite3.connect(learning) as conn:
            assert M054.verify(conn) is True
        clear_snapshots(tmp_path)

    def test_names_are_unique_per_profile_not_globally(self, tmp_path) -> None:
        learning, _memory = current_store(tmp_path)
        _a_view(learning)
        with sqlite3.connect(learning) as conn:
            conn.execute(
                "INSERT INTO saved_views (profile_id, view_id, name, name_key, query, "
                "created_at, updated_at) VALUES ('work', 'v2', 'Work', 'work', 'q', "
                "'t', 't')")
            with pytest.raises(sqlite3.IntegrityError):
                conn.execute(
                    "INSERT INTO saved_views (profile_id, view_id, name, name_key, "
                    "query, created_at, updated_at) VALUES ('default', 'v3', 'WORK', "
                    "'work', 'q', 't', 't')")
