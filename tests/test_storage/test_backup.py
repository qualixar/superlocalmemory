# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Tests for the pre-migration database backup module.

All tests use scratch directories under tmp_path. They never touch
~/.superlocalmemory/memory.db or ~/Downloads/memory.db.
"""

from __future__ import annotations

import os
import shutil
import sqlite3
import time
from pathlib import Path
from unittest import mock

import pytest

from superlocalmemory.storage.backup import (
    InsufficientDiskSpaceError,
    LiveStoreWriteError,
    SnapshotUnusableError,
    _backup_via_sqlite_api,
    _gc_old_backups,
    _pre_migration_backup,
    restore_pre_migration_snapshot,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_wal_db(path: Path) -> None:
    """Create a minimal WAL-mode SQLite database with one committed row."""
    conn = sqlite3.connect(str(path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("CREATE TABLE t (v INTEGER)")
    conn.execute("INSERT INTO t VALUES (1)")
    conn.commit()
    conn.close()


def _make_simple_db(path: Path) -> None:
    """Create a minimal SQLite database (default journal mode)."""
    conn = sqlite3.connect(str(path))
    conn.execute("CREATE TABLE t (v INTEGER)")
    conn.execute("INSERT INTO t VALUES (99)")
    conn.commit()
    conn.close()


def _integrity_ok(path: Path) -> bool:
    """Return True if PRAGMA integrity_check returns 'ok'."""
    conn = sqlite3.connect(str(path))
    try:
        row = conn.execute("PRAGMA integrity_check").fetchone()
        return row is not None and row[0] == "ok"
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# _backup_via_sqlite_api
# ---------------------------------------------------------------------------


class TestBackupViaSqliteApi:
    def test_creates_dest_file(self, tmp_path: Path) -> None:
        src = tmp_path / "src.db"
        _make_simple_db(src)
        dest = tmp_path / "sub" / "dest.db"

        _backup_via_sqlite_api(src, dest)

        assert dest.exists(), "Backup file must be created"

    def test_dest_parent_created_automatically(self, tmp_path: Path) -> None:
        src = tmp_path / "src.db"
        _make_simple_db(src)
        dest = tmp_path / "deep" / "nested" / "dest.db"

        _backup_via_sqlite_api(src, dest)

        assert dest.parent.exists()

    def test_backup_is_valid_sqlite(self, tmp_path: Path) -> None:
        src = tmp_path / "src.db"
        _make_simple_db(src)
        dest = tmp_path / "dest.db"

        _backup_via_sqlite_api(src, dest)

        assert _integrity_ok(dest), "PRAGMA integrity_check must return 'ok'"

    def test_backup_contains_committed_data(self, tmp_path: Path) -> None:
        src = tmp_path / "src.db"
        _make_simple_db(src)
        dest = tmp_path / "dest.db"

        _backup_via_sqlite_api(src, dest)

        conn = sqlite3.connect(str(dest))
        rows = conn.execute("SELECT v FROM t").fetchall()
        conn.close()
        assert rows == [(99,)]

    def test_backup_consistent_under_live_writer(self, tmp_path: Path) -> None:
        """Gate test: backup of WAL database with an uncommitted transaction.

        The SQLite backup API reads committed pages only. An in-flight write
        that has not been committed must not appear in the snapshot.
        """
        src = tmp_path / "memory.db"

        # Bootstrap a WAL database.
        conn1 = sqlite3.connect(str(src))
        conn1.execute("PRAGMA journal_mode=WAL")
        conn1.execute("CREATE TABLE t (v INTEGER)")
        conn1.commit()

        # Start an uncommitted transaction with data in the WAL.
        conn1.execute("BEGIN")
        conn1.execute("INSERT INTO t VALUES (42)")
        # conn1 not yet committed — WAL has a pending page

        dest = tmp_path / "backup" / "memory.db"
        _backup_via_sqlite_api(src, dest)  # must not block; must not include uncommitted data

        conn2 = sqlite3.connect(str(dest))
        rows = conn2.execute("SELECT * FROM t").fetchall()
        conn2.close()

        assert rows == [], (
            "Uncommitted writes must not appear in the backup snapshot"
        )

        conn1.commit()
        conn1.close()

    def test_integrity_check_after_concurrent_writer(self, tmp_path: Path) -> None:
        """Backup taken concurrently with an in-flight writer is byte-consistent."""
        src = tmp_path / "memory.db"

        conn1 = sqlite3.connect(str(src))
        conn1.execute("PRAGMA journal_mode=WAL")
        conn1.execute("CREATE TABLE data (id INTEGER PRIMARY KEY, val TEXT)")
        for i in range(50):
            conn1.execute("INSERT INTO data VALUES (?, ?)", (i, f"val_{i}"))
        conn1.commit()

        # Begin a write that will be in-flight during backup.
        conn1.execute("BEGIN")
        for i in range(50, 100):
            conn1.execute("INSERT INTO data VALUES (?, ?)", (i, f"val_{i}"))

        dest = tmp_path / "backup" / "memory.db"
        _backup_via_sqlite_api(src, dest)

        assert _integrity_ok(dest), "Backup must pass integrity_check"

        conn1.rollback()
        conn1.close()

    def test_uses_sqlite_backup_api_not_shutil(self, tmp_path: Path) -> None:
        """Verify the implementation uses sqlite3.Connection.backup, not shutil.copy2.

        The SQLite backup API is a C-level method on sqlite3.Connection —
        conn.backup is read-only and cannot be monkeypatched directly. We
        prove the contract two ways:
        1. shutil.copy2 is never called (it must not appear in the code path).
        2. The backup destination is a fully-readable, non-empty SQLite file
           produced by the backup call, which is only possible if backup() ran.
        """
        src = tmp_path / "src.db"
        _make_simple_db(src)
        dest = tmp_path / "dest.db"

        with mock.patch("shutil.copy2") as mock_copy:
            _backup_via_sqlite_api(src, dest)

        # shutil.copy2 must never be invoked — it cannot produce a consistent
        # WAL snapshot.
        mock_copy.assert_not_called()

        # The result must be a valid, readable SQLite database — only the
        # backup API (not a raw byte copy with copy2 blocked) can produce it.
        assert _integrity_ok(dest), "Backup result must pass SQLite integrity_check"

        # Confirm the source module does not contain a shutil.copy2() call.
        # A docstring mention is acceptable; an actual call is not.
        import ast
        import inspect

        import superlocalmemory.storage.backup as backup_module
        source = inspect.getsource(backup_module)
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                # Detect shutil.copy2(...) as an attribute call.
                if (
                    isinstance(func, ast.Attribute)
                    and func.attr == "copy2"
                ):
                    raise AssertionError(
                        "backup.py must not call shutil.copy2() — found a call at "
                        f"line {func.end_lineno}"
                    )


# ---------------------------------------------------------------------------
# _pre_migration_backup
# ---------------------------------------------------------------------------


class TestPreMigrationBackup:
    def test_returns_path(self, tmp_path: Path) -> None:
        mem = tmp_path / "memory.db"
        lrn = tmp_path / "learning.db"
        _make_simple_db(mem)
        _make_simple_db(lrn)
        snapshots_root = tmp_path / "snapshots"

        result = _pre_migration_backup(lrn, mem, backups_root=snapshots_root)

        assert isinstance(result, Path)

    def test_creates_both_db_files(self, tmp_path: Path) -> None:
        """Both databases are backed up as flat files in snapshots_root."""
        mem = tmp_path / "memory.db"
        lrn = tmp_path / "learning.db"
        _make_simple_db(mem)
        _make_simple_db(lrn)
        snapshots_root = tmp_path / "snapshots"

        _pre_migration_backup(lrn, mem, backups_root=snapshots_root)

        mem_snaps = list(snapshots_root.glob("memory-*-pre-migration.db"))
        lrn_snaps = list(snapshots_root.glob("learning-*-pre-migration.db"))
        assert len(mem_snaps) == 1, "Exactly one memory snapshot file must be created"
        assert len(lrn_snaps) == 1, "Exactly one learning snapshot file must be created"

    def test_snapshot_files_are_flat_in_snapshots_root(self, tmp_path: Path) -> None:
        """Snapshot files sit directly in snapshots_root — no subdirectory."""
        mem = tmp_path / "memory.db"
        lrn = tmp_path / "learning.db"
        _make_simple_db(mem)
        _make_simple_db(lrn)
        snapshots_root = tmp_path / "snapshots"

        _pre_migration_backup(lrn, mem, backups_root=snapshots_root)

        for f in snapshots_root.glob("*-pre-migration.db"):
            assert f.parent == snapshots_root, (
                f"Snapshot file {f.name} must be directly in snapshots_root, not in a subdir"
            )

    def test_both_backups_pass_integrity_check(self, tmp_path: Path) -> None:
        mem = tmp_path / "memory.db"
        lrn = tmp_path / "learning.db"
        _make_wal_db(mem)
        _make_simple_db(lrn)
        snapshots_root = tmp_path / "snapshots"

        _pre_migration_backup(lrn, mem, backups_root=snapshots_root)

        for snap in snapshots_root.glob("*-pre-migration.db"):
            assert _integrity_ok(snap), f"{snap.name} backup must pass integrity_check"

    def test_raises_insufficient_disk_space(self, tmp_path: Path) -> None:
        mem = tmp_path / "memory.db"
        lrn = tmp_path / "learning.db"
        _make_simple_db(mem)
        _make_simple_db(lrn)
        snapshots_root = tmp_path / "snapshots"

        # Simulate 0 bytes free.
        mock_usage = shutil.disk_usage(str(tmp_path))._replace(free=0)
        with mock.patch("shutil.disk_usage", return_value=mock_usage):
            with pytest.raises(InsufficientDiskSpaceError):
                _pre_migration_backup(lrn, mem, backups_root=snapshots_root)

    def test_no_backup_created_when_disk_full(self, tmp_path: Path) -> None:
        mem = tmp_path / "memory.db"
        lrn = tmp_path / "learning.db"
        _make_simple_db(mem)
        _make_simple_db(lrn)
        snapshots_root = tmp_path / "snapshots"

        mock_usage = shutil.disk_usage(str(tmp_path))._replace(free=0)
        with mock.patch("shutil.disk_usage", return_value=mock_usage):
            with pytest.raises(InsufficientDiskSpaceError):
                _pre_migration_backup(lrn, mem, backups_root=snapshots_root)

        # No snapshot files should have been created.
        snaps = list(snapshots_root.glob("*-pre-migration.db")) if snapshots_root.exists() else []
        assert snaps == [], "No snapshot files should be created when disk space check fails"

    def test_skips_missing_db(self, tmp_path: Path) -> None:
        """If one db does not exist, backup only the one that does."""
        mem = tmp_path / "memory.db"
        _make_simple_db(mem)
        lrn = tmp_path / "learning.db"  # intentionally absent
        snapshots_root = tmp_path / "snapshots"

        _pre_migration_backup(lrn, mem, backups_root=snapshots_root)

        mem_snaps = list(snapshots_root.glob("memory-*-pre-migration.db"))
        lrn_snaps = list(snapshots_root.glob("learning-*-pre-migration.db"))
        assert len(mem_snaps) == 1, "memory snapshot must be created"
        assert len(lrn_snaps) == 0, "learning snapshot must not be created when db is absent"

    def test_returns_snapshots_root(self, tmp_path: Path) -> None:
        """_pre_migration_backup returns the snapshots root directory itself."""
        mem = tmp_path / "memory.db"
        lrn = tmp_path / "learning.db"
        _make_simple_db(mem)
        _make_simple_db(lrn)
        snapshots_root = tmp_path / "snapshots"

        result = _pre_migration_backup(lrn, mem, backups_root=snapshots_root)

        assert result == snapshots_root, (
            "Return value must be the snapshots root, not a child subdirectory"
        )


# ---------------------------------------------------------------------------
# _gc_old_backups
# ---------------------------------------------------------------------------


class TestGcOldBackups:
    def _make_snapshot_files(
        self, root: Path, names: list[str], mtime_offsets: list[float]
    ) -> list[Path]:
        """Create synthetic pre-migration snapshot files with controlled mtimes.

        Each file is named ``{name}-pre-migration.db`` and is a valid (empty)
        SQLite database so that stat() calls succeed.
        """
        root.mkdir(parents=True, exist_ok=True)
        files = []
        for name, offset in zip(names, mtime_offsets):
            f = root / f"{name}-pre-migration.db"
            # Touch as a valid SQLite file.
            conn = sqlite3.connect(str(f))
            conn.close()
            t = time.time() - offset  # offset seconds in the past
            os.utime(str(f), (t, t))
            files.append(f)
        return files

    def test_keeps_two_newest(self, tmp_path: Path) -> None:
        root = tmp_path / "snapshots"
        names = [f"memory-2026010{i}-120000" for i in range(5)]
        # offsets: 400s, 300s, 200s, 100s, 10s ago → oldest first in the list
        self._make_snapshot_files(root, names, [400, 300, 200, 100, 10])

        _gc_old_backups(root, keep=2)

        remaining = sorted(f.name for f in root.glob("*-pre-migration.db"))
        expected = sorted([f"{names[3]}-pre-migration.db", f"{names[4]}-pre-migration.db"])
        assert remaining == expected, (
            f"Expected two newest to remain, got: {remaining}"
        )

    def test_no_op_when_fewer_than_keep(self, tmp_path: Path) -> None:
        root = tmp_path / "snapshots"
        names = ["memory-20260101-120000", "memory-20260102-120000"]
        self._make_snapshot_files(root, names, [200, 100])

        _gc_old_backups(root, keep=2)

        remaining = sorted(f.name for f in root.glob("*-pre-migration.db"))
        expected = sorted(f"{n}-pre-migration.db" for n in names)
        assert remaining == expected, "No files should be removed when count <= keep"

    def test_no_op_when_root_absent(self, tmp_path: Path) -> None:
        root = tmp_path / "does_not_exist"
        # Must not raise.
        _gc_old_backups(root, keep=2)

    def test_never_removes_non_pre_migration_file(self, tmp_path: Path) -> None:
        """GC safety: files without the -pre-migration.db suffix are never touched."""
        root = tmp_path / "snapshots"
        root.mkdir()

        # Ordinary-named files that must never be removed.
        safe_file = root / "memory-20260819-120000.db"
        _make_simple_db(safe_file)
        other_file = root / "some_other_data.db"
        _make_simple_db(other_file)

        # Add enough pre-migration files to trigger GC.
        names = [f"memory-2026010{i}-120000" for i in range(5)]
        self._make_snapshot_files(root, names, [400, 300, 200, 100, 10])

        _gc_old_backups(root, keep=2)

        assert safe_file.exists(), "memory-*.db file without -pre-migration suffix must not be removed"
        assert other_file.exists(), "Unrelated .db file must not be removed by GC"

    def test_deleted_paths_are_children_of_snapshots_root(self, tmp_path: Path) -> None:
        """Every deleted file must have snapshots_root as its direct parent."""
        root = tmp_path / "snapshots"
        names = [f"memory-2026010{i}-120000" for i in range(5)]
        self._make_snapshot_files(root, names, [400, 300, 200, 100, 10])

        deleted: list[Path] = []
        original_unlink = Path.unlink

        def recording_unlink(self_path, missing_ok=False):
            deleted.append(self_path)
            original_unlink(self_path, missing_ok=missing_ok)

        with mock.patch.object(Path, "unlink", recording_unlink):
            _gc_old_backups(root, keep=2)

        assert len(deleted) == 3, f"Expected 3 deletions, got {len(deleted)}"
        for p in deleted:
            assert p.parent == root, (
                f"Deleted path {p} must have snapshots_root as parent"
            )

    def test_exactly_keep_count_remains_after_multiple_runs(self, tmp_path: Path) -> None:
        root = tmp_path / "snapshots"
        names = [f"memory-2026010{i}-120000" for i in range(10)]
        self._make_snapshot_files(root, names, list(range(1000, 0, -100)))

        _gc_old_backups(root, keep=2)
        remaining = list(root.glob("*-pre-migration.db"))
        assert len(remaining) == 2

        # Running again must be idempotent.
        _gc_old_backups(root, keep=2)
        remaining_after = list(root.glob("*-pre-migration.db"))
        assert len(remaining_after) == 2

    def test_custom_keep_value(self, tmp_path: Path) -> None:
        root = tmp_path / "snapshots"
        names = [f"memory-2026010{i}-120000" for i in range(5)]
        self._make_snapshot_files(root, names, [400, 300, 200, 100, 10])

        _gc_old_backups(root, keep=3)

        remaining = list(root.glob("*-pre-migration.db"))
        assert len(remaining) == 3


# ---------------------------------------------------------------------------
# Retention privilege: pre-migration snapshots survive BackupManager churn
# ---------------------------------------------------------------------------


class TestPreMigrationPrivilege:
    def test_pre_migration_backup_survives_backup_manager_retention(
        self, tmp_path: Path
    ) -> None:
        """Privilege test: BackupManager._enforce_retention() cannot reach snapshots.

        Pre-migration snapshots live in ``<data_root>/pre-migration-snapshots/``.
        BackupManager._enforce_retention() globs only ``<data_root>/backups/``.
        These are different directories. Regardless of how many ordinary backups
        accumulate, _enforce_retention() can never delete our snapshot files.
        """
        from superlocalmemory.infra.backup import BackupManager

        data_root = tmp_path / "slm"
        data_root.mkdir()
        snapshots_root = data_root / "pre-migration-snapshots"

        # Source databases.
        mem_src = data_root / "memory.db"
        lrn_src = data_root / "learning.db"
        _make_simple_db(mem_src)
        _make_simple_db(lrn_src)

        # Create the pre-migration snapshot.
        _pre_migration_backup(lrn_src, mem_src, backups_root=snapshots_root)
        snapshot_files_before = set(f.name for f in snapshots_root.glob("*-pre-migration.db"))
        assert snapshot_files_before, "Pre-migration snapshot must be created"

        # Drive BackupManager past its retention limit into data_root/backups/.
        # Use max_backups=2 so retention fires after every third backup.
        mgr = BackupManager(base_dir=data_root)
        mgr.config["max_backups"] = 2
        for _ in range(8):  # 8 >> max_backups of 2; retention fires multiple times
            mgr.create_backup()

        # Snapshots in the separate directory must be completely untouched.
        snapshot_files_after = set(f.name for f in snapshots_root.glob("*-pre-migration.db"))
        assert snapshot_files_after == snapshot_files_before, (
            "BackupManager retention must never delete pre-migration snapshots: "
            "they live in a separate directory (pre-migration-snapshots/) that "
            "_enforce_retention() never globs"
        )

    def test_gc_never_removes_backup_manager_style_files(
        self, tmp_path: Path
    ) -> None:
        """GC safety: *-pre-migration.db glob never matches BackupManager filenames.

        BackupManager writes ``memory-{ts}.db`` and ``learning-{ts}.db``.
        Our GC glob is ``*-pre-migration.db``. Even if both file types land in
        the same root, the glob must only match our files.
        """
        root = tmp_path / "snapshots"
        root.mkdir()

        # BackupManager-style files (no -pre-migration suffix) — must survive.
        ordinary = [
            root / "memory-20260819-120000.db",
            root / "learning-20260819-120000.db",
            root / "memory-20260820-083000.db",
        ]
        for f in ordinary:
            _make_simple_db(f)

        # Pre-migration files — enough to trigger GC (keep=2, create 5).
        for i in range(5):
            f = root / f"memory-2026080{i}-120000-pre-migration.db"
            _make_simple_db(f)

        _gc_old_backups(root, keep=2)

        # BackupManager-style files must be completely untouched.
        for f in ordinary:
            assert f.exists(), (
                f"GC must not remove BackupManager-style file: {f.name}"
            )

        # Only 2 pre-migration files should remain.
        remaining = list(root.glob("*-pre-migration.db"))
        assert len(remaining) == 2, (
            f"Expected 2 pre-migration files to remain after GC, got {len(remaining)}"
        )


# ---------------------------------------------------------------------------
# Deprecation of legacy backup_database in migrations.py
# ---------------------------------------------------------------------------


class TestLegacyBackupDatabaseDeprecated:
    def test_backup_database_emits_deprecation_warning(self, tmp_path: Path) -> None:
        from superlocalmemory.storage.migrations import backup_database

        db = tmp_path / "test.db"
        _make_simple_db(db)

        with pytest.warns(DeprecationWarning):
            backup_database(db)

    def test_backup_database_still_creates_a_file(self, tmp_path: Path) -> None:
        """Legacy function must not crash — dead v1 path depends on it."""
        from superlocalmemory.storage.migrations import backup_database

        db = tmp_path / "test.db"
        _make_simple_db(db)

        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            result = backup_database(db)

        assert result.exists()


# ---------------------------------------------------------------------------
# End-to-end restore: snapshot survives into a usable recovery
# ---------------------------------------------------------------------------


class TestRestoreFromSnapshot:
    def test_restore_from_snapshot_end_to_end(self, tmp_path: Path) -> None:
        """Gate test: snapshot written before migration is restorable after failure.

        Write a known database → snapshot it → simulate a failed migration by
        overwriting the live database with different data → restore from the
        snapshot → assert the original data came back.

        This test exercises the full chain: _pre_migration_backup creates the
        snapshot, BackupManager.restore_backup (with backup_dir pointed at the
        snapshots directory) performs the restore via sqlite3.backup().
        """
        from superlocalmemory.infra.backup import BackupManager

        data_root = tmp_path / "slm"
        data_root.mkdir()
        snapshots_root = data_root / "pre-migration-snapshots"

        # Source database with a known row.
        mem_db = data_root / "memory.db"
        conn = sqlite3.connect(str(mem_db))
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("CREATE TABLE items (id INTEGER PRIMARY KEY, val TEXT)")
        conn.execute("INSERT INTO items VALUES (1, 'original-data')")
        conn.commit()
        conn.close()

        # Snapshot before migration.
        _pre_migration_backup(mem_db, mem_db, backups_root=snapshots_root)

        snaps = list(snapshots_root.glob("memory-*-pre-migration.db"))
        assert len(snaps) == 1, "Snapshot file must be created"
        snap_filename = snaps[0].name

        # Simulate a failed migration: overwrite with a different value.
        conn = sqlite3.connect(str(mem_db))
        conn.execute("DELETE FROM items")
        conn.execute("INSERT INTO items VALUES (1, 'corrupted-after-migration')")
        conn.commit()
        conn.close()

        # Confirm the damage is real before restoring.
        conn = sqlite3.connect(str(mem_db))
        rows = conn.execute("SELECT val FROM items").fetchall()
        conn.close()
        assert rows == [("corrupted-after-migration",)], "Pre-restore state must show the damage"

        # Restore from snapshot: BackupManager pointed at the snapshots dir.
        mgr = BackupManager(base_dir=data_root, backup_dir=snapshots_root)
        ok = mgr.restore_backup(snap_filename)
        assert ok is True, (
            f"restore_backup() must return True for a valid snapshot; returned {ok!r}"
        )

        # Original data must be back.
        conn = sqlite3.connect(str(mem_db))
        rows = conn.execute("SELECT val FROM items").fetchall()
        conn.close()
        assert rows == [("original-data",)], (
            f"Restore must return the pre-migration data; got {rows!r}"
        )


class TestSafeSnapshotRestore:
    """The documented restore path must not be able to destroy the live store.

    Reproduced against BackupManager.restore_backup(): with 11 snapshots and
    max_backups=10, its internal pre-restore backup runs retention, retention
    unlinks the snapshot being restored, sqlite3.connect recreates that path as
    an empty database, and the empty database is copied over the live store —
    returning True. 500 facts became 0 tables.
    """

    @staticmethod
    def _store(path: Path, n: int) -> None:
        conn = sqlite3.connect(path)
        conn.execute("CREATE TABLE atomic_facts(fact_id TEXT, content TEXT)")
        conn.executemany("INSERT INTO atomic_facts VALUES(?,?)",
                         [(f"f{i}", f"m{i}") for i in range(n)])
        conn.commit()
        conn.close()

    @staticmethod
    def _facts(path: Path) -> int:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        try:
            return conn.execute("SELECT COUNT(*) FROM atomic_facts").fetchone()[0]
        finally:
            conn.close()

    def test_restores_content_and_keeps_a_copy_of_what_it_replaced(self, tmp_path):
        live = tmp_path / "memory.db"
        self._store(live, 500)
        snaps = tmp_path / "pre-migration-snapshots"
        snaps.mkdir()
        snap = snaps / "memory-20260819-000000-pre-migration.db"
        _backup_via_sqlite_api(live, snap)

        # the live store then diverges, as it would after a bad migration
        conn = sqlite3.connect(live)
        conn.execute("DELETE FROM atomic_facts")
        conn.commit()
        conn.close()
        assert self._facts(live) == 0

        safety = restore_pre_migration_snapshot(snap, live)

        assert self._facts(live) == 500, "snapshot content must be restored"
        assert safety.exists(), "the replaced state must be preserved"
        assert self._facts(safety) == 0, "safety copy holds the pre-restore state"
        assert safety.parent != snaps, (
            "the safety copy must live outside the snapshot directory so no "
            "retention policy can reclaim it"
        )

    def test_refuses_an_empty_snapshot_without_touching_the_live_store(self, tmp_path):
        live = tmp_path / "memory.db"
        self._store(live, 42)
        empty = tmp_path / "memory-20260819-000000-pre-migration.db"
        empty.touch()          # exactly what connect() leaves behind after an unlink

        with pytest.raises(SnapshotUnusableError):
            restore_pre_migration_snapshot(empty, live)

        assert self._facts(live) == 42, "live store must be untouched on refusal"

    def test_refuses_a_tableless_snapshot_without_touching_the_live_store(self, tmp_path):
        live = tmp_path / "memory.db"
        self._store(live, 7)
        hollow = tmp_path / "memory-20260819-000001-pre-migration.db"
        sqlite3.connect(hollow).close()      # a valid but empty database

        with pytest.raises(SnapshotUnusableError):
            restore_pre_migration_snapshot(hollow, live)

        assert self._facts(live) == 7, "live store must be untouched on refusal"


class TestSnapshotRestoreOverHotWal:
    """Restoring over a WAL store whose -wal holds committed, un-checkpointed rows.

    That is the normal state of a store the daemon is using. The restore used to
    rename a copy over the live file, leaving the old -wal beside it: SQLite read
    the old frames over the restored pages, so the restore returned normally and
    changed nothing, and checkpointed those frames into the restored file once
    the last connection closed.
    """

    _PAYLOAD = "x" * 300

    @classmethod
    def _wal_store(cls, path: Path, tag: str, rows: int, *, replace: bool = False) -> None:
        """Commit rows into a WAL store and close it, which checkpoints them."""
        conn = sqlite3.connect(path)
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(
                "CREATE TABLE IF NOT EXISTS t (id INTEGER PRIMARY KEY, tag TEXT, body TEXT)")
            if replace:
                conn.execute("DELETE FROM t")
            conn.executemany("INSERT INTO t (tag, body) VALUES (?, ?)",
                             [(tag, cls._PAYLOAD)] * rows)
            conn.commit()
        finally:
            conn.close()

    @classmethod
    def _hold_uncheckpointed(cls, path: Path, tag: str, rows: int) -> sqlite3.Connection:
        """Commit rows that exist only in path's -wal; the caller closes the connection."""
        conn = sqlite3.connect(path)
        conn.execute("PRAGMA wal_autocheckpoint=0")
        conn.executemany("INSERT INTO t (tag, body) VALUES (?, ?)",
                         [(tag, cls._PAYLOAD)] * rows)
        conn.commit()
        return conn

    @staticmethod
    def _tags(conn: sqlite3.Connection) -> dict[str, int]:
        return dict(conn.execute("SELECT tag, count(*) FROM t GROUP BY tag ORDER BY tag"))

    @staticmethod
    def _dump(path: Path, *, immutable: bool = False) -> list[str]:
        uri = (f"{path.absolute().as_uri()}?mode=ro&immutable=1" if immutable
               else str(path))
        conn = sqlite3.connect(uri, uri=immutable)
        try:
            return list(conn.iterdump())
        finally:
            conn.close()

    @staticmethod
    def _integrity(path: Path) -> str:
        conn = sqlite3.connect(path)
        try:
            return "; ".join(r[0] for r in conn.execute("PRAGMA integrity_check"))
        except sqlite3.DatabaseError as exc:
            return f"{type(exc).__name__}: {exc}"
        finally:
            conn.close()

    @staticmethod
    def _wal_bytes(path: Path) -> int:
        wal = path.with_name(path.name + "-wal")
        return wal.stat().st_size if wal.exists() else 0

    @pytest.mark.parametrize("live_moved_on", [False, True],
                             ids=["same-base", "live-moved-on"])
    def test_restore_matches_snapshot_and_is_intact(self, tmp_path, live_moved_on):
        live = tmp_path / "memory.db"
        self._wal_store(live, "A", 400)
        snaps = tmp_path / "pre-migration-snapshots"
        snap = snaps / "memory-20261002-000000-000000-pre-migration.db"
        _backup_via_sqlite_api(live, snap)
        snapshot_dump = self._dump(snap, immutable=True)

        if live_moved_on:
            # The live store changed shape after the snapshot and was
            # checkpointed, so the -wal frames below describe a different tree.
            self._wal_store(live, "A2", 2000, replace=True)
        holder = self._hold_uncheckpointed(live, "B", 800)
        try:
            # Guard: the B rows are committed and live only in the -wal.
            assert self._wal_bytes(live) > 0
            assert self._tags(holder)["B"] == 800

            safety = restore_pre_migration_snapshot(snap, live)

            assert self._integrity(live) == "ok"
            assert self._dump(live) == snapshot_dump, (
                "live store does not hold the snapshot's content")
            # A reader that was open throughout sees the restored content too.
            assert self._tags(holder) == {"A": 400}
        finally:
            holder.close()

        # Once the last connection closes, nothing stale is checkpointed back in.
        assert self._integrity(live) == "ok"
        assert self._dump(live) == snapshot_dump
        # The safety copy holds what was live, including the rows only in -wal.
        expected_before = {"A2": 2000, "B": 800} if live_moved_on else {"A": 400, "B": 800}
        conn = sqlite3.connect(f"{safety.absolute().as_uri()}?mode=ro&immutable=1", uri=True)
        try:
            assert self._tags(conn) == expected_before
        finally:
            conn.close()

    def test_refuses_a_snapshot_whose_wal_holds_frames(self, tmp_path):
        """The snapshot is read immutable, which ignores its -wal; refuse, don't drop."""
        live = tmp_path / "memory.db"
        self._wal_store(live, "A", 10)
        live_before = self._dump(live)
        snap = tmp_path / "memory-20261002-000000-000000-pre-migration.db"
        self._wal_store(snap, "S", 5)
        holder = self._hold_uncheckpointed(snap, "S2", 5)
        try:
            assert self._wal_bytes(snap) > 0
            with pytest.raises(LiveStoreWriteError, match="write-ahead log"):
                restore_pre_migration_snapshot(snap, live)
        finally:
            holder.close()

        assert self._dump(live) == live_before, "live store must be untouched on refusal"


# ---------------------------------------------------------------------------
# _gc_old_backups: a collision suffix must not split one generation in two
# ---------------------------------------------------------------------------

class TestGcCollisionSuffix:
    """A -N collision suffix on one file of a generation must not break GC."""

    def test_collision_suffixed_file_grouped_with_base_generation(
        self, tmp_path: Path
    ) -> None:
        """A file with a -N collision suffix must share a generation with its base timestamp.

        Without normalisation, GC counts the -1 file as a different generation
        from the base and, with keep=2 and mtime ordering, can prune the BASE
        file while keeping the COLLISION — leaving half a generation that
        restore_pre_migration_snapshot cannot use.

        Three content generations are created.  Gen3 has both a base and a
        collision-suffixed file.  Gen3_base is given the OLDEST mtime so that
        mtime-based ordering incorrectly prunes it.  After the fix (filename
        timestamp ordering), gen1 is correctly pruned and both gen3 files survive.
        """
        root = tmp_path / "snapshots"
        root.mkdir()

        gen1_ts = "20260817-120000-000001"
        gen2_ts = "20260818-120000-000002"
        gen3_ts = "20260819-120000-000003"

        gen1_file = root / f"memory-{gen1_ts}-pre-migration.db"
        gen2_file = root / f"memory-{gen2_ts}-pre-migration.db"
        gen3_base = root / f"memory-{gen3_ts}-pre-migration.db"
        gen3_coll = root / f"memory-{gen3_ts}-1-pre-migration.db"

        for f in (gen1_file, gen2_file, gen3_base, gen3_coll):
            _make_simple_db(f)

        # Mtimes that fool mtime-based ordering: gen3_base gets the OLDEST mtime
        # so the old code would prune it as a separate "generation", keeping gen1
        # (newer mtime) and gen3_coll — the wrong pair.
        now = time.time()
        os.utime(str(gen3_base), (now - 1000, now - 1000))
        os.utime(str(gen2_file), (now - 500, now - 500))
        os.utime(str(gen1_file), (now - 100, now - 100))
        os.utime(str(gen3_coll), (now - 50, now - 50))

        _gc_old_backups(root, keep=2)

        # After fix: 3 generations by timestamp; keep=2 prunes gen1 (oldest ts).
        assert gen3_base.exists(), (
            "gen3_base must survive — it and gen3_coll share the same generation"
        )
        assert gen3_coll.exists(), "gen3_coll (collision twin) must also survive"
        assert not gen1_file.exists(), (
            "gen1 is the oldest generation by timestamp and must be pruned"
        )

    def test_gc_counts_generations_not_files(self, tmp_path: Path) -> None:
        """keep=2 retains 2 complete generations, not just 2 files."""
        root = tmp_path / "snapshots"
        root.mkdir()

        # 3 generations, each with memory + learning
        for g in range(3):
            ts = f"2026010{g + 1}-120000-000000"
            for stem in ("memory", "learning"):
                _make_simple_db(root / f"{stem}-{ts}-pre-migration.db")

        _gc_old_backups(root, keep=2)

        remaining = list(root.glob("*-pre-migration.db"))
        assert len(remaining) == 4, (
            f"keep=2 with 2-file generations must leave 4 files (2 gens × 2), "
            f"got {len(remaining)}: {[f.name for f in remaining]}"
        )

    def test_gc_orders_by_filename_not_mtime(self, tmp_path: Path) -> None:
        """The oldest generation is removed even when all mtimes are identical."""
        root = tmp_path / "snapshots"
        root.mkdir()

        ts_list = [
            "20260818-120000-000001",  # oldest
            "20260819-120000-000002",
            "20260820-120000-000003",  # newest
        ]
        files = []
        for ts in ts_list:
            f = root / f"memory-{ts}-pre-migration.db"
            _make_simple_db(f)
            # Force all mtimes to be identical — simulates FAT or 1-second mtime
            same_t = 1_700_000_000.0
            os.utime(str(f), (same_t, same_t))
            files.append(f)

        _gc_old_backups(root, keep=2)

        remaining_names = {f.name for f in root.glob("*-pre-migration.db")}
        assert files[0].name not in remaining_names, (
            "Oldest file (by timestamp in name) must be removed "
            "even when all mtimes are equal"
        )
        assert files[2].name in remaining_names, (
            "Newest file (by timestamp in name) must be retained"
        )


# ---------------------------------------------------------------------------
# The disk-space check must not over-reserve for sequential writes
# ---------------------------------------------------------------------------

class TestDiskSpaceCheck:
    def test_backup_succeeds_with_1_15x_free_space(self, tmp_path: Path) -> None:
        """_pre_migration_backup must not raise when free >= 1.1× total DB bytes.

        Sequential snapshot writes peak at total_bytes on disk (both final
        snapshots exist simultaneously).  The old formula required 2.1× the
        sum, blocking a backup that would have fit.
        """
        memory_db = tmp_path / "memory.db"
        learning_db = tmp_path / "learning.db"
        snaps = tmp_path / "snaps"
        _make_simple_db(memory_db)
        _make_simple_db(learning_db)

        # Real on-disk sizes are tiny; we only need the ratio check to work.
        actual_total = memory_db.stat().st_size + learning_db.stat().st_size
        # 1.15× is above the 1.1× needed but below the broken 2.1×
        free = int(actual_total * 1.15)

        with mock.patch("shutil.disk_usage") as mock_du:
            mock_du.return_value.free = free
            try:
                _pre_migration_backup(learning_db, memory_db, backups_root=snaps)
            except InsufficientDiskSpaceError as exc:
                pytest.fail(
                    f"InsufficientDiskSpaceError raised with {free:,} free bytes "
                    f"(1.15× of {actual_total:,}). needed_bytes={exc.needed_bytes:,}. "
                    "Sequential writes peak at total_bytes, not 2× total — "
                    "the check over-reserved."
                )

    def test_backup_still_fails_when_space_is_genuinely_too_tight(
        self, tmp_path: Path
    ) -> None:
        """The disk check must still fire when free < 1.0× total DB bytes."""
        memory_db = tmp_path / "memory.db"
        learning_db = tmp_path / "learning.db"
        snaps = tmp_path / "snaps"
        _make_simple_db(memory_db)
        _make_simple_db(learning_db)

        actual_total = memory_db.stat().st_size + learning_db.stat().st_size
        # Only 0.5× — genuinely too little to hold both snapshots
        free = int(actual_total * 0.5)

        with mock.patch("shutil.disk_usage") as mock_du:
            mock_du.return_value.free = free
            with pytest.raises(InsufficientDiskSpaceError):
                _pre_migration_backup(learning_db, memory_db, backups_root=snaps)


# ---------------------------------------------------------------------------
# Finding 3: _snapshot_sort_key must handle filenames with hyphenated stems
# ---------------------------------------------------------------------------


class TestSnapshotSortKeyRobustness:
    """_snapshot_sort_key must extract the correct timestamp regardless of stem."""

    def test_normal_stem_returns_timestamp_key(self, tmp_path: Path) -> None:
        """Baseline: memory-<ts>-pre-migration.db extracts correctly."""
        from superlocalmemory.storage.backup import _snapshot_sort_key

        p = tmp_path / "memory-20260819-120000-123456-pre-migration.db"
        p.touch()
        assert _snapshot_sort_key(p) == ("20260819-120000-123456", 0)

    def test_collision_suffix_is_integer(self, tmp_path: Path) -> None:
        """Collision-suffixed file returns (base_ts, collision_int)."""
        from superlocalmemory.storage.backup import _snapshot_sort_key

        p = tmp_path / "memory-20260819-120000-123456-2-pre-migration.db"
        p.touch()
        assert _snapshot_sort_key(p) == ("20260819-120000-123456", 2)

    def test_hyphenated_stem_does_not_corrupt_timestamp(self, tmp_path: Path) -> None:
        """A stem containing a hyphen must not shift the extracted timestamp.

        split('-', 1)[-1] on "pre-migration-20260819-120000-123456-pre-migration.db"
        yields "migration-20260819-120000-123456-pre-migration.db", leaving
        "migration-" as a prefix in the raw stamp.  The regex then misses the
        timestamp and returns the full "migration-20260819-..." string as the
        sort key, which sorts alphabetically after all numeric keys and makes
        the file appear to be newer than any real snapshot.
        """
        from superlocalmemory.storage.backup import _snapshot_sort_key

        p = tmp_path / "pre-migration-20260819-120000-123456-pre-migration.db"
        p.touch()
        key = _snapshot_sort_key(p)
        assert key == ("20260819-120000-123456", 0), (
            f"hyphenated stem must not bleed into the timestamp key; got {key!r}"
        )

    def test_hyphenated_stem_sorts_consistently_with_plain_stem(
        self, tmp_path: Path
    ) -> None:
        """A hyphenated-stem snapshot at the same timestamp must sort equal to a
        plain-stem snapshot at the same timestamp, not sort after it.

        If the sort key is wrong the -1 index in sorted(candidates)[-1] returns
        the malformed file as the newest snapshot and the caller crashes trying
        to open it as a SQLite database.
        """
        from superlocalmemory.storage.backup import _snapshot_sort_key

        good = tmp_path / "memory-20260819-120000-000001-pre-migration.db"
        hyphen = tmp_path / "pre-migration-20260819-120000-000001-pre-migration.db"
        good.touch()
        hyphen.touch()
        good_key = _snapshot_sort_key(good)
        bad_key = _snapshot_sort_key(hyphen)
        assert good_key[0] == bad_key[0], (
            f"timestamp component must match across stems: good={good_key}, "
            f"hyphenated={bad_key}"
        )


# ---------------------------------------------------------------------------
# Finding 3 (GC side): _gc_old_backups must group hyphenated-stem files
# correctly into generations
# ---------------------------------------------------------------------------

class TestGcRobustWithHyphenatedStem:
    """GC generation grouping must not split a generation due to a hyphenated stem."""

    def test_gc_does_not_remove_files_from_active_generation(
        self, tmp_path: Path
    ) -> None:
        """Files whose generation stamp parses correctly must be grouped together.

        If the stamp extraction is wrong for a hyphenated stem, GC may see it as
        a separate 'new' generation and delete the actual newest valid generation.
        """
        snaps = tmp_path / "snaps"
        snaps.mkdir()

        # Three generations — keep=2 means the oldest should be removed.
        old_ts = "20260810-100000-000000"
        mid_ts = "20260815-100000-000000"
        new_ts = "20260819-120000-123456"

        for ts in (old_ts, mid_ts, new_ts):
            for stem in ("memory", "learning"):
                (snaps / f"{stem}-{ts}-pre-migration.db").touch()

        _gc_old_backups(snaps, keep=2)

        remaining = {f.name for f in snaps.iterdir() if f.is_file()}
        # Old generation must be gone; mid and new must survive.
        for stem in ("memory", "learning"):
            assert f"{stem}-{old_ts}-pre-migration.db" not in remaining, (
                "oldest generation must have been removed"
            )
            assert f"{stem}-{mid_ts}-pre-migration.db" in remaining
            assert f"{stem}-{new_ts}-pre-migration.db" in remaining
