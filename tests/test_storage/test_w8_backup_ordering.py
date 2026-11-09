# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Tests for backup wiring in apply_all (task 8.1).

TDD RED phase: tests fail before the wiring is added.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from unittest import mock

import pytest

from superlocalmemory.storage import migration_runner as mr
from superlocalmemory.storage.backup import (
    InsufficientDiskSpaceError,
    _pre_migration_backup,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _minimal_dbs(tmp_path: Path) -> tuple[Path, Path]:
    """Create bare-minimum learning.db + memory.db that apply_all accepts."""
    learning_db = tmp_path / "learning.db"
    memory_db = tmp_path / "memory.db"
    for db in (learning_db, memory_db):
        conn = sqlite3.connect(str(db))
        conn.execute("PRAGMA journal_mode=WAL")
        conn.close()
    return learning_db, memory_db


# ---------------------------------------------------------------------------
# 1. Backup is created by apply_all (non-dry-run)
# ---------------------------------------------------------------------------

class TestApplyAllCallsBackup:

    def test_backup_dir_present_in_result_details(self, tmp_path):
        """apply_all() result['details']['_backup'] is a non-empty string."""
        learning_db, memory_db = _minimal_dbs(tmp_path)
        backups_root = tmp_path / "backups"

        with mock.patch(
            "superlocalmemory.storage.migration_runner._pre_migration_backup",
            wraps=lambda ld, md, **kw: _pre_migration_backup(ld, md, backups_root=backups_root),
        ):
            result = mr.apply_all(learning_db, memory_db)

        assert "_backup" in result["details"], (
            "apply_all() must store backup path in result['details']['_backup']"
        )
        assert result["details"]["_backup"]  # non-empty string

    def test_backup_directory_actually_exists(self, tmp_path):
        """The backup directory named in result['details']['_backup'] exists on disk."""
        learning_db, memory_db = _minimal_dbs(tmp_path)
        backups_root = tmp_path / "backups"

        with mock.patch(
            "superlocalmemory.storage.migration_runner._pre_migration_backup",
            wraps=lambda ld, md, **kw: _pre_migration_backup(ld, md, backups_root=backups_root),
        ):
            result = mr.apply_all(learning_db, memory_db)

        backup_dir = Path(result["details"]["_backup"])
        assert backup_dir.exists(), f"Backup directory {backup_dir} must exist after apply_all"

    def test_backup_skipped_on_dry_run(self, tmp_path):
        """apply_all(dry_run=True) does NOT call _pre_migration_backup."""
        learning_db, memory_db = _minimal_dbs(tmp_path)

        with mock.patch(
            "superlocalmemory.storage.migration_runner._pre_migration_backup"
        ) as patched:
            mr.apply_all(learning_db, memory_db, dry_run=True)

        patched.assert_not_called()

    def test_dry_run_result_has_no_backup_key(self, tmp_path):
        """apply_all(dry_run=True) result details does NOT contain '_backup'."""
        learning_db, memory_db = _minimal_dbs(tmp_path)
        result = mr.apply_all(learning_db, memory_db, dry_run=True)
        assert "_backup" not in result["details"]


# ---------------------------------------------------------------------------
# 2. Backup precedes the first in_progress row (ordering guarantee)
# ---------------------------------------------------------------------------

class TestBackupPrecedesMigrationLog:

    def test_backup_dir_exists_before_first_in_progress(self, tmp_path):
        """Backup directory exists at the moment the first in_progress row is written."""
        learning_db, memory_db = _minimal_dbs(tmp_path)
        backups_root = tmp_path / "backups"

        call_order: list[str] = []

        original_backup = _pre_migration_backup

        def tracking_backup(ld, md, **kw):
            result = original_backup(ld, md, backups_root=backups_root)
            call_order.append(f"backup:{result}")
            return result

        original_upsert = None
        try:
            from superlocalmemory.storage import _migration_internals as _mi
            original_upsert = _mi._upsert_log

            def tracking_upsert(conn, name, ddl_hash, status):
                if status == "in_progress":
                    call_order.append(f"in_progress:{name}")
                return original_upsert(conn, name, ddl_hash, status)

            with mock.patch(
                "superlocalmemory.storage.migration_runner._pre_migration_backup",
                side_effect=tracking_backup,
            ), mock.patch.object(_mi, "_upsert_log", side_effect=tracking_upsert):
                mr.apply_all(learning_db, memory_db)

        except ImportError:
            pytest.skip("_migration_internals not importable — skip ordering check")

        # Verify backup happened before any in_progress record
        backup_events = [e for e in call_order if e.startswith("backup:")]
        in_progress_events = [e for e in call_order if e.startswith("in_progress:")]

        if in_progress_events:
            assert backup_events, "No backup event recorded before in_progress writes"
            first_backup_idx = call_order.index(backup_events[0])
            first_in_progress_idx = call_order.index(in_progress_events[0])
            assert first_backup_idx < first_in_progress_idx, (
                f"Backup (index {first_backup_idx}) must precede "
                f"first in_progress (index {first_in_progress_idx})"
            )


# ---------------------------------------------------------------------------
# 3. InsufficientDiskSpaceError aborts apply_all cleanly
# ---------------------------------------------------------------------------

class TestInsufficientDiskAborts:

    def test_migration_aborts_when_disk_space_insufficient(self, tmp_path):
        """apply_all() propagates InsufficientDiskSpaceError without running migrations."""
        learning_db, memory_db = _minimal_dbs(tmp_path)

        with mock.patch(
            "superlocalmemory.storage.migration_runner._pre_migration_backup",
            side_effect=InsufficientDiskSpaceError(1_000_000_000, 100),
        ):
            with pytest.raises(InsufficientDiskSpaceError):
                mr.apply_all(learning_db, memory_db)

    def test_no_migration_log_written_on_disk_abort(self, tmp_path):
        """When InsufficientDiskSpaceError fires, migration_log must have no in_progress rows."""
        learning_db, memory_db = _minimal_dbs(tmp_path)

        with mock.patch(
            "superlocalmemory.storage.migration_runner._pre_migration_backup",
            side_effect=InsufficientDiskSpaceError(1_000_000_000, 100),
        ):
            try:
                mr.apply_all(learning_db, memory_db)
            except InsufficientDiskSpaceError:
                pass

        # Neither DB should have a migration_log with in_progress rows
        for db_path in (learning_db, memory_db):
            try:
                conn = sqlite3.connect(str(db_path))
                rows = conn.execute(
                    "SELECT * FROM migration_log WHERE status = 'in_progress'"
                ).fetchall()
                conn.close()
                assert rows == [], f"Found in_progress rows in {db_path} after disk abort"
            except sqlite3.OperationalError:
                pass  # table doesn't exist yet = no writes, correct


# ---------------------------------------------------------------------------
# 4. _gc_old_backups is called after backup (retention)
# ---------------------------------------------------------------------------

class TestGcCalledAfterBackup:

    def test_gc_called_on_successful_backup(self, tmp_path):
        """apply_all() calls _gc_old_backups after a successful backup."""
        learning_db, memory_db = _minimal_dbs(tmp_path)
        backups_root = tmp_path / "backups"

        # The mock must return what production returns: _pre_migration_backup
        # returns the snapshots ROOT, not a child directory. An earlier version
        # of this test returned a child, which made `backup_dir.parent`
        # accidentally equal `backups_root` and hid a real defect — GC was being
        # pointed one level above the snapshots and pruned nothing, so snapshots
        # grew without bound at roughly half a gigabyte each.
        with mock.patch(
            "superlocalmemory.storage.migration_runner._pre_migration_backup",
            return_value=backups_root,
        ) as _pb, mock.patch(
            "superlocalmemory.storage.migration_runner._gc_old_backups"
        ) as gc_mock:
            backups_root.mkdir(parents=True, exist_ok=True)
            mr.apply_all(learning_db, memory_db)

        gc_mock.assert_called_once()
        # The root itself; ``protect`` (the copy just taken) is keyword-only.
        assert gc_mock.call_args.args == (backups_root,)


# ---------------------------------------------------------------------------
# 5. Backup directory is gitignored and npmignored
# ---------------------------------------------------------------------------

class TestBackupDirIgnored:

    def test_snapshot_dirs_ignored_by_git_and_npm(self):
        """Both backup directories must be ignored by git AND npm.

        Snapshots contain the user's memories in full. One escaping into a
        commit or an npm tarball would publish them. The earlier version of
        this test checked only `backups/`, and only in `.gitignore`, with an
        `or "backups"` clause that could not fail.
        """
        root = Path(__file__).parents[2]
        for name in (".gitignore", ".npmignore"):
            path = root / name
            assert path.exists(), f"{name} missing from repo root"
            content = path.read_text(encoding="utf-8")
            for entry in ("backups/", "pre-migration-snapshots/"):
                assert entry in content, f"{entry!r} must be listed in {name}"

    def test_backups_dir_in_npmignore(self):
        """backups/ must appear in .npmignore so backup dirs are not published."""
        npmignore = Path(__file__).parents[2] / ".npmignore"
        assert npmignore.exists(), ".npmignore missing from repo root"
        content = npmignore.read_text(encoding="utf-8")
        assert "backups/" in content or "backups" in content, (
            "'backups/' must be in .npmignore to prevent publishing backup dirs"
        )


# ---------------------------------------------------------------------------
# 6. Regression guard — backup stays beside the databases, never in canonical root
# ---------------------------------------------------------------------------

class TestNoLeakToCanonicalDataRoot:
    """apply_all(dry_run=False) must derive the snapshot directory from the
    database paths, not from canonical_data_root().

    The conftest redirects canonical_data_root() to _TEST_DATA_DIR. These
    tests place databases in a DIFFERENT directory (tmp_path / "dbs") so we
    can assert: backup went to dbs/, not to canonical_data_root().
    """

    def _make_dbs(self, db_dir: Path) -> tuple[Path, Path]:
        db_dir.mkdir(parents=True, exist_ok=True)
        learning_db = db_dir / "learning.db"
        memory_db = db_dir / "memory.db"
        with sqlite3.connect(str(learning_db)) as c:
            c.execute("PRAGMA user_version = 1")
        with sqlite3.connect(str(memory_db)) as c:
            c.execute("PRAGMA user_version = 1")
        return learning_db, memory_db

    def test_backup_lands_beside_databases_not_in_canonical_root(self, tmp_path):
        """Snapshot must go to db_dir/pre-migration-snapshots/, where db_dir
        is memory_db.parent — not to canonical_data_root().

        The conftest sets canonical_data_root() -> _TEST_DATA_DIR.
        Databases are in tmp_path / "dbs", which is a DIFFERENT directory.
        After apply_all, canonical_data_root() / "pre-migration-snapshots"
        must NOT exist (i.e. no backup leaked into the canonical root)."""
        from superlocalmemory.infra.data_root import canonical_data_root

        canonical = canonical_data_root()
        db_dir = tmp_path / "dbs"
        learning_db, memory_db = self._make_dbs(db_dir)

        # Canonical root and db_dir must be different for this test to be valid.
        assert canonical != db_dir, (
            "Test precondition violated: canonical_data_root() == db_dir; "
            "the regression guard requires them to be separate paths"
        )

        mr.apply_all(learning_db, memory_db, dry_run=False)

        leaked = canonical / "pre-migration-snapshots"
        assert not leaked.exists(), (
            f"Backup leaked into canonical_data_root(): {leaked} was created. "
            "backups_root must be derived from memory_db.parent, not from "
            "canonical_data_root()."
        )

    def test_backup_lands_in_db_parent_dir(self, tmp_path):
        """Snapshot files must appear in memory_db.parent / pre-migration-snapshots."""
        db_dir = tmp_path / "dbs"
        learning_db, memory_db = self._make_dbs(db_dir)

        mr.apply_all(learning_db, memory_db, dry_run=False)

        expected = db_dir / "pre-migration-snapshots"
        assert expected.exists(), (
            f"Snapshot directory not found at {expected}. "
            "backups_root must be memory_db.parent / 'pre-migration-snapshots'."
        )
        snapshots = list(expected.glob("*-pre-migration.db"))
        assert len(snapshots) >= 2, (
            f"Expected ≥2 snapshot files in {expected}, found {len(snapshots)}."
        )
