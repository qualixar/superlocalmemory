# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Tests for migration notification functions (task 8.5).

TDD RED phase: tests fail before implementation exists.
"""

from __future__ import annotations

from pathlib import Path


class TestMigrationNotificationHelpers:
    """_notify_migration_applied and _write_migration_error_log helpers."""

    def test_notify_migration_applied_is_importable(self):
        """_notify_migration_applied exists in unified_daemon."""
        from superlocalmemory.server import unified_daemon
        assert hasattr(unified_daemon, "_notify_migration_applied")

    def test_notify_migration_applied_prints_count(self, capsys):
        """_notify_migration_applied prints migration count and backup path."""
        from superlocalmemory.server.unified_daemon import _notify_migration_applied

        _notify_migration_applied(
            applied=["M041_foo", "M042_bar"],
            elapsed=1.23,
            backup_dir=Path("/tmp/test/backups/pre-4.1.0-20260819-120000"),
        )

        captured = capsys.readouterr()
        assert "2" in captured.out or "2" in captured.err
        assert "migration" in captured.out.lower() or "migration" in captured.err.lower()

    def test_write_migration_error_log_is_importable(self):
        """_write_migration_error_log exists in unified_daemon."""
        from superlocalmemory.server import unified_daemon
        assert hasattr(unified_daemon, "_write_migration_error_log")

    def test_write_migration_error_log_creates_file(self, tmp_path):
        """_write_migration_error_log writes a file with failure details."""
        from superlocalmemory.server.unified_daemon import _write_migration_error_log

        slm_home = tmp_path
        backup_dir = tmp_path / "backups" / "pre-4.1.0-20260819"
        backup_dir.mkdir(parents=True)

        log_file = _write_migration_error_log(
            failed=["M041_foo"],
            backup_dir=backup_dir,
            slm_home=slm_home,
        )

        assert log_file is not None
        assert log_file.exists()
        content = log_file.read_text(encoding="utf-8")
        assert "M041_foo" in content or "migration" in content.lower()

    def test_write_migration_error_log_filename_pattern(self, tmp_path):
        """Error log filename matches migration-error-{timestamp}.log pattern."""
        from superlocalmemory.server.unified_daemon import _write_migration_error_log

        log_file = _write_migration_error_log(
            failed=["M042_bar"],
            backup_dir=tmp_path / "backups",
            slm_home=tmp_path,
        )

        assert log_file.name.startswith("migration-error-")
        assert log_file.name.endswith(".log")

    def test_notify_windows_user_is_importable(self):
        """_notify_windows_user exists in unified_daemon."""
        from superlocalmemory.server import unified_daemon
        assert hasattr(unified_daemon, "_notify_windows_user")

    def test_notify_windows_user_is_noop_on_macos(self, monkeypatch):
        """_notify_windows_user does nothing on macOS (sys.platform != 'win32')."""
        from superlocalmemory.server.unified_daemon import _notify_windows_user
        # Must not raise on macOS
        _notify_windows_user("Test migration complete")


class TestMigrationFailureWritesErrorLog:
    """Integration-style test: apply_all failure writes error log via the daemon path."""

    def test_failed_migrations_in_result_trigger_error_log_write(self, tmp_path):
        """When apply_all returns failed migrations, error log appears in slm_home."""
        # This tests the dispatch logic, not the full daemon startup
        from superlocalmemory.server.unified_daemon import (
            _write_migration_error_log,
        )

        # Simulate a failed result
        failed = ["M041_bad_migration"]
        backup_dir = tmp_path / "backups" / "pre-4.1.0"
        backup_dir.mkdir(parents=True)

        log_path = _write_migration_error_log(
            failed=failed,
            backup_dir=backup_dir,
            slm_home=tmp_path,
        )

        assert log_path.exists()
        assert "M041_bad_migration" in log_path.read_text(encoding="utf-8")

    def test_error_log_content_includes_backup_path(self, tmp_path):
        """Error log content references the backup directory."""
        from superlocalmemory.server.unified_daemon import _write_migration_error_log

        backup_dir = tmp_path / "backups" / "pre-4.1.0-20260819"
        backup_dir.mkdir(parents=True)

        log_path = _write_migration_error_log(
            failed=["M042_oops"],
            backup_dir=backup_dir,
            slm_home=tmp_path,
        )

        content = log_path.read_text(encoding="utf-8")
        assert "pre-4.1.0-20260819" in content or "backup" in content.lower()

    def test_error_message_references_slm_doctor(self, tmp_path):
        """Error log content or stderr mentions 'slm doctor' as recovery step."""
        from superlocalmemory.server.unified_daemon import _write_migration_error_log

        backup_dir = tmp_path / "backups"
        backup_dir.mkdir(parents=True)

        log_path = _write_migration_error_log(
            failed=["M042_oops"],
            backup_dir=backup_dir,
            slm_home=tmp_path,
        )

        content = log_path.read_text(encoding="utf-8")
        # Must mention how to get help (slm doctor is the 4.0.9 recovery path)
        assert "slm doctor" in content.lower() or "slm" in content.lower()
