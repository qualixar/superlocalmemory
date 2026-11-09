# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Tests for PidManager — PID file read/write/cleanup with atomic JSON operations.

TDD order 1: PidManager has no dependencies, test first.
Tests: T5, T6, T7, T14, T15.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest


class TestPidManagerWriteRead:
    """T5: PID file JSON roundtrip preserves all fields."""

    def test_pid_file_write_read_roundtrip(self, tmp_pid_file: Path) -> None:
        from superlocalmemory.infra.pid_manager import PidManager

        mgr = PidManager(tmp_pid_file)
        mgr.register(1234, 5678)
        records = mgr.read_all()

        assert len(records) == 1
        assert records[0].pid == 1234
        assert records[0].ppid == 5678
        assert records[0].started_at  # Non-empty ISO timestamp

    def test_register_replaces_stale_same_pid(self, tmp_pid_file: Path) -> None:
        """Re-registering same PID replaces the old entry (no duplicates)."""
        from superlocalmemory.infra.pid_manager import PidManager

        mgr = PidManager(tmp_pid_file)
        mgr.register(1234, 5678)
        mgr.register(1234, 9999)
        records = mgr.read_all()

        assert len(records) == 1
        assert records[0].ppid == 9999

    def test_multiple_pids_registered(self, tmp_pid_file: Path) -> None:
        """Multiple distinct PIDs are all stored."""
        from superlocalmemory.infra.pid_manager import PidManager

        mgr = PidManager(tmp_pid_file)
        mgr.register(100, 200)
        mgr.register(300, 400)
        mgr.register(500, 600)
        records = mgr.read_all()

        assert len(records) == 3
        pids = {r.pid for r in records}
        assert pids == {100, 300, 500}

    def test_read_nonexistent_file(self, tmp_pid_file: Path) -> None:
        """Reading from a nonexistent file returns empty list."""
        from superlocalmemory.infra.pid_manager import PidManager

        mgr = PidManager(tmp_pid_file)
        assert mgr.read_all() == []


class TestPidManagerCleanup:
    """T6: PID removed from file cleanly."""

    def test_pid_file_remove_on_exit(self, tmp_pid_file: Path) -> None:
        from superlocalmemory.infra.pid_manager import PidManager

        mgr = PidManager(tmp_pid_file)
        mgr.register(1234, 5678)
        assert mgr.remove(1234) is True
        assert mgr.read_all() == []

    def test_remove_nonexistent_pid(self, tmp_pid_file: Path) -> None:
        """Removing a PID not in the file returns False."""
        from superlocalmemory.infra.pid_manager import PidManager

        mgr = PidManager(tmp_pid_file)
        mgr.register(1234, 5678)
        assert mgr.remove(9999) is False
        assert len(mgr.read_all()) == 1


class TestPidManagerCorruption:
    """T7: Corrupt PID file is deleted and recreated, not crashed on."""

    def test_pid_file_corruption_recovery(self, tmp_pid_file: Path) -> None:
        tmp_pid_file.write_text("{invalid json !!!", encoding="utf-8")
        from superlocalmemory.infra.pid_manager import PidManager

        mgr = PidManager(tmp_pid_file)
        records = mgr.read_all()
        assert records == []

        # Should work after recovery
        mgr.register(9999, 1111)
        assert len(mgr.read_all()) == 1

    def test_malformed_structure(self, tmp_pid_file: Path) -> None:
        """PID file with valid JSON but wrong structure returns empty."""
        tmp_pid_file.write_text(json.dumps({"wrong_key": []}), encoding="utf-8")
        from superlocalmemory.infra.pid_manager import PidManager

        mgr = PidManager(tmp_pid_file)
        assert mgr.read_all() == []


class TestCleanupDeadPids:
    """T14: cleanup_dead() removes stale entries."""

    def test_cleanup_dead_pids(self, tmp_pid_file: Path) -> None:
        from superlocalmemory.infra.pid_manager import PidManager

        mgr = PidManager(tmp_pid_file)
        my_pid = os.getpid()

        # Register current PID (alive) and two fake PIDs (dead)
        mgr.register(my_pid, os.getppid())
        mgr.register(99998, 88888)
        mgr.register(99999, 88889)

        removed = mgr.cleanup_dead()

        assert removed == 2
        records = mgr.read_all()
        pids = [r.pid for r in records]
        assert my_pid in pids
        assert 99998 not in pids
        assert 99999 not in pids


class TestAtomicWrite:
    """T15: Atomic write survives disk failure."""

    def test_atomic_write_on_crash(self, tmp_pid_file: Path) -> None:
        from superlocalmemory.infra.pid_manager import PidManager

        mgr = PidManager(tmp_pid_file)
        mgr.register(1234, 5678)

        # Read current state as baseline
        original_content = tmp_pid_file.read_text(encoding="utf-8")

        # Mock os.replace to simulate crash during atomic write
        with patch("os.replace", side_effect=OSError("Simulated disk failure")):
            mgr.register(9999, 8888)  # This write should fail silently

        # Original file should still be valid and readable
        restored = mgr.read_all()
        assert len(restored) >= 1
        assert any(r.pid == 1234 for r in restored)
        assert tmp_pid_file.read_text(encoding="utf-8") == original_content


class TestPidRecord:
    """PidRecord dataclass tests."""

    def test_to_dict_from_dict_roundtrip(self) -> None:
        from superlocalmemory.infra.pid_manager import PidRecord

        record = PidRecord(pid=1234, ppid=5678, started_at="2026-03-30T14:25:01")
        d = record.to_dict()
        restored = PidRecord.from_dict(d)

        assert restored.pid == 1234
        assert restored.ppid == 5678
        assert restored.started_at == "2026-03-30T14:25:01"

    def test_frozen_immutability(self) -> None:
        from superlocalmemory.infra.pid_manager import PidRecord

        record = PidRecord(pid=1, ppid=2, started_at="now")
        with pytest.raises(AttributeError):
            record.pid = 999  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Coverage gap tests: read_all OSError on unlink (lines 100-101)
# ---------------------------------------------------------------------------
class TestReadAllUnlinkError:
    """Cover OSError during unlink of corrupt PID file."""

    def test_corrupt_file_unlink_fails(self, tmp_pid_file: Path) -> None:
        """OSError on unlink of corrupt file is handled gracefully."""
        tmp_pid_file.write_text("{invalid json !!!", encoding="utf-8")
        from superlocalmemory.infra.pid_manager import PidManager

        mgr = PidManager(tmp_pid_file)

        with patch.object(Path, "unlink", side_effect=OSError("Permission denied")):
            records = mgr.read_all()

        assert records == []


# ---------------------------------------------------------------------------
# Coverage gap tests: read_all OSError on read (lines 104-106)
# ---------------------------------------------------------------------------
class TestReadAllOSError:
    """Cover OSError when reading the PID file."""

    def test_os_error_on_read(self, tmp_pid_file: Path) -> None:
        """OSError on reading PID file returns empty list."""
        tmp_pid_file.write_text('{"pids": []}', encoding="utf-8")
        from superlocalmemory.infra.pid_manager import PidManager

        mgr = PidManager(tmp_pid_file)

        with patch.object(Path, "read_text", side_effect=OSError("I/O error")):
            records = mgr.read_all()

        assert records == []


# ---------------------------------------------------------------------------
# Coverage gap tests: _write_all temp file cleanup (lines 141-142)
# ---------------------------------------------------------------------------
class TestWriteAllTempFileCleanup:
    """Cover the finally block that cleans up temp files on write failure."""

    def test_temp_file_cleaned_on_write_failure(self, tmp_pid_file: Path) -> None:
        """Temp file is cleaned up when os.replace fails."""
        from superlocalmemory.infra.pid_manager import PidManager

        mgr = PidManager(tmp_pid_file)
        mgr.register(1234, 5678)  # Initial write succeeds

        # Now mock os.replace to fail -- temp file should be cleaned up
        with patch("os.replace", side_effect=OSError("Disk full")):
            mgr.register(9999, 8888)  # Should fail, temp file cleaned

        # Verify no stray temp files left behind
        temp_files = list(tmp_pid_file.parent.glob("slm-pids-*.tmp"))
        assert len(temp_files) == 0

    def test_temp_file_cleanup_unlink_fails(self, tmp_pid_file: Path) -> None:
        """OSError in finally cleanup of temp file is handled gracefully."""
        from superlocalmemory.infra.pid_manager import PidManager

        mgr = PidManager(tmp_pid_file)


        def fail_replace(src, dst):
            raise OSError("Disk full")

        # We need replace to fail AND unlink to fail
        with patch("os.replace", side_effect=fail_replace):
            with patch.object(Path, "unlink", side_effect=OSError("Cannot delete")):
                mgr.register(9999, 8888)  # Should not raise


# ---------------------------------------------------------------------------
# Coverage gap tests: cleanup_dead PermissionError (lines 187-188)
# ---------------------------------------------------------------------------
class TestCleanupDeadPermissionError:
    """Cover PermissionError in cleanup_dead keeping the record."""

    def test_cleanup_dead_permission_error_keeps_record(
        self, tmp_pid_file: Path
    ) -> None:
        """PermissionError on kill check keeps the record alive."""
        from superlocalmemory.infra.pid_manager import PidManager

        mgr = PidManager(tmp_pid_file)
        mgr.register(55555, 11111)

        original_kill = os.kill

        def mock_kill(pid, sig):
            if pid == 55555 and sig == 0:
                raise PermissionError("Operation not permitted")
            return original_kill(pid, sig)

        with patch("os.kill", side_effect=mock_kill):
            removed = mgr.cleanup_dead()

        assert removed == 0
        records = mgr.read_all()
        pids = [r.pid for r in records]
        assert 55555 in pids
