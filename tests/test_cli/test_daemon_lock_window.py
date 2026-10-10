# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""4.1.22 polish: the start-lock window has no evidence of its own.

Right after another process wins ``daemon.lock`` and before it writes a
descriptor, neither ``this_process_is_spawning()`` (a per-process flag) nor
``starting_descriptor()`` (nothing written yet) sees anything. Before this
fix:

  * ``ensure_daemon()``'s lock-contention branch fell into a flat 60 s wait
    (``_wait_for_daemon(timeout=60)``), ignoring ``SLM_DAEMON_START_WAIT_S``
    and able to outlive a host's own ~60 s tool-call timeout.
  * ``daemon_diagnosis.describe()`` reported ``no_daemon`` ("no daemon is
    registered for this data root") while a daemon genuinely was starting.

These tests hold the real lock file from a SEPARATE file descriptor in the
same test process: BSD-style ``flock`` blocks per-fd even within one process
(see the v3.4.42 restart self-deadlock fix in this same module), so this is
a faithful in-process stand-in for "another process holds the lock."
"""

from __future__ import annotations

import sys
import time
from contextlib import contextmanager
from unittest.mock import patch

import pytest

pytestmark = pytest.mark.skipif(
    sys.platform == "win32", reason="flock-based lock simulation is POSIX-only",
)

# Imported after the skip mark: Windows has no fcntl, and a module-level import
# failed collection there, which stopped every Windows test shard before a
# single test ran.
fcntl = pytest.importorskip(
    "fcntl", reason="flock-based lock simulation is POSIX-only",
)


@contextmanager
def _held_by_another_fd(lock_path):
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = open(lock_path, "w", encoding="utf-8")
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        fd.close()


class TestStartLockIsHeld:
    def test_false_when_the_lock_file_does_not_exist(self, tmp_path):
        from superlocalmemory.cli import daemon as _daemon

        with patch.object(_daemon, "_LOCK_FILE", tmp_path / "daemon.lock"):
            assert _daemon.start_lock_is_held() is False

    def test_false_when_nobody_holds_it(self, tmp_path):
        from superlocalmemory.cli import daemon as _daemon

        lock_path = tmp_path / "daemon.lock"
        lock_path.write_text("", encoding="utf-8")
        with patch.object(_daemon, "_LOCK_FILE", lock_path):
            assert _daemon.start_lock_is_held() is False

    def test_true_while_another_fd_holds_it(self, tmp_path):
        from superlocalmemory.cli import daemon as _daemon

        lock_path = tmp_path / "daemon.lock"
        with patch.object(_daemon, "_LOCK_FILE", lock_path):
            with _held_by_another_fd(lock_path):
                assert _daemon.start_lock_is_held() is True
            # released: the probe must see that too, not latch "held" forever
            assert _daemon.start_lock_is_held() is False


class TestStartInProgressSeesTheLock:
    def test_start_in_progress_true_when_lock_held_with_no_descriptor(self, tmp_path):
        from superlocalmemory.cli import daemon as _daemon
        from superlocalmemory.cli import daemon_startup as _startup

        lock_path = tmp_path / "daemon.lock"
        with patch.object(_daemon, "_LOCK_FILE", lock_path), \
             patch.object(_startup, "starting_descriptor", return_value=None):
            with _held_by_another_fd(lock_path):
                assert _startup.lock_is_held_by_another_process() is True
                assert _startup.start_in_progress() is True
            assert _startup.lock_is_held_by_another_process() is False

    def test_never_raises_when_the_probe_itself_errors(self, monkeypatch):
        from superlocalmemory.cli import daemon as _daemon
        from superlocalmemory.cli import daemon_startup as _startup

        monkeypatch.setattr(
            _daemon, "start_lock_is_held",
            lambda: (_ for _ in ()).throw(RuntimeError("boom")),
        )
        assert _startup.lock_is_held_by_another_process() is False


class TestWaitForStartingDaemonHonorsTheLock:
    def test_waits_instead_of_returning_none_at_once(self, tmp_path, monkeypatch):
        """Without the fix this returned None immediately (no descriptor, this
        process is not the spawner) even though a start was under way."""
        from superlocalmemory.cli import daemon as _daemon
        from superlocalmemory.cli import daemon_startup as _startup

        lock_path = tmp_path / "daemon.lock"
        monkeypatch.setattr(_startup, "_expired_at", {})
        with patch.object(_daemon, "_LOCK_FILE", lock_path), \
             patch.object(_daemon, "read_descriptor", return_value=None):
            with _held_by_another_fd(lock_path):
                started = time.monotonic()
                result = _startup.wait_for_starting_daemon(seconds=0.4)
                elapsed = time.monotonic() - started

        assert result is None  # the holder never produced a matching daemon
        assert elapsed >= 0.35  # it actually waited the budget, not 0s

    def test_still_returns_none_instantly_when_nothing_is_starting(self, tmp_path):
        from superlocalmemory.cli import daemon as _daemon
        from superlocalmemory.cli import daemon_startup as _startup

        lock_path = tmp_path / "daemon.lock"  # never created, never held
        with patch.object(_daemon, "_LOCK_FILE", lock_path), \
             patch.object(_daemon, "read_descriptor", return_value=None):
            started = time.monotonic()
            result = _startup.wait_for_starting_daemon(seconds=5.0)
            elapsed = time.monotonic() - started

        assert result is None
        assert elapsed < 1.0  # no evidence at all -> no wait


class TestEnsureDaemonLockContentionIsBounded:
    def test_bounded_by_the_start_wait_budget_not_a_flat_60s(self, tmp_path, monkeypatch):
        """4.1.22 bug: this used to call _wait_for_daemon(timeout=60). With the
        lock held by someone else and no daemon ever appearing, ensure_daemon
        must return within the configured budget, nowhere near 60 s."""
        from superlocalmemory.cli import daemon as _daemon

        lock_path = tmp_path / "daemon.lock"
        monkeypatch.setenv("SLM_DAEMON_START_WAIT_S", "0.4")
        # The test suite's isolation plugin blocks ensure_daemon() from doing
        # anything by default (SLM_TEST_ISOLATION=1); opt this test back in
        # so it actually reaches the lock-contention branch under test.
        monkeypatch.setenv("SLM_TEST_ALLOW_DAEMON_SPAWN", "1")
        with patch.object(_daemon, "_LOCK_FILE", lock_path), \
             patch.object(_daemon, "is_daemon_running", return_value=False), \
             patch.object(_daemon, "read_descriptor", return_value=None):
            with _held_by_another_fd(lock_path):
                started = time.monotonic()
                result = _daemon.ensure_daemon()
                elapsed = time.monotonic() - started

        assert result is False
        assert elapsed < 10.0  # nowhere near the old flat 60 s
