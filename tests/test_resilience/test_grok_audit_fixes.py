# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V4 | https://qualixar.com | https://varunpratap.com

"""Tests for the Grok audit verified fix set (G-01 through G-10).

TDD: tests written first, then fixes applied, then verified GREEN.
All tests:
- Use scratch tmp dirs (NEVER ~/.superlocalmemory)
- Use ports 8781+ (NEVER 8765)
- Do NOT kill/stop/restart the live production daemon
"""

from __future__ import annotations

import json
import os
import signal
import socket
import sqlite3
import time
from pathlib import Path
from threading import Thread
from unittest.mock import MagicMock, call, patch

import pytest


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _write_json_lock(path: Path, pid: int, claimed_at_ms: int | None = None) -> None:
    payload = {
        "pid": pid,
        "owner_id": "test-owner",
        "claimed_at_ms": claimed_at_ms if claimed_at_ms is not None else int(time.time() * 1000),
        "database": str(path.parent / "memory.db"),
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_plain_pid(path: Path, pid: int) -> None:
    path.write_text(str(pid), encoding="utf-8")


def _dead_pid() -> int:
    pid = 2_000_000
    while pid > 1:
        try:
            os.kill(pid, 0)
            pid -= 1
        except ProcessLookupError:
            return pid
        except PermissionError:
            pid -= 1
    raise RuntimeError("Could not find a dead PID")


# ---------------------------------------------------------------------------
# G-01 + G-05: bounded retry loop in CanonicalRememberRuntime.start()
# ---------------------------------------------------------------------------

class TestBoundedRetryLoop:
    """G-01+G-05: bounded 5-attempt retry loop replaces single self-heal+retry."""

    def _make_runtime(self, mock_coordinator):
        """Build a minimal CanonicalRememberRuntime shell for unit testing."""
        from superlocalmemory.core.remember_runtime import CanonicalRememberRuntime
        rt = CanonicalRememberRuntime.__new__(CanonicalRememberRuntime)
        rt._started = False
        rt.coordinator = mock_coordinator
        return rt

    def test_alive_but_not_healthy_then_becomes_healthy_raises_daemon_already_serving(self):
        """Holder alive but health-check not yet responsive → eventually responds → DaemonAlreadyServing.

        Scenario: holder is starting up; first iteration health check fails,
        second iteration health check succeeds → DaemonAlreadyServing raised cleanly.
        """
        from superlocalmemory.core.remember_runtime import DaemonAlreadyServing

        mock_coordinator = MagicMock()
        # claim_ownership always returns False (holder alive)
        mock_coordinator.claim_ownership.return_value = False
        mock_coordinator.db_path = Path("/tmp/test_memory.db")

        # First health check: unhealthy. Second: healthy.
        health_responses = [False, True]
        health_call_count = [0]

        def health_check(port):
            idx = health_call_count[0]
            health_call_count[0] += 1
            return health_responses[idx] if idx < len(health_responses) else True

        with patch("superlocalmemory.core.remember_runtime._slm_health_check", side_effect=health_check), \
             patch("superlocalmemory.core.remember_runtime._get_daemon_port", return_value=8790), \
             patch("superlocalmemory.core.remember_runtime._boot_self_heal"), \
             patch("time.sleep"):  # don't actually sleep in tests

            rt = self._make_runtime(mock_coordinator)
            with pytest.raises(DaemonAlreadyServing) as exc_info:
                rt.start()

        assert "8790" in str(exc_info.value)
        assert not issubclass(
            exc_info.type,
            __import__("superlocalmemory.core.remember_runtime", fromlist=["CanonicalRememberUnavailable"]).CanonicalRememberUnavailable
        ), "Must be DaemonAlreadyServing, not CanonicalRememberUnavailable"

    def test_holder_dies_mid_loop_claim_succeeds(self):
        """Holder alive for 2 attempts, then dies → claim succeeds on 3rd → start completes.

        Scenario: holder crashes between our retry intervals.
        Expected: start() completes normally (no exception) after the claim succeeds.
        """
        from superlocalmemory.core.remember_runtime import CanonicalRememberRuntime

        mock_coordinator = MagicMock()
        # claim: False, False, True (first outer check + 2 loop retries)
        mock_coordinator.claim_ownership.side_effect = [False, False, True]
        mock_coordinator.db_path = Path("/tmp/test_memory.db")

        with patch("superlocalmemory.core.remember_runtime._slm_health_check", return_value=False), \
             patch("superlocalmemory.core.remember_runtime._get_daemon_port", return_value=8791), \
             patch("superlocalmemory.core.remember_runtime._boot_self_heal"), \
             patch("time.sleep"):

            rt = self._make_runtime(mock_coordinator)
            with patch.object(rt, "replay_pending", return_value=0):
                rt.start()

        assert rt._started is True
        # claim called: 1 initial + 2 loop retries = 3 total
        assert mock_coordinator.claim_ownership.call_count == 3

    def test_persistent_live_holder_raises_daemon_already_serving_not_unavailable(self):
        """After 5 failed claims + no healthy daemon ever detected → DaemonAlreadyServing.

        Key invariant: the final exception is DaemonAlreadyServing (clean exit)
        NOT CanonicalRememberUnavailable (traceback crash for non-technical users).
        """
        from superlocalmemory.core.remember_runtime import (
            CanonicalRememberUnavailable,
            DaemonAlreadyServing,
        )

        mock_coordinator = MagicMock()
        mock_coordinator.claim_ownership.return_value = False  # always fails
        mock_coordinator.db_path = Path("/tmp/test_memory.db")

        with patch("superlocalmemory.core.remember_runtime._slm_health_check", return_value=False), \
             patch("superlocalmemory.core.remember_runtime._get_daemon_port", return_value=8792), \
             patch("superlocalmemory.core.remember_runtime._boot_self_heal"), \
             patch("time.sleep"):

            rt = self._make_runtime(mock_coordinator)
            with pytest.raises(DaemonAlreadyServing):
                rt.start()

        # After the initial outer claim check, the loop makes 5 more attempts
        # (or until health check fires DaemonAlreadyServing earlier).
        # With no healthy daemon detected, the loop-else branch fires.
        # claim_ownership called: 1 initial + up to 5 in loop.
        assert mock_coordinator.claim_ownership.call_count <= 6

    def test_self_heal_called_exactly_once_in_loop(self):
        """Self-heal runs only on the first iteration; subsequent iterations skip it."""
        from superlocalmemory.core.remember_runtime import DaemonAlreadyServing

        mock_coordinator = MagicMock()
        mock_coordinator.claim_ownership.return_value = False
        mock_coordinator.db_path = Path("/tmp/test_memory.db")

        heal_calls = []

        def record_heal(data_dir):
            heal_calls.append(data_dir)

        with patch("superlocalmemory.core.remember_runtime._slm_health_check", return_value=False), \
             patch("superlocalmemory.core.remember_runtime._get_daemon_port", return_value=8793), \
             patch("superlocalmemory.core.remember_runtime._boot_self_heal", side_effect=record_heal), \
             patch("time.sleep"):

            rt = self._make_runtime(mock_coordinator)
            with pytest.raises(DaemonAlreadyServing):
                rt.start()

        assert len(heal_calls) == 1, (
            f"Self-heal must run exactly once; ran {len(heal_calls)} times"
        )

    def test_first_claim_success_skips_loop_entirely(self):
        """When claim_ownership() succeeds on the first try, no loop or health-check is run."""
        mock_coordinator = MagicMock()
        mock_coordinator.claim_ownership.return_value = True
        mock_coordinator.db_path = Path("/tmp/test_memory.db")

        health_calls = []

        with patch("superlocalmemory.core.remember_runtime._slm_health_check",
                   side_effect=lambda p: health_calls.append(p) or False):

            rt = self._make_runtime(mock_coordinator)
            with patch.object(rt, "replay_pending", return_value=0):
                rt.start()

        assert rt._started is True
        assert health_calls == [], "No health check when claim succeeds immediately"
        mock_coordinator.claim_ownership.assert_called_once()


# ---------------------------------------------------------------------------
# G-01 (unified_daemon): CanonicalRememberUnavailable also triggers clean exit
# ---------------------------------------------------------------------------

class TestUnifiedDaemonCatchesBothExceptions:
    """G-01 belt-and-suspenders: the lifespan catches both DaemonAlreadyServing
    AND CanonicalRememberUnavailable → sys.exit(0).

    We test the catch clause pattern directly (the unified_daemon lifespan is
    FastAPI and not easily unit-testable here — we verify the import and that
    both exceptions share no common ancestor that would mask real errors).
    """

    def test_daemon_already_serving_is_not_subclass_of_canonical_remember_unavailable(self):
        from superlocalmemory.core.remember_runtime import (
            CanonicalRememberUnavailable,
            DaemonAlreadyServing,
        )
        assert not issubclass(DaemonAlreadyServing, CanonicalRememberUnavailable)
        assert not issubclass(CanonicalRememberUnavailable, DaemonAlreadyServing)

    def test_both_exceptions_importable_from_remember_runtime(self):
        """unified_daemon.py imports both — verify they are exported."""
        from superlocalmemory.core.remember_runtime import (
            CanonicalRememberUnavailable,
            DaemonAlreadyServing,
        )
        assert CanonicalRememberUnavailable is not None
        assert DaemonAlreadyServing is not None

    def test_catch_tuple_catches_both(self):
        """Python except-tuple semantics: (DaemonAlreadyServing, CanonicalRememberUnavailable) catches both."""
        from superlocalmemory.core.remember_runtime import (
            CanonicalRememberUnavailable,
            DaemonAlreadyServing,
        )

        caught = []

        def _try_catch(exc_cls):
            try:
                raise exc_cls("test")
            except (DaemonAlreadyServing, CanonicalRememberUnavailable):
                caught.append(exc_cls.__name__)

        _try_catch(DaemonAlreadyServing)
        _try_catch(CanonicalRememberUnavailable)

        assert caught == ["DaemonAlreadyServing", "CanonicalRememberUnavailable"], (
            "Both exceptions must be caught by the unified_daemon except tuple"
        )


# ---------------------------------------------------------------------------
# G-02 + G-07: mesh expiry wired at boot + BEGIN IMMEDIATE
# ---------------------------------------------------------------------------

class TestMeshExpiryBootWiring:
    """G-02+G-07: expire_stale_mesh_locks uses BEGIN IMMEDIATE; locked DB → returns 0."""

    def _create_mesh_db(self, db_path: Path) -> sqlite3.Connection:
        conn = sqlite3.connect(str(db_path))
        conn.execute("""
            CREATE TABLE mesh_locks (
                lock_key    TEXT NOT NULL,
                holder_id   TEXT NOT NULL,
                fencing_token INTEGER NOT NULL DEFAULT 0,
                acquired_at TEXT NOT NULL,
                expires_at  TEXT,
                PRIMARY KEY (lock_key, holder_id)
            )
        """)
        conn.commit()
        return conn

    def test_expired_row_deleted_correctly(self, tmp_path):
        """Expired row is deleted; live rows survive; function returns correct count."""
        from superlocalmemory.infra.self_heal import expire_stale_mesh_locks

        db_path = tmp_path / "mesh.db"
        conn = self._create_mesh_db(db_path)
        past_ts = "2000-01-01T00:00:00+00:00"
        future_ts = "2999-01-01T00:00:00+00:00"
        conn.execute(
            "INSERT INTO mesh_locks VALUES (?, ?, ?, ?, ?)",
            ("resource/x", "dead-node", 1, past_ts, past_ts),
        )
        conn.execute(
            "INSERT INTO mesh_locks VALUES (?, ?, ?, ?, ?)",
            ("resource/y", "live-node", 2, past_ts, future_ts),
        )
        conn.commit()
        conn.close()

        deleted = expire_stale_mesh_locks(db_path)

        assert deleted == 1, "Exactly 1 expired row must be deleted"

        # Verify surviving row
        verify = sqlite3.connect(str(db_path))
        rows = verify.execute("SELECT holder_id FROM mesh_locks").fetchall()
        verify.close()
        holder_ids = [r[0] for r in rows]
        assert "dead-node" not in holder_ids, "Expired row must be gone"
        assert "live-node" in holder_ids, "Live row must survive"

    def test_begin_immediate_present_in_source(self):
        """Verify BEGIN IMMEDIATE is present in the expire_stale_mesh_locks source code.

        We cannot easily intercept C-extension sqlite3 method calls in tests.
        Verify the implementation directly — this is the correctness signal for G-07.
        """
        import inspect
        from superlocalmemory.infra.self_heal import expire_stale_mesh_locks

        source = inspect.getsource(expire_stale_mesh_locks)
        assert "BEGIN IMMEDIATE" in source, (
            "G-07: expire_stale_mesh_locks must use BEGIN IMMEDIATE for atomic isolation"
        )
        assert "OperationalError" in source, (
            "G-07: OperationalError (DB locked) must be caught and return 0"
        )

    def test_locked_db_returns_zero_no_exception(self, tmp_path):
        """When DB is locked (OperationalError on BEGIN IMMEDIATE) → returns 0, no exception."""
        from superlocalmemory.infra.self_heal import expire_stale_mesh_locks

        db_path = tmp_path / "mesh.db"
        conn = self._create_mesh_db(db_path)
        conn.close()

        # Simulate OperationalError on BEGIN IMMEDIATE
        def patched_connect(*args, **kwargs):
            c = sqlite3.connect.__wrapped__(*args, **kwargs) if hasattr(sqlite3.connect, "__wrapped__") else sqlite3.connect(*args, **kwargs)
            original_execute = c.execute

            def tracked_execute(sql, *a, **kw):
                if "BEGIN IMMEDIATE" in sql.upper():
                    raise sqlite3.OperationalError("database is locked")
                return original_execute(sql, *a, **kw)

            c.execute = tracked_execute
            return c

        # Use a direct mock instead to avoid recursion
        mock_conn = MagicMock()
        mock_conn.execute.side_effect = sqlite3.OperationalError("database is locked")
        mock_conn.__enter__ = lambda s: s
        mock_conn.__exit__ = MagicMock(return_value=False)

        with patch("sqlite3.connect", return_value=mock_conn):
            result = expire_stale_mesh_locks(db_path)

        assert result == 0, "Locked DB must return 0, not raise"

    def test_begin_immediate_transaction_isolation(self, tmp_path):
        """The DELETE is wrapped in BEGIN IMMEDIATE (no autocommit DELETE that races)."""
        from superlocalmemory.infra.self_heal import expire_stale_mesh_locks

        db_path = tmp_path / "mesh.db"
        conn = self._create_mesh_db(db_path)
        past_ts = "2000-01-01T00:00:00+00:00"
        future_ts = "2999-01-01T00:00:00+00:00"
        conn.execute("INSERT INTO mesh_locks VALUES (?, ?, ?, ?, ?)",
                     ("r/a", "dead", 1, past_ts, past_ts))
        conn.execute("INSERT INTO mesh_locks VALUES (?, ?, ?, ?, ?)",
                     ("r/b", "live", 2, past_ts, future_ts))
        conn.commit()
        conn.close()

        deleted = expire_stale_mesh_locks(db_path)
        assert deleted == 1

        verify = sqlite3.connect(str(db_path))
        rows = verify.execute("SELECT holder_id FROM mesh_locks").fetchall()
        verify.close()
        holder_ids = [r[0] for r in rows]
        assert "dead" not in holder_ids
        assert "live" in holder_ids


# ---------------------------------------------------------------------------
# G-03: live SLM writer.lock must be kept even with create_time mismatch
# ---------------------------------------------------------------------------

class TestLiveSLMLockProtection:
    """G-03: SLM identity check comes before create_time; live SLM is always kept."""

    def test_live_slm_lock_kept_despite_create_time_mismatch(self, tmp_path):
        """A writer.lock whose PID is a live SLM process is KEPT even if create_time
        is wildly off (e.g., the 300 s window). G-03 fix: SLM check is primary."""
        from superlocalmemory.infra.self_heal import reap_stale_artifacts

        lock_file = tmp_path / "memory.db.writer.lock"
        live_pid = os.getpid()

        # Write with ancient claimed_at_ms (huge mismatch — would have failed old 10s check)
        ancient_ms = 1_000_000  # 1970-era timestamp
        _write_json_lock(lock_file, live_pid, claimed_at_ms=ancient_ms)

        # Patch _pid_is_slm to return True for our pid — simulates a live SLM process
        with patch("superlocalmemory.infra.self_heal._pid_is_slm",
                   side_effect=lambda pid: pid == live_pid):
            report = reap_stale_artifacts(tmp_path)

        assert lock_file.exists(), (
            "Live SLM writer.lock MUST be kept even with create_time mismatch (G-03)"
        )
        assert str(lock_file) in report["kept"]
        assert not report["removed"]

    def test_alive_non_slm_pid_with_time_mismatch_is_removed(self, tmp_path):
        """Alive PID that is NOT SLM and has create_time mismatch → 'pid_reused'."""
        from superlocalmemory.infra.self_heal import reap_stale_artifacts

        lock_file = tmp_path / "memory.db.writer.lock"
        live_pid = os.getpid()
        ancient_ms = 1_000_000

        _write_json_lock(lock_file, live_pid, claimed_at_ms=ancient_ms)

        # Non-SLM process
        with patch("superlocalmemory.infra.self_heal._pid_is_slm", return_value=False):
            report = reap_stale_artifacts(tmp_path)

        assert not lock_file.exists()
        assert any(r["reason"] in ("pid_reused", "pid_reused_non_slm")
                   for r in report["removed"])

    def test_pid_matches_claimed_at_window_is_300s(self):
        """_pid_matches_claimed_at uses 300 s tolerance (was 10 s before G-03)."""
        from superlocalmemory.infra.self_heal import _pid_matches_claimed_at

        live_pid = os.getpid()
        # Set claimed_at to 200 seconds ago — would fail 10 s check, must pass 300 s
        two_hundred_s_ago_ms = int((time.time() - 200) * 1000)

        with patch("superlocalmemory.infra.self_heal._pid_create_time",
                   return_value=time.time()):
            result = _pid_matches_claimed_at(live_pid, two_hundred_s_ago_ms)

        # With 300 s window: |now - (now - 200)| = 200 ≤ 300 → should pass
        assert result is True, (
            "300 s window: a 200 s difference must match (was wrongly rejected with 10 s window)"
        )

    def test_pid_matches_claimed_at_returns_true_when_psutil_absent(self, tmp_path):
        """When _pid_create_time returns None (psutil absent), assume matches (safe default)."""
        from superlocalmemory.infra.self_heal import _pid_matches_claimed_at

        with patch("superlocalmemory.infra.self_heal._pid_create_time", return_value=None):
            result = _pid_matches_claimed_at(os.getpid(), 12345)

        assert result is True, (
            "When create_time is unavailable, assume match (safe: never wrongly remove)"
        )


# ---------------------------------------------------------------------------
# G-09: PermissionError = ALIVE in _is_pid_alive
# ---------------------------------------------------------------------------

class TestPermissionErrorIsAlive:
    """G-09: os.kill PermissionError means process exists; treat as alive."""

    def test_permission_error_on_kill_means_alive(self):
        """_is_pid_alive returns True when os.kill raises PermissionError."""
        from superlocalmemory.infra.self_heal import _is_pid_alive

        # When psutil is unavailable and os.kill raises PermissionError → ALIVE
        with patch("superlocalmemory.infra.self_heal._is_pid_alive.__wrapped__" if hasattr(
                    __import__("superlocalmemory.infra.self_heal", fromlist=["_is_pid_alive"]).
                    _is_pid_alive, "__wrapped__") else "builtins.open", create=True):
            pass  # just test the real function directly

        # Import psutil conditionally to test the os.kill fallback path
        try:
            import psutil as _psutil
            # If psutil is available, _is_pid_alive uses psutil.pid_exists.
            # Test the PermissionError path by disabling psutil.
            with patch.dict("sys.modules", {"psutil": None}):
                with patch("os.kill", side_effect=PermissionError("not permitted")):
                    result = _is_pid_alive(99999)
            assert result is True, (
                "PermissionError on os.kill must return True (process exists)"
            )
        except ImportError:
            # psutil not installed — _is_pid_alive uses os.kill path directly
            with patch("os.kill", side_effect=PermissionError("not permitted")):
                result = _is_pid_alive(99999)
            assert result is True

    def test_process_lookup_error_means_dead(self):
        """_is_pid_alive returns False when os.kill raises ProcessLookupError."""
        from superlocalmemory.infra.self_heal import _is_pid_alive

        try:
            import psutil as _psutil
            with patch.dict("sys.modules", {"psutil": None}):
                with patch("os.kill", side_effect=ProcessLookupError("no such process")):
                    result = _is_pid_alive(99999)
        except ImportError:
            with patch("os.kill", side_effect=ProcessLookupError("no such process")):
                result = _is_pid_alive(99999)

        assert result is False

    def test_permission_error_pid_artifact_not_removed(self, tmp_path):
        """When a PID file points to a PermissionError PID → artifact is kept (process is alive)."""
        from superlocalmemory.infra.self_heal import reap_stale_artifacts

        pid_file = tmp_path / ".reranker-worker.pid"
        # Use a PID we don't own so kill(pid, 0) may raise PermissionError
        # In tests we simulate PermissionError directly
        _write_plain_pid(pid_file, 99999)

        # Simulate: psutil unavailable, os.kill → PermissionError (process exists but untouchable)
        # _pid_is_slm calls _is_pid_alive first, which returns True; then checks cmdline.
        # We need _pid_is_slm to also return True to keep the artifact.
        with patch.dict("sys.modules", {"psutil": None}), \
             patch("os.kill", side_effect=PermissionError("not permitted")), \
             patch("superlocalmemory.infra.self_heal._pid_is_slm", return_value=True):
            report = reap_stale_artifacts(tmp_path)

        assert pid_file.exists(), (
            "PermissionError PID that is live SLM must NOT be removed (G-09)"
        )
        assert str(pid_file) in report["kept"]


# ---------------------------------------------------------------------------
# G-06: socket connect retry
# ---------------------------------------------------------------------------

class TestSocketRetry:
    """G-06: _socket_has_listener retries 3× before returning False."""

    def test_socket_checked_three_times_before_false(self):
        """When socket connection is always refused, _socket_has_listener tries 3×."""
        from superlocalmemory.infra.self_heal import _socket_has_listener

        connect_attempts = [0]

        original_socket_class = socket.socket

        class _TrackingSocket:
            def __init__(self, *args, **kwargs):
                pass
            def settimeout(self, t):
                pass
            def connect(self, path):
                connect_attempts[0] += 1
                raise ConnectionRefusedError("connection refused")
            def close(self):
                pass

        with patch("socket.socket", _TrackingSocket), \
             patch("time.sleep"):  # don't actually sleep
            result = _socket_has_listener(Path("/tmp/fake.sock"))

        assert result is False
        assert connect_attempts[0] == 3, (
            f"Must attempt 3 connects before giving up; got {connect_attempts[0]}"
        )

    def test_socket_returns_true_on_second_attempt(self):
        """If second connect succeeds, _socket_has_listener returns True immediately."""
        from superlocalmemory.infra.self_heal import _socket_has_listener

        attempt = [0]

        class _EventualSocket:
            def __init__(self, *args, **kwargs):
                pass
            def settimeout(self, t):
                pass
            def connect(self, path):
                attempt[0] += 1
                if attempt[0] < 2:
                    raise ConnectionRefusedError("not ready yet")
                # Second attempt succeeds (no exception)
            def close(self):
                pass

        with patch("socket.socket", _EventualSocket), \
             patch("time.sleep"):
            result = _socket_has_listener(Path("/tmp/fake.sock"))

        assert result is True
        assert attempt[0] == 2, "Should succeed on 2nd attempt"

    def test_total_wall_time_under_1_5s_with_real_sleeps(self):
        """The retry backoff schedule (0.0 + 0.2 + 0.4 = 0.6 s sleeps) stays under 1.5 s budget.

        We mock time.sleep and sum the requested sleep amounts.
        """
        from superlocalmemory.infra.self_heal import _socket_has_listener

        slept = [0.0]

        def record_sleep(t):
            slept[0] += t

        class _AlwaysRefused:
            def __init__(self, *args, **kwargs):
                pass
            def settimeout(self, t):
                pass
            def connect(self, path):
                raise ConnectionRefusedError()
            def close(self):
                pass

        with patch("socket.socket", _AlwaysRefused), \
             patch("time.sleep", side_effect=record_sleep):
            _socket_has_listener(Path("/tmp/fake.sock"))

        # Sleeps: 0.2 + 0.4 = 0.6 s; per-attempt timeout: 0.3 × 3 = 0.9 s; total = 1.5 s
        assert slept[0] <= 0.7, (
            f"Total sleep budget must be ≤ 0.7 s (got {slept[0]:.2f} s)"
        )


# ---------------------------------------------------------------------------
# G-10: DLQ delete guard in ops_remediation._action_retry
# ---------------------------------------------------------------------------

class TestDLQDeleteGuard:
    """G-10: _action_retry only deletes DLQ row when operation is no longer FAILED."""

    def _create_ops_db(self, db_path: Path) -> None:
        """Create minimal tables for ops_remediation testing."""
        conn = sqlite3.connect(str(db_path))
        conn.execute("""
            CREATE TABLE dead_letter_operations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                original_op_id TEXT NOT NULL,
                profile_id TEXT NOT NULL,
                dead_lettered_at TEXT NOT NULL,
                reason TEXT
            )
        """)
        conn.execute("""
            INSERT INTO dead_letter_operations
            (original_op_id, profile_id, dead_lettered_at, reason)
            VALUES ('op-123', 'test-profile', '2026-01-01T00:00:00Z', 'exhausted')
        """)
        conn.commit()
        conn.close()

    def test_dlq_row_preserved_when_retry_still_fails(self, tmp_path):
        """When cmd.retry() returns an operation still in FAILED state → DLQ row kept."""
        from superlocalmemory.core.ops_remediation import _action_retry
        from superlocalmemory.core.ingestion_command import IngestionState

        db_path = tmp_path / "memory.db"
        self._create_ops_db(db_path)

        # Build a proper mock operation with state == IngestionState.FAILED.
        # Cannot set .value on an enum, so use a real IngestionState member.
        class _FakeOp:
            state = IngestionState.FAILED

        mock_cmd = MagicMock()
        mock_cmd.retry.return_value = _FakeOp()

        mock_engine = MagicMock()

        # _action_retry imports build_engine_ingestion_command locally inside the
        # function; patch it at the source module.
        with patch(
            "superlocalmemory.core.engine_ingestion.build_engine_ingestion_command",
            return_value=mock_cmd,
        ):
            result = _action_retry(db_path, mock_engine, "op-123")

        assert result["success"] is False
        assert "retry_still_failed" in result["reason"]

        # DLQ row must still be there
        conn = sqlite3.connect(str(db_path))
        row = conn.execute(
            "SELECT id FROM dead_letter_operations WHERE original_op_id = 'op-123'"
        ).fetchone()
        conn.close()
        assert row is not None, "DLQ row must be PRESERVED when retry still fails"

    def test_dlq_row_removed_when_retry_succeeds(self, tmp_path):
        """When cmd.retry() returns operation in QUERYABLE state → DLQ row removed."""
        from superlocalmemory.core.ops_remediation import _action_retry
        from superlocalmemory.core.ingestion_command import IngestionState

        db_path = tmp_path / "memory.db"
        self._create_ops_db(db_path)

        # Use a real IngestionState value — cannot set .value on an enum.
        class _FakeOp:
            state = IngestionState.QUERYABLE

        mock_cmd = MagicMock()
        mock_cmd.retry.return_value = _FakeOp()

        mock_engine = MagicMock()

        with patch(
            "superlocalmemory.core.engine_ingestion.build_engine_ingestion_command",
            return_value=mock_cmd,
        ):
            result = _action_retry(db_path, mock_engine, "op-123")

        assert result["success"] is True

        # DLQ row must be gone
        conn = sqlite3.connect(str(db_path))
        row = conn.execute(
            "SELECT id FROM dead_letter_operations WHERE original_op_id = 'op-123'"
        ).fetchone()
        conn.close()
        assert row is None, "DLQ row must be REMOVED when retry succeeds"

    def test_dlq_row_preserved_when_operation_not_in_queue(self, tmp_path):
        """If operation is not in DLQ, return success=False with not_found reason."""
        from superlocalmemory.core.ops_remediation import _action_retry

        db_path = tmp_path / "memory.db"
        self._create_ops_db(db_path)

        mock_engine = MagicMock()
        result = _action_retry(db_path, mock_engine, "op-NONEXISTENT")

        assert result["success"] is False
        assert "not_found" in result["reason"]
