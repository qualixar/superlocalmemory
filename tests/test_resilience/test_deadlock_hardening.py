# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V4 | https://qualixar.com | https://varunpratap.com

"""No-Deadlock Hardening Tests — H1–H4 (TDD, RED → GREEN).

Six required tests from the No-Deadlock Hardening Plan:

1. kill-9 daemon mid-write → restart recovers, no manual lock cleanup
2. double-spawn race → exactly one serves; the other exits gracefully (not crash)
3. stale .reranker-worker.pid / .embedding.lock from dead owner → reaped on boot
4. simulated reboot: all pid files present but every PID dead → clean single-daemon recovery
5. team mode: node holding mesh lock is killed → its lease expires; another node acquires;
   fencing rejects dead node's stale token
6. PID-REUSE SAFETY: stale pid file whose PID was reused by an UNRELATED live process →
   must NOT be treated as ours (no false adoption, no wrongful kill)

All tests:
- Use scratch tmp dirs (NEVER ~/.superlocalmemory)
- Use ports 8781+ (NEVER 8765)
- No pytest-xdist, no pytest-timeout flags
- Do NOT kill/stop/restart the live production daemon on port 8765
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
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_json_lock(path: Path, pid: int, claimed_at_ms: int | None = None) -> None:
    """Write a *.writer.lock JSON metadata file as WriteCoordinator does."""
    payload = {
        "pid": pid,
        "owner_id": "test-owner",
        "claimed_at_ms": claimed_at_ms if claimed_at_ms is not None else int(time.time() * 1000),
        "database": str(path.parent / "memory.db"),
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_plain_pid(path: Path, pid: int) -> None:
    """Write a plain-text PID file."""
    path.write_text(str(pid), encoding="utf-8")


def _dead_pid() -> int:
    """Return a PID that is provably NOT alive on this machine."""
    pid = 2_000_000
    while pid > 1:
        try:
            os.kill(pid, 0)
            pid -= 1
        except ProcessLookupError:
            return pid
        except PermissionError:
            pid -= 1
    raise RuntimeError("Could not find a dead PID — something is wrong with the OS")


def _alive_unrelated_pid() -> int:
    """Return a PID that is alive but NOT an SLM process.

    Uses the current test process itself — guaranteed alive, guaranteed not SLM.
    """
    return os.getpid()


# ---------------------------------------------------------------------------
# T1: kill-9 → restart recovers without manual lock cleanup
# ---------------------------------------------------------------------------

class TestKill9Recovery:
    """T1 — kill -9 the daemon mid-write → restart must recover, no manual cleanup."""

    def test_stale_json_lock_from_dead_owner_cleaned_on_restart(self, tmp_path):
        """A *.writer.lock with a dead owner PID is removed by reap_stale_artifacts.

        This simulates the kill-9 scenario:
        - A writer.lock JSON file with a dead PID exists (left by a killed daemon).
        - The portalocker OS advisory lock is already gone (auto-released on death).
        - reap_stale_artifacts must remove the metadata file so the next daemon
          can claim the writer cleanly.
        """
        from superlocalmemory.infra.self_heal import reap_stale_artifacts

        lock_file = tmp_path / "memory.db.writer.lock"
        dead = _dead_pid()
        _write_json_lock(lock_file, dead)

        assert lock_file.exists(), "Precondition: lock file must be present"

        report = reap_stale_artifacts(tmp_path)

        assert not lock_file.exists(), (
            "kill-9 recovery: stale writer.lock from dead owner must be removed"
        )
        assert len(report["removed"]) == 1
        assert report["removed"][0]["reason"] == "dead_owner_pid"
        assert not report["errors"]

    def test_reap_is_idempotent(self, tmp_path):
        """Running reap twice must not fail or produce spurious errors."""
        from superlocalmemory.infra.self_heal import reap_stale_artifacts

        lock_file = tmp_path / "memory.db.writer.lock"
        _write_json_lock(lock_file, _dead_pid())

        report1 = reap_stale_artifacts(tmp_path)
        report2 = reap_stale_artifacts(tmp_path)  # file already gone

        assert len(report1["removed"]) == 1
        assert report2["removed"] == []  # idempotent
        assert not report2["errors"]

    def test_live_owner_lock_not_touched(self, tmp_path):
        """A writer.lock whose PID is a live SLM-looking process is left alone."""
        from superlocalmemory.infra.self_heal import reap_stale_artifacts, _pid_create_time

        lock_file = tmp_path / "memory.db.writer.lock"
        live_pid = os.getpid()

        # claimed_at_ms MUST match the actual process create_time within the
        # 10-second tolerance.  Using time.time() here would be wrong — the
        # test process started minutes (or hours) ago, not right now.
        create_time = _pid_create_time(live_pid)
        if create_time is not None:
            claimed_at_ms = int(create_time * 1000)
        else:
            # psutil unavailable — skip create_time check by passing None
            claimed_at_ms = None
        _write_json_lock(lock_file, live_pid, claimed_at_ms=claimed_at_ms)

        with patch(
            "superlocalmemory.infra.self_heal._pid_is_slm",
            side_effect=lambda pid: pid == live_pid,
        ):
            report = reap_stale_artifacts(tmp_path)

        assert lock_file.exists(), "Live owner's lock must NOT be removed"
        assert str(lock_file) in report["kept"]
        assert not report["removed"]


# ---------------------------------------------------------------------------
# T2: double-spawn race → one serves, the other exits gracefully
# ---------------------------------------------------------------------------

class TestDoubleSpawnRace:
    """T2 — two daemon instances start ~simultaneously; exactly one serves."""

    def test_daemon_already_serving_raised_when_healthy_daemon_present(self, tmp_path):
        """H2: claim_ownership() → False + healthy health-check → DaemonAlreadyServing.

        Simulates: a healthy daemon is serving; a second process tries to start.
        Expected: DaemonAlreadyServing is raised (caught by lifespan → sys.exit(0)).
        NEVER: CanonicalRememberUnavailable (scary crash).
        """
        from superlocalmemory.core.remember_runtime import (
            DaemonAlreadyServing,
            _get_daemon_port,
            _slm_health_check,
        )

        # The claim returns False (another daemon holds the lock)
        mock_coordinator = MagicMock()
        mock_coordinator.claim_ownership.return_value = False
        mock_coordinator.db_path = tmp_path / "memory.db"

        # A healthy daemon is responding on the port
        with patch(
            "superlocalmemory.core.remember_runtime._slm_health_check",
            return_value=True,
        ):
            with patch(
                "superlocalmemory.core.remember_runtime._get_daemon_port",
                return_value=8781,
            ):
                from superlocalmemory.core.remember_runtime import CanonicalRememberRuntime

                runtime = CanonicalRememberRuntime.__new__(CanonicalRememberRuntime)
                runtime._started = False
                runtime.coordinator = mock_coordinator

                with pytest.raises(DaemonAlreadyServing) as exc_info:
                    runtime.start()

        assert "8781" in str(exc_info.value), (
            "DaemonAlreadyServing message must include the port"
        )
        # The bounded retry loop (G-01) calls claim_ownership once outside the
        # loop, then once at the start of each loop iteration before the health
        # check. With a healthy daemon, the health check fires on the first loop
        # iteration → claim called twice before DaemonAlreadyServing is raised.
        assert mock_coordinator.claim_ownership.call_count >= 1, (
            "claim_ownership must have been called at least once"
        )

    def test_no_crash_on_unhealthy_holder_then_self_heal_retry(self, tmp_path):
        """H2/G-01: claim fails → no healthy daemon → self-heal runs once → claim succeeds.

        Bounded retry loop (G-01): the loop calls claim_ownership at the START of
        each iteration BEFORE the health check and self-heal.  To reach the
        self-heal branch on iteration 0, the loop's first claim must also fail.

        side_effect schedule:
          call 1 (outer guard): False → enter loop
          call 2 (loop iter 0): False → health=False → self-heal (first iter)
          call 3 (loop iter 1): True → break, proceed
        """
        from superlocalmemory.core.remember_runtime import CanonicalRememberRuntime

        mock_coordinator = MagicMock()
        # Three calls: outer=False, loop-iter0=False, loop-iter1=True
        mock_coordinator.claim_ownership.side_effect = [False, False, True]
        mock_coordinator.db_path = tmp_path / "memory.db"

        self_heal_calls: list[Path] = []

        def _record_self_heal(data_dir: Path) -> None:
            self_heal_calls.append(data_dir)

        with patch(
            "superlocalmemory.core.remember_runtime._slm_health_check",
            return_value=False,
        ), patch(
            "superlocalmemory.core.remember_runtime._get_daemon_port",
            return_value=8782,
        ), patch(
            "superlocalmemory.core.remember_runtime._boot_self_heal",
            side_effect=_record_self_heal,
        ), patch("time.sleep"):  # don't actually sleep 1s between retries
            runtime = CanonicalRememberRuntime.__new__(CanonicalRememberRuntime)
            runtime._started = False
            runtime.coordinator = mock_coordinator

            mock_coordinator.register_handler = MagicMock()
            mock_coordinator.start = MagicMock()
            with patch.object(runtime, "replay_pending", return_value=0):
                runtime.start()

        assert mock_coordinator.claim_ownership.call_count == 3, (
            "With bounded loop: outer(False) + iter0(False) + iter1(True) = 3 calls"
        )
        assert len(self_heal_calls) == 1, "Must call _boot_self_heal exactly once"
        assert runtime._started is True

    def test_daemon_already_serving_is_not_canonical_remember_unavailable(self):
        """DaemonAlreadyServing is a distinct exception — not CanonicalRememberUnavailable."""
        from superlocalmemory.core.remember_runtime import (
            CanonicalRememberUnavailable,
            DaemonAlreadyServing,
        )

        assert not issubclass(DaemonAlreadyServing, CanonicalRememberUnavailable), (
            "DaemonAlreadyServing must be a separate exception so the lifespan"
            " can catch it without masking other errors"
        )
        assert issubclass(DaemonAlreadyServing, RuntimeError)


# ---------------------------------------------------------------------------
# T3: stale .reranker-worker.pid / .embedding.lock from dead owner → reaped
# ---------------------------------------------------------------------------

class TestStalePlainPidFiles:
    """T3 — stale plain-pid/lock files from a dead owner are reaped on boot."""

    @pytest.mark.parametrize("filename", [
        ".reranker-worker.pid",
        ".embedding.lock",
        ".config.json.lock",
        "daemon.pid",
    ])
    def test_dead_owner_plain_pid_file_is_removed(self, tmp_path, filename):
        """A plain-text PID file pointing to a dead PID is removed by self-heal."""
        from superlocalmemory.infra.self_heal import reap_stale_artifacts

        path = tmp_path / filename
        _write_plain_pid(path, _dead_pid())

        report = reap_stale_artifacts(tmp_path)

        assert not path.exists(), f"{filename}: file with dead PID must be removed"
        removed_names = [Path(r["path"]).name for r in report["removed"]]
        assert filename in removed_names
        assert any(r["reason"] == "dead_owner_pid" for r in report["removed"])

    def test_fresh_worker_can_spawn_after_stale_files_removed(self, tmp_path):
        """After dead-PID files are reaped, a new worker can write a fresh PID file."""
        from superlocalmemory.infra.self_heal import reap_stale_artifacts

        reranker_pid = tmp_path / ".reranker-worker.pid"
        embedding_lock = tmp_path / ".embedding.lock"

        _write_plain_pid(reranker_pid, _dead_pid())
        _write_plain_pid(embedding_lock, _dead_pid())

        report = reap_stale_artifacts(tmp_path)
        assert not reranker_pid.exists()
        assert not embedding_lock.exists()

        # Simulate a fresh worker writing its own PID
        fresh_pid = os.getpid()
        reranker_pid.write_text(str(fresh_pid), encoding="utf-8")

        assert reranker_pid.read_text(encoding="utf-8") == str(fresh_pid), (
            "Fresh worker can write its PID after stale file is cleared"
        )

    def test_stale_socket_with_no_listener_removed(self):
        """A socket file with no active listener is removed by self-heal.

        Note: Unix socket paths have a hard OS limit (~108 bytes on macOS /
        Linux). pytest's tmp_path is often too long. We create our own short
        directory under the system temp root to stay within the limit.
        """
        import shutil
        import tempfile
        from superlocalmemory.infra.self_heal import reap_stale_artifacts

        # Use a short parent to keep the full socket path under the OS limit
        short_dir = Path(tempfile.mkdtemp(prefix="slm_t3_"))
        try:
            sock_path = short_dir / "hook_daemon.sock"
            # Create a bound (but NOT listening) Unix socket file
            s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                s.bind(str(sock_path))
                # Intentionally do NOT call s.listen() — connection will fail
            finally:
                s.close()

            assert sock_path.exists(), "Precondition: socket file must exist"

            report = reap_stale_artifacts(short_dir)

            assert not sock_path.exists(), (
                "Socket file with no listener must be removed"
            )
            removed_names = [Path(r["path"]).name for r in report["removed"]]
            assert "hook_daemon.sock" in removed_names
        finally:
            shutil.rmtree(str(short_dir), ignore_errors=True)


# ---------------------------------------------------------------------------
# T4: simulated reboot — all PIDs dead → clean single-daemon recovery
# ---------------------------------------------------------------------------

class TestSimulatedRebootRecovery:
    """T4 — all pid files present but all PIDs dead → clean single-daemon recovery."""

    def test_full_reboot_scenario_all_files_cleared(self, tmp_path):
        """After a reboot, all SLM artifacts from dead PIDs are removed in one pass.

        Simulates: machine rebooted, PID namespace recycled, all old PIDs are dead.
        Expected: reap_stale_artifacts removes every artifact in one idempotent call.
        """
        from superlocalmemory.infra.self_heal import reap_stale_artifacts

        dead = _dead_pid()

        # Create all artifact types with dead PIDs (as they would be post-reboot)
        writer_lock = tmp_path / "memory.db.writer.lock"
        _write_json_lock(writer_lock, dead)

        for name in (".reranker-worker.pid", ".embedding.lock", ".config.json.lock", "daemon.pid"):
            _write_plain_pid(tmp_path / name, dead)

        files_before = list(tmp_path.iterdir())
        assert len(files_before) == 5, f"Precondition: 5 artifact files; got {len(files_before)}"

        report = reap_stale_artifacts(tmp_path)

        assert len(report["removed"]) == 5, (
            f"All 5 dead-PID artifacts must be removed; got {report['removed']}"
        )
        assert not report["kept"]
        assert not report["errors"]

        # Data dir is now clear — a fresh daemon can claim the writer
        remaining = [p for p in tmp_path.iterdir()]
        assert remaining == [], (
            f"Data dir must be empty after full reboot recovery; got {remaining}"
        )

    def test_reap_empty_dir_is_safe(self, tmp_path):
        """reap_stale_artifacts on an empty directory returns an empty report."""
        from superlocalmemory.infra.self_heal import reap_stale_artifacts

        report = reap_stale_artifacts(tmp_path)
        assert report == {"removed": [], "kept": [], "errors": []}

    def test_reap_nonexistent_dir_is_safe(self, tmp_path):
        """reap_stale_artifacts on a non-existent directory returns an empty report."""
        from superlocalmemory.infra.self_heal import reap_stale_artifacts

        phantom = tmp_path / "no_such_dir"
        report = reap_stale_artifacts(phantom)
        assert report == {"removed": [], "kept": [], "errors": []}


# ---------------------------------------------------------------------------
# T5: team mode — mesh lock TTL expiry + fencing rejection of dead node
# ---------------------------------------------------------------------------

class TestMeshLockExpiry:
    """T5 — team mode: dead node's mesh lock expires; live node acquires; fencing rejects."""

    def _create_mesh_db(self, db_path: Path) -> sqlite3.Connection:
        """Create a minimal mesh_locks SQLite table matching the production schema."""
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

    def test_expired_mesh_lock_deleted_by_expire_stale_mesh_locks(self, tmp_path):
        """H4: expire_stale_mesh_locks deletes rows where expires_at <= now."""
        from superlocalmemory.infra.self_heal import expire_stale_mesh_locks

        db_path = tmp_path / "mesh.db"
        conn = self._create_mesh_db(db_path)

        # Row 1: expired 1 hour ago (should be deleted)
        past_ts = "2000-01-01T00:00:00+00:00"
        conn.execute(
            "INSERT INTO mesh_locks VALUES (?, ?, ?, ?, ?)",
            ("resource/x", "dead-node", 1, past_ts, past_ts),
        )
        # Row 2: sentinel _NEVER_EXPIRES (must NOT be deleted)
        conn.execute(
            "INSERT INTO mesh_locks VALUES (?, ?, ?, ?, ?)",
            ("resource/y", "live-node", 2, past_ts, "9999-12-31T23:59:59Z"),
        )
        # Row 3: expires in the future (must NOT be deleted)
        future_ts = "2999-01-01T00:00:00+00:00"
        conn.execute(
            "INSERT INTO mesh_locks VALUES (?, ?, ?, ?, ?)",
            ("resource/z", "live-node-2", 3, past_ts, future_ts),
        )
        conn.commit()
        conn.close()

        deleted = expire_stale_mesh_locks(db_path)

        assert deleted == 1, f"Exactly 1 expired row must be deleted; got {deleted}"

        # Verify surviving rows
        conn2 = sqlite3.connect(str(db_path))
        remaining = conn2.execute(
            "SELECT holder_id FROM mesh_locks ORDER BY holder_id"
        ).fetchall()
        conn2.close()
        holder_ids = [r[0] for r in remaining]
        assert "dead-node" not in holder_ids, "Expired row must be gone"
        assert "live-node" in holder_ids, "_NEVER_EXPIRES sentinel must survive"
        assert "live-node-2" in holder_ids, "Future-expires row must survive"

    def test_expired_row_blocks_dead_node_fencing_token(self, tmp_path):
        """After TTL expiry, a dead node's stale fencing token is rejected.

        Fencing model: a DELETE that includes the token in the WHERE clause will
        not match any row (the row was already deleted by TTL expiry), so the
        dead node's release attempt is a silent no-op — correct behavior.
        """
        from superlocalmemory.infra.self_heal import expire_stale_mesh_locks

        db_path = tmp_path / "mesh.db"
        conn = self._create_mesh_db(db_path)

        # Dead node's expired row
        past_ts = "2000-01-01T00:00:00+00:00"
        dead_token = 42
        conn.execute(
            "INSERT INTO mesh_locks VALUES (?, ?, ?, ?, ?)",
            ("resource/shared", "dead-node", dead_token, past_ts, past_ts),
        )
        # Live node's valid row (future expiry sentinel)
        conn.execute(
            "INSERT INTO mesh_locks VALUES (?, ?, ?, ?, ?)",
            ("resource/shared", "live-node", 43, past_ts, "9999-12-31T23:59:59Z"),
        )
        conn.commit()
        conn.close()

        # Expire stale rows — dead-node's row is removed by TTL
        expire_stale_mesh_locks(db_path)

        # Dead node tries to release with its stale fencing token —
        # the token-conditional DELETE finds no matching row (already gone).
        conn2 = sqlite3.connect(str(db_path))
        cur = conn2.execute(
            "DELETE FROM mesh_locks WHERE lock_key = ? AND holder_id = ? AND fencing_token = ?",
            ("resource/shared", "dead-node", dead_token),
        )
        conn2.commit()
        rows_deleted = cur.rowcount
        conn2.close()

        assert rows_deleted == 0, (
            "Stale fencing token from dead node must not delete any row "
            "(the row was already expired)"
        )

        # Live node's row is untouched
        conn3 = sqlite3.connect(str(db_path))
        live = conn3.execute(
            "SELECT holder_id FROM mesh_locks WHERE holder_id = 'live-node'"
        ).fetchone()
        conn3.close()
        assert live is not None, "Live node's lock must survive dead-node expiry"

    def test_expire_stale_mesh_locks_noop_when_table_absent(self, tmp_path):
        """expire_stale_mesh_locks returns 0 gracefully when the table doesn't exist."""
        from superlocalmemory.infra.self_heal import expire_stale_mesh_locks

        db_path = tmp_path / "empty.db"
        # Create an empty DB with no tables
        conn = sqlite3.connect(str(db_path))
        conn.close()

        deleted = expire_stale_mesh_locks(db_path)
        assert deleted == 0  # fail-soft, no exception


# ---------------------------------------------------------------------------
# T6: PID-REUSE SAFETY — stale pid file reused by UNRELATED live process
# ---------------------------------------------------------------------------

class TestPidReuseSafety:
    """T6 — PID-reuse: stale pid file whose PID now belongs to an unrelated live process."""

    def test_plain_pid_file_with_reused_non_slm_pid_removed_not_killed(self, tmp_path):
        """A plain PID file pointing to a live but non-SLM process is treated as stale.

        The LIVE unrelated process must NEVER be killed. Only the stale file is removed.
        PID-reuse safety: alive PID + not SLM cmdline → treat as dead owner.
        """
        from superlocalmemory.infra.self_heal import reap_stale_artifacts

        # Our own test process: guaranteed alive, guaranteed not an SLM process
        live_unrelated_pid = _alive_unrelated_pid()

        pid_file = tmp_path / ".reranker-worker.pid"
        _write_plain_pid(pid_file, live_unrelated_pid)

        # _pid_is_slm must return False for our test process (it's pytest, not SLM)
        # We patch it to be explicit and test-environment-independent.
        with patch(
            "superlocalmemory.infra.self_heal._pid_is_slm",
            side_effect=lambda pid: False,  # all pids are non-SLM in this test
        ):
            report = reap_stale_artifacts(tmp_path)

        assert not pid_file.exists(), (
            "Stale pid file with a reused non-SLM PID must be removed"
        )
        removed_names = [Path(r["path"]).name for r in report["removed"]]
        assert ".reranker-worker.pid" in removed_names
        assert any(r["reason"] == "pid_reused_non_slm" for r in report["removed"]), (
            "Reason must be 'pid_reused_non_slm' not 'dead_owner_pid'"
        )

        # Critical: the live unrelated process must NOT have been killed.
        try:
            os.kill(live_unrelated_pid, 0)
        except ProcessLookupError:
            pytest.fail(
                f"PID {live_unrelated_pid} was killed — self_heal MUST NOT kill "
                "unrelated live processes"
            )

    def test_json_writer_lock_with_reused_pid_removed(self, tmp_path):
        """A writer.lock JSON file where PID was reused by an unrelated process is removed.

        Uses create_time mismatch detection: the claimed_at_ms in the JSON does not
        match the live process's actual start time → PID was reused → remove artifact.
        """
        from superlocalmemory.infra.self_heal import reap_stale_artifacts

        live_unrelated_pid = _alive_unrelated_pid()
        lock_file = tmp_path / "memory.db.writer.lock"

        # Write a JSON lock with the current live PID but a wildly wrong timestamp
        # (year 2000 in milliseconds) — guaranteed mismatch with actual create_time.
        ancient_ms = 978307200000  # 2001-01-01T00:00:00Z in ms
        _write_json_lock(lock_file, live_unrelated_pid, claimed_at_ms=ancient_ms)

        report = reap_stale_artifacts(tmp_path)

        assert not lock_file.exists(), (
            "Writer.lock with a PID-reuse create_time mismatch must be removed"
        )
        assert any(
            r["reason"] in ("pid_reused", "pid_reused_non_slm")
            for r in report["removed"]
        ), f"Expected pid_reused reason; got {report['removed']}"

    def test_valid_live_slm_pid_file_never_removed(self, tmp_path):
        """A plain PID file pointing to a live, verified SLM process is never removed.

        Simulates: the process is alive AND its command line contains 'superlocalmemory'.
        Expected: the file is kept; no process is killed.
        """
        from superlocalmemory.infra.self_heal import reap_stale_artifacts

        pid_file = tmp_path / ".reranker-worker.pid"
        live_pid = os.getpid()
        _write_plain_pid(pid_file, live_pid)

        # Pretend the live process is an SLM process
        with patch(
            "superlocalmemory.infra.self_heal._pid_is_slm",
            side_effect=lambda pid: pid == live_pid,
        ):
            report = reap_stale_artifacts(tmp_path)

        assert pid_file.exists(), (
            "PID file of a live SLM process must NOT be removed"
        )
        assert str(pid_file) in report["kept"]
        assert not report["removed"]
