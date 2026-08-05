# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""TDD RED: Writer stall watchdog + circuit breaker tests.

Tests:
  (a) Stalled item → subsequent submit() raises WriterStalledError fast
  (b) Health/status surfaces writer_stalled
  (c) After item completes, breaker resets and writes resume
  (d) Healthy-path unaffected (fast-path items finish normally)
"""

from __future__ import annotations

import sqlite3
import threading
import time

import pytest


def _install_write_commits(db_path) -> None:
    from superlocalmemory.storage.migrations import M032_write_coordinator_admission
    conn = sqlite3.connect(db_path)
    try:
        M032_write_coordinator_admission.apply(conn)
        conn.commit()
    finally:
        conn.close()


def _make_coordinator(db_path, stall_threshold: float = 30.0):
    from superlocalmemory.storage.write_coordinator import WriteCoordinator
    return WriteCoordinator(
        db_path,
        owner_id="stall-test",
        stall_threshold=stall_threshold,
    )


def _admission_payload(name: str) -> dict:
    return {
        "journal_id": f"journal:{name}",
        "request_hash": f"hash:{name}",
        "profile_id": "default",
        "idempotency_key": f"idem:{name}",
    }


class TestWriterStalledError:
    """WriterStalledError must be importable and an instance of WriteCoordinatorError."""

    def test_writer_stalled_error_importable(self):
        from superlocalmemory.storage.write_coordinator import (
            WriterStalledError,
            WriteCoordinatorError,
        )
        assert issubclass(WriterStalledError, WriteCoordinatorError)

    def test_writer_stalled_error_carries_message(self):
        from superlocalmemory.storage.write_coordinator import WriterStalledError
        err = WriterStalledError("write subsystem stalled; admin remediation required")
        assert "stalled" in str(err)


class TestStallThresholdConstructor:
    """WriteCoordinator accepts stall_threshold kwarg without changing healthy path."""

    def test_constructor_accepts_stall_threshold(self, tmp_path):
        from superlocalmemory.storage.write_coordinator import WriteCoordinator
        wc = WriteCoordinator(tmp_path / "memory.db", stall_threshold=5.0)
        assert wc.stall_threshold == 5.0

    def test_constructor_default_threshold_is_thirty_seconds(self, tmp_path):
        from superlocalmemory.storage.write_coordinator import WriteCoordinator
        wc = WriteCoordinator(tmp_path / "memory.db")
        assert wc.stall_threshold == 30.0

    def test_healthy_execute_unaffected_by_watchdog(self, tmp_path):
        """Normal sub-1s writes must NOT be affected by the watchdog."""
        from superlocalmemory.storage.write_coordinator import (
            WriteCoordinator,
            WriteDeadlineExceededError,
        )
        db_path = tmp_path / "memory.db"
        _install_write_commits(db_path)
        wc = WriteCoordinator(db_path, owner_id="healthy-test", stall_threshold=30.0)
        assert wc.claim_ownership()
        try:
            wc.execute(
                "CREATE TABLE IF NOT EXISTS sanity(v TEXT)",
                timeout=2.0,
            )
            wc.execute("INSERT INTO sanity VALUES ('ok')", timeout=2.0)
            rows = wc.execute("SELECT v FROM sanity", timeout=2.0)
            assert rows[0]["v"] == "ok"
        finally:
            wc.stop()


class TestStallCircuitBreaker:
    """Stalled worker → fast WriterStalledError for subsequent submitters."""

    def test_stalled_handler_triggers_fast_error_on_next_submit(self, tmp_path):
        """When a handler sleeps > stall_threshold, next submit raises WriterStalledError."""
        from superlocalmemory.storage.write_coordinator import (
            CommandKind,
            WriteCommand,
            WriteCoordinator,
            WriteResult,
            WriterStalledError,
        )

        db_path = tmp_path / "memory.db"
        _install_write_commits(db_path)
        stall_threshold = 0.3  # very short for testing

        wc = WriteCoordinator(db_path, owner_id="stall-cb", stall_threshold=stall_threshold)
        assert wc.claim_ownership()

        slow_started = threading.Event()

        def slow_handler(conn, _cap, cmd):
            slow_started.set()
            time.sleep(1.0)  # much longer than stall_threshold=0.3
            return WriteResult.from_receipt(cmd, {"operation_id": f"op:{cmd.command_id}"})

        wc.register_handler(CommandKind.ADMISSION, slow_handler)
        wc.execute("CREATE TABLE IF NOT EXISTS t(v TEXT)", timeout=2.0)

        # Submit slow command in background
        slow_cmd = WriteCommand.create(
            CommandKind.ADMISSION,
            {**_admission_payload("slow"), "journal_id": "j:slow"},
        )
        slow_future: list[Exception | None] = [None]

        def _submit_slow():
            try:
                wc.submit(slow_cmd, timeout=2.0)
            except Exception as exc:
                slow_future[0] = exc

        t = threading.Thread(target=_submit_slow, daemon=True)
        t.start()

        # Wait for slow handler to start
        slow_started.wait(timeout=2.0)

        # Sleep past stall_threshold
        time.sleep(stall_threshold + 0.2)

        # Next submit should fail fast with WriterStalledError
        fast_cmd = WriteCommand.create(
            CommandKind.ADMISSION,
            {**_admission_payload("fast"), "journal_id": "j:fast"},
        )
        t0 = time.monotonic()
        with pytest.raises(WriterStalledError, match="stalled"):
            wc.submit(fast_cmd, timeout=5.0)
        elapsed = time.monotonic() - t0

        # Must be fast (< 1s), not hanging
        assert elapsed < 1.0, f"WriterStalledError should be raised fast, took {elapsed:.2f}s"

        t.join(timeout=3.0)
        wc.stop()

    def test_writer_stalled_flag_exposed_on_coordinator(self, tmp_path):
        """writer_stalled property is False initially and True after a stall."""
        from superlocalmemory.storage.write_coordinator import (
            CommandKind,
            WriteCommand,
            WriteCoordinator,
            WriteResult,
        )

        db_path = tmp_path / "memory.db"
        _install_write_commits(db_path)
        stall_threshold = 0.3

        wc = WriteCoordinator(db_path, owner_id="stall-prop", stall_threshold=stall_threshold)
        assert wc.claim_ownership()
        assert wc.writer_stalled is False

        slow_started = threading.Event()

        def slow_handler(conn, _cap, cmd):
            slow_started.set()
            time.sleep(1.5)
            return WriteResult.from_receipt(cmd, {"operation_id": f"op:{cmd.command_id}"})

        wc.register_handler(CommandKind.ADMISSION, slow_handler)
        wc.execute("CREATE TABLE IF NOT EXISTS t(v TEXT)", timeout=2.0)

        slow_cmd = WriteCommand.create(
            CommandKind.ADMISSION,
            {**_admission_payload("prop"), "journal_id": "j:prop"},
        )

        def _submit_slow():
            try:
                wc.submit(slow_cmd, timeout=3.0)
            except Exception:
                pass

        t = threading.Thread(target=_submit_slow, daemon=True)
        t.start()
        slow_started.wait(timeout=2.0)
        time.sleep(stall_threshold + 0.2)

        assert wc.writer_stalled is True

        t.join(timeout=3.0)
        # After item completes, stall resets
        time.sleep(0.1)
        assert wc.writer_stalled is False

        wc.stop()

    def test_stall_resets_after_item_completes(self, tmp_path):
        """After stalled item completes, new writes resume normally."""
        from superlocalmemory.storage.write_coordinator import (
            CommandKind,
            WriteCommand,
            WriteCoordinator,
            WriteResult,
            WriterStalledError,
        )

        db_path = tmp_path / "memory.db"
        _install_write_commits(db_path)
        stall_threshold = 0.3

        wc = WriteCoordinator(db_path, owner_id="stall-reset", stall_threshold=stall_threshold)
        assert wc.claim_ownership()

        slow_started = threading.Event()
        slow_done = threading.Event()

        def slow_handler(conn, _cap, cmd):
            slow_started.set()
            time.sleep(0.6)  # > stall_threshold
            slow_done.set()
            return WriteResult.from_receipt(cmd, {"operation_id": f"op:{cmd.command_id}"})

        wc.register_handler(CommandKind.ADMISSION, slow_handler)
        wc.execute("CREATE TABLE IF NOT EXISTS t(v TEXT)", timeout=2.0)

        slow_cmd = WriteCommand.create(
            CommandKind.ADMISSION,
            {**_admission_payload("reset"), "journal_id": "j:reset"},
        )

        def _submit_slow():
            try:
                wc.submit(slow_cmd, timeout=3.0)
            except Exception:
                pass

        t = threading.Thread(target=_submit_slow, daemon=True)
        t.start()
        slow_started.wait(timeout=2.0)
        time.sleep(stall_threshold + 0.1)

        # Stalled — fast error
        with pytest.raises(WriterStalledError):
            wc.submit(
                WriteCommand.create(
                    CommandKind.ADMISSION,
                    {**_admission_payload("duringStall"), "journal_id": "j:ds"},
                ),
                timeout=2.0,
            )

        # Wait for slow to finish
        t.join(timeout=3.0)
        slow_done.wait(timeout=3.0)
        time.sleep(0.2)  # let coordinator reset

        # Now writer should be healthy again
        wc.execute("INSERT INTO t VALUES ('after-reset')", timeout=2.0)
        rows = wc.execute("SELECT v FROM t", timeout=2.0)
        assert any(r["v"] == "after-reset" for r in rows)

        wc.stop()


class TestStallHealthInfo:
    """writer_stalled info is exposed via the coordinator for health reporting."""

    def test_inflight_info_none_when_idle(self, tmp_path):
        from superlocalmemory.storage.write_coordinator import WriteCoordinator
        db_path = tmp_path / "memory.db"
        wc = WriteCoordinator(db_path, owner_id="inflight-test")
        assert wc.inflight_info() == {"stalled": False, "op_id": None, "age_s": None}
