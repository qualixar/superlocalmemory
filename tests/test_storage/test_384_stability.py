# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""SLM 3.8.4 Stability — TDD test suite.

Five tests corresponding to Fixes A–F.  All should FAIL on v3.8.3 and PASS
after the fixes are applied.

Test coverage:
  1. Lock-storm regression: 8 concurrent writers, zero "database is locked".
  2. Graph-pruner batched lock: concurrent writer never waits > 100 ms per batch.
  3. Legacy pending.db dead-letter: items transition after _MAX_RETRY_COUNT retries.
  4. M018 dead-letter table: exhausted operations moved to dead_letter_operations.
  5. EventBus resilience: emit() never raises on a briefly locked database.
"""

from __future__ import annotations

import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import pytest

from superlocalmemory.storage import schema as real_schema
from superlocalmemory.storage.database import DatabaseManager
from superlocalmemory.storage.migrations import M018_ingestion_operations as M018

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _init_db(db: DatabaseManager) -> None:
    """Apply base schema so that tables like graph_edges, memories exist."""
    db.initialize(real_schema)
    # Also apply M018 for ingestion_operations
    with db.raw_connection() as conn:
        M018.apply(conn)


def _insert_orphan_edges(db: DatabaseManager, profile_id: str, n: int) -> None:
    """Insert n fake graph_edges whose source/target exist in NO fact/entity table.

    These will all be found by _remove_orphan_edges and queued for deletion —
    making them the worst-case scenario for lock hold duration.
    """
    # Use 'entity' — the only non-CHECK-constrained common type
    created_at = "2026-01-01T00:00:00"
    rows = [
        (f"e{i}", profile_id, f"orphan-src-{i}", f"orphan-tgt-{i}", "entity", 1.0, created_at)
        for i in range(n)
    ]
    # Insert in small batches to avoid one huge transaction
    batch_size = 200
    for start in range(0, len(rows), batch_size):
        chunk = rows[start:start + batch_size]
        with db.transaction():
            for row in chunk:
                db.execute(
                    "INSERT OR IGNORE INTO graph_edges "
                    "(edge_id, profile_id, source_id, target_id, edge_type, weight, created_at) "
                    "VALUES (?,?,?,?,?,?,?)",
                    row,
                )


# ---------------------------------------------------------------------------
# Test 1: Lock storm regression
# ---------------------------------------------------------------------------

class TestLockStormRegression:
    """Fix B: serialise execute() via RLock; Fix C: EventBus busy_timeout.

    8 threads each do 50 write+emit cycles. Before the fix these races produce
    'database is locked' errors because execute() outside transaction() holds
    no lock and EventBus opens bare connections with no busy_timeout PRAGMA.
    After fixes the RLock serialises all DML and EventBus waits properly.
    """

    def test_no_database_locked_under_8_concurrent_writers(
        self, tmp_path: Path
    ) -> None:
        db = DatabaseManager(tmp_path / "memory.db")
        _init_db(db)

        # Import EventBus here so it is NOT imported before DB is ready
        from superlocalmemory.infra.event_bus import EventBus
        EventBus.reset_instance(tmp_path / "memory.db")
        # Pass db so EventBus routes writes through DatabaseManager (Fix C)
        bus = EventBus(tmp_path / "memory.db", db=db)

        errors: list[str] = []
        lock_errors: list[str] = []

        def writer(thread_id: int) -> None:
            for i in range(50):
                try:
                    with db.transaction():
                        db.execute(
                            "INSERT OR IGNORE INTO memories "
                            "(memory_id, profile_id, content, created_at) "
                            "VALUES (?,?,?,?)",
                            (
                                f"mem-{thread_id}-{i}",
                                "default",
                                f"content {thread_id} {i}",
                                time.time(),
                            ),
                        )
                    # EventBus emit goes through DatabaseManager (Fix C)
                    bus.emit("memory.stored", {"thread": thread_id, "i": i})
                except Exception as exc:
                    msg = str(exc).lower()
                    if "database is locked" in msg or "locked" in msg:
                        lock_errors.append(
                            f"Thread {thread_id}, iter {i}: {exc}"
                        )
                    else:
                        errors.append(f"Thread {thread_id}, iter {i}: {exc}")

        threads = [threading.Thread(target=writer, args=(n,)) for n in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=60)

        assert not lock_errors, (
            f"Lock storm detected ({len(lock_errors)} errors): "
            f"{lock_errors[:3]}"
        )
        # Other errors (e.g. table not found) should not exist either
        assert not errors, f"Unexpected errors: {errors[:3]}"


# ---------------------------------------------------------------------------
# Test 2: Graph pruner batched lock hold
# ---------------------------------------------------------------------------

class TestGraphPrunerBatchedLock:
    """Fix A: prune_graph accepts DatabaseManager as first arg and uses batched
    transactions routed through the shared writer lock.

    Before Fix A: prune_graph(db_path: str | Path, ...) opens its own
    sqlite3 connection and holds ONE long BEGIN...COMMIT for ALL deletes.
    Passing a DatabaseManager as the first arg would create a wrong database
    (sqlite3.connect() on the object repr string) and prune nothing.

    After Fix A: prune_graph(db: DatabaseManager | str | Path, ...) accepts
    a DatabaseManager directly; edges are pruned from the correct database.
    The function also routes DELETEs through batched db.transaction() calls.
    """

    def test_prune_graph_accepts_database_manager_and_prunes_edges(
        self, tmp_path: Path
    ) -> None:
        db = DatabaseManager(tmp_path / "memory.db")
        _init_db(db)

        # Insert 5 000 orphan edges (orphan = source/target not in facts/entities)
        _insert_orphan_edges(db, "default", 5_000)

        before = db.execute(
            "SELECT COUNT(*) AS c FROM graph_edges WHERE profile_id='default'"
        )
        assert int(before[0]["c"]) == 5_000, "Setup: edge insert failed"

        from superlocalmemory.core.graph_pruner import prune_graph

        # Pass DatabaseManager directly (Fix A API change)
        stats = prune_graph(db, "default")

        after = db.execute(
            "SELECT COUNT(*) AS c FROM graph_edges WHERE profile_id='default'"
        )
        edges_after = int(after[0]["c"])

        # All 5000 edges were orphans → should be removed
        assert edges_after < 5_000, (
            f"Expected orphan edges to be pruned, but count unchanged at {edges_after}. "
            "prune_graph may have operated on a wrong (empty) database — "
            "Fix A not applied."
        )
        assert stats.get("orphans_removed", 0) > 0 or stats.get("hub_edges_removed", 0) > 0, (
            f"Stats show no removal: {stats}. Pruner ran on wrong DB."
        )

    def test_prune_graph_batched_lock_hold(self, tmp_path: Path) -> None:
        """Concurrent writer never waits > 200 ms per write while prune runs.

        This is a best-effort timing test — it passes even without Fix A on
        fast hardware.  The primary correctness test is above.  This verifies
        the batching doesn't introduce a regression.
        """
        db = DatabaseManager(tmp_path / "memory.db")
        _init_db(db)
        _insert_orphan_edges(db, "default", 3_000)

        write_latencies: list[float] = []
        stop_event = threading.Event()
        writer_errors: list[str] = []

        def concurrent_writer() -> None:
            idx = 0
            while not stop_event.is_set():
                t0 = time.perf_counter()
                try:
                    db.execute(
                        "INSERT OR REPLACE INTO memories "
                        "(memory_id, profile_id, content, created_at) "
                        "VALUES (?,?,?,?)",
                        (f"probe-{idx}", "default", "probe", "2026-01-01T00:00:00"),
                    )
                except Exception as exc:
                    writer_errors.append(str(exc))
                write_latencies.append(time.perf_counter() - t0)
                idx += 1
                time.sleep(0.001)

        wt = threading.Thread(target=concurrent_writer, daemon=True)
        wt.start()
        time.sleep(0.02)

        from superlocalmemory.core.graph_pruner import prune_graph
        prune_graph(db, "default")

        stop_event.set()
        wt.join(timeout=10)

        assert not writer_errors, f"Writer errors during prune: {writer_errors[:3]}"
        assert write_latencies, "Writer never ran"
        max_ms = max(write_latencies) * 1000
        # 200 ms is generous — WAL busy_timeout is 10s; actual batched hold is < 10 ms
        assert max_ms < 200.0, (
            f"Write stall during prune: {max_ms:.1f} ms (limit 200 ms)"
        )


# ---------------------------------------------------------------------------
# Test 3: Legacy pending.db dead-letter cap
# ---------------------------------------------------------------------------

class TestPendingDeadLetterCap:
    """Fix F: pending_store.mark_failed caps at _MAX_RETRY_COUNT and
    transitions to 'dead_letter' status.

    Before the fix: status is always reset to 'pending' — items retry forever.
    After: after _MAX_RETRY_COUNT failures the item is dead-lettered.
    """

    def test_pending_dead_lettered_after_max_retries(
        self, tmp_path: Path
    ) -> None:
        from superlocalmemory.cli.pending_store import (
            _MAX_RETRY_COUNT,  # type: ignore[attr-defined]  # added by Fix F
            mark_failed,
            store_pending,
        )

        row_id = store_pending(
            "persistent failure content", base_dir=tmp_path
        )

        for attempt in range(_MAX_RETRY_COUNT):
            mark_failed(row_id, "simulated error", base_dir=tmp_path)

        # Read the row directly from the DB
        db_path = tmp_path / "pending.db"
        conn = sqlite3.connect(str(db_path))
        try:
            row = conn.execute(
                "SELECT status, next_retry_at FROM pending_memories WHERE id=?",
                (row_id,),
            ).fetchone()
        finally:
            conn.close()

        assert row is not None, "Row disappeared"
        assert row[0] == "dead_letter", (
            f"Expected status='dead_letter' after {_MAX_RETRY_COUNT} retries, "
            f"got '{row[0]}'"
        )
        assert row[1] is None, (
            "Dead-lettered item should have next_retry_at=NULL"
        )

    def test_dead_lettered_item_not_in_work_queue(
        self, tmp_path: Path
    ) -> None:
        from superlocalmemory.cli.pending_store import (
            _MAX_RETRY_COUNT,  # type: ignore[attr-defined]
            get_pending,
            mark_failed,
            store_pending,
        )

        row_id = store_pending("will be dead-lettered", base_dir=tmp_path)
        for _ in range(_MAX_RETRY_COUNT):
            mark_failed(row_id, "simulated", base_dir=tmp_path)

        work_queue = get_pending(base_dir=tmp_path)
        assert not any(r["id"] == row_id for r in work_queue), (
            "Dead-lettered item MUST NOT appear in work queue"
        )


# ---------------------------------------------------------------------------
# Test 4: M018 dead-letter operations table
# ---------------------------------------------------------------------------

class TestM018DeadLetterTable:
    """Fix E: ingestion operations exhausted after _MAX_AUTOMATIC_MATERIALIZATION_ATTEMPTS
    are moved to dead_letter_operations, not silently dropped to FAILED.

    Before the fix: silently drops to FAILED, never visible to operators.
    After: row appears in dead_letter_operations with original_op_id and error.
    """

    def _make_db(self, tmp_path: Path) -> DatabaseManager:
        """Return a DatabaseManager with ingestion_operations + dead_letter_operations tables."""
        db = DatabaseManager(tmp_path / "memory.db")
        _init_db(db)
        # Apply M031 migration to create dead_letter_operations
        from superlocalmemory.storage.migrations import M031_dead_letter_operations as M031
        with db.raw_connection() as conn:
            M031.apply(conn)
        return db

    def test_m018_operation_dead_lettered_after_10_attempts(
        self, tmp_path: Path
    ) -> None:
        from superlocalmemory.core.ingestion_command import (
            _MAX_AUTOMATIC_MATERIALIZATION_ATTEMPTS,
            IngestionCommand,
            IngestionOperationRepository,
            IngestionRequest,
        )

        db = self._make_db(tmp_path)
        repo = IngestionOperationRepository(db)

        # Fake write_queryable: just returns a dummy fact ID
        def _fake_write_queryable(request: IngestionRequest, op_id: str) -> list[str]:
            return [f"fact-{uuid.uuid4().hex}"]

        # Fake materializer: always raises
        def _fake_materialize(operation: Any) -> list[str]:
            raise RuntimeError("persistent failure")

        cmd = IngestionCommand(
            repository=repo,
            write_queryable=_fake_write_queryable,
            materialize=_fake_materialize,
        )

        request = IngestionRequest(
            content="test evidence for dead-letter",
            profile_id="default",
            source_type="test",
            idempotency_key=uuid.uuid4().hex,
        )
        operation = cmd.submit(request)
        op_id = operation.operation_id

        # The first nine real claims remain retryable and must be recorded as
        # nine, not ten. claim_enriching(), not finish_enriching(), owns the
        # single attempt-count increment.
        for attempt in range(_MAX_AUTOMATIC_MATERIALIZATION_ATTEMPTS - 1):
            try:
                cmd.materialize(op_id)
            except Exception:
                pass  # materialize errors are expected

        before_boundary = dict(db.execute(
            "SELECT attempt_count FROM ingestion_operations WHERE operation_id = ?",
            (op_id,),
        )[0])
        assert (
            before_boundary["attempt_count"]
            == _MAX_AUTOMATIC_MATERIALIZATION_ATTEMPTS - 1
        )
        assert db.execute(
            "SELECT 1 FROM dead_letter_operations WHERE original_op_id = ?",
            (op_id,),
        ) == []

        # The tenth claim is the exact dead-letter boundary.
        cmd.materialize(op_id)

        # Verify NOT in the materializable work queue
        materializable = repo.list_materializable(limit=100)
        assert not any(
            op.operation_id == op_id for op in materializable
        ), "Exhausted operation must NOT appear in work queue"

        # Verify IS in dead_letter_operations (Fix E)
        dead = db.execute(
            "SELECT * FROM dead_letter_operations WHERE original_op_id = ?",
            (op_id,),
        )
        assert len(dead) == 1, (
            f"Expected 1 dead_letter_operations row, found {len(dead)}. "
            "Fix E not applied."
        )
        row = dict(dead[0])
        assert row["attempt_count"] == _MAX_AUTOMATIC_MATERIALIZATION_ATTEMPTS, (
            f"Expected attempt_count={_MAX_AUTOMATIC_MATERIALIZATION_ATTEMPTS}, "
            f"got {row['attempt_count']}"
        )


# ---------------------------------------------------------------------------
# Test 5: EventBus resilience under locked DB
# ---------------------------------------------------------------------------

class TestEventBusResilience:
    """Fix C: EventBus routes through DatabaseManager; no 'database is locked'
    even when a competing connection holds the write lock for 500 ms.

    Before the fix: bare sqlite3.connect() with 5-second Python default timeout
    and no PRAGMA busy_timeout — can raise on any contention > 5s.
    After: routes through DatabaseManager (which has 10s busy_timeout + RLock).
    """

    def test_event_bus_resilient_on_busy_db(self, tmp_path: Path) -> None:
        db = DatabaseManager(tmp_path / "memory.db")
        _init_db(db)

        from superlocalmemory.infra.event_bus import EventBus
        EventBus.reset_instance(tmp_path / "memory.db")
        bus = EventBus(tmp_path / "memory.db", db=db)

        # Hold the SQLite write lock externally for 500 ms
        lock_held = threading.Event()
        lock_released = threading.Event()

        def hold_lock_briefly() -> None:
            conn = sqlite3.connect(str(tmp_path / "memory.db"))
            conn.execute("BEGIN IMMEDIATE")
            lock_held.set()
            time.sleep(0.5)
            conn.rollback()
            conn.close()
            lock_released.set()

        lock_holder = threading.Thread(target=hold_lock_briefly, daemon=True)
        lock_holder.start()

        # Wait for lock to be established
        lock_held.wait(timeout=5)

        try:
            # emit() should NOT raise even though the lock is held for 500 ms
            # because Fix C routes through DatabaseManager's 10s busy_timeout
            bus.emit("memory.stored", {"test": "resilience"})
        except Exception as exc:
            pytest.fail(
                f"EventBus.emit() raised while DB was briefly locked: {exc!r}. "
                "Fix C not applied or bus not using DatabaseManager."
            )
        finally:
            lock_holder.join(timeout=5)
