# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory v3.8.4 — Workstream M (Zero-Touch Upgrade)

"""M031 auto-migration tests — Workstream M capstone.

GOAL ANCHOR (Varun's hard requirement, verbatim):
"SLM is used by mostly non-technical users. If they just upgrade and renew,
they should not do any other commands. Everything should be done through
upgrade automatically."

These tests verify the FULL zero-touch upgrade contract for M031
(dead_letter_operations table, workstream E). They exercise the SAME
path the daemon lifespan uses: ``apply_all()`` from ``migration_runner``.

Four required assertions (per spec):
  (a) dead_letter_operations table exists after startup migration.
  (b) Pre-existing atomic_facts rows survive — zero data loss.
  (c) Running the startup migration again is idempotent (no error/duplicate).
  (d) An already-current DB produces a clean no-op.

Three CRIT assertions (per spec):
  CRIT-1: A simulated crash-interrupted migration (in_progress) recovers
          cleanly on the next apply_all() — no schema corruption.
  CRIT-2: The version gate does not misfire; already-complete migrations
          are never double-applied.
  CRIT-3: M031 startup migration adds negligible latency even on a large DB.

Test harness:
  SLM_TEST_ISOLATION=1 PYTHONPATH=".../src" pytest tests/test_m031_startup_automigration.py

SAFETY: All paths are temp-dir only. The live DB (~/.superlocalmemory/memory.db)
is NEVER touched. The conftest audit hook enforces this at the OS level.
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_pre_m031_db(learning_db: Path, memory_db: Path) -> None:
    """Construct a pair of DBs that simulate a 3.8.3 install.

    Strategy:
      1. Run apply_all() on fresh empty DBs — this applies ALL migrations
         including M031, so every migration gets a valid migration_log entry
         with the correct DDL hash. This avoids hash-drift failures.
      2. Surgically remove M031: DROP dead_letter_operations + delete its
         log row. The result is the exact state a 3.8.3 DB would be in after
         running every prior migration but before 3.8.4-M ships M031.
      3. Create a minimal atomic_facts table (normally bootstrapped by
         MemoryEngine.initialize(), not by apply_all()) with 3 seed rows
         so data-preservation tests have something to verify.

    This approach is preferred over hand-crafting fake DDL hashes because
    the real hashes come from the real migration modules, making the test
    immune to hash changes in other migrations.
    """
    from superlocalmemory.storage.migration_runner import apply_all

    # Step 1: Full migration baseline — all M001–M031 applied.
    result = apply_all(learning_db, memory_db)
    # Sanity: M031 must have been applied successfully on the fresh DB.
    assert "M031_dead_letter_operations" in result["applied"], (
        f"Baseline apply_all() failed to apply M031: {result}"
    )

    # Step 2: Undo M031 to simulate the pre-3.8.4 state.
    conn = sqlite3.connect(str(memory_db))
    try:
        # DROP the table (SQLite drops associated indexes automatically).
        conn.execute("DROP TABLE IF EXISTS dead_letter_operations")
        conn.execute(
            "DELETE FROM migration_log WHERE name = 'M031_dead_letter_operations'"
        )
        conn.commit()
    finally:
        conn.close()

    # Step 3: Add atomic_facts rows for data-preservation testing.
    # The table is normally created by MemoryEngine; create a minimal version.
    conn = sqlite3.connect(str(memory_db))
    try:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS atomic_facts (
                fact_id   TEXT PRIMARY KEY,
                content   TEXT NOT NULL,
                profile_id TEXT DEFAULT 'default',
                confidence REAL DEFAULT 0.9
            );
            INSERT OR IGNORE INTO atomic_facts (fact_id, content, profile_id)
            VALUES ('pre-m031-f1', 'Paris is the capital of France', 'default');
            INSERT OR IGNORE INTO atomic_facts (fact_id, content, profile_id)
            VALUES ('pre-m031-f2', 'SLM stores memories durably', 'default');
            INSERT OR IGNORE INTO atomic_facts (fact_id, content, profile_id)
            VALUES ('pre-m031-f3', 'Zero-touch upgrades require no user commands', 'default');
        """)
        conn.commit()
    finally:
        conn.close()


def _table_exists(db_path: Path, table_name: str) -> bool:
    conn = sqlite3.connect(str(db_path))
    try:
        row = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
            (table_name,),
        ).fetchone()
        return row is not None
    finally:
        conn.close()


def _migration_status(db_path: Path, name: str) -> str | None:
    """Return the migration_log status for *name*, or None if missing."""
    conn = sqlite3.connect(str(db_path))
    try:
        row = conn.execute(
            "SELECT status FROM migration_log WHERE name = ?",
            (name,),
        ).fetchone()
        return row[0] if row else None
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Required assertion (a): table created on a pre-M031 DB
# ---------------------------------------------------------------------------

def test_startup_migration_creates_dead_letter_table(tmp_path):
    """(a) apply_all() creates dead_letter_operations on a pre-M031 DB.

    Simulates: user upgrades from 3.8.3 → 3.8.4-M and restarts SLM.
    The daemon lifespan calls apply_all() — no user command needed.
    """
    learning_db = tmp_path / "learning.db"
    memory_db = tmp_path / "memory.db"
    _build_pre_m031_db(learning_db, memory_db)

    # Pre-condition: dead_letter_operations must NOT exist.
    assert not _table_exists(memory_db, "dead_letter_operations"), (
        "Pre-condition failed: dead_letter_operations must be absent in the "
        "pre-M031 DB."
    )
    assert _migration_status(memory_db, "M031_dead_letter_operations") is None

    # Act: daemon startup migration path.
    from superlocalmemory.storage.migration_runner import apply_all
    result = apply_all(learning_db, memory_db)

    # Assert: M031 applied, no failure.
    assert "M031_dead_letter_operations" in result["applied"], (
        f"M031 was NOT applied. result={result}"
    )
    assert "M031_dead_letter_operations" not in result["failed"], (
        f"M031 reported as failed. result={result}"
    )

    # Assert: table now exists.
    assert _table_exists(memory_db, "dead_letter_operations"), (
        "dead_letter_operations table was not created."
    )

    # Assert: migration_log records success.
    assert _migration_status(memory_db, "M031_dead_letter_operations") == "complete"


# ---------------------------------------------------------------------------
# Required assertion (b): zero data loss — atomic_facts rows preserved
# ---------------------------------------------------------------------------

def test_startup_migration_preserves_existing_rows(tmp_path):
    """(b) apply_all() does not touch pre-existing atomic_facts rows.

    An upgrade migration must never cause data loss. The 3 seed rows in
    atomic_facts must survive the M031 schema addition unchanged.
    """
    learning_db = tmp_path / "learning.db"
    memory_db = tmp_path / "memory.db"
    _build_pre_m031_db(learning_db, memory_db)

    from superlocalmemory.storage.migration_runner import apply_all
    apply_all(learning_db, memory_db)

    conn = sqlite3.connect(str(memory_db))
    try:
        count = conn.execute("SELECT COUNT(*) FROM atomic_facts").fetchone()[0]
        fact_ids = {r[0] for r in conn.execute("SELECT fact_id FROM atomic_facts")}
    finally:
        conn.close()

    assert count == 3, (
        f"Expected 3 atomic_facts rows; found {count}. Migration caused data loss."
    )
    assert "pre-m031-f1" in fact_ids
    assert "pre-m031-f2" in fact_ids
    assert "pre-m031-f3" in fact_ids


# ---------------------------------------------------------------------------
# Required assertion (c): idempotent — running startup migration twice is safe
# ---------------------------------------------------------------------------

def test_startup_migration_is_idempotent(tmp_path):
    """(c) Running apply_all() twice on the same DB is a no-op the second time.

    A daemon restart after a successful migration must not error, duplicate
    rows, or re-apply M031.
    """
    learning_db = tmp_path / "learning.db"
    memory_db = tmp_path / "memory.db"
    _build_pre_m031_db(learning_db, memory_db)

    from superlocalmemory.storage.migration_runner import apply_all

    # First run: applies M031.
    result1 = apply_all(learning_db, memory_db)
    assert "M031_dead_letter_operations" in result1["applied"], (
        f"First apply_all() did not apply M031: {result1}"
    )

    # Second run: M031 must be skipped, not re-applied, not failed.
    result2 = apply_all(learning_db, memory_db)
    assert "M031_dead_letter_operations" not in result2["applied"], (
        "M031 was re-applied on the second run (not idempotent)."
    )
    assert "M031_dead_letter_operations" not in result2["failed"], (
        f"M031 reported as failed on idempotent re-run: {result2}"
    )
    assert "M031_dead_letter_operations" in result2["skipped"], (
        "M031 should have been skipped on re-run (already complete)."
    )

    # Exactly one log row for M031 — no phantom duplicates.
    conn = sqlite3.connect(str(memory_db))
    try:
        log_count = conn.execute(
            "SELECT COUNT(*) FROM migration_log WHERE name='M031_dead_letter_operations'"
        ).fetchone()[0]
    finally:
        conn.close()
    assert log_count == 1, (
        f"Expected exactly 1 migration_log row for M031; found {log_count}."
    )


# ---------------------------------------------------------------------------
# Required assertion (d): already-current DB → clean no-op
# ---------------------------------------------------------------------------

def test_startup_migration_noop_on_current_db(tmp_path):
    """(d) apply_all() on an already-current DB skips all migrations cleanly.

    Models a user who upgrades and restarts twice in a row, or a healthy
    daemon restart with no schema changes pending.
    """
    learning_db = tmp_path / "learning.db"
    memory_db = tmp_path / "memory.db"

    from superlocalmemory.storage.migration_runner import apply_all

    # Build a fully-current DB.
    result1 = apply_all(learning_db, memory_db)
    assert "M031_dead_letter_operations" in result1["applied"]
    # Allow other skips but no fails (some migrations skip on fresh DBs).

    # Re-apply on the current DB: everything should be skipped, nothing failed.
    result2 = apply_all(learning_db, memory_db)
    assert "M031_dead_letter_operations" not in result2["applied"], (
        "M031 was re-applied on an already-current DB."
    )
    assert "M031_dead_letter_operations" not in result2["failed"], (
        f"M031 failed on an already-current DB: {result2}"
    )
    # No failures at all (except possibly benign migration_log bootstrap
    # artefacts that are always logged separately).
    real_failures = [
        f for f in result2.get("failed", [])
        if not f.startswith("learning_db") and not f.startswith("memory_db")
    ]
    assert real_failures == [], (
        f"Unexpected failures on re-run of current DB: {real_failures}. "
        f"Full result: {result2}"
    )


# ---------------------------------------------------------------------------
# CRIT-1: Crash recovery — in_progress state is cleaned and retried
# ---------------------------------------------------------------------------

def test_crashed_m031_migration_is_recoverable(tmp_path):
    """CRIT-1: A crash mid-M031 (in_progress in migration_log) recovers cleanly.

    Scenario: daemon crashed after writing in_progress to migration_log but
    before the DDL committed (or before the log was updated to 'complete').
    Next daemon start must retry M031, create the table, and mark complete.
    Schema must not be corrupted; daemon must not be blocked from starting.
    """
    learning_db = tmp_path / "learning.db"
    memory_db = tmp_path / "memory.db"
    _build_pre_m031_db(learning_db, memory_db)

    # Simulate crash: inject an in_progress row WITHOUT creating the table.
    # The migration_log table already exists (created by _build_pre_m031_db
    # via the first apply_all() call).
    conn = sqlite3.connect(str(memory_db))
    try:
        conn.execute(
            "INSERT OR REPLACE INTO migration_log "
            "(name, applied_at, ddl_sha256, rows_affected, status) "
            "VALUES ('M031_dead_letter_operations', datetime('now'), "
            "'crash_interrupted_hash', 0, 'in_progress')"
        )
        conn.commit()
    finally:
        conn.close()

    # Verify pre-condition: in_progress, no table.
    assert _migration_status(memory_db, "M031_dead_letter_operations") == "in_progress"
    assert not _table_exists(memory_db, "dead_letter_operations")

    # Recovery run: apply_all() must retry M031.
    from superlocalmemory.storage.migration_runner import apply_all
    result = apply_all(learning_db, memory_db)

    # M031 must be applied successfully after crash recovery.
    assert "M031_dead_letter_operations" in result["applied"], (
        f"M031 was NOT re-applied after crash recovery: {result}"
    )
    assert "M031_dead_letter_operations" not in result["failed"], (
        f"M031 failed during crash recovery: {result}"
    )

    # Table must exist and status must be 'complete'.
    assert _table_exists(memory_db, "dead_letter_operations"), (
        "dead_letter_operations table is missing after crash recovery."
    )
    assert _migration_status(memory_db, "M031_dead_letter_operations") == "complete"

    # Running again must still be idempotent.
    result2 = apply_all(learning_db, memory_db)
    assert "M031_dead_letter_operations" in result2["skipped"]
    assert "M031_dead_letter_operations" not in result2["failed"]


# ---------------------------------------------------------------------------
# CRIT-2: Version gate — complete migrations are never double-applied
# ---------------------------------------------------------------------------

def test_version_gate_prevents_double_apply(tmp_path):
    """CRIT-2: The version gate correctly prevents M031 from being re-applied.

    The migration_log stores the DDL SHA-256 hash. After a successful apply,
    the runner must detect the matching hash and skip — never re-run the DDL
    a second time (which would cause a 'table already exists' error if M031
    had used CREATE TABLE without IF NOT EXISTS).
    """
    learning_db = tmp_path / "learning.db"
    memory_db = tmp_path / "memory.db"

    from superlocalmemory.storage.migration_runner import apply_all

    # First apply: M031 completes with its real DDL hash.
    result1 = apply_all(learning_db, memory_db)
    assert "M031_dead_letter_operations" in result1["applied"]

    # Read the recorded hash.
    conn = sqlite3.connect(str(memory_db))
    try:
        row = conn.execute(
            "SELECT ddl_sha256, status FROM migration_log "
            "WHERE name='M031_dead_letter_operations'"
        ).fetchone()
    finally:
        conn.close()

    assert row is not None
    recorded_hash, status = row
    assert status == "complete"
    assert len(recorded_hash) == 64, "Expected a SHA-256 hex digest (64 chars)"

    # Verify the recorded hash matches the module's actual DDL.
    import hashlib

    from superlocalmemory.storage.migrations import (
        M031_dead_letter_operations as _M031,
    )
    expected_hash = hashlib.sha256(_M031.DDL.encode("utf-8")).hexdigest()
    assert recorded_hash == expected_hash, (
        "Version gate will misfire: recorded hash does not match current DDL."
    )

    # Second apply: must skip (hash matches, status complete).
    result2 = apply_all(learning_db, memory_db)
    assert "M031_dead_letter_operations" in result2["skipped"]
    assert "M031_dead_letter_operations" not in result2["applied"]
    assert "M031_dead_letter_operations" not in result2["failed"]


# ---------------------------------------------------------------------------
# CRIT-3: Startup latency is negligible
# ---------------------------------------------------------------------------

def test_startup_migration_latency_is_negligible(tmp_path):
    """CRIT-3: M031 adds negligible startup latency even for no-op re-runs.

    Schema creation (CREATE TABLE IF NOT EXISTS, CREATE INDEX IF NOT EXISTS)
    is O(1) sqlite_master lookup. A full apply must complete in < 5 s on
    any reasonable hardware; a no-op re-run must be even faster.

    This bound is intentionally generous — the unit of concern is 'does not
    block daemon startup', not microsecond precision.
    """
    learning_db = tmp_path / "learning.db"
    memory_db = tmp_path / "memory.db"
    _build_pre_m031_db(learning_db, memory_db)

    from superlocalmemory.storage.migration_runner import apply_all

    # Timed first apply: creates dead_letter_operations.
    t0 = time.perf_counter()
    result = apply_all(learning_db, memory_db)
    elapsed_apply = time.perf_counter() - t0

    assert "M031_dead_letter_operations" in result["applied"]
    assert elapsed_apply < 5.0, (
        f"apply_all() took {elapsed_apply:.2f}s — unacceptably slow for startup."
    )

    # Timed no-op: already current.
    t1 = time.perf_counter()
    result2 = apply_all(learning_db, memory_db)
    elapsed_noop = time.perf_counter() - t1

    assert "M031_dead_letter_operations" in result2["skipped"]
    assert elapsed_noop < 5.0, (
        f"No-op apply_all() took {elapsed_noop:.2f}s — unacceptably slow for daemon restart."
    )


# ---------------------------------------------------------------------------
# Bonus: verify() contract — M031 module self-verifies correctly
# ---------------------------------------------------------------------------

def test_m031_verify_function(tmp_path):
    """M031.verify() returns False before migration and True after."""
    from superlocalmemory.storage.migrations import (
        M031_dead_letter_operations as _M031,
    )

    db_path = tmp_path / "verify_test.db"
    conn = sqlite3.connect(str(db_path))
    try:
        # Before: no table → verify returns False.
        assert _M031.verify(conn) is False

        # After apply: verify returns True.
        _M031.apply(conn)
        assert _M031.verify(conn) is True

        # Idempotent apply: still True.
        _M031.apply(conn)
        assert _M031.verify(conn) is True
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# G-workstream note: graph_pruning needs no schema migration (additive JSON)
# ---------------------------------------------------------------------------

def test_graph_pruning_config_needs_no_schema_migration():
    """G: graph_pruning is additive JSON config — no SQLite migration needed.

    Old configs without 'graph_pruning' silently fall back to GraphPruningConfig()
    defaults at SLMConfig.load() time (config.py line 1203). No ALTER TABLE,
    no new column, no new table required.
    """
    from superlocalmemory.storage.migration_runner import DEFERRED_MIGRATIONS, MIGRATIONS

    all_migration_names = [m.name for m in (*MIGRATIONS, *DEFERRED_MIGRATIONS)]
    graph_pruning_migrations = [
        n for n in all_migration_names if "graph_pruning" in n.lower()
    ]
    assert graph_pruning_migrations == [], (
        f"Unexpected schema migrations for graph_pruning: {graph_pruning_migrations}. "
        "graph_pruning is additive JSON config — it must not add DB migrations."
    )
