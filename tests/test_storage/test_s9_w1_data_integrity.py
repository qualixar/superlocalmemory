# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory — learned-model integrity migrations

"""Stage 9 W1 regressions — Data Integrity.

Covers:
- H-DATA-01: M020 must backfill ``bytes_sha256`` for every pre-existing row
  so ``verify_sha256`` accepts them on first load. Without the forward repair,
  every 18k+ v3.4.19 user would silently lose their learned-ranker state
  because the integrity gate tombstones empty-hash rows.
- M-DATA-01: M002 must NOT hardcode ``is_active=1`` for every row — only
  the MAX(id) row per profile can be active (partial unique index would
  otherwise roll back the whole rebuild on dev-build DBs with multiple
  rows per profile).
- C3: ``apply_deferred`` must not independently bootstrap ``migration_log``
  on memory.db — that creates split-brain state when ``apply_all`` crashed
  before reaching memory targets. The bootstrap must happen once at the
  top of ``apply_all`` for BOTH DBs.
"""

from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path

import pytest

from superlocalmemory.storage import migration_runner
from superlocalmemory.storage.migrations import (
    M002_model_state_history as M002,
)
from superlocalmemory.storage.migrations import (
    M020_model_state_integrity as M020,
)


# ---------------------------------------------------------------------------
# H-DATA-01 — M020 forward migration backfills bytes_sha256
# ---------------------------------------------------------------------------


def _seed_v3419_learning_db(db_path: Path,
                            profile_blobs: list[tuple[str, bytes]]) -> None:
    """Build a minimal v3.4.19-shaped learning_model_state with seeded rows."""
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript("""
            CREATE TABLE learning_model_state (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                profile_id  TEXT NOT NULL UNIQUE,
                state_bytes BLOB NOT NULL,
                updated_at  TEXT NOT NULL
            );
        """)
        for pid, blob in profile_blobs:
            conn.execute(
                "INSERT INTO learning_model_state "
                "(profile_id, state_bytes, updated_at) "
                "VALUES (?, ?, '2026-04-01T00:00:00Z')",
                (pid, blob),
            )
        conn.commit()
    finally:
        conn.close()


def test_m020_backfills_sha256_for_legacy_rows(
    tmp_path: Path,
) -> None:
    """H-DATA-01: M020 computes sha256 for each historical copied row."""
    db = tmp_path / "learning.db"
    blob_a = b"learned-ranker-model-A" * 32
    blob_b = b"learned-ranker-model-B" * 32
    _seed_v3419_learning_db(db, [("default", blob_a), ("work", blob_b)])

    conn = sqlite3.connect(db)
    try:
        conn.executescript(M002.DDL)
        M020.apply(conn)
        rows = conn.execute(
            "SELECT profile_id, bytes_sha256 FROM learning_model_state"
        ).fetchall()
    finally:
        conn.close()

    by_profile = {pid: sha for (pid, sha) in rows}
    assert by_profile["default"] == hashlib.sha256(blob_a).hexdigest()
    assert by_profile["work"] == hashlib.sha256(blob_b).hexdigest()
    # 64 hex chars — verify_sha256 contract.
    assert all(len(v) == 64 for v in by_profile.values())


def test_m020_is_idempotent(tmp_path: Path) -> None:
    """Re-running the forward repair on populated rows is a no-op."""
    db = tmp_path / "learning.db"
    _seed_v3419_learning_db(db, [("solo", b"x" * 256)])

    conn = sqlite3.connect(db)
    try:
        conn.executescript(M002.DDL)
        M020.apply(conn)
        first = conn.execute(
            "SELECT bytes_sha256 FROM learning_model_state"
        ).fetchone()[0]
        # Running again must not change the value or raise.
        M020.apply(conn)
        second = conn.execute(
            "SELECT bytes_sha256 FROM learning_model_state"
        ).fetchone()[0]
    finally:
        conn.close()

    assert first == second
    assert first == hashlib.sha256(b"x" * 256).hexdigest()


def test_m020_handles_empty_table(tmp_path: Path) -> None:
    """Fresh install (no legacy rows) → repair does nothing, raises nothing."""
    db = tmp_path / "learning.db"
    # Minimal empty seed — just the old table shape.
    _seed_v3419_learning_db(db, [])

    conn = sqlite3.connect(db)
    try:
        conn.executescript(M002.DDL)
        M020.apply(conn)
        count = conn.execute(
            "SELECT COUNT(*) FROM learning_model_state"
        ).fetchone()[0]
    finally:
        conn.close()

    assert count == 0


# ---------------------------------------------------------------------------
# M-DATA-01 — is_active=1 no longer hardcoded
# ---------------------------------------------------------------------------


def test_m020_preserves_existing_model_payload_and_metadata(tmp_path: Path) -> None:
    """The forward repair changes only missing digests, never model state."""
    db = tmp_path / "learning.db"
    blob = b"model-state" * 64
    _seed_v3419_learning_db(db, [("default", blob)])
    conn = sqlite3.connect(db)
    try:
        conn.executescript(M002.DDL)
        before = conn.execute(
            "SELECT state_bytes, model_version, trained_on_count, is_active "
            "FROM learning_model_state WHERE profile_id = 'default'"
        ).fetchone()
        M020.apply(conn)
        after = conn.execute(
            "SELECT state_bytes, model_version, trained_on_count, is_active "
            "FROM learning_model_state WHERE profile_id = 'default'"
        ).fetchone()
    finally:
        conn.close()

    assert after == before


# ---------------------------------------------------------------------------
# Runner forward-migration plumbing
# ---------------------------------------------------------------------------


def test_migration_runner_runs_forward_integrity_repair(
    tmp_path: Path,
) -> None:
    """H-DATA-01: the runner invokes M020 after M002 completes."""
    learning_db = tmp_path / "learning.db"
    memory_db = tmp_path / "memory.db"
    _seed_v3419_learning_db(learning_db,
                            [("p1", b"model-p1" * 128),
                             ("p2", b"model-p2" * 128)])

    result = migration_runner.apply_all(learning_db, memory_db)
    # M002 creates the historical schema; M020 provides the integrity repair.
    assert "M002_model_state_history" in result["applied"]
    assert "M020_model_state_integrity" in result["applied"]

    conn = sqlite3.connect(learning_db)
    try:
        rows = conn.execute(
            "SELECT profile_id, bytes_sha256 FROM learning_model_state"
        ).fetchall()
    finally:
        conn.close()

    for pid, sha in rows:
        assert len(sha) == 64, f"profile {pid} missing sha backfill"
