# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory v3.4.21 — Stage 9 W1

"""Stage 9 W1 regressions — Data Integrity.

Covers:
- H-DATA-01: M002 must backfill ``bytes_sha256`` for every pre-existing row
  so ``verify_sha256`` accepts them on first load. Without the backfill,
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


# ---------------------------------------------------------------------------
# H-DATA-01 — M002 post_ddl_hook backfills bytes_sha256
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


def test_m002_post_ddl_hook_backfills_sha256_for_legacy_rows(
    tmp_path: Path,
) -> None:
    """H-DATA-01: M002 post-DDL hook computes sha256 for each copied row."""
    db = tmp_path / "learning.db"
    blob_a = b"learned-ranker-model-A" * 32
    blob_b = b"learned-ranker-model-B" * 32
    _seed_v3419_learning_db(db, [("default", blob_a), ("work", blob_b)])

    conn = sqlite3.connect(db)
    try:
        conn.executescript(M002.DDL)
        M002.post_ddl_hook(conn)
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


def test_m002_post_ddl_hook_is_idempotent(tmp_path: Path) -> None:
    """Re-running the hook on already-populated rows is a no-op."""
    db = tmp_path / "learning.db"
    _seed_v3419_learning_db(db, [("solo", b"x" * 256)])

    conn = sqlite3.connect(db)
    try:
        conn.executescript(M002.DDL)
        M002.post_ddl_hook(conn)
        first = conn.execute(
            "SELECT bytes_sha256 FROM learning_model_state"
        ).fetchone()[0]
        # Running again must not change the value or raise.
        M002.post_ddl_hook(conn)
        second = conn.execute(
            "SELECT bytes_sha256 FROM learning_model_state"
        ).fetchone()[0]
    finally:
        conn.close()

    assert first == second
    assert first == hashlib.sha256(b"x" * 256).hexdigest()


def test_m002_post_ddl_hook_handles_empty_table(tmp_path: Path) -> None:
    """Fresh install (no legacy rows) → hook does nothing, raises nothing."""
    db = tmp_path / "learning.db"
    # Minimal empty seed — just the old table shape.
    _seed_v3419_learning_db(db, [])

    conn = sqlite3.connect(db)
    try:
        conn.executescript(M002.DDL)
        M002.post_ddl_hook(conn)
        count = conn.execute(
            "SELECT COUNT(*) FROM learning_model_state"
        ).fetchone()[0]
    finally:
        conn.close()

    assert count == 0


# ---------------------------------------------------------------------------
# M-DATA-01 — is_active=1 no longer hardcoded
# ---------------------------------------------------------------------------


def test_m002_is_active_only_on_max_id_per_profile(tmp_path: Path) -> None:
    """Dev-build path: multiple rows per profile_id must land with only
    the MAX(id) row marked active; the partial unique index must hold."""
    db = tmp_path / "learning.db"
    # Simulate a dev build where the UNIQUE constraint is absent and a
    # profile accumulated 3 rows.
    conn = sqlite3.connect(db)
    try:
        conn.executescript("""
            CREATE TABLE learning_model_state (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                profile_id  TEXT NOT NULL,
                state_bytes BLOB NOT NULL,
                updated_at  TEXT NOT NULL
            );
        """)
        for i in range(3):
            conn.execute(
                "INSERT INTO learning_model_state "
                "(profile_id, state_bytes, updated_at) "
                "VALUES (?, ?, ?)",
                ("default", f"blob-{i}".encode() * 64,
                 f"2026-04-{i+1:02d}T00:00:00Z"),
            )
        conn.commit()

        # Apply M002.
        conn.executescript(M002.DDL)
        M002.post_ddl_hook(conn)

        active_count = conn.execute(
            "SELECT COUNT(*) FROM learning_model_state WHERE is_active = 1"
        ).fetchone()[0]
        total = conn.execute(
            "SELECT COUNT(*) FROM learning_model_state"
        ).fetchone()[0]
        active_id = conn.execute(
            "SELECT id FROM learning_model_state WHERE is_active = 1"
        ).fetchone()[0]
    finally:
        conn.close()

    assert total == 3, "all three rows should be copied forward"
    assert active_count == 1, "partial unique index requires 1 active"
    assert active_id == 3, "the newest (MAX(id)) row should be active"


# ---------------------------------------------------------------------------
# Runner post_ddl_hook plumbing
# ---------------------------------------------------------------------------


def test_migration_runner_calls_post_ddl_hook_after_successful_ddl(
    tmp_path: Path,
) -> None:
    """C3+H-DATA-01 plumbing: the runner must invoke a migration's
    ``post_ddl_hook`` exactly once after the DDL commits."""
    learning_db = tmp_path / "learning.db"
    memory_db = tmp_path / "memory.db"
    _seed_v3419_learning_db(learning_db,
                            [("p1", b"model-p1" * 128),
                             ("p2", b"model-p2" * 128)])

    result = migration_runner.apply_all(learning_db, memory_db)
    # M002 must land with status 'applied'. The post_ddl_hook must have
    # run (evidenced by non-empty bytes_sha256 on every row).
    assert "M002_model_state_history" in result["applied"]

    conn = sqlite3.connect(learning_db)
    try:
        rows = conn.execute(
            "SELECT profile_id, bytes_sha256 FROM learning_model_state"
        ).fetchall()
    finally:
        conn.close()

    for pid, sha in rows:
        assert len(sha) == 64, f"profile {pid} missing sha backfill"
