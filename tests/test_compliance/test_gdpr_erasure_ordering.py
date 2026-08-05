# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Tests for two behaviors in GDPRCompliance.forget_profile:

1. When the DB wrapper exposes no ``db_path``, the context-cache purge must
   emit a WARNING rather than silently skipping — so operators can detect the
   gap in erasure coverage.

2. The context-cache purge must execute BEFORE the main-DB profile row is
   deleted.  That ordering makes the erase crash-recoverable: a crash between
   the two steps leaves the profile present in the main DB, so a retry of
   forget_profile can reach and complete the erasure.  The reverse order would
   leave cache PII permanently orphaned with no retry path.

All tests use real SQLite databases and real function implementations.
No MagicMock is used to hide the path under test.
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from superlocalmemory.compliance.gdpr import GDPRCompliance
from superlocalmemory.storage import schema as real_schema
from superlocalmemory.storage.database import DatabaseManager
from superlocalmemory.storage.models import AtomicFact, MemoryRecord


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_cache_db(path: Path, profile_id: str, n_rows: int = 3) -> None:
    """Seed a minimal active_brain_cache.db with context_entries rows."""
    conn = sqlite3.connect(str(path), isolation_level=None)
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS context_entries (
            profile_id  TEXT NOT NULL DEFAULT 'default',
            session_id  TEXT NOT NULL,
            topic_sig   TEXT NOT NULL,
            content     TEXT NOT NULL,
            fact_ids    TEXT NOT NULL,
            provenance  TEXT NOT NULL DEFAULT 'tool_observation',
            computed_at INTEGER NOT NULL,
            byte_size   INTEGER NOT NULL,
            PRIMARY KEY (profile_id, session_id, topic_sig)
        ) WITHOUT ROWID;
        """
    )
    ts = int(time.time())
    for i in range(n_rows):
        conn.execute(
            "INSERT OR REPLACE INTO context_entries "
            "(profile_id, session_id, topic_sig, content, fact_ids, computed_at, byte_size) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (profile_id, f"sess_{i}", f"sig_{i}", f"pii_data_{i}", "[]", ts, 10),
        )
    conn.close()


def _count_cache_rows(path: Path, profile_id: str) -> int:
    """Return the number of context_entries rows for profile_id in the cache DB."""
    if not path.exists():
        return 0
    try:
        conn = sqlite3.connect(str(path))
        try:
            count = conn.execute(
                "SELECT COUNT(*) FROM context_entries WHERE profile_id = ?",
                (profile_id,),
            ).fetchone()[0]
        finally:
            conn.close()
        return count
    except Exception:
        return 0


def _profile_exists_in_main_db(mgr: DatabaseManager, profile_id: str) -> bool:
    rows = mgr.execute(
        "SELECT 1 FROM profiles WHERE profile_id = ?", (profile_id,)
    )
    return bool(rows)


class _NoDbPathWrapper:
    """Thin delegation wrapper over a real DatabaseManager that intentionally
    does not expose the ``db_path`` attribute.

    This exercises the ``getattr(self._db, "db_path", None) is None`` branch
    in GDPRCompliance.forget_profile without using MagicMock — every call
    routes to the real DatabaseManager, so the actual SQL and schema logic
    is exercised throughout.
    """

    def __init__(self, real_mgr: DatabaseManager) -> None:
        self._real = real_mgr

    def execute(self, sql: str, params: tuple[Any, ...] = ()) -> list:
        return self._real.execute(sql, params)
    # No db_path attribute — getattr(..., None) returns None for this wrapper.


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def main_db(tmp_path: Path) -> DatabaseManager:
    db_path = tmp_path / "erasure_test.db"
    mgr = DatabaseManager(db_path)
    mgr.initialize(real_schema)
    return mgr


@pytest.fixture()
def seeded_db(main_db: DatabaseManager) -> DatabaseManager:
    """Main DB pre-populated with a non-default profile, memories, and facts."""
    main_db.execute(
        "INSERT OR IGNORE INTO profiles (profile_id, name) VALUES ('alice', 'Alice')"
    )
    main_db.store_memory(MemoryRecord(
        memory_id="m1", profile_id="alice", content="Alice personal info"
    ))
    main_db.store_fact(AtomicFact(
        fact_id="f1", memory_id="m1", profile_id="alice", content="Alice fact"
    ))
    return main_db


# ---------------------------------------------------------------------------
# DEFECT 1 — observability when DB wrapper lacks db_path
# ---------------------------------------------------------------------------

class TestCachePurgeObservabilityWhenNoDbPath:
    """When the DB wrapper exposes no db_path, the erase path must emit at
    least one WARNING rather than silently skipping the cache purge."""

    def test_warning_emitted_when_fallback_root_lookup_also_fails(
        self,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Both db_path absent AND DEFAULT_BASE_DIR unavailable → 'skipped' WARNING."""
        db_path = tmp_path / "no_dbpath.db"
        real_mgr = DatabaseManager(db_path)
        real_mgr.initialize(real_schema)
        real_mgr.execute(
            "INSERT OR IGNORE INTO profiles (profile_id, name) VALUES ('bob', 'Bob')"
        )

        # Wrap in a no-db_path delegation object (real DB, no db_path attr).
        wrapper = _NoDbPathWrapper(real_mgr)
        gdpr = GDPRCompliance(wrapper)  # type: ignore[arg-type]

        # Patch DEFAULT_BASE_DIR to None so _Path(None) raises TypeError,
        # setting data_root = None, which fires the "purge skipped" warning.
        import superlocalmemory.core.config as _cfg_mod
        with (
            patch.object(_cfg_mod, "DEFAULT_BASE_DIR", None),
            caplog.at_level("WARNING", logger="superlocalmemory.compliance.gdpr"),
        ):
            gdpr.forget_profile("bob")

        warning_text = " ".join(caplog.messages)
        assert "bob" in warning_text, (
            "WARNING must identify the affected profile_id"
        )
        assert any(
            phrase in warning_text
            for phrase in (
                "context-cache purge skipped",
                "data root could not be resolved",
                "canonical root lookup failed",
            )
        ), f"Expected a 'purge skipped' WARNING; captured: {caplog.messages}"

    def test_default_root_is_never_used_for_destructive_fallback(
        self,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A wrapper without db_path must not erase another installation root."""
        db_path = tmp_path / "fallback_test.db"
        real_mgr = DatabaseManager(db_path)
        real_mgr.initialize(real_schema)
        real_mgr.execute(
            "INSERT OR IGNORE INTO profiles (profile_id, name) VALUES ('carol', 'Carol')"
        )

        # Seed a real cache file at the fallback location (tmp_path).
        cache_path = tmp_path / "active_brain_cache.db"
        _make_cache_db(cache_path, "carol", n_rows=4)
        assert _count_cache_rows(cache_path, "carol") == 4, (
            "pre-condition: cache must be seeded"
        )

        # Wrap the real DB in a no-db_path object so the fallback branch fires.
        wrapper = _NoDbPathWrapper(real_mgr)
        gdpr = GDPRCompliance(wrapper)  # type: ignore[arg-type]

        # Redirect DEFAULT_BASE_DIR to tmp_path so the purge finds the cache file.
        import superlocalmemory.core.config as _cfg_mod
        with (
            patch.object(_cfg_mod, "DEFAULT_BASE_DIR", tmp_path),
            caplog.at_level("WARNING", logger="superlocalmemory.compliance.gdpr"),
        ):
            gdpr.forget_profile("carol")

        # The default installation is not authoritative for this wrapper.
        assert _count_cache_rows(cache_path, "carol") == 4

        # A WARNING about using the fallback must have been emitted.
        warning_text = " ".join(caplog.messages)
        assert "carol" in warning_text, "WARNING must name the affected profile_id"
        assert "data root could not be resolved" in warning_text


# ---------------------------------------------------------------------------
# DEFECT 2 — ordering: cache purge must precede main-DB profile deletion
# ---------------------------------------------------------------------------

class TestCachePurgeOrderingBeforeMainDelete:
    """The context-cache purge must complete before the profile row is removed
    from the main DB — this ordering proves crash-recoverability."""

    def test_cache_purge_runs_before_profiles_delete(
        self,
        tmp_path: Path,
        seeded_db: DatabaseManager,
    ) -> None:
        """Record call order via real-object spies; assert cache-purge index
        is strictly less than profiles-delete index."""
        call_order: list[str] = []

        # Seed a real cache at the location derived from db_path.
        cache_at_db_dir = seeded_db.db_path.parent / "active_brain_cache.db"
        _make_cache_db(cache_at_db_dir, "alice", n_rows=2)

        # Spy wraps the real purge_profile_from_cache_db — records the call order
        # while still running the real implementation.
        from superlocalmemory.core import context_cache as _ctx_mod
        real_purge = _ctx_mod.purge_profile_from_cache_db

        def spy_purge(db_path: Path, profile_id: str) -> int:
            call_order.append("cache_purge")
            return real_purge(db_path, profile_id)

        # Spy on the DB execute method to detect "DELETE FROM profiles".
        real_execute = seeded_db.execute

        def spy_execute(sql: str, params: tuple[Any, ...] = ()) -> list:
            # Match the exact lowercase SQL used by gdpr.py — no .upper() needed.
            if "DELETE FROM profiles" in sql:
                call_order.append("profiles_delete")
            return real_execute(sql, params)

        seeded_db.execute = spy_execute  # type: ignore[method-assign]

        gdpr = GDPRCompliance(seeded_db)
        try:
            with patch(
                "superlocalmemory.core.context_cache.purge_profile_from_cache_db",
                side_effect=spy_purge,
            ):
                gdpr.forget_profile("alice")
        finally:
            seeded_db.execute = real_execute  # type: ignore[method-assign]

        assert "cache_purge" in call_order, (
            "purge_profile_from_cache_db was never called during forget_profile"
        )
        assert "profiles_delete" in call_order, (
            "DELETE FROM profiles was never observed during forget_profile"
        )
        cache_idx = call_order.index("cache_purge")
        delete_idx = call_order.index("profiles_delete")
        assert cache_idx < delete_idx, (
            f"Cache purge (position {cache_idx}) must precede profile deletion "
            f"(position {delete_idx}); full call_order={call_order}"
        )

    def test_both_cache_and_main_db_rows_cleared_after_erase(
        self,
        tmp_path: Path,
        seeded_db: DatabaseManager,
    ) -> None:
        """End-to-end: after forget_profile, both main-DB rows and cache rows
        for the erased profile are gone."""
        cache_path = seeded_db.db_path.parent / "active_brain_cache.db"
        _make_cache_db(cache_path, "alice", n_rows=3)

        assert _profile_exists_in_main_db(seeded_db, "alice"), (
            "pre-condition: profile must exist"
        )
        assert _count_cache_rows(cache_path, "alice") == 3, (
            "pre-condition: cache must be seeded"
        )

        gdpr = GDPRCompliance(seeded_db)
        result = gdpr.forget_profile("alice")

        # Profile row must be gone from the main DB.
        assert not _profile_exists_in_main_db(seeded_db, "alice"), (
            "Profile row must be deleted from the main DB"
        )

        # Memory rows must be gone.
        mem_rows = seeded_db.execute(
            "SELECT 1 FROM memories WHERE profile_id = ?", ("alice",)
        )
        assert not mem_rows, "Memory rows must be deleted from the main DB"

        # Cache rows must be gone.
        assert _count_cache_rows(cache_path, "alice") == 0, (
            "Cache rows must be purged during erasure"
        )

        # Return value must account for both the main-DB and the cache purge.
        assert result.get("profiles") == 1
        assert result.get("context_cache", 0) >= 1, (
            "Return value must report the count of cache rows deleted"
        )


class TestLearningDatabaseUsesActiveDataRoot:
    """Erasure must target sidecar databases next to the active memory DB."""

    def test_custom_root_learning_data_erased_without_touching_default_root(
        self,
        tmp_path: Path,
    ) -> None:
        from superlocalmemory.learning.database import LearningDatabase
        import superlocalmemory.core.config as config_module

        active_root = tmp_path / "active-root"
        default_root = tmp_path / "default-root"
        active_root.mkdir()
        default_root.mkdir()

        mgr = DatabaseManager(active_root / "memory.db")
        mgr.initialize(real_schema)
        mgr.execute(
            "INSERT OR IGNORE INTO profiles (profile_id, name) VALUES ('alice', 'Alice')"
        )

        active_learning = LearningDatabase(active_root / "learning.db")
        default_learning = LearningDatabase(default_root / "learning.db")
        active_learning.store_signal("alice", "q", "f1", "recall_hit")
        default_learning.store_signal("alice", "q", "f2", "recall_hit")

        with patch.object(config_module, "DEFAULT_BASE_DIR", default_root):
            result = GDPRCompliance(mgr).forget_profile("alice")

        assert result["erasure_complete"] == 1
        assert active_learning.get_signal_count("alice") == 0
        assert default_learning.get_signal_count("alice") == 1

    def test_cache_partial_failure_does_not_block_main_db_delete(
        self,
        tmp_path: Path,
        seeded_db: DatabaseManager,
    ) -> None:
        """If the cache purge raises unexpectedly, the main-DB deletion must
        still complete — the outer try/except in forget_profile must isolate
        the cache error from the critical delete path."""

        def raise_purge(*args: Any, **kwargs: Any) -> int:
            raise OSError("simulated cache I/O failure")

        gdpr = GDPRCompliance(seeded_db)
        with patch(
            "superlocalmemory.core.context_cache.purge_profile_from_cache_db",
            side_effect=raise_purge,
        ):
            result = gdpr.forget_profile("alice")

        # Main-DB profile row must be deleted even when the cache purge raises.
        assert not _profile_exists_in_main_db(seeded_db, "alice"), (
            "Main-DB profile row must be deleted even when the cache purge raises"
        )
        assert result.get("profiles") == 1
