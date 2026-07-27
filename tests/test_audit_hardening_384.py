# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com
"""Audit-hardening tests for 3.8.4 stability release.

Covers the 5 fixes from the iron-pattern Stage 8/9 audit:
  FIX-1  (H-1)  — /health handler NameError on _TEST_ISOLATION_ALLOWED
  FIX-2  (C1)   — warm-embed pool TOCTOU race on concurrent first-store_fast()
  FIX-3  (H2+stats) — prune_events() bounded batches + correct non-zero stats
  FIX-4  (L1)   — SLM_STORE_FAST_EMBED_TIMEOUT_MS non-int env var raises ValueError
  FIX-5  (M-2)  — mesh peer ::ffff:127.0.0.1 mis-categorised as remote
"""

from __future__ import annotations

import asyncio
import os
import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ===========================================================================
# FIX-1: /health handler must not raise NameError for _TEST_ISOLATION_ALLOWED
# ===========================================================================

class TestFix1HealthNameError:
    """GET /health via TestClient must not raise NameError.

    Before the fix, _TEST_ISOLATION_ALLOWED was referenced without being
    imported into unified_daemon.py, causing NameError on the testclient path.
    """

    def test_health_testclient_no_name_error(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        """GET /health with SLM_TEST_ISOLATION=1 must not raise NameError."""
        from fastapi.testclient import TestClient

        monkeypatch.setenv("SLM_DATA_DIR", str(tmp_path))
        monkeypatch.setenv("SLM_TEST_ISOLATION", "1")
        monkeypatch.setenv("SLM_DAEMON_PORT", "0")

        from superlocalmemory.server import unified_daemon
        monkeypatch.setattr(unified_daemon, "_ACTIVE_DAEMON_DESCRIPTOR", None)

        app = unified_daemon.create_app()

        # TestClient sets client.host = "testclient".
        # Before the fix this raises NameError; after the fix it returns 200.
        client = TestClient(app, raise_server_exceptions=True)
        resp = client.get("/health")
        # 200 means no NameError — the /health handler completed successfully.
        assert resp.status_code == 200, (
            f"Expected 200 from /health, got {resp.status_code}: {resp.text}"
        )

    def test_health_request_none_path_still_works(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        """Calling the endpoint directly with request=None (internal/test path)
        still succeeds — this is the trusted path and must not regress."""
        monkeypatch.setenv("SLM_DATA_DIR", str(tmp_path))
        monkeypatch.setenv("SLM_TEST_ISOLATION", "1")
        monkeypatch.setenv("SLM_DAEMON_PORT", "0")

        from superlocalmemory.server import unified_daemon
        monkeypatch.setattr(unified_daemon, "_ACTIVE_DAEMON_DESCRIPTOR", None)

        app = unified_daemon.create_app()
        route = next(
            r for r in app.routes if getattr(r, "path", None) == "/health"
        )
        # request=None → trusted path; must return full payload, no NameError.
        payload = asyncio.run(route.endpoint())
        assert payload["status"] == "ok"


# ===========================================================================
# FIX-2: Warm-embed pool TOCTOU — exactly ONE pool under concurrent init
# ===========================================================================

class TestFix2EmbedPoolRace:
    """Concurrent first-calls to the warm-guard pool init must produce exactly
    one ThreadPoolExecutor (no orphaned second pool)."""

    def _make_engine(self, tmp_path: Path):
        from superlocalmemory.core.config import SLMConfig
        from superlocalmemory.core.engine import MemoryEngine
        from superlocalmemory.storage.models import Mode

        cfg = SLMConfig.for_mode(Mode.B, base_dir=tmp_path)
        engine = MemoryEngine(cfg)
        engine._require_full = lambda _: None
        engine._ensure_init()
        return engine

    def test_concurrent_pool_init_creates_one_executor(
        self, tmp_path: Path,
    ) -> None:
        """N threads race to init the pool on first store_fast() call.

        After all threads finish, the pool must be a single ThreadPoolExecutor
        (identity check). No orphan pools, no deadlock.
        """
        import concurrent.futures as _cf

        engine = self._make_engine(tmp_path)
        # Start with pool=None to force concurrent init
        engine._store_fast_embed_pool = None

        # Warm embedder that is "available" instantly
        embedder = MagicMock()
        embedder._available = True
        embedder._config = None
        embedder.embed.return_value = [0.1, 0.2, 0.3]
        embedder.compute_fisher_params.return_value = ([0.1], [0.9])
        engine._embedder = embedder

        N = 10
        pools_seen: list = []
        barrier = threading.Barrier(N)
        errors: list = []

        def _race_init():
            barrier.wait()  # all threads start simultaneously
            try:
                engine.store_fast(f"concurrent content {threading.current_thread().name}")
                pools_seen.append(id(engine._store_fast_embed_pool))
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=_race_init) for _ in range(N)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert not errors, f"Threads raised exceptions: {errors}"

        pool = engine._store_fast_embed_pool
        assert pool is not None, "Pool must have been created"
        assert isinstance(pool, _cf.ThreadPoolExecutor), (
            "Pool must be a ThreadPoolExecutor"
        )
        # All threads must see the same pool identity
        unique_pool_ids = set(pools_seen)
        assert len(unique_pool_ids) == 1, (
            f"Expected 1 unique pool, found {len(unique_pool_ids)}: {unique_pool_ids} "
            "(TOCTOU race: multiple pools were created)"
        )

    def test_lock_attribute_exists_on_engine(self, tmp_path: Path) -> None:
        """The pool guard is a non-reentrant lock that protects first init."""
        engine = self._make_engine(tmp_path)
        assert hasattr(engine, "_store_fast_embed_pool_lock"), (
            "MemoryEngine must have _store_fast_embed_pool_lock"
        )
        lock = engine._store_fast_embed_pool_lock
        # ``threading.Lock`` is a factory on supported CPython versions, not a
        # class suitable for ``isinstance``.  Test the safety property instead.
        assert lock.acquire(blocking=False)
        try:
            assert lock.acquire(blocking=False) is False
        finally:
            lock.release()


# ===========================================================================
# FIX-3: prune_events() bounded batches + correct non-zero stats (db path)
# ===========================================================================

class TestFix3PruneEventsBounded:
    """prune_events() via DatabaseManager path must:
      - return non-zero counts for matched rows
      - delete rows in bounded LIMIT batches (not one unbounded txn)
      - not hold the write lock for the entire dataset at once
    """

    def _seed_events(
        self,
        bus,
        *,
        n_hot_old: int = 5,
        n_warm_old: int = 5,
        n_archive_old: int = 5,
    ) -> None:
        """Insert events using the EventBus schema (created by _init_schema)."""
        conn = sqlite3.connect(str(bus.db_path))
        conn.execute("PRAGMA busy_timeout=5000")
        now = datetime.now(timezone.utc)

        # hot→warm: tier='hot', old enough, importance < 5
        hot_ts = (now - timedelta(hours=60)).isoformat()
        for i in range(n_hot_old):
            conn.execute(
                "INSERT INTO memory_events (event_type, tier, importance, created_at) "
                "VALUES (?, ?, ?, ?)",
                ("memory.stored", "hot", 1, hot_ts),
            )

        # warm→cold: tier='warm', old enough
        warm_ts = (now - timedelta(days=20)).isoformat()
        for i in range(n_warm_old):
            conn.execute(
                "INSERT INTO memory_events (event_type, tier, importance, created_at) "
                "VALUES (?, ?, ?, ?)",
                ("memory.recalled", "warm", 3, warm_ts),
            )

        # archive: very old (beyond cold_hours)
        archive_ts = (now - timedelta(days=40)).isoformat()
        for i in range(n_archive_old):
            conn.execute(
                "INSERT INTO memory_events (event_type, tier, importance, created_at) "
                "VALUES (?, ?, ?, ?)",
                ("memory.deleted", "cold", 2, archive_ts),
            )

        conn.commit()
        conn.close()

    def test_prune_via_db_path_returns_nonzero_counts(
        self, tmp_path: Path,
    ) -> None:
        """With rows matching all three prune conditions, stats must be non-zero."""
        db_path = tmp_path / "events.db"

        from superlocalmemory.infra.event_bus import EventBus
        from superlocalmemory.storage.database import DatabaseManager

        EventBus.reset_instance(db_path)
        bus = EventBus.get_instance(db_path)
        # Seed after schema init (EventBus._init_schema() already ran)
        self._seed_events(bus, n_hot_old=3, n_warm_old=4, n_archive_old=2)
        # Wire the DatabaseManager so we exercise the db path
        bus._db = DatabaseManager(db_path)

        stats = bus.prune_events(hot_hours=48, warm_hours=14 * 24, cold_hours=30 * 24)

        assert "error" not in stats, f"prune_events returned error: {stats}"
        assert stats["hot_to_warm"] >= 3, (
            f"Expected ≥3 hot→warm transitions, got {stats['hot_to_warm']}"
        )
        assert stats["warm_to_cold"] >= 4, (
            f"Expected ≥4 warm→cold deletions, got {stats['warm_to_cold']}"
        )
        assert stats["archived"] >= 2, (
            f"Expected ≥2 archive deletions, got {stats['archived']}"
        )

    def test_prune_via_fallback_path_returns_nonzero_counts(
        self, tmp_path: Path,
    ) -> None:
        """Fallback (direct connection) path must also return non-zero counts.
        This path was already correct; ensure we haven't broken it."""
        db_path = tmp_path / "events_fb.db"

        from superlocalmemory.infra.event_bus import EventBus

        EventBus.reset_instance(db_path)
        bus = EventBus.get_instance(db_path)
        self._seed_events(bus, n_hot_old=2, n_warm_old=3, n_archive_old=1)
        bus._db = None  # force fallback path

        stats = bus.prune_events(hot_hours=48, warm_hours=14 * 24, cold_hours=30 * 24)

        assert "error" not in stats, f"prune_events fallback returned error: {stats}"
        assert stats["warm_to_cold"] >= 3
        assert stats["archived"] >= 1

    def test_prune_zero_rows_returns_zero_stats(self, tmp_path: Path) -> None:
        """Empty table → stats must be all zeros (no KeyError, no crash)."""
        db_path = tmp_path / "events_empty.db"

        from superlocalmemory.infra.event_bus import EventBus
        from superlocalmemory.storage.database import DatabaseManager

        # Create bus (creates schema), don't seed any rows
        EventBus.reset_instance(db_path)
        bus = EventBus.get_instance(db_path)
        bus._db = DatabaseManager(db_path)

        stats = bus.prune_events(hot_hours=48, warm_hours=14 * 24, cold_hours=30 * 24)

        assert "error" not in stats, f"Empty-table prune returned error: {stats}"
        assert stats["hot_to_warm"] == 0
        assert stats["warm_to_cold"] == 0
        assert stats["archived"] == 0


# ===========================================================================
# FIX-4: Non-int SLM_STORE_FAST_EMBED_TIMEOUT_MS must not crash store_fast
# ===========================================================================

class TestFix4EnvVarGuard:
    """SLM_STORE_FAST_EMBED_TIMEOUT_MS='500ms' (non-int) must not raise
    ValueError and must fall back to the 500ms default."""

    def _make_engine(self, tmp_path: Path):
        from superlocalmemory.core.config import SLMConfig
        from superlocalmemory.core.engine import MemoryEngine
        from superlocalmemory.storage.models import Mode

        cfg = SLMConfig.for_mode(Mode.B, base_dir=tmp_path)
        engine = MemoryEngine(cfg)
        engine._require_full = lambda _: None
        engine._ensure_init()
        return engine

    @pytest.mark.parametrize("bad_value", [
        "500ms", "abc", "0.5", "500 ms", "", "None",
    ])
    def test_non_int_env_var_does_not_crash_store_fast(
        self, tmp_path: Path, bad_value: str,
    ) -> None:
        """store_fast() must survive malformed SLM_STORE_FAST_EMBED_TIMEOUT_MS."""
        engine = self._make_engine(tmp_path)
        embedder = MagicMock()
        embedder._available = True
        embedder._config = None
        embedder.embed.return_value = [0.1, 0.2]
        embedder.compute_fisher_params.return_value = ([0.1], [0.9])
        engine._embedder = embedder

        with patch.dict(os.environ, {"SLM_STORE_FAST_EMBED_TIMEOUT_MS": bad_value}):
            # Must not raise — should fall back to default 500ms
            fact_ids = engine.store_fast("content with bad env var")

        assert fact_ids, "store_fast must return fact_ids even with bad env var"

    def test_valid_int_env_var_still_works(self, tmp_path: Path) -> None:
        """Normal integer value must still work correctly."""
        engine = self._make_engine(tmp_path)
        embedder = MagicMock()
        embedder._available = True
        embedder._config = None
        embedder.embed.return_value = [0.1, 0.2]
        embedder.compute_fisher_params.return_value = ([0.1], [0.9])
        engine._embedder = embedder

        with patch.dict(os.environ, {"SLM_STORE_FAST_EMBED_TIMEOUT_MS": "200"}):
            fact_ids = engine.store_fast("content with valid env var")

        assert fact_ids, "store_fast must work with valid integer env var"


# ===========================================================================
# FIX-5: Mesh peer display — ::ffff:127.0.0.1 must categorise as local
# ===========================================================================

class TestFix5MeshLoopbackDisplay:
    """_mesh_read_model() must categorise IPv4-mapped loopback as local.

    Before the fix, only the literal set {"127.0.0.1", "::1", "localhost"} was
    checked, so ::ffff:127.0.0.1 fell through to remote. After the fix,
    is_loopback() handles IPv4-mapped forms.
    """

    def _make_peer(self, host: str) -> dict:
        """Minimal peer record with a recent heartbeat."""
        now = datetime.now(timezone.utc)
        return {
            "host": host,
            "session_id": "test-session",
            "last_heartbeat": (now - timedelta(seconds=10)).isoformat(),
        }

    def test_ipv4_mapped_loopback_categorised_as_local(self) -> None:
        """::ffff:127.0.0.1 must appear in the local list, not remote."""
        from superlocalmemory.server.routes.mesh import _mesh_read_model

        records = [self._make_peer("::ffff:127.0.0.1")]
        remote, local = _mesh_read_model(records)

        assert len(local) == 1, (
            f"::ffff:127.0.0.1 must be categorised as local, "
            f"but was in remote: {remote}"
        )
        assert len(remote) == 0, (
            f"Remote list must be empty for loopback peer, got: {remote}"
        )

    @pytest.mark.parametrize("host", [
        "127.0.0.1",
        "::1",
        "localhost",
        "::ffff:127.0.0.1",
        "::ffff:127.0.0.2",
    ])
    def test_all_loopback_forms_categorised_as_local(self, host: str) -> None:
        """All standard loopback variants must land in the local bucket."""
        from superlocalmemory.server.routes.mesh import _mesh_read_model

        records = [self._make_peer(host)]
        remote, local = _mesh_read_model(records)

        assert len(local) == 1, (
            f"Host '{host}' must be categorised as local, but remote={remote}"
        )

    def test_remote_host_still_categorised_as_remote(self) -> None:
        """A genuine remote IP must remain in the remote bucket."""
        from superlocalmemory.server.routes.mesh import _mesh_read_model

        records = [self._make_peer("10.0.0.5")]
        remote, local = _mesh_read_model(records)

        assert len(remote) == 1, (
            f"10.0.0.5 must be categorised as remote, but local={local}"
        )
        assert len(local) == 0

    def test_display_categorisation_uses_is_loopback_not_literal_set(
        self,
    ) -> None:
        """Confirm the fix: _LOOPBACK_HOSTS literal set is NOT used for
        classification (it would miss ::ffff:127.0.0.1). The presence of
        ::ffff:127.0.0.1 in local proves is_loopback() is being called."""
        from superlocalmemory.server.routes.mesh import _LOOPBACK_HOSTS, _mesh_read_model

        # Confirm the literal set does NOT contain the mapped form
        assert "::ffff:127.0.0.1" not in _LOOPBACK_HOSTS, (
            "_LOOPBACK_HOSTS should not contain ::ffff:127.0.0.1 "
            "(it's the old set we replaced)"
        )

        # But the function must still categorise it as local
        records = [self._make_peer("::ffff:127.0.0.1")]
        remote, local = _mesh_read_model(records)
        assert len(local) == 1, (
            "::ffff:127.0.0.1 classified as remote — is_loopback() not being used"
        )
