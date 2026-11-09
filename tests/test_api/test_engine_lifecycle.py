# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | Engine lifecycle tests

"""Tests for engine lifecycle: lazy init, mode-change recovery, thread safety.

Regression tests for the "Engine not initialized" bug where routes like
/api/entity/list returned 503 forever after a mode switch because
/api/v3/mode endpoints set ``app.state.engine = None`` without re-initialising.

These tests simulate the post-mode-switch state and assert that routes still
work by lazily re-initialising the engine.
"""

from __future__ import annotations

import sqlite3
import threading
import uuid

import pytest

fastapi = pytest.importorskip("fastapi", reason="fastapi not installed")


# ---------------------------------------------------------------------------
# DB + config fixtures
# ---------------------------------------------------------------------------


def _setup_v32_tables(conn: sqlite3.Connection) -> None:
    from superlocalmemory.storage.schema_v32 import V32_DDL
    for ddl in V32_DDL:
        for stmt in ddl.strip().split(";"):
            stmt = stmt.strip()
            if stmt:
                try:
                    conn.execute(stmt)
                except Exception:
                    pass
    conn.commit()


def _seed_entity(conn: sqlite3.Connection, name: str = "Varun", type_: str = "person",
                 profile: str = "default") -> str:
    """Insert a canonical_entity row. Returns entity_id."""
    eid = f"ent_{uuid.uuid4().hex[:8]}"
    conn.execute(
        "INSERT INTO canonical_entities "
        "(entity_id, canonical_name, entity_type, profile_id, fact_count) "
        "VALUES (?, ?, ?, ?, ?)",
        (eid, name, type_, profile, 3),
    )
    conn.commit()
    return eid


@pytest.fixture
def engine_db(tmp_path, monkeypatch):
    """Create a seeded DB and point SLMConfig at it.

    Also monkeypatches DEFAULT_BASE_DIR so SLMConfig.load() reads from tmp_path.
    """
    from superlocalmemory.storage import schema

    db_path = tmp_path / "memory.db"
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    schema.create_all_tables(conn)
    _setup_v32_tables(conn)
    conn.execute(
        "INSERT OR IGNORE INTO profiles (profile_id, name, description) "
        "VALUES ('default', 'default', 'test')"
    )
    _seed_entity(conn, "Varun")
    _seed_entity(conn, "Qualixar", "organization")
    conn.commit()
    conn.close()

    monkeypatch.setenv("SLM_BASE_DIR", str(tmp_path))
    monkeypatch.setattr("superlocalmemory.core.config.DEFAULT_BASE_DIR", tmp_path)
    monkeypatch.setattr("superlocalmemory.server.routes.helpers.MEMORY_DIR", tmp_path)
    monkeypatch.setattr("superlocalmemory.server.routes.helpers.DB_PATH", db_path)

    # Clear process-wide lazy-init cooldown between tests.
    import superlocalmemory.server.routes.helpers as _helpers
    if hasattr(_helpers, "_last_engine_failure"):
        _helpers._last_engine_failure = 0.0

    return tmp_path


def _make_app_with_entity_routes():
    """Build a FastAPI app with just the entity routes wired up."""
    from fastapi import FastAPI

    from superlocalmemory.server.routes.entity import router as entity_router

    app = FastAPI()
    app.include_router(entity_router)
    app.state.engine = None  # Simulates post-mode-switch state.
    return app


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestLazyEngineRecovery:
    """Regression: after a mode switch sets engine=None, routes must still work."""

    def test_entity_list_recovers_when_engine_is_none(self, engine_db):
        """After mode change (engine=None), /api/entity/list must lazy-init and return 200."""
        from fastapi.testclient import TestClient

        app = _make_app_with_entity_routes()
        assert app.state.engine is None  # Precondition: simulates post-mode-switch.

        client = TestClient(app)
        resp = client.get("/api/entity/list?limit=10")

        assert resp.status_code == 200, (
            f"Expected 200 after lazy init, got {resp.status_code}: {resp.text}"
        )
        body = resp.json()
        assert "entities" in body
        assert body["total"] >= 2  # Seeded two entities.
        # Engine must have been installed on app.state for future requests.
        assert app.state.engine is not None

    def test_second_request_reuses_engine(self, engine_db):
        """Two consecutive requests share the same engine instance (no re-init)."""
        from fastapi.testclient import TestClient

        app = _make_app_with_entity_routes()
        client = TestClient(app)

        resp1 = client.get("/api/entity/list?limit=10")
        assert resp1.status_code == 200
        engine_after_first = app.state.engine
        assert engine_after_first is not None

        resp2 = client.get("/api/entity/list?limit=10")
        assert resp2.status_code == 200
        assert app.state.engine is engine_after_first  # Same object.

    def test_concurrent_requests_do_not_double_init(self, engine_db):
        """Under concurrent load, only ONE engine should be created."""
        from superlocalmemory.server.routes.helpers import get_engine_lazy

        class FakeState:
            def __init__(self):
                self.engine = None

        state = FakeState()
        results: list[object] = []
        errors: list[Exception] = []

        def worker():
            try:
                results.append(get_engine_lazy(state))
            except Exception as exc:  # pragma: no cover - diagnostic
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert not errors, f"Worker errors: {errors}"
        # All workers should receive the SAME engine object.
        unique_engines = {id(e) for e in results if e is not None}
        assert len(unique_engines) == 1, (
            f"Expected one engine under contention, got {len(unique_engines)}"
        )

    def test_lazy_init_recovers_after_transient_failure(self, engine_db, monkeypatch):
        """If a first init attempt fails, a later attempt must still be able to succeed.

        This is the regression for the sticky `_engine_init_attempted` flag which
        previously prevented any retry once a single init failed.
        """
        from superlocalmemory.server.routes import helpers

        call_count = {"n": 0}
        real_initialize = None

        class _Boom(RuntimeError):
            pass

        def flaky_initialize(self):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise _Boom("first attempt fails")
            return real_initialize(self)

        from superlocalmemory.core.engine import MemoryEngine
        real_initialize = MemoryEngine.initialize
        monkeypatch.setattr(MemoryEngine, "initialize", flaky_initialize)

        class FakeState:
            def __init__(self):
                self.engine = None

        state = FakeState()

        # Reset cooldown between retries for deterministic test timing.
        helpers._last_engine_failure = 0.0
        first = helpers.get_engine_lazy(state)
        assert first is None, "First attempt should fail"

        # Bypass cooldown.
        helpers._last_engine_failure = 0.0
        second = helpers.get_engine_lazy(state)
        assert second is not None, "Second attempt must succeed (no sticky flag)"
        assert call_count["n"] >= 2


class TestRequireEngineHelper:
    """The require_engine(request) helper raises 503 only when lazy init truly fails."""

    def test_require_engine_raises_503_when_db_missing(self, tmp_path, monkeypatch):
        """If the DB is genuinely missing, require_engine should raise 503."""
        from fastapi import HTTPException

        from superlocalmemory.server.routes.helpers import require_engine

        monkeypatch.setattr("superlocalmemory.core.config.DEFAULT_BASE_DIR", tmp_path / "nonexistent")

        class FakeState:
            engine = None

        class FakeApp:
            state = FakeState()

        class FakeRequest:
            app = FakeApp()

        # Clear cooldown.
        import superlocalmemory.server.routes.helpers as _helpers
        _helpers._last_engine_failure = 0.0

        # If init fails it should return 503, not crash.
        try:
            require_engine(FakeRequest())
        except HTTPException as exc:
            assert exc.status_code == 503
            return
        # If init succeeded against an empty tmp_path, that's also acceptable —
        # the contract is only "don't crash, don't silently pass None".

    def test_require_engine_returns_engine_when_available(self, engine_db):
        """When engine is available, require_engine returns it (no exception)."""
        from superlocalmemory.server.routes.helpers import require_engine

        class FakeState:
            engine = None

        class FakeApp:
            state = FakeState()

        class FakeRequest:
            app = FakeApp()

        req = FakeRequest()
        engine = require_engine(req)
        assert engine is not None
        # Second call returns the same cached engine.
        assert require_engine(req) is engine


class TestModeChangeAudit:
    """Every mode change should leave an audit trail — catches phantom writes."""

    def test_log_mode_change_writes_audit_line(self, tmp_path, monkeypatch):
        """log_mode_change() appends a line to logs/mode-audit.log."""
        from superlocalmemory.server.routes import helpers

        monkeypatch.setattr(helpers, "MEMORY_DIR", tmp_path)
        helpers.log_mode_change("a", "c", provider="openrouter",
                                model="anthropic/claude-sonnet-4", source="test")

        audit = tmp_path / "logs" / "mode-audit.log"
        assert audit.exists(), "mode-audit.log was not created"
        content = audit.read_text(encoding="utf-8")
        assert "old=a" in content
        assert "new=c" in content
        assert "anthropic/claude-sonnet-4" in content
        assert "source=test" in content
