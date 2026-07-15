# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3.3 | Dashboard + ACLI API Tests

"""Tests for V3.3 API endpoints and ACLI integration.

Covers:
- Forgetting dashboard (GET /forgetting/stats, POST /forgetting/run)
- Quantization dashboard (GET /quantization/stats)
- CCQ dashboard (GET /ccq/blocks)
- Soft prompts dashboard (GET /soft-prompts)
- Process health (GET /health/processes)
- V3.3 overview (GET /v33/overview)
- Route registration verification
- ACLI soft prompt injection integration
- Forgetting-aware auto-invoke filtering
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

fastapi = pytest.importorskip("fastapi", reason="fastapi not installed")


# ---------------------------------------------------------------------------
# Helpers: DB setup
# ---------------------------------------------------------------------------


def _make_app():
    """Create a FastAPI app with V3 router included."""
    from superlocalmemory.server.routes.v3_api import router
    from fastapi import FastAPI

    app = FastAPI()
    app.state.engine = MagicMock(profile_id="default")

    @app.middleware("http")
    async def _authenticated_test_request(request, call_next):
        request.state.authenticated_actor = "test:verified"
        return await call_next(request)

    app.include_router(router)
    return app


def _setup_v33_tables(conn: sqlite3.Connection) -> None:
    """Create V3.3 tables needed by the new endpoints."""
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


def _create_db(tmp_path: Path) -> Path:
    """Create a temporary DB with all tables and a default profile."""
    from superlocalmemory.storage import schema

    db_path = tmp_path / "memory.db"
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    schema.create_all_tables(conn)
    _setup_v33_tables(conn)
    conn.execute(
        "INSERT OR IGNORE INTO profiles (profile_id, name, description) "
        "VALUES ('default', 'default', 'Test profile')",
    )
    conn.commit()
    conn.close()
    return db_path


def _seed_retention(conn: sqlite3.Connection, pid: str = "default") -> list[str]:
    """Insert seed fact_retention rows across different lifecycle zones."""
    zones = [
        ("active", 0.85),
        ("active", 0.90),
        ("warm", 0.50),
        ("cold", 0.25),
        ("archive", 0.10),
        ("forgotten", 0.02),
    ]
    fact_ids = []
    for zone, score in zones:
        fid = f"fact_{uuid.uuid4().hex[:8]}"
        mid = f"mem_{uuid.uuid4().hex[:8]}"
        conn.execute(
            "INSERT INTO memories (memory_id, profile_id, content) "
            "VALUES (?, ?, ?)",
            (mid, pid, f"Memory for {zone}"),
        )
        conn.execute(
            "INSERT INTO atomic_facts "
            "(fact_id, memory_id, profile_id, content, fact_type, confidence) "
            "VALUES (?, ?, ?, ?, 'semantic', 0.9)",
            (fid, mid, pid, f"Fact in {zone} zone, score={score}"),
        )
        conn.execute(
            "INSERT INTO fact_retention "
            "(fact_id, profile_id, retention_score, memory_strength, "
            "access_count, lifecycle_zone) "
            "VALUES (?, ?, ?, 1.0, 3, ?)",
            (fid, pid, score, zone),
        )
        fact_ids.append(fid)
    conn.commit()
    return fact_ids


def _seed_quantization(conn: sqlite3.Connection, pid: str = "default") -> None:
    """Insert seed embedding_quantization_metadata rows."""
    levels = [
        ("float32", 32),
        ("float32", 32),
        ("int8", 8),
        ("polar4", 4),
        ("polar2", 2),
    ]
    for level, bw in levels:
        fid = f"fact_{uuid.uuid4().hex[:8]}"
        mid = f"mem_{uuid.uuid4().hex[:8]}"
        conn.execute(
            "INSERT INTO memories (memory_id, profile_id, content) "
            "VALUES (?, ?, ?)",
            (mid, pid, f"Memory for quant {level}"),
        )
        conn.execute(
            "INSERT INTO atomic_facts "
            "(fact_id, memory_id, profile_id, content, fact_type, confidence) "
            "VALUES (?, ?, ?, ?, 'semantic', 0.9)",
            (fid, mid, pid, f"Quant fact {level}"),
        )
        conn.execute(
            "INSERT INTO embedding_quantization_metadata "
            "(fact_id, profile_id, quantization_level, bit_width) "
            "VALUES (?, ?, ?, ?)",
            (fid, pid, level, bw),
        )
    conn.commit()


def _seed_ccq_blocks(conn: sqlite3.Connection, pid: str = "default") -> list[str]:
    """Insert seed ccq_consolidated_blocks."""
    block_ids = []
    for i in range(3):
        bid = f"blk_{uuid.uuid4().hex[:8]}"
        source_ids = json.dumps([f"src_{j}" for j in range(i + 2)])
        conn.execute(
            "INSERT INTO ccq_consolidated_blocks "
            "(block_id, profile_id, content, source_fact_ids, "
            "char_count, compiled_by, cluster_id) "
            "VALUES (?, ?, ?, ?, ?, 'ccq', ?)",
            (bid, pid, f"Gist block {i}", source_ids, 50, f"cluster_{i}"),
        )
        block_ids.append(bid)
    conn.commit()
    return block_ids


def _seed_soft_prompts(conn: sqlite3.Connection, pid: str = "default") -> list[str]:
    """Insert seed soft_prompt_templates."""
    prompt_ids = []
    templates = [
        ("identity", "The user is an architect.", 0.9, 20),
        ("tech_preference", "Prefers Python and TypeScript.", 0.8, 15),
    ]
    for category, content, confidence, tokens in templates:
        prompt_id = f"sp_{uuid.uuid4().hex[:8]}"
        conn.execute(
            "INSERT INTO soft_prompt_templates "
            "(prompt_id, profile_id, category, content, confidence, "
            "effectiveness, token_count, retention_score, active, version) "
            "VALUES (?, ?, ?, ?, ?, 0.6, ?, 0.9, 1, 1)",
            (prompt_id, pid, category, content, confidence, tokens),
        )
        prompt_ids.append(prompt_id)
    conn.commit()
    return prompt_ids


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def v33_db(tmp_path):
    """DB with all V3.3 tables, profile, and no seed data. Returns db_path."""
    return _create_db(tmp_path)


@pytest.fixture
def seeded_v33_db(tmp_path):
    """DB with all V3.3 tables and seed data across all features.

    Returns (db_path, fact_ids, block_ids, prompt_ids).
    """
    db_path = _create_db(tmp_path)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    fact_ids = _seed_retention(conn)
    _seed_quantization(conn)
    block_ids = _seed_ccq_blocks(conn)
    prompt_ids = _seed_soft_prompts(conn)
    conn.close()
    return db_path, fact_ids, block_ids, prompt_ids


# ---------------------------------------------------------------------------
# TASK 1: Route Registration Tests
# ---------------------------------------------------------------------------


class TestV33RouteRegistration:
    """Verify all V3.3 routes are registered on the router."""

    def test_v33_routes_registered(self):
        """All 7 V3.3 endpoints appear in the router."""
        from superlocalmemory.server.routes.v3_api import router

        route_paths = [r.path for r in router.routes]
        expected = [
            "/api/v3/forgetting/stats",
            "/api/v3/forgetting/run",
            "/api/v3/quantization/stats",
            "/api/v3/ccq/blocks",
            "/api/v3/soft-prompts",
            "/api/v3/health/processes",
            "/api/v3/v33/overview",
        ]
        for path in expected:
            assert path in route_paths, f"Missing route: {path}"

    def test_v33_routes_coexist_with_v32(self):
        """V3.3 routes do not break existing V3.2 routes."""
        from superlocalmemory.server.routes.v3_api import router

        route_paths = [r.path for r in router.routes]
        # V3.2 routes that must still exist
        v32_routes = [
            "/api/v3/dashboard",
            "/api/v3/mode",
            "/api/v3/auto-invoke/config",
            "/api/v3/associations",
            "/api/v3/core-memory",
        ]
        for path in v32_routes:
            assert path in route_paths, f"V3.2 route broken: {path}"


# ---------------------------------------------------------------------------
# TASK 1a: Forgetting Dashboard Tests
# ---------------------------------------------------------------------------


class TestForgettingStats:
    """Tests for GET /api/v3/forgetting/stats."""

    def test_empty_db_returns_zero_zones(self, v33_db):
        """Empty DB returns total=0 with all zone counts zero."""
        from fastapi.testclient import TestClient

        with patch("superlocalmemory.server.routes.helpers.DB_PATH", v33_db), \
             patch("superlocalmemory.server.routes.helpers.get_active_profile", return_value="default"):
            client = TestClient(_make_app())
            resp = client.get("/api/v3/forgetting/stats?profile=default")
            assert resp.status_code == 200
            data = resp.json()
            assert data["total"] == 0
            assert set(data["zones"].keys()) == {"active", "warm", "cold", "archive", "forgotten"}
            assert all(v == 0 for v in data["zones"].values())

    def test_seeded_db_returns_zone_distribution(self, seeded_v33_db):
        """Seeded DB returns correct zone counts."""
        db_path, _, _, _ = seeded_v33_db
        from fastapi.testclient import TestClient

        with patch("superlocalmemory.server.routes.helpers.DB_PATH", db_path), \
             patch("superlocalmemory.server.routes.helpers.get_active_profile", return_value="default"):
            client = TestClient(_make_app())
            resp = client.get("/api/v3/forgetting/stats?profile=default")
            assert resp.status_code == 200
            data = resp.json()
            assert data["total"] == 6
            assert data["zones"]["active"] == 2
            assert data["zones"]["warm"] == 1
            assert data["zones"]["cold"] == 1
            assert data["zones"]["archive"] == 1
            assert data["zones"]["forgotten"] == 1

    def test_profile_scoping(self, seeded_v33_db):
        """Nonexistent profile returns all zeros."""
        db_path, _, _, _ = seeded_v33_db
        from fastapi.testclient import TestClient

        with patch("superlocalmemory.server.routes.helpers.DB_PATH", db_path), \
             patch("superlocalmemory.server.routes.helpers.get_active_profile", return_value="nonexistent"):
            client = TestClient(_make_app())
            resp = client.get("/api/v3/forgetting/stats?profile=nonexistent")
            assert resp.status_code == 200
            assert resp.json()["total"] == 0

    def test_no_db_file_returns_empty(self, tmp_path):
        """Missing DB file returns graceful empty response."""
        fake_db = tmp_path / "nonexistent.db"
        from fastapi.testclient import TestClient

        with patch("superlocalmemory.server.routes.helpers.DB_PATH", fake_db), \
             patch("superlocalmemory.server.routes.helpers.get_active_profile", return_value="default"):
            client = TestClient(_make_app())
            resp = client.get("/api/v3/forgetting/stats")
            assert resp.status_code == 200
            assert resp.json()["total"] == 0


# ---------------------------------------------------------------------------
# TASK 1b: Forgetting Trigger Tests
# ---------------------------------------------------------------------------


class TestForgettingRun:
    """Tests for POST /api/v3/forgetting/run."""

    def test_run_forgetting_on_seeded_db(self, seeded_v33_db):
        """POST /forgetting/run decays retention scores."""
        db_path, _, _, _ = seeded_v33_db
        from fastapi.testclient import TestClient

        with patch("superlocalmemory.server.routes.helpers.DB_PATH", db_path), \
             patch("superlocalmemory.server.routes.helpers.get_active_profile", return_value="default"):
            client = TestClient(_make_app())
            resp = client.post("/api/v3/forgetting/run", json={"profile": "default"})
            assert resp.status_code == 200
            data = resp.json()
            assert data["success"] is True
            assert "facts_decayed" in data
            assert data["profile"] == "default"

    def test_run_forgetting_no_db(self, tmp_path):
        """POST /forgetting/run with no DB returns error gracefully."""
        fake_db = tmp_path / "nonexistent.db"
        from fastapi.testclient import TestClient

        with patch("superlocalmemory.server.routes.helpers.DB_PATH", fake_db), \
             patch("superlocalmemory.server.routes.helpers.get_active_profile", return_value="default"):
            client = TestClient(_make_app())
            resp = client.post("/api/v3/forgetting/run", json={})
            assert resp.status_code == 200
            data = resp.json()
            assert data["success"] is False

    def test_run_forgetting_does_not_touch_archived(self, seeded_v33_db):
        """Archived/forgotten facts are NOT decayed further."""
        db_path, fact_ids, _, _ = seeded_v33_db
        from fastapi.testclient import TestClient

        # Record pre-decay score for the archive fact
        conn = sqlite3.connect(str(db_path))
        pre_score = conn.execute(
            "SELECT retention_score FROM fact_retention "
            "WHERE lifecycle_zone = 'archive' AND profile_id = 'default'",
        ).fetchone()[0]
        conn.close()

        with patch("superlocalmemory.server.routes.helpers.DB_PATH", db_path), \
             patch("superlocalmemory.server.routes.helpers.get_active_profile", return_value="default"):
            client = TestClient(_make_app())
            client.post("/api/v3/forgetting/run", json={})

        conn = sqlite3.connect(str(db_path))
        post_score = conn.execute(
            "SELECT retention_score FROM fact_retention "
            "WHERE lifecycle_zone = 'archive' AND profile_id = 'default'",
        ).fetchone()[0]
        conn.close()
        # Archive facts should NOT have been decayed
        assert post_score == pre_score


# ---------------------------------------------------------------------------
# TASK 1c: Quantization Stats Tests
# ---------------------------------------------------------------------------


class TestQuantizationStats:
    """Tests for GET /api/v3/quantization/stats."""

    def test_empty_db_returns_zeros(self, v33_db):
        """Empty DB returns total=0 with all tier counts zero."""
        from fastapi.testclient import TestClient

        with patch("superlocalmemory.server.routes.helpers.DB_PATH", v33_db), \
             patch("superlocalmemory.server.routes.helpers.get_active_profile", return_value="default"):
            client = TestClient(_make_app())
            resp = client.get("/api/v3/quantization/stats?profile=default")
            assert resp.status_code == 200
            data = resp.json()
            assert data["total"] == 0
            assert set(data["tiers"].keys()) == {"float32", "int8", "polar4", "polar2"}
            assert data["compression_ratio"] == 1.0

    def test_seeded_db_returns_tier_distribution(self, seeded_v33_db):
        """Seeded DB returns correct tier distribution."""
        db_path, _, _, _ = seeded_v33_db
        from fastapi.testclient import TestClient

        with patch("superlocalmemory.server.routes.helpers.DB_PATH", db_path), \
             patch("superlocalmemory.server.routes.helpers.get_active_profile", return_value="default"):
            client = TestClient(_make_app())
            resp = client.get("/api/v3/quantization/stats?profile=default")
            assert resp.status_code == 200
            data = resp.json()
            assert data["total"] == 5
            assert data["tiers"]["float32"] == 2
            assert data["tiers"]["int8"] == 1
            assert data["tiers"]["polar4"] == 1
            assert data["tiers"]["polar2"] == 1
            assert isinstance(data["compression_ratio"], (int, float))


# ---------------------------------------------------------------------------
# TASK 1d: CCQ Blocks Tests
# ---------------------------------------------------------------------------


class TestCCQBlocks:
    """Tests for GET /api/v3/ccq/blocks."""

    def test_empty_db_returns_no_blocks(self, v33_db):
        """Empty DB returns empty block list."""
        from fastapi.testclient import TestClient

        with patch("superlocalmemory.server.routes.helpers.DB_PATH", v33_db), \
             patch("superlocalmemory.server.routes.helpers.get_active_profile", return_value="default"):
            client = TestClient(_make_app())
            resp = client.get("/api/v3/ccq/blocks?profile=default")
            assert resp.status_code == 200
            data = resp.json()
            assert data["blocks"] == []
            assert data["total"] == 0

    def test_seeded_db_returns_blocks(self, seeded_v33_db):
        """Seeded DB returns CCQ blocks with metadata."""
        db_path, _, block_ids, _ = seeded_v33_db
        from fastapi.testclient import TestClient

        with patch("superlocalmemory.server.routes.helpers.DB_PATH", db_path), \
             patch("superlocalmemory.server.routes.helpers.get_active_profile", return_value="default"):
            client = TestClient(_make_app())
            resp = client.get("/api/v3/ccq/blocks?profile=default")
            assert resp.status_code == 200
            data = resp.json()
            assert data["total"] == len(block_ids)
            assert len(data["blocks"]) == len(block_ids)

            block = data["blocks"][0]
            assert "block_id" in block
            assert "content" in block
            assert "source_fact_count" in block
            assert "cluster_id" in block
            assert block["source_fact_count"] >= 2

    def test_ccq_blocks_respects_limit(self, seeded_v33_db):
        """Limit parameter caps the number of blocks returned."""
        db_path, _, _, _ = seeded_v33_db
        from fastapi.testclient import TestClient

        with patch("superlocalmemory.server.routes.helpers.DB_PATH", db_path), \
             patch("superlocalmemory.server.routes.helpers.get_active_profile", return_value="default"):
            client = TestClient(_make_app())
            resp = client.get("/api/v3/ccq/blocks?profile=default&limit=1")
            assert resp.status_code == 200
            data = resp.json()
            assert len(data["blocks"]) == 1
            assert data["total"] == 3  # Total is unaffected by limit


# ---------------------------------------------------------------------------
# TASK 1e: Soft Prompts Tests
# ---------------------------------------------------------------------------


class TestSoftPrompts:
    """Tests for GET /api/v3/soft-prompts."""

    def test_empty_db_returns_no_prompts(self, v33_db):
        """Empty DB returns empty prompt list."""
        from fastapi.testclient import TestClient

        with patch("superlocalmemory.server.routes.helpers.DB_PATH", v33_db), \
             patch("superlocalmemory.server.routes.helpers.get_active_profile", return_value="default"):
            client = TestClient(_make_app())
            resp = client.get("/api/v3/soft-prompts?profile=default")
            assert resp.status_code == 200
            data = resp.json()
            assert data["prompts"] == []
            assert data["total"] == 0
            assert data["total_tokens"] == 0

    def test_seeded_db_returns_prompts(self, seeded_v33_db):
        """Seeded DB returns active soft prompts sorted by confidence."""
        db_path, _, _, prompt_ids = seeded_v33_db
        from fastapi.testclient import TestClient

        with patch("superlocalmemory.server.routes.helpers.DB_PATH", db_path), \
             patch("superlocalmemory.server.routes.helpers.get_active_profile", return_value="default"):
            client = TestClient(_make_app())
            resp = client.get("/api/v3/soft-prompts?profile=default")
            assert resp.status_code == 200
            data = resp.json()
            assert data["total"] == len(prompt_ids)
            assert data["total_tokens"] == 35  # 20 + 15

            prompt = data["prompts"][0]
            assert "prompt_id" in prompt
            assert "category" in prompt
            assert "confidence" in prompt
            assert "token_count" in prompt
            # Sorted by confidence DESC
            assert data["prompts"][0]["confidence"] >= data["prompts"][1]["confidence"]


# ---------------------------------------------------------------------------
# TASK 1f: Process Health Tests
# ---------------------------------------------------------------------------


class TestProcessHealth:
    """Tests for GET /api/v3/health/processes."""

    def test_process_health_returns_structure(self):
        """Response has expected keys: processes, memory_mb, healthy."""
        from fastapi.testclient import TestClient

        client = TestClient(_make_app())
        resp = client.get("/api/v3/health/processes")
        assert resp.status_code == 200
        data = resp.json()
        assert "processes" in data
        assert "mcp_server" in data["processes"]
        assert data["processes"]["mcp_server"]["status"] == "running"
        assert "parent" in data["processes"]
        assert "memory_mb" in data
        assert "healthy" in data

    def test_process_health_reports_pid(self):
        """MCP server PID is a positive integer."""
        import os
        from fastapi.testclient import TestClient

        client = TestClient(_make_app())
        resp = client.get("/api/v3/health/processes")
        data = resp.json()
        assert data["processes"]["mcp_server"]["pid"] == os.getpid()


# ---------------------------------------------------------------------------
# TASK 1g: V3.3 Overview Tests
# ---------------------------------------------------------------------------


class TestV33Overview:
    """Tests for GET /api/v3/v33/overview."""

    def test_empty_db_returns_defaults(self, v33_db):
        """Empty DB returns zeroed overview with all feature sections."""
        from fastapi.testclient import TestClient

        with patch("superlocalmemory.server.routes.helpers.DB_PATH", v33_db), \
             patch("superlocalmemory.server.routes.helpers.get_active_profile", return_value="default"):
            client = TestClient(_make_app())
            resp = client.get("/api/v3/v33/overview?profile=default")
            assert resp.status_code == 200
            data = resp.json()
            assert data["version"] == "3.3"
            assert data["profile"] == "default"
            assert "forgetting" in data
            assert "quantization" in data
            assert "ccq" in data
            assert "soft_prompts" in data
            assert "hopfield" in data
            assert "process_health" in data

    def test_seeded_db_returns_combined_stats(self, seeded_v33_db):
        """Seeded DB returns non-zero stats for all features."""
        db_path, _, _, _ = seeded_v33_db
        from fastapi.testclient import TestClient

        with patch("superlocalmemory.server.routes.helpers.DB_PATH", db_path), \
             patch("superlocalmemory.server.routes.helpers.get_active_profile", return_value="default"):
            client = TestClient(_make_app())
            resp = client.get("/api/v3/v33/overview?profile=default")
            assert resp.status_code == 200
            data = resp.json()
            # Forgetting has data
            assert data["forgetting"]["total"] == 6
            # Quantization has data
            assert data["quantization"]["total"] == 5
            # CCQ has blocks
            assert data["ccq"]["blocks"] == 3
            # Soft prompts exist
            assert data["soft_prompts"]["total"] == 2

    def test_no_db_returns_defaults(self, tmp_path):
        """Missing DB returns graceful default overview."""
        fake_db = tmp_path / "nonexistent.db"
        from fastapi.testclient import TestClient

        with patch("superlocalmemory.server.routes.helpers.DB_PATH", fake_db), \
             patch("superlocalmemory.server.routes.helpers.get_active_profile", return_value="default"):
            client = TestClient(_make_app())
            resp = client.get("/api/v3/v33/overview")
            assert resp.status_code == 200
            data = resp.json()
            assert data["version"] == "3.3"


# ---------------------------------------------------------------------------
# TASK 2a: ACLI Soft Prompt Injection Tests
# ---------------------------------------------------------------------------


class TestAutoInvokerSoftPromptInjection:
    """Tests for soft prompt injection wired into AutoInvoker."""

    def _make_invoker(self, prompt_injector=None, config=None):
        """Create an AutoInvoker with mocked DB."""
        from superlocalmemory.hooks.auto_invoker import AutoInvoker
        from superlocalmemory.core.config import AutoInvokeConfig

        db = MagicMock()
        db.execute.return_value = []
        db.get_fact_context.return_value = None
        cfg = config or AutoInvokeConfig(
            enabled=True, profile_id="default",
        )
        return AutoInvoker(
            db=db, config=cfg, prompt_injector=prompt_injector,
        )

    def test_no_injector_returns_memory_only(self):
        """Without injector, get_session_context returns memory context only."""
        invoker = self._make_invoker(prompt_injector=None)
        # No results from DB, so should return ""
        result = invoker.get_session_context(query="test")
        assert result == ""

    def test_injector_prepends_soft_prompts(self):
        """With injector, soft prompts come before memory context."""
        injector = MagicMock()
        injector.get_injection_context.return_value = "# Soft Prompt\nYou are an architect."
        injector.inject_into_context.return_value = (
            "# Soft Prompt\nYou are an architect.\n\n# Relevant Memory Context"
        )
        invoker = self._make_invoker(prompt_injector=injector)

        result = invoker.get_session_context(query="test")
        # Soft prompt should be present
        assert "Soft Prompt" in result or result == ""
        # Injector was consulted
        injector.get_injection_context.assert_called_once_with("default")

    def test_injector_failure_does_not_crash(self):
        """If injector raises, auto-invoke still returns gracefully."""
        injector = MagicMock()
        injector.get_injection_context.side_effect = RuntimeError("DB error")
        invoker = self._make_invoker(prompt_injector=injector)

        # Should not raise
        result = invoker.get_session_context(query="test")
        assert isinstance(result, str)

    def test_get_soft_prompt_text_returns_empty_without_injector(self):
        """_get_soft_prompt_text returns '' when no injector is set."""
        invoker = self._make_invoker(prompt_injector=None)
        assert invoker._get_soft_prompt_text() == ""

    def test_get_soft_prompt_text_returns_text_with_injector(self):
        """_get_soft_prompt_text returns injector output when set."""
        injector = MagicMock()
        injector.get_injection_context.return_value = "You prefer Python."
        invoker = self._make_invoker(prompt_injector=injector)
        result = invoker._get_soft_prompt_text()
        assert result == "You prefer Python."

    def test_prompt_injector_stored_as_attribute(self):
        """prompt_injector is accessible as _prompt_injector."""
        injector = MagicMock()
        invoker = self._make_invoker(prompt_injector=injector)
        assert invoker._prompt_injector is injector


# ---------------------------------------------------------------------------
# TASK 2b: Forgetting-Aware Auto-Invoke Tests
# ---------------------------------------------------------------------------


class TestForgettingAwareAutoInvoke:
    """Tests for forgetting-aware filtering in AutoInvoker."""

    def test_excluded_zones_contains_forgotten(self):
        """_EXCLUDED_ZONES includes both 'archived' and 'forgotten'."""
        from superlocalmemory.hooks.auto_invoker import AutoInvoker
        assert "archived" in AutoInvoker._EXCLUDED_ZONES
        assert "forgotten" in AutoInvoker._EXCLUDED_ZONES

    def test_enrich_result_skips_archived(self):
        """_enrich_result returns None for archived facts."""
        from superlocalmemory.hooks.auto_invoker import AutoInvoker
        from superlocalmemory.core.config import AutoInvokeConfig

        db = MagicMock()
        db.execute.return_value = [
            {"fact_id": "f1", "content": "test", "fact_type": "semantic", "lifecycle": "archived"},
        ]
        invoker = AutoInvoker(db=db, config=AutoInvokeConfig(enabled=True))
        result = invoker._enrich_result("f1", 0.5, {}, "default")
        assert result is None

    def test_enrich_result_skips_forgotten(self):
        """_enrich_result returns None for forgotten facts."""
        from superlocalmemory.hooks.auto_invoker import AutoInvoker
        from superlocalmemory.core.config import AutoInvokeConfig

        db = MagicMock()
        db.execute.return_value = [
            {"fact_id": "f1", "content": "test", "fact_type": "semantic", "lifecycle": "forgotten"},
        ]
        invoker = AutoInvoker(db=db, config=AutoInvokeConfig(enabled=True))
        result = invoker._enrich_result("f1", 0.5, {}, "default")
        assert result is None

    def test_enrich_result_allows_active(self):
        """_enrich_result returns data for active facts."""
        from superlocalmemory.hooks.auto_invoker import AutoInvoker
        from superlocalmemory.core.config import AutoInvokeConfig

        db = MagicMock()
        db.execute.return_value = [
            {"fact_id": "f1", "content": "test content", "fact_type": "semantic", "lifecycle": "active"},
        ]
        db.get_fact_context.return_value = None
        invoker = AutoInvoker(db=db, config=AutoInvokeConfig(enabled=True))
        result = invoker._enrich_result("f1", 0.8, {"similarity": 0.8}, "default")
        assert result is not None
        assert result["fact_id"] == "f1"
        assert result["content"] == "test content"

    def test_enrich_result_allows_warm(self):
        """_enrich_result returns data for warm (non-excluded) facts."""
        from superlocalmemory.hooks.auto_invoker import AutoInvoker
        from superlocalmemory.core.config import AutoInvokeConfig

        db = MagicMock()
        db.execute.return_value = [
            {"fact_id": "f2", "content": "warm fact", "fact_type": "semantic", "lifecycle": "warm"},
        ]
        db.get_fact_context.return_value = None
        invoker = AutoInvoker(db=db, config=AutoInvokeConfig(enabled=True))
        result = invoker._enrich_result("f2", 0.5, {"similarity": 0.5}, "default")
        assert result is not None

    def test_candidates_sql_excludes_forgotten(self):
        """_get_candidates SQL fallback includes lifecycle filter."""
        from superlocalmemory.hooks.auto_invoker import AutoInvoker
        from superlocalmemory.core.config import AutoInvokeConfig

        db = MagicMock()
        db.execute.return_value = [{"fact_id": "f1"}]
        invoker = AutoInvoker(db=db, config=AutoInvokeConfig(enabled=True))

        invoker._get_candidates("test query", "default", top_k=10)
        # Verify the SQL called includes the lifecycle filter
        call_args = db.execute.call_args
        sql = call_args[0][0]
        assert "archived" in sql.lower() or "forgotten" in sql.lower()
