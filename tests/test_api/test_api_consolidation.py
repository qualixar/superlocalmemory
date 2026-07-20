# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3.2 | Phase 6 API Tests

"""Tests for consolidation, core-memory, and vector-store API endpoints.

Covers:
- GET /api/v3/consolidation/status
- POST /api/v3/consolidation/trigger
- GET /api/v3/core-memory
- PUT /api/v3/core-memory/{block_id}
- GET /api/v3/vector-store/status
"""

from __future__ import annotations

import sqlite3
from unittest.mock import MagicMock, patch

import pytest

fastapi = pytest.importorskip("fastapi", reason="fastapi not installed")


def _make_app():
    """Create a minimal FastAPI app with v3 routes."""
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


def _mock_slm_config():
    """Build a mock SLMConfig with consolidation defaults."""
    from superlocalmemory.core.config import (
        ConsolidationConfig,
        EmbeddingConfig,
        SLMConfig,
    )
    from superlocalmemory.storage.models import Mode

    config = SLMConfig.for_mode(Mode.A)
    return config


class TestConsolidationStatus:
    """Tests for GET /api/v3/consolidation/status."""

    def test_get_consolidation_status_initial(self, empty_db):
        """Returns config flags even when no consolidation has run."""
        from fastapi.testclient import TestClient

        mock_cfg = _mock_slm_config()
        with patch("superlocalmemory.server.routes.helpers.DB_PATH", empty_db), \
             patch("superlocalmemory.server.routes.helpers.get_active_profile", return_value="default"), \
             patch("superlocalmemory.core.config.SLMConfig.load", return_value=mock_cfg):
            client = TestClient(_make_app())
            resp = client.get("/api/v3/consolidation/status?profile=default")
            assert resp.status_code == 200
            data = resp.json()

            assert "enabled" in data
            assert "triggers" in data
            assert "session_end" in data["triggers"]
            assert "idle_timeout" in data["triggers"]
            assert "step_count" in data["triggers"]
            assert "store_count_since_last" in data


class TestConsolidationTrigger:
    """Tests for POST /api/v3/consolidation/trigger."""

    def test_trigger_consolidation_returns_success(self, seeded_db):
        """POST trigger returns success when consolidation engine runs."""
        db_path, _, _, _ = seeded_db
        from fastapi.testclient import TestClient

        mock_cfg = _mock_slm_config()
        mock_cfg_copy = _mock_slm_config()

        # Mock the consolidation engine to avoid needing full deps
        mock_engine = MagicMock()
        mock_engine.consolidate.return_value = {
            "profile_id": "default",
            "lightweight": False,
            "blocks": 2,
            "compressed": 0,
        }

        with patch("superlocalmemory.server.routes.helpers.get_active_profile", return_value="default"), \
             patch("superlocalmemory.core.worker_pool.WorkerPool.shared", side_effect=Exception("no pool")), \
             patch("superlocalmemory.core.config.SLMConfig.load", return_value=mock_cfg), \
             patch("superlocalmemory.core.consolidation_engine.ConsolidationEngine", return_value=mock_engine):
            client = TestClient(_make_app())
            resp = client.post(
                "/api/v3/consolidation/trigger",
                json={"lightweight": False, "profile": "default"},
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["success"] is True

    def test_trigger_consolidation_lightweight(self, seeded_db):
        """POST trigger with lightweight=true calls consolidation correctly."""
        db_path, _, _, _ = seeded_db
        from fastapi.testclient import TestClient

        mock_engine = MagicMock()
        mock_engine.consolidate.return_value = {
            "profile_id": "default",
            "lightweight": True,
            "blocks": 1,
        }

        mock_cfg = _mock_slm_config()
        with patch("superlocalmemory.server.routes.helpers.get_active_profile", return_value="default"), \
             patch("superlocalmemory.core.worker_pool.WorkerPool.shared", side_effect=Exception("no pool")), \
             patch("superlocalmemory.core.config.SLMConfig.load", return_value=mock_cfg), \
             patch("superlocalmemory.core.consolidation_engine.ConsolidationEngine", return_value=mock_engine):
            client = TestClient(_make_app())
            resp = client.post(
                "/api/v3/consolidation/trigger",
                json={"lightweight": True},
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["success"] is True
            mock_engine.consolidate.assert_called_once_with(
                profile_id="default", lightweight=True,
            )


class TestCoreMemory:
    """Tests for GET/PUT /api/v3/core-memory endpoints."""

    def test_get_core_memory_empty(self, empty_db):
        """GET /api/v3/core-memory returns empty blocks list."""
        from fastapi.testclient import TestClient

        with patch("superlocalmemory.server.routes.helpers.DB_PATH", empty_db), \
             patch("superlocalmemory.server.routes.helpers.get_active_profile", return_value="default"):
            client = TestClient(_make_app())
            resp = client.get("/api/v3/core-memory?profile=default")
            assert resp.status_code == 200
            data = resp.json()
            assert data["blocks"] == []
            assert data["total_chars"] == 0
            assert data["char_limit"] == 2000

    def test_get_core_memory_with_blocks(self, seeded_db):
        """GET /api/v3/core-memory returns populated blocks."""
        db_path, _, _, block_ids = seeded_db
        from fastapi.testclient import TestClient

        with patch("superlocalmemory.server.routes.helpers.DB_PATH", db_path), \
             patch("superlocalmemory.server.routes.helpers.get_active_profile", return_value="default"):
            client = TestClient(_make_app())
            resp = client.get("/api/v3/core-memory?profile=default")
            assert resp.status_code == 200
            data = resp.json()
            assert len(data["blocks"]) == len(block_ids)
            assert data["total_chars"] > 0

            block = data["blocks"][0]
            assert "block_id" in block
            assert "block_type" in block
            assert "content" in block
            assert "char_count" in block
            assert "version" in block
            assert "compiled_by" in block

    def test_put_core_memory_block(self, seeded_db):
        """PUT /api/v3/core-memory/{block_id} updates content."""
        db_path, _, _, block_ids = seeded_db
        from fastapi.testclient import TestClient

        with patch("superlocalmemory.server.routes.helpers.DB_PATH", db_path), \
             patch("superlocalmemory.server.routes.helpers.get_active_profile", return_value="default"):
            client = TestClient(_make_app())

            bid = block_ids[0]
            new_content = "Updated core memory content for testing"
            resp = client.put(
                f"/api/v3/core-memory/{bid}",
                json={"content": new_content},
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["content"] == new_content
            assert data["char_count"] == len(new_content)
            assert data["version"] == 2
            assert data["compiled_by"] == "manual"

            # Verify via GET
            resp2 = client.get("/api/v3/core-memory?profile=default")
            blocks = resp2.json()["blocks"]
            updated = [b for b in blocks if b["block_id"] == bid][0]
            assert updated["content"] == new_content

    def test_put_core_memory_block_not_found(self, empty_db):
        """PUT with nonexistent block_id returns 404."""
        from fastapi.testclient import TestClient

        with patch("superlocalmemory.server.routes.helpers.DB_PATH", empty_db), \
             patch("superlocalmemory.server.routes.helpers.get_active_profile", return_value="default"):
            client = TestClient(_make_app())
            resp = client.put(
                "/api/v3/core-memory/nonexistent_block",
                json={"content": "test"},
            )
            assert resp.status_code == 404

    def test_put_core_memory_block_missing_content(self, seeded_db):
        """PUT without content field returns 400."""
        db_path, _, _, block_ids = seeded_db
        from fastapi.testclient import TestClient

        with patch("superlocalmemory.server.routes.helpers.DB_PATH", db_path), \
             patch("superlocalmemory.server.routes.helpers.get_active_profile", return_value="default"):
            client = TestClient(_make_app())
            resp = client.put(
                f"/api/v3/core-memory/{block_ids[0]}",
                json={},
            )
            assert resp.status_code == 400


class TestVectorStoreStatus:
    """Tests for GET /api/v3/vector-store/status."""

    def test_get_vector_store_status(self, empty_db):
        """GET /api/v3/vector-store/status returns health info."""
        from fastapi.testclient import TestClient

        mock_cfg = _mock_slm_config()
        with patch("superlocalmemory.server.routes.helpers.DB_PATH", empty_db), \
             patch("superlocalmemory.core.config.SLMConfig.load", return_value=mock_cfg):
            client = TestClient(_make_app())
            resp = client.get("/api/v3/vector-store/status")
            assert resp.status_code == 200
            data = resp.json()

            assert "available" in data
            assert "provider" in data
            assert data["provider"] == "sqlite-vec"
            assert "dimension" in data
            assert isinstance(data["dimension"], int)
            assert "embedding_model" in data
            assert "total_vectors" in data
            assert "binary_quantization" in data
