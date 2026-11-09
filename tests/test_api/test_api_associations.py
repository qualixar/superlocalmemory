# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3.2 | Phase 6 API Tests

"""Tests for GET /api/v3/associations and GET /api/v3/associations/stats.

Verifies:
- Empty DB returns empty edges list
- Seeded DB returns edges with source/target previews
- Type filter works correctly
- Stats endpoint returns correct aggregates
- Profile scoping (Rule 01)
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

fastapi = pytest.importorskip("fastapi", reason="fastapi not installed")


def _make_app_with_db(db_path):
    """Create a FastAPI app with routes patched to use given db_path."""
    from fastapi import FastAPI

    from superlocalmemory.server.routes.v3_api import router

    app = FastAPI()
    app.include_router(router)
    return app


class TestAssociations:
    """Tests for the associations API endpoints."""

    def test_get_associations_empty(self, empty_db):
        """GET /api/v3/associations returns empty list when no edges exist."""
        from fastapi.testclient import TestClient

        with patch("superlocalmemory.server.routes.helpers.DB_PATH", empty_db), \
             patch("superlocalmemory.server.routes.helpers.get_active_profile", return_value="default"):
            app = _make_app_with_db(empty_db)
            client = TestClient(app)
            resp = client.get("/api/v3/associations?profile=default")
            assert resp.status_code == 200
            data = resp.json()
            assert data["edges"] == []
            assert data["total"] == 0

    def test_get_associations_with_data(self, seeded_db):
        """GET /api/v3/associations returns edges with previews."""
        db_path, fact_ids, edge_ids, _ = seeded_db
        from fastapi.testclient import TestClient

        with patch("superlocalmemory.server.routes.helpers.DB_PATH", db_path), \
             patch("superlocalmemory.server.routes.helpers.get_active_profile", return_value="default"):
            app = _make_app_with_db(db_path)
            client = TestClient(app)
            resp = client.get("/api/v3/associations?profile=default")
            assert resp.status_code == 200
            data = resp.json()
            assert len(data["edges"]) > 0
            assert data["total"] == len(edge_ids)

            edge = data["edges"][0]
            assert "edge_id" in edge
            assert "source_fact_id" in edge
            assert "target_fact_id" in edge
            assert "association_type" in edge
            assert "weight" in edge
            assert "source_preview" in edge
            assert "target_preview" in edge
            assert len(edge["source_preview"]) > 0

    def test_get_associations_filtered_by_type(self, seeded_db):
        """GET /api/v3/associations?type=auto_link filters correctly."""
        db_path, _, _, _ = seeded_db
        from fastapi.testclient import TestClient

        with patch("superlocalmemory.server.routes.helpers.DB_PATH", db_path), \
             patch("superlocalmemory.server.routes.helpers.get_active_profile", return_value="default"):
            app = _make_app_with_db(db_path)
            client = TestClient(app)
            resp = client.get("/api/v3/associations?type=auto_link&profile=default")
            assert resp.status_code == 200
            data = resp.json()
            for edge in data["edges"]:
                assert edge["association_type"] == "auto_link"

    def test_get_associations_stats(self, seeded_db):
        """GET /api/v3/associations/stats returns correct aggregates."""
        db_path, _, edge_ids, _ = seeded_db
        from fastapi.testclient import TestClient

        with patch("superlocalmemory.server.routes.helpers.DB_PATH", db_path), \
             patch("superlocalmemory.server.routes.helpers.get_active_profile", return_value="default"):
            app = _make_app_with_db(db_path)
            client = TestClient(app)
            resp = client.get("/api/v3/associations/stats?profile=default")
            assert resp.status_code == 200
            data = resp.json()

            assert "total_edges" in data
            assert data["total_edges"] == len(edge_ids)
            assert "by_type" in data
            assert isinstance(data["by_type"], dict)
            assert "avg_weight" in data
            assert isinstance(data["avg_weight"], float)
            assert "community_count" in data
            assert "top_connected_facts" in data

    def test_get_associations_respects_profile(self, seeded_db):
        """GET /api/v3/associations?profile=nonexistent returns empty (Rule 01)."""
        db_path, _, _, _ = seeded_db
        from fastapi.testclient import TestClient

        with patch("superlocalmemory.server.routes.helpers.DB_PATH", db_path), \
             patch("superlocalmemory.server.routes.helpers.get_active_profile", return_value="nonexistent"):
            app = _make_app_with_db(db_path)
            client = TestClient(app)
            resp = client.get("/api/v3/associations?profile=nonexistent")
            assert resp.status_code == 200
            data = resp.json()
            assert data["edges"] == []
            assert data["total"] == 0
