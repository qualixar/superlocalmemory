# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3.2 | Phase 6 API Tests

"""Tests for GET/PUT /api/v3/auto-invoke/config endpoints.

Verifies:
- Default config returned on first GET
- PUT updates persist and GET returns updated values
- Invalid min_score is rejected with 400
- Weights are treated as multipliers (no sum constraint)
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

fastapi = pytest.importorskip("fastapi", reason="fastapi not installed")


def _patch_memory_dir(tmp_path):
    """Patch MEMORY_DIR and config path so tests use tmp_path."""
    return patch.multiple(
        "superlocalmemory.server.routes.v3_api",
        _load_auto_invoke_json=_make_loader(tmp_path),
        _save_auto_invoke_json=_make_saver(tmp_path),
    )


def _config_path(tmp_path: Path) -> Path:
    return tmp_path / "config.json"


def _make_loader(tmp_path: Path):
    def _load() -> dict:
        cp = _config_path(tmp_path)
        if cp.exists():
            try:
                data = json.loads(cp.read_text(encoding="utf-8"))
                return data.get("auto_invoke", {})
            except Exception:
                pass
        return {}
    return _load


def _make_saver(tmp_path: Path):
    def _save(auto_invoke_data: dict) -> None:
        cp = _config_path(tmp_path)
        cfg: dict = {}
        if cp.exists():
            try:
                cfg = json.loads(cp.read_text(encoding="utf-8"))
            except Exception:
                pass
        cfg["auto_invoke"] = auto_invoke_data
        cp.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    return _save


class TestAutoInvokeConfig:
    """Tests for auto-invoke config API endpoints."""

    def test_get_auto_invoke_config_returns_defaults(self, tmp_path):
        """GET /api/v3/auto-invoke/config returns sensible defaults."""
        with _patch_memory_dir(tmp_path):
            from fastapi import FastAPI
            from fastapi.testclient import TestClient

            from superlocalmemory.server.routes.v3_api import router

            app = FastAPI()
            app.include_router(router)
            client = TestClient(app)

            resp = client.get("/api/v3/auto-invoke/config")
            assert resp.status_code == 200
            data = resp.json()

            assert "enabled" in data
            assert data["enabled"] is True
            assert "min_score" in data
            assert isinstance(data["min_score"], (int, float))
            assert "weights" in data
            assert "similarity" in data["weights"]
            assert "act_r_mode" in data

    def test_put_auto_invoke_config_updates(self, tmp_path):
        """PUT then GET round-trips config updates correctly."""
        with _patch_memory_dir(tmp_path):
            from fastapi import FastAPI
            from fastapi.testclient import TestClient

            from superlocalmemory.server.routes.v3_api import router

            app = FastAPI()
            app.include_router(router)
            client = TestClient(app)

            # Update config
            resp = client.put(
                "/api/v3/auto-invoke/config",
                json={"enabled": True, "min_score": 0.15},
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["enabled"] is True
            assert data["min_score"] == 0.15

            # Verify persistence via GET
            resp2 = client.get("/api/v3/auto-invoke/config")
            assert resp2.status_code == 200
            data2 = resp2.json()
            assert data2["enabled"] is True
            assert data2["min_score"] == 0.15

    def test_put_auto_invoke_config_validates_weights(self, tmp_path):
        """PUT with arbitrary weight values is accepted (multipliers, not probabilities)."""
        with _patch_memory_dir(tmp_path):
            from fastapi import FastAPI
            from fastapi.testclient import TestClient

            from superlocalmemory.server.routes.v3_api import router

            app = FastAPI()
            app.include_router(router)
            client = TestClient(app)

            resp = client.put(
                "/api/v3/auto-invoke/config",
                json={"weights": {"similarity": 0.8, "recency": 0.5, "frequency": 0.3, "trust": 0.2}},
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["weights"]["similarity"] == 0.8

    def test_put_auto_invoke_config_rejects_invalid_min_score(self, tmp_path):
        """PUT with negative min_score returns 400."""
        with _patch_memory_dir(tmp_path):
            from fastapi import FastAPI
            from fastapi.testclient import TestClient

            from superlocalmemory.server.routes.v3_api import router

            app = FastAPI()
            app.include_router(router)
            client = TestClient(app)

            resp = client.put(
                "/api/v3/auto-invoke/config",
                json={"min_score": -1},
            )
            assert resp.status_code == 400
            assert "error" in resp.json()
