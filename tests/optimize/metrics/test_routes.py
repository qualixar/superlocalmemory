# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later
"""Tests for optimize API routes (server/routes/optimize.py)."""

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def _client(tmp_path, monkeypatch):
    """Create a test client with isolated config + db."""
    from fastapi import FastAPI
    from superlocalmemory.server.routes.optimize import router
    from superlocalmemory.optimize.config.store import ConfigStore
    from superlocalmemory.optimize.storage.db import CacheDB
    from superlocalmemory.optimize.metrics.counters import MetricsCollector

    # Reset singleton
    MetricsCollector._instance = None

    store = ConfigStore(config_path=tmp_path / "optimize.json")
    db = CacheDB(tmp_path / "llmcache.db")

    monkeypatch.setattr(
        "superlocalmemory.server.routes.optimize.__name__", "optimize"
    )

    # Patch the imports inside the route handlers
    import superlocalmemory.server.routes.optimize as mod

    original_get_store = None
    try:
        from superlocalmemory.optimize.config.store import ConfigStore as CS
        monkeypatch.setattr(mod, "ConfigStore", lambda: store)
    except Exception:
        pass

    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)
    yield client

    MetricsCollector._instance = None


def test_get_config(_client):
    resp = _client.get("/api/optimize/config")
    assert resp.status_code == 200
    data = resp.json()
    assert "enabled" in data
    assert "cache_enabled" in data
    assert "compress_enabled" in data


def test_put_config(_client):
    resp = _client.put("/api/optimize/config", json={"enabled": True})
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"


def test_get_savings(_client):
    resp = _client.get("/api/optimize/savings")
    assert resp.status_code == 200
    data = resp.json()
    # Mandatory contract keys (INTERFACE-CONTRACT §5)
    assert "tokens_saved_input" in data
    assert "tokens_saved_output" in data
    assert "calls_skipped" in data
    assert "compress_ratio" in data
    assert "cost_saved" in data
    assert "usd" in data["cost_saved"]
    assert "inr" in data["cost_saved"]
    assert "hit_rate" in data
    assert "cache_bytes" in data
    assert "entries" in data
    # Non-conflicting diagnostic extras
    assert "hits" in data
    assert "pricing_date" in data


def test_get_stats(_client):
    resp = _client.get("/api/optimize/stats")
    assert resp.status_code == 200
    data = resp.json()
    assert "hits" in data
    assert "misses" in data
    assert "tokens_saved_input" in data


def test_delete_cache_clear(_client):
    resp = _client.delete("/api/optimize/cache/clear?tenant=default")
    assert resp.status_code == 200
    data = resp.json()
    assert "deleted" in data
