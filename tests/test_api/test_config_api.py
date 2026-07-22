# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com

"""Tests for config API endpoints.

Covers:
- GET/PUT /api/v3/storage/config
- GET/PUT /api/v3/daemon/config
- GET/PUT /api/v3/mesh/config
- GET/PUT /api/v3/trust/config
- GET/PUT /api/v3/forgetting/config
- POST /api/evolution/disable

TDD approach: tests are written FIRST and verify:
1. GET returns defaults when no config.json exists
2. PUT persists values (reload from JSON and assert stuck)
3. PUT validates input (422 on bad input)
4. restart_required flag present for port/backend changes
5. mode is preserved across any config PUT
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Generator

import pytest

fastapi = pytest.importorskip("fastapi", reason="fastapi not installed")

from fastapi import FastAPI
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_app(monkeypatch, tmp_path: Path) -> TestClient:
    """Create a TestClient wired to config_api.router with MEMORY_DIR mocked."""
    # Import lazily so monkeypatch fires first
    monkeypatch.setattr(
        "superlocalmemory.server.routes.config_api.MEMORY_DIR", tmp_path
    )
    from superlocalmemory.server.routes.config_api import router

    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def _make_evo_app(monkeypatch, tmp_path: Path) -> TestClient:
    """Create a TestClient wired to the evolution router with MEMORY_DIR mocked."""
    monkeypatch.setattr(
        "superlocalmemory.server.routes.evolution.MEMORY_DIR", tmp_path
    )
    from superlocalmemory.server.routes.evolution import router

    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def _read_config(tmp_path: Path) -> dict:
    p = tmp_path / "config.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text())


def _write_config(tmp_path: Path, data: dict) -> None:
    p = tmp_path / "config.json"
    p.write_text(json.dumps(data, indent=2))


# ---------------------------------------------------------------------------
# Storage config
# ---------------------------------------------------------------------------


class TestStorageConfigGet:
    def test_returns_defaults_when_no_config(self, tmp_path, monkeypatch):
        client = _make_app(monkeypatch, tmp_path)
        resp = client.get("/api/v3/storage/config")
        assert resp.status_code == 200
        data = resp.json()
        assert "graph_backend" in data
        assert "vector_backend" in data
        assert "base_dir" in data
        assert data["graph_backend"] == "auto"
        assert data["vector_backend"] == "auto"

    def test_reflects_existing_config_values(self, tmp_path, monkeypatch):
        _write_config(tmp_path, {
            "graph_backend": "sqlite",
            "vector_backend": "lancedb",
        })
        client = _make_app(monkeypatch, tmp_path)
        resp = client.get("/api/v3/storage/config")
        assert resp.status_code == 200
        data = resp.json()
        assert data["graph_backend"] == "sqlite"
        assert data["vector_backend"] == "lancedb"


class TestStorageConfigPut:
    def test_persists_graph_backend(self, tmp_path, monkeypatch):
        client = _make_app(monkeypatch, tmp_path)
        resp = client.put("/api/v3/storage/config", json={"graph_backend": "sqlite"})
        assert resp.status_code == 200
        saved = _read_config(tmp_path)
        assert saved["graph_backend"] == "sqlite"

    def test_persists_vector_backend(self, tmp_path, monkeypatch):
        client = _make_app(monkeypatch, tmp_path)
        resp = client.put("/api/v3/storage/config", json={"vector_backend": "lancedb"})
        assert resp.status_code == 200
        saved = _read_config(tmp_path)
        assert saved["vector_backend"] == "lancedb"

    def test_returns_restart_required_true(self, tmp_path, monkeypatch):
        client = _make_app(monkeypatch, tmp_path)
        resp = client.put("/api/v3/storage/config", json={"graph_backend": "cozo"})
        assert resp.status_code == 200
        assert resp.json()["restart_required"] is True

    def test_rejects_invalid_graph_backend(self, tmp_path, monkeypatch):
        client = _make_app(monkeypatch, tmp_path)
        resp = client.put("/api/v3/storage/config", json={"graph_backend": "nosuchdb"})
        assert resp.status_code == 422

    def test_rejects_invalid_vector_backend(self, tmp_path, monkeypatch):
        client = _make_app(monkeypatch, tmp_path)
        resp = client.put("/api/v3/storage/config", json={"vector_backend": "faiss"})
        assert resp.status_code == 422

    def test_rejects_unknown_keys(self, tmp_path, monkeypatch):
        client = _make_app(monkeypatch, tmp_path)
        resp = client.put("/api/v3/storage/config", json={"unknown_field": "value"})
        assert resp.status_code == 422

    def test_base_dir_is_read_only(self, tmp_path, monkeypatch):
        """base_dir in PUT body must be ignored — it is read-only."""
        client = _make_app(monkeypatch, tmp_path)
        # Write initial config with a base_dir
        _write_config(tmp_path, {"base_dir": "/original/path"})
        client2 = _make_app(monkeypatch, tmp_path)
        resp = client2.put(
            "/api/v3/storage/config",
            json={"graph_backend": "sqlite", "base_dir": "/hacked/path"},
        )
        assert resp.status_code == 422

    def test_mode_preserved_across_put(self, tmp_path, monkeypatch):
        _write_config(tmp_path, {"mode": "b", "graph_backend": "auto"})
        client = _make_app(monkeypatch, tmp_path)
        resp = client.put("/api/v3/storage/config", json={"graph_backend": "sqlite"})
        assert resp.status_code == 200
        saved = _read_config(tmp_path)
        assert saved.get("mode") == "b"


# ---------------------------------------------------------------------------
# Daemon config
# ---------------------------------------------------------------------------


class TestDaemonConfigGet:
    def test_returns_defaults_when_no_config(self, tmp_path, monkeypatch):
        client = _make_app(monkeypatch, tmp_path)
        resp = client.get("/api/v3/daemon/config")
        assert resp.status_code == 200
        data = resp.json()
        assert data["idle_timeout"] == 0
        assert data["port"] == 8765
        assert data["legacy_port"] == 8767
        assert data["enable_legacy_port"] is True

    def test_reflects_existing_config_values(self, tmp_path, monkeypatch):
        _write_config(tmp_path, {
            "daemon_idle_timeout": 3600,
            "daemon_port": 9000,
        })
        client = _make_app(monkeypatch, tmp_path)
        resp = client.get("/api/v3/daemon/config")
        assert resp.status_code == 200
        data = resp.json()
        assert data["idle_timeout"] == 3600
        assert data["port"] == 9000


class TestDaemonConfigPut:
    def test_persists_idle_timeout(self, tmp_path, monkeypatch):
        client = _make_app(monkeypatch, tmp_path)
        resp = client.put("/api/v3/daemon/config", json={"idle_timeout": 600})
        assert resp.status_code == 200
        saved = _read_config(tmp_path)
        assert saved["daemon_idle_timeout"] == 600

    def test_persists_port(self, tmp_path, monkeypatch):
        client = _make_app(monkeypatch, tmp_path)
        resp = client.put("/api/v3/daemon/config", json={"port": 9090})
        assert resp.status_code == 200
        saved = _read_config(tmp_path)
        assert saved["daemon_port"] == 9090

    def test_persists_enable_legacy_port(self, tmp_path, monkeypatch):
        client = _make_app(monkeypatch, tmp_path)
        resp = client.put("/api/v3/daemon/config", json={"enable_legacy_port": False})
        assert resp.status_code == 200
        saved = _read_config(tmp_path)
        assert saved["daemon_enable_legacy_port"] is False

    def test_restart_required_true_for_port_change(self, tmp_path, monkeypatch):
        client = _make_app(monkeypatch, tmp_path)
        resp = client.put("/api/v3/daemon/config", json={"port": 9091})
        assert resp.status_code == 200
        assert resp.json()["restart_required"] is True

    def test_restart_required_false_for_idle_timeout_only(self, tmp_path, monkeypatch):
        """idle_timeout alone does not require restart."""
        client = _make_app(monkeypatch, tmp_path)
        resp = client.put("/api/v3/daemon/config", json={"idle_timeout": 120})
        assert resp.status_code == 200
        assert resp.json()["restart_required"] is False

    def test_rejects_port_zero(self, tmp_path, monkeypatch):
        client = _make_app(monkeypatch, tmp_path)
        resp = client.put("/api/v3/daemon/config", json={"port": 0})
        assert resp.status_code == 422

    def test_rejects_port_too_large(self, tmp_path, monkeypatch):
        client = _make_app(monkeypatch, tmp_path)
        resp = client.put("/api/v3/daemon/config", json={"port": 99999})
        assert resp.status_code == 422

    def test_rejects_negative_idle_timeout(self, tmp_path, monkeypatch):
        client = _make_app(monkeypatch, tmp_path)
        resp = client.put("/api/v3/daemon/config", json={"idle_timeout": -1})
        assert resp.status_code == 422

    def test_rejects_unknown_keys(self, tmp_path, monkeypatch):
        client = _make_app(monkeypatch, tmp_path)
        resp = client.put("/api/v3/daemon/config", json={"bad_key": True})
        assert resp.status_code == 422

    def test_mode_preserved_across_put(self, tmp_path, monkeypatch):
        _write_config(tmp_path, {"mode": "c", "daemon_port": 8765})
        client = _make_app(monkeypatch, tmp_path)
        resp = client.put("/api/v3/daemon/config", json={"idle_timeout": 300})
        assert resp.status_code == 200
        saved = _read_config(tmp_path)
        assert saved.get("mode") == "c"


# ---------------------------------------------------------------------------
# Mesh config
# ---------------------------------------------------------------------------


class TestMeshConfigGet:
    def test_returns_default_enabled_true(self, tmp_path, monkeypatch):
        client = _make_app(monkeypatch, tmp_path)
        resp = client.get("/api/v3/mesh/config")
        assert resp.status_code == 200
        assert resp.json()["enabled"] is True

    def test_reflects_disabled_state(self, tmp_path, monkeypatch):
        _write_config(tmp_path, {"mesh_enabled": False})
        client = _make_app(monkeypatch, tmp_path)
        resp = client.get("/api/v3/mesh/config")
        assert resp.status_code == 200
        assert resp.json()["enabled"] is False


class TestMeshConfigPut:
    def test_persists_enabled_false(self, tmp_path, monkeypatch):
        client = _make_app(monkeypatch, tmp_path)
        resp = client.put("/api/v3/mesh/config", json={"enabled": False})
        assert resp.status_code == 200
        saved = _read_config(tmp_path)
        assert saved["mesh_enabled"] is False

    def test_persists_enabled_true(self, tmp_path, monkeypatch):
        _write_config(tmp_path, {"mesh_enabled": False})
        client = _make_app(monkeypatch, tmp_path)
        resp = client.put("/api/v3/mesh/config", json={"enabled": True})
        assert resp.status_code == 200
        saved = _read_config(tmp_path)
        assert saved["mesh_enabled"] is True

    def test_rejects_non_bool(self, tmp_path, monkeypatch):
        client = _make_app(monkeypatch, tmp_path)
        resp = client.put("/api/v3/mesh/config", json={"enabled": "yes"})
        assert resp.status_code == 422

    def test_rejects_unknown_keys(self, tmp_path, monkeypatch):
        client = _make_app(monkeypatch, tmp_path)
        resp = client.put("/api/v3/mesh/config", json={"enabled": True, "secret": "x"})
        assert resp.status_code == 422

    def test_mode_preserved_across_put(self, tmp_path, monkeypatch):
        _write_config(tmp_path, {"mode": "b", "mesh_enabled": True})
        client = _make_app(monkeypatch, tmp_path)
        resp = client.put("/api/v3/mesh/config", json={"enabled": False})
        assert resp.status_code == 200
        saved = _read_config(tmp_path)
        assert saved.get("mode") == "b"


# ---------------------------------------------------------------------------
# Trust config
# ---------------------------------------------------------------------------


class TestTrustConfigGet:
    def test_returns_defaults_when_no_config(self, tmp_path, monkeypatch):
        client = _make_app(monkeypatch, tmp_path)
        resp = client.get("/api/v3/trust/config")
        assert resp.status_code == 200
        data = resp.json()
        assert "use_trust_weighting" in data
        assert "trust_first_party" in data
        assert "promotion_min_trust" in data
        assert data["use_trust_weighting"] is True
        assert data["trust_first_party"] is False
        assert data["promotion_min_trust"] == pytest.approx(0.5)

    def test_reflects_saved_retrieval_values(self, tmp_path, monkeypatch):
        _write_config(tmp_path, {
            "retrieval": {"use_trust_weighting": False},
        })
        client = _make_app(monkeypatch, tmp_path)
        resp = client.get("/api/v3/trust/config")
        assert resp.status_code == 200
        assert resp.json()["use_trust_weighting"] is False

    def test_reflects_saved_injection_values(self, tmp_path, monkeypatch):
        _write_config(tmp_path, {
            "injection": {"trust_first_party": True},
        })
        client = _make_app(monkeypatch, tmp_path)
        resp = client.get("/api/v3/trust/config")
        assert resp.status_code == 200
        assert resp.json()["trust_first_party"] is True

    def test_reflects_saved_consolidation_values(self, tmp_path, monkeypatch):
        _write_config(tmp_path, {
            "consolidation": {"promotion_min_trust": 0.8},
        })
        client = _make_app(monkeypatch, tmp_path)
        resp = client.get("/api/v3/trust/config")
        assert resp.status_code == 200
        assert resp.json()["promotion_min_trust"] == pytest.approx(0.8)


class TestTrustConfigPut:
    def test_persists_use_trust_weighting(self, tmp_path, monkeypatch):
        client = _make_app(monkeypatch, tmp_path)
        resp = client.put("/api/v3/trust/config", json={"use_trust_weighting": False})
        assert resp.status_code == 200
        saved = _read_config(tmp_path)
        assert saved["retrieval"]["use_trust_weighting"] is False

    def test_persists_trust_first_party(self, tmp_path, monkeypatch):
        client = _make_app(monkeypatch, tmp_path)
        resp = client.put("/api/v3/trust/config", json={"trust_first_party": True})
        assert resp.status_code == 200
        saved = _read_config(tmp_path)
        assert saved["injection"]["trust_first_party"] is True

    def test_persists_promotion_min_trust(self, tmp_path, monkeypatch):
        client = _make_app(monkeypatch, tmp_path)
        resp = client.put("/api/v3/trust/config", json={"promotion_min_trust": 0.7})
        assert resp.status_code == 200
        saved = _read_config(tmp_path)
        assert saved["consolidation"]["promotion_min_trust"] == pytest.approx(0.7)

    def test_preserves_other_retrieval_fields(self, tmp_path, monkeypatch):
        """PUT trust/config must not clobber unrelated retrieval fields."""
        _write_config(tmp_path, {
            "retrieval": {
                "use_trust_weighting": True,
                "top_k": 25,
                "min_similarity": 0.6,
            }
        })
        client = _make_app(monkeypatch, tmp_path)
        resp = client.put("/api/v3/trust/config", json={"use_trust_weighting": False})
        assert resp.status_code == 200
        saved = _read_config(tmp_path)
        assert saved["retrieval"]["top_k"] == 25
        assert saved["retrieval"]["min_similarity"] == pytest.approx(0.6)

    def test_preserves_other_injection_fields(self, tmp_path, monkeypatch):
        """PUT trust/config must not clobber unrelated injection fields."""
        _write_config(tmp_path, {
            "injection": {
                "trust_first_party": False,
                "enabled": True,
                "total_budget_tokens_a": 2000,
            }
        })
        client = _make_app(monkeypatch, tmp_path)
        resp = client.put("/api/v3/trust/config", json={"trust_first_party": True})
        assert resp.status_code == 200
        saved = _read_config(tmp_path)
        assert saved["injection"]["enabled"] is True
        assert saved["injection"]["total_budget_tokens_a"] == 2000

    def test_rejects_invalid_promotion_min_trust_above_one(self, tmp_path, monkeypatch):
        client = _make_app(monkeypatch, tmp_path)
        resp = client.put("/api/v3/trust/config", json={"promotion_min_trust": 1.5})
        assert resp.status_code == 422

    def test_rejects_invalid_promotion_min_trust_negative(self, tmp_path, monkeypatch):
        client = _make_app(monkeypatch, tmp_path)
        resp = client.put("/api/v3/trust/config", json={"promotion_min_trust": -0.1})
        assert resp.status_code == 422

    def test_rejects_unknown_keys(self, tmp_path, monkeypatch):
        client = _make_app(monkeypatch, tmp_path)
        resp = client.put("/api/v3/trust/config", json={"hacked_key": True})
        assert resp.status_code == 422

    def test_mode_preserved_across_put(self, tmp_path, monkeypatch):
        _write_config(tmp_path, {"mode": "c", "retrieval": {"use_trust_weighting": True}})
        client = _make_app(monkeypatch, tmp_path)
        resp = client.put("/api/v3/trust/config", json={"use_trust_weighting": False})
        assert resp.status_code == 200
        saved = _read_config(tmp_path)
        assert saved.get("mode") == "c"


# ---------------------------------------------------------------------------
# Forgetting config
# ---------------------------------------------------------------------------


class TestForgettingConfigGet:
    def test_returns_defaults_when_no_config(self, tmp_path, monkeypatch):
        client = _make_app(monkeypatch, tmp_path)
        resp = client.get("/api/v3/forgetting/config")
        assert resp.status_code == 200
        data = resp.json()
        # All ForgettingConfig fields present
        assert "enabled" in data
        assert "alpha" in data
        assert "archive_threshold" in data
        assert "forget_threshold" in data
        assert "scheduler_interval_minutes" in data
        assert "core_memory_immune" in data
        # Defaults
        assert data["enabled"] is True
        assert data["alpha"] == pytest.approx(2.0)
        assert data["archive_threshold"] == pytest.approx(0.2)
        assert data["forget_threshold"] == pytest.approx(0.05)

    def test_reflects_existing_forgetting_section(self, tmp_path, monkeypatch):
        _write_config(tmp_path, {
            "forgetting": {"enabled": False, "alpha": 3.5, "archive_threshold": 0.3}
        })
        client = _make_app(monkeypatch, tmp_path)
        resp = client.get("/api/v3/forgetting/config")
        assert resp.status_code == 200
        data = resp.json()
        assert data["enabled"] is False
        assert data["alpha"] == pytest.approx(3.5)
        assert data["archive_threshold"] == pytest.approx(0.3)


class TestForgettingConfigPut:
    def test_persists_enabled(self, tmp_path, monkeypatch):
        client = _make_app(monkeypatch, tmp_path)
        resp = client.put("/api/v3/forgetting/config", json={"enabled": False})
        assert resp.status_code == 200
        saved = _read_config(tmp_path)
        assert saved["forgetting"]["enabled"] is False

    def test_persists_thresholds(self, tmp_path, monkeypatch):
        client = _make_app(monkeypatch, tmp_path)
        resp = client.put(
            "/api/v3/forgetting/config",
            json={"archive_threshold": 0.25, "forget_threshold": 0.03},
        )
        assert resp.status_code == 200
        saved = _read_config(tmp_path)
        assert saved["forgetting"]["archive_threshold"] == pytest.approx(0.25)
        assert saved["forgetting"]["forget_threshold"] == pytest.approx(0.03)

    def test_partial_update_preserves_other_fields(self, tmp_path, monkeypatch):
        """PUT on forgetting/config with one field must not clobber others."""
        _write_config(tmp_path, {
            "forgetting": {
                "enabled": True,
                "alpha": 3.0,
                "beta": 2.0,
                "archive_threshold": 0.2,
                "forget_threshold": 0.05,
            }
        })
        client = _make_app(monkeypatch, tmp_path)
        resp = client.put("/api/v3/forgetting/config", json={"alpha": 4.0})
        assert resp.status_code == 200
        saved = _read_config(tmp_path)
        fg = saved["forgetting"]
        assert fg["alpha"] == pytest.approx(4.0)
        assert fg["beta"] == pytest.approx(2.0)     # preserved
        assert fg["enabled"] is True                 # preserved

    def test_rejects_archive_threshold_above_one(self, tmp_path, monkeypatch):
        client = _make_app(monkeypatch, tmp_path)
        resp = client.put("/api/v3/forgetting/config", json={"archive_threshold": 1.5})
        assert resp.status_code == 422

    def test_rejects_forget_threshold_above_one(self, tmp_path, monkeypatch):
        client = _make_app(monkeypatch, tmp_path)
        resp = client.put("/api/v3/forgetting/config", json={"forget_threshold": -0.1})
        assert resp.status_code == 422

    def test_rejects_scheduler_interval_zero(self, tmp_path, monkeypatch):
        client = _make_app(monkeypatch, tmp_path)
        resp = client.put(
            "/api/v3/forgetting/config", json={"scheduler_interval_minutes": 0}
        )
        assert resp.status_code == 422

    def test_rejects_unknown_keys(self, tmp_path, monkeypatch):
        client = _make_app(monkeypatch, tmp_path)
        resp = client.put("/api/v3/forgetting/config", json={"nonsense": 99})
        assert resp.status_code == 422

    def test_mode_preserved_across_put(self, tmp_path, monkeypatch):
        _write_config(tmp_path, {"mode": "b", "forgetting": {"enabled": True}})
        client = _make_app(monkeypatch, tmp_path)
        resp = client.put("/api/v3/forgetting/config", json={"enabled": False})
        assert resp.status_code == 200
        saved = _read_config(tmp_path)
        assert saved.get("mode") == "b"


# ---------------------------------------------------------------------------
# Evolution disable
# ---------------------------------------------------------------------------


class TestEvolutionDisable:
    def test_disable_persists_enabled_false(self, tmp_path, monkeypatch):
        _write_config(tmp_path, {"evolution": {"enabled": True, "backend": "auto"}})
        client = _make_evo_app(monkeypatch, tmp_path)
        resp = client.post("/api/evolution/disable")
        assert resp.status_code == 200
        saved = _read_config(tmp_path)
        assert saved["evolution"]["enabled"] is False

    def test_disable_when_no_existing_config(self, tmp_path, monkeypatch):
        client = _make_evo_app(monkeypatch, tmp_path)
        resp = client.post("/api/evolution/disable")
        assert resp.status_code == 200
        saved = _read_config(tmp_path)
        assert saved["evolution"]["enabled"] is False

    def test_disable_preserves_other_evolution_keys(self, tmp_path, monkeypatch):
        _write_config(tmp_path, {
            "evolution": {"enabled": True, "backend": "claude", "max_evolutions_per_cycle": 5}
        })
        client = _make_evo_app(monkeypatch, tmp_path)
        resp = client.post("/api/evolution/disable")
        assert resp.status_code == 200
        saved = _read_config(tmp_path)
        evo = saved["evolution"]
        assert evo["enabled"] is False
        assert evo["backend"] == "claude"              # preserved
        assert evo["max_evolutions_per_cycle"] == 5    # preserved

    def test_disable_preserves_mode(self, tmp_path, monkeypatch):
        _write_config(tmp_path, {"mode": "c", "evolution": {"enabled": True}})
        client = _make_evo_app(monkeypatch, tmp_path)
        resp = client.post("/api/evolution/disable")
        assert resp.status_code == 200
        saved = _read_config(tmp_path)
        assert saved.get("mode") == "c"

    def test_disable_returns_ok_true(self, tmp_path, monkeypatch):
        client = _make_evo_app(monkeypatch, tmp_path)
        resp = client.post("/api/evolution/disable")
        assert resp.status_code == 200
        assert resp.json()["ok"] is True
