# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""TDD tests for Workstream G — Graph Thinning Config (#84).

Covers:
  (a) GraphPruningConfig round-trip: set graph_pruning, save(), load() → values persist
  (b) prune_graph respects min_edge_weight (edges below weight removed) + max_degree
  (c) GET /api/v3/graph/config returns config; PUT updates + persists (survives reload)

RED phase: all tests should FAIL before implementation.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

# ===========================================================================
# (a) Config round-trip tests
# ===========================================================================


class TestGraphPruningConfigDataclass:
    """GraphPruningConfig defaults match current _MAX_DEGREE_PER_NODE behaviour."""

    def test_default_max_degree_is_100(self):
        from superlocalmemory.core.config import GraphPruningConfig
        cfg = GraphPruningConfig()
        assert cfg.max_degree_per_node == 100

    def test_default_min_edge_weight_is_zero(self):
        from superlocalmemory.core.config import GraphPruningConfig
        cfg = GraphPruningConfig()
        assert cfg.min_edge_weight == 0.0

    def test_default_enabled_is_true(self):
        from superlocalmemory.core.config import GraphPruningConfig
        cfg = GraphPruningConfig()
        assert cfg.enabled is True

    def test_custom_values_stored(self):
        from superlocalmemory.core.config import GraphPruningConfig
        cfg = GraphPruningConfig(max_degree_per_node=50, min_edge_weight=0.3, enabled=False)
        assert cfg.max_degree_per_node == 50
        assert cfg.min_edge_weight == pytest.approx(0.3)
        assert cfg.enabled is False


class TestSLMConfigHasGraphPruningField:
    """SLMConfig exposes graph_pruning with correct defaults."""

    def test_slm_config_has_graph_pruning_field(self):
        from superlocalmemory.core.config import SLMConfig
        from superlocalmemory.storage.models import Mode
        cfg = SLMConfig.for_mode(Mode.A)
        assert hasattr(cfg, "graph_pruning")

    def test_graph_pruning_defaults_are_100_and_zero(self):
        from superlocalmemory.core.config import SLMConfig
        from superlocalmemory.storage.models import Mode
        cfg = SLMConfig.for_mode(Mode.A)
        assert cfg.graph_pruning.max_degree_per_node == 100
        assert cfg.graph_pruning.min_edge_weight == pytest.approx(0.0)
        assert cfg.graph_pruning.enabled is True


class TestGraphPruningConfigRoundTrip:
    """save() writes graph_pruning section; load() reads it back correctly."""

    def test_save_persists_graph_pruning_section(self, tmp_path):
        from superlocalmemory.core.config import GraphPruningConfig, SLMConfig
        from superlocalmemory.storage.models import Mode

        cfg = SLMConfig.for_mode(Mode.A, base_dir=tmp_path)
        cfg.graph_pruning = GraphPruningConfig(max_degree_per_node=50, min_edge_weight=0.3)
        cfg.save(tmp_path / "config.json")

        raw = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
        assert "graph_pruning" in raw
        assert raw["graph_pruning"]["max_degree_per_node"] == 50
        assert raw["graph_pruning"]["min_edge_weight"] == pytest.approx(0.3)

    def test_load_restores_saved_values(self, tmp_path):
        from superlocalmemory.core.config import GraphPruningConfig, SLMConfig
        from superlocalmemory.storage.models import Mode

        cfg = SLMConfig.for_mode(Mode.A, base_dir=tmp_path)
        cfg.graph_pruning = GraphPruningConfig(max_degree_per_node=75, min_edge_weight=0.25)
        cfg.save(tmp_path / "config.json")

        loaded = SLMConfig.load(tmp_path / "config.json")
        assert loaded.graph_pruning.max_degree_per_node == 75
        assert loaded.graph_pruning.min_edge_weight == pytest.approx(0.25)

    def test_save_then_load_enabled_false(self, tmp_path):
        from superlocalmemory.core.config import GraphPruningConfig, SLMConfig
        from superlocalmemory.storage.models import Mode

        cfg = SLMConfig.for_mode(Mode.A, base_dir=tmp_path)
        cfg.graph_pruning = GraphPruningConfig(enabled=False)
        cfg.save(tmp_path / "config.json")

        loaded = SLMConfig.load(tmp_path / "config.json")
        assert loaded.graph_pruning.enabled is False


class TestGraphPruningConfigOldConfigCompat:
    """load() on config.json without graph_pruning key silently defaults."""

    def test_old_config_missing_key_uses_defaults(self, tmp_path):
        from superlocalmemory.core.config import SLMConfig

        # Write a minimal config.json without graph_pruning
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps({"mode": "a"}), encoding="utf-8")

        loaded = SLMConfig.load(config_path)
        assert loaded.graph_pruning.max_degree_per_node == 100
        assert loaded.graph_pruning.min_edge_weight == pytest.approx(0.0)
        assert loaded.graph_pruning.enabled is True


class TestGraphPruningConfigValidation:
    """load() clamps out-of-range values rather than crashing."""

    def test_min_edge_weight_clamped_below_zero(self, tmp_path):
        from superlocalmemory.core.config import SLMConfig

        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps({
            "mode": "a",
            "graph_pruning": {"max_degree_per_node": 100, "min_edge_weight": -0.5, "enabled": True}
        }), encoding="utf-8")
        loaded = SLMConfig.load(config_path)
        assert loaded.graph_pruning.min_edge_weight == pytest.approx(0.0)

    def test_min_edge_weight_clamped_above_one(self, tmp_path):
        from superlocalmemory.core.config import SLMConfig

        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps({
            "mode": "a",
            "graph_pruning": {"max_degree_per_node": 100, "min_edge_weight": 1.5, "enabled": True}
        }), encoding="utf-8")
        loaded = SLMConfig.load(config_path)
        assert loaded.graph_pruning.min_edge_weight == pytest.approx(1.0)

    def test_max_degree_zero_reset_to_100(self, tmp_path):
        from superlocalmemory.core.config import SLMConfig

        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps({
            "mode": "a",
            "graph_pruning": {"max_degree_per_node": 0, "min_edge_weight": 0.0, "enabled": True}
        }), encoding="utf-8")
        loaded = SLMConfig.load(config_path)
        assert loaded.graph_pruning.max_degree_per_node == 100


# ===========================================================================
# (b) prune_graph parameter tests
# ===========================================================================


def _raw_insert_edge(conn, eid, src, tgt, weight, profile_id="default"):
    """Insert a raw graph_edge for testing (source/target are treated as entity IDs)."""
    conn.execute(
        "INSERT INTO graph_edges (edge_id, profile_id, source_id, target_id, "
        "edge_type, weight, created_at) VALUES (?, ?, ?, ?, 'semantic', ?, '2026-01-01')",
        (eid, profile_id, src, tgt, weight),
    )


def _raw_insert_entity(conn, entity_id, profile_id="default"):
    """Insert a canonical_entity so edges referencing it are not treated as orphans."""
    conn.execute(
        "INSERT OR IGNORE INTO canonical_entities "
        "(entity_id, profile_id, canonical_name, entity_type) "
        "VALUES (?, ?, ?, 'CONCEPT')",
        (entity_id, profile_id, entity_id),
    )


@pytest.fixture()
def pruner_db(tmp_path):
    """Return a DatabaseManager with schema initialised."""
    from superlocalmemory.storage import schema as real_schema
    from superlocalmemory.storage.database import DatabaseManager

    mgr = DatabaseManager(tmp_path / "prune_test.db")
    mgr.initialize(real_schema)
    return mgr


class TestPruneGraphMinEdgeWeight:
    """prune_graph removes edges below min_edge_weight floor."""

    def test_removes_edges_below_weight_floor(self, pruner_db):
        from superlocalmemory.core.graph_pruner import prune_graph

        conn = sqlite3.connect(str(pruner_db.db_path))
        conn.row_factory = sqlite3.Row
        # Insert entities so edges are not treated as orphans
        for eid in ("fa", "fb", "fc", "fd"):
            _raw_insert_entity(conn, eid)
        _raw_insert_edge(conn, "e1", "fa", "fb", 0.1)
        _raw_insert_edge(conn, "e2", "fc", "fd", 0.5)
        _raw_insert_edge(conn, "e3", "fa", "fd", 0.9)
        conn.commit()
        conn.close()

        prune_graph(pruner_db, "default", min_edge_weight=0.4)

        remaining = pruner_db.execute(
            "SELECT edge_id FROM graph_edges WHERE profile_id = 'default'", ()
        )
        edge_ids = {dict(r)["edge_id"] for r in remaining}
        assert "e1" not in edge_ids, "Edge with weight 0.1 should be pruned (below 0.4)"
        assert "e2" in edge_ids, "Edge with weight 0.5 should survive"
        assert "e3" in edge_ids, "Edge with weight 0.9 should survive"

    def test_zero_min_weight_removes_nothing(self, pruner_db):
        """Default min_edge_weight=0.0 must not remove any edge by weight."""
        from superlocalmemory.core.graph_pruner import prune_graph

        conn = sqlite3.connect(str(pruner_db.db_path))
        conn.row_factory = sqlite3.Row
        for eid in ("fa", "fb"):
            _raw_insert_entity(conn, eid)
        _raw_insert_edge(conn, "e1", "fa", "fb", 0.01)
        conn.commit()
        conn.close()

        prune_graph(pruner_db, "default", min_edge_weight=0.0)

        remaining = pruner_db.execute(
            "SELECT edge_id FROM graph_edges WHERE profile_id = 'default'", ()
        )
        edge_ids = {dict(r)["edge_id"] for r in remaining}
        assert "e1" in edge_ids, "Zero floor must not remove any edge"

    def test_stats_include_low_weight_removed(self, pruner_db):
        """prune_graph stats should count edges removed by weight floor."""
        from superlocalmemory.core.graph_pruner import prune_graph

        conn = sqlite3.connect(str(pruner_db.db_path))
        conn.row_factory = sqlite3.Row
        for eid in ("fa", "fb", "fc", "fd"):
            _raw_insert_entity(conn, eid)
        _raw_insert_edge(conn, "e1", "fa", "fb", 0.1)
        _raw_insert_edge(conn, "e2", "fc", "fd", 0.8)
        conn.commit()
        conn.close()

        stats = prune_graph(pruner_db, "default", min_edge_weight=0.5)
        assert stats.get("low_weight_removed", 0) >= 1, (
            "stats must include at least 1 low_weight_removed"
        )


class TestPruneGraphMaxDegree:
    """prune_graph respects custom max_degree parameter."""

    def test_max_degree_override_limits_hub_edges(self, pruner_db):
        from superlocalmemory.core.graph_pruner import prune_graph

        conn = sqlite3.connect(str(pruner_db.db_path))
        hub = "hub_node"
        _raw_insert_entity(conn, hub)
        # Insert 20 source entities that all point to hub (20 in-edges for hub)
        for i in range(20):
            eid = f"src_{i}"
            _raw_insert_entity(conn, eid)
            _raw_insert_edge(conn, f"e_{i}", eid, hub, 0.5 + i * 0.02)
        conn.commit()
        conn.close()

        prune_graph(pruner_db, "default", cap_degree=True, max_degree=10)

        rows = pruner_db.execute(
            "SELECT COUNT(*) AS cnt FROM graph_edges "
            "WHERE target_id = ? AND profile_id = 'default'",
            (hub,),
        )
        count = int(dict(rows[0])["cnt"])
        assert count <= 10, f"Expected ≤ 10 in-edges for hub, got {count}"

    def test_default_max_degree_preserves_100_behaviour(self, pruner_db):
        """Calling prune_graph without max_degree must behave exactly as before (default 100)."""
        from superlocalmemory.core.graph_pruner import _MAX_DEGREE_PER_NODE, prune_graph

        # Confirm the default bakes in exactly 100
        assert _MAX_DEGREE_PER_NODE == 100

        conn = sqlite3.connect(str(pruner_db.db_path))
        hub = "hub2"
        _raw_insert_entity(conn, hub)
        for i in range(50):
            eid = f"s_{i}"
            _raw_insert_entity(conn, eid)
            conn.execute(
                "INSERT INTO graph_edges (edge_id, profile_id, source_id, target_id, "
                "edge_type, weight, created_at) VALUES (?, 'default', ?, ?, 'semantic', ?, '2026-01-01')",
                (f"ev_{i}", eid, hub, 0.5 + i * 0.001),
            )
        conn.commit()
        conn.close()

        stats = prune_graph(pruner_db, "default", cap_degree=True)
        # 50 edges < 100 degree cap → none should be removed by hub-cap
        assert stats["hub_edges_removed"] == 0


# ===========================================================================
# (c) API endpoint tests
# ===========================================================================


fastapi = pytest.importorskip("fastapi", reason="fastapi not installed")


def _make_app(monkeypatch, tmp_path: Path):
    """Build a TestClient with config_api router, MEMORY_DIR patched."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    monkeypatch.setattr(
        "superlocalmemory.server.routes.config_api.MEMORY_DIR", tmp_path
    )
    from superlocalmemory.server.routes.config_api import router

    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def _read_raw(tmp_path: Path) -> dict:
    p = tmp_path / "config.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def _write_raw(tmp_path: Path, data: dict) -> None:
    (tmp_path / "config.json").write_text(json.dumps(data, indent=2), encoding="utf-8")


class TestGraphConfigGet:
    """GET /api/v3/graph/config returns current graph pruning configuration."""

    def test_returns_defaults_when_no_config(self, tmp_path, monkeypatch):
        client = _make_app(monkeypatch, tmp_path)
        resp = client.get("/api/v3/graph/config")
        assert resp.status_code == 200
        data = resp.json()
        assert data["max_degree_per_node"] == 100
        assert data["min_edge_weight"] == pytest.approx(0.0)
        assert data["enabled"] is True

    def test_reflects_saved_values(self, tmp_path, monkeypatch):
        _write_raw(tmp_path, {
            "graph_pruning": {
                "max_degree_per_node": 60,
                "min_edge_weight": 0.2,
                "enabled": False,
            }
        })
        client = _make_app(monkeypatch, tmp_path)
        resp = client.get("/api/v3/graph/config")
        assert resp.status_code == 200
        data = resp.json()
        assert data["max_degree_per_node"] == 60
        assert data["min_edge_weight"] == pytest.approx(0.2)
        assert data["enabled"] is False


class TestGraphConfigPut:
    """PUT /api/v3/graph/config persists and survives reload."""

    def test_persists_max_degree_per_node(self, tmp_path, monkeypatch):
        client = _make_app(monkeypatch, tmp_path)
        resp = client.put("/api/v3/graph/config", json={"max_degree_per_node": 75})
        assert resp.status_code == 200
        saved = _read_raw(tmp_path)
        assert saved["graph_pruning"]["max_degree_per_node"] == 75

    def test_persists_min_edge_weight(self, tmp_path, monkeypatch):
        client = _make_app(monkeypatch, tmp_path)
        resp = client.put("/api/v3/graph/config", json={"min_edge_weight": 0.15})
        assert resp.status_code == 200
        saved = _read_raw(tmp_path)
        assert saved["graph_pruning"]["min_edge_weight"] == pytest.approx(0.15)

    def test_persists_enabled_false(self, tmp_path, monkeypatch):
        client = _make_app(monkeypatch, tmp_path)
        resp = client.put("/api/v3/graph/config", json={"enabled": False})
        assert resp.status_code == 200
        saved = _read_raw(tmp_path)
        assert saved["graph_pruning"]["enabled"] is False

    def test_partial_update_preserves_other_fields(self, tmp_path, monkeypatch):
        """PUT with one field must not reset the other fields to defaults."""
        _write_raw(tmp_path, {
            "graph_pruning": {
                "max_degree_per_node": 60,
                "min_edge_weight": 0.3,
                "enabled": True,
            }
        })
        client = _make_app(monkeypatch, tmp_path)
        resp = client.put("/api/v3/graph/config", json={"max_degree_per_node": 80})
        assert resp.status_code == 200
        saved = _read_raw(tmp_path)
        gp = saved["graph_pruning"]
        assert gp["max_degree_per_node"] == 80
        assert gp["min_edge_weight"] == pytest.approx(0.3)  # preserved
        assert gp["enabled"] is True  # preserved

    def test_survives_reload(self, tmp_path, monkeypatch):
        """Values persisted by PUT must still be present after a fresh load()."""
        client = _make_app(monkeypatch, tmp_path)
        client.put("/api/v3/graph/config", json={"max_degree_per_node": 42, "min_edge_weight": 0.1})

        from superlocalmemory.core.config import SLMConfig
        reloaded = SLMConfig.load(tmp_path / "config.json")
        assert reloaded.graph_pruning.max_degree_per_node == 42
        assert reloaded.graph_pruning.min_edge_weight == pytest.approx(0.1)

    def test_mode_preserved_across_put(self, tmp_path, monkeypatch):
        _write_raw(tmp_path, {"mode": "b", "graph_pruning": {"max_degree_per_node": 100}})
        client = _make_app(monkeypatch, tmp_path)
        resp = client.put("/api/v3/graph/config", json={"max_degree_per_node": 50})
        assert resp.status_code == 200
        saved = _read_raw(tmp_path)
        assert saved.get("mode") == "b"

    def test_rejects_max_degree_zero(self, tmp_path, monkeypatch):
        client = _make_app(monkeypatch, tmp_path)
        resp = client.put("/api/v3/graph/config", json={"max_degree_per_node": 0})
        assert resp.status_code == 422

    def test_rejects_negative_max_degree(self, tmp_path, monkeypatch):
        client = _make_app(monkeypatch, tmp_path)
        resp = client.put("/api/v3/graph/config", json={"max_degree_per_node": -5})
        assert resp.status_code == 422

    def test_rejects_min_edge_weight_negative(self, tmp_path, monkeypatch):
        client = _make_app(monkeypatch, tmp_path)
        resp = client.put("/api/v3/graph/config", json={"min_edge_weight": -0.1})
        assert resp.status_code == 422

    def test_rejects_min_edge_weight_above_one(self, tmp_path, monkeypatch):
        client = _make_app(monkeypatch, tmp_path)
        resp = client.put("/api/v3/graph/config", json={"min_edge_weight": 1.5})
        assert resp.status_code == 422

    def test_rejects_unknown_keys(self, tmp_path, monkeypatch):
        client = _make_app(monkeypatch, tmp_path)
        resp = client.put("/api/v3/graph/config", json={"unknown_key": 99})
        assert resp.status_code == 422

    def test_response_includes_all_fields(self, tmp_path, monkeypatch):
        client = _make_app(monkeypatch, tmp_path)
        resp = client.put("/api/v3/graph/config", json={"max_degree_per_node": 50})
        assert resp.status_code == 200
        data = resp.json()
        assert "max_degree_per_node" in data
        assert "min_edge_weight" in data
        assert "enabled" in data
