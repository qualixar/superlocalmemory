# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3

"""Tests for CozoDBGraphBackend — Sprint 2."""

from __future__ import annotations

import pytest
import shutil
from pathlib import Path

from superlocalmemory.graph.cozo_backend import (
    CozoDBGraphBackend,
    CozoDBNotAvailable,
    _COZO_AVAILABLE,
)

# CozoDB is an optional backend (pip install superlocalmemory[cozo]). When it
# is absent these tests cannot run — skip cleanly instead of erroring at
# fixture setup, so the suite stays honestly green on installs without it.
pytestmark = pytest.mark.skipif(
    not _COZO_AVAILABLE,
    reason="CozoDB optional dependency not installed (pip install superlocalmemory[cozo])",
)


@pytest.fixture
def backend():
    """Create temporary CozoDB backend."""
    path = "/tmp/test_slm_cozo_backend"
    be = CozoDBGraphBackend(path)
    yield be
    be.close()
    shutil.rmtree(path, ignore_errors=True)


@pytest.fixture
def populated(backend):
    """Backend with test graph: e1→e2→e3, e1→e3, e4→e1."""
    backend.add_entity("e1", "Node1", "concept")
    backend.add_entity("e2", "Node2", "concept")
    backend.add_entity("e3", "Node3", "concept")
    backend.add_entity("e4", "Node4", "isolated")
    backend.add_edge("e1", "e2", "links", 1.0)
    backend.add_edge("e2", "e3", "links", 1.0)
    backend.add_edge("e1", "e3", "direct", 0.5)
    return backend


class TestLifecycle:
    """Open, close, health check."""

    def test_open_and_close(self):
        be = CozoDBGraphBackend("/tmp/test_cozo_lifecycle")
        health = be.health_check()
        assert health["status"] == "active"
        be.close()
        shutil.rmtree("/tmp/test_cozo_lifecycle", ignore_errors=True)

    def test_entities_persist(self, populated):
        health = populated.health_check()
        assert health["entities"] == 4
        assert health["edges"] == 3


class TestSpreadingActivation:
    """BFS traversal with decay."""

    def test_seeds_get_score_one(self, populated):
        results = populated.spreading_activation(["e1"], depth=2, top_k=10)
        scores = dict(results)
        assert scores["e1"] == 1.0

    def test_decay_per_hop(self, populated):
        results = populated.spreading_activation(["e1"], depth=2, decay=0.5, top_k=10)
        scores = dict(results)
        assert scores["e1"] == 1.0
        assert scores["e2"] < 1.0  # 1 hop → decay applied
        assert scores.get("e3", 0) <= scores["e2"]  # 2 hops → more decay

    def test_empty_seeds(self, populated):
        assert populated.spreading_activation([]) == []

    def test_depth_limit(self, populated):
        results = populated.spreading_activation(["e1"], depth=1, top_k=10)
        scores = dict(results)
        # depth=1 means only e1+e2 (1 hop). e3 is 2 hops away, excluded.
        assert "e2" in scores
        # e3 may or may not appear depending on direct edge e1→e3

    def test_isolated_node(self, populated):
        results = populated.spreading_activation(["e4"], depth=3, top_k=10)
        scores = dict(results)
        assert scores["e4"] == 1.0
        assert len(scores) == 1  # Only e4, no edges


class TestPageRank:
    """Iterative PageRank."""

    def test_pagerank_returns_all_entities(self, populated):
        pr = populated.pagerank()
        assert len(pr) == 4
        assert all(v > 0 for v in pr.values())

    def test_empty_backend(self, backend):
        pr = backend.pagerank()
        assert pr == {}


class TestCommunityDetection:
    """Connected components."""

    def test_connected_component(self, populated):
        communities = populated.community_detect()
        assert len(set(communities.values())) == 2  # {e1,e2,e3} + {e4}

    def test_same_community(self, populated):
        communities = populated.community_detect()
        assert communities["e1"] == communities["e2"] == communities["e3"]
        assert communities["e4"] != communities["e1"]


class TestShortestPath:
    """BFS shortest path."""

    def test_same_node(self, populated):
        assert populated.shortest_path("e1", "e1") == ["e1"]

    def test_direct_edge(self, populated):
        path = populated.shortest_path("e1", "e2")
        assert path == ["e1", "e2"]

    def test_two_hop(self, populated):
        path = populated.shortest_path("e1", "e3")
        assert path in (["e1", "e3"], ["e1", "e2", "e3"])

    def test_no_path(self, populated):
        assert populated.shortest_path("e1", "e4") == []


class TestTierSync:
    """sync_tier_changes modifies entity tiers."""

    def test_tier_sync_no_crash(self, populated):
        """sync_tier_changes should not raise."""
        populated.sync_tier_changes(added=["e1"], removed=["e4"])
