# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Tests for CodeGraph MCP tools (Phase 5).

Tests key tools: build_code_graph, get_blast_radius, query_graph,
semantic_search_code, list_graph_stats, code_memory_search,
code_stale_check. Includes error cases.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from superlocalmemory.code_graph.database import CodeGraphDatabase
from superlocalmemory.code_graph.models import (
    CodeMemoryLink,
    EdgeKind,
    FileRecord,
    GraphEdge,
    GraphNode,
    LinkType,
    NodeKind,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _insert_test_graph(db: CodeGraphDatabase) -> dict[str, str]:
    """Insert a small test graph. Returns {name: node_id}."""
    nodes = {
        "auth_handler": GraphNode(
            node_id="n1",
            kind=NodeKind.FILE,
            name="handler.py",
            qualified_name="src/auth/handler.py",
            file_path="src/auth/handler.py",
            language="python",
        ),
        "authenticate_user": GraphNode(
            node_id="n2",
            kind=NodeKind.FUNCTION,
            name="authenticate_user",
            qualified_name="src/auth/handler.py::authenticate_user",
            file_path="src/auth/handler.py",
            line_start=10,
            line_end=80,
            language="python",
        ),
        "validate_token": GraphNode(
            node_id="n3",
            kind=NodeKind.FUNCTION,
            name="validate_token",
            qualified_name="src/auth/utils.py::validate_token",
            file_path="src/auth/utils.py",
            line_start=5,
            line_end=20,
            language="python",
        ),
        "UserService": GraphNode(
            node_id="n4",
            kind=NodeKind.CLASS,
            name="UserService",
            qualified_name="src/user/service.py::UserService",
            file_path="src/user/service.py",
            language="python",
        ),
    }

    for node in nodes.values():
        db.upsert_node(node)

    # Edges
    db.upsert_edge(GraphEdge(
        edge_id="e1",
        kind=EdgeKind.CALLS,
        source_node_id="n2",
        target_node_id="n3",
        file_path="src/auth/handler.py",
        line=15,
    ))
    db.upsert_edge(GraphEdge(
        edge_id="e2",
        kind=EdgeKind.CONTAINS,
        source_node_id="n1",
        target_node_id="n2",
        file_path="src/auth/handler.py",
        line=10,
    ))

    # File record
    db.upsert_file_record(FileRecord(
        file_path="src/auth/handler.py",
        content_hash="abc",
        mtime=1000.0,
        language="python",
        node_count=2,
        edge_count=2,
    ))

    return {name: node.node_id for name, node in nodes.items()}


def _run_async(coro):
    """Run an async function synchronously."""
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Fixture: tools with mock server
# ---------------------------------------------------------------------------


class _ToolRegistry:
    """Captures tools registered via @server.tool()."""

    def __init__(self):
        self.tools: dict[str, callable] = {}

    def tool(self, *args, **kwargs):
        def decorator(func):
            self.tools[func.__name__] = func
            return func
        return decorator

    def __getattr__(self, name):
        return MagicMock()


@pytest.fixture
def tools(db: CodeGraphDatabase) -> dict:
    """Register all 22 tools and return {name: func} dict."""
    from superlocalmemory.mcp import tools_code_graph

    # Reset service singleton
    tools_code_graph._service = None

    registry = _ToolRegistry()
    tools_code_graph.register_code_graph_tools(registry, MagicMock())
    return registry.tools


@pytest.fixture
def tools_with_graph(db: CodeGraphDatabase, tools: dict) -> tuple[dict, dict]:
    """Tools with a pre-populated graph."""
    from superlocalmemory.mcp import tools_code_graph
    from superlocalmemory.code_graph.config import CodeGraphConfig
    from superlocalmemory.code_graph.service import CodeGraphService

    ids = _insert_test_graph(db)

    # Set up the service singleton to point to our test DB
    config = CodeGraphConfig(
        enabled=True,
        db_path=db.db_path,
    )
    svc = CodeGraphService(config)
    # Force DB to use our pre-populated one
    svc._db = db
    tools_code_graph._service = svc

    return tools, ids


# ---------------------------------------------------------------------------
# Registration tests
# ---------------------------------------------------------------------------


class TestRegistration:
    """Test that all 22 tools are registered."""

    def test_all_22_tools_registered(self, tools: dict) -> None:
        assert len(tools) == 22

    def test_graph_tools_registered(self, tools: dict) -> None:
        expected = {
            "build_code_graph", "update_code_graph", "get_blast_radius",
            "get_review_context", "query_graph", "semantic_search_code",
            "list_graph_stats", "find_large_functions", "list_flows",
            "get_flow", "get_affected_flows", "list_communities",
            "get_community", "get_architecture_overview", "detect_changes",
            "refactor_preview", "apply_refactor",
        }
        assert expected.issubset(set(tools.keys()))

    def test_bridge_tools_registered(self, tools: dict) -> None:
        expected = {
            "code_memory_search", "code_entity_history",
            "enrich_blast_radius", "code_stale_check", "link_memory_to_code",
        }
        assert expected.issubset(set(tools.keys()))


# ---------------------------------------------------------------------------
# Error cases: graph not built
# ---------------------------------------------------------------------------


class TestGraphNotBuilt:
    """Test graceful errors when graph is not built."""

    def test_list_graph_stats_no_graph(self, tools: dict) -> None:
        from superlocalmemory.mcp import tools_code_graph
        tools_code_graph._service = None
        result = _run_async(tools["list_graph_stats"]())
        # Either returns error or success with built=False
        assert isinstance(result, dict)
        assert "success" in result

    def test_query_graph_invalid_pattern(self, tools_with_graph) -> None:
        tools, ids = tools_with_graph
        result = _run_async(tools["query_graph"](
            pattern="invalid_pattern", target="foo"
        ))
        assert result["success"] is False
        assert "Invalid pattern" in result["error"]


# ---------------------------------------------------------------------------
# Graph tool tests
# ---------------------------------------------------------------------------


class TestQueryGraph:
    """Test query_graph tool."""

    def test_callers_of(self, tools_with_graph) -> None:
        tools, ids = tools_with_graph
        result = _run_async(tools["query_graph"](
            pattern="callers_of",
            target="src/auth/utils.py::validate_token",
        ))
        assert result["success"] is True
        assert result["pattern"] == "callers_of"
        # authenticate_user calls validate_token
        assert len(result["results"]) >= 1

    def test_callees_of(self, tools_with_graph) -> None:
        tools, ids = tools_with_graph
        result = _run_async(tools["query_graph"](
            pattern="callees_of",
            target="src/auth/handler.py::authenticate_user",
        ))
        assert result["success"] is True
        assert len(result["results"]) >= 1

    def test_contains(self, tools_with_graph) -> None:
        tools, ids = tools_with_graph
        result = _run_async(tools["query_graph"](
            pattern="contains",
            target="src/auth/handler.py",
        ))
        assert result["success"] is True

    def test_target_not_found(self, tools_with_graph) -> None:
        tools, ids = tools_with_graph
        result = _run_async(tools["query_graph"](
            pattern="callers_of",
            target="nonexistent_function",
        ))
        assert result["success"] is True
        assert result["results"] == []


class TestSemanticSearch:
    """Test semantic_search_code tool."""

    def test_search_by_name(self, tools_with_graph) -> None:
        tools, ids = tools_with_graph
        result = _run_async(tools["semantic_search_code"](
            query="authenticate",
        ))
        assert result["success"] is True
        assert isinstance(result["results"], list)

    def test_search_with_kind_filter(self, tools_with_graph) -> None:
        tools, ids = tools_with_graph
        result = _run_async(tools["semantic_search_code"](
            query="handler",
            kind="function",
        ))
        assert result["success"] is True


class TestListGraphStats:
    """Test list_graph_stats tool."""

    def test_returns_stats(self, tools_with_graph) -> None:
        tools, ids = tools_with_graph
        result = _run_async(tools["list_graph_stats"]())
        assert result["success"] is True
        assert result["total_nodes"] >= 4
        assert result["total_edges"] >= 2
        assert result["built"] is True


class TestFindLargeFunctions:
    """Test find_large_functions tool."""

    def test_finds_large_function(self, tools_with_graph) -> None:
        tools, ids = tools_with_graph
        result = _run_async(tools["find_large_functions"](threshold=50))
        assert result["success"] is True
        # authenticate_user has 70 lines (80-10)
        assert len(result["functions"]) >= 1
        assert result["functions"][0]["lines"] >= 50

    def test_no_large_functions(self, tools_with_graph) -> None:
        tools, ids = tools_with_graph
        result = _run_async(tools["find_large_functions"](threshold=500))
        assert result["success"] is True
        assert len(result["functions"]) == 0


class TestGetBlastRadius:
    """Test get_blast_radius tool."""

    def test_basic_blast_radius(self, tools_with_graph) -> None:
        tools, ids = tools_with_graph
        result = _run_async(tools["get_blast_radius"](
            changed_files="src/auth/handler.py",
        ))
        assert result["success"] is True
        assert isinstance(result["changed_nodes"], list)
        assert isinstance(result["impacted_nodes"], list)

    def test_no_files_error(self, tools_with_graph) -> None:
        tools, ids = tools_with_graph
        result = _run_async(tools["get_blast_radius"](
            changed_files="",
        ))
        assert result["success"] is False


# ---------------------------------------------------------------------------
# Bridge tool tests
# ---------------------------------------------------------------------------


class TestCodeMemorySearch:
    """Test code_memory_search bridge tool."""

    def test_search_with_no_links(self, tools_with_graph) -> None:
        tools, ids = tools_with_graph
        result = _run_async(tools["code_memory_search"](
            code_entity="authenticate_user",
        ))
        assert result["success"] is True
        assert result["memories"] == []

    def test_search_with_links(self, tools_with_graph, db) -> None:
        tools, ids = tools_with_graph
        # Add a link
        link = CodeMemoryLink(
            link_id="lnk1",
            code_node_id="n2",
            slm_fact_id="fact_123",
            link_type=LinkType.BUG_FIX,
            confidence=0.9,
            created_at="2026-01-01T00:00:00",
        )
        db.upsert_link(link)

        result = _run_async(tools["code_memory_search"](
            code_entity="authenticate_user",
        ))
        assert result["success"] is True
        assert len(result["memories"]) == 1
        assert result["memories"][0]["fact_id"] == "fact_123"

    def test_search_entity_not_found(self, tools_with_graph) -> None:
        tools, ids = tools_with_graph
        result = _run_async(tools["code_memory_search"](
            code_entity="nonexistent_func",
        ))
        assert result["success"] is True
        assert result["matched_node"] is None


class TestCodeStaleCheck:
    """Test code_stale_check bridge tool."""

    def test_no_stale(self, tools_with_graph) -> None:
        tools, ids = tools_with_graph
        result = _run_async(tools["code_stale_check"]())
        assert result["success"] is True
        assert result["total_stale"] == 0

    def test_with_stale_link(self, tools_with_graph, db) -> None:
        tools, ids = tools_with_graph
        # Add a stale link
        link = CodeMemoryLink(
            link_id="lnk_stale",
            code_node_id="n2",
            slm_fact_id="fact_old",
            link_type=LinkType.MENTIONS,
            confidence=0.8,
            created_at="2026-01-01T00:00:00",
            last_verified="2026-01-01T00:00:00",
            is_stale=True,
        )
        db.upsert_link(link)

        result = _run_async(tools["code_stale_check"]())
        assert result["success"] is True
        assert result["total_stale"] == 1
        assert result["stale_memories"][0]["fact_id"] == "fact_old"


class TestLinkMemoryToCode:
    """Test link_memory_to_code bridge tool."""

    def test_valid_link(self, tools_with_graph) -> None:
        tools, ids = tools_with_graph
        result = _run_async(tools["link_memory_to_code"](
            fact_id="my_fact_1",
            code_entity="authenticate_user",
            link_type="bug_fix",
        ))
        assert result["success"] is True
        assert result["confidence"] == 1.0
        assert "link_id" in result

    def test_invalid_link_type(self, tools_with_graph) -> None:
        tools, ids = tools_with_graph
        result = _run_async(tools["link_memory_to_code"](
            fact_id="my_fact",
            code_entity="authenticate_user",
            link_type="invalid_type",
        ))
        assert result["success"] is False
        assert "Invalid link_type" in result["error"]

    def test_entity_not_found(self, tools_with_graph) -> None:
        tools, ids = tools_with_graph
        result = _run_async(tools["link_memory_to_code"](
            fact_id="my_fact",
            code_entity="nonexistent",
        ))
        assert result["success"] is False
        assert "not found" in result["error"]


class TestCodeEntityHistory:
    """Test code_entity_history bridge tool."""

    def test_empty_history(self, tools_with_graph) -> None:
        tools, ids = tools_with_graph
        result = _run_async(tools["code_entity_history"](
            code_entity="authenticate_user",
        ))
        assert result["success"] is True
        assert result["total_memories"] == 0
        assert result["timeline"] == []

    def test_with_history(self, tools_with_graph, db) -> None:
        tools, ids = tools_with_graph
        for i in range(3):
            db.upsert_link(CodeMemoryLink(
                link_id=f"lnk_{i}",
                code_node_id="n2",
                slm_fact_id=f"fact_{i}",
                link_type=LinkType.MENTIONS,
                confidence=0.8,
                created_at=f"2026-01-0{i+1}T00:00:00",
            ))

        result = _run_async(tools["code_entity_history"](
            code_entity="authenticate_user",
        ))
        assert result["success"] is True
        assert result["total_memories"] == 3
        assert len(result["timeline"]) == 3


# ---------------------------------------------------------------------------
# B1 (3.7.9): update_code_graph repo_path containment + git hardening (RCE)
# ---------------------------------------------------------------------------


def _no_git_run(*_args, **_kwargs):
    raise AssertionError("git subprocess must not run for a rejected repo_path")


class TestUpdateRepoPathSecurity:
    """update_code_graph must contain untrusted repo_path under $HOME and never
    invoke git in an untrusted cwd. Regression for the RCE where a hostile repo's
    .git config/hooks (core.fsmonitor / core.hooksPath) execute under ``git``.
    """

    def test_traversal_repo_path_rejected(self, tools_with_graph, monkeypatch):
        import subprocess
        tools, _ = tools_with_graph
        monkeypatch.setattr(subprocess, "run", _no_git_run)
        res = _run_async(tools["update_code_graph"](repo_path="../../../../etc"))
        assert res["success"] is False
        assert "Invalid repo_path" in res["error"]

    def test_absolute_path_outside_home_rejected(
        self, tools_with_graph, tmp_path, monkeypatch
    ):
        import subprocess
        tools, _ = tools_with_graph
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
        monkeypatch.setattr(subprocess, "run", _no_git_run)
        res = _run_async(tools["update_code_graph"](repo_path="/etc"))
        assert res["success"] is False
        assert "Invalid repo_path" in res["error"]

    def test_home_path_without_dot_git_rejected(
        self, tools_with_graph, tmp_path, monkeypatch
    ):
        import subprocess
        tools, _ = tools_with_graph
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
        monkeypatch.setattr(subprocess, "run", _no_git_run)
        plain_dir = tmp_path / "not_a_repo"
        plain_dir.mkdir()
        res = _run_async(tools["update_code_graph"](repo_path=str(plain_dir)))
        assert res["success"] is False
        assert "Not a git repository" in res["error"]

    def test_valid_repo_invokes_hardened_git(
        self, tools_with_graph, tmp_path, monkeypatch
    ):
        import subprocess
        tools, _ = tools_with_graph
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
        repo = tmp_path / "repo"
        (repo / ".git").mkdir(parents=True)

        captured: dict = {}

        class _Result:
            stdout = ""

        def _fake_run(cmd, **kwargs):
            captured["cmd"] = cmd
            captured["cwd"] = kwargs.get("cwd")
            captured["env"] = kwargs.get("env") or {}
            return _Result()

        monkeypatch.setattr(subprocess, "run", _fake_run)
        res = _run_async(tools["update_code_graph"](repo_path=str(repo)))

        assert captured["cmd"][0] == "git"
        assert "core.hooksPath=/dev/null" in captured["cmd"]
        assert "core.fsmonitor=" in captured["cmd"]
        assert captured["env"].get("GIT_CONFIG_NOSYSTEM") == "1"
        assert captured["cwd"] == str(repo.resolve())
        assert res["success"] is True
