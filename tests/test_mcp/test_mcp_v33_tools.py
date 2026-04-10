# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under the Elastic License 2.0 - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Tests for V3.3 MCP tools.

Covers:
    - register_v33_tools registers the expected 6 tools
    - forget tool returns decay cycle stats
    - quantize tool returns EAP cycle stats
    - consolidate_cognitive tool returns pipeline results
    - get_soft_prompts tool returns prompt list
    - reap_processes tool returns orphan stats
    - get_retention_stats tool returns zone distribution
    - All tools handle errors gracefully (return success=False)
    - No duplicate names with existing tool registrations

Part of Qualixar | Author: Varun Pratap Bhardwaj
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helper: mock server that captures @server.tool() decorated functions
# ---------------------------------------------------------------------------


class _MockServer:
    """Minimal mock that captures @server.tool() decorated functions."""

    def __init__(self):
        self._tools: dict[str, object] = {}

    def tool(self):
        def decorator(fn):
            self._tools[fn.__name__] = fn
            return fn

        return decorator


# ---------------------------------------------------------------------------
# Helper: mock engine
# ---------------------------------------------------------------------------


def _make_mock_engine(profile_id: str = "test-profile"):
    """Create a mock engine with the attributes the tools need."""
    engine = MagicMock()
    engine.profile_id = profile_id

    # Mock _config sub-attributes
    engine._config.forgetting = MagicMock()
    engine._config.quantization = MagicMock()

    # Mock _db.execute to return empty by default
    engine._db.execute.return_value = []
    engine._db.db_path = MagicMock()

    return engine


def _run(coro):
    """Run an async coroutine in a sync test."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ---------------------------------------------------------------------------
# Registration tests
# ---------------------------------------------------------------------------


class TestV33ToolRegistration:
    """Verify V3.3 tool registration counts and names."""

    def test_registers_expected_count(self):
        """register_v33_tools registers exactly 7 tools (v3.3.12: +run_maintenance)."""
        from superlocalmemory.mcp.tools_v33 import register_v33_tools

        srv = _MockServer()
        get_engine = MagicMock()
        register_v33_tools(srv, get_engine)

        assert len(srv._tools) == 7, (
            f"Expected 7 V3.3 tools, got {len(srv._tools)}: {sorted(srv._tools.keys())}"
        )

    def test_expected_tool_names(self):
        """All 7 expected tool names are present."""
        from superlocalmemory.mcp.tools_v33 import register_v33_tools

        srv = _MockServer()
        register_v33_tools(srv, MagicMock())

        expected = {
            "forget",
            "quantize",
            "consolidate_cognitive",
            "get_soft_prompts",
            "reap_processes",
            "get_retention_stats",
            "run_maintenance",
        }
        assert set(srv._tools.keys()) == expected

    def test_no_overlap_with_core_tools(self):
        """V3.3 tool names don't collide with core tools."""
        from superlocalmemory.mcp.tools_v33 import register_v33_tools
        from superlocalmemory.mcp.tools_core import register_core_tools

        srv_core = _MockServer()
        register_core_tools(srv_core, MagicMock())

        srv_v33 = _MockServer()
        register_v33_tools(srv_v33, MagicMock())

        overlap = set(srv_core._tools) & set(srv_v33._tools)
        assert len(overlap) == 0, f"Name collision: {overlap}"

    def test_no_overlap_with_active_tools(self):
        """V3.3 tool names don't collide with active tools."""
        from superlocalmemory.mcp.tools_v33 import register_v33_tools
        from superlocalmemory.mcp.tools_active import register_active_tools

        srv_active = _MockServer()
        register_active_tools(srv_active, MagicMock())

        srv_v33 = _MockServer()
        register_v33_tools(srv_v33, MagicMock())

        overlap = set(srv_active._tools) & set(srv_v33._tools)
        assert len(overlap) == 0, f"Name collision: {overlap}"

    def test_no_overlap_with_v28_tools(self):
        """V3.3 tool names don't collide with v28 tools."""
        from superlocalmemory.mcp.tools_v33 import register_v33_tools
        from superlocalmemory.mcp.tools_v28 import register_v28_tools

        srv_v28 = _MockServer()
        register_v28_tools(srv_v28, MagicMock())

        srv_v33 = _MockServer()
        register_v33_tools(srv_v33, MagicMock())

        overlap = set(srv_v28._tools) & set(srv_v33._tools)
        assert len(overlap) == 0, f"Name collision: {overlap}"


# ---------------------------------------------------------------------------
# forget tool tests
# ---------------------------------------------------------------------------


class TestForgetTool:
    """Tests for the forget (Ebbinghaus decay) MCP tool."""

    def _get_tool(self):
        from superlocalmemory.mcp.tools_v33 import register_v33_tools

        srv = _MockServer()
        engine = _make_mock_engine()
        get_engine = MagicMock(return_value=engine)
        register_v33_tools(srv, get_engine)
        return srv._tools["forget"], engine

    def test_forget_returns_success_with_stats(self):
        """forget dry_run=False returns zone counts from fact_retention."""
        tool, engine = self._get_tool()

        mock_result = {
            "total": 100,
            "active": 50,
            "warm": 20,
            "cold": 15,
            "archive": 10,
            "forgotten": 5,
            "transitions": 8,
        }

        with (
            patch(
                "superlocalmemory.learning.forgetting_scheduler.ForgettingScheduler"
            ) as MockSched,
            patch("superlocalmemory.math.ebbinghaus.EbbinghausCurve"),
        ):
            MockSched.return_value.run_decay_cycle.return_value = mock_result
            result = _run(tool(dry_run=False))

        assert result["success"] is True
        assert result["total"] == 100
        assert result["transitions"] == 8
        assert result["dry_run"] is False

    def test_forget_dry_run_returns_zone_distribution(self):
        """forget dry_run=True returns zone counts from fact_retention."""
        tool, engine = self._get_tool()

        mock_rows = [
            {"lifecycle_zone": "active", "cnt": 50},
            {"lifecycle_zone": "warm", "cnt": 20},
            {"lifecycle_zone": "cold", "cnt": 10},
            {"lifecycle_zone": "archive", "cnt": 5},
            {"lifecycle_zone": "forgotten", "cnt": 3},
        ]
        engine._db.execute.return_value = mock_rows

        result = _run(tool())

        assert result["success"] is True
        assert result["total"] == 88
        assert result["dry_run_zones"]["active"] == 50
        assert result["dry_run_zones"]["warm"] == 20
        assert result["dry_run_zones"]["cold"] == 10

    def test_forget_with_profile(self):
        """forget tool uses provided profile_id."""
        tool, engine = self._get_tool()

        mock_result = {
            "total": 0,
            "active": 0,
            "warm": 0,
            "cold": 0,
            "archive": 0,
            "forgotten": 0,
            "transitions": 0,
        }

        with (
            patch(
                "superlocalmemory.learning.forgetting_scheduler.ForgettingScheduler"
            ) as MockSched,
            patch("superlocalmemory.math.ebbinghaus.EbbinghausCurve"),
        ):
            MockSched.return_value.run_decay_cycle.return_value = mock_result
            result = _run(tool(profile_id="custom-profile", dry_run=False))

        assert result["success"] is True
        MockSched.return_value.run_decay_cycle.assert_called_once_with(
            "custom-profile",
            force=True,
        )

    def test_forget_handles_error(self):
        """forget tool returns success=False on exception."""
        tool, engine = self._get_tool()

        with patch(
            "superlocalmemory.math.ebbinghaus.EbbinghausCurve",
            side_effect=RuntimeError("test error"),
        ):
            result = _run(tool())

        assert result["success"] is False
        assert "test error" in result["error"]


# ---------------------------------------------------------------------------
# quantize tool tests
# ---------------------------------------------------------------------------


class TestQuantizeTool:
    """Tests for the quantize (EAP) MCP tool."""

    def _get_tool(self):
        from superlocalmemory.mcp.tools_v33 import register_v33_tools

        srv = _MockServer()
        engine = _make_mock_engine()
        get_engine = MagicMock(return_value=engine)
        register_v33_tools(srv, get_engine)
        return srv._tools["quantize"], engine

    def test_quantize_returns_success_with_stats(self):
        """quantize tool (dry_run=False) returns EAP cycle stats."""
        tool, engine = self._get_tool()

        mock_result = {
            "total": 50,
            "downgrades": 10,
            "upgrades": 3,
            "skipped": 35,
            "deleted": 2,
            "errors": 0,
        }

        with (
            patch("superlocalmemory.dynamics.eap_scheduler.EAPScheduler") as MockEAP,
            patch("superlocalmemory.math.ebbinghaus.EbbinghausCurve"),
            patch("superlocalmemory.storage.quantized_store.QuantizedEmbeddingStore"),
            patch("superlocalmemory.math.polar_quant.PolarQuantEncoder"),
            patch("superlocalmemory.math.qjl.QJLEncoder"),
        ):
            MockEAP.return_value.run_eap_cycle.return_value = mock_result
            result = _run(tool(dry_run=False))

        assert result["success"] is True
        assert result["downgrades"] == 10
        assert result["upgrades"] == 3
        assert result["dry_run"] is False

    def test_quantize_handles_error(self):
        """quantize tool returns success=False on exception."""
        tool, engine = self._get_tool()

        with patch(
            "superlocalmemory.math.ebbinghaus.EbbinghausCurve",
            side_effect=RuntimeError("eap crash"),
        ):
            result = _run(tool())

        assert result["success"] is False
        assert "eap crash" in result["error"]


# ---------------------------------------------------------------------------
# consolidate_cognitive tool tests
# ---------------------------------------------------------------------------


@dataclass
class _MockCCQResult:
    """Minimal CCQ pipeline result for testing."""

    clusters_processed: int = 3
    blocks_created: int = 2
    facts_archived: int = 15
    compression_ratio: float = 0.45
    bytes_before: int = 1000
    bytes_after: int = 450
    audit_ids: list = field(default_factory=list)
    errors: list = field(default_factory=list)


class TestConsolidateCognitiveTool:
    """Tests for the consolidate_cognitive MCP tool."""

    def _get_tool(self):
        from superlocalmemory.mcp.tools_v33 import register_v33_tools

        srv = _MockServer()
        engine = _make_mock_engine()
        get_engine = MagicMock(return_value=engine)
        register_v33_tools(srv, get_engine)
        return srv._tools["consolidate_cognitive"], engine

    def test_consolidate_returns_success(self):
        """consolidate_cognitive returns pipeline results."""
        tool, engine = self._get_tool()

        with patch(
            "superlocalmemory.encoding.cognitive_consolidator.CognitiveConsolidator"
        ) as MockCCQ:
            MockCCQ.return_value.run_pipeline.return_value = _MockCCQResult()
            result = _run(tool())

        assert result["success"] is True
        assert result["clusters_processed"] == 3
        assert result["blocks_created"] == 2
        assert result["facts_archived"] == 15
        assert result["compression_ratio"] == 0.45

    def test_consolidate_handles_error(self):
        """consolidate_cognitive returns success=False on error."""
        tool, engine = self._get_tool()

        with patch(
            "superlocalmemory.encoding.cognitive_consolidator.CognitiveConsolidator",
            side_effect=RuntimeError("ccq fail"),
        ):
            result = _run(tool())

        assert result["success"] is False
        assert "ccq fail" in result["error"]


# ---------------------------------------------------------------------------
# get_soft_prompts tool tests
# ---------------------------------------------------------------------------


class TestGetSoftPromptsTool:
    """Tests for the get_soft_prompts MCP tool."""

    def _get_tool(self):
        from superlocalmemory.mcp.tools_v33 import register_v33_tools

        srv = _MockServer()
        engine = _make_mock_engine()
        get_engine = MagicMock(return_value=engine)
        register_v33_tools(srv, get_engine)
        return srv._tools["get_soft_prompts"], engine

    def test_returns_empty_when_no_prompts(self):
        """get_soft_prompts returns empty list when no prompts exist."""
        tool, engine = self._get_tool()
        engine._db.execute.return_value = []

        result = _run(tool())

        assert result["success"] is True
        assert result["count"] == 0
        assert result["prompts"] == []

    def test_returns_prompts_from_db(self):
        """get_soft_prompts returns formatted prompts from DB."""
        tool, engine = self._get_tool()

        mock_row = {
            "prompt_id": "sp-001",
            "category": "tech_preference",
            "content": "User prefers Python 3.13",
            "confidence": 0.85,
            "effectiveness": 0.72,
            "token_count": 12,
            "version": 1,
            "created_at": "2026-03-30 10:00:00",
        }
        engine._db.execute.return_value = [mock_row]

        result = _run(tool())

        assert result["success"] is True
        assert result["count"] == 1
        assert result["prompts"][0]["prompt_id"] == "sp-001"
        assert result["prompts"][0]["confidence"] == 0.85

    def test_handles_error(self):
        """get_soft_prompts returns success=False on error."""
        tool, engine = self._get_tool()
        engine._db.execute.side_effect = RuntimeError("db down")

        result = _run(tool())

        assert result["success"] is False
        assert "db down" in result["error"]


# ---------------------------------------------------------------------------
# reap_processes tool tests
# ---------------------------------------------------------------------------


class TestReapProcessesTool:
    """Tests for the reap_processes MCP tool."""

    def _get_tool(self):
        from superlocalmemory.mcp.tools_v33 import register_v33_tools

        srv = _MockServer()
        engine = _make_mock_engine()
        get_engine = MagicMock(return_value=engine)
        register_v33_tools(srv, get_engine)
        return srv._tools["reap_processes"], engine

    def test_reap_dry_run(self):
        """reap_processes in dry_run mode reports but doesn't kill."""
        tool, _ = self._get_tool()

        mock_result = {
            "total_found": 5,
            "orphans_found": 2,
            "killed": 0,
            "skipped": 2,
            "errors": [],
            "processes": [],
        }

        with patch(
            "superlocalmemory.infra.process_reaper.cleanup_all_orphans",
            return_value=mock_result,
        ):
            result = _run(tool())

        assert result["success"] is True
        assert result["dry_run"] is True
        assert result["total_found"] == 5
        assert result["orphans_found"] == 2
        assert result["killed"] == 0

    def test_reap_execute(self):
        """reap_processes with dry_run=False kills orphans."""
        tool, _ = self._get_tool()

        mock_result = {
            "total_found": 5,
            "orphans_found": 2,
            "killed": 2,
            "skipped": 0,
            "errors": [],
            "processes": [],
        }

        with patch(
            "superlocalmemory.infra.process_reaper.cleanup_all_orphans",
            return_value=mock_result,
        ):
            result = _run(tool(dry_run=False))

        assert result["success"] is True
        assert result["dry_run"] is False
        assert result["killed"] == 2

    def test_reap_handles_error(self):
        """reap_processes returns success=False on error."""
        tool, _ = self._get_tool()

        with patch(
            "superlocalmemory.infra.process_reaper.cleanup_all_orphans",
            side_effect=RuntimeError("ps failed"),
        ):
            result = _run(tool())

        assert result["success"] is False
        assert "ps failed" in result["error"]


# ---------------------------------------------------------------------------
# get_retention_stats tool tests
# ---------------------------------------------------------------------------


class TestGetRetentionStatsTool:
    """Tests for the get_retention_stats MCP tool."""

    def _get_tool(self):
        from superlocalmemory.mcp.tools_v33 import register_v33_tools

        srv = _MockServer()
        engine = _make_mock_engine()
        get_engine = MagicMock(return_value=engine)
        register_v33_tools(srv, get_engine)
        return srv._tools["get_retention_stats"], engine

    def test_returns_zone_distribution(self):
        """get_retention_stats returns zone counts and averages."""
        tool, engine = self._get_tool()

        mock_rows = [
            {"lifecycle_zone": "active", "cnt": 50, "avg_score": 0.92},
            {"lifecycle_zone": "warm", "cnt": 20, "avg_score": 0.65},
            {"lifecycle_zone": "cold", "cnt": 10, "avg_score": 0.35},
            {"lifecycle_zone": "archive", "cnt": 5, "avg_score": 0.12},
            {"lifecycle_zone": "forgotten", "cnt": 3, "avg_score": 0.02},
        ]
        engine._db.execute.return_value = mock_rows

        result = _run(tool())

        assert result["success"] is True
        assert result["total"] == 88
        assert result["active"] == 50
        assert result["warm"] == 20
        assert result["cold"] == 10
        assert result["archive"] == 5
        assert result["forgotten"] == 3

    def test_returns_empty_when_no_data(self):
        """get_retention_stats returns zeros when no retention data."""
        tool, engine = self._get_tool()
        engine._db.execute.return_value = []

        result = _run(tool())

        assert result["success"] is True
        assert result["total"] == 0
        assert result["active"] == 0

    def test_handles_error(self):
        """get_retention_stats returns success=False on error."""
        tool, engine = self._get_tool()
        engine._db.execute.side_effect = RuntimeError("db error")

        result = _run(tool())

        assert result["success"] is False
        assert "db error" in result["error"]

    def test_uses_custom_profile(self):
        """get_retention_stats uses provided profile_id."""
        tool, engine = self._get_tool()
        engine._db.execute.return_value = []

        result = _run(tool(profile_id="custom"))

        assert result["success"] is True
        assert result["profile"] == "custom"
        # Verify query used custom profile
        call_args = engine._db.execute.call_args
        assert call_args[0][1] == ("custom",)


# ---------------------------------------------------------------------------
# Maintenance Tools tests
# ---------------------------------------------------------------------------


class TestRunMaintenanceTool:
    """Tests for the run_maintenance MCP tool."""

    def _get_tool(self):
        from superlocalmemory.mcp.tools_v33 import register_v33_tools

        srv = _MockServer()
        engine = _make_mock_engine()
        get_engine = MagicMock(return_value=engine)
        register_v33_tools(srv, get_engine)
        return srv._tools["run_maintenance"], engine

    def test_behavioral_patterns_mined(self):
        """run_maintenance behavioral section uses correct ConsolidationWorker args."""
        tool, engine = self._get_tool()

        with (
            patch("superlocalmemory.learning.consolidation_worker.ConsolidationWorker", autospec=True) as MockCW,
            patch(
                "superlocalmemory.core.maintenance.run_maintenance",
                return_value={"updated": 0},
            ),
            patch(
                "superlocalmemory.learning.forgetting_scheduler.ForgettingScheduler"
            ) as MockSched,
            patch("superlocalmemory.math.ebbinghaus.EbbinghausCurve"),
        ):
            MockSched.return_value.run_decay_cycle.return_value = {
                "total": 0,
                "active": 0,
                "warm": 0,
                "cold": 0,
                "archive": 0,
                "forgotten": 0,
                "transitions": 0,
            }
            MockCW.return_value._generate_patterns.return_value = 7
            result = _run(tool())

        assert result["success"] is True
        assert result["behavioral"]["patterns_mined"] == 7
        init_args = MockCW.call_args[0]
        expected_memory = engine._db.db_path
        expected_learning = engine._db.db_path.parent / "learning.db"
        MockCW.assert_called_once_with(expected_memory, expected_learning)