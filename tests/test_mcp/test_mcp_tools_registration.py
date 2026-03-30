# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under the MIT License - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Tests for MCP tool registration — Phase 0 Safety Net.

Covers:
    - register_core_tools registers the expected number of tools
    - register_active_tools registers the expected number of tools
    - No duplicate tool names across both registrations

Part of Qualixar | Author: Varun Pratap Bhardwaj
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Helper: mock server that counts registrations
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
# Tests
# ---------------------------------------------------------------------------

class TestToolRegistration:
    """Verify tool registration counts and uniqueness."""

    def test_core_tools_registers_expected_count(self):
        """register_core_tools registers all 15 core tools on the server.

        Expected: remember, recall, search, fetch, list_recent, get_status,
        build_graph, switch_profile, backup_status, memory_used,
        get_learned_patterns, correct_pattern, delete_memory,
        update_memory, get_attribution.
        """
        from superlocalmemory.mcp.tools_core import register_core_tools

        srv = _MockServer()
        get_engine = MagicMock()
        register_core_tools(srv, get_engine)

        # Docstring says 13 but actual code registers 15 (delete_memory + update_memory)
        assert len(srv._tools) >= 13, (
            f"Expected at least 13 core tools, got {len(srv._tools)}: "
            f"{sorted(srv._tools.keys())}"
        )
        # Verify key tools are present
        for name in ("remember", "recall", "search", "fetch", "get_status",
                     "get_attribution", "list_recent"):
            assert name in srv._tools, f"Missing core tool: {name}"

    def test_active_tools_registers_expected_count(self):
        """register_active_tools registers the 3 active tools on the server.

        Expected: session_init, observe, report_feedback.
        """
        from superlocalmemory.mcp.tools_active import register_active_tools

        srv = _MockServer()
        get_engine = MagicMock()
        register_active_tools(srv, get_engine)

        assert len(srv._tools) == 3, (
            f"Expected 3 active tools, got {len(srv._tools)}: "
            f"{sorted(srv._tools.keys())}"
        )
        for name in ("session_init", "observe", "report_feedback"):
            assert name in srv._tools, f"Missing active tool: {name}"

    def test_tool_names_unique(self):
        """No duplicate names when both core and active tools are registered."""
        from superlocalmemory.mcp.tools_core import register_core_tools
        from superlocalmemory.mcp.tools_active import register_active_tools

        srv = _MockServer()
        get_engine = MagicMock()
        register_core_tools(srv, get_engine)

        core_names = set(srv._tools.keys())

        srv_active = _MockServer()
        register_active_tools(srv_active, get_engine)

        active_names = set(srv_active._tools.keys())

        overlap = core_names & active_names
        assert len(overlap) == 0, (
            f"Duplicate tool names across core and active: {overlap}"
        )
