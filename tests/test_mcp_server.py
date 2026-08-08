# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Tests for V3 MCP Server — Task 14 of V3 build.

Verifies that all MCP modules can be imported, tools can be registered,
and the server singleton is properly initialised.

Part of Qualixar | Author: Varun Pratap Bhardwaj
"""

import pytest

pytestmark = pytest.mark.slow


# -- Import tests -------------------------------------------------------------

def test_server_import():
    """FastMCP server instance is importable and non-None."""
    from superlocalmemory.mcp.server import server
    assert server is not None


def test_core_tools_importable():
    """register_core_tools is importable and callable."""
    from superlocalmemory.mcp.tools_core import register_core_tools
    assert callable(register_core_tools)


def test_v28_tools_importable():
    """register_v28_tools is importable and callable."""
    from superlocalmemory.mcp.tools_v28 import register_v28_tools
    assert callable(register_v28_tools)


def test_v3_tools_importable():
    """register_v3_tools is importable and callable."""
    from superlocalmemory.mcp.tools_v3 import register_v3_tools
    assert callable(register_v3_tools)


def test_resources_importable():
    """register_resources is importable and callable."""
    from superlocalmemory.mcp.resources import register_resources
    assert callable(register_resources)


def test_backward_compat_tools():
    """tools.py re-exports from sub-modules correctly."""
    from superlocalmemory.mcp.tools import (
        register_core_tools,
        register_v28_tools,
        register_v3_tools,
    )
    assert callable(register_core_tools)
    assert callable(register_v28_tools)
    assert callable(register_v3_tools)


# -- Singleton tests -----------------------------------------------------------

def test_get_engine_function_exists():
    """get_engine is importable and callable."""
    from superlocalmemory.mcp.server import get_engine
    assert callable(get_engine)


def test_reset_engine_clears_singleton():
    """reset_engine sets the internal _engine to None."""
    from superlocalmemory.mcp import server as srv_mod
    srv_mod._engine = "sentinel"
    srv_mod.reset_engine()
    assert srv_mod._engine is None


# -- Registration smoke tests -------------------------------------------------

def test_server_has_name():
    """FastMCP server has the expected name."""
    from superlocalmemory.mcp.server import server
    assert server.name == "SuperLocalMemory V3"


def test_tools_core_helper_format():
    """_format_results returns correct structure for empty input."""
    from superlocalmemory.mcp.tools_core import _format_results
    assert _format_results([]) == []


def test_v3_mode_description():
    """_mode_description returns capability-only strings for known modes."""
    from superlocalmemory.mcp.tools_v3 import _mode_description
    assert "Local Guardian" in _mode_description("a")
    assert "Smart Local" in _mode_description("b")
    assert "Full Power" in _mode_description("c")
    assert "Unknown" in _mode_description("z")
    for mode in ("a", "b", "c"):
        text = _mode_description(mode).lower()
        assert "ai act" not in text
        assert "compliance" not in text
        assert "compliant" not in text
