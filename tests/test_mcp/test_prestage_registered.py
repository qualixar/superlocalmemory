# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V4

"""W3: prestage_context must be registered on the MCP server, not dead code."""

from __future__ import annotations

import importlib
import os
import sys


def _load_server():
    os.environ["SLM_MCP_EMBEDDED"] = "1"
    os.environ.setdefault("SLM_DISABLE_WARMUP_SIDE_EFFECTS", "1")
    # Force full tool exposure so registration is not filtered away by a
    # narrow profile env left over from other tests.
    os.environ["SLM_MCP_ALL_TOOLS"] = "1"
    for key in ("SLM_MCP_PROFILE", "SLM_MCP_TOOLS"):
        os.environ.pop(key, None)
    mod_name = "superlocalmemory.mcp.server"
    if mod_name in sys.modules:
        del sys.modules[mod_name]
    # Also drop tools_context so re-registration is fresh if needed.
    for name in list(sys.modules):
        if name.startswith("superlocalmemory.mcp.tools_"):
            del sys.modules[name]
    return importlib.import_module(mod_name)


def test_prestage_context_absent_from_counted_profiles():
    """prestage_context is registered but must NOT grow the counted profiles.

    Brain evidence deliberately grew the named profiles in V4.0.4; this test
    only guards that prestage_context remains raw-server-only. Reachability is
    covered by ``test_prestage_context_registered_on_server``.
    """
    mod = _load_server()
    assert "prestage_context" not in mod._ESSENTIAL_TOOLS
    assert "prestage_context" not in mod._PROFILE_DEFINITIONS["full"]
    assert "prestage_context" not in mod._PROFILE_DEFINITIONS["power"]
    assert len(mod._ESSENTIAL_TOOLS) == 47
    assert len(mod._PROFILE_DEFINITIONS["full"]) == 47
    assert len(mod._PROFILE_DEFINITIONS["power"]) == 59


def test_prestage_context_registered_on_server():
    mod = _load_server()
    server = mod.server
    # mcp 2.0 MCPServer / FastMCP tool registry surfaces.
    tools = None
    for attr in ("_tool_manager", "_tools", "tools"):
        if hasattr(server, attr):
            tools = getattr(server, attr)
            break
    names: set[str] = set()
    if tools is None and hasattr(server, "list_tools"):
        # async list — not used here
        pass
    if hasattr(tools, "_tools"):
        names = set(tools._tools.keys())
    elif isinstance(tools, dict):
        names = set(tools.keys())
    else:
        # Fallback: inspect registered callables on the FastMCP/MCPServer.
        tm = getattr(server, "_tool_manager", None) or getattr(server, "_mcp_server", None)
        if tm is not None:
            raw = getattr(tm, "_tools", None) or getattr(tm, "tools", None) or {}
            if isinstance(raw, dict):
                names = set(raw.keys())
    assert "prestage_context" in names, (
        f"prestage_context not registered; found {sorted(names)[:40]}"
    )
