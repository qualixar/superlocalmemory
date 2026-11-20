# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V4 | https://qualixar.com | https://varunpratap.com

"""The one list of MCP tool registrations, shared by the MCP server and ``slm tool``.

``mcp/server.py`` calls :func:`register_tools` on its (possibly filtered) target;
``slm tool`` calls it on a fresh, unfiltered :class:`SLMFastMCP`. Both therefore
see the same names, descriptions, input schemas and admission gates: a tool added
here appears in both places, and a tool can never be callable from one and not
described by the other.

Resources are not tools and stay registered by ``server.py`` itself.
"""

from __future__ import annotations

from typing import Any, Callable


def register_tools(
    target: Any,
    get_engine: Callable[[], Any],
    choose_pool: Callable[[], Any],
    prestage_recall: Callable[..., Any],
) -> None:
    """Register every SuperLocalMemory MCP tool on ``target``, in server order."""
    from superlocalmemory.mcp.tools_active import register_active_tools
    from superlocalmemory.mcp.tools_brain import register_brain_tools
    from superlocalmemory.mcp.tools_code_graph import register_code_graph_tools
    from superlocalmemory.mcp.tools_context import register_prestage_tool
    from superlocalmemory.mcp.tools_core import register_core_tools
    from superlocalmemory.mcp.tools_evolution import register_evolution_tools
    from superlocalmemory.mcp.tools_kinds import register_kind_tools
    from superlocalmemory.mcp.tools_learning import register_learning_tools
    from superlocalmemory.mcp.tools_loops import register_loop_tools
    from superlocalmemory.mcp.tools_media import (
        register_document_tools,
        register_media_tools,
    )
    from superlocalmemory.mcp.tools_media_upload import register_upload_link_tool
    from superlocalmemory.mcp.tools_mesh import register_mesh_tools
    from superlocalmemory.mcp.tools_ops import register_ops_tools
    from superlocalmemory.mcp.tools_optimize import register_optimize_tools
    from superlocalmemory.mcp.tools_summaries import register_summary_tools
    from superlocalmemory.mcp.tools_v3 import register_v3_tools
    from superlocalmemory.mcp.tools_v28 import register_v28_tools
    from superlocalmemory.mcp.tools_v33 import register_v33_tools
    from superlocalmemory.mcp.tools_views import register_view_tools

    register_core_tools(target, get_engine)
    register_v28_tools(target, get_engine)
    register_v3_tools(target, get_engine)
    register_active_tools(target, get_engine)
    register_v33_tools(target, get_engine)
    register_code_graph_tools(target, get_engine)
    register_mesh_tools(target, get_engine)  # v3.4.4: mesh P2P tools
    register_learning_tools(target, get_engine)  # v3.4.7: two-way learning
    register_evolution_tools(target, get_engine)  # v3.4.11: skill evolution
    register_optimize_tools(target)  # v3.6.11: Surface B optimize tools
    register_loop_tools(target, get_engine, choose_pool)  # v3.8.0: bounded loops
    register_ops_tools(target, get_engine)  # operational recovery
    register_brain_tools(target, get_engine)  # v4.0.2 portable Brain receipts
    register_summary_tools(target, get_engine)  # v4.0.8 #113 summary reads
    register_view_tools(target, get_engine)  # 4.1.21 #113 saved views
    register_kind_tools(target, get_engine)  # 4.1.19 WP8 memory kinds
    register_media_tools(target)  # image tools for local AI apps (host-only)
    register_document_tools(target)  # document tools for local AI apps (host-only)
    register_upload_link_tool(target)  # one-time upload link (remote-only)
    register_prestage_tool(target, prestage_recall)


__all__ = ["register_tools"]
