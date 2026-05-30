# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""SuperLocalMemory V3 — MCP Server.

Clean MCP server calling V3 MemoryEngine. Supports all MCP-compatible IDEs.

Part of Qualixar | Author: Varun Pratap Bhardwaj
"""

from __future__ import annotations

# CRITICAL: Set BEFORE any torch/transformers import to prevent Metal/MPS
# GPU memory reservation on Apple Silicon.
import os as _os
_os.environ.setdefault('PYTORCH_MPS_HIGH_WATERMARK_RATIO', '0.0')
_os.environ.setdefault('PYTORCH_MPS_MEM_LIMIT', '0')
_os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
_os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
_os.environ.setdefault('TORCH_DEVICE', 'cpu')
# LIGHT engine contract: suppress the top-level dep check in __init__.py
# that unconditionally imports onnxruntime. MCP runs LIGHT-only — ONNX
# must never load in this process.
_os.environ.setdefault('SLM_SKIP_DEP_CHECK', '1')

import logging
import sys

from mcp.server.fastmcp import FastMCP

logger = logging.getLogger(__name__)

server = FastMCP("SuperLocalMemory V3")

# Lazy engine singleton -------------------------------------------------------

import threading as _threading
_engine = None
_engine_lock = _threading.Lock()


def get_engine():
    """Return (or create) the singleton LIGHT MemoryEngine.

    FastMCP may call tools concurrently from multiple threads. The
    double-checked lock keeps construction single-shot even if two
    tool invocations race on a cold process — without it we would
    double-run the schema migrations and build two ``AdaptiveLearner``
    instances over the same DB file.
    """
    global _engine
    if _engine is not None:
        return _engine
    with _engine_lock:
        if _engine is not None:
            return _engine
        from superlocalmemory.core.config import SLMConfig
        from superlocalmemory.core.engine import MemoryEngine
        from superlocalmemory.core.engine_capabilities import Capabilities

        config = SLMConfig.load()
        new_engine = MemoryEngine(config, capabilities=Capabilities.LIGHT)
        new_engine.initialize()
        _engine = new_engine
    return _engine


def reset_engine():
    """Reset engine singleton (for testing or mode switch)."""
    global _engine
    with _engine_lock:
        _engine = None


# Register tools and resources -------------------------------------------------
#
# Essential-only default: 25 base tools + 8 mesh tools = 33 registered
# when mesh is enabled. Set ``SLM_MCP_ALL_TOOLS=1`` to expose the full
# toolset. Rationale: IDEs cap at 50-100 tools total (Cursor,
# Antigravity, Windsurf) and a maximal SLM registration crowds out
# other MCP servers the user may have installed.
# Admin/diagnostics tools remain available via CLI (`slm <command>`).
# Set SLM_MCP_ALL_TOOLS=1 to enable all 38 tools (power users).

import os as _os_reg

_ESSENTIAL_TOOLS: set[str] = {
    # Core memory operations (8)
    "remember", "recall", "search", "fetch",
    "list_recent", "delete_memory", "update_memory", "get_status",
    # Session lifecycle (3)
    "session_init", "observe", "close_session",
    # Feedback / learning signals — reachable Dash-Core path for
    # thumbs-up / pin / drift signals.
    "report_feedback",
    # Memory management (2)
    "forget", "run_maintenance",
    # Infinite memory + learning (4)
    "consolidate_cognitive", "get_soft_prompts",
    "set_mode", "report_outcome",
    # v3.4.7: Two-way learning (4)
    "log_tool_event", "get_assertions",
    "reinforce_assertion", "contradict_assertion",
    # v3.4.11: Skill evolution (3)
    "evolve_skill", "skill_health", "skill_lineage",
}

# v3.4.4: Mesh tools — enabled if mesh_enabled in config or SLM_MCP_MESH_TOOLS=1
_mesh_tools_enabled = _os_reg.environ.get("SLM_MCP_MESH_TOOLS", "").lower() in ("1", "true")
if not _mesh_tools_enabled:
    try:
        from superlocalmemory.core.config import SLMConfig
        _cfg = SLMConfig.load()
        _mesh_tools_enabled = getattr(_cfg, "mesh_enabled", True)  # default True in v3.4.3+
    except Exception:
        _mesh_tools_enabled = True  # Safe default — mesh broker is always in daemon

if _mesh_tools_enabled:
    _ESSENTIAL_TOOLS.update({
        "mesh_summary", "mesh_peers", "mesh_send", "mesh_inbox",
        "mesh_state", "mesh_lock", "mesh_events", "mesh_status",
    })

_ESSENTIAL_TOOLS = frozenset(_ESSENTIAL_TOOLS)

_all_tools = _os_reg.environ.get("SLM_MCP_ALL_TOOLS") == "1"

# v3.4.45: Minimal mode — explicit user allowlist via SLM_MCP_TOOLS env var.
# Format: comma-separated tool names, e.g. "remember,recall,session_init,search"
# Use case: Claude Code consumer plans with tight context budgets where the
# 25-tool essential set is still too many. Power users override to expose
# exactly the tools they invoke. Falls back to _ESSENTIAL_TOOLS when unset.
_user_allowlist_str = _os_reg.environ.get("SLM_MCP_TOOLS", "").strip()


class _FilteredServer:
    """Wraps FastMCP to only register essential tools.

    Non-essential tools are silently skipped (not registered on the MCP
    server). They remain available via CLI. When SLM_MCP_ALL_TOOLS=1,
    all tools are registered (bypass filter). When SLM_MCP_TOOLS is set,
    that user allowlist is used instead of _ESSENTIAL_TOOLS.
    """
    __slots__ = ("_server", "_allowed")

    def __init__(self, real_server: FastMCP, allowed: frozenset[str]) -> None:
        self._server = real_server
        self._allowed = allowed

    def tool(self, *args, **kwargs):
        def decorator(func):
            if func.__name__ in self._allowed:
                return self._server.tool(*args, **kwargs)(func)
            return func  # Skip registration — still importable, just not MCP-visible
        return decorator

    def __getattr__(self, name):
        return getattr(self._server, name)


# Choose registration target (precedence: ALL > user allowlist > essential)
if _all_tools:
    _target = server
elif _user_allowlist_str:
    _user_allowlist = frozenset(t.strip() for t in _user_allowlist_str.split(",") if t.strip())
    _target = _FilteredServer(server, _user_allowlist)
else:
    _target = _FilteredServer(server, _ESSENTIAL_TOOLS)

from superlocalmemory.mcp.tools_core import register_core_tools
from superlocalmemory.mcp.tools_v28 import register_v28_tools
from superlocalmemory.mcp.tools_v3 import register_v3_tools
from superlocalmemory.mcp.tools_active import register_active_tools
from superlocalmemory.mcp.tools_v33 import register_v33_tools
from superlocalmemory.mcp.resources import register_resources
from superlocalmemory.mcp.tools_code_graph import register_code_graph_tools
from superlocalmemory.mcp.tools_mesh import register_mesh_tools
from superlocalmemory.mcp.tools_learning import register_learning_tools
from superlocalmemory.mcp.tools_evolution import register_evolution_tools

register_core_tools(_target, get_engine)
register_v28_tools(_target, get_engine)
register_v3_tools(_target, get_engine)
register_active_tools(_target, get_engine)
register_v33_tools(_target, get_engine)
register_resources(server, get_engine)  # Resources always registered (not tools)
register_code_graph_tools(_target, get_engine)  # CodeGraph: filtered like other tools (SLM_MCP_ALL_TOOLS=1 to show all)
register_mesh_tools(_target, get_engine)  # v3.4.4: Mesh P2P tools — ships with SLM, no separate slm-mesh needed
register_learning_tools(_target, get_engine)  # v3.4.7: Two-way learning tools
register_evolution_tools(_target, get_engine)  # v3.4.11: Skill evolution tools


# V3.3.21: Eager engine warmup — start initializing BEFORE first tool call.
# The MCP server process starts when the IDE launches. Previously, the engine
# was lazy-loaded on first tool call → 23s cold start for the user.
# Now: engine starts warming in a background thread immediately. By the time
# the first tool call arrives (1-2s later), the engine is already warm.
# This applies to ALL IDEs: Claude Code, Cursor, Antigravity, Gemini CLI, etc.
def _eager_warmup() -> None:
    """Pre-warm LIGHT engine + ensure daemon is running + auto-register mesh.

    LIGHT engine init is cheap (DB only, ~100 ms). The real reason this
    stays in a background thread is the follow-on side effects
    (``ensure_daemon``, ``auto_register_mesh``) which do I/O.
    """
    import logging
    _logger = logging.getLogger(__name__)
    try:
        get_engine()
        _logger.info("MCP engine pre-warmed successfully")
    except Exception as exc:
        _logger.warning("MCP engine pre-warmup failed: %s", exc)

    # Measurement / test harnesses set this to skip daemon-start and
    # mesh-register. The LIGHT engine init above still runs.
    if _os.environ.get("SLM_DISABLE_WARMUP_SIDE_EFFECTS") == "1":
        return

    # V3.4.4: Also ensure daemon is running for dashboard/mesh/health features.
    # This runs in background — doesn't block MCP tool registration.
    try:
        from superlocalmemory.cli.daemon import ensure_daemon
        if ensure_daemon():
            _logger.info("Daemon auto-started by MCP server")
    except Exception as exc:
        _logger.warning("Daemon auto-start failed: %s", exc)

    # V3.4.6: Auto-register this MCP session as a mesh peer immediately.
    # Previously, registration was lazy (only on first mesh tool call).
    # Now every Claude session appears on the mesh from startup.
    try:
        from superlocalmemory.mcp.tools_mesh import auto_register_mesh
        auto_register_mesh()
        _logger.info("Mesh peer auto-registered at startup")
    except Exception as exc:
        _logger.warning("Mesh auto-register failed: %s", exc)

import threading
_warmup_thread = threading.Thread(target=_eager_warmup, daemon=True, name="mcp-warmup")
_warmup_thread.start()


# V3.4.57: Parent watchdog — self-terminate when the IDE/Claude session dies.
# FastMCP relies on stdin EOF to stop, but stdin EOF is NOT guaranteed on
# crash or force-quit. Without this, every abnormal exit leaves an orphaned
# slm mcp process consuming ~100-200 MB indefinitely. 22 orphans caused a
# daemon deadlock on May 30 2026 (241% CPU, session_init timeouts).
def _parent_watchdog() -> None:
    """Exit when parent IDE process (Claude Code, Cursor, etc.) dies.

    Polls os.getppid() every 10 seconds. On macOS/Linux, when a parent
    dies the child is reparented to PID 1 (init) — getppid() returns 1.
    Also validates the original parent is still alive via os.kill(ppid, 0).
    Uses os._exit(0) to bypass any atexit handlers that might hang.
    """
    import os as _os_wd, time as _time
    _wlog = logging.getLogger(__name__ + ".watchdog")
    initial_ppid = _os_wd.getppid()
    if initial_ppid <= 1:
        return  # Already reparented at startup — don't self-terminate
    while True:
        _time.sleep(10)
        try:
            current_ppid = _os_wd.getppid()
            if current_ppid != initial_ppid or current_ppid <= 1:
                _wlog.info("Parent PID changed (%d→%d), self-terminating", initial_ppid, current_ppid)
                _os_wd._exit(0)
            _os_wd.kill(initial_ppid, 0)  # Raises ProcessLookupError if dead
        except ProcessLookupError:
            _wlog.info("Parent PID %d gone, self-terminating", initial_ppid)
            _os_wd._exit(0)
        except Exception:
            pass  # Transient errors — keep watching


_watchdog_thread = threading.Thread(target=_parent_watchdog, daemon=True, name="parent-watchdog")
_watchdog_thread.start()


if __name__ == "__main__":
    server.run(transport="stdio")
