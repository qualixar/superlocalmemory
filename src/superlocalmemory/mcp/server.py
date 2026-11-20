# ruff: noqa: E402
# Bootstrap environment and transport seams must precede tool imports.
# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V4 | https://qualixar.com | https://varunpratap.com

"""SuperLocalMemory V4 — MCP Server.

Clean MCP server calling V3 MemoryEngine. Supports all MCP-compatible IDEs.

Part of Qualixar | Author: Varun Pratap Bhardwaj
"""

from __future__ import annotations

# CRITICAL: Set BEFORE any torch/transformers import to prevent Metal/MPS
# GPU memory reservation on Apple Silicon.
import os as _os

_os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")
_os.environ.setdefault("PYTORCH_MPS_MEM_LIMIT", "0")
_os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
_os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
_os.environ.setdefault("TORCH_DEVICE", "cpu")
# LIGHT engine contract: suppress the top-level dep check in __init__.py
# that unconditionally imports onnxruntime. MCP runs LIGHT-only — ONNX
# must never load in this process.
_os.environ.setdefault("SLM_SKIP_DEP_CHECK", "1")

import logging
import sys

from superlocalmemory import __version__ as _slm_version
from superlocalmemory.mcp.http_transport import SLMFastMCP

logger = logging.getLogger(__name__)

# mcp 2.0.0: MCPServer takes version= directly (no private _mcp_server poke).
server = SLMFastMCP("SuperLocalMemory V4", version=_slm_version)

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
    resident = _daemon_engine()
    if resident is not None:
        return _request_engine(resident)
    if _engine is not None:
        return _request_engine(_engine)
    with _engine_lock:
        if _engine is not None:
            return _request_engine(_engine)
        from superlocalmemory.core.config import SLMConfig
        from superlocalmemory.core.engine import MemoryEngine
        from superlocalmemory.core.engine_capabilities import Capabilities

        config = SLMConfig.load()
        new_engine = MemoryEngine(config, capabilities=Capabilities.LIGHT)
        new_engine.initialize()
        _engine = new_engine
    return _request_engine(_engine)


def _daemon_engine():
    """The daemon's resident engine when this server runs inside the daemon, else None.

    Building a second engine there ran deferred migrations and every verify() on the
    daemon's event loop, so /health went unanswered for minutes on a large store. Read
    at call time, so the engine a mode switch puts in place is the one used.
    """
    if _os.environ.get("SLM_MCP_EMBEDDED") != "1":
        return None
    import sys

    daemon = sys.modules.get("superlocalmemory.server.unified_daemon")
    return getattr(daemon, "_engine", None) if daemon is not None else None


def _request_engine(engine):
    """A lightweight engine view whose default workspace is pinned to this request.

    Storage and services remain shared. Only the routing fields are copied;
    a caller cannot mutate the singleton's profile or config through this view.
    """
    from copy import copy

    from superlocalmemory.server.profile_runtime import current_request_pin

    pin = current_request_pin()
    if pin is None:
        return engine
    view = copy(engine)
    view._profile_id = pin.profile_id
    view._config = copy(engine._config)
    view._config.active_profile = pin.profile_id
    return view


def reset_engine():
    """Reset engine singleton (for testing or mode switch)."""
    global _engine
    with _engine_lock:
        _engine = None


# Register tools and resources -------------------------------------------------
#
# Essential-only default: 48 base tools + 9 mesh tools = 57 registered
# when mesh is enabled. Set ``SLM_MCP_ALL_TOOLS=1`` to expose the full
# toolset. Rationale: IDEs cap at 50-100 tools total (Cursor,
# Antigravity, Windsurf) and a maximal SLM registration crowds out
# other MCP servers the user may have installed.
# Admin/diagnostics tools remain available via CLI (`slm <command>`).
# Set SLM_MCP_ALL_TOOLS=1 to enable all 107 tools (power users).

import os as _os_reg

from superlocalmemory.mcp import tool_surface as _tool_surface  # noqa: E402

# The essential set, mesh and image rules live in mcp/tool_surface (shared with `slm tool`).
_mesh_tools_enabled = _tool_surface.mesh_tools_enabled()
_media_tools_enabled = _tool_surface.media_tools_enabled()
_ESSENTIAL_TOOLS = _tool_surface.essential_tools()

_all_tools = _os_reg.environ.get("SLM_MCP_ALL_TOOLS") == "1"

# v3.4.45: Minimal mode — explicit user allowlist via SLM_MCP_TOOLS env var.
# Format: comma-separated tool names, e.g. "remember,recall,session_init,search"
# Use case: Claude Code consumer plans with tight context budgets where the
# 25-tool essential set is still too many. Power users override to expose
# exactly the tools they invoke. Falls back to _ESSENTIAL_TOOLS when unset.
_user_allowlist_str = _os_reg.environ.get("SLM_MCP_TOOLS", "").strip()

# ---------------------------------------------------------------------------
# Named profile definitions (introduced in v3.6.14)
# Extracted to mcp/profiles.py (v3.8.0) — pure data, no side effects.
# All names re-exported here for backward compatibility with existing tests
# and any code that imports them from this module.
# ---------------------------------------------------------------------------

# SLM_MCP_PROFILE from the host's config, else the set saved by `slm tools set`.
# The profile names are re-exported: tests and older callers read them from this module.
from superlocalmemory.mcp.profile_names import effective_profile as _effective_profile  # noqa: E402
from superlocalmemory.mcp.profiles import (  # noqa: E402, F401 (after env setups above)
    _MEDIA_TOOLS,
    _PROFILE_ALIASES,
    _PROFILE_DEFINITIONS,
)

_profile = _effective_profile()


_resolve_profile_allowed = _tool_surface.resolve_profile_allowed


class _FilteredServer:
    """Wraps FastMCP to only register essential tools.

    Non-essential tools are silently skipped (not registered on the MCP
    server). They remain available via CLI. When SLM_MCP_ALL_TOOLS=1,
    all tools are registered (bypass filter). When SLM_MCP_TOOLS is set,
    that user allowlist is used instead of _ESSENTIAL_TOOLS.
    """

    __slots__ = ("_server", "_allowed")

    def __init__(self, real_server: SLMFastMCP, allowed: frozenset[str]) -> None:
        self._server = real_server
        self._allowed = allowed
        real_server.configure_tool_surface(allowed)

    def tool(self, *args, **kwargs):
        def decorator(func):
            if func.__name__ in self._allowed:
                return self._server.tool(*args, **kwargs)(func)
            return func  # Skip registration — still importable, just not MCP-visible

        return decorator

    def __getattr__(self, name):
        return getattr(self._server, name)


# Choose registration target (precedence: ALL > user allowlist > profile > essential)
_allowed_tools = _tool_surface.surface_for(
    all_tools=_all_tools,
    allowlist=_user_allowlist_str,
    profile=_profile,
    essential=_ESSENTIAL_TOOLS,
    resolve=_resolve_profile_allowed,
)
_target = server if _allowed_tools is None else _FilteredServer(server, _allowed_tools)

# One registration list for the MCP server and `slm tool` (mcp/tool_registry), so the
# two can never disagree about which tools exist or what arguments they take.
from superlocalmemory.mcp import prestage_bridge as _prestage_bridge
from superlocalmemory.mcp._daemon_proxy import choose_pool as _choose_loop_pool
from superlocalmemory.mcp.resources import register_resources
from superlocalmemory.mcp.tool_registry import register_tools
from superlocalmemory.mcp.tools_media import register_media_resources

_memories_from = _prestage_bridge.memories_from


def _prestage_recall(query: str, limit: int, profile_id: str, as_of: str | None = None):
    """Bridge engine.recall -> prestage_context recall_fn shape (mcp/prestage_bridge)."""
    return _prestage_bridge.prestage_recall(get_engine, query, limit, profile_id, as_of)


register_tools(_target, get_engine, _choose_loop_pool, _prestage_recall)
register_resources(server, get_engine)  # Resources always registered (not tools)
register_media_resources(server)


# Keep stdio MCP processes thin until a tool truly needs a local LIGHT engine.
# Every open IDE/task owns a stdio process; eagerly opening memory.db in all of
# them multiplied RAM and SQLite writers on machines with many long-lived
# sessions. The shared daemon owns model warmup and common remember/recall work.
def _eager_warmup() -> None:
    """Ensure the shared daemon is running without opening a per-client engine.

    Each local stdio host session joins the broker and maintains its heartbeat.
    Embedded daemon instances skip this warmup entirely.
    """
    _logger = logging.getLogger(__name__)

    # Measurement / test harnesses set this to skip daemon-start.
    if _os.environ.get("SLM_DISABLE_WARMUP_SIDE_EFFECTS") == "1":
        return

    # V3.4.4: Also ensure daemon is running for dashboard/mesh/health features.
    # This runs in background — doesn't block MCP tool registration.
    try:
        from superlocalmemory.cli.daemon import ensure_daemon

        if ensure_daemon():
            _logger.info("Daemon auto-started by MCP server")
        # Even when the daemon is not up yet, the session keeps retrying to join.
        from superlocalmemory.mcp.tools_mesh import auto_register_mesh

        auto_register_mesh()
    except Exception as exc:
        _logger.warning("Daemon auto-start failed: %s", exc)


import threading

# v3.6.7: Suppress standalone-process behaviours when the MCP server is
# imported inside the daemon (SLM_MCP_EMBEDDED=1). Three threads are safe
# to run in a dedicated `slm mcp` subprocess but harmful inside the daemon:
#   mcp-warmup      — ensures the shared daemon only; never creates an engine.
#   parent-watchdog — calls os._exit(0) if its parent IDE quits, which would
#                     kill the daemon along with it.
#   stdin-eof-monitor — monitors stdin pipe; meaningless inside the daemon.
_embedded_in_daemon = _os.environ.get("SLM_MCP_EMBEDDED") == "1"

if not _embedded_in_daemon:
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
    import os as _os_wd
    import time as _time

    _wlog = logging.getLogger(__name__ + ".watchdog")
    initial_ppid = _os_wd.getppid()
    if initial_ppid <= 1:
        return  # Already reparented at startup — don't self-terminate
    while True:
        _time.sleep(10)
        try:
            current_ppid = _os_wd.getppid()
            if current_ppid != initial_ppid or current_ppid <= 1:
                _wlog.info(
                    "Parent PID changed (%d→%d), self-terminating", initial_ppid, current_ppid
                )
                _os_wd._exit(0)
            from superlocalmemory.core.platform_utils import is_pid_alive

            if not is_pid_alive(initial_ppid):
                _wlog.info("Parent PID %d gone, self-terminating", initial_ppid)
                _os_wd._exit(0)
        except Exception:
            pass  # Transient errors — keep watching


if not _embedded_in_daemon:
    _watchdog_thread = threading.Thread(
        target=_parent_watchdog, daemon=True, name="parent-watchdog"
    )
    _watchdog_thread.start()


# V3.5.9: Stdin EOF monitor — complements the parent watchdog for the case where
# the IDE starts a new MCP session WITHOUT quitting the app. The old stdio pipe
# is abandoned (write-end closed) but the parent process stays alive, so the
# watchdog never fires. Without this, each IDE reconnect adds a zombie process
# (22 seen in production causing 12 GB swap on the M5 Pro).
#
# Uses kqueue(2) KQ_EV_EOF on macOS — fires when the write-end of the stdin pipe
# closes WITHOUT consuming any bytes, so it cannot race with FastMCP's asyncio
# stdin reader. On Linux (no kqueue), the watchdog alone provides coverage.
def _stdin_eof_monitor() -> None:
    """Exit this stdio MCP process when the host closes its input stream."""
    import os as _os_eof
    import select as _sel
    import time as _time

    from superlocalmemory.mcp._stdin_guard import eof_action

    _mlog = logging.getLogger(__name__ + ".stdin_monitor")
    if not hasattr(_sel, "kqueue"):
        return  # Linux / non-macOS: watchdog covers process death
    # Bounded grace for the FastMCP reader to drain a buffered final request
    # before we tear down. ~2 s ceiling (40 × 50 ms): a genuine EOF means the
    # session is ending regardless, so we never wait indefinitely.
    _DRAIN_POLL_S = 0.05
    _DRAIN_MAX_POLLS = 40
    try:
        fd = sys.stdin.fileno()
        kq = _sel.kqueue()
        ke = _sel.kevent(fd, filter=_sel.KQ_FILTER_READ, flags=_sel.KQ_EV_ADD | _sel.KQ_EV_EOF)
        kq.control([ke], 0)  # register without waiting
        while True:
            evs = kq.control(None, 4, 30.0)  # 30 s poll — low cost
            for ev in evs:
                action = eof_action(ev.flags, ev.data, _sel.KQ_EV_EOF)
                if action == "ignore":
                    continue
                if action == "drain":
                    # Write-end closed but unread bytes remain. Let the
                    # FastMCP reader consume the final request(s); poll until
                    # drained or the grace ceiling elapses.
                    for _ in range(_DRAIN_MAX_POLLS):
                        _time.sleep(_DRAIN_POLL_S)
                        recheck = kq.control(None, 4, 0)  # non-blocking
                        if not recheck:
                            break
                        ev = recheck[0]
                        if eof_action(ev.flags, ev.data, _sel.KQ_EV_EOF) != "drain":
                            break
                _mlog.info("stdin write-end closed (kqueue EOF, drained), self-terminating")
                _os_eof._exit(0)
    except Exception as exc:
        _mlog.debug("stdin EOF monitor error: %s — watchdog will cover", exc)


# D-49: every daemon request this process makes is an MCP tool call, so decisions it records
# (a promotion, a settlement, a label) carry the channel ``mcp`` (core/agent_attribution).
# Set, not defaulted: the channel is the door, and a daemon started by `slm tool` inherits
# that process's ``cli`` (its embedded MCP server must still say ``mcp``).
_os.environ["SLM_CLIENT_CHANNEL"] = "mcp"

if not _embedded_in_daemon:
    _stdin_monitor_thread = threading.Thread(
        target=_stdin_eof_monitor, daemon=True, name="stdin-eof-monitor"
    )
    _stdin_monitor_thread.start()


if __name__ == "__main__":
    server.run(transport="stdio")
