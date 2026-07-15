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

from superlocalmemory.mcp.http_transport import SLMFastMCP

logger = logging.getLogger(__name__)

server = SLMFastMCP("SuperLocalMemory V3")

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
    # v3.6.11: Surface B Optimize tools (5)
    "slm_compress", "slm_retrieve", "slm_cache_set", "slm_cache_get", "slm_optimize_stats",
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

# ---------------------------------------------------------------------------
# v3.6.14 WP-01: Named profile definitions
# ---------------------------------------------------------------------------

_PROFILE_CORE = frozenset({  # 14
    "remember", "recall", "search", "fetch", "list_recent", "update_memory", "forget",
    "session_init", "close_session",
    "slm_compress", "slm_retrieve", "slm_cache_set", "slm_cache_get", "slm_optimize_stats",
})
_PROFILE_CODE = _PROFILE_CORE | frozenset({  # 20
    "build_code_graph", "get_blast_radius", "query_graph",
    "semantic_search_code", "get_review_context", "detect_changes",
})
_PROFILE_FULL_MESH = frozenset({  # 8
    "mesh_summary", "mesh_peers", "mesh_send", "mesh_inbox",
    "mesh_state", "mesh_lock", "mesh_events", "mesh_status",
})
_PROFILE_FULL = frozenset({  # 30 base — EXPLICIT literal, NOT runtime _ESSENTIAL_TOOLS (OQ-2)
    "remember", "recall", "search", "fetch", "list_recent", "delete_memory", "update_memory",
    "get_status", "session_init", "observe", "close_session", "report_feedback", "forget",
    "run_maintenance", "consolidate_cognitive", "get_soft_prompts", "set_mode", "report_outcome",
    "log_tool_event", "get_assertions", "reinforce_assertion", "contradict_assertion",
    "evolve_skill", "skill_health", "skill_lineage",
    "slm_compress", "slm_retrieve", "slm_cache_set", "slm_cache_get", "slm_optimize_stats",
}) | _PROFILE_FULL_MESH  # 38
_PROFILE_POWER = _PROFILE_FULL | frozenset({  # 50
    "get_version", "get_mode", "health", "consistency_check", "recall_trace",
    "get_lifecycle_status", "set_retention_policy", "compact_memories",
    "get_behavioral_patterns", "audit_trail", "quantize", "get_retention_stats",
})
_PROFILE_MESH = _PROFILE_FULL_MESH  # 8

_PROFILE_DEFINITIONS: dict[str, frozenset[str]] = {
    "core": _PROFILE_CORE,
    "code": _PROFILE_CODE,
    "full": _PROFILE_FULL,
    "power": _PROFILE_POWER,
    "mesh": _PROFILE_MESH,
}  # "whole" intentionally absent — maps to raw server (D-2 LOCKED)

# Compatibility aliases published by the v3.6 README.  Keep these explicit so
# a stale client configuration has one deterministic meaning and emits a
# migration warning.  Any other value is a configuration error (fail closed).
_PROFILE_ALIASES: dict[str, str] = {
    "core14": "core",
    "code20": "code",
    "full38": "full",
    "power50": "power",
    "mesh8": "mesh",
    "whole81": "whole",
}

_profile = _os_reg.environ.get("SLM_MCP_PROFILE", "").strip().lower()


def _resolve_profile_allowed(
    profile: str,
    definitions: dict[str, frozenset[str]],
    essential: frozenset[str],
) -> frozenset[str] | None:
    """Resolve a profile name to its allowed tool frozenset, or None for raw-server routes.

    Returns:
        None  — for "" (no selection) or "whole" (raw server, all tools).
        frozenset — the profile's tool set for known profiles.

    Raises:
        ValueError: if ``profile`` is neither canonical nor an explicitly
            supported compatibility alias.
    """
    canonical = _PROFILE_ALIASES.get(profile, profile)
    if canonical != profile:
        logger.warning(
            "SLM_MCP_PROFILE=%r is deprecated; use %r instead.",
            profile,
            canonical,
        )
    if not canonical or canonical == "whole":
        return None
    if canonical in definitions:
        return definitions[canonical]
    valid = ", ".join((*sorted(definitions), "whole"))
    raise ValueError(
        f"SLM_MCP_PROFILE={profile!r} is not recognised; valid profiles: {valid}"
    )


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


# Choose registration target (precedence: ALL > user allowlist > profile > essential)
if _all_tools:
    _target = server                                                              # tier1 (unchanged)
elif _user_allowlist_str:
    _user_allowlist = frozenset(t.strip() for t in _user_allowlist_str.split(",") if t.strip())
    _target = _FilteredServer(server, _user_allowlist)                           # tier2 (unchanged)
elif _profile == "whole":
    _target = server                                                              # NEW raw-server
elif _profile:
    _profile_allowed = _resolve_profile_allowed(
        _profile, _PROFILE_DEFINITIONS, _ESSENTIAL_TOOLS
    )
    _target = (
        server
        if _profile_allowed is None
        else _FilteredServer(server, _profile_allowed)
    )
else:
    _target = _FilteredServer(server, _ESSENTIAL_TOOLS)                          # default (unchanged)

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
from superlocalmemory.mcp.tools_optimize import register_optimize_tools
register_optimize_tools(_target)  # v3.6.11: Surface B Optimize tools (proxy-free)


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

# v3.6.7: Suppress standalone-process behaviours when the MCP server is
# imported inside the daemon (SLM_MCP_EMBEDDED=1). Three threads are safe
# to run in a dedicated `slm mcp` subprocess but harmful inside the daemon:
#   mcp-warmup      — creates a LIGHT engine duplicate; daemon has a FULL one.
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


if not _embedded_in_daemon:
    _watchdog_thread = threading.Thread(target=_parent_watchdog, daemon=True, name="parent-watchdog")
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
    """Exit when the IDE closes our stdin pipe (kqueue — macOS only).

    V3.6.4: kqueue ``EVFILT_READ`` reports ``EV_EOF`` *together with*
    still-readable bytes (``ev.data > 0``) when the write-end closes while a
    final request is buffered. Exiting on the EOF flag alone (pre-3.6.4)
    dropped that in-flight request and self-terminated a session that still
    had work to deliver — strict MCP hosts (e.g. the Hermes agent) then
    logged a keepalive failure and respawned the process. We now defer
    termination until the buffer is genuinely drained (see ``_stdin_guard``).
    """
    import select as _sel, os as _os_eof, time as _time
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


if not _embedded_in_daemon:
    _stdin_monitor_thread = threading.Thread(target=_stdin_eof_monitor, daemon=True, name="stdin-eof-monitor")
    _stdin_monitor_thread.start()


if __name__ == "__main__":
    server.run(transport="stdio")
