# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""SLM Mesh MCP Tools — P2P agent communication via the unified daemon.

v3.4.4: These tools ship WITH SuperLocalMemory, no separate slm-mesh install needed.
End users get full mesh functionality from `pip install superlocalmemory`.

All tools communicate with the daemon's Python mesh broker on port 8765.
Auto-heartbeat keeps the session alive as long as the MCP server is running.

8 tools: mesh_summary, mesh_peers, mesh_send, mesh_inbox,
         mesh_state, mesh_lock, mesh_events, mesh_status
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
import uuid
from typing import Callable

from mcp.types import ToolAnnotations

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# M03: Simple circuit breaker for mesh_send
# ---------------------------------------------------------------------------
# Prevents runaway retries (~30s stall) when the daemon is dead.
# State machine: CLOSED → OPEN (after 3 consecutive failures)
#               OPEN → HALF_OPEN (after 60s cooldown)
#               HALF_OPEN → CLOSED (on one successful probe)
#               HALF_OPEN → OPEN (on probe failure)

_CB_FAILURE_THRESHOLD = 3
_CB_COOLDOWN_SECONDS = 60

# Client-side mesh message cap (mirrors the broker's MAX_MESSAGE_SIZE).
MAX_MESSAGE_SIZE = 4096

_CB_STATE_CLOSED = "closed"
_CB_STATE_OPEN = "open"
_CB_STATE_HALF_OPEN = "half_open"


class _SendCircuitBreaker:
    """Thread-safe circuit breaker scoped to mesh_send daemon calls."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._state: str = _CB_STATE_CLOSED
        self._failure_count: int = 0
        self._opened_at: float = 0.0

    def reset(self) -> None:
        """Reset to CLOSED; used by tests for isolation."""
        with self._lock:
            self._state = _CB_STATE_CLOSED
            self._failure_count = 0
            self._opened_at = 0.0

    def is_open(self) -> bool:
        with self._lock:
            if self._state == _CB_STATE_OPEN:
                if time.monotonic() - self._opened_at >= _CB_COOLDOWN_SECONDS:
                    self._state = _CB_STATE_HALF_OPEN
                    return False  # allow one probe
                return True
            return False

    def allow_request(self) -> bool:
        """Return True if the call should proceed; False if circuit is OPEN.

        State transitions inside the lock prevent concurrent probe races:
        - CLOSED → allow
        - OPEN (cooldown not elapsed) → block
        - OPEN (cooldown elapsed) → transition to HALF_OPEN, allow one probe,
          and immediately re-enter OPEN so any concurrent call is blocked until
          the probe result comes in via record_success / record_failure.
        - HALF_OPEN → block (probe already dispatched and not yet resolved)
        """
        with self._lock:
            if self._state == _CB_STATE_CLOSED:
                return True
            if self._state == _CB_STATE_OPEN:
                if time.monotonic() - self._opened_at >= _CB_COOLDOWN_SECONDS:
                    # Grant exactly one probe by briefly entering HALF_OPEN then
                    # going back to OPEN.  Subsequent callers are blocked until
                    # record_success or record_failure resolves the probe.
                    self._state = _CB_STATE_HALF_OPEN
                    return True
                return False
            # HALF_OPEN: probe is in flight — block until resolved
            return False

    def record_success(self) -> None:
        with self._lock:
            self._state = _CB_STATE_CLOSED
            self._failure_count = 0

    def record_failure(self) -> None:
        with self._lock:
            if self._state == _CB_STATE_HALF_OPEN:
                # Probe failed — reopen immediately
                self._state = _CB_STATE_OPEN
                self._opened_at = time.monotonic()
                return
            self._failure_count += 1
            if self._failure_count >= _CB_FAILURE_THRESHOLD:
                self._state = _CB_STATE_OPEN
                self._opened_at = time.monotonic()
                logger.warning(
                    "mesh_send circuit breaker OPEN after %d consecutive failures; "
                    "fast-failing for %ds",
                    self._failure_count,
                    _CB_COOLDOWN_SECONDS,
                )


_SEND_CIRCUIT = _SendCircuitBreaker()

# Unique peer ID for this MCP server session
_PEER_ID = str(uuid.uuid4())[:12]
_SESSION_SUMMARY = ""
_PROJECT_PATH = ""  # v3.4.6: detected from cwd or CLAUDE_PROJECT_DIR
_HEARTBEAT_INTERVAL = 25  # seconds (broker marks stale at 30s, dead at 60s)
_HEARTBEAT_THREAD: threading.Thread | None = None
_REGISTERED = False


def _detect_project_path() -> str:
    """Detect current project path from env or cwd."""
    return (
        os.environ.get("CLAUDE_PROJECT_DIR")
        or os.environ.get("PROJECT_PATH")
        or os.getcwd()
    )


def _mesh_request(method: str, path: str, body: dict | None = None) -> dict | None:
    """Send an exact-instance, capability-authenticated mesh request."""
    try:
        from superlocalmemory.cli.daemon import daemon_request

        return daemon_request(method, f"/mesh{path}", body)
    except Exception as exc:
        logger.debug("Mesh request failed: %s %s — %s", method, path, exc)
        return None


def _ensure_registered() -> None:
    """Register this session with the mesh broker if not already."""
    global _REGISTERED, _PROJECT_PATH, _PEER_ID
    if _REGISTERED:
        return

    _PROJECT_PATH = _detect_project_path()
    result = _mesh_request("POST", "/register", {
        "peer_id": _PEER_ID,
        "session_id": os.environ.get("CLAUDE_SESSION_ID", _PEER_ID),
        "summary": _SESSION_SUMMARY or "SLM MCP session",
        "project_path": _PROJECT_PATH,
        # Peer identity is the canonical SLM_AGENT_ID (same var memory
        # attribution uses), so Antigravity/Hermes/Cursor/etc. show as
        # themselves instead of collapsing to "claude_code". Fall back to the
        # legacy CLAUDE_AGENT_TYPE, then a generic default.
        "agent_type": (
            os.environ.get("SLM_AGENT_ID")
            or os.environ.get("CLAUDE_AGENT_TYPE")
            or "claude_code"
        ),
    })
    if result:
        # v3.6.12 (mesh-1): the broker mints its OWN peer_id (RegisterRequest has
        # no peer_id field, so our body value is dropped by pydantic). Adopt the
        # broker's id BEFORE starting the heartbeat, otherwise heartbeat/send/
        # inbox all target a non-existent peer → 404s and the session is reaped.
        _PEER_ID = result.get("peer_id", _PEER_ID)
        _REGISTERED = True
        _start_heartbeat()
        pending = result.get("pending_messages", 0)
        if pending > 0:
            logger.info("Mesh: %d pending messages waiting", pending)


def _start_heartbeat() -> None:
    """Background thread that sends heartbeat to keep session alive."""
    global _HEARTBEAT_THREAD
    if _HEARTBEAT_THREAD is not None:
        return

    def heartbeat_loop():
        while True:
            time.sleep(_HEARTBEAT_INTERVAL)
            try:
                _mesh_request("POST", "/heartbeat", {"peer_id": _PEER_ID})
            except Exception:
                pass

    _HEARTBEAT_THREAD = threading.Thread(target=heartbeat_loop, daemon=True, name="mesh-heartbeat")
    _HEARTBEAT_THREAD.start()
    logger.info("Mesh heartbeat started (peer_id=%s, interval=%ds)", _PEER_ID, _HEARTBEAT_INTERVAL)


def auto_register_mesh() -> None:
    """Called from server.py warmup to register this session immediately.

    v3.4.6: Sessions register at MCP startup, not lazily on first tool call.
    This ensures every Claude session is visible on the mesh from the start.
    """
    _ensure_registered()


def register_mesh_tools(server, get_engine: Callable) -> None:
    """Register all 8 mesh MCP tools."""

    @server.tool()
    async def mesh_summary(summary: str = "") -> dict:
        """Register this session and describe what you're working on.

        Call this at the start of every session. Other agents can see your summary
        and send you messages. The session stays alive via automatic heartbeat.
        v3.4.6: Sessions auto-register at MCP startup, but calling this updates
        the summary so other sessions know what you're doing.

        Args:
            summary: What this session is working on (e.g. "Fixing auth bug in api.py")
        """
        global _SESSION_SUMMARY
        _SESSION_SUMMARY = summary or "Active session"

        await asyncio.to_thread(_ensure_registered)

        # Update summary
        result = await asyncio.to_thread(
            _mesh_request, "POST", "/summary",
            {"peer_id": _PEER_ID, "summary": _SESSION_SUMMARY},
        )

        return {
            "peer_id": _PEER_ID,
            "summary": _SESSION_SUMMARY,
            "project_path": _PROJECT_PATH,
            "registered": _REGISTERED,
            "heartbeat_active": _HEARTBEAT_THREAD is not None and _HEARTBEAT_THREAD.is_alive(),
            "broker_response": result,
        }

    @server.tool(annotations=ToolAnnotations(readOnlyHint=True))
    async def mesh_peers() -> dict:
        """List all active peer sessions on this machine.

        Shows other Claude Code, Cursor, or AI agent sessions that are
        connected to the same SLM mesh network.
        """
        await asyncio.to_thread(_ensure_registered)
        result = await asyncio.to_thread(_mesh_request, "GET", "/peers")
        peers = (result or {}).get("peers", [])
        return {
            "peers": peers,
            "count": len(peers),
            "my_peer_id": _PEER_ID,
        }

    @server.tool()
    async def mesh_send(to: str, message: str) -> dict:
        """Send a message to another peer session, broadcast, or project.

        Args:
            to: Target — one of:
                - A peer_id from mesh_peers (direct message)
                - "broadcast" (all active + future sessions within 48h)
                - "project:/path/to/dir" (all sessions in that project directory)
            message: The message content (max 4KB — use file paths for large data)
        """
        # Enforce the documented 4KB notification cap client-side too (the
        # broker also caps, but fail fast without a round-trip).
        if len(message.encode("utf-8")) > MAX_MESSAGE_SIZE:
            return {"ok": False, "error": (
                f"message too large (max {MAX_MESSAGE_SIZE} bytes) — "
                "reference a file path instead")}
        # M03: circuit breaker — fast-fail if daemon is repeatedly unreachable
        if not _SEND_CIRCUIT.allow_request():
            return {
                "ok": False,
                "error": (
                    "mesh_send circuit open: daemon unreachable after repeated failures. "
                    f"Retrying in {_CB_COOLDOWN_SECONDS}s."
                ),
            }

        await asyncio.to_thread(_ensure_registered)
        result = await asyncio.to_thread(
            _mesh_request, "POST", "/send",
            {"from_peer": _PEER_ID, "to_peer": to, "content": message},
        )
        # Circuit tracks daemon-unreachable (None) only — not valid broker errors
        # like "recipient not found", which are application-level, not failures.
        if result is None:
            _SEND_CIRCUIT.record_failure()
            return {"ok": False, "error": "Failed to send message"}
        _SEND_CIRCUIT.record_success()
        return result

    @server.tool()
    async def mesh_inbox() -> dict:
        """Read messages sent to this session.

        Returns unread messages: direct + broadcast + project-targeted.
        Broadcast/project messages are delivered to ALL matching sessions.
        Messages auto-expire after 48 hours.
        """
        await asyncio.to_thread(_ensure_registered)
        from urllib.parse import quote
        project = _PROJECT_PATH or _detect_project_path()
        messages = await asyncio.to_thread(
            _mesh_request, "GET",
            f"/inbox/{_PEER_ID}?project_path={quote(project, safe='')}",
        )
        msg_list = (messages or {}).get("messages", [])
        # Auto-mark unread messages as read. v3.6.12 (failopen-2): use .get("id")
        # — a malformed broker message without an "id" key used to raise KeyError
        # out to the agent, violating the never-raise contract.
        unread_ids = [m["id"] for m in msg_list
                      if not m.get("read") and m.get("id") is not None]
        if unread_ids:
            await asyncio.to_thread(
                _mesh_request, "POST", f"/inbox/{_PEER_ID}/read",
                {"message_ids": unread_ids},
            )
        return {
            "messages": msg_list,
            "count": len(msg_list),
            "unread": len(unread_ids),
        }

    @server.tool()
    async def mesh_state(key: str = "", value: str = "", action: str = "get") -> dict:
        """Get or set shared state across all sessions.

        Shared state is visible to authenticated peers. Use it for non-secret
        coordination metadata such as feature flags and task assignments.
        Credentials, tokens, passwords, and API keys are rejected.

        Args:
            key: State key name
            value: Value to set (only for action="set")
            action: "get" (read all or one key), "set" (write a key)
        """
        await asyncio.to_thread(_ensure_registered)

        if action == "set" and key:
            result = await asyncio.to_thread(
                _mesh_request, "POST", "/state",
                {"key": key, "value": value, "set_by": _PEER_ID},
            )
            return result or {"ok": False, "error": "Failed to set state"}

        if key:
            result = await asyncio.to_thread(_mesh_request, "GET", f"/state/{key}")
            return result or {"key": key, "value": None}

        result = await asyncio.to_thread(_mesh_request, "GET", "/state")
        return result or {"state": {}}

    @server.tool()
    async def mesh_lock(
        file_path: str,
        action: str = "query",
    ) -> dict:
        """Manage file locks across sessions.

        Before editing a shared file, check if another session has it locked.

        Args:
            file_path: Absolute path to the file
            action: "query" (check lock), "acquire" (lock file), "release" (unlock)
        """
        # Require a non-empty absolute path; a relative/blank path is ambiguous
        # and lets a caller probe arbitrary strings via the coordination store.
        if not file_path or not (
            file_path.startswith("/")
            or (len(file_path) >= 3 and file_path[1] == ":")
        ):
            return {"ok": False, "error": "file_path must be a non-empty absolute path"}
        await asyncio.to_thread(_ensure_registered)
        result = await asyncio.to_thread(
            _mesh_request, "POST", "/lock",
            {"file_path": file_path, "action": action, "locked_by": _PEER_ID},
        )
        return result or {"ok": False, "error": "Lock operation failed"}

    @server.tool(annotations=ToolAnnotations(readOnlyHint=True))
    async def mesh_events() -> dict:
        """Get recent mesh events (peer joins, leaves, messages, state changes).

        Shows the activity log of the mesh network.
        """
        result = await asyncio.to_thread(_mesh_request, "GET", "/events")
        return result or {"events": []}

    @server.tool(annotations=ToolAnnotations(readOnlyHint=True))
    async def mesh_status() -> dict:
        """Get mesh broker health and statistics.

        Shows broker uptime, peer count, and connection status.
        """
        result = await asyncio.to_thread(_mesh_request, "GET", "/status")
        if result:
            result["my_peer_id"] = _PEER_ID
            result["heartbeat_active"] = _HEARTBEAT_THREAD is not None and _HEARTBEAT_THREAD.is_alive()
        return result or {
            "broker_up": False,
            "error": "Cannot reach mesh broker. Is the daemon running? (slm serve start)",
        }
