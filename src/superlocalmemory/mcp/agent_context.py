"""Per-HTTP-request agent ID resolution — ContextVar home.

Kept in a standalone module so both tools_core and tools_active can import
it without creating circular dependencies (server.py → tools_core → here,
and unified_daemon.py → here independently).

Priority chain (HTTP-first, stdio-fallback):
  1. ContextVar set by _AgentIDExtractorASGI middleware from /mcp/{agent_id} URL path.
  2. SLM_AGENT_ID environment variable (stdio transport legacy).
  3. Hard-coded "mcp_client" sentinel.
"""
from __future__ import annotations

import contextvars
import os
import re

# v3.6.12 (parity-1): default is "" (the "no agent routed" sentinel), NOT the
# user-visible "mcp_client". Sanitized agent ids are [A-Za-z0-9._-], so "" can
# never collide — a client that explicitly routes to /mcp/mcp_client is now
# distinguishable from a bare /mcp/ request with no agent segment.
_current_agent_id: contextvars.ContextVar[str] = contextvars.ContextVar(
    "slm_agent_id", default=""
)

# Agent ids arrive from an untrusted URL path segment. They are ATTRIBUTION
# metadata, never an authenticated principal — but they reach loggers, the
# agent registry, and SQL-bound attribution columns, so we hard-restrict the
# charset at the single extraction chokepoint. This neutralises log-injection
# (CRLF / ANSI), oversized ids, and any path-ish characters in one place.
_AGENT_ID_SANITIZE = re.compile(r"[^A-Za-z0-9._-]")
_AGENT_ID_MAX_LEN = 64


def sanitize_agent_id(raw: str) -> str:
    """Coerce an untrusted agent-id segment to a safe, bounded token."""
    return _AGENT_ID_SANITIZE.sub("_", raw)[:_AGENT_ID_MAX_LEN]


def get_current_agent_id(env_fallback: bool = True) -> str:
    """Return the agent_id for the current asyncio task.

    For HTTP transport the ASGI wrapper sets the ContextVar from the URL path
    before the request reaches any MCP tool, so this returns the URL-derived id.
    For stdio transport the ContextVar holds its default ("mcp_client") and we
    fall through to the SLM_AGENT_ID env var instead.
    """
    ctx_id = _current_agent_id.get()
    if ctx_id:
        return ctx_id  # an explicitly-routed agent id (incl. "mcp_client")
    if env_fallback:
        return os.environ.get("SLM_AGENT_ID", "mcp_client")
    return "mcp_client"


class AgentIDExtractorASGI:
    """ASGI wrapper that maps ``/mcp/{agent_id}`` → the agent-id ContextVar.

    Mounted at ``/mcp`` in unified_daemon. IMPORTANT: Starlette's ``Mount``
    (≥0.35 / 1.x) does NOT strip the mount prefix from ``scope["path"]`` — it
    records the prefix in ``scope["root_path"]`` and leaves ``path`` as the full
    request path (e.g. ``/mcp/claude`` with ``root_path == "/mcp"``). So we
    compute the mount-relative sub-path ourselves as ``path[len(root_path):]``.

    Flow for ``POST /mcp/claude``:
      sub-path ``/claude`` → agent id ``claude`` → set ContextVar → rewrite the
      scope path to ``{root_path}/`` so the inner FastMCP app (Starlette, route
      ``/``) sees the same mount-relative ``/`` it sees for a bare ``/mcp/``.

    Backward compatible: bare ``/mcp/`` has sub-path ``/`` → no agent segment →
    the request passes through untouched and the ContextVar keeps its
    ``"mcp_client"`` default.

    Per-request isolation is guaranteed by ContextVar + ``reset(token)`` in a
    ``finally``, so concurrent HTTP sessions never see each other's agent id.
    """

    __slots__ = ("_app",)

    def __init__(self, inner) -> None:
        self._app = inner

    async def __call__(self, scope, receive, send):
        if scope.get("type") == "http":
            root_path: str = scope.get("root_path", "")
            full_path: str = scope.get("path", "/")
            # Mount-relative sub-path (what comes AFTER /mcp). When a root_path
            # is present the request path MUST start with it (Starlette Mount
            # guarantees this); if it somehow does not, treat it as no-agent and
            # pass through untouched rather than mis-parsing the full path.
            if root_path:
                if not full_path.startswith(root_path):
                    await self._app(scope, receive, send)
                    return
                subpath = full_path[len(root_path):]
            else:
                subpath = full_path
            first = subpath.lstrip("/").split("/")[0]
            if first:
                first = sanitize_agent_id(first)
                token = _current_agent_id.set(first)
                # Rewrite the path so the inner app sees the bare mount root,
                # exactly as it would for a no-agent /mcp/ request.
                new_full = (root_path + "/") if root_path else "/"
                new_scope = {
                    **scope,
                    "path": new_full,
                    "raw_path": new_full.encode(),
                }
                try:
                    await self._app(new_scope, receive, send)
                finally:
                    _current_agent_id.reset(token)
                return
        await self._app(scope, receive, send)
