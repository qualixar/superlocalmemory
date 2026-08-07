# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V4 | https://qualixar.com | https://varunpratap.com

"""Resource-safe Streamable-HTTP integration for mcp==2.0.0.

mcp 2.0.0 deleted ``mcp.server.fastmcp.FastMCP``. Replacement is
``mcp.server.mcpserver.MCPServer`` with the same ``.tool()`` decorator and
``run(transport="stdio")``.

Fully-stateless is the default (see ``remote_mode.mcp_stateless`` and
``unified_daemon._configure_mcp_transport_settings``). Under
``stateless_http=True``:

* ``session_idle_timeout`` is illegal (SDK raises RuntimeError) and unused —
  there are no transport sessions to reap.
* the EventStore (SSE Last-Event-ID resumability) is not used.
* therefore the old ``SLMFastMCP.streamable_http_app()`` override that
  pre-created a ``StreamableHTTPSessionManager`` is gone — kwargs go
  straight to ``MCPServer.streamable_http_app(...)``.

Application-level ``session_init`` / ``close_session`` are orthogonal
(session_id is a str param persisted in the memories table) and unchanged.
"""

from __future__ import annotations

import logging

from mcp.server.mcpserver import MCPServer
from sse_starlette.sse import EventSourceResponse
from starlette.types import Receive, Scope, Send

from superlocalmemory import __version__

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SSE resource guard (still useful if a host opts into non-json SSE responses)
# ---------------------------------------------------------------------------


class ClosingEventSourceResponse(EventSourceResponse):
    """EventSourceResponse that closes the async iterator it consumes."""

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            close = getattr(self.body_iterator, "aclose", None)
            if close is not None:
                await close()


def install_streamable_http_resource_guard() -> None:
    """Install the response owner used by MCP's Streamable-HTTP transport."""
    from mcp.server import streamable_http

    streamable_http.EventSourceResponse = ClosingEventSourceResponse


# ---------------------------------------------------------------------------
# SLMFastMCP — thin MCPServer wrapper (name kept for import stability)
# ---------------------------------------------------------------------------


class SLMFastMCP(MCPServer):
    """MCPServer with SLM release identity.

    Named ``SLMFastMCP`` for backward-compatible imports. Behaviour is
    fully-stateless by default at the *call site* via
    ``streamable_http_app(stateless_http=True, json_response=True, ...)`` —
    this class does not override that method.

    ``version`` is passed to ``MCPServer.__init__`` directly (no private
    ``_mcp_server.version`` poke — that attribute is gone in mcp 2.0.0).
    """

    def __init__(self, *args, product_version: str = __version__, **kwargs) -> None:
        # MCPServer takes version= as a first-class kwarg.
        kwargs.setdefault("version", product_version)
        super().__init__(*args, **kwargs)
        # Optional SSE resource guard for hosts that still request event-stream.
        install_streamable_http_resource_guard()
