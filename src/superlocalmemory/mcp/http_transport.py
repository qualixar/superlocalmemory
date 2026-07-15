# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Resource-safe FastMCP Streamable-HTTP integration.

MCP SDK 1.27.1 passes an AnyIO ``MemoryObjectReceiveStream`` directly to
``EventSourceResponse`` for every JSON-RPC POST.  The response consumes that
stream but does not close it after normal iteration, leaving one receive
endpoint per request for the garbage collector.  The response is the owner of
that per-request iterator, so SLM closes it at the response boundary.
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP
from sse_starlette.sse import EventSourceResponse
from starlette.types import Receive, Scope, Send


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


class SLMFastMCP(FastMCP):
    """FastMCP with deterministic per-request SSE resource cleanup."""

    def streamable_http_app(self):
        install_streamable_http_resource_guard()
        return super().streamable_http_app()
