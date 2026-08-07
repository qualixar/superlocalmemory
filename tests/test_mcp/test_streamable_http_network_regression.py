"""Network regression for Streamable-HTTP tool-result completion (mcp 2.0.0).

Uses the official mcp 2.0 Client over a real Uvicorn socket. Fully-stateless
+ json_response is the production default.
"""

from __future__ import annotations

import asyncio
import contextlib
import socket
import threading
from collections.abc import AsyncIterator
from typing import Callable

import pytest


@contextlib.asynccontextmanager
async def _running_mcp_server(
    register_tools: Callable[[object], None] | None = None,
) -> AsyncIterator[str]:
    """Serve a stateless SLM MCP app on a loopback socket for one test."""
    import uvicorn
    from fastapi import FastAPI

    from superlocalmemory.mcp.http_transport import SLMFastMCP

    mcp = SLMFastMCP("streamable-http-regression")

    if register_tools is None:
        @mcp.tool()
        async def recall(query: str, limit: int = 2) -> dict[str, object]:
            """Return a deliberately non-trivial response without a live database."""
            return {
                "query": query,
                "limit": limit,
                "results": [{"content": "x" * 65_536, "score": 0.99}],
            }
    else:
        register_tools(mcp)

    mcp_app = mcp.streamable_http_app(
        streamable_http_path="/",
        stateless_http=True,
        json_response=True,
        event_store=None,
        host="127.0.0.1",
    )

    @contextlib.asynccontextmanager
    async def lifespan(_app: FastAPI):
        async with mcp_app.router.lifespan_context(mcp_app):
            yield

    app = FastAPI(lifespan=lifespan)
    app.mount("/mcp", mcp_app)

    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.bind(("127.0.0.1", 0))
    _, port = probe.getsockname()
    probe.close()

    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    )
    task = asyncio.create_task(server.serve())
    try:
        for _ in range(100):
            if server.started:
                break
            await asyncio.sleep(0.02)
        assert server.started, "Uvicorn did not start the MCP regression server"
        yield f"http://127.0.0.1:{port}/mcp"
    finally:
        server.should_exit = True
        await asyncio.wait_for(task, timeout=10)


@pytest.mark.asyncio
async def test_stateless_streamable_http_tool_result_completes_over_network() -> None:
    """A large tools/call result must complete through the official Client."""
    from mcp.client import Client

    async with _running_mcp_server() as endpoint:
        started = asyncio.get_running_loop().time()
        async with Client(endpoint, mode="auto") as client:
            tools = await asyncio.wait_for(client.list_tools(), timeout=10)
            assert any(tool.name == "recall" for tool in tools.tools)

            result = await asyncio.wait_for(
                client.call_tool(
                    "recall", {"query": "transport regression", "limit": 2}
                ),
                timeout=10,
            )

        elapsed = asyncio.get_running_loop().time() - started
        assert result is not None
        # Prefer isError attribute when present
        if hasattr(result, "isError"):
            assert not result.isError
        if hasattr(result, "content"):
            assert result.content
        assert elapsed < 10, f"Streamable HTTP tools/call took {elapsed:.3f}s"


@pytest.mark.asyncio
async def test_core_recall_resolves_daemon_proxy_off_event_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mounted HTTP recall must not synchronously probe the daemon on its own loop."""
    from mcp.client import Client

    from superlocalmemory.mcp import _daemon_proxy
    from superlocalmemory.mcp.tools_core import register_core_tools

    event_loop_thread = threading.get_ident()
    factory_threads: list[int] = []

    class _Pool:
        def recall(self, **kwargs: object) -> dict[str, object]:
            return {
                "ok": True,
                "results": [{"content": "x" * 65_536, "score": 0.99}],
                "result_count": 1,
                "query_type": "semantic",
                "channel_weights": {"semantic": 1.0},
            }

    def _choose_pool() -> _Pool:
        factory_threads.append(threading.get_ident())
        return _Pool()

    monkeypatch.setattr(_daemon_proxy, "choose_pool", _choose_pool)

    async with _running_mcp_server(
        lambda server: register_core_tools(server, lambda: None)
    ) as endpoint:
        async with Client(endpoint, mode="auto") as client:
            result = await asyncio.wait_for(
                client.call_tool("recall", {"query": "transport regression"}),
                timeout=10,
            )

    assert result is not None
    if hasattr(result, "isError"):
        assert not result.isError
    assert factory_threads
    assert all(thread_id != event_loop_thread for thread_id in factory_threads)


@pytest.mark.asyncio
async def test_core_remember_resolves_daemon_proxy_off_event_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mounted HTTP remember fallback must not probe its own event loop."""
    from mcp.client import Client

    from superlocalmemory.cli import daemon as daemon_client
    from superlocalmemory.mcp import _daemon_proxy
    from superlocalmemory.mcp.tools_core import register_core_tools

    event_loop_thread = threading.get_ident()
    factory_threads: list[int] = []

    class _Pool:
        def store(
            self,
            content: str,
            metadata: dict[str, object],
        ) -> dict[str, object]:
            return {
                "ok": True,
                "fact_ids": ["http-remember-witness"],
                "count": 1,
                "materialization_state": "complete",
            }

    def _choose_pool() -> _Pool:
        factory_threads.append(threading.get_ident())
        return _Pool()

    monkeypatch.setattr(daemon_client, "is_daemon_running", lambda: False)
    monkeypatch.setattr(_daemon_proxy, "choose_pool", _choose_pool)

    async with _running_mcp_server(
        lambda server: register_core_tools(server, lambda: None)
    ) as endpoint:
        async with Client(endpoint, mode="auto") as client:
            result = await asyncio.wait_for(
                client.call_tool(
                    "remember",
                    {"content": "HTTP remember thread-boundary witness"},
                ),
                timeout=10,
            )

    assert result is not None
    if hasattr(result, "isError"):
        assert not result.isError
    assert factory_threads
    assert all(thread_id != event_loop_thread for thread_id in factory_threads)
