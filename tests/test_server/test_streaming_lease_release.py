# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE file
# Part of SuperLocalMemory V3
"""Streaming requests must not hold the profile-runtime operation lease.

ROOT CAUSE (production, observed repeatedly)
--------------------------------------------
``ProfileRuntimeMiddleware`` held one operation lease for the ENTIRE duration
of every non-exempt HTTP request. Two endpoints stream indefinitely:

  * ``/events/stream`` — a ``while True: await sleep(1)`` SSE loop that runs
    for as long as a dashboard browser tab is open (i.e. forever).
  * ``/api/v3/chat/stream`` — streams LLM tokens for up to 120s.

While such a connection was open the lease never released, so the drain loop in
``ProfileRuntime.transition()`` could never reach zero in-flight operations and
EVERY profile switch timed out with HTTP 503 — the daemon appeared "dead" on
switch even though it was healthy.

FIX
---
The lease guards the SYNCHRONOUS route work that produces the response, not the
streaming of the body. The middleware now releases the lease at
``http.response.start`` (by then the handler has returned its Response;
streaming generators re-acquire their own short lease for any engine work).

These tests drive the middleware at the raw ASGI layer so they reproduce the
exact wedge without a live daemon.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from superlocalmemory.server.profile_runtime import (
    ProfileRuntime,
    ProfileRuntimeMiddleware,
)


def _http_scope(path: str, method: str = "GET") -> dict:
    return {"type": "http", "path": path, "method": method, "headers": []}


async def _noop_receive() -> dict:
    return {"type": "http.request", "body": b"", "more_body": False}


async def test_streaming_request_releases_lease_at_response_start() -> None:
    """A long-lived SSE body must release the operation lease the instant the
    response starts — NOT hold it for the connection's lifetime.

    This is the exact production scenario: an open ``/events/stream`` tab must
    not block a concurrent profile switch.
    """
    runtime = ProfileRuntime("alpha")
    app_state = SimpleNamespace(profile_runtime=runtime)

    start_sent = asyncio.Event()
    allow_body = asyncio.Event()

    async def streaming_app(scope, receive, send) -> None:
        # Emit response start (as StreamingResponse does before iterating the
        # generator), then block — simulating an open SSE connection.
        await send({"type": "http.response.start", "status": 200, "headers": []})
        start_sent.set()
        await allow_body.wait()
        await send({"type": "http.response.body", "body": b"data: x\n\n",
                    "more_body": False})

    middleware = ProfileRuntimeMiddleware(streaming_app, app_state=app_state)

    sent: list[dict] = []

    async def capture_send(message) -> None:
        sent.append(message)

    task = asyncio.create_task(
        middleware(_http_scope("/events/stream"), _noop_receive, capture_send)
    )

    # Wait until the streaming app has emitted response.start.
    await asyncio.wait_for(start_sent.wait(), timeout=2.0)
    # Give the event loop a tick so the middleware's _send wrapper processes it.
    await asyncio.sleep(0)

    # THE REGRESSION ASSERTION: the lease is already released even though the
    # SSE body is still streaming (allow_body has not been set).
    assert runtime._active_operations == 0, (
        "Streaming request still holds the operation lease after "
        "http.response.start — a profile switch would 503 while the SSE "
        "connection is open."
    )

    # And a real profile switch must succeed WHILE the stream is still open.
    committed: list[str] = []
    await asyncio.to_thread(
        runtime.transition, "beta", lambda prev, target: committed.append(target)
    )
    assert committed == ["beta"], "Switch must commit while an SSE stream is open"
    assert runtime.snapshot.profile_id == "beta"

    # Let the streaming body finish and confirm no double-release / underflow.
    allow_body.set()
    await asyncio.wait_for(task, timeout=2.0)
    assert runtime._active_operations == 0


async def test_normal_request_holds_lease_until_response_start() -> None:
    """A normal buffered request holds the lease across the handler and releases
    exactly once — the lease must still protect synchronous route work.
    """
    runtime = ProfileRuntime("alpha")
    app_state = SimpleNamespace(profile_runtime=runtime)

    in_handler = asyncio.Event()
    resume_handler = asyncio.Event()

    async def normal_app(scope, receive, send) -> None:
        # Simulate synchronous route work that must run under the lease.
        in_handler.set()
        await resume_handler.wait()
        assert runtime._active_operations == 1, (
            "Lease must be held for the duration of the handler's work"
        )
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"{}",
                    "more_body": False})

    middleware = ProfileRuntimeMiddleware(normal_app, app_state=app_state)

    async def capture_send(message) -> None:
        pass

    task = asyncio.create_task(
        middleware(_http_scope("/status"), _noop_receive, capture_send)
    )
    await asyncio.wait_for(in_handler.wait(), timeout=2.0)
    # While the handler runs, the lease is held.
    assert runtime._active_operations == 1
    resume_handler.set()
    await asyncio.wait_for(task, timeout=2.0)
    # Released exactly once after completion.
    assert runtime._active_operations == 0


async def test_handler_error_before_response_start_releases_lease() -> None:
    """A handler that raises BEFORE emitting response.start must still release
    its lease (the finally-block safety net), never leaking a lease.
    """
    runtime = ProfileRuntime("alpha")
    app_state = SimpleNamespace(profile_runtime=runtime)

    async def failing_app(scope, receive, send) -> None:
        raise RuntimeError("boom before response.start")

    middleware = ProfileRuntimeMiddleware(failing_app, app_state=app_state)

    async def capture_send(message) -> None:
        pass

    with pytest.raises(RuntimeError, match="boom"):
        await middleware(_http_scope("/status"), _noop_receive, capture_send)

    # Safety net released the lease despite the error.
    assert runtime._active_operations == 0
    # And a switch still works — no leaked lease wedges the daemon.
    committed: list[str] = []
    await asyncio.to_thread(
        runtime.transition, "beta", lambda prev, target: committed.append(target)
    )
    assert committed == ["beta"]


async def test_transition_request_is_exempt_from_leasing() -> None:
    """The profile-switch route itself must not take an operation lease
    (it would wait on its own drain forever)."""
    runtime = ProfileRuntime("alpha")
    app_state = SimpleNamespace(profile_runtime=runtime)

    saw_lease: list[int] = []

    async def switch_app(scope, receive, send) -> None:
        saw_lease.append(runtime._active_operations)
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"{}",
                    "more_body": False})

    middleware = ProfileRuntimeMiddleware(switch_app, app_state=app_state)

    async def capture_send(message) -> None:
        pass

    await middleware(
        _http_scope("/api/profiles/beta/switch", "POST"),
        _noop_receive,
        capture_send,
    )
    assert saw_lease == [0], "Transition requests must not hold an operation lease"
    assert runtime._active_operations == 0
