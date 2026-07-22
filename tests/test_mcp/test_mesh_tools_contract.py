"""Sandbox contracts for the eight daemon-backed Mesh MCP tools."""

from __future__ import annotations

import asyncio


class _ToolCollector:
    def __init__(self) -> None:
        self.tools: dict[str, object] = {}

    def tool(self, *args, **kwargs):
        def decorate(func):
            self.tools[func.__name__] = func
            return func

        return decorate


def test_mesh_mcp_tools_use_one_broker_identity_and_unread_delivery(monkeypatch) -> None:
    """Every exposed Mesh tool uses the broker identity minted at registration."""
    import superlocalmemory.mcp.tools_mesh as mesh_tools

    collector = _ToolCollector()
    mesh_tools.register_mesh_tools(collector, lambda: None)

    calls: list[tuple[str, str, dict | None]] = []

    def request(method: str, path: str, body: dict | None = None) -> dict:
        calls.append((method, path, body))
        if path == "/register":
            return {"peer_id": "broker-peer", "pending_messages": 0}
        if path.startswith("/inbox/broker-peer?"):
            return {
                "messages": [
                    {"id": 7, "content": "one delivery", "read": False},
                    {"content": "malformed receipt", "read": False},
                ]
            }
        if path == "/peers":
            return {"peers": [{"peer_id": "other-peer"}]}
        if path == "/events":
            return {"events": [{"event_type": "message_sent"}]}
        if path == "/status":
            return {"broker_up": True, "peer_count": 2}
        if path == "/state" and method == "GET":
            return {"state": {"release": {"value": "v3.7"}}}
        if path == "/state/release":
            return {"key": "release", "value": "v3.7"}
        return {"ok": True}

    monkeypatch.setattr(mesh_tools, "_mesh_request", request)
    monkeypatch.setattr(mesh_tools, "_start_heartbeat", lambda: None)
    monkeypatch.setattr(mesh_tools, "_REGISTERED", False)
    monkeypatch.setattr(mesh_tools, "_PEER_ID", "local-provisional")
    monkeypatch.setattr(mesh_tools, "_HEARTBEAT_THREAD", None)
    monkeypatch.setattr(mesh_tools, "_PROJECT_PATH", "/sandbox/project")
    monkeypatch.setenv("PROJECT_PATH", "/sandbox/project")

    async def exercise() -> None:
        assert (await collector.tools["mesh_summary"]("audit mesh"))["peer_id"] == "broker-peer"
        assert (await collector.tools["mesh_peers"]())["count"] == 1
        assert (await collector.tools["mesh_send"]("other-peer", "ready"))["ok"] is True
        inbox = await collector.tools["mesh_inbox"]()
        assert inbox["unread"] == 1
        assert inbox["count"] == 2
        assert (await collector.tools["mesh_state"]())["state"]["release"]["value"] == "v3.7"
        assert (await collector.tools["mesh_state"]("release"))["value"] == "v3.7"
        assert (await collector.tools["mesh_state"]("release", "v3.7", "set"))["ok"] is True
        assert (await collector.tools["mesh_lock"]("/sandbox/file.py", "acquire"))["ok"] is True
        assert len((await collector.tools["mesh_events"]())["events"]) == 1
        assert (await collector.tools["mesh_status"]())["my_peer_id"] == "broker-peer"

    asyncio.run(exercise())

    assert calls[0] == (
        "POST",
        "/register",
        {
            "peer_id": "local-provisional",
            "session_id": "local-provisional",
            "summary": "audit mesh",
            "project_path": "/sandbox/project",
            "agent_type": "claude_code",
        },
    )
    assert ("POST", "/send", {"from_peer": "broker-peer", "to_peer": "other-peer", "content": "ready"}) in calls
    assert ("POST", "/inbox/broker-peer/read", {"message_ids": [7]}) in calls
    assert ("POST", "/state", {"key": "release", "value": "v3.7", "set_by": "broker-peer"}) in calls
    assert ("POST", "/lock", {"file_path": "/sandbox/file.py", "action": "acquire", "locked_by": "broker-peer"}) in calls


# ---------------------------------------------------------------------------
# M03: Circuit-breaker tests for mesh_send
# ---------------------------------------------------------------------------

class _CollectorWithAnnotations:
    """Collector that also accepts ToolAnnotations keyword from @server.tool."""
    def __init__(self) -> None:
        self.tools: dict[str, object] = {}

    def tool(self, *args, **kwargs):
        def decorate(func):
            self.tools[func.__name__] = func
            return func
        return decorate


def _make_mesh_send_tool(monkeypatch):
    """Return the mesh_send coroutine with a fresh module-level circuit state."""
    import importlib
    import superlocalmemory.mcp.tools_mesh as mesh_tools
    # Force fresh module state
    monkeypatch.setattr(mesh_tools, "_REGISTERED", True)
    monkeypatch.setattr(mesh_tools, "_PEER_ID", "cb-peer")
    monkeypatch.setattr(mesh_tools, "_PROJECT_PATH", "/cb/proj")
    monkeypatch.setattr(mesh_tools, "_HEARTBEAT_THREAD", object())  # non-None

    collector = _CollectorWithAnnotations()
    mesh_tools.register_mesh_tools(collector, lambda: None)
    return collector.tools["mesh_send"], mesh_tools


def test_circuit_breaker_opens_after_3_failures(monkeypatch) -> None:
    """After 3 consecutive None returns from _mesh_request the circuit must
    open and subsequent calls must fast-fail without calling _mesh_request."""
    import asyncio

    send_fn, mesh_tools = _make_mesh_send_tool(monkeypatch)

    send_call_count = 0

    def failing_request(method, path, body=None):
        nonlocal send_call_count
        send_call_count += 1
        return None  # simulates daemon unreachable

    monkeypatch.setattr(mesh_tools, "_mesh_request", failing_request)
    # Reset circuit state so tests are isolated
    if hasattr(mesh_tools, "_SEND_CIRCUIT"):
        mesh_tools._SEND_CIRCUIT.reset()

    async def run():
        # Calls 1-3: should reach _mesh_request and record failures
        for _ in range(3):
            result = await send_fn("other-peer", "ping")
            assert result.get("ok") is False

        calls_before_open = send_call_count

        # Call 4+: circuit should be OPEN — must NOT call _mesh_request again
        result = await send_fn("other-peer", "ping-after-open")
        assert result.get("ok") is False
        assert "circuit" in result.get("error", "").lower(), (
            f"Expected circuit-open error, got: {result}"
        )
        # _mesh_request must NOT have been called for the 4th attempt
        assert send_call_count == calls_before_open, (
            "mesh_request was called even though circuit should be open"
        )

    asyncio.run(run())


def test_circuit_breaker_half_open_allows_one_trial(monkeypatch) -> None:
    """After the 60s cooldown the circuit moves to HALF_OPEN and allows one
    probe.  A successful probe resets the circuit to CLOSED."""
    import asyncio
    import time as _time

    send_fn, mesh_tools = _make_mesh_send_tool(monkeypatch)

    if hasattr(mesh_tools, "_SEND_CIRCUIT"):
        cb = mesh_tools._SEND_CIRCUIT
        cb.reset()
    else:
        pytest.skip("Circuit breaker not yet implemented (_SEND_CIRCUIT missing)")

    # Drive circuit OPEN
    def always_none(method, path, body=None):
        return None

    monkeypatch.setattr(mesh_tools, "_mesh_request", always_none)

    async def open_circuit():
        for _ in range(3):
            await send_fn("p", "msg")

    asyncio.run(open_circuit())
    assert cb.is_open()

    # Simulate passage of 60s by monkeypatching time.monotonic
    original_monotonic = _time.monotonic
    base = original_monotonic()
    monkeypatch.setattr(_time, "monotonic", lambda: base + 61)

    probe_calls = []

    def probe_request(method, path, body=None):
        probe_calls.append((method, path))
        return {"ok": True}  # success

    monkeypatch.setattr(mesh_tools, "_mesh_request", probe_request)

    async def half_open_probe():
        result = await send_fn("p", "probe")
        return result

    result = asyncio.run(half_open_probe())
    assert result.get("ok") is True, f"Probe should succeed: {result}"
    assert len(probe_calls) == 1, "Half-open must allow exactly one probe"
    assert not cb.is_open(), "Circuit must close after a successful probe"


def test_circuit_breaker_success_resets_failure_count(monkeypatch) -> None:
    """Two failures followed by one success must reset the failure count so the
    circuit stays closed (does not trip after one more failure)."""
    import asyncio

    send_fn, mesh_tools = _make_mesh_send_tool(monkeypatch)

    if hasattr(mesh_tools, "_SEND_CIRCUIT"):
        mesh_tools._SEND_CIRCUIT.reset()
    else:
        pytest.skip("Circuit breaker not yet implemented")

    call_results = iter([None, None, {"ok": True}, None])

    def varying_request(method, path, body=None):
        return next(call_results, {"ok": True})

    monkeypatch.setattr(mesh_tools, "_mesh_request", varying_request)

    async def run():
        await send_fn("p", "fail-1")   # failure 1
        await send_fn("p", "fail-2")   # failure 2
        r = await send_fn("p", "ok")   # success → resets counter
        assert r.get("ok") is True
        # One more failure — counter should be 1, not tripping the breaker at 3
        r2 = await send_fn("p", "fail-again")
        assert r2.get("ok") is False
        # Circuit should still be CLOSED (only 1 consecutive failure since reset)
        assert not mesh_tools._SEND_CIRCUIT.is_open()

    asyncio.run(run())
