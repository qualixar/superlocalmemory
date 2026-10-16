# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""A hook hands its events to the daemon, or spools them; it never drops them.

The daemon is proven to be this account's own (``daemon.json`` + /health)
before anything is sent, the whole hand-off is capped so a hung daemon cannot
stall the user's tool call, and every path that is not an ack ends in the
spool.
"""

from __future__ import annotations

import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from superlocalmemory.hooks import event_delivery as ed
from superlocalmemory.hooks import event_spool


class _Daemon(BaseHTTPRequestHandler):
    health_body = b"{}"
    mode = "ack"  # ack | hang | refuse | fail
    received: list[dict] = []
    headers_seen: list[dict] = []

    def _send(self, status: int, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        self._send(200, type(self).health_body)

    def do_POST(self) -> None:  # noqa: N802
        body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        cls = type(self)
        cls.headers_seen.append({k.lower(): v for k, v in self.headers.items()})
        if cls.mode == "hang":
            time.sleep(2.0)
        if cls.mode == "refuse":
            return self._send(403, b'{"detail": "Invalid daemon capability"}')
        if cls.mode == "fail":
            return self._send(200, b'{"ok": false}')
        cls.received.extend(json.loads(body)["events"])
        return self._send(200, b'{"ok": true, "accepted": 1, "dropped": 0}')

    def log_message(self, *args) -> None:
        pass


@pytest.fixture()
def data_root(tmp_path: Path, monkeypatch) -> Path:
    root = tmp_path / "data"
    root.mkdir()
    monkeypatch.setenv("SLM_DATA_DIR", str(root))
    return root


@pytest.fixture()
def daemon(data_root: Path):
    from superlocalmemory.infra.daemon_identity import build_descriptor, write_descriptor

    _Daemon.mode, _Daemon.received, _Daemon.headers_seen = "ack", [], []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Daemon)
    server.daemon_threads = True
    port = int(server.server_address[1])
    descriptor = build_descriptor(data_root=data_root, port=port, version="test",
                                  pid=os.getpid(), state="ready")
    write_descriptor(descriptor, data_root=data_root)
    _Daemon.health_body = json.dumps({"status": "ok", **descriptor.public_health_fields()}).encode()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield _Daemon, descriptor
    finally:
        server.shutdown()
        thread.join(timeout=5)


def _events(n: int = 2) -> list[dict]:
    return [ed.tool_event(session_id="s1", tool_name=f"T{i}", profile_id="default",
                          tool_input={"i": i}, response_text="ok") for i in range(n)]


def _spooled(data_root: Path) -> list[dict]:
    out = []
    for path in sorted((data_root / "hook_spool").glob("*.jsonl")):
        out += [json.loads(x) for x in path.read_text().splitlines() if x]
    return out


def test_events_go_to_the_daemon_with_its_capability(data_root, daemon) -> None:
    handler, descriptor = daemon
    events = _events()
    assert ed.deliver(events, profile="default") == "daemon"
    assert [e["event_id"] for e in handler.received] == [e["event_id"] for e in events]
    assert handler.headers_seen[0]["x-slm-daemon-capability"] == descriptor.capability
    assert _spooled(data_root) == []


def test_no_daemon_means_the_spool_and_no_network(data_root) -> None:
    started = time.perf_counter()
    events = _events()
    assert ed.deliver(events, profile="default") == "spooled"
    assert (time.perf_counter() - started) < 0.1
    assert [e["event_id"] for e in _spooled(data_root)] == [e["event_id"] for e in events]


@pytest.mark.parametrize("mode", ["refuse", "fail"])
def test_anything_but_an_ack_is_spooled(data_root, daemon, mode) -> None:
    daemon[0].mode = mode
    events = _events()
    assert ed.deliver(events, profile="default") == "spooled"
    assert len(_spooled(data_root)) == 2


def test_a_hung_daemon_cannot_hold_the_hook_past_its_cap(data_root, daemon, monkeypatch) -> None:
    monkeypatch.setenv("SLM_HOOK_DELIVERY_TIMEOUT_MS", "150")
    daemon[0].mode = "hang"
    started = time.perf_counter()
    assert ed.deliver(_events(), profile="default") == "spooled"
    assert (time.perf_counter() - started) < 0.25
    assert len(_spooled(data_root)) == 2


def test_an_unproven_port_never_receives_events(data_root, daemon) -> None:
    daemon[0].health_body = b'{"status": "ok", "service": "someone-else"}'
    assert ed.deliver(_events(), profile="default") == "spooled"
    assert daemon[0].received == [] and daemon[0].headers_seen == []


def test_a_full_spool_reports_the_drop(data_root, monkeypatch) -> None:
    monkeypatch.setenv("SLM_HOOK_SPOOL_MAX_BYTES", "4096")
    outcomes = {ed.deliver(_events(1), profile="default") for _ in range(40)}
    assert "spooled" in outcomes and any(o.startswith("dropped_") for o in outcomes)
    assert event_spool.read_stats(data_root / "hook_spool").dropped > 0


def test_tool_event_summaries_are_capped_and_scrubbed() -> None:
    secret = "sk-" + "A" * 30
    event = ed.tool_event(session_id="s", tool_name="Bash", profile_id="default",
                          tool_input={"cmd": f"echo {secret}"}, response_text="word " * 1000,
                          project_path="/p" * 400)
    assert secret not in event["input_summary"]
    assert len(event["output_summary"]) == 500 and len(event["project_path"]) == 500
    assert len(event["event_id"]) == 32 and event["kind"] == "tool_event"


# ---------------------------------------------------------------------------
# The hook socket (POSIX): the daemon's existing owner-only hook channel
# ---------------------------------------------------------------------------

posix_only = pytest.mark.skipif(not hasattr(__import__("socket"), "AF_UNIX"),
                                reason="AF_UNIX hook socket is POSIX-only")


@pytest.fixture()
def hook_socket(data_root, monkeypatch):
    from superlocalmemory.hooks.hook_daemon import HookDaemon
    from superlocalmemory.server import hook_event_writer as hew

    woke: list[int] = []
    monkeypatch.setattr(hew, "wake", lambda: woke.append(1))
    daemon = HookDaemon(queue_db_path=data_root / "recall_queue.db")
    daemon.start()
    try:
        yield woke
    finally:
        daemon.stop()


@posix_only
def test_the_hook_socket_is_the_fast_path(data_root, hook_socket) -> None:
    ed.deliver(_events(1), profile="default")  # warm imports
    events = _events()
    started = time.perf_counter()
    assert ed.deliver(events, profile="default") == "daemon"
    assert (time.perf_counter() - started) < 0.05
    # The daemon's ack means "spooled by the daemon, writer woken".
    assert [e["event_id"] for e in _spooled(data_root)][-2:] == [e["event_id"] for e in events]
    assert hook_socket == [1, 1]


@posix_only
def test_the_socket_refuses_malformed_events(data_root, hook_socket) -> None:
    from superlocalmemory.hooks.hook_daemon import try_socket_events

    assert try_socket_events([{"kind": "tool_event"}], timeout=1.0) == {
        "ok": False, "error": "invalid_events"}
    assert _spooled(data_root) == [] and hook_socket == []


@posix_only
def test_a_silent_socket_is_spooled_within_the_cap(data_root, monkeypatch) -> None:
    import socket as _socket

    from superlocalmemory.hooks.hook_daemon import _default_sock_path, resolve_sock_path

    monkeypatch.setenv("SLM_HOOK_DELIVERY_TIMEOUT_MS", "150")
    path = resolve_sock_path(_default_sock_path(), create=True)
    listener = _socket.socket(_socket.AF_UNIX, _socket.SOCK_STREAM)
    listener.bind(str(path))
    listener.listen(4)  # accepts into the backlog, never answers
    try:
        started = time.perf_counter()
        assert ed.deliver(_events(), profile="default") == "spooled"
        assert (time.perf_counter() - started) < 0.25
        assert len(_spooled(data_root)) == 2
    finally:
        listener.close()
        path.unlink(missing_ok=True)


def test_the_wait_ends_inside_the_hook_budget(monkeypatch) -> None:
    """Work the hook did before delivering is paid out of the same budget."""
    monkeypatch.delenv("SLM_HOOK_DELIVERY_TIMEOUT_MS", raising=False)
    now = time.perf_counter()
    assert ed._deadline_s() == ed.DEFAULT_DELIVERY_TIMEOUT_MS / 1000  # noqa: SLF001
    assert 0.1 < ed._deadline_s(now) <= ed.DEFAULT_DELIVERY_TIMEOUT_MS / 1000  # noqa: SLF001
    late = time.perf_counter() - (ed.HOOK_BUDGET_MS - 40) / 1000
    # At most the 40 ms left (less whatever this line itself took under load).
    assert 0.01 <= ed._deadline_s(late) <= 0.041  # noqa: SLF001
    assert ed._deadline_s(now - 5) == 0.01  # noqa: SLF001 - floor, never zero


def test_a_hook_that_ran_long_still_ends_inside_its_budget(data_root, daemon) -> None:
    daemon[0].mode = "hang"
    started = time.perf_counter() - 0.1  # the hook already spent 100 ms
    assert ed.deliver(_events(), profile="default", started=started) == "spooled"
    # A hung daemon would hold the hook for seconds; the margin covers the spool
    # write and scheduler noise on a busy machine (206 ms was seen under load).
    assert (time.perf_counter() - started) * 1000 < ed.HOOK_BUDGET_MS + 250


@posix_only
def test_an_oversized_socket_request_is_cut_off_unanswered(data_root, hook_socket) -> None:
    """The socket used to buffer a request line without bound."""
    import socket as _socket

    from superlocalmemory.hooks.hook_daemon import _MAX_REQUEST_BYTES, owned_socket_path

    client = _socket.socket(_socket.AF_UNIX, _socket.SOCK_STREAM)
    client.settimeout(5)
    client.connect(str(owned_socket_path()))
    try:
        try:
            client.sendall(b"x" * (_MAX_REQUEST_BYTES + 65536))
        except (BrokenPipeError, ConnectionResetError):
            pass
        assert client.recv(64) == b""  # closed, no answer
    finally:
        client.close()
    assert _spooled(data_root) == []
