"""4.1.22: daemon-backed calls made while the daemon is still starting.

Before 4.1.22 the first ``remember`` after an MCP host started SLM failed at
once with ``DAEMON_UNAVAILABLE (daemon_unreachable)``, because the daemon the
MCP server had just spawned needed seconds to answer ``/health`` and every
call waited only for one 2 s probe. These tests pin the replacement contract:

* a start in progress (descriptor ``starting`` + live process, or this process
  spawning) gets a bounded wait, then the call proceeds;
* the wait is bounded, is not repeated by back-to-back retries, and ends
  with a truthful ``daemon_starting`` diagnosis — never a fake success;
* nothing starting means no wait (old behaviour), and the wait never spawns.

Slow-start daemons are small stub processes (a real PID that answers
``/health`` with the descriptor's identity after a delay); one test drives the
real ``slm mcp`` server against a real cold daemon.
"""

from __future__ import annotations

import json
import os
import queue
import socket
import subprocess
import sys
import threading
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from superlocalmemory.infra.daemon_identity import (
    build_descriptor,
    process_create_time_for,
    write_descriptor,
)
from superlocalmemory.infra.process_identity import process_start_token_for

_STUB = r"""
import json, pathlib, socket, sys, time
port, delay, mode = int(sys.argv[1]), float(sys.argv[2]), sys.argv[3]
health_path = pathlib.Path(sys.argv[4])
if mode != "slow":
    time.sleep(delay)
if mode == "exit":
    sys.exit(0)
srv = socket.socket(); srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
srv.bind(("127.0.0.1", port)); srv.listen(16)
while True:
    conn, _ = srv.accept()
    if mode == "silent":
        continue  # accept, never answer: a port reserved before HTTP is up
    if mode == "slow":
        time.sleep(delay)  # alive and listening already; slow to answer
    data = conn.recv(65536).decode("latin-1")
    health = health_path.read_text()
    stopping = data.startswith("POST /stop")
    if data.startswith("GET /health"):
        body = health
    elif stopping:
        body = json.dumps({"status": "stopping"})
    else:
        body = json.dumps({"ok": True, "fact_ids": ["stub-fact"]})
    raw = body.encode()
    conn.sendall(b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: "
                 + str(len(raw)).encode() + b"\r\nConnection: close\r\n\r\n" + raw)
    conn.close()
    if stopping:
        srv.close()
        sys.exit(0)
"""


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def _publish(descriptor, proc):
    """Publish ``descriptor`` for ``proc`` (the PID is only known after spawn)."""
    final = replace(
        descriptor,
        pid=proc.pid,
        process_create_time=process_create_time_for(proc.pid),
        process_start_token=process_start_token_for(proc.pid),
    )
    write_descriptor(final, data_root=Path(os.environ["SLM_DATA_DIR"]))
    return final


@pytest.fixture
def no_spawn(monkeypatch):
    """Any spawn attempt during these tests is a defect."""
    from superlocalmemory.cli import daemon

    def _forbidden(*_a, **_k):
        raise AssertionError("a starting daemon must never trigger a second spawn")

    monkeypatch.setattr(daemon, "_start_daemon_subprocess", _forbidden)


def _identity_stub(work: Path, *, delay, mode="serve", state="starting"):
    """Spawn a stub whose health echoes the descriptor it is published under.

    The descriptor holds the PID, which is only known after the spawn, so the
    stub reads its health body from a file written right after.
    """
    from superlocalmemory import __version__

    port = _free_port()
    work.mkdir(parents=True, exist_ok=True)
    health_file = work / "health.json"
    script = work / "stub.py"
    script.write_text(_STUB, encoding="utf-8")
    proc = subprocess.Popen(
        [sys.executable, str(script), str(port), str(delay), mode, str(health_file)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    base = build_descriptor(port=port, version=__version__, pid=proc.pid, state=state)
    descriptor = _publish(base, proc)
    health_file.write_text(
        json.dumps({"status": "ok", **descriptor.public_health_fields()}), encoding="utf-8",
    )
    return proc, descriptor


@pytest.fixture
def stubs(tmp_path):
    started: list[subprocess.Popen] = []

    def start(**kwargs):
        proc, descriptor = _identity_stub(tmp_path / f"stub{len(started)}", **kwargs)
        started.append(proc)
        return proc, descriptor

    yield start
    for proc in started:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)


def test_a_starting_daemon_is_waited_for_and_the_call_goes_through(stubs, no_spawn):
    from superlocalmemory.cli import daemon

    _proc, _descriptor = stubs(delay=1.5)
    began = time.monotonic()
    response = daemon.daemon_request("POST", "/remember", {"content": "probe"})
    took = time.monotonic() - began

    assert response is not None and response.get("fact_ids") == ["stub-fact"], response
    assert 1.0 <= took < 8.0, took


def test_the_wait_is_bounded_and_ends_with_a_truthful_starting_answer(
    stubs, no_spawn, monkeypatch,
):
    from superlocalmemory.cli import daemon
    from superlocalmemory.mcp._daemon_proxy import daemon_unavailable_error

    monkeypatch.setenv("SLM_DAEMON_START_WAIT_S", "1.5")
    proc, _descriptor = stubs(delay=600)  # never listens within the test
    began = time.monotonic()
    assert daemon.daemon_request("POST", "/remember", {"content": "probe"}) is None
    took = time.monotonic() - began
    assert 1.4 <= took < 2.5, took

    diagnosis = daemon.describe_daemon_unavailability()
    assert diagnosis["reason"] == "daemon_starting"
    assert str(proc.pid) in diagnosis["message"]
    assert "not sent" in diagnosis["message"]
    assert "Retry in about" in diagnosis["hint"]
    assert daemon_unavailable_error().startswith("DAEMON_UNAVAILABLE (daemon_starting):")

    # remember retries three times inside one tool call; they must not each
    # wait the whole 1.5 s budget again. One short probe is allowed: on
    # Windows a connect to a port nobody bound takes the full 0.5 s probe.
    began = time.monotonic()
    assert daemon.daemon_request("POST", "/remember", {"content": "probe"}) is None
    assert time.monotonic() - began < 1.0


def test_a_port_reserved_before_http_is_up_cannot_stretch_the_wait(
    stubs, no_spawn, monkeypatch,
):
    """Connects succeed and then hang: each probe is cut to the time left."""
    from superlocalmemory.cli import daemon

    monkeypatch.setenv("SLM_DAEMON_START_WAIT_S", "1")
    stubs(delay=0, mode="silent")
    time.sleep(0.5)  # let the stub bind
    began = time.monotonic()
    assert daemon.daemon_request("GET", "/health") is None
    # one pre-existing 2 s probe + a 1 s budget + scheduling slack
    assert time.monotonic() - began < 2.5  # 0.5 s first probe + 1 s budget + slack


def test_a_slow_but_alive_health_reply_is_not_reported_unavailable(
    stubs, no_spawn, monkeypatch,
):
    """4.1.22 correction (CRIT #1): shortening the Windows fail-fast probe
    must not narrow POSIX's own patience for a real, busy-but-alive daemon.
    A real stub process binds and accepts the connection immediately (it is
    not "starting"), then takes 1.2 s to answer /health -- well under the
    2 s POSIX budget and well over the 0.5 s Windows-only one, so this
    would wrongly read DAEMON_UNAVAILABLE if the short probe ever leaked
    onto POSIX. Since issue #319 the same holds on Windows for a ready
    daemon whose process is alive."""
    from superlocalmemory.cli import daemon
    from superlocalmemory.cli.daemon_startup import STARTING_PROBE_S

    # Checked on the read limit the real call path hands the health fetch, not
    # on a stopwatch: a 1.2 s stub reply against the 2 s limit failed on a
    # loaded machine, where the stub itself was late.
    _proc, _descriptor = stubs(delay=0, mode="slow", state="ready")
    real_fetch, limits = daemon._fetch_health, []

    def spy(port, timeout=2.0):
        limits.append(timeout)
        return real_fetch(port, timeout=timeout)

    monkeypatch.setattr(daemon, "_fetch_health", spy)
    health = None
    for _ in range(100):  # the stub binds asynchronously; liveness only
        health = daemon.daemon_request("GET", "/health")
        if health is not None:
            break
        time.sleep(0.1)
    assert health is not None and health.get("status") == "ok", health
    assert limits and limits[0] == 2.0 > STARTING_PROBE_S, limits


def test_nothing_starting_keeps_the_old_fail_fast_behaviour(stubs, no_spawn):
    from superlocalmemory.cli import daemon

    stubs(delay=600, state="ready")  # alive, "ready", never answers
    began = time.monotonic()
    assert daemon.daemon_request("POST", "/remember", {"content": "probe"}) is None
    assert time.monotonic() - began < 1.0
    assert daemon.describe_daemon_unavailability()["reason"] == "daemon_unreachable"


def test_a_daemon_that_dies_while_starting_ends_the_wait_early(
    stubs, no_spawn, monkeypatch,
):
    from superlocalmemory.cli import daemon

    monkeypatch.setenv("SLM_DAEMON_START_WAIT_S", "15")
    proc, _descriptor = stubs(delay=0.5, mode="exit")
    began = time.monotonic()
    assert daemon.daemon_request("POST", "/remember", {"content": "probe"}) is None
    assert time.monotonic() - began < 5.0
    proc.wait(timeout=5)
    assert daemon.describe_daemon_unavailability()["reason"] == "daemon_process_exited"


def test_a_pinned_descriptor_never_waits(stubs, no_spawn):
    """``slm serve stop`` pins the descriptor it inspected; it must not stall."""
    from superlocalmemory.cli import daemon

    _proc, descriptor = stubs(delay=600)
    began = time.monotonic()
    assert daemon.daemon_request("POST", "/stop", expected_descriptor=descriptor) is None
    assert time.monotonic() - began < 1.0


def test_an_in_process_spawn_is_waited_for_and_never_duplicated(no_spawn, monkeypatch):
    """A second thread of the process that is spawning must not spawn again."""
    from superlocalmemory.cli import daemon, daemon_startup

    monkeypatch.setenv("SLM_TEST_ALLOW_DAEMON_SPAWN", "1")
    monkeypatch.setenv("SLM_DAEMON_START_WAIT_S", "1")
    with daemon_startup.spawning():
        assert daemon.describe_daemon_unavailability()["reason"] == "daemon_starting"
        began = time.monotonic()
        assert daemon.ensure_daemon() is False  # bounded wait, no second spawn
        assert daemon.daemon_request("GET", "/health") is None
        assert time.monotonic() - began < 3.5
    assert not daemon_startup.this_process_is_spawning()


def test_stop_waits_for_a_starting_daemon_instead_of_leaving_it_running(stubs):
    """Before 4.1.22 `slm serve stop` during a start said "not running" and the
    daemon kept running (it leaked from this very test file once)."""
    from superlocalmemory.cli import daemon

    proc, _descriptor = stubs(delay=1.5)
    began = time.monotonic()
    assert daemon.stop_daemon() is True
    assert 1.0 <= time.monotonic() - began < 20.0
    proc.wait(timeout=5)
    assert proc.returncode == 0


@pytest.mark.parametrize(
    ("raw", "cap", "expected"),
    [
        ("", None, 20.0), ("5", None, 5.0), ("0", None, 0.0), ("-3", None, 0.0),
        ("999", None, 45.0), ("nan", None, 20.0), ("junk", None, 20.0), ("", 3.0, 3.0),
    ],
)
def test_the_start_wait_budget_is_clamped(monkeypatch, raw, cap, expected):
    from superlocalmemory.cli.daemon_startup import start_wait_budget

    monkeypatch.setenv("SLM_DAEMON_START_WAIT_S", raw)
    assert start_wait_budget(cap) == expected


class _RecordingDaemonModule:
    """A duck-typed stand-in for the ``daemon`` module ``probe_health`` takes:
    records the timeout it was actually asked to read with, instead of
    depending on how fast any given OS refuses a connect to a closed port
    (that varies by platform and is exactly what made the Windows bug
    invisible on POSIX CI: the OS difference never showed up in a test that
    only checks the final True/False answer)."""

    def __init__(self, *, reachable: bool):
        self.reachable = reachable
        self.fetch_calls: list[float] = []

    def _fetch_health(self, port, timeout=2.0):
        self.fetch_calls.append(timeout)
        return {"status": "ok"} if self.reachable else None


@pytest.mark.parametrize("state", ["starting", "ready", "", None])
def test_health_probe_timeout_is_short_on_every_path(state, monkeypatch):
    """The fix for the Windows fail-fast regression, pinned without a real
    socket: the one-shot probe used whenever nothing proves a start is in
    progress must ask for the same short read as the starting-daemon polling
    loop already does on Windows -- not the old, much longer default that
    cost a closed port the full read timeout on a platform slow to refuse
    it. Forced to win32 regardless of where this test actually runs: the
    short bound is Windows-specific (see test_non_starting_probe_stays_2s_
    on_posix below), not a universal one."""
    from superlocalmemory.cli import daemon_startup
    from superlocalmemory.cli.daemon_startup import STARTING_PROBE_S, health_probe_timeout

    monkeypatch.setattr(daemon_startup.sys, "platform", "win32")
    descriptor = SimpleNamespace(state=state) if state is not None else object()
    assert health_probe_timeout(descriptor) == STARTING_PROBE_S


@pytest.mark.parametrize("state", ["ready", "", None])
def test_non_starting_probe_stays_2s_on_posix(state, monkeypatch):
    """4.1.22 correction: the Windows fail-fast fix must not narrow POSIX's
    patience for a real, busy-but-alive daemon. A refused connect on POSIX
    is fast regardless of the timeout value, so there is nothing to gain
    there from shortening it, and something to lose (see the module test
    below with a real slow-but-alive stub)."""
    from superlocalmemory.cli import daemon_startup
    from superlocalmemory.cli.daemon_startup import health_probe_timeout

    monkeypatch.setattr(daemon_startup.sys, "platform", "darwin")
    descriptor = SimpleNamespace(state=state) if state is not None else object()
    assert health_probe_timeout(descriptor) == 2.0


def test_probe_health_passes_the_given_timeout_straight_through():
    from superlocalmemory.cli import daemon_startup

    stub = _RecordingDaemonModule(reachable=True)
    assert daemon_startup.probe_health(stub, port=1, remaining=2.0) == {"status": "ok"}
    assert stub.fetch_calls == [2.0]


def test_probe_health_returns_none_when_unreachable():
    from superlocalmemory.cli import daemon_startup

    stub = _RecordingDaemonModule(reachable=False)
    assert daemon_startup.probe_health(stub, port=1, remaining=2.0) is None


# ---------------------------------------------------------------------------
# The real composed path: a host starts `slm mcp` on an empty data root and
# calls remember at once, while the daemon it just spawned is still starting.
# ---------------------------------------------------------------------------

def _mcp_env(tmp_path: Path, port: int) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith("SLM_")}
    home = tmp_path / "home"
    cache = home / ".cache"
    home.mkdir()
    (tmp_path / "data").mkdir()
    src = Path(__file__).resolve().parents[2] / "src"
    env.update(
        HOME=str(home), USERPROFILE=str(home), SLM_DATA_DIR=str(tmp_path / "data"),
        SLM_DAEMON_PORT=str(port), SLM_DISABLE_LEGACY_PORT="1",
        SLM_NON_INTERACTIVE="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
        HF_HOME=str(cache / "huggingface"), XDG_CACHE_HOME=str(cache),
        OMP_NUM_THREADS="1", TOKENIZERS_PARALLELISM="false",
        PYTHONPATH=os.pathsep.join(filter(None, [str(src), env.get("PYTHONPATH", "")])),
    )
    return env


def _stop_spawned_daemon(env: dict[str, str], data_root: Path) -> None:
    """Stop the daemon this test caused, by its own data dir; never leak it.

    ``slm serve stop`` is the supported path (it waits out a start). If a
    descriptor still names a live process afterwards, that exact PID — and
    only it — is terminated.
    """
    import psutil

    try:
        subprocess.run(
            [sys.executable, "-m", "superlocalmemory.cli.main", "serve", "stop"],
            env=env, capture_output=True, text=True, timeout=150,
        )
    except subprocess.TimeoutExpired:
        pass
    try:
        pid = int(json.loads((data_root / "daemon.json").read_text(encoding="utf-8"))["pid"])
    except (OSError, ValueError, KeyError, TypeError):
        pid_file = data_root / "daemon.pid"
        try:
            pid = int(pid_file.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            return
    try:
        owned = psutil.Process(pid)
        if "superlocalmemory.server.unified_daemon" not in " ".join(owned.cmdline()):
            return
        children = owned.children(recursive=True)
        owned.terminate()
        _gone, alive = psutil.wait_procs([owned, *children], timeout=20)
        for leftover in alive:
            leftover.kill()
    except psutil.Error:
        pass


def test_first_remember_after_a_real_cold_start_succeeds(tmp_path):
    port = _free_port()
    env = _mcp_env(tmp_path, port)
    proc = subprocess.Popen(
        [sys.executable, "-m", "superlocalmemory.cli.main", "mcp"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        text=True, env=env,
    )
    lines: "queue.Queue[str]" = queue.Queue()
    threading.Thread(target=lambda: [lines.put(x) for x in proc.stdout], daemon=True).start()

    def send(message: dict) -> None:
        proc.stdin.write(json.dumps(message) + "\n")
        proc.stdin.flush()

    def reply(request_id: int, limit: float = 90.0) -> dict:
        end = time.monotonic() + limit
        while time.monotonic() < end:
            try:
                message = json.loads(lines.get(timeout=1))
            except (queue.Empty, ValueError):
                continue
            if message.get("id") == request_id:
                return message
        raise AssertionError(f"no reply to request {request_id}")

    def tool(request_id: int, name: str, arguments: dict) -> dict:
        send({"jsonrpc": "2.0", "id": request_id, "method": "tools/call",
              "params": {"name": name, "arguments": arguments}})
        content = reply(request_id)["result"]["content"]
        return json.loads(content[0]["text"])

    try:
        send({"jsonrpc": "2.0", "id": 1, "method": "initialize",
              "params": {"protocolVersion": "2024-11-05", "capabilities": {},
                         "clientInfo": {"name": "cold-start-test", "version": "0"}}})
        reply(1)
        send({"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}})
        # A loaded machine can take longer than the wait budget to start. Then the
        # only acceptable answer is the truthful "still starting" one, and a host
        # retrying as told must get through. Every attempt stays bounded.
        request_id = 2
        deadline = time.monotonic() + 240.0
        while True:
            began = time.monotonic()
            stored = tool(request_id, "remember",
                          {"content": "The cold start probe shelf code is K-77."})
            took = time.monotonic() - began
            # Bounded: far inside the ~60 s a host allows a tool call.
            assert took < 50.0, took
            if stored.get("success") is True:
                break
            assert stored.get("retryable") is True, stored
            assert "daemon_starting" in str(stored.get("error", "")), stored
            assert time.monotonic() < deadline, "SLM never finished starting"
            request_id += 1
            time.sleep(10)
        assert stored.get("fact_ids"), stored
        found = tool(request_id + 1, "recall", {"query": "cold start probe shelf code"})
        assert found.get("success") is True, found
    finally:
        try:
            proc.stdin.close()
            proc.wait(timeout=15)
        except Exception:
            proc.kill()
            proc.wait(timeout=5)
        _stop_spawned_daemon(env, tmp_path / "data")


def test_the_first_probe_comes_out_of_the_same_budget(stubs, no_spawn, monkeypatch):
    """The first health probe ran before the bounded wait and then the wait got
    the whole budget again: a 1.5 s budget took 3 s on a loaded machine."""
    from superlocalmemory.cli import daemon, daemon_startup

    monkeypatch.setenv("SLM_DAEMON_START_WAIT_S", "1.5")
    stubs(delay=600)  # never listens within the test
    real_probe = daemon_startup.probe_health
    calls = []

    def slow_first_probe(d, port, remaining, **options):
        calls.append(remaining)
        if len(calls) == 1:
            time.sleep(0.8)
            return None
        return real_probe(d, port, remaining, **options)

    monkeypatch.setattr(daemon_startup, "probe_health", slow_first_probe)
    began = time.monotonic()
    assert daemon.daemon_request("POST", "/remember", {"content": "probe"}) is None
    took = time.monotonic() - began
    assert took < 1.5 + 0.6, took  # old: >= 0.8 + 1.5


# Issue #319: on Windows a READY daemon whose process is alive was given the
# same 0.5 s /health read as a port nobody bound, so a busy-but-healthy
# daemon came back as DAEMON_UNAVAILABLE. The short read exists only to stop
# a closed port's slow refusal from costing 2 s; a live ready daemon is not
# that case.
def test_windows_ready_daemon_with_live_process_gets_the_posix_read(monkeypatch):
    from superlocalmemory.cli import daemon, daemon_startup
    from superlocalmemory.cli.daemon_startup import health_probe_timeout

    monkeypatch.setattr(daemon_startup.sys, "platform", "win32")
    monkeypatch.delenv("SLM_DAEMON_HEALTH_PROBE_S", raising=False)
    monkeypatch.setattr(daemon, "_is_pid_alive", lambda pid: pid == 4242)
    assert health_probe_timeout(SimpleNamespace(state="ready", pid=4242)) == 2.0
    # Dead process or no PID on record: still the fast-fail read.
    assert health_probe_timeout(SimpleNamespace(state="ready", pid=7)) == 0.5
    assert health_probe_timeout(SimpleNamespace(state="ready")) == 0.5
    # A starting daemon keeps the short polling read everywhere.
    assert health_probe_timeout(SimpleNamespace(state="starting", pid=4242)) == 0.5


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("5", 5.0), ("0.01", 0.25), ("600", 10.0), ("nonsense", 2.0), ("", 2.0)],
)
def test_health_probe_read_is_configurable_within_bounds(raw, expected, monkeypatch):
    from superlocalmemory.cli import daemon, daemon_startup
    from superlocalmemory.cli.daemon_startup import health_probe_timeout

    monkeypatch.setattr(daemon_startup.sys, "platform", "win32")
    monkeypatch.setattr(daemon, "_is_pid_alive", lambda pid: True)
    monkeypatch.setenv("SLM_DAEMON_HEALTH_PROBE_S", raw)
    assert health_probe_timeout(SimpleNamespace(state="ready", pid=1)) == expected
    # The setting never lengthens a starting daemon's polling read.
    assert health_probe_timeout(SimpleNamespace(state="starting", pid=1)) == 0.5


# Issue #319 follow-up: the long read for a live daemon must not make a
# closed port slow on Windows, and the configured read must really be used.
def test_windows_closed_port_fails_on_the_short_connect(monkeypatch):
    from superlocalmemory.cli import daemon_startup

    monkeypatch.setattr(daemon_startup.sys, "platform", "win32")
    tried = []
    monkeypatch.setattr(daemon_startup, "loopback_accepts",
                        lambda port, timeout: tried.append(timeout) or False)
    stub = _RecordingDaemonModule(reachable=True)
    assert daemon_startup.probe_health(stub, port=1, remaining=2.0,
                                       connect_first=True) is None
    assert tried == [daemon_startup.STARTING_PROBE_S]
    assert stub.fetch_calls == []  # never waited the long read


def test_windows_listening_daemon_gets_the_whole_read(monkeypatch):
    from superlocalmemory.cli import daemon_startup

    monkeypatch.setattr(daemon_startup.sys, "platform", "win32")
    monkeypatch.setattr(daemon_startup, "loopback_accepts", lambda port, timeout: True)
    stub = _RecordingDaemonModule(reachable=True)
    assert daemon_startup.probe_health(stub, port=1, remaining=5.0, ceiling=10.0,
                                       connect_first=True) == {"status": "ok"}
    assert stub.fetch_calls == [5.0]


def test_connect_first_is_windows_only(monkeypatch):
    from superlocalmemory.cli import daemon_startup

    monkeypatch.setattr(daemon_startup.sys, "platform", "darwin")
    monkeypatch.setattr(daemon_startup, "loopback_accepts",
                        lambda port, timeout: pytest.fail("POSIX refuses at once"))
    stub = _RecordingDaemonModule(reachable=True)
    assert daemon_startup.probe_health(stub, port=1, remaining=2.0,
                                       connect_first=True) == {"status": "ok"}


def test_configured_read_reaches_the_http_call(monkeypatch):
    """SLM_DAEMON_HEALTH_PROBE_S above 2 s must not be cut back to 2 s."""
    import urllib.request

    from superlocalmemory.cli import daemon

    seen = []

    def fake_urlopen(url, timeout):
        seen.append(timeout)
        raise OSError("not listening")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    assert daemon._fetch_health(1, timeout=5.0) is None
    assert daemon._fetch_health(1, timeout=60.0) is None
    assert seen == [5.0, 10.0]


def test_loopback_accepts_reports_a_closed_port():
    import socket

    from superlocalmemory.cli import daemon_startup

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen(1)
        port = sock.getsockname()[1]
        assert daemon_startup.loopback_accepts(port, 0.5) is True
    assert daemon_startup.loopback_accepts(port, 0.5) is False
