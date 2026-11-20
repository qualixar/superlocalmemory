# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Real daemons on a pytest data folder and port: one owner, the right settings.

1. A client that auto-starts the daemon directly hands it a scrubbed
   environment: the client's agent name, channel and tool profile never reach
   the shared daemon.
2. When the installed service starts its daemon while an unowned daemon holds
   the folder (the state an older client leaves behind), the service's daemon
   takes over: the unowned one stops, the service's one serves, and exactly one
   daemon owns the folder. The "service" is a real daemon process whose
   service-manager queries are stubbed to name it, so no launchd/systemd job
   is ever registered.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

psutil = pytest.importorskip("psutil")

pytestmark = [
    pytest.mark.skipif(os.environ.get("SLM_SKIP_REAL_DAEMON") == "1",
                       reason="real daemon start disabled by SLM_SKIP_REAL_DAEMON=1"),
    pytest.mark.skipif(sys.platform == "win32", reason="POSIX process environment"),
]

CLIENT_ENV = {
    "SLM_AGENT_ID": "agent-under-test",
    "SLM_CLIENT_CHANNEL": "mcp",
    "SLM_MCP_PROFILE": "whole",
    "SLM_SKIP_DEP_CHECK": "1",
}
_READY_S = 120


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _record(data_dir: Path) -> dict | None:
    try:
        return json.loads((data_dir / "daemon.json").read_text())
    except (OSError, ValueError):
        return None


def _wait_ready(data_dir: Path, pid: int | None = None, deadline_s: float = _READY_S) -> dict:
    deadline = time.monotonic() + deadline_s
    while time.monotonic() < deadline:
        record = _record(data_dir)
        if record and record.get("state") == "ready" and (pid is None or record["pid"] == pid):
            return record
        time.sleep(0.5)
    raise AssertionError(f"no ready daemon (pid={pid}); last record: {_record(data_dir)}")


def _stop(pid: int) -> None:
    try:
        proc = psutil.Process(pid)
        proc.terminate()
        proc.wait(timeout=60)
    except psutil.NoSuchProcess:
        return
    except psutil.TimeoutExpired:
        proc.kill()


@pytest.fixture
def namespace(tmp_path, monkeypatch):
    port = _free_port()
    monkeypatch.setenv("SLM_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("SLM_DAEMON_PORT", str(port))
    monkeypatch.setenv("SLM_TEST_ALLOW_DAEMON_SPAWN", "1")
    monkeypatch.setenv("SLM_INSTANCE_LOCK_WAIT_S", "5")
    started: list[int] = []
    yield tmp_path, port, started
    for pid in started:
        _stop(pid)


def test_auto_started_daemon_never_inherits_the_clients_identity(namespace, monkeypatch):
    data_dir, port, started = namespace
    for key, value in CLIENT_ENV.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("SLM_RANKING", "operator-choice")
    from superlocalmemory.cli import daemon

    assert daemon.ensure_daemon(port=port) is True
    record = _wait_ready(data_dir)
    started.append(record["pid"])
    environment = psutil.Process(record["pid"]).environ()
    leaked = sorted(set(CLIENT_ENV) & set(environment))
    assert leaked == [], f"client-scoped variables reached the daemon: {leaked}"
    assert environment.get("SLM_RANKING") == "operator-choice"


_SERVICE_DAEMON = textwrap.dedent("""
    import importlib, os, sys
    from pathlib import Path
    from superlocalmemory.cli import daemon_supervisor as supervisor
    port = int(sys.argv[1])
    job = supervisor.SupervisorJob("macos", "com.example.slm.e2e", Path("/x"), port,
                                   {"SLM_RANKING": "v2-ensemble"})
    supervisor.jobs_for = lambda p: [job] if p == port else []
    supervisor.job_pid = lambda j: os.getpid()
    daemon = importlib.import_module("superlocalmemory.server.unified_daemon")
    daemon._cli_main(["unified_daemon", "--start", f"--port={port}"])
""")


def test_service_daemon_takes_the_folder_from_an_unowned_daemon(namespace):
    data_dir, port, started = namespace
    base = {k: v for k, v in os.environ.items() if k not in CLIENT_ENV}
    # The unowned daemon, started the way an older client did: detached, with
    # the client's environment.
    unowned = subprocess.Popen(
        [sys.executable, "-m", "superlocalmemory.server.unified_daemon",
         "--start", f"--port={port}"],
        env={**base, **CLIENT_ENV}, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    started.append(unowned.pid)
    _wait_ready(data_dir, unowned.pid)

    service = subprocess.Popen(
        [sys.executable, "-c", _SERVICE_DAEMON, str(port)],
        env={**base, "SLM_RANKING": "v2-ensemble"},
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True,
    )
    started.append(service.pid)

    record = _wait_ready(data_dir, service.pid)
    assert unowned.wait(timeout=60) is not None, "the unowned daemon must hand over"
    assert service.poll() is None, "the service's daemon must keep serving"
    owners = [p.pid for p in psutil.process_iter(["pid"])
              if p.pid in (unowned.pid, service.pid) and p.is_running()
              and p.status() != psutil.STATUS_ZOMBIE]
    assert owners == [service.pid]
    environment = psutil.Process(service.pid).environ()
    assert environment.get("SLM_RANKING") == "v2-ensemble"
    assert "SLM_AGENT_ID" not in environment
    assert record["port"] == port
