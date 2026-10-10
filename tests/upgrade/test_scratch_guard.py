# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Fast tests for the rules that keep a fixture run away from other SLM daemons."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _slm_env as env  # noqa: E402

DAEMON_ARGV = [sys.executable, "-c", "import time; time.sleep(120)", "superlocalmemory.server.unified_daemon"]


@pytest.fixture
def scratch(tmp_path):
    home = tmp_path / "home"
    data = home / env.DATA_SUBDIR
    data.mkdir(parents=True)
    return home, data


def spawn(home: Path, argv=None) -> subprocess.Popen:
    e = {**os.environ, "HOME": str(home)}
    p = subprocess.Popen(argv or DAEMON_ARGV, env=e, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(0.3)
    return p


@pytest.fixture
def procs():
    made: list[subprocess.Popen] = []
    yield made
    for p in made:
        p.kill()
        p.wait()


def make_instance(scratch, version="3.4.21", port=8851) -> env.Instance:
    home, data = scratch
    return env.Instance(bin_dir=Path("/nonexistent"), data_dir=data, home=home, port=port, version=version)


def test_instance_refuses_the_live_daemon_ports(scratch):
    for port in (8765, 8767):
        with pytest.raises(ValueError):
            make_instance(scratch, port=port)


def test_free_port_stays_in_range_and_skips_listeners():
    seen = {env.free_port() for _ in range(30)}
    assert all(env.PORT_LO <= p <= env.PORT_HI for p in seen)
    assert not seen & env.PROTECTED_PORTS
    with socket.socket() as srv:
        srv.bind(("127.0.0.1", 0))
        srv.listen()
        busy = srv.getsockname()[1]
        with pytest.raises(RuntimeError):
            env.free_port(busy, busy)


def test_legacy_start_refuses_when_default_port_is_taken(scratch, monkeypatch):
    with socket.socket() as srv:
        srv.bind(("127.0.0.1", 0))
        srv.listen()
        monkeypatch.setattr(env, "PROTECTED_PORTS", frozenset({srv.getsockname()[1]}))
        inst = make_instance(scratch, version="3.4.21")
        monkeypatch.setattr(inst, "run", lambda *a, **k: pytest.fail("slm must not run"))
        with pytest.raises(RuntimeError, match="already listening"):
            inst.serve_start(wait=1)


def test_legacy_stop_never_calls_slm_serve_stop(scratch, procs, monkeypatch):
    home, data = scratch
    mine = spawn(home)
    procs.append(mine)
    (data / "daemon.pid").write_text(str(mine.pid))
    inst = make_instance(scratch, version="3.4.21")
    calls = []
    monkeypatch.setattr(inst, "run", lambda *a, **k: calls.append(a) or (0, "", "", 0.0))
    assert inst.serve_stop() is True
    assert mine.wait(timeout=5) is not None
    assert not any(a[:2] == ("serve", "stop") for a in calls)


def test_stop_leaves_a_foreign_process_alone(scratch, procs, tmp_path):
    other_home = tmp_path / "other"
    other_home.mkdir()
    foreign = spawn(other_home)
    procs.append(foreign)
    (scratch[1] / "daemon.pid").write_text(str(foreign.pid))
    inst = make_instance(scratch)
    assert inst.stop_scratch() is True  # nothing of ours is running
    assert foreign.poll() is None


def test_stop_leaves_a_non_slm_process_with_our_home_alone(scratch, procs):
    other = spawn(scratch[0], argv=[sys.executable, "-c", "import time; time.sleep(120)"])
    procs.append(other)
    (scratch[1] / "daemon.pid").write_text(str(other.pid))
    assert make_instance(scratch).stop_scratch() is True
    assert other.poll() is None


def test_stop_survives_garbage_and_missing_pid_files(scratch):
    inst = make_instance(scratch)
    assert inst.scratch_pid() is None
    (scratch[1] / "daemon.pid").write_text("not a number")
    assert inst.scratch_pid() is None
    (scratch[1] / "daemon.pid").write_text("999999999")
    assert inst.stop_scratch() is True


def test_nonlegacy_stop_uses_the_cli_then_falls_back_to_the_scratch_pid(scratch, procs, monkeypatch):
    home, data = scratch
    mine = spawn(home)
    procs.append(mine)
    (data / "daemon.pid").write_text(str(mine.pid))
    inst = make_instance(scratch, version="4.1.24")
    inst.stop_wait = 1
    calls = []
    monkeypatch.setattr(inst, "run", lambda *a, **k: calls.append(a) or (0, "", "", 0.0))
    assert inst.serve_stop() is True
    assert ("serve", "stop") in calls
    assert mine.wait(timeout=5) is not None


class _Health(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        body = json.dumps({"status": "ok"}).encode()
        self.send_response(200)
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


def test_health_only_trusts_the_scratch_daemon(scratch, procs):
    home, data = scratch
    srv = HTTPServer(("127.0.0.1", 0), _Health)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        port = srv.server_address[1]
        (data / "daemon.port").write_text(str(port))
        inst = make_instance(scratch, version="4.1.24")
        assert inst.health() is None  # something answers, but no scratch pid file
        mine = spawn(home)
        procs.append(mine)
        (data / "daemon.pid").write_text(str(mine.pid))
        assert inst.health() == {"status": "ok"}
        mine.kill()
        mine.wait()
        assert inst.health() is None  # pid gone
    finally:
        srv.shutdown()


def test_session_stops_the_daemon_when_the_body_raises(scratch, monkeypatch):
    inst = make_instance(scratch, version="4.1.24")
    events = []
    monkeypatch.setattr(inst, "serve_start", lambda wait=180: events.append("start") or (True, 0.1, ""))
    monkeypatch.setattr(inst, "serve_stop", lambda: events.append("stop") or True)
    with pytest.raises(KeyError):
        with inst.session():
            raise KeyError("boom")
    assert events == ["start", "stop"]


def test_session_raises_when_the_daemon_will_not_exit(scratch, monkeypatch):
    inst = make_instance(scratch, version="4.1.24")
    monkeypatch.setattr(inst, "serve_start", lambda wait=180: (True, 0.1, ""))
    monkeypatch.setattr(inst, "serve_stop", lambda: False)
    with pytest.raises(RuntimeError, match="still running"):
        with inst.session():
            pass


def test_require_version_checks_the_installed_package():
    have = env.package_version(Path(sys.executable), "pytest")
    assert have
    env.require_version(Path(sys.executable), have, package="pytest")
    with pytest.raises(RuntimeError, match="expected"):
        env.require_version(Path(sys.executable), "0.0.0", package="pytest")
