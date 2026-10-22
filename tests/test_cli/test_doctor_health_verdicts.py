"""An unreadable /health is a failed daemon; optional features never mask a proven one."""

import json
import threading
import time
from argparse import Namespace
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from superlocalmemory.cli import commands
from superlocalmemory.cli.doctor_daemon import CHECK_TIMEOUT, check_daemon
from superlocalmemory.infra.daemon_identity import build_descriptor, write_descriptor


def _serve(free_port, behavior, started_ago=3600):
    import os

    record = build_descriptor(port=free_port, version="test", pid=os.getpid(),
                              started_at=time.time() - started_ago)
    write_descriptor(record)
    stop = threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            try:
                if behavior == "silent_features" and self.path != "/health":
                    stop.wait(8)  # accepted, never answered: a real socket timeout
                    return
                self.send_response(200)
                self.end_headers()
                if behavior == "stream" or (behavior == "features" and self.path != "/health"):
                    until = time.monotonic() + CHECK_TIMEOUT + 1
                    while time.monotonic() < until and not stop.wait(0.05):
                        self.wfile.write(b" ")
                        self.wfile.flush()
                    return
                if behavior == "large":
                    self.wfile.write(b" " * 100_000)
                    return
                body = (record.public_health_fields() if self.path == "/health"
                        else {"media": {}})
                self.wfile.write(json.dumps(body).encode())
            except (BrokenPipeError, ConnectionResetError):
                pass

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", free_port), Handler)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01})
    thread.start()

    def close():
        stop.set()
        server.shutdown()
        server.server_close()
        thread.join(2)

    return close


@pytest.mark.parametrize("behavior", ["stream", "large"])
def test_unreadable_health_fails_with_exit_one(free_port, capsys, behavior):
    close = _serve(free_port, behavior)
    began = time.monotonic()
    try:
        result = check_daemon()
        assert time.monotonic() - began < CHECK_TIMEOUT + 0.3
        assert result.status == "FAIL"
        assert "health" in result.detail.lower()
        with pytest.raises(SystemExit) as exc:
            commands.cmd_doctor(Namespace(json=True, quick=True))
        assert exc.value.code == 1
    finally:
        close()
    checks = {c["name"]: c for c in json.loads(capsys.readouterr().out)["data"]["checks"]}
    assert checks["Daemon"]["status"] == "FAIL"


def test_feature_timeout_keeps_proven_daemon_pass(free_port, capsys):
    close = _serve(free_port, "features")
    began = time.monotonic()
    try:
        result = check_daemon()
        assert time.monotonic() - began < CHECK_TIMEOUT + 0.3
        assert result.status == "PASS"
        assert result.feature_warning
        try:
            commands.cmd_doctor(Namespace(json=True, quick=True))
        except SystemExit:
            pass
    finally:
        close()
    checks = {c["name"]: c for c in json.loads(capsys.readouterr().out)["data"]["checks"]}
    assert checks["Daemon"]["status"] == "PASS"
    assert checks["Daemon features"]["status"] == "WARN"

def test_whole_doctor_has_one_deadline_naming_unfinished_check(monkeypatch, capsys):
    release = threading.Event()

    def hang():
        release.wait(10)
        return {"enabled": False}

    def checks(_args, state):
        # The real checks before Surface B probe the host's daemon and store and can
        # outlast a 1 s budget on their own, so the deadline would name one of them.
        state["checks"].append({"name": "Python", "status": "PASS", "detail": "ok", "fix": ""})
        state["stage"] = "Optimize (Surface B)"
        commands._gather_optimize_surface_b()

    monkeypatch.setattr(commands, "_gather_optimize_surface_b", hang)
    monkeypatch.setattr(commands, "_cmd_doctor_checks", checks)
    monkeypatch.setattr(commands, "DOCTOR_TIMEOUT", 1.0, raising=False)
    began = time.monotonic()
    try:
        with pytest.raises(SystemExit) as exc:
            commands.cmd_doctor(Namespace(json=True, quick=True))
        assert time.monotonic() - began < 3
        assert exc.value.code == 1
    finally:
        release.set()
    data = json.loads(capsys.readouterr().out)["data"]
    deadline = [c for c in data["checks"] if c["name"] == "Doctor deadline"]
    assert deadline and deadline[0]["status"] == "FAIL"
    assert "Optimize (Surface B)" in deadline[0]["detail"]
    assert data["summary"]["failed"] >= 1


def test_deadline_text_mode_names_check_and_exits_one(monkeypatch, capsys):
    release = threading.Event()
    monkeypatch.setattr(commands, "_gather_optimize_surface_b",
                        lambda: release.wait(10) or {"enabled": False})
    monkeypatch.setattr(commands, "DOCTOR_TIMEOUT", 1.0, raising=False)
    try:
        with pytest.raises(SystemExit) as exc:
            commands.cmd_doctor(Namespace(json=False, quick=True))
    finally:
        release.set()
    out = capsys.readouterr().out
    assert exc.value.code == 1
    assert "Optimize (Surface B)" in out and "did not finish" in out
    assert "Summary:" in out


def test_a_daemon_that_just_started_is_told_to_run_doctor_again_not_restart(free_port):
    from superlocalmemory.cli import doctor_daemon

    close = _serve(free_port, "silent_features", started_ago=5)
    began = time.monotonic()
    try:
        result = check_daemon()
        elapsed = time.monotonic() - began
    finally:
        close()
    assert result.status == "PASS" and result.feature_warning == ""
    assert "still starting" in result.feature_note
    assert CHECK_TIMEOUT < elapsed < CHECK_TIMEOUT + doctor_daemon.WARMUP_TIMEOUT + 1.5
