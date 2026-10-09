# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""The record guardian restores the daemon record; clients fall back on health.

The guardian runs only inside the process that holds the instance lock. The
client fallback never writes the record: it waits for the daemon to republish.
"""
from __future__ import annotations

import json
import logging
import subprocess
import sys
import threading
import time
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from superlocalmemory.infra.daemon_identity import (
    build_descriptor,
    descriptor_path,
    publish_if_owner,
    read_descriptor,
    write_descriptor,
)
from superlocalmemory.infra.instance_lock import InstanceLock


@pytest.fixture
def root(tmp_path, monkeypatch):
    data = tmp_path / "slm-data"
    monkeypatch.setenv("SLM_DATA_DIR", str(data))
    monkeypatch.setenv("SLM_TEST_ALLOW_DAEMON_SPAWN", "1")
    return data


@pytest.fixture
def lock(tmp_path):
    held = InstanceLock(tmp_path / "locks" / "daemon.instance.lock")
    yield held
    held.release()


def _dead_pid() -> int:
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    return proc.pid


def _ours(port: int = 8765):
    return build_descriptor(port=port, version="t", state="ready")


def _guardian(ours, lock, interval_s: float = 5.0):
    from superlocalmemory.daemon.record_guardian import RecordGuardian

    return RecordGuardian(
        descriptor_provider=lambda: ours, lock=lock, interval_s=interval_s,
    )


# 8 -- guardian --------------------------------------------------------------

def test_not_owner_never_touches_disk(root, lock):
    assert _guardian(_ours(), lock).check_once() == "not_owner"
    assert not descriptor_path().exists()


def test_no_descriptor_yet(root, lock):
    from superlocalmemory.daemon.record_guardian import RecordGuardian

    assert lock.try_acquire()
    guardian = RecordGuardian(descriptor_provider=lambda: None, lock=lock)
    assert guardian.check_once() == "no_descriptor"


def test_matching_record_is_ok_and_deleted_record_is_republished(root, lock):
    ours = _ours()
    assert lock.try_acquire()
    publish_if_owner(ours, lock)
    guardian = _guardian(ours, lock)
    assert guardian.check_once() == "ok"
    descriptor_path().unlink()
    assert guardian.check_once() == "republished"
    assert read_descriptor().instance_id == ours.instance_id
    assert guardian.check_once() == "ok"


def test_mirror_drift_is_repaired(root, lock):
    ours = _ours()
    assert lock.try_acquire()
    publish_if_owner(ours, lock)
    descriptor_path().with_name("daemon.pid").write_text("1")
    assert _guardian(ours, lock).check_once() == "republished"
    assert descriptor_path().with_name("daemon.pid").read_text() == str(ours.pid)


def test_foreign_record_with_dead_pid_is_replaced(root, lock):
    ours = _ours()
    write_descriptor(build_descriptor(
        port=9999, version="t", pid=_dead_pid(), state="ready"))
    assert lock.try_acquire()
    assert _guardian(ours, lock).check_once() == "republished"
    assert read_descriptor().instance_id == ours.instance_id


def test_foreign_record_with_live_pid_is_replaced_with_a_warning(
    root, lock, caplog,
):
    ours = _ours()
    sleeper = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        write_descriptor(build_descriptor(
            port=9999, version="t", pid=sleeper.pid, state="ready"))
        assert lock.try_acquire()
        with caplog.at_level(logging.WARNING):
            assert _guardian(ours, lock).check_once() == "republished"
    finally:
        sleeper.kill()
        sleeper.wait()
    assert read_descriptor().instance_id == ours.instance_id
    assert any(r.levelno == logging.WARNING for r in caplog.records)


def test_guardian_thread_restores_a_deleted_record_and_stops(root, lock):
    ours = _ours()
    assert lock.try_acquire()
    publish_if_owner(ours, lock)
    guardian = _guardian(ours, lock, interval_s=0.2)
    guardian.start()
    guardian.start()                       # idempotent
    try:
        assert guardian.health()["state"] == "running"
        descriptor_path().unlink()
        deadline = time.monotonic() + 6.0
        while time.monotonic() < deadline and not descriptor_path().exists():
            time.sleep(0.05)
        assert read_descriptor().instance_id == ours.instance_id
    finally:
        guardian.stop()
    assert guardian.health()["state"] == "stopped"
    assert not [t for t in threading.enumerate()
                if t.name == "slm-record-guardian" and t.is_alive()]


# 9 -- client fallback -------------------------------------------------------

class _Health(BaseHTTPRequestHandler):
    payload: dict = {}

    def do_GET(self):  # noqa: N802 - stdlib handler naming
        body = json.dumps(type(self).payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        pass


@pytest.fixture
def health_server():
    started = []

    def _start(payload: dict) -> int:
        handler = type("H", (_Health,), {"payload": payload})
        server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        server.daemon_threads = True
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        started.append((server, thread))
        return int(server.server_address[1])

    yield _start
    for server, thread in started:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _republish_later(descriptor, delay: float = 0.5) -> threading.Thread:
    thread = threading.Thread(
        target=lambda: (time.sleep(delay), write_descriptor(descriptor)),
        daemon=True,
    )
    thread.start()
    return thread


def _serve_same_account(health_server):
    probe = build_descriptor(port=1, version="t", state="ready")
    handler_payload: dict = {}
    port = health_server(handler_payload)
    live = replace(probe, port=port)
    handler_payload.update(live.public_health_fields())
    return live, handler_payload


def test_dead_record_with_same_account_health_waits_for_republish(
    root, health_server,
):
    from superlocalmemory.cli import daemon as daemon_mod

    live, _ = _serve_same_account(health_server)
    write_descriptor(replace(live, pid=_dead_pid(), instance_id="stale"))
    writer = _republish_later(live)
    assert daemon_mod.is_daemon_running() is True
    writer.join(timeout=5)


def test_foreign_account_health_fails_without_waiting(root, health_server):
    from superlocalmemory.cli import daemon as daemon_mod

    live, payload = _serve_same_account(health_server)
    payload["owner_id"] = "uid:someone-else"
    write_descriptor(replace(live, pid=_dead_pid(), instance_id="stale"))
    started = time.monotonic()
    assert daemon_mod.is_daemon_running() is False
    assert time.monotonic() - started < 1.0


def test_dead_record_without_any_health_fails_immediately(root):
    from superlocalmemory.cli import daemon as daemon_mod

    write_descriptor(build_descriptor(
        port=1, version="t", pid=_dead_pid(), state="ready"))
    started = time.monotonic()
    assert daemon_mod.is_daemon_running() is False
    assert time.monotonic() - started < 1.0


def test_malformed_record_with_health_waits_for_republish(
    root, health_server, monkeypatch,
):
    from superlocalmemory.cli import daemon as daemon_mod

    live, _ = _serve_same_account(health_server)
    descriptor_path().parent.mkdir(parents=True, exist_ok=True)
    descriptor_path().write_text("{not json")
    monkeypatch.setattr(daemon_mod, "_get_port", lambda: live.port)
    writer = _republish_later(live)
    assert daemon_mod.is_daemon_running() is True
    writer.join(timeout=5)


def test_owned_process_alive_follows_the_same_fallback(root, health_server):
    from superlocalmemory.cli import daemon as daemon_mod

    live, _ = _serve_same_account(health_server)
    write_descriptor(replace(live, pid=_dead_pid(), instance_id="stale"))
    writer = _republish_later(live)
    assert daemon_mod.owned_daemon_process_alive() is True
    writer.join(timeout=5)


# -- hardening -----------------------------------------------------------------

def test_health_that_proves_the_record_needs_no_wait(root, health_server):
    from superlocalmemory.cli import daemon as daemon_mod

    live, payload = _serve_same_account(health_server)
    # the process is invisible to this client (dead pid here), health still
    # names exactly the recorded instance
    dead = replace(live, pid=_dead_pid())
    payload.update(dead.public_health_fields())
    write_descriptor(dead)
    started = time.monotonic()
    assert daemon_mod._health_fallback(dead.port) is True
    assert time.monotonic() - started < 1.0


def test_missing_record_with_held_lock_waits_for_republish(root, health_server):
    from superlocalmemory.cli import daemon as daemon_mod
    from superlocalmemory.infra.instance_lock import instance_lock_path

    live, _ = _serve_same_account(health_server)

    held = InstanceLock(instance_lock_path())
    assert held.try_acquire()
    try:
        writer = _republish_later(live)
        assert daemon_mod.owned_daemon_process_alive() is True
        descriptor_path().unlink()
        writer = _republish_later(live)
        assert daemon_mod.is_daemon_running() is True
        writer.join(timeout=5)
    finally:
        held.release()


def test_lock_file_replaced_means_lock_lost(root, lock, caplog):
    ours = _ours()
    assert lock.try_acquire() and lock.still_owns_file()
    publish_if_owner(ours, lock)
    lock.path.unlink()
    assert lock.still_owns_file() is False
    descriptor_path().unlink()
    guardian = _guardian(ours, lock)
    with caplog.at_level(logging.ERROR):
        assert guardian.check_once() == "lock_lost"
        assert guardian.check_once() == "lock_lost"
    assert not descriptor_path().exists()
    assert publish_if_owner(ours, lock) is False
    assert len([r for r in caplog.records if r.levelno == logging.ERROR]) == 1


def test_state_drift_is_repaired_with_the_current_descriptor(root, lock):
    starting = replace(_ours(), state="starting")
    ready = replace(starting, state="ready")
    assert lock.try_acquire()
    publish_if_owner(starting, lock)
    from superlocalmemory.daemon.record_guardian import RecordGuardian

    guardian = RecordGuardian(descriptor_provider=lambda: ready, lock=lock)
    assert guardian.check_once() == "republished"
    assert read_descriptor().state == "ready"
    calls = iter([starting, ready])
    stale = RecordGuardian(
        descriptor_provider=lambda: next(calls, ready), lock=lock,
    )
    descriptor_path().unlink()
    assert stale.check_once() == "republished"
    assert read_descriptor().state == "ready"


def test_mirror_failure_does_not_fail_the_publish(root, lock, caplog):
    from unittest.mock import patch

    assert lock.try_acquire()
    with patch(
        "superlocalmemory.infra.daemon_identity.write_mirror_atomic",
        side_effect=OSError("held open"),
    ), caplog.at_level(logging.WARNING):
        assert publish_if_owner(_ours(), lock) is True
    assert read_descriptor() is not None
    assert any("mirror" in r.getMessage() for r in caplog.records)


def test_restart_record_temp_never_survives_a_failure(root):
    from unittest.mock import patch

    from superlocalmemory.cli.commands import _write_restart_record

    logs = descriptor_path().parent / "logs"
    with patch("os.replace", side_effect=OSError("nope")):
        _write_restart_record(1.0, [])
    assert not list(logs.glob("*.tmp"))


def test_malformed_record_uses_the_port_mirror(root, health_server):
    from superlocalmemory.cli import daemon as daemon_mod

    live, _ = _serve_same_account(health_server)
    descriptor_path().parent.mkdir(parents=True, exist_ok=True)
    descriptor_path().write_text("{not json")
    descriptor_path().with_name("daemon.port").write_text(str(live.port))
    writer = _republish_later(live)
    assert daemon_mod.is_daemon_running() is True
    writer.join(timeout=5)


def test_ensure_daemon_starts_when_ownership_ends_during_the_wait(
    root, monkeypatch,
):
    from superlocalmemory.cli import daemon as daemon_mod
    from superlocalmemory.infra.instance_lock import instance_lock_path

    held = InstanceLock(instance_lock_path())
    assert held.try_acquire()
    monkeypatch.setenv("SLM_DAEMON_START_WAIT_S", "5")
    monkeypatch.setattr(daemon_mod, "is_daemon_running", lambda: False)
    monkeypatch.setattr(daemon_mod, "_has_tcp_listener", lambda port: False)
    started = []
    monkeypatch.setattr(
        daemon_mod, "_start_daemon_subprocess",
        lambda port=None: started.append(port) or True,
    )
    releaser = threading.Timer(0.6, held.release)
    releaser.start()
    try:
        assert daemon_mod.ensure_daemon() is True
    finally:
        releaser.join()
        held.release()
    assert started


def test_start_server_failure_after_publish_still_cleans_up(root, monkeypatch):
    from unittest.mock import patch

    from superlocalmemory.infra.instance_lock import get_instance_lock
    from superlocalmemory.server import unified_daemon

    monkeypatch.setattr(unified_daemon, "install_thread_dump_signal", lambda: None)
    monkeypatch.setattr(unified_daemon, "_cleanup_process_descriptor", lambda d: None)
    monkeypatch.setattr(
        unified_daemon, "_serve_owned",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    with pytest.raises(RuntimeError):
        unified_daemon.start_server(port=0)
    assert unified_daemon._RECORD_GUARDIAN is None
    assert not get_instance_lock().held
    assert not [t for t in threading.enumerate()
                if t.name == "slm-record-guardian" and t.is_alive()]
