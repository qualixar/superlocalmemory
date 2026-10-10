# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""The daemon record is written only by the process that owns the data folder.

A per-data-folder instance lock decides who the daemon is. Only its holder may
write ``daemon.json`` and the pid/port mirrors; a starter that lost the race
never touches them. All locks here live under a pytest data folder; real
cross-process contention uses lightweight child interpreters.
"""
from __future__ import annotations

import inspect
import json
import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from superlocalmemory.infra.daemon_identity import (
    build_descriptor,
    descriptor_path,
    read_descriptor,
    write_descriptor,
)
from superlocalmemory.infra.instance_lock import (
    INSTANCE_LOCK_NAME,
    InstanceLock,
    acquire_with_backoff,
    instance_lock_is_held,
    instance_lock_path,
    wait_until_instance_lock_free,
)

posix_only = pytest.mark.skipif(
    sys.platform == "win32", reason="POSIX file modes and kill semantics",
)

_HOLDER = textwrap.dedent("""
    import sys
    from pathlib import Path
    from superlocalmemory.infra.instance_lock import InstanceLock
    lock = InstanceLock(Path(sys.argv[1]))
    print("held" if lock.try_acquire() else "busy", flush=True)
    sys.stdin.readline()
""")

_PROBE = textwrap.dedent("""
    import sys
    from pathlib import Path
    from superlocalmemory.infra.instance_lock import InstanceLock
    lock = InstanceLock(Path(sys.argv[1]))
    print("1" if lock.try_acquire() else "0")
""")

_RACER = textwrap.dedent("""
    import os, sys, time
    from pathlib import Path
    from superlocalmemory.infra.daemon_identity import build_descriptor
    from superlocalmemory.infra.instance_lock import InstanceLock
    from superlocalmemory.infra.daemon_identity import publish_if_owner
    gate, lock_path = Path(sys.argv[1]), Path(sys.argv[2])
    while not gate.exists():
        time.sleep(0.005)
    lock = InstanceLock(lock_path)
    won = lock.try_acquire()
    if won:
        publish_if_owner(
            build_descriptor(port=int(sys.argv[3]), version="t", state="ready"),
            lock,
        )
    print("won" if won else "lost", flush=True)
    sys.stdin.readline()
""")


@pytest.fixture
def root(tmp_path, monkeypatch):
    data = tmp_path / "slm-data"
    monkeypatch.setenv("SLM_DATA_DIR", str(data))
    monkeypatch.setenv("SLM_TEST_ALLOW_DAEMON_SPAWN", "1")
    return data


def _spawn(script: str, *args: str) -> subprocess.Popen:
    return subprocess.Popen(
        [sys.executable, "-c", script, *args],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
        env=dict(os.environ),
    )


def _finish(proc: subprocess.Popen) -> None:
    try:
        if proc.stdin:
            proc.stdin.close()
        proc.wait(timeout=10)
    finally:
        if proc.poll() is None:
            proc.kill()
        if proc.stdout:
            proc.stdout.close()


def _probe(lock_path: Path) -> str:
    out = subprocess.run(
        [sys.executable, "-c", _PROBE, str(lock_path)],
        capture_output=True, text=True, timeout=30, env=dict(os.environ),
    )
    return out.stdout.strip()


def _descriptor(**kw):
    return build_descriptor(port=kw.pop("port", 8765), version="t", **kw)


# 1 -- the lock -------------------------------------------------------------

def test_lock_is_exclusive_across_processes_and_releases(root):
    path = instance_lock_path()
    assert path.name == INSTANCE_LOCK_NAME
    lock = InstanceLock(path)
    assert not path.parent.exists()          # no I/O in __init__
    assert lock.try_acquire() and lock.held
    assert lock.try_acquire()                # idempotent
    assert path.parent.is_dir()
    assert _probe(path) == "0"
    lock.release()
    lock.release()                           # idempotent
    assert not lock.held
    assert path.exists()                     # never deleted
    assert _probe(path) == "1"


def test_lock_context_manager_and_free_probe(root):
    path = instance_lock_path()
    with InstanceLock(path) as lock:
        assert lock.held
        assert instance_lock_is_held()
    assert not instance_lock_is_held()


def test_killed_holder_frees_the_lock(root):
    path = instance_lock_path()
    holder = _spawn(_HOLDER, str(path))
    try:
        assert holder.stdout.readline().strip() == "held"
        assert instance_lock_is_held()
        holder.kill()
        holder.wait(timeout=10)
        assert wait_until_instance_lock_free(timeout_s=5.0)
    finally:
        _finish(holder)


def test_distinct_data_roots_have_independent_locks(tmp_path):
    a = InstanceLock(tmp_path / "a" / INSTANCE_LOCK_NAME)
    b = InstanceLock(tmp_path / "b" / INSTANCE_LOCK_NAME)
    try:
        assert a.try_acquire() and b.try_acquire()
    finally:
        a.release()
        b.release()


# 2 / 3 -- publish and clear only as owner ------------------------------------

def test_publish_without_lock_writes_nothing(root, tmp_path):
    from superlocalmemory.infra.daemon_identity import publish_if_owner

    lock = InstanceLock(tmp_path / "lock")
    assert publish_if_owner(_descriptor(), lock) is False
    assert not descriptor_path().exists()
    assert not descriptor_path().with_name("daemon.pid").exists()
    assert not descriptor_path().with_name("daemon.port").exists()


def test_publish_as_owner_writes_all_three_files_atomically(root, tmp_path):
    from superlocalmemory.infra.daemon_identity import publish_if_owner

    descriptor = _descriptor(port=8123)
    with InstanceLock(tmp_path / "lock") as lock:
        assert publish_if_owner(descriptor, lock) is True
    base = descriptor_path().parent
    assert read_descriptor().instance_id == descriptor.instance_id
    assert (base / "daemon.pid").read_text() == str(descriptor.pid)
    assert (base / "daemon.port").read_text() == "8123"
    assert not list(base.glob("*.tmp"))
    if sys.platform != "win32":
        for name in ("daemon.json", "daemon.pid", "daemon.port"):
            assert (base / name).stat().st_mode & 0o777 == 0o600


def test_clear_is_owner_and_instance_checked(root, tmp_path):
    from superlocalmemory.infra.daemon_identity import (
        clear_descriptor_if_owner,
        publish_if_owner,
    )

    ours = _descriptor()
    base = descriptor_path().parent
    lock = InstanceLock(tmp_path / "lock")
    other = _descriptor(pid=os.getpid())
    write_descriptor(other)
    # not the lock owner: untouched
    assert clear_descriptor_if_owner(other.instance_id, lock) is False
    assert descriptor_path().exists()
    # owner, but a different instance is on disk: untouched
    assert lock.try_acquire()
    try:
        assert clear_descriptor_if_owner(ours.instance_id, lock) is False
        assert descriptor_path().exists()
        # owner and same instance: record and matching mirrors removed
        publish_if_owner(ours, lock)
        assert clear_descriptor_if_owner(ours.instance_id, lock) is True
        assert not descriptor_path().exists()
        assert not (base / "daemon.pid").exists()
        assert not (base / "daemon.port").exists()
        assert clear_descriptor_if_owner(ours.instance_id, lock) is False
    finally:
        lock.release()


# 4 -- real race --------------------------------------------------------------

def test_two_racing_processes_exactly_one_owns_the_record(root, tmp_path):
    gate = tmp_path / "go"
    lock_path = instance_lock_path()
    procs = [_spawn(_RACER, str(gate), str(lock_path), str(9000 + i))
             for i in range(2)]
    try:
        gate.write_text("go")
        verdicts = [p.stdout.readline().strip() for p in procs]
        assert sorted(verdicts) == ["lost", "won"]
        winner = procs[verdicts.index("won")]
        record = read_descriptor()
        assert record is not None and record.pid == winner.pid
    finally:
        for proc in procs:
            _finish(proc)


# 5 / 6 -- start_server -------------------------------------------------------

def test_start_server_refuses_when_another_process_holds_the_lock(
    root, monkeypatch,
):
    from superlocalmemory.server import unified_daemon

    sentinel = _descriptor(pid=os.getpid())
    write_descriptor(sentinel)
    before = descriptor_path().read_bytes()
    holder = _spawn(_HOLDER, str(instance_lock_path()))
    monkeypatch.setenv("SLM_INSTANCE_LOCK_WAIT_S", "0.3")
    binds = MagicMock(side_effect=AssertionError("no socket may be created"))
    monkeypatch.setattr(unified_daemon, "install_thread_dump_signal", lambda: None)
    try:
        assert holder.stdout.readline().strip() == "held"
        with patch("socket.socket", binds):
            unified_daemon.start_server(port=0)
    finally:
        _finish(holder)
    binds.assert_not_called()
    assert descriptor_path().read_bytes() == before
    assert not descriptor_path().with_name("daemon.pid").exists()


def test_start_server_takes_the_lock_before_binding_or_publishing():
    from superlocalmemory.server import unified_daemon

    source = inspect.getsource(unified_daemon.start_server)
    acquire = source.index("acquire_with_backoff")
    assert acquire < source.index("listener.bind")
    assert acquire < source.index("socket.socket(")
    assert acquire < source.index("_publish_process_descriptor")


# 7 -- the incident regression ------------------------------------------------

def test_parent_never_writes_the_record_when_the_child_does_not_win(
    root, monkeypatch,
):
    from superlocalmemory.cli import daemon as daemon_mod

    winner = _descriptor(pid=os.getpid(), state="ready")
    write_descriptor(winner)
    base = descriptor_path().parent
    (base / "daemon.pid").write_text(str(winner.pid))
    (base / "daemon.port").write_text(str(winner.port))
    snapshot = {n: (base / n).read_bytes()
                for n in ("daemon.json", "daemon.pid", "daemon.port")}

    fake = MagicMock()
    fake.pid = 2_000_000
    monkeypatch.setattr(daemon_mod, "is_daemon_running", lambda: False)
    monkeypatch.setattr(daemon_mod, "_has_tcp_listener", lambda port: False)
    monkeypatch.setattr(daemon_mod, "_wait_for_daemon", lambda timeout=60: False)
    with patch("subprocess.Popen", return_value=fake):
        assert daemon_mod._start_daemon_subprocess(port=8765) is False
    for name, data in snapshot.items():
        assert (base / name).read_bytes() == data, name


def test_parent_writes_no_record_files_at_all(root, monkeypatch):
    from superlocalmemory.cli import daemon as daemon_mod

    fake = MagicMock()
    fake.pid = 2_000_001
    monkeypatch.setattr(daemon_mod, "is_daemon_running", lambda: False)
    monkeypatch.setattr(daemon_mod, "_has_tcp_listener", lambda port: False)
    monkeypatch.setattr(daemon_mod, "_wait_for_daemon", lambda timeout=60: False)
    with patch("subprocess.Popen", return_value=fake):
        daemon_mod._start_daemon_subprocess(port=8765)
    base = descriptor_path().parent
    assert not (base / "daemon.json").exists()
    assert not (base / "daemon.pid").exists()
    assert not (base / "daemon.port").exists()


# 10 -- restart waits for the data-folder lock -----------------------------------

_RELEASER = textwrap.dedent("""
    import sys, time
    from pathlib import Path
    from superlocalmemory.infra.instance_lock import InstanceLock
    lock = InstanceLock(Path(sys.argv[1]))
    assert lock.try_acquire()
    print("held", flush=True)
    time.sleep(float(sys.argv[2]))
    lock.release()
    print("released", flush=True)
    sys.stdin.readline()
""")


def test_acquire_backoff_outlasts_a_holder_that_lets_go(root):
    holder = _spawn(_RELEASER, str(instance_lock_path()), "1.5")
    mine = InstanceLock(instance_lock_path())
    try:
        assert holder.stdout.readline().strip() == "held"
        assert not mine.try_acquire()
        assert acquire_with_backoff(mine, 10.0)
    finally:
        mine.release()
        _finish(holder)


def test_acquire_backoff_gives_up_when_the_holder_stays(root):
    holder = _spawn(_HOLDER, str(instance_lock_path()))
    mine = InstanceLock(instance_lock_path())
    try:
        assert holder.stdout.readline().strip() == "held"
        assert not acquire_with_backoff(mine, 0.4)
    finally:
        _finish(holder)


def test_start_server_waits_for_an_old_daemon_to_release(root, monkeypatch):
    from superlocalmemory.infra.instance_lock import get_instance_lock
    from superlocalmemory.server import unified_daemon

    class Reached(Exception):
        pass

    monkeypatch.setattr(unified_daemon, "install_thread_dump_signal", lambda: None)
    monkeypatch.setenv("SLM_INSTANCE_LOCK_WAIT_S", "10")
    holder = _spawn(_RELEASER, str(instance_lock_path()), "1.5")
    try:
        assert holder.stdout.readline().strip() == "held"
        with patch("socket.socket", side_effect=Reached):
            with pytest.raises(Reached):
                unified_daemon.start_server(port=0)
        assert get_instance_lock().held
    finally:
        get_instance_lock().release()
        _finish(holder)


def test_probe_never_makes_the_daemon_give_up(root):
    import threading

    stop = threading.Event()

    def probe_loop():
        while not stop.is_set():
            instance_lock_is_held()

    thread = threading.Thread(target=probe_loop, daemon=True)
    mine = InstanceLock(instance_lock_path())
    thread.start()
    try:
        for _ in range(50):
            assert acquire_with_backoff(mine, 5.0)
            mine.release()
    finally:
        stop.set()
        thread.join(timeout=5)
        mine.release()


def test_ensure_daemon_never_spawns_into_an_owned_folder(root, monkeypatch):
    import threading

    from superlocalmemory.cli import daemon as daemon_mod

    monkeypatch.setenv("SLM_DAEMON_START_WAIT_S", "0.5")
    monkeypatch.setattr(daemon_mod, "_has_tcp_listener", lambda port: False)
    holder = _spawn(_HOLDER, str(instance_lock_path()))
    results: list[bool] = []
    try:
        assert holder.stdout.readline().strip() == "held"
        with patch("subprocess.Popen") as popen:
            threads = [threading.Thread(
                target=lambda: results.append(daemon_mod.ensure_daemon()))
                for _ in range(5)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=30)
        popen.assert_not_called()
    finally:
        _finish(holder)
    assert results == [False] * 5


def test_ensure_daemon_does_not_unlink_the_start_lock(root, monkeypatch):
    from superlocalmemory.cli import daemon as daemon_mod

    monkeypatch.setattr(daemon_mod, "is_daemon_running", lambda: False)
    monkeypatch.setattr(daemon_mod, "_has_tcp_listener", lambda port: False)
    monkeypatch.setattr(daemon_mod, "_start_daemon_subprocess", lambda port=None: True)
    assert daemon_mod.ensure_daemon() is True
    assert daemon_mod._lock_file_path().exists()


def test_cleanup_stops_the_guardian_before_clearing_the_record(root):
    from superlocalmemory.infra.instance_lock import get_instance_lock
    from superlocalmemory.server import unified_daemon

    lock = get_instance_lock()
    ours = _descriptor(state="ready")
    assert lock.try_acquire()
    unified_daemon._ACTIVE_DAEMON_DESCRIPTOR = ours
    try:
        from superlocalmemory.infra.daemon_identity import publish_if_owner
        publish_if_owner(ours, lock)
        unified_daemon._start_record_guardian()
        # a foreign write over the live owner is corrected by the guardian
        write_descriptor(_descriptor(pid=os.getpid(), state="ready"))
        deadline = time.monotonic() + 6.0
        while time.monotonic() < deadline:
            rec = read_descriptor()
            if rec is not None and rec.instance_id == ours.instance_id:
                break
            time.sleep(0.1)
        assert read_descriptor().instance_id == ours.instance_id
        unified_daemon._cleanup_process_descriptor(ours)
        unified_daemon._cleanup_process_descriptor(ours)   # twice is safe
        assert unified_daemon._RECORD_GUARDIAN is None
        assert not descriptor_path().exists()
    finally:
        unified_daemon._ACTIVE_DAEMON_DESCRIPTOR = None
        unified_daemon._stop_record_guardian()
        lock.release()


def test_restart_record_is_written_on_failure_and_success(root):
    from superlocalmemory.cli.commands import _write_restart_record

    logs = descriptor_path().parent / "logs"
    _write_restart_record(1.0, [{"step": 1, "name": "x", "status": "fail", "detail": ""}])
    data = json.loads((logs / "restart-last.json").read_text())
    assert data["success"] is False and data["started_at"] == 1.0
    _write_restart_record(2.0, [{"step": 1, "name": "x", "status": "ok", "detail": ""}])
    data = json.loads((logs / "restart-last.json").read_text())
    assert data["success"] is True and data["finished_at"] >= 2.0
    assert not list(logs.glob("*.tmp"))


def _run_restart(monkeypatch, capsys, started: MagicMock):
    from argparse import Namespace

    from superlocalmemory.cli import commands
    from superlocalmemory.cli import daemon as daemon_mod
    from superlocalmemory.infra import instance_lock as lock_mod

    real_wait = lock_mod.wait_until_instance_lock_free
    monkeypatch.setattr(
        lock_mod, "wait_until_instance_lock_free",
        lambda timeout_s=30.0, **kw: real_wait(timeout_s=0.5, **kw),
    )
    monkeypatch.setattr(daemon_mod, "owned_daemon_process_alive", lambda: False)
    monkeypatch.setattr(daemon_mod, "_start_daemon_subprocess", started)
    commands.cmd_restart(Namespace(json=True, dashboard=False))
    payload = json.loads(capsys.readouterr().out)
    data = payload.get("data", payload)
    return data["steps"]


def test_restart_stops_when_the_previous_daemon_still_holds_the_lock(
    root, monkeypatch, capsys,
):
    started = MagicMock(return_value=False)
    holder = _spawn(_HOLDER, str(instance_lock_path()))
    try:
        assert holder.stdout.readline().strip() == "held"
        steps = _run_restart(monkeypatch, capsys, started)
    finally:
        _finish(holder)
    lock_step = [s for s in steps if "lock" in s["name"].lower()
                 and str(s["step"]).startswith("1")]
    assert lock_step and lock_step[0]["status"] == "fail"
    started.assert_not_called()
    record = json.loads((descriptor_path().parent / "logs" / "restart-last.json").read_text())
    assert record["success"] is False and record["steps"] == steps


def test_restart_proceeds_to_start_when_the_lock_is_free(
    root, monkeypatch, capsys,
):
    started = MagicMock(return_value=False)
    steps = _run_restart(monkeypatch, capsys, started)
    lock_step = [s for s in steps if str(s["step"]) == "1b"]
    assert lock_step and lock_step[0]["status"] == "ok"
    started.assert_called_once()
