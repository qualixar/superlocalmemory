# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""SLM Daemon — client functions for communicating with the unified daemon.

The unified daemon (server/unified_daemon.py) runs as a single FastAPI/uvicorn
process on port 8765, with port 8767 as a backward-compat TCP redirect.

This module contains CLIENT functions used by CLI commands:
  - is_daemon_running(): check if daemon is alive
  - ensure_daemon(): start daemon if not running
  - stop_daemon(): gracefully stop the daemon
  - daemon_request(): send HTTP request to daemon

The actual daemon server code is in server/unified_daemon.py.

Part of Qualixar | Author: Varun Pratap Bhardwaj
License: AGPL-3.0-or-later
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from dataclasses import replace

from superlocalmemory.infra.daemon_identity import (
    build_descriptor,
    descriptor_matches_health,
    descriptor_path,
    process_create_time_for,
    read_descriptor,
    write_descriptor,
)
from superlocalmemory.infra.data_root import (
    assert_no_durable_root_conflict,
    state_path,
)

logger = logging.getLogger(__name__)

try:
    _DEFAULT_PORT = int(os.environ.get("SLM_DAEMON_PORT", "") or 8765)
except ValueError:
    _DEFAULT_PORT = 8765
_LEGACY_PORT = 8767   # backward-compat redirect
_DEFAULT_IDLE_TIMEOUT = 0  # v3.4.3: 24/7 default (was 1800)
_PID_FILE = None  # test-only override; runtime resolution stays dynamic
_PORT_FILE = None  # test-only override; runtime resolution stays dynamic


# ---------------------------------------------------------------------------
# Client: check if daemon running + send requests
# ---------------------------------------------------------------------------

def _is_pid_alive(pid: int) -> bool:
    """Cross-platform check if a process with given PID exists."""
    try:
        import psutil
        return psutil.pid_exists(pid)
    except ImportError:
        try:
            os.kill(pid, 0)
            return True
        except (ProcessLookupError, PermissionError):
            return False


def _descriptor_process_is_alive(descriptor) -> bool:
    """Reject stale descriptors when a PID has been reused by another process."""
    if not _is_pid_alive(descriptor.pid):
        return False
    try:
        import psutil

        actual = float(psutil.Process(descriptor.pid).create_time())
    except ImportError:
        return True
    except Exception:
        return False
    return abs(actual - float(descriptor.process_create_time)) <= 1.0


def is_daemon_running() -> bool:
    """Return True only for a daemon owned by this canonical data namespace.

    A PID or an HTTP 200 proves liveness, not ownership. V3.7 requires the
    private local descriptor and the health endpoint to agree on namespace,
    process instance, capability fingerprint, owner, PID, protocol, and port.
    """
    local_descriptor_path = descriptor_path()
    descriptor = read_descriptor()
    if descriptor is not None:
        if not _descriptor_process_is_alive(descriptor):
            return False
        if descriptor.state == "starting":
            return True
        health = _fetch_health(descriptor.port)
        return health is not None and descriptor_matches_health(descriptor, health)

    # A malformed or foreign descriptor must fail closed; never fall through
    # to legacy PID/port adoption in the same namespace.
    if local_descriptor_path.exists():
        return False

    legacy = _verified_legacy_health()
    return legacy is not None


def _fetch_health(port: int) -> dict | None:
    """Fetch loopback health without following cross-namespace discovery."""
    try:
        import urllib.request

        expected_url = f"http://127.0.0.1:{port}/health"
        response = urllib.request.urlopen(
            expected_url, timeout=2,
        )
        if response.status != 200:
            return None
        geturl = getattr(response, "geturl", None)
        final_url = geturl() if callable(geturl) else None
        if final_url is not None and final_url != expected_url:
            return None
        payload = json.loads(response.read().decode())
        return payload if isinstance(payload, dict) else None
    except Exception:
        return None


def _is_verified_legacy_process(pid: int) -> bool:
    """One-release bridge for a same-root V3.6 unified-daemon process."""
    if not _is_pid_alive(pid):
        return False
    try:
        import psutil

        process = psutil.Process(pid)
        command = " ".join(process.cmdline())
        return "superlocalmemory.server.unified_daemon" in command
    except Exception:
        return False


def _verified_legacy_health() -> dict | None:
    """Accept legacy health only with a verified same-root daemon PID file."""
    pid_file = descriptor_path().with_name("daemon.pid")
    port_file = descriptor_path().with_name("daemon.port")
    try:
        pid = int(pid_file.read_text().strip())
        port = int(port_file.read_text().strip()) if port_file.exists() else _DEFAULT_PORT
    except (OSError, ValueError):
        return None
    if not _is_verified_legacy_process(pid):
        return None
    health = _fetch_health(port)
    if health is None or int(health.get("pid", -1)) != pid:
        return None
    # Identity-bearing health without a descriptor is not legacy and cannot
    # be adopted. It belongs to another namespace or stale state.
    if health.get("daemon_protocol") is not None:
        return None
    return {**health, "_legacy_port": port}


def _get_port() -> int:
    descriptor = read_descriptor()
    if descriptor is not None:
        return descriptor.port
    if descriptor_path().exists():
        return _DEFAULT_PORT
    legacy = _verified_legacy_health()
    if legacy is not None:
        return int(legacy["_legacy_port"])
    return _DEFAULT_PORT


def daemon_request(method: str, path: str, body: dict | None = None) -> dict | None:
    """Send a request only after validating the owned daemon identity."""
    descriptor = read_descriptor()
    capability: str | None = None
    if descriptor is not None:
        health = _fetch_health(descriptor.port)
        if health is None or not descriptor_matches_health(descriptor, health):
            return None
        if method.upper() == "GET" and path == "/health":
            return health
        port = descriptor.port
        capability = descriptor.capability
    elif descriptor_path().exists():
        return None
    else:
        legacy = _verified_legacy_health()
        if legacy is None:
            return None
        if method.upper() == "GET" and path == "/health":
            return {key: value for key, value in legacy.items() if key != "_legacy_port"}
        port = int(legacy["_legacy_port"])
    try:
        import urllib.request
        url = f"http://127.0.0.1:{port}{path}"
        data = json.dumps(body).encode() if body else None
        headers = {"Content-Type": "application/json"} if data else {}
        if capability is not None:
            headers["X-SLM-Daemon-Capability"] = capability
            headers["X-SLM-Target-Instance"] = descriptor.instance_id
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        resp = urllib.request.urlopen(req, timeout=30)
        return json.loads(resp.read().decode())
    except Exception:
        return None


_LOCK_FILE = None  # test-only override; runtime resolution stays dynamic


def _pid_file_path():
    return _PID_FILE or state_path("daemon.pid")


def _port_file_path():
    return _PORT_FILE or state_path("daemon.port")


def _lock_file_path():
    return _LOCK_FILE or state_path("daemon.lock")


def _start_daemon_subprocess() -> bool:
    """Spawn the unified daemon subprocess and wait for readiness.

    v3.4.42: Extracted from ensure_daemon() so callers that already hold
    daemon.lock (e.g. cmd_restart Step 2) can start the daemon WITHOUT
    triggering a second flock acquisition. BSD-style flock blocks per-fd
    even within the same process, so the previous code path produced a
    self-deadlock when called from Step 3 of `slm restart`: the lock held
    by Step 2 caused ensure_daemon's own flock to fail with EWOULDBLOCK,
    falling into the wait-for-someone-else branch and timing out at 60s
    even though the daemon would have started cleanly.

    PRECONDITION: caller has either acquired daemon.lock OR is certain no
    other CLI/MCP process is racing to start a daemon (e.g. we just killed
    everything in `slm restart` Step 1).

    Returns True if daemon is reachable on the health endpoint within
    60 seconds, False otherwise.
    """
    if is_daemon_running():
        return True
    assert_no_durable_root_conflict()

    import subprocess

    from superlocalmemory import __version__ as _slm_version
    # v3.6.9 (#33): pass SLM_DAEMON_PORT as explicit --port= so the daemon
    # binds the right port even when the env var reaches the subprocess.
    _target_port = _DEFAULT_PORT
    cmd = [
        sys.executable, "-m", "superlocalmemory.server.unified_daemon",
        "--start", f"--port={_target_port}",
    ]
    log_dir = state_path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "daemon.log"

    kwargs: dict = {}
    if sys.platform == "win32":
        kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
    else:
        kwargs["start_new_session"] = True

    # v3.4.60: Force OMP_NUM_THREADS=1 in daemon env BEFORE Python imports
    # numpy/torch/lightgbm. Setting it in __init__.py is too late on M5 Pro —
    # by the time superlocalmemory.__init__ runs, libomp has already been
    # initialized by an earlier import, causing the SIGSEGV at
    # __kmp_suspend_initialize_thread when lightgbm forks its worker pool.
    # Forcing serial OpenMP eliminates the parallel barrier race entirely.
    daemon_env = os.environ.copy()
    daemon_env["OMP_NUM_THREADS"] = "1"
    daemon_env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    bootstrap_descriptor = build_descriptor(
        port=_target_port,
        version=_slm_version,
        pid=os.getpid(),
        state="starting",
    )
    daemon_env["SLM_DAEMON_INSTANCE_ID"] = bootstrap_descriptor.instance_id
    daemon_env["SLM_DAEMON_CAPABILITY"] = bootstrap_descriptor.capability
    kwargs["env"] = daemon_env

    with open(log_file, "a") as lf:
        proc = subprocess.Popen(cmd, stdout=lf, stderr=lf, **kwargs)

    # Publish the exact child identity immediately so concurrent callers know
    # this namespace is warming up. If the child won the race and already
    # published the same instance as ready, never overwrite it with starting.
    child_descriptor = replace(
        bootstrap_descriptor,
        pid=proc.pid,
        process_create_time=process_create_time_for(proc.pid),
    )
    current = read_descriptor()
    if not (
        current is not None
        and current.instance_id == child_descriptor.instance_id
        and current.pid == child_descriptor.pid
        and current.state == "ready"
    ):
        write_descriptor(child_descriptor)

    # One-release compatibility mirrors; never sufficient for ownership.
    _pid_file_path().write_text(str(proc.pid))
    _port_file_path().write_text(str(_target_port))

    return _wait_for_daemon(timeout=60)


def ensure_daemon() -> bool:
    """Start daemon if not running. Returns True if daemon is ready.

    v3.4.4 BULLETPROOF:
      1. If PID alive → return True immediately (even if warming up)
      2. File lock prevents two callers from starting concurrent daemons
      3. After starting, waits for PID file (not health check) — fast detection
      4. Cross-platform: macOS + Windows + Linux

    v3.4.42: Refactored to delegate the actual subprocess start to
    `_start_daemon_subprocess()`. Callers that already hold daemon.lock
    (e.g. `slm restart` Step 3) should call that helper directly to avoid
    the same-process flock self-deadlock that returned a false-negative
    "failed to start" while the daemon was actually starting cleanly.
    """
    if is_daemon_running():
        return True
    if (
        os.environ.get("SLM_TEST_ISOLATION") == "1"
        and os.environ.get("SLM_TEST_ALLOW_DAEMON_SPAWN") != "1"
    ):
        logger.debug(
            "pytest isolation blocked daemon spawn; use an owned daemon fixture",
        )
        return False

    # File lock — prevent concurrent starts from multiple CLI/MCP calls
    lock_fd = None
    try:
        lock_file = _lock_file_path()
        lock_file.parent.mkdir(parents=True, exist_ok=True)
        lock_fd = open(lock_file, "w")

        # Cross-platform file locking
        if sys.platform == "win32":
            import msvcrt
            try:
                msvcrt.locking(lock_fd.fileno(), msvcrt.LK_NBLCK, 1)
            except (IOError, OSError):
                # Another process is starting the daemon — just wait for it
                lock_fd.close()
                return _wait_for_daemon(timeout=60)
        else:
            import fcntl
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except (IOError, OSError):
                lock_fd.close()
                return _wait_for_daemon(timeout=60)

        # Re-check after acquiring lock (another process may have started it)
        if is_daemon_running():
            return True

        # v3.6.9 (#36): TCP-level check catches a systemd-started daemon that
        # has bound the port but hasn't written a PID file yet (e.g. different
        # HOME for the service user vs. the SSH user).  If the port is already
        # bound, don't start a second daemon — wait for HTTP readiness instead.
        try:
            import socket as _socket
            with _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM) as _s:
                _s.settimeout(1)
                if _s.connect_ex(("127.0.0.1", _DEFAULT_PORT)) == 0:
                    return _wait_for_daemon(timeout=30)
        except Exception:
            pass

        # Start unified daemon in background — delegated to helper so the
        # same logic can be reused by callers that already hold the lock.
        return _start_daemon_subprocess()

    except Exception as exc:
        # Daemon auto-start is the entry point for dashboard / mesh /
        # health features; failure here silently disables all of them.
        # Log at WARNING so operators can see it in production logs.
        logger.warning("ensure_daemon error: %s (run `slm doctor`)", exc)
        return False
    finally:
        if lock_fd:
            try:
                lock_fd.close()
            except Exception:
                pass
            try:
                _lock_file_path().unlink(missing_ok=True)
            except Exception:
                pass


def _wait_for_daemon(timeout: int = 60) -> bool:
    """Wait for matching owned health; liveness alone is never readiness."""
    for _ in range(timeout * 2):  # check every 0.5s
        time.sleep(0.5)
        descriptor = read_descriptor()
        if descriptor is not None:
            if not _descriptor_process_is_alive(descriptor):
                continue
            health = _fetch_health(descriptor.port)
            if health is not None and descriptor_matches_health(descriptor, health):
                return True
            continue
        if descriptor_path().exists():
            continue
        if _verified_legacy_health() is not None:
            return True
    return False


def stop_daemon() -> bool:
    """Stop only the daemon proven to belong to this data namespace.

    Machine-wide process-name scans are forbidden: they can kill another SLM
    installation or a user's live workers during tests. V3.7 uses the owned
    HTTP capability; the daemon itself terminates its child process tree.
    """
    if read_descriptor() is None and _verified_legacy_health() is None:
        return False
    response = daemon_request("POST", "/stop")
    return bool(response and response.get("status") == "stopping")
