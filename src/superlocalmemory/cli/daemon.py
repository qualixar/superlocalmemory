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
import signal
import sys
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
import threading
from threading import Thread

from ._lazy_init import slm_home

logger = logging.getLogger(__name__)

try:
    _DEFAULT_PORT = int(os.environ.get("SLM_DAEMON_PORT", "") or 8765)
except ValueError:
    _DEFAULT_PORT = 8765
_LEGACY_PORT = 8767   # backward-compat redirect
_DEFAULT_IDLE_TIMEOUT = 0  # v3.4.3: 24/7 default (was 1800)
# Resolved via slm_home() so the client finds the daemon started by
# server/unified_daemon.py when SLM_DATA_DIR/SLM_HOME point off-home.
_PID_FILE = slm_home() / "daemon.pid"
_PORT_FILE = slm_home() / "daemon.port"


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


def is_daemon_running() -> bool:
    """Check if daemon is alive via PID file + HTTP health check.

    v3.4.4 FIX: If PID is alive, returns True EVEN IF health check fails.
    This prevents starting duplicate daemons when the existing one is
    warming up (Ollama processing, model download, embedding init).

    Priority:
      1. PID file exists AND process alive → True (daemon warming up or ready)
      2. No PID file → try health check on known ports (MCP/hook started daemon)
      3. PID file stale (process dead) → clean up, return False
    """
    if _PID_FILE.exists():
        try:
            pid = int(_PID_FILE.read_text().strip())
            if _is_pid_alive(pid):
                # PID alive = daemon exists. Don't check health — it might be warming up.
                # This is the critical fix: NEVER start a second daemon if PID is alive.
                return True
            else:
                # Process died — clean up stale PID file
                _PID_FILE.unlink(missing_ok=True)
                _PORT_FILE.unlink(missing_ok=True)
        except (ValueError, OSError):
            _PID_FILE.unlink(missing_ok=True)

    # No PID file — maybe daemon was started by MCP/hook without PID file.
    # Try health check on known ports as last resort.
    for try_port in (_DEFAULT_PORT, _LEGACY_PORT):
        try:
            import urllib.request
            resp = urllib.request.urlopen(
                f"http://127.0.0.1:{try_port}/health", timeout=2,
            )
            if resp.status == 200:
                # Daemon running without PID file — write one for future checks
                try:
                    import json as _json
                    data = _json.loads(resp.read().decode())
                    pid = data.get("pid")
                    if pid:
                        _PID_FILE.parent.mkdir(parents=True, exist_ok=True)
                        _PID_FILE.write_text(str(pid))
                        _PORT_FILE.write_text(str(try_port))
                except Exception:
                    pass
                return True
        except Exception:
            continue
    return False


def _get_port() -> int:
    if _PORT_FILE.exists():
        try:
            return int(_PORT_FILE.read_text().strip())
        except ValueError:
            pass
    return _DEFAULT_PORT


def daemon_request(method: str, path: str, body: dict | None = None) -> dict | None:
    """Send request to daemon. Returns parsed JSON or None on failure."""
    port = _get_port()
    try:
        import urllib.request
        url = f"http://127.0.0.1:{port}{path}"
        data = json.dumps(body).encode() if body else None
        headers = {"Content-Type": "application/json"} if data else {}
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        resp = urllib.request.urlopen(req, timeout=30)
        return json.loads(resp.read().decode())
    except Exception:
        return None


_LOCK_FILE = Path.home() / ".superlocalmemory" / "daemon.lock"


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

    import subprocess
    # v3.6.9 (#33): pass SLM_DAEMON_PORT as explicit --port= so the daemon
    # binds the right port even when the env var reaches the subprocess.
    _target_port = _DEFAULT_PORT
    cmd = [
        sys.executable, "-m", "superlocalmemory.server.unified_daemon",
        "--start", f"--port={_target_port}",
    ]
    log_dir = Path.home() / ".superlocalmemory" / "logs"
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
    kwargs["env"] = daemon_env

    with open(log_file, "a") as lf:
        proc = subprocess.Popen(cmd, stdout=lf, stderr=lf, **kwargs)

    # Write PID immediately so other callers see it during warmup
    _PID_FILE.write_text(str(proc.pid))
    _PORT_FILE.write_text(str(_target_port))

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

    # File lock — prevent concurrent starts from multiple CLI/MCP calls
    lock_fd = None
    try:
        _LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
        lock_fd = open(_LOCK_FILE, "w")

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
                _LOCK_FILE.unlink(missing_ok=True)
            except Exception:
                pass


def _wait_for_daemon(timeout: int = 60) -> bool:
    """Wait for daemon to become reachable. Checks PID alive first (fast),
    then health endpoint (confirms HTTP server is bound)."""
    for _ in range(timeout * 2):  # check every 0.5s
        time.sleep(0.5)
        if is_daemon_running():
            # PID is alive — now optionally check if HTTP is ready
            port = _get_port()
            try:
                import urllib.request
                urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2)
                return True  # HTTP is ready
            except Exception:
                # PID alive but HTTP not ready — daemon is warming up, that's OK
                return True
    return False


def stop_daemon() -> bool:
    """Stop ALL SLM daemon processes and their workers.

    v3.4.7: Nuclear cleanup — finds and kills ALL processes matching
    superlocalmemory.server.unified_daemon, embedding_worker, recall_worker,
    reranker_worker. Not just the PID file daemon. Multiple daemons can
    accumulate from rapid restarts, MCP warmups, and concurrent sessions.
    """
    killed = 0

    try:
        import psutil
        my_pid = os.getpid()
        targets = [
            "superlocalmemory.server.unified_daemon",
            "superlocalmemory.core.embedding_worker",
            "superlocalmemory.core.recall_worker",
            "superlocalmemory.core.reranker_worker",
        ]

        for proc in psutil.process_iter(["pid", "cmdline"]):
            try:
                if proc.pid == my_pid:
                    continue
                cmdline = " ".join(proc.info.get("cmdline") or [])
                if any(t in cmdline for t in targets):
                    # Kill children first, then process
                    for child in proc.children(recursive=True):
                        try:
                            child.kill()
                            killed += 1
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            pass
                    proc.kill()
                    killed += 1
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

    except ImportError:
        # Fallback: pkill by pattern
        try:
            import subprocess as _sp
            for pattern in [
                "superlocalmemory.server.unified_daemon",
                "superlocalmemory.core.embedding_worker",
                "superlocalmemory.core.recall_worker",
                "superlocalmemory.core.reranker_worker",
            ]:
                result = _sp.run(
                    ["pkill", "-9", "-f", pattern],
                    capture_output=True, timeout=5,
                )
                if result.returncode == 0:
                    killed += 1
        except Exception:
            pass

    # Clean up PID/port files + worker PID files
    _PID_FILE.unlink(missing_ok=True)
    _PORT_FILE.unlink(missing_ok=True)
    # v3.4.13: Clean worker PID files (singleton guards)
    for pidfile in (".embedding-worker.pid", ".reranker-worker.pid"):
        (Path.home() / ".superlocalmemory" / pidfile).unlink(missing_ok=True)

    # v3.4.13: Wait for ALL workers to actually die before returning.
    # Without this, `slm restart` starts a new daemon before old workers exit,
    # causing duplicate embedding_workers (1.6GB each).
    if killed:
        logger.info("Stopped %d SLM processes, waiting for exit...", killed)
        _wait_for_workers_dead(timeout=10)

    return True


def _wait_for_workers_dead(timeout: int = 10) -> None:
    """Wait until no SLM worker processes remain alive."""
    targets = [
        "superlocalmemory.server.unified_daemon",
        "superlocalmemory.core.embedding_worker",
        "superlocalmemory.core.recall_worker",
        "superlocalmemory.core.reranker_worker",
    ]
    my_pid = os.getpid()
    deadline = time.time() + timeout

    while time.time() < deadline:
        alive = False
        try:
            import psutil
            for proc in psutil.process_iter(["pid", "cmdline"]):
                try:
                    if proc.pid == my_pid:
                        continue
                    cmdline = " ".join(proc.info.get("cmdline") or [])
                    if any(t in cmdline for t in targets):
                        alive = True
                        break
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
        except ImportError:
            # No psutil — just wait a fixed time
            time.sleep(3)
            return

        if not alive:
            return
        time.sleep(0.5)

    logger.warning("Some SLM workers still alive after %ds timeout", timeout)
