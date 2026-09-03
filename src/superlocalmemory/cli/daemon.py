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
import socket
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
from superlocalmemory.infra.process_identity import (
    compare_start_tokens,
    process_start_token_for,
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
_EXPECTED_DESCRIPTOR_UNSET = object()


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


_CREATE_TIME_TOLERANCE_SECONDS = 1.0


def _health_proves_descriptor_ownership(descriptor) -> bool:
    """Return whether the live health endpoint proves this exact daemon.

    This is a *stronger* ownership proof than any process-table comparison. To
    pass, a process listening on the descriptor's port must echo the random
    128-bit ``instance_id`` and the SHA-256 fingerprint of the 256-bit
    capability token -- both of which exist only inside the mode-0600
    ``daemon.json`` -- alongside its own PID, namespace, owner and port. A
    process that merely inherited a recycled PID cannot produce any of that.
    """
    health = _fetch_health(descriptor.port)
    if health is None:
        return False
    return descriptor_matches_health(descriptor, health)


def _resolve_descriptor_liveness(descriptor) -> tuple[bool, str]:
    """Return ``(is_alive, evidence)`` for the descriptor's recorded process.

    Ownership is decided by the strongest available evidence, never by the
    wall clock alone:

    1. The PID must exist and must not be a zombie.
    2. A clock-independent start token settles it exactly, with no tolerance.
       This is the path that fixes issue #104: under WSL2 the boot time behind
       ``psutil.create_time`` drifts against the wall clock during a session,
       so a recorded creation time stops matching the *same* live process
       (~35s after ~4 minutes).  A start token cannot drift, so no tolerance
       constant is needed and none can silently expire.
    3. Otherwise fall back to comparing creation times, for descriptors written
       by an older release and for platforms with no token (Windows, where the
       kernel creation time is already immune to clock adjustment).
    4. A creation-time mismatch is *not* proof of PID reuse -- it is exactly
       what a stepped clock looks like -- so before condemning a running
       daemon, ask the daemon to prove its identity over loopback. Only if that
       cryptographic proof also fails is the process declared foreign.
    """
    if not _is_pid_alive(descriptor.pid):
        return False, "process_exited"
    try:
        import psutil
    except ImportError:
        # Without psutil, PID existence is the only signal there is.
        return True, "pid_exists_without_psutil"
    try:
        process = psutil.Process(descriptor.pid)
        # A terminated daemon can remain in the process table briefly as a
        # zombie while its parent reaps it.  PID existence is therefore not
        # liveness and must not block a namespace-owned restart.
        if not process.is_running() or process.status() == psutil.STATUS_ZOMBIE:
            return False, "process_zombie"
        actual_create_time = float(process.create_time())
    except Exception:
        return False, "process_unreadable"

    recorded_token = getattr(descriptor, "process_start_token", None)
    if recorded_token:
        verdict = compare_start_tokens(
            recorded_token, process_start_token_for(descriptor.pid),
        )
        if verdict is True:
            return True, "start_token_match"
        if verdict is False:
            return False, "start_token_mismatch"

    drift = abs(actual_create_time - float(descriptor.process_create_time))
    if drift <= _CREATE_TIME_TOLERANCE_SECONDS:
        return True, "create_time_match"

    if _health_proves_descriptor_ownership(descriptor):
        logger.debug(
            "descriptor creation time drifted by %.3fs for pid %s; owned "
            "daemon confirmed by health identity instead",
            drift, descriptor.pid,
        )
        return True, "health_identity_match"
    return False, "identity_mismatch"


def _descriptor_process_is_alive(descriptor) -> bool:
    """Reject stale descriptors when a PID has been reused by another process."""
    return _resolve_descriptor_liveness(descriptor)[0]


def _is_port_available(port: int) -> bool:
    """Return whether the daemon port can be exclusively bound right now."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as candidate:
            if sys.platform == "win32":
                # Winsock SO_REUSEADDR can bind an address that is still
                # occupied, so it cannot prove shutdown completion. Request
                # exclusive ownership where available and otherwise use the
                # default non-reuse bind contract.
                exclusive = getattr(socket, "SO_EXCLUSIVEADDRUSE", None)
                if exclusive is not None:
                    candidate.setsockopt(socket.SOL_SOCKET, exclusive, 1)
            else:
                # On POSIX, mirror Uvicorn's reuse contract so a closed
                # listener's TIME_WAIT sockets do not block a safe restart.
                candidate.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            candidate.bind(("127.0.0.1", port))
        return True
    except OSError:
        return False


def _has_tcp_listener(port: int) -> bool:
    """Return whether a process is actively accepting on the daemon port."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as candidate:
            candidate.settimeout(0.5)
            return candidate.connect_ex(("127.0.0.1", port)) == 0
    except OSError:
        return False


def wait_for_owned_daemon_shutdown(
    descriptor,
    timeout: float = 25.0,
    *,
    legacy_pid: int | None = None,
    legacy_port: int | None = None,
) -> bool:
    """Wait for the stopped instance *and* its TCP listener to be gone.

    Restart must never spawn a replacement just because the descriptor was
    removed: a graceful shutdown can remove it before Uvicorn releases the
    port.  The 25-second budget covers Uvicorn's 10-second graceful drain plus
    SLM worker cleanup. A descriptor carries process creation time, so PID
    reuse cannot make this wait target an unrelated process.
    """
    port = (
        descriptor.port
        if descriptor is not None
        else legacy_port if legacy_port is not None else _DEFAULT_PORT
    )
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        process_alive = bool(
            _descriptor_process_is_alive(descriptor)
            if descriptor is not None
            else legacy_pid is not None and _is_verified_legacy_process(legacy_pid)
        )
        if not process_alive and _is_port_available(port):
            return True
        time.sleep(0.1)
    return False


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


def owned_daemon_process_alive() -> bool:
    """Return whether an owned daemon *process* is alive, HTTP aside.

    ``is_daemon_running()`` additionally requires a live, matching
    ``/health`` response, which conflates two different questions: "is there
    a process I need to stop" and "is it ready to serve requests right now."
    A daemon whose event loop is synchronously blocked by a long-running
    handler (``/maintenance/run``, ``/consolidate/cognitive``) cannot answer
    the second for the duration of that call, but the answer to the first is
    still yes. Callers that only need to decide whether a stop is owed
    (``slm restart`` Step 1) should use this instead, so a transiently busy
    daemon is not skipped as "already stopped" while it keeps running and
    holding the port — which then made Step 3 refuse to start a second
    daemon on the still-occupied port and fail the whole restart.
    """
    descriptor = read_descriptor()
    if descriptor is not None:
        return _descriptor_process_is_alive(descriptor)
    if descriptor_path().exists():
        return False
    return _verified_legacy_health() is not None


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


def _health_is_owned(health: dict, *, port: int | None = None) -> bool:
    """Return whether an answering health payload belongs to this namespace.

    Descriptor present: the payload must echo the descriptor's identity
    (namespace, instance, capability, PID, port). No descriptor: only a
    verified legacy same-root daemon counts — and when the probed ``port``
    is given, the occupant must BE that legacy daemon (same PID answering
    on the legacy port). Anything else answering is foreign by definition.

    4.1.14 audit: the port check closes the spoof where a JSON /health on
    the probed port plus a leftover daemon.pid skipped the loud fail-fast.
    """
    descriptor = read_descriptor()
    if descriptor is not None:
        try:
            return bool(descriptor_matches_health(descriptor, health))
        except Exception:
            return False
    try:
        legacy = _verified_legacy_health()
    except Exception:
        return False
    if legacy is None:
        return False
    if port is None:
        return True
    try:
        return (
            int(health.get("pid", -1)) == int(legacy.get("pid", -2))
            and int(port) == int(legacy.get("_legacy_port", -3))
        )
    except (TypeError, ValueError):
        return False


class DaemonRefused(RuntimeError):
    """The daemon answered, and the answer was no.

    Raised for HTTP 401 and 403 only. Distinct from ``daemon_request``
    returning ``None``, which means the daemon could not be reached or did not
    answer usefully. Callers that fall back to a direct engine write MUST let
    this propagate or exit on it: falling back after a refusal performs, as the
    machine owner, exactly the write the workspace just declined.
    """

    def __init__(self, status: int, path: str = "") -> None:
        self.status = int(status)
        self.path = path
        super().__init__(
            f"the daemon refused this request (HTTP {status})"
            + (f" for {path}" if path else "")
        )


class DaemonConflict(RuntimeError):
    """A deterministic daemon conflict that the caller must resolve."""

    def __init__(self, detail: str) -> None:
        self.detail = detail or "daemon request conflicted with current state"
        super().__init__(self.detail)


class DaemonNotFound(RuntimeError):
    """The daemon answered 404 with a structured error body.

    Raised only when the caller passes ``preserve_not_found=True``: a live
    daemon refusing an unknown id (e.g. per-request routing to a missing
    profile) is an answer, not an outage, and collapsing it to None made
    ``unknown_profile`` indistinguishable from a dead daemon (#audit).
    """

    def __init__(self, status: int, code: str, message: str, path: str = "") -> None:
        self.status = int(status)
        self.code = code or "not_found"
        self.message = message or "daemon returned 404"
        super().__init__(self.message + (f" for {path}" if path else ""))


def daemon_request(
    method: str,
    path: str,
    body: dict | None = None,
    *,
    timeout_seconds: float = 30.0,
    expected_descriptor=_EXPECTED_DESCRIPTOR_UNSET,
    expected_legacy: dict | None = None,
    verify_health: bool = True,
    preserve_conflict: bool = False,
    preserve_not_found: bool = False,
) -> dict | None:
    """Send a request only after validating the owned daemon identity.

    ``verify_health`` — when True (the default), a ``GET /health`` preflight
    must succeed and match the descriptor before the real request is sent.
    That preflight needs the daemon's event loop to be free to answer HTTP,
    which is a *readiness* question, not a *liveness* one: a daemon whose
    loop is synchronously blocked by a long-running handler (e.g.
    ``/maintenance/run``, ``/consolidate/cognitive``, neither of which is
    offloaded to a thread the way ``/recall`` was for exactly this reason in
    v3.4.52) cannot answer /health for the duration of that call even though
    the process is fully alive and listening. Callers that have already
    proven process-level ownership some other way (e.g. ``stop_daemon()`` via
    ``_descriptor_process_is_alive``) should pass ``verify_health=False`` so a
    busy-but-alive daemon does not get misreported as not running. Only
    meaningful for the descriptor path — the legacy bridge has no capability
    header and still needs health to identify its target.
    """
    legacy = None
    if expected_legacy is not None:
        # Legacy daemons have no capability header. Bind the compatibility
        # request to the captured PID+port and refuse to adopt a replacement
        # descriptor or a different legacy process during this stop.
        if read_descriptor() is not None:
            return None
        current_legacy = _verified_legacy_health()
        if current_legacy is None or (
            int(current_legacy.get("pid", -1))
            != int(expected_legacy.get("pid", -2))
            or int(current_legacy.get("_legacy_port", -1))
            != int(expected_legacy.get("_legacy_port", -2))
        ):
            return None
        descriptor = None
        legacy = current_legacy
    else:
        descriptor = (
            read_descriptor()
            if expected_descriptor is _EXPECTED_DESCRIPTOR_UNSET
            else expected_descriptor
        )
    capability: str | None = None
    target_instance: str | None = None
    if descriptor is not None:
        if verify_health:
            health = _fetch_health(descriptor.port)
            if health is None or not descriptor_matches_health(descriptor, health):
                return None
            if method.upper() == "GET" and path == "/health":
                return health
        port = descriptor.port
        capability = descriptor.capability
        target_instance = descriptor.instance_id
    elif descriptor_path().exists():
        return None
    else:
        legacy = legacy or _verified_legacy_health()
        if legacy is None:
            return None
        if method.upper() == "GET" and path == "/health":
            return {key: value for key, value in legacy.items() if key != "_legacy_port"}
        port = int(legacy["_legacy_port"])
    try:
        import urllib.error
        import urllib.request
        url = f"http://127.0.0.1:{port}{path}"
        data = json.dumps(body).encode() if body else None
        headers = {"Content-Type": "application/json"} if data else {}
        if capability is not None and target_instance is not None:
            headers["X-SLM-Daemon-Capability"] = capability
            headers["X-SLM-Target-Instance"] = target_instance
        # Daemon ownership proves that this CLI targets the local instance; it
        # does not replace a dashboard user's profile-scoped authorization in
        # governed workspaces. The user opts in by supplying an explicit
        # session through the process environment (never logged or persisted).
        user_session = os.environ.get("SLM_USER_SESSION", "").strip()
        if user_session:
            headers["X-SLM-User-Session"] = user_session
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        resp = urllib.request.urlopen(req, timeout=timeout_seconds)
        return json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        # A refusal is an answer, not a failure to get one. Returning None here
        # made "you are not allowed to do this" indistinguishable from "the
        # daemon is not running", and every caller that falls back to a local
        # engine write treated the first as the second -- so a workspace that
        # required a login refused the write over HTTP and then performed it
        # locally as the machine owner.
        if exc.code in (401, 403):
            raise DaemonRefused(exc.code, path) from exc
        if exc.code == 409 and preserve_conflict:
            detail = "daemon request conflicted with current state"
            try:
                payload = json.loads(exc.read().decode())
                if isinstance(payload, dict) and payload.get("detail"):
                    detail = str(payload["detail"])
            except Exception:
                pass
            raise DaemonConflict(detail) from exc
        if exc.code == 404 and preserve_not_found:
            code, message = "not_found", "daemon returned 404"
            try:
                payload = json.loads(exc.read().decode())
                if isinstance(payload, dict):
                    err = payload.get("error", {})
                    if isinstance(err, dict):
                        code = str(err.get("code", code))
                        message = str(err.get("message", message))
            except Exception:
                pass
            raise DaemonNotFound(exc.code, code, message, path) from exc
        return None
    except Exception:
        return None


_LOCK_FILE = None  # test-only override; runtime resolution stays dynamic


def _pid_file_path():
    return _PID_FILE or state_path("daemon.pid")


def _port_file_path():
    return _PORT_FILE or state_path("daemon.port")


def _lock_file_path():
    return _LOCK_FILE or state_path("daemon.lock")


def _start_daemon_subprocess(*, port: int | None = None) -> bool:
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
    # Never create a descriptor for a child that cannot own the listener.
    # A closed connection in TIME_WAIT is not a listener and is safe: the
    # server reserves its socket with SO_REUSEADDR during bootstrap.
    # 4.1.14 audit: bind the port ensure_daemon probed — probing a custom
    # port and then spawning on the default opened the browser on a port
    # with no daemon (or refused a start the probe had cleared).
    _target_port = port if port is not None else _DEFAULT_PORT
    if _has_tcp_listener(_target_port):
        logger.warning("daemon port %d is already owned; refusing a second start", _target_port)
        return False
    assert_no_durable_root_conflict()

    import subprocess

    from superlocalmemory import __version__ as _slm_version
    # v3.6.9 (#33): pass SLM_DAEMON_PORT as explicit --port= so the daemon
    # binds the right port even when the env var reaches the subprocess.
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
        process_start_token=process_start_token_for(proc.pid),
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


def ensure_daemon(*, port: int | None = None) -> bool:
    """Start daemon if not running. Returns True if daemon is ready.

    ``port`` — when supplied, the daemon is started (or verified) on this port
    instead of the configured default.  The dashboard passes its own ``--port``
    here so the bind authority matches the URL shown to the user.

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
        #
        # 4.1.14 (#132): probe the CONFIGURED port, not the default constant —
        # a foreign squatter on 8765 must not divert startup when this
        # namespace serves elsewhere. And when the occupant answers HTTP
        # with non-SLM identity, fail fast and loud instead of burning a
        # 30 s wait: an answering foreign service can never become our
        # daemon. Silence (nothing answering) keeps the old wait — a slow
        # starter is indistinguishable from a raw squat.
        probe_port = port if port is not None else _get_port()
        if _has_tcp_listener(probe_port):
            occupant = _fetch_health(probe_port)
            if occupant is not None and not _health_is_owned(
                occupant, port=probe_port,
            ):
                logger.error(
                    "SLM daemon will not start: port %d is occupied by a "
                    "foreign service (answered HTTP without SLM identity). "
                    "Free the port or point this namespace elsewhere, then "
                    "run `slm restart`.",
                    probe_port,
                )
                return False
            return _wait_for_daemon(timeout=30)

        # Start unified daemon in background — delegated to helper so the
        # same logic can be reused by callers that already hold the lock.
        # 4.1.14 audit: the probed port travels with the spawn.
        return _start_daemon_subprocess(port=port)

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


_GENERIC_UNAVAILABLE = {
    "reason": "unknown",
    "message": "Owned daemon is unavailable; retry later.",
    "hint": "Run `slm doctor`, then `slm restart` if it stays down.",
}

_LIVENESS_DIAGNOSIS = {
    "process_exited": (
        "daemon_process_exited",
        "the recorded daemon process (pid {pid}) is no longer running",
        "Start it again with `slm start`.",
    ),
    "process_zombie": (
        "daemon_process_exited",
        "the recorded daemon process (pid {pid}) has exited and is awaiting "
        "reaping by its parent",
        "Start it again with `slm start`.",
    ),
    "process_unreadable": (
        "daemon_process_unreadable",
        "the recorded daemon process (pid {pid}) could not be inspected; it "
        "may belong to another user",
        "Run `slm restart` to publish a fresh descriptor.",
    ),
    "start_token_mismatch": (
        "pid_reused_by_another_process",
        "pid {pid} is alive but is a different process than the daemon that "
        "wrote {path}; the daemon exited and its pid was recycled",
        "Run `slm restart` to publish a fresh descriptor.",
    ),
    "identity_mismatch": (
        "daemon_identity_mismatch",
        "pid {pid} did not match the process identity recorded in {path} and "
        "the process on port {port} did not prove it owns that identity; the "
        "recorded creation time can also diverge on its own if this machine's "
        "clock is stepped (common under WSL2)",
        "Run `slm restart` to publish a fresh descriptor.",
    ),
}


def describe_daemon_unavailability() -> dict[str, str]:
    """Explain *why* the owned daemon cannot be used, in actionable terms.

    "Owned daemon is unavailable" is true of a stopped daemon, a recycled PID,
    an unreachable port and an identity mismatch alike, which left issue #104's
    reporter with nothing to act on. This names the specific evidence instead.
    Diagnosis is best-effort and never raises: a broken diagnosis must not
    replace the caller's real error.
    """
    try:
        return _describe_daemon_unavailability()
    except Exception:  # noqa: BLE001 - diagnosis is advisory only
        return dict(_GENERIC_UNAVAILABLE)


def _describe_daemon_unavailability() -> dict[str, str]:
    path = descriptor_path()
    descriptor = read_descriptor()
    if descriptor is None:
        if path.exists():
            return {
                "reason": "descriptor_unusable",
                "message": (
                    f"{path} is unreadable, malformed, or belongs to another "
                    f"data root or user."
                ),
                "hint": "Run `slm restart` to publish a fresh descriptor.",
            }
        if _verified_legacy_health() is not None:
            return {
                "reason": "legacy_daemon_request_failed",
                "message": (
                    "a pre-descriptor daemon answered health but rejected or "
                    "dropped the request."
                ),
                "hint": "Run `slm restart` to upgrade it to an owned daemon.",
            }
        return {
            "reason": "no_daemon",
            "message": f"no daemon is registered for this data root ({path} is absent).",
            "hint": "Run `slm start`.",
        }

    alive, evidence = _resolve_descriptor_liveness(descriptor)
    if not alive:
        reason, template, hint = _LIVENESS_DIAGNOSIS.get(
            evidence,
            (
                "daemon_identity_mismatch",
                "pid {pid} did not match the identity recorded in {path}",
                "Run `slm restart` to publish a fresh descriptor.",
            ),
        )
        return {
            "reason": reason,
            "message": template.format(
                pid=descriptor.pid, port=descriptor.port, path=path,
            ) + ".",
            "hint": hint,
        }

    health = _fetch_health(descriptor.port)
    if health is None:
        return {
            "reason": "daemon_unreachable",
            "message": (
                f"the owned daemon (pid {descriptor.pid}) is running but did "
                f"not answer http://127.0.0.1:{descriptor.port}/health within "
                f"2s."
            ),
            "hint": (
                "Check `slm logs` for a stalled request, or `slm restart` if "
                "it stays unresponsive."
            ),
        }
    if not descriptor_matches_health(descriptor, health):
        return {
            "reason": "port_owned_by_another_daemon",
            "message": (
                f"port {descriptor.port} answered health but with a different "
                f"daemon identity than {path} records."
            ),
            "hint": (
                "Another SuperLocalMemory instance holds that port. Stop it, "
                "or set SLM_DAEMON_PORT to a free port."
            ),
        }
    return {
        "reason": "request_rejected",
        "message": (
            f"the owned daemon (pid {descriptor.pid}) is healthy but rejected "
            f"or dropped this request."
        ),
        "hint": "Check `slm logs` for the failing request.",
    }


def stop_daemon() -> bool:
    """Stop only the daemon proven to belong to this data namespace.

    Machine-wide process-name scans are forbidden: they can kill another SLM
    installation or a user's live workers during tests. V3.7 uses the owned
    HTTP capability; the daemon itself terminates its child process tree.
    Success means the owned process exited and released its listener, not just
    that the asynchronous stop request was accepted.

    A busy daemon is not a dead daemon. ``daemon_request()`` normally
    preflights every call with ``GET /health`` before sending it, but that
    preflight needs the daemon's single-threaded event loop to be free to
    answer HTTP. ``/maintenance/run`` and ``/consolidate/cognitive`` run
    multi-second (sometimes multi-minute) synchronous work directly inline in
    their handlers with no thread offload, which blocks *every* request on
    that loop, health included, for as long as they run. Reproduced live: a
    genuine ``/maintenance/run`` call held the loop long enough that 15/15
    health polls during the window timed out at exactly the 2s cap while
    ``ps``/``lsof`` proved the process never stopped listening — which is
    exactly the "Daemon was not running" false report this fixes. Process
    liveness (PID + clock-independent start token, proven below via
    ``_descriptor_process_is_alive``) is the fact that actually matters for
    "should I try to stop this," so it is checked directly and the mutating
    ``/stop`` POST is sent with ``verify_health=False`` once that is proven —
    the daemon still authenticates the request by its capability header on
    arrival, so this loses no ownership guarantee, only the redundant,
    stall-prone preflight round trip.
    """
    descriptor = read_descriptor()
    legacy = _verified_legacy_health() if descriptor is None else None
    if descriptor is None and legacy is None:
        return False
    if descriptor is not None:
        if not _descriptor_process_is_alive(descriptor):
            return False
        response = daemon_request(
            "POST",
            "/stop",
            expected_descriptor=descriptor,
            verify_health=False,
        )
    else:
        if legacy is None:
            return False
        response = daemon_request(
            "POST",
            "/stop",
            expected_legacy=legacy,
        )
    if not response or response.get("status") != "stopping":
        return False
    if descriptor is not None:
        return wait_for_owned_daemon_shutdown(descriptor)
    if legacy is None:
        return False
    return wait_for_owned_daemon_shutdown(
        None,
        legacy_pid=int(legacy["pid"]),
        legacy_port=int(legacy["_legacy_port"]),
    )
