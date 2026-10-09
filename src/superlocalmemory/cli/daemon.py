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

from superlocalmemory.cli import daemon_startup as _startup
from superlocalmemory.infra.daemon_identity import (
    build_descriptor,
    descriptor_matches_health,
    descriptor_path,
    health_is_other_account,
    health_is_same_account,
    read_descriptor,
)
from superlocalmemory.infra.data_root import (
    assert_no_durable_root_conflict,
    state_path,
)
from superlocalmemory.infra.instance_lock import instance_lock_is_held
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
        from superlocalmemory.core.platform_utils import is_pid_alive
        return is_pid_alive(pid)


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


def _wait_for_republished_record(
    timeout_s: float = 6.0, poll_s: float = 0.25,
) -> bool:
    """Wait for the owning daemon to publish a record that names a live process.

    Only the daemon writes the record; a client that finds it stale or damaged
    while health still answers waits for the daemon's guardian to repair it.
    """
    deadline = time.monotonic() + timeout_s
    while True:
        descriptor = read_descriptor()
        if descriptor is not None and _descriptor_process_is_alive(descriptor):
            if descriptor.state == "starting":
                return True
            health = _fetch_health(descriptor.port)
            if health is not None and descriptor_matches_health(descriptor, health):
                return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(poll_s)


def _health_fallback(port: int) -> bool:
    """The record is stale or damaged: trust only a same-account daemon's repair.

    A daemon of another account or data folder never passes, and nothing here
    reads or returns a capability (health carries none).
    """
    health = _fetch_health(port)
    if health is None or not health_is_same_account(health):
        return False
    return _wait_for_republished_record()


def is_daemon_running() -> bool:
    """Return True only for a daemon owned by this canonical data namespace.

    A PID or an HTTP 200 proves liveness, not ownership. V3.7 requires the
    private local descriptor and the health endpoint to agree on namespace,
    process instance, capability fingerprint, owner, PID, protocol, and port.
    A stale or damaged record next to a healthy same-account daemon is waited
    out (the daemon repairs it) rather than reported as "not running".
    """
    local_descriptor_path = descriptor_path()
    descriptor = read_descriptor()
    if descriptor is not None:
        if not _descriptor_process_is_alive(descriptor):
            return _health_fallback(descriptor.port)
        if descriptor.state == "starting":
            return True
        health = _fetch_health(descriptor.port)
        return health is not None and descriptor_matches_health(descriptor, health)

    # A malformed or foreign descriptor never falls through to legacy PID/port
    # adoption in the same namespace; only the health fallback may rescue it.
    if local_descriptor_path.exists():
        return _health_fallback(_get_port())

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

    A stale or damaged record beside a healthy same-account daemon takes the
    same health fallback as ``is_daemon_running``.
    """
    descriptor = read_descriptor()
    if descriptor is not None:
        if _descriptor_process_is_alive(descriptor):
            return True
        port = descriptor.port
    elif descriptor_path().exists():
        port = _get_port()
    else:
        return _verified_legacy_health() is not None
    if not _health_fallback(port):
        return False
    repaired = read_descriptor()
    return repaired is not None and _descriptor_process_is_alive(repaired)


def _fetch_health(port: int, timeout: float = 2.0) -> dict | None:
    """Fetch loopback health without following cross-namespace discovery."""
    try:
        import urllib.request

        expected_url = f"http://127.0.0.1:{port}/health"
        response = urllib.request.urlopen(
            expected_url, timeout=max(0.05, min(float(timeout), 2.0)),
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


def _process_is_this_account(process) -> bool:
    """The process runs as the account running this code.

    On a computer shared by several accounts, another account's daemon can
    hold a PID a stale ``daemon.pid`` here still names. Windows has no uid,
    so there the account is the user name the process runs as. A process
    whose account cannot be read is not counted as this account's.
    """
    getuid = getattr(os, "getuid", None)
    try:
        if getuid is not None:
            return int(process.uids().real) == int(getuid())
        from superlocalmemory.core.platform_utils import current_account

        return str(process.username()).casefold() == current_account().casefold()
    except Exception:
        return False


def _is_verified_legacy_process(pid: int) -> bool:
    """One-release bridge for a same-root V3.6 unified-daemon process."""
    if not _is_pid_alive(pid):
        return False
    try:
        import psutil

        process = psutil.Process(pid)
        if not _process_is_this_account(process):
            return False
        command = " ".join(process.cmdline())
        return "superlocalmemory.server.unified_daemon" in command
    except Exception:
        return False


def _verified_legacy_health() -> dict | None:
    """Accept legacy health only with a verified same-root daemon PID file."""
    pid_file = descriptor_path().with_name("daemon.pid")
    port_file = descriptor_path().with_name("daemon.port")
    try:
        pid = int(pid_file.read_text(encoding="utf-8").strip())
        port = int(port_file.read_text(encoding="utf-8").strip()) if port_file.exists() else _DEFAULT_PORT
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


def owned_daemon_answers(port: int) -> bool:
    """This account's own daemon answers on ``port`` (proven, not assumed).

    Use before sending anything to a loopback port: on a shared computer the
    port may belong to another account's SuperLocalMemory, or to anything.
    """
    health = _fetch_health(int(port))
    return health is not None and _health_is_owned(health, port=int(port))


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

    def __init__(self, status: int, code: str, message: str, path: str = "", *,
                 error_code: str = "", error_message: str = "") -> None:
        self.status = int(status)
        self.code = code or "not_found"
        self.message = message or "daemon returned 404"
        # A per-request-routing route answers an unknown profile with
        # ``{"error": {"code": "unknown_profile", ...}}`` rather than a
        # ``detail``. Kept apart from ``code`` so no existing caller changes;
        # a routed caller reads it (mcp/tools_core._routed_daemon_call).
        self.error_code = error_code
        self.error_message = error_message
        super().__init__(self.message + (f" for {path}" if path else ""))


def not_found_from(payload: object, path: str = "") -> DaemonNotFound:
    """The :class:`DaemonNotFound` for a 404 response body.

    L3-11: every 404 in this codebase is a plain FastAPI
    HTTPException(404, detail="...") -- {"detail": "..."} or {"detail": {...}}
    -- so ``code`` and ``message`` come from ``detail``. The routed-profile
    routes' ``{"error": {...}}`` body fills ``error_code``/``error_message``.
    """
    code, message = "not_found", "daemon returned 404"
    error_code, error_message = "", ""
    if isinstance(payload, dict):
        detail = payload.get("detail")
        if isinstance(detail, dict):
            code = str(detail.get("code", code))
            message = str(detail.get("message", message))
        elif isinstance(detail, str) and detail:
            message = detail
        error = payload.get("error")
        if isinstance(error, dict) and isinstance(error.get("code"), str):
            error_code = error["code"]
            error_message = str(error.get("message", ""))
    return DaemonNotFound(404, code, message, path,
                          error_code=error_code, error_message=error_message)


class DaemonUnprocessable(RuntimeError):
    """The daemon answered 422: it refused the request itself, before any work.

    Raised only when the caller passes ``preserve_unprocessable=True``. Without
    it a 422 collapses to ``None``, which callers read as "daemon unavailable"
    and retry - pointless for a request that will be refused every time.
    """

    def __init__(self, code: str, message: str) -> None:
        self.code = code or "INVALID_REQUEST"
        self.message = message or "the daemon refused this request"
        super().__init__(self.message)


def _unprocessable(exc) -> DaemonUnprocessable:
    """Read ``{"detail": {"code", "message"}}``, ``{"detail": "text"}``, or
    FastAPI's own list-shaped validation-error body —
    ``{"detail": [{"type": ..., "loc": [...], "msg": "..."}, ...]}`` — which a
    request that fails Pydantic's own field validation (e.g. a fact_id over
    the field's max_length) gets before any route code runs (L3-11). Without
    this branch the message silently went empty for exactly that shape.
    """
    try:
        detail = json.loads(exc.read().decode()).get("detail")
    except Exception:  # noqa: BLE001 - an unreadable body still means "refused"
        detail = None
    if isinstance(detail, dict):
        return DaemonUnprocessable(str(detail.get("code") or ""),
                                   str(detail.get("message") or ""))
    if isinstance(detail, list):
        messages = [
            str(item.get("msg", "")) for item in detail
            if isinstance(item, dict) and item.get("msg")
        ]
        return DaemonUnprocessable("", "; ".join(messages) or "the request was invalid")
    message = detail if isinstance(detail, str) else ""
    # The daemon's own wording for a key reused for another request.
    code = "IDEMPOTENCY_CONFLICT" if message.startswith("idempotency key ") else ""
    return DaemonUnprocessable(code, message)


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
    preserve_unprocessable: bool = False,
    start_wait_seconds: float | None = None,
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

    ``start_wait_seconds`` — 4.1.22: when the daemon is *starting* (this
    process is spawning it, or its descriptor says so), wait up to this long
    (default: the ``daemon_startup`` budget, never more than
    ``timeout_seconds``) for it to answer, instead of failing at once. Nothing
    starting means no wait. Callers pinning a descriptor never wait.
    """
    unpinned = expected_descriptor is _EXPECTED_DESCRIPTOR_UNSET and expected_legacy is None
    wait_cap = (
        timeout_seconds if start_wait_seconds is None
        else min(start_wait_seconds, timeout_seconds)
    )
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
    if descriptor is None and unpinned and _startup.this_process_is_spawning():
        ready = _startup.wait_for_starting_daemon(cap=wait_cap)
        descriptor = ready[0] if ready is not None else read_descriptor()
    capability: str | None = None
    target_instance: str | None = None
    if descriptor is not None:
        if verify_health:
            probe_began = time.monotonic()
            health = _startup.probe_health(
                sys.modules[__name__], descriptor.port, _startup.health_probe_timeout(descriptor),
            )
            if health is None and unpinned:
                ready = _startup.wait_for_starting_daemon(
                    cap=wait_cap, already_waited=time.monotonic() - probe_began)
                if ready is not None:
                    descriptor, health = ready
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
        from superlocalmemory.core import outbound_http  # never via a proxy
        resp = outbound_http.urlopen(req, timeout=timeout_seconds)
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
            # The reason is read from the body (not_found_from): discarding
            # it made "Memory not found" or "no such profile" look alike.
            try:
                payload = json.loads(exc.read().decode())
            except Exception:
                payload = None
            raise not_found_from(payload, path) from exc
        if exc.code == 422 and preserve_unprocessable:
            raise _unprocessable(exc) from exc
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


def start_lock_is_held() -> bool:
    """Is some OTHER process holding ``daemon.lock`` right now?

    A non-mutating probe: tries the same non-blocking exclusive lock
    ``ensure_daemon`` takes, and releases it at once if acquired. Used by
    ``daemon_startup`` (4.1.22) to tell a cross-process start-in-progress
    (the lock is held, but the other process has not written a descriptor
    yet) apart from "nothing is starting" -- before this, that window had no
    signal at all, so a caller fell into ``ensure_daemon``'s old flat 60 s
    wait instead of the bounded start-wait budget, or a diagnosis call
    reported "no daemon" while one was actually starting.
    """
    lock_file = _lock_file_path()
    if not lock_file.exists():
        return False
    try:
        lock_fd = open(lock_file, "w", encoding="utf-8")
    except OSError:
        return False
    try:
        if sys.platform == "win32":
            import msvcrt
            try:
                msvcrt.locking(lock_fd.fileno(), msvcrt.LK_NBLCK, 1)
            except (IOError, OSError):
                return True
            try:
                msvcrt.locking(lock_fd.fileno(), msvcrt.LK_UNLCK, 1)
            except (IOError, OSError):
                pass
            return False
        else:
            import fcntl
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except (IOError, OSError):
                return True
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
            except (IOError, OSError):
                pass
            return False
    finally:
        try:
            lock_fd.close()
        except Exception:
            pass


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

    with open(log_file, "a", encoding="utf-8") as lf:
        proc = subprocess.Popen(cmd, stdout=lf, stderr=lf, **kwargs)

    # The child writes the record itself once it owns the data folder. The
    # parent never does: a child that loses a start race must not overwrite
    # the record of the daemon that won it.
    return _wait_for_daemon(timeout=60)


def _wait_bounded_for_lock_holder() -> bool:
    """Wait (bounded) for whoever holds ``daemon.lock`` to finish starting it.

    4.1.22: the file-lock branch of ``ensure_daemon`` used to wait a flat
    60 s here regardless of ``SLM_DAEMON_START_WAIT_S`` -- long enough to
    outlive a host's own ~60 s tool-call timeout before this process ever
    reported anything back. The holder has no descriptor to read yet in the
    window right after it wins the lock, so ``wait_for_starting_daemon``
    treats a currently-held lock as its own evidence of a start in progress
    (see ``daemon.start_lock_is_held``).
    """
    _startup.wait_for_starting_daemon()
    return is_daemon_running()


def _data_folder_is_owned() -> bool:
    """A daemon owns this data folder: it holds the lock or the record names it."""
    if instance_lock_is_held():
        return True
    descriptor = read_descriptor()
    return descriptor is not None and _descriptor_process_is_alive(descriptor)


def _wait_while_owned() -> bool:
    """Never spawn into an owned folder: wait the start budget for owned health."""
    deadline = time.monotonic() + _startup.start_wait_budget()
    while not is_daemon_running():
        if time.monotonic() >= deadline or not _data_folder_is_owned():
            return is_daemon_running()
        time.sleep(0.25)
    return True


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
    if _data_folder_is_owned():
        return _wait_while_owned()
    if _startup.this_process_is_spawning():
        # 4.1.22: another thread of THIS process holds the start lock and is
        # spawning. Never spawn twice; wait the bounded start budget only.
        _startup.wait_for_starting_daemon()
        return is_daemon_running()

    # File lock — prevent concurrent starts from multiple CLI/MCP calls
    lock_fd = None
    spawn_mark = _startup.spawning()
    marked = False
    try:
        lock_file = _lock_file_path()
        lock_file.parent.mkdir(parents=True, exist_ok=True)
        lock_fd = open(lock_file, "w", encoding="utf-8")

        # Cross-platform file locking
        if sys.platform == "win32":
            import msvcrt
            try:
                msvcrt.locking(lock_fd.fileno(), msvcrt.LK_NBLCK, 1)
            except (IOError, OSError):
                # Another process holds the start lock — wait the same
                # bounded start budget a same-process contender gets (4.1.22;
                # this used to be a flat 60 s wait that could outlive a
                # host's own tool-call timeout).
                lock_fd.close()
                return _wait_bounded_for_lock_holder()
        else:
            import fcntl
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except (IOError, OSError):
                lock_fd.close()
                return _wait_bounded_for_lock_holder()
        spawn_mark.__enter__()  # released in finally; see daemon_startup
        marked = True

        # Re-check after acquiring lock (another process may have started it)
        if is_daemon_running():
            return True
        if _data_folder_is_owned():
            return _wait_while_owned()

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
            if occupant is not None and health_is_other_account(occupant):
                logger.error(
                    "SLM daemon will not start: port %d is used by SuperLocalMemory "
                    "running for another account on this computer. Each account "
                    "needs its own port: set SLM_DAEMON_PORT (for example %d) for "
                    "this account, then run `slm restart`.",
                    probe_port, probe_port + 1,
                )
                return False
            if occupant is not None and not _health_is_owned(
                occupant, port=probe_port,
            ):
                # 4.1.14 audit: when the occupant PID matches this
                # namespace's stale daemon.pid, say so — the owned daemon
                # likely moved ports, and "foreign service" would send the
                # operator hunting the wrong process.
                _stale_hint = ""
                try:
                    _pid_text = (
                        descriptor_path().with_name("daemon.pid").read_text(encoding="utf-8").strip()
                    )
                    if _pid_text and str(occupant.get("pid")) == _pid_text:
                        _stale_hint = (
                            " The occupant PID matches this namespace's stale "
                            "daemon.pid — the owned daemon likely moved ports; "
                            "run `slm restart` instead of hunting a foreign process."
                        )
                except (OSError, ValueError):
                    pass
                logger.error(
                    "SLM daemon will not start: port %d is occupied by a "
                    "foreign service (answered HTTP without SLM identity).%s "
                    "Free the port or point this namespace elsewhere, then "
                    "run `slm restart`.",
                    probe_port,
                    _stale_hint,
                )
                return False
            return _wait_for_daemon(timeout=30)

        # Start unified daemon in background — delegated to helper so the
        # same logic can be reused by callers that already hold the lock.
        # 4.1.14 audit: the probed port travels with the spawn — probing a
        # custom port and spawning the default would serve elsewhere than
        # verified.
        return _start_daemon_subprocess(port=probe_port)

    except Exception as exc:
        # Daemon auto-start is the entry point for dashboard / mesh /
        # health features; failure here silently disables all of them.
        # Log at WARNING so operators can see it in production logs.
        logger.warning("ensure_daemon error: %s (run `slm doctor`)", exc)
        return False
    finally:
        if marked:
            spawn_mark.__exit__(None, None, None)
        if lock_fd:
            try:
                lock_fd.close()
            except Exception:
                pass
            # The lock file is never unlinked: removing a locked file lets a
            # third starter lock a fresh inode while this one is still held.


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


# The diagnosis tables and body live in daemon_diagnosis (4.1.22); these names
# stay importable from here.
from superlocalmemory.cli.daemon_diagnosis import (  # noqa: E402, F401 - re-exported
    _GENERIC_UNAVAILABLE,
    _LIVENESS_DIAGNOSIS,
)


def describe_daemon_unavailability() -> dict[str, str]:
    """Explain *why* the owned daemon cannot be used, in actionable terms.

    "Owned daemon is unavailable" is true of a stopped daemon, a recycled PID,
    an unreachable port, an identity mismatch and a daemon still starting
    alike, which left issue #104's reporter with nothing to act on. This names
    the specific evidence instead. Best-effort and never raises.
    """
    try:
        return _describe_daemon_unavailability()
    except Exception:  # noqa: BLE001 - diagnosis is advisory only
        return dict(_GENERIC_UNAVAILABLE)


def _describe_daemon_unavailability() -> dict[str, str]:
    from superlocalmemory.cli import daemon_diagnosis

    return daemon_diagnosis.describe(sys.modules[__name__])


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
        if descriptor.state == "starting":
            # 4.1.22: a starting daemon cannot take /stop yet. Returning False
            # here printed "not running" and left it running; wait for it.
            # Bounded (never a flat hang forever) and configurable via
            # SLM_DAEMON_STOP_WAIT_S -- a daemon wedged permanently in
            # "starting" must not make this command hang indefinitely.
            ready = _startup.wait_for_starting_daemon(seconds=_startup.stop_wait_budget())
            descriptor = ready[0] if ready is not None else descriptor
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
