# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""A bounded wait for a daemon that is still starting.

A fresh daemon needs seconds (a laptop with an empty store) to about a minute
(a small shared box loading models) before ``/health`` answers. Before 4.1.22
every daemon-backed call made in that window failed at once with
``DAEMON_UNAVAILABLE (daemon_unreachable)``, so the first ``remember`` after an
MCP host started SLM was lost to a race the caller could not see.

The rule here:

* wait only when there is evidence the daemon is starting — this process is
  spawning it, or the descriptor says ``starting`` and its process is alive;
* wait at most :func:`start_wait_budget` seconds (default 20 s);
* never spawn anything and never report success: when the budget runs out the
  caller gets its usual "unavailable" answer, which the diagnosis then names
  ``daemon_starting`` with a retry hint.

Why 20 s: the hosts that call SLM's MCP tools cancel a tool call at about 60 s
(Codex ``tool_timeout_sec`` defaults to 60; the Claude desktop app cancels
stdio tool calls near 60 s). A daemon request itself may take up to its own
30 s timeout, so 20 s of waiting plus 30 s of work stays about 10 s under the
host's limit, and it covers the 5-11 s cold start measured on a developer Mac.
``SLM_DAEMON_START_WAIT_S`` overrides it (clamped to 0-45 s; 0 disables).
"""

from __future__ import annotations

import os
import sys
import threading
import time
from contextlib import contextmanager
from typing import Any, Iterator

DEFAULT_START_WAIT_S = 20.0
MAX_START_WAIT_S = 45.0
# `slm serve stop` waits out a cold start before giving up on a daemon that is
# still "starting" (a small host's first model load can take about a
# minute). Bounded and configurable (4.1.22 polish) so a daemon wedged
# forever in "starting" cannot make the command hang indefinitely regardless
# of host size.
DEFAULT_STOP_WAIT_S = 90.0
MAX_STOP_WAIT_S = 180.0
RETRY_HINT_S = 10
_POLL_S = 0.25
# A starting daemon binds its port before HTTP is up, so a 2 s health read
# would just hang; while the descriptor says "starting" one probe is short.
#
# The same short bound also covers the one-shot probe when NOTHING shows a
# start in progress (``health_probe_timeout``'s other branch, used by the
# ordinary fail-fast path): POSIX refuses a loopback connect to a port
# nobody ever bound almost instantly, but Windows does not -- CI measured
# ~2 s for that refusal, which silently consumed the old 2 s health-read
# timeout there and made "nothing is starting" exactly as slow as a start
# actually in progress. A caller that specifically needs to tolerate a
# daemon that IS listening but slow to answer (busy, not starting) has its
# own opt-out already: ``daemon_request(..., verify_health=False)``.
STARTING_PROBE_S = 0.5
# Back-to-back retries inside one tool call (remember tries three times, 50-
# 100 ms apart) must not each wait the full budget again: a wait that ran out
# this recently for the same daemon instance is not repeated.
_REWAIT_GRACE_S = 2.0

_state_lock = threading.Lock()
_spawning_count = 0
_spawn_finished = threading.Event()
_spawn_finished.set()
_expired_at: dict[str, float] = {}


def start_wait_budget(cap: float | None = None) -> float:
    """Seconds a call may wait for a starting daemon (never negative)."""
    raw = os.environ.get("SLM_DAEMON_START_WAIT_S", "").strip()
    try:
        value = float(raw) if raw else DEFAULT_START_WAIT_S
    except ValueError:
        value = DEFAULT_START_WAIT_S
    if value != value:  # NaN
        value = DEFAULT_START_WAIT_S
    value = max(0.0, min(value, MAX_START_WAIT_S))
    if cap is not None:
        value = min(value, max(0.0, float(cap)))
    return value


def stop_wait_budget() -> float:
    """Seconds ``slm serve stop`` may wait for a starting daemon to finish
    starting before giving up (never negative, never unbounded)."""
    raw = os.environ.get("SLM_DAEMON_STOP_WAIT_S", "").strip()
    try:
        value = float(raw) if raw else DEFAULT_STOP_WAIT_S
    except ValueError:
        value = DEFAULT_STOP_WAIT_S
    if value != value:  # NaN
        value = DEFAULT_STOP_WAIT_S
    return max(0.0, min(value, MAX_STOP_WAIT_S))


@contextmanager
def spawning() -> Iterator[None]:
    """Mark that THIS process is spawning the daemon for the block's duration."""
    global _spawning_count
    with _state_lock:
        _spawning_count += 1
        _spawn_finished.clear()
    try:
        yield
    finally:
        with _state_lock:
            _spawning_count -= 1
            if _spawning_count <= 0:
                _spawning_count = 0
                _spawn_finished.set()


def this_process_is_spawning() -> bool:
    return not _spawn_finished.is_set()


def starting_descriptor() -> Any | None:
    """The descriptor of a live daemon that says it is still starting."""
    from superlocalmemory.cli import daemon as _d

    descriptor = _d.read_descriptor()
    if descriptor is None or getattr(descriptor, "state", "") != "starting":
        return None
    return descriptor if _d._descriptor_process_is_alive(descriptor) else None


def lock_is_held_by_another_process() -> bool:
    """Is some OTHER process holding the start lock, with no descriptor yet?

    The gap this closes (4.1.22): right after another process wins
    ``daemon.lock`` and before it writes a descriptor, neither
    ``this_process_is_spawning()`` (that flag lives in the OTHER process's
    memory) nor ``starting_descriptor()`` (nothing written yet) sees
    anything -- so without this check, a caller in THIS process saw "no
    evidence of starting" and either waited the old unbounded 60 s
    (``ensure_daemon``'s lock-contention branch) or reported "no daemon" (the
    diagnosis) while a start was genuinely under way.
    """
    from superlocalmemory.cli import daemon as _d

    try:
        return _d.start_lock_is_held()
    except Exception:
        return False


def start_in_progress() -> bool:
    """Evidence that the daemon is starting right now."""
    return (
        this_process_is_spawning()
        or starting_descriptor() is not None
        or lock_is_held_by_another_process()
    )


def _recently_expired(instance_id: str) -> bool:
    with _state_lock:
        at = _expired_at.get(instance_id)
    return at is not None and time.monotonic() - at < _REWAIT_GRACE_S


def _mark_expired(instance_id: str) -> None:
    with _state_lock:
        _expired_at.clear()  # one daemon per root; keep the map tiny
        _expired_at[instance_id] = time.monotonic()


def probe_health(d: Any, port: int, remaining: float) -> dict | None:
    """One health probe that cannot outlive the wait's own deadline.

    A starting daemon reserves its port early, so a connect can succeed and
    then hang until the read limit; capping by ``remaining`` keeps the whole
    wait inside its budget.
    """
    limit = max(0.05, min(2.0, remaining))
    try:
        return d._fetch_health(port, timeout=limit)
    except TypeError:  # a stand-in that takes only the port
        return d._fetch_health(port)


def health_probe_timeout(descriptor: Any) -> float:
    """Health read limit for ``descriptor``.

    Short (``STARTING_PROBE_S``) while the descriptor says it is starting
    (the polling loop needs each attempt to stay quick), and ALSO short on
    Windows specifically when nothing suggests a start is in progress (the
    ordinary fail-fast path): a bare "nobody is listening" answer is not
    fast there the way it is on POSIX (see ``STARTING_PROBE_S``'s comment),
    so the old 2 s default cost that path the same latency as an actual
    stuck start. POSIX refuses that connect immediately regardless of the
    timeout value, so shortening it there gains nothing and only narrows
    the window for a real, busy-but-alive daemon's /health to answer -- a
    genuine regression the Windows fix must not cause on macOS/Linux. A
    caller that already knows the daemon to be alive and merely wants to
    tolerate a slow-to-answer, busy event loop uses ``verify_health=False``
    instead of a longer timeout here.
    """
    if getattr(descriptor, "state", "") == "starting":
        return STARTING_PROBE_S
    if sys.platform == "win32" and not _recorded_process_is_alive(descriptor):
        # Issue #319: only a descriptor whose process is gone (or unknown)
        # needs the short read. A live, ready daemon on Windows is the same
        # busy-but-alive case POSIX already waits for; giving it 0.5 s turned
        # a slow /health answer into DAEMON_UNAVAILABLE.
        return STARTING_PROBE_S
    return ready_probe_seconds()


READY_PROBE_S = 2.0
_READY_PROBE_BOUNDS = (0.25, 10.0)


def ready_probe_seconds() -> float:
    """The /health read for a ready daemon: ``SLM_DAEMON_HEALTH_PROBE_S``,
    clamped to 0.25-10 s, default 2 s. Never applies to a starting daemon."""
    raw = os.environ.get("SLM_DAEMON_HEALTH_PROBE_S", "").strip()
    try:
        value = float(raw) if raw else READY_PROBE_S
    except ValueError:
        return READY_PROBE_S
    if value != value:  # NaN
        return READY_PROBE_S
    low, high = _READY_PROBE_BOUNDS
    return min(high, max(low, value))


def _recorded_process_is_alive(descriptor: Any) -> bool:
    pid = getattr(descriptor, "pid", None)
    if not isinstance(pid, int) or pid <= 0:
        return False
    from superlocalmemory.cli import daemon as _d

    try:
        return bool(_d._is_pid_alive(pid))
    except Exception:  # a liveness check must never break the probe
        return False


def wait_for_starting_daemon(
    *, cap: float | None = None, seconds: float | None = None,
    already_waited: float = 0.0,
) -> tuple[Any, dict] | None:
    """Wait (bounded) for a starting daemon; return ``(descriptor, health)``.

    Returns ``None`` at once when nothing shows a start in progress, so a
    stopped or wedged daemon behaves exactly as before. Returns ``None`` when
    the process exits or the budget runs out. Never spawns. ``seconds``
    replaces the configured budget (``slm serve stop`` waits longer than a
    tool call may).
    """
    from superlocalmemory.cli import daemon as _d

    first = starting_descriptor()
    lock_held = (
        first is None
        and not this_process_is_spawning()
        and lock_is_held_by_another_process()
    )
    if first is None and not this_process_is_spawning() and not lock_held:
        return None
    if seconds is None and first is not None and _recently_expired(first.instance_id):
        return None
    budget = start_wait_budget(cap) if seconds is None else max(0.0, float(seconds))
    # Time the caller already spent on this request (its own first probe)
    # comes out of the same budget: the wait is the call's, not a second one.
    budget = max(0.0, budget - max(0.0, float(already_waited)))
    deadline = time.monotonic() + budget
    last_instance = getattr(first, "instance_id", "")
    while True:
        descriptor = _d.read_descriptor()
        if descriptor is not None:
            last_instance = descriptor.instance_id
            if not _d._descriptor_process_is_alive(descriptor):
                return None  # it exited: the diagnosis says so
            health = probe_health(_d, descriptor.port, deadline - time.monotonic())
            if health is not None and _d.descriptor_matches_health(descriptor, health):
                return descriptor, health
        elif not this_process_is_spawning() and not lock_is_held_by_another_process():
            return None  # no descriptor and nobody — by any signal — is starting one
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            if last_instance:
                _mark_expired(last_instance)
            return None
        time.sleep(min(_POLL_S, remaining))


def starting_diagnosis(descriptor: Any) -> dict[str, str]:
    """The truthful answer for a call made while the daemon is starting."""
    try:
        age = max(0, int(time.time() - float(descriptor.process_create_time)))
        detail = f" (pid {descriptor.pid}, launched {age}s ago)"
    except (TypeError, ValueError, AttributeError):
        pid = getattr(descriptor, "pid", None)
        detail = f" (pid {pid})" if pid else ""
    return {
        "reason": "daemon_starting",
        "message": (
            f"SuperLocalMemory is still starting{detail}; "
            "a first start loads its models and can take up to a minute. "
            "This request was not sent."
        ),
        "hint": f"Retry in about {RETRY_HINT_S} seconds.",
    }
