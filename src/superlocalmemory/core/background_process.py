# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Run one pure computation in a spawned, low-priority child process.

For work that is all interpreter time -- decoding thousands of stored vectors,
graph algorithms over hundreds of thousands of edges. On the daemon's own
interpreter such work holds the interpreter lock in long stretches, and every
recall thread then waits up to the switch interval each time one of its
database reads hands the lock back (measured: 60 facts by id 15 -> 518 ms
beside one busy thread). A child process shares nothing with the recalls.

The function and its arguments must be importable and picklable; the child
starts fresh (``spawn``), so it never inherits the daemon's threads or locks.

The child never outlives the call: on a timeout or an error it is stopped
before this returns, and a child whose parent died (a killed daemon) exits on
its own within about a second. A process pool could not promise the first: its
shutdown cancels only work that has not started, so a timed-out computation ran
on to the end (up to 15 minutes) with nobody waiting for it.
"""

from __future__ import annotations

import multiprocessing
import os
import threading
import time
from collections.abc import Callable
from typing import Any

NICE_INCREMENT = 10
#: How often a child checks that its parent is still alive.
PARENT_CHECK_SECONDS = 0.5
_STOP_WAIT_SECONDS = 2.0


def _lower_priority() -> None:
    try:
        os.nice(NICE_INCREMENT)
    except (AttributeError, OSError):  # Windows has no nice; priority is best effort
        pass


def _exit_with_parent(parent_pid: int) -> None:
    """In the child: exit at once when the process that started it is gone."""
    def _watch() -> None:
        while True:
            time.sleep(PARENT_CHECK_SECONDS)
            if os.getppid() != parent_pid:
                os._exit(1)

    threading.Thread(target=_watch, daemon=True, name="parent-watch").start()


def _child_main(conn: Any, parent_pid: int, fn: Callable[..., Any], args: tuple) -> None:
    _lower_priority()
    _exit_with_parent(parent_pid)
    try:
        result = (True, fn(*args))
    except BaseException as exc:  # noqa: BLE001 -- handed to the caller to raise
        result = (False, exc)
    try:
        conn.send(result)
    except Exception:  # noqa: BLE001 -- an unpicklable error is reported by name
        conn.send((False, RuntimeError(f"{type(result[1]).__name__}: {result[1]}")))
    finally:
        conn.close()


def _stop(proc: Any) -> None:
    if proc.is_alive():
        proc.terminate()
        proc.join(_STOP_WAIT_SECONDS)
    if proc.is_alive():
        proc.kill()
    proc.join(_STOP_WAIT_SECONDS)


#: Memory limits (MB) the daemon's watchdog applies to particular children, by PID.
#: A ``spawn`` child's command line looks like every other worker's, so a child that
#: needs more than the default (the graph-metrics helper on a large graph, #304)
#: carries its own limit here for as long as it runs.
_MEMORY_LIMITS: dict[int, int] = {}
_LIMITS_LOCK = threading.Lock()


def memory_limits() -> dict[int, int]:
    """A copy of the PID -> MB limits of the children running now."""
    with _LIMITS_LOCK:
        return dict(_MEMORY_LIMITS)


def run_in_child(fn: Callable[..., Any], *args: Any, timeout: float,
                 memory_limit_mb: int | None = None) -> Any:
    """``fn(*args)`` in a fresh child process. Raises what it raised, or TimeoutError.

    ``memory_limit_mb`` replaces the watchdog's per-worker limit for this child.
    """
    ctx = multiprocessing.get_context("spawn")
    receive, send = ctx.Pipe(duplex=False)
    proc = ctx.Process(target=_child_main, args=(send, os.getpid(), fn, args), daemon=True)
    proc.start()
    send.close()
    if memory_limit_mb and proc.pid:
        with _LIMITS_LOCK:
            _MEMORY_LIMITS[proc.pid] = int(memory_limit_mb)
    try:
        if not receive.poll(timeout):
            raise TimeoutError(f"background computation exceeded {timeout:.0f}s")
        try:
            ok, value = receive.recv()
        except EOFError as exc:
            raise RuntimeError("background process ended without a result") from exc
    finally:
        receive.close()
        _stop(proc)
        with _LIMITS_LOCK:
            _MEMORY_LIMITS.pop(proc.pid, None)
    if ok:
        return value
    raise value


__all__ = ["NICE_INCREMENT", "PARENT_CHECK_SECONDS", "memory_limits", "run_in_child"]
