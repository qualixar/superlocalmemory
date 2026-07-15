# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Cross-platform utilities for subprocess management and resource monitoring.

V3.4.24: Consolidates Windows/POSIX branching from 10+ files into one module.
Replaces the Unix-only ``resource`` module with ``psutil`` on Windows.
Inspired by community PR #14 (GuillaumeG / Tyrin451).
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading


def popen_platform_kwargs() -> dict:
    """Platform-appropriate kwargs for subprocess.Popen.

    POSIX: ``start_new_session=True`` — prevents terminal signals bleeding.
    Windows: ``CREATE_NO_WINDOW`` — prevents console window popup.
    """
    if sys.platform == "win32":
        # CREATE_NO_WINDOW = 0x08000000 — only defined on Windows.
        flag = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
        return {"creationflags": flag}
    return {"start_new_session": True}


def get_rss_mb() -> float:
    """Current process RSS in megabytes.

    POSIX: ``resource.getrusage`` (stdlib). Windows: ``psutil``.
    Returns 0.0 if measurement is unavailable.
    """
    if sys.platform != "win32":
        try:
            import resource
            ru_maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            if sys.platform == "darwin":
                return ru_maxrss / 1024 / 1024  # macOS: bytes
            return ru_maxrss / 1024  # Linux: kilobytes
        except Exception:
            return 0.0
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1024 / 1024
    except Exception:
        return 0.0


def is_pid_alive(pid: int) -> bool:
    """Check whether a process with *pid* is alive.

    POSIX: ``os.kill(pid, 0)`` — signal 0 checks existence.
    Windows: ``psutil.pid_exists()`` with ``os.kill`` fallback.
    """
    if pid <= 0:
        return False
    if sys.platform != "win32":
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False  # ESRCH — no such process
        except PermissionError:
            return True   # EPERM — process EXISTS, we just can't signal it
        except OSError:
            return False
    try:
        import psutil
        return psutil.pid_exists(pid)
    except ImportError:
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False  # ESRCH — no such process
        except PermissionError:
            return True   # EPERM — process EXISTS, we just can't signal it
        except OSError:
            return False


def kill_process(pid: int) -> bool:
    """Send SIGTERM (POSIX) or taskkill /F /T (Windows).

    Returns True if the signal was sent successfully.
    """
    if pid <= 0:
        return False
    if sys.platform == "win32":
        try:
            subprocess.call(
                ["taskkill", "/F", "/T", "/PID", str(pid)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return True
        except Exception:
            return False
    try:
        import signal
        os.kill(pid, signal.SIGTERM)
        return True
    except OSError:
        return False


def start_parent_watchdog(
    *, stop_event: threading.Event | None = None,
) -> threading.Thread | None:
    """Self-terminate when the parent process dies.

    Prevents orphaned workers (500+ MB each) after parent crash/kill.
    V3.3.7 origin: 33 GB consumed by orphaned workers.
    V3.4.24: Consolidated from 3 separate worker files.
    """
    try:
        parent_pid = os.getppid()
    except AttributeError:
        return
    if parent_pid <= 1:
        return

    stop = stop_event or threading.Event()

    def _watch() -> None:
        while not stop.wait(5):
            if not is_pid_alive(parent_pid):
                os._exit(0)

    t = threading.Thread(target=_watch, daemon=True, name="parent-watchdog")
    t.start()
    return t
