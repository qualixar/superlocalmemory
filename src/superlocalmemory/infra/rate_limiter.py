# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com
"""Sliding-window rate limiter for per-agent request throttling.

Pure stdlib -- no external dependencies.  Thread-safe.

Defaults (configurable via env vars):
    SLM_RATE_LIMIT_WRITE  = 100 req / window
    SLM_RATE_LIMIT_READ   = 300 req / window
    SLM_RATE_LIMIT_WINDOW = 60  seconds
"""

import logging
import os
import threading
import time
from collections import defaultdict
from typing import Dict, List, Tuple

logger = logging.getLogger("superlocalmemory.ratelimit")

# ---------------------------------------------------------------------------
# Module-level defaults (overridable via environment)
# ---------------------------------------------------------------------------
WRITE_LIMIT = int(os.environ.get("SLM_RATE_LIMIT_WRITE", "100"))
READ_LIMIT = int(os.environ.get("SLM_RATE_LIMIT_READ", "300"))
WINDOW_SECONDS = int(os.environ.get("SLM_RATE_LIMIT_WINDOW", "60"))


class RateLimiter:
    """Thread-safe sliding-window rate limiter.

    Each *client_id* (agent name, IP, etc.) gets its own independent
    request window.  Expired timestamps are pruned lazily on every call
    to ``allow()`` or ``is_allowed()``.

    Args:
        max_requests: Maximum requests allowed per window.
        window_seconds: Length of the sliding window in seconds.
    """

    def __init__(
        self,
        max_requests: int = 100,
        window_seconds: int = 60,
    ) -> None:
        self.max_requests = max_requests
        self.window = window_seconds
        self._requests: Dict[str, List[float]] = defaultdict(list)
        self._lock = threading.Lock()

    # ----- public API -----

    def allow(self, client_id: str) -> bool:
        """Check **and record** a request for *client_id*.

        Returns ``True`` when the request is allowed, ``False`` when the
        client has exceeded its limit for the current window.
        """
        allowed, _ = self.is_allowed(client_id)
        return allowed

    def is_allowed(self, client_id: str) -> Tuple[bool, int]:
        """Check and record a request.

        Returns:
            ``(allowed, remaining)`` -- whether the request is permitted
            and how many requests remain in the current window.
        """
        now = time.time()
        cutoff = now - self.window

        with self._lock:
            # Prune expired timestamps
            self._requests[client_id] = [
                t for t in self._requests[client_id] if t > cutoff
            ]

            current = len(self._requests[client_id])

            if current >= self.max_requests:
                return False, 0

            self._requests[client_id].append(now)
            return True, self.max_requests - current - 1

    def remaining(self, client_id: str) -> int:
        """Return how many requests *client_id* has left without recording one."""
        now = time.time()
        cutoff = now - self.window

        with self._lock:
            active = [t for t in self._requests.get(client_id, []) if t > cutoff]
            return max(0, self.max_requests - len(active))

    def reset(self, client_id: str) -> None:
        """Clear all recorded requests for *client_id*."""
        with self._lock:
            self._requests.pop(client_id, None)

    def cleanup(self) -> int:
        """Remove stale entries for clients with no recent requests.

        Returns:
            Number of client entries removed.
        """
        now = time.time()
        cutoff = now - self.window * 2  # keep 2 windows of data

        with self._lock:
            stale = [
                k
                for k, v in self._requests.items()
                if not v or max(v) < cutoff
            ]
            for k in stale:
                del self._requests[k]
            return len(stale)

    def configure(
        self,
        max_requests: int | None = None,
        window_seconds: int | None = None,
    ) -> None:
        """Reconfigure limits at runtime (thread-safe).

        Recorded request timestamps are preserved; only the ceilings change,
        so a raised limit takes effect immediately for the current window.
        """
        with self._lock:
            if max_requests is not None:
                self.max_requests = max(1, int(max_requests))
            if window_seconds is not None:
                self.window = max(1, int(window_seconds))

    def get_stats(self) -> dict:
        """Return a snapshot of limiter state."""
        with self._lock:
            return {
                "max_requests": self.max_requests,
                "window_seconds": self.window,
                "tracked_clients": len(self._requests),
            }


# ---------------------------------------------------------------------------
# Module-level convenience singletons
# ---------------------------------------------------------------------------
write_limiter = RateLimiter(max_requests=WRITE_LIMIT, window_seconds=WINDOW_SECONDS)
read_limiter = RateLimiter(max_requests=READ_LIMIT, window_seconds=WINDOW_SECONDS)


# ---------------------------------------------------------------------------
# Runtime-configurable limits (task #47) — dashboard-editable thresholds.
#
# The enforcement middleware builds its own limiter instances; it registers
# them here by role so a single set_limits() call reconfigures every live
# limiter at once (no restart). Loopback limiters derive from write/read,
# matching the startup derivation in unified_daemon.
# ---------------------------------------------------------------------------

_MANAGED: List[Tuple[str, "RateLimiter"]] = []
_MANAGED_LOCK = threading.Lock()
_CURRENT: Dict[str, int] = {
    "write": WRITE_LIMIT, "read": READ_LIMIT, "window": WINDOW_SECONDS,
}


def _loopback_write(write: int) -> int:
    return max(300, int(write) * 10)


def _loopback_read(read: int) -> int:
    return max(2000, int(read) * 20)


def register_managed(role: str, limiter: "RateLimiter") -> None:
    """Register an enforcement limiter so set_limits() can reconfigure it.

    role is one of: 'write', 'read', 'lb_write', 'lb_read'.
    """
    with _MANAGED_LOCK:
        _MANAGED.append((role, limiter))


def reset_managed() -> None:
    """Drop all registered limiters (app teardown / test isolation)."""
    with _MANAGED_LOCK:
        _MANAGED.clear()


def get_limits() -> Dict[str, int]:
    """Return the current effective write/read/window limits."""
    with _MANAGED_LOCK:
        return dict(_CURRENT)


def set_limits(
    write: int | None = None,
    read: int | None = None,
    window: int | None = None,
) -> Dict[str, int]:
    """Update limits and reconfigure every registered limiter live.

    Partial updates are supported (pass only what changes). Loopback
    limiters re-derive from the new write/read. Returns the new effective
    limits. Values are floored at 1.
    """
    with _MANAGED_LOCK:
        if write is not None:
            _CURRENT["write"] = max(1, int(write))
        if read is not None:
            _CURRENT["read"] = max(1, int(read))
        if window is not None:
            _CURRENT["window"] = max(1, int(window))
        w, r, win = _CURRENT["write"], _CURRENT["read"], _CURRENT["window"]
        lb_w, lb_r = _loopback_write(w), _loopback_read(r)
        for role, limiter in _MANAGED:
            if role == "write":
                limiter.configure(max_requests=w, window_seconds=win)
            elif role == "read":
                limiter.configure(max_requests=r, window_seconds=win)
            elif role == "lb_write":
                limiter.configure(max_requests=lb_w, window_seconds=win)
            elif role == "lb_read":
                limiter.configure(max_requests=lb_r, window_seconds=win)
        return dict(_CURRENT)
