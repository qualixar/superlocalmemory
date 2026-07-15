# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Layered token-bucket rate limiter.

Part of Qualixar | Author: Varun Pratap Bhardwaj
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

from superlocalmemory.core.error_envelope import ErrorCode


class RateLimitedError(Exception):
    def __init__(
        self,
        layer: str,
        retry_after_ms: int,
        message: str | None = None,
        **extras: object,
    ) -> None:
        self.layer = layer
        self.retry_after_ms = retry_after_ms
        self.code = ErrorCode.RATE_LIMITED
        self.extras = extras
        super().__init__(message or f"{layer} rate limit exceeded")


class TokenBucket:
    __slots__ = ("rate", "capacity", "tokens", "last_refill")

    def __init__(self, rate_per_sec: float, capacity: float | None = None) -> None:
        self.rate = float(rate_per_sec)
        self.capacity = float(capacity) if capacity is not None else float(rate_per_sec)
        self.tokens = self.capacity
        self.last_refill = time.monotonic()

    def _refill(self, now: float) -> None:
        delta = now - self.last_refill
        if delta > 0:
            self.tokens = min(self.capacity, self.tokens + delta * self.rate)
            self.last_refill = now

    def can_consume(self, cost: float = 1.0) -> bool:
        self._refill(time.monotonic())
        return self.tokens >= cost

    def try_consume(self, cost: float = 1.0) -> bool:
        now = time.monotonic()
        self._refill(now)
        if self.tokens >= cost:
            self.tokens -= cost
            return True
        return False

    def consume(self, cost: float = 1.0) -> None:
        self.tokens -= cost

    def ms_to_refill(self, cost: float = 1.0) -> int:
        self._refill(time.monotonic())
        needed = max(0.0, cost - self.tokens)
        if self.rate <= 0:
            return 1_000_000
        return int((needed / self.rate) * 1000) + 1


@dataclass
class _PidEntry:
    bucket: TokenBucket
    last_used: float = field(default_factory=time.monotonic)


class LayeredRateLimiter:
    def __init__(
        self,
        *,
        global_rps: float = 100.0,
        per_pid_rps: float = 30.0,
        per_agent_rps: float = 10.0,
        idle_ttl_s: float = 60.0,
    ) -> None:
        self._global = TokenBucket(global_rps)
        self._per_pid_rps = per_pid_rps
        self._per_agent_rps = per_agent_rps
        self._idle_ttl_s = idle_ttl_s
        self._per_pid: dict[int, _PidEntry] = {}
        self._per_agent: dict[str, _PidEntry] = {}
        self._lock = threading.RLock()

    def _sweep(self, now: float) -> None:
        cutoff = now - self._idle_ttl_s
        for k in [p for p, e in self._per_pid.items() if e.last_used < cutoff]:
            del self._per_pid[k]
        for k in [a for a, e in self._per_agent.items() if e.last_used < cutoff]:
            del self._per_agent[k]

    def _get_pid(self, pid: int, now: float) -> TokenBucket:
        entry = self._per_pid.get(pid)
        if entry is None:
            entry = _PidEntry(TokenBucket(self._per_pid_rps), now)
            self._per_pid[pid] = entry
        entry.last_used = now
        return entry.bucket

    def _get_agent(self, agent_id: str, now: float) -> TokenBucket:
        entry = self._per_agent.get(agent_id)
        if entry is None:
            entry = _PidEntry(TokenBucket(self._per_agent_rps), now)
            self._per_agent[agent_id] = entry
        entry.last_used = now
        return entry.bucket

    def check_and_consume(self, *, pid: int, agent_id: str | None = None) -> None:
        """Peek all layers; raise on any reject without touching others.

        On admission, consume one token from each applicable bucket.
        """
        with self._lock:
            now = time.monotonic()
            self._sweep(now)
            pid_bucket = self._get_pid(pid, now)
            agent_bucket = self._get_agent(agent_id, now) if agent_id else None
            if not self._global.can_consume():
                raise RateLimitedError(
                    "global", self._global.ms_to_refill(),
                )
            if not pid_bucket.can_consume():
                raise RateLimitedError(
                    "per-pid", pid_bucket.ms_to_refill(), pid=pid,
                )
            if agent_bucket is not None and not agent_bucket.can_consume():
                raise RateLimitedError(
                    "per-agent", agent_bucket.ms_to_refill(), agent=agent_id,
                )
            self._global.consume()
            pid_bucket.consume()
            if agent_bucket is not None:
                agent_bucket.consume()

    def per_pid_size(self) -> int:
        with self._lock:
            return len(self._per_pid)

    def per_agent_size(self) -> int:
        with self._lock:
            return len(self._per_agent)
