# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""A registry for the background services that run inside the daemon.

Each service is a small object with a name, ``start``, ``stop`` and ``health``.
The registry starts them in dependency order, stops them in reverse, never lets
one failing stop prevent the rest from stopping, and reports a health snapshot.
"""

from __future__ import annotations

import logging
import threading
from typing import Protocol

logger = logging.getLogger(__name__)


class BackgroundService(Protocol):
    """What the registry needs from a service."""

    name: str

    def start(self) -> None: ...

    def stop(self, timeout_s: float) -> bool:
        """Return True when the service stopped cleanly."""
        ...

    def health(self) -> dict:
        """``{"state": "running" | "stopped" | "failed", "detail": str}``."""
        ...


class ServiceRegistry:
    """Thread-safe registry of named background services."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._services: dict[str, BackgroundService] = {}
        self._after: dict[str, tuple[str, ...]] = {}
        self._started: list[str] = []

    def register(
        self, service: BackgroundService, *, after: tuple[str, ...] = (),
    ) -> None:
        """Add a service; names are unique. ``after`` is checked at start_all."""
        with self._lock:
            if service.name in self._services:
                raise ValueError(f"service already registered: {service.name}")
            self._services[service.name] = service
            self._after[service.name] = tuple(after)

    def unregister(self, name: str) -> None:
        """Forget a service (it is not stopped); unknown names are ignored."""
        with self._lock:
            self._services.pop(name, None)
            self._after.pop(name, None)
            if name in self._started:
                self._started.remove(name)

    def get(self, name: str) -> BackgroundService | None:
        with self._lock:
            return self._services.get(name)

    def start(self, name: str) -> None:
        """Start one registered service."""
        with self._lock:
            service = self._services[name]
            if name not in self._started:
                self._started.append(name)
        service.start()

    def stop(self, name: str, timeout_s: float) -> bool:
        """Stop one service; an unknown name has nothing to stop (True)."""
        with self._lock:
            service = self._services.get(name)
            if name in self._started:
                self._started.remove(name)
        if service is None:
            return True
        return self._stop_safely(service, timeout_s)

    @staticmethod
    def _stop_safely(service: BackgroundService, timeout_s: float) -> bool:
        try:
            return bool(service.stop(timeout_s))
        except Exception as exc:  # noqa: BLE001 - one failure must not block the rest
            logger.warning("service %s failed to stop: %s", service.name, exc)
            return False

    def _start_order(self) -> list[str]:
        """Topological order by ``after``; raises ValueError on a cycle or gap."""
        with self._lock:
            after = dict(self._after)
        for name, deps in after.items():
            for dep in deps:
                if dep not in after:
                    raise ValueError(f"service {name} runs after unknown service {dep}")
        order: list[str] = []
        state: dict[str, int] = {}

        def visit(name: str) -> None:
            if state.get(name) == 2:
                return
            if state.get(name) == 1:
                raise ValueError(f"service dependency cycle at {name}")
            state[name] = 1
            for dep in after[name]:
                visit(dep)
            state[name] = 2
            order.append(name)

        for name in after:
            visit(name)
        return order

    def start_all(self) -> None:
        """Start every service, dependencies first. Validates before starting."""
        for name in self._start_order():
            self.start(name)

    def stop_all(self, timeout_s: float = 10.0) -> dict[str, bool]:
        """Stop in reverse start order; services never started stop last."""
        with self._lock:
            started, self._started = list(self._started), []
            services = dict(self._services)
        names = list(reversed(started))
        names += [n for n in reversed(list(services)) if n not in started]
        return {
            name: self._stop_safely(services[name], timeout_s)
            for name in names if name in services
        }

    def snapshot(self) -> dict[str, dict]:
        """Each service's health; a raising ``health`` reads as failed."""
        with self._lock:
            services = dict(self._services)
        result: dict[str, dict] = {}
        for name, service in services.items():
            try:
                result[name] = dict(service.health())
            except Exception as exc:  # noqa: BLE001
                result[name] = {"state": "failed", "detail": repr(exc)}
        return result
