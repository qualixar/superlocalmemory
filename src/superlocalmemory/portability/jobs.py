# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""The daemon's background threads for imports and rollbacks: start, stop, ask if one is alive.

A job is one thread per run, started by a request and owned by the daemon process. The run's
progress lives in ``import_runs`` (durable); this registry only knows which runs have a live
thread, so a run that says "running" with no thread behind it (the daemon was restarted) can be
told apart and closed. Threads are daemon threads: they never keep the daemon from exiting.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable

logger = logging.getLogger(__name__)


class Jobs:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._live: dict[str, tuple[threading.Thread, threading.Event]] = {}

    def start(self, run_id: str, target: Callable[[threading.Event], object]) -> None:
        stop = threading.Event()

        def body() -> None:
            try:
                target(stop)
            except Exception:  # noqa: BLE001 - the job reports its own end; never kill the daemon
                logger.exception("import job %s ended with an error", run_id)
            finally:
                with self._lock:
                    self._live.pop(run_id, None)

        thread = threading.Thread(target=body, name=f"slm-import-{run_id[-8:]}", daemon=True)
        with self._lock:
            if run_id in self._live:
                raise RuntimeError("that run already has a job")
            self._live[run_id] = (thread, stop)
        thread.start()

    def is_live(self, run_id: str) -> bool:
        with self._lock:
            found = self._live.get(run_id)
        return bool(found and found[0].is_alive())

    def stop(self, run_id: str) -> None:
        with self._lock:
            found = self._live.get(run_id)
        if found:
            found[1].set()

    def join(self, run_id: str, timeout: float) -> bool:
        """True when the job has ended (or never ran) within ``timeout`` seconds."""
        with self._lock:
            found = self._live.get(run_id)
        if not found:
            return True
        found[0].join(max(0.0, timeout))
        return not found[0].is_alive()

    def stop_all(self, timeout: float = 5.0) -> None:
        with self._lock:
            live = list(self._live.values())
        for _thread, stop in live:
            stop.set()
        for thread, _stop in live:
            thread.join(timeout)


#: The daemon's one registry.
JOBS = Jobs()

__all__ = ["JOBS", "Jobs"]
