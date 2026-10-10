# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""The document job service: takes queued document jobs one at a time.

Registered with the daemon's service registry. While images and documents are off
it is idle: no thread, no database, no process. Turning them on (or saving a
document) wakes it. ``stop`` asks the running job to stop at the next page; the
job goes back to the queue and resumes at the first page not yet recorded.
"""

from __future__ import annotations

import logging
import os
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from superlocalmemory.documents.pipeline import JobContext, JobRunner

logger = logging.getLogger(__name__)

SERVICE_NAME = "document-jobs"
PARSE_SCRIPT = Path(__file__).resolve().parents[1] / "runtimes" / "pdf_parse.py"
_ACTIVE: "DocumentJobService | None" = None


def _env_number(name: str, default: float) -> float:
    try:
        return float(os.environ[name])
    except (KeyError, ValueError):
        return default


@dataclass(frozen=True)
class Limits:
    page_timeout_s: float = 30.0
    job_timeout_s: float = 1200.0
    rss_limit_mb: int = 1600
    max_pages: int = 500
    lease_s: float = 300.0
    poll_s: float = 1.0
    idle_s: float = 5.0

    @classmethod
    def from_env(cls) -> "Limits":
        return cls(max_pages=int(_env_number("SLM_DOC_MAX_PAGES", 500)),
                   rss_limit_mb=int(_env_number("SLM_DOC_PARSE_RSS_MB", 1600)))


@dataclass(frozen=True)
class RunnerDeps:
    store_factory: Callable[[], Any]
    client_supplier: Callable[[], Any]
    runtime_supplier: Callable[[], Any]
    config_supplier: Callable[[], Any]
    python_supplier: Callable[[], Path | None]
    enabled: Callable[[], bool]
    script: Path = PARSE_SCRIPT
    limits: Limits = Limits()
    owns_store: bool = True


class DocumentJobService:
    name = SERVICE_NAME

    def __init__(self, deps: RunnerDeps) -> None:
        self._deps = deps
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._closed = False
        self._store: Any = None
        self._failure = ""
        self._owner = f"{SERVICE_NAME}:{os.getpid()}:{uuid.uuid4().hex[:8]}"

    # -- registry protocol --------------------------------------------------------
    def start(self) -> None:
        self._closed = False
        self._ensure_thread()

    def wake(self) -> None:
        """Called when work may have arrived; starts the thread only when the feature is on."""
        if not self._closed:
            self._ensure_thread()
        self._wake.set()

    def stop(self, timeout_s: float) -> bool:
        self._closed = True
        self._stop.set()
        self._wake.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=max(0.0, timeout_s))
            if thread.is_alive():
                return False
        self._release_store()
        return True

    def health(self) -> dict:
        thread = self._thread
        if thread is not None and thread.is_alive():
            return {"state": "running", "detail": ""}
        return {"state": "failed" if self._failure else "stopped", "detail": self._failure}

    # -- test and shutdown hooks ----------------------------------------------------
    def request_stop(self) -> None:
        """Ask the running job to stop at its next page (``stop`` also joins the thread)."""
        self._stop.set()

    def clear_stop(self) -> None:
        self._stop.clear()

    # -- the loop -------------------------------------------------------------------
    def _ensure_thread(self) -> None:
        with self._lock:
            if (self._thread is not None and self._thread.is_alive()) or not self._deps.enabled():
                return
            self._stop.clear()
            self._thread = threading.Thread(target=self._loop, name=SERVICE_NAME, daemon=True)
            self._thread.start()

    def _loop(self) -> None:
        while not self._stop.is_set():
            worked = False
            try:
                if not self._deps.enabled():
                    break
                worked = self.process_next()
                self._failure = ""
            except Exception as exc:  # noqa: BLE001 - the service keeps running
                self._failure = type(exc).__name__
                logger.warning("document job service error (%s)", self._failure)
            if not worked:
                self._wake.wait(timeout=self._deps.limits.idle_s)
                self._wake.clear()

    def _store_ref(self) -> Any:
        if self._store is None:
            self._store = self._deps.store_factory()
        return self._store

    def _release_store(self) -> None:
        store, self._store = self._store, None
        if store is not None and self._deps.owns_store:
            store.close()

    def process_next(self) -> bool:
        """Run one queued job to its end (or to a stop); False when there was nothing to run."""
        runtime, config = self._deps.runtime_supplier(), self._deps.config_supplier()
        store = self._store_ref()
        if runtime is None or config is None or store is None:
            return False
        job = store.claim_job(self._owner, self._deps.limits.lease_s, kinds=("document",))
        if job is None:
            return False
        client, python = self._deps.client_supplier(), self._deps.python_supplier()
        if client is None or python is None:
            store.release_job(job["job_id"], self._owner)
            return False
        ctx = JobContext(store=store, client=client, runtime=runtime, config=config, python=python,
                         script=self._deps.script, limits=self._deps.limits, owner=self._owner,
                         should_stop=self._stop.is_set)
        JobRunner(ctx, job).run()
        return True


# -- daemon wiring ---------------------------------------------------------------------

def default_service(application: Any) -> DocumentJobService:
    """The service wired to the running daemon; ``application`` only needs a ``state`` attribute."""
    from superlocalmemory.media import open_media_store
    from superlocalmemory.runtimes.features import media_enabled
    from superlocalmemory.runtimes.media_env import media_env
    from superlocalmemory.runtimes.worker_client import media_embedder

    state = application.state

    def runtime() -> Any:
        found = getattr(state, "canonical_remember_runtime", None)
        return found if found is not None and getattr(found, "ready", False) else None

    def python() -> Path | None:
        env = media_env()
        return Path(env.python()) if env.status().state == "ready" else None

    return DocumentJobService(RunnerDeps(
        store_factory=open_media_store, client_supplier=media_embedder, runtime_supplier=runtime,
        config_supplier=lambda: getattr(getattr(state, "engine", None), "_config", None),
        python_supplier=python, enabled=media_enabled, limits=Limits.from_env()))


def wake_active() -> None:
    """Nudge the registered service after a document is queued; does nothing before registration."""
    service = _ACTIVE
    if service is not None:
        service.wake()


def start_document_jobs(application: Any, registry: Any) -> None:
    """Register (once) and start the service; it stays idle while the feature is off."""
    global _ACTIVE
    if registry.get(SERVICE_NAME) is None:
        registry.register(default_service(application))
    _ACTIVE = registry.get(SERVICE_NAME)
    registry.start(SERVICE_NAME)


def stop_document_jobs(registry: Any, timeout_s: float = 5.0) -> bool:
    global _ACTIVE
    _ACTIVE = None
    return registry.stop(SERVICE_NAME, timeout_s)
