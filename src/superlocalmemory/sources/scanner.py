# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""The background service that runs source scans. Idle, with no thread, while the feature is off."""

from __future__ import annotations

import json
import logging
import os
import threading
import uuid
from typing import Any, Callable

from superlocalmemory.media.store_jobs import utc_stamp
from superlocalmemory.sources import locks
from superlocalmemory.sources.host import SourceHost
from superlocalmemory.sources.reconcile import scan_source
from superlocalmemory.sources.store import SourceStore
from superlocalmemory.sources.folder_watch import DEBOUNCE_S, SourceWatcher

logger = logging.getLogger(__name__)

SERVICE_NAME = "source-scan"
THREAD_NAME = "slm-source-scan"
#: Reconciliation is the truth; a watcher only hints. Every source is looked at this often.
RESCAN_INTERVAL_S = 15 * 60.0
_LEASE_S = 120.0


class SourceScanService:
    name = SERVICE_NAME

    def __init__(self, host: SourceHost, *, poll_s: float = 2.0,
                 interval_s: float = RESCAN_INTERVAL_S, watch_debounce_s: float = DEBOUNCE_S,
                 observer_factory: Callable[[], Any] | None = None) -> None:
        self._host = host
        self._watch_args = (watch_debounce_s, observer_factory)
        self._watcher: SourceWatcher | None = None
        self._poll_s = poll_s
        self._interval_s = interval_s
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._closed = False
        self._failure = ""
        self._owner = f"{SERVICE_NAME}:{os.getpid()}:{uuid.uuid4().hex[:8]}"

    # -- registry protocol -----------------------------------------------------------
    def start(self) -> None:
        self._closed = False
        self._watcher = SourceWatcher(self._host, self._changed, debounce_s=self._watch_args[0],
                                      observer_factory=self._watch_args[1])
        self._ensure_thread()

    def wake(self) -> None:
        if not self._closed:
            self._ensure_thread()
        self._wake.set()

    def stop(self, timeout_s: float) -> bool:
        self._closed = True
        self._stop.set()
        self._wake.set()
        watcher_done = self._watcher.stop(timeout_s) if self._watcher is not None else True
        thread = self._thread
        if thread is not None:
            thread.join(timeout=max(0.0, timeout_s))
            return watcher_done and not thread.is_alive()
        return watcher_done

    def health(self) -> dict:
        thread = self._thread
        if thread is not None and thread.is_alive():
            return {"state": "running", "detail": ""}
        return {"state": "failed" if self._failure else "stopped", "detail": self._failure}

    # -- the loop ----------------------------------------------------------------------
    def _enabled(self) -> bool:
        from superlocalmemory.runtimes.features import sources_enabled

        return sources_enabled(self._host.data_root)

    def _ensure_thread(self) -> None:
        if not self._enabled():
            return
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop.clear()
            self._thread = threading.Thread(target=self._run, name=THREAD_NAME, daemon=True)
            self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            worked = False
            try:
                worked = self._pass()
                self._failure = ""
            except Exception as exc:  # noqa: BLE001 - the loop must survive
                self._failure = type(exc).__name__
                logger.warning("folder scan pass failed (%s)", self._failure)
            if not worked:
                self._wake.wait(self._poll_s)
                self._wake.clear()

    def _pass(self) -> bool:
        if not self._enabled():
            self._sync_watch([])
            self._stop.wait(self._poll_s)
            return False
        from superlocalmemory.media import open_media_store

        media = open_media_store(data_root=self._host.data_root)
        if media is None:
            return False
        try:
            store = SourceStore(media)
            self._queue_due(store)
            self._sync_watch([] if self._host.remote_on() else store.list_sources(states=("active",)))
            job = media.claim_job(self._owner, _LEASE_S, kinds=("source_scan",))
            return self._run_job(store, media, job) if job else False
        finally:
            media.close()

    def _sync_watch(self, active: list[dict[str, Any]]) -> None:
        if self._watcher is not None and not self._stop.is_set():
            self._watcher.sync(active)

    def _changed(self, source_id: str, relpaths: list[str] | None) -> None:
        """The watcher saw these files change: read just them, under the source's lock."""
        from superlocalmemory.media import open_media_store

        media = open_media_store(data_root=self._host.data_root)
        if media is None:
            return
        try:
            store = SourceStore(media)
            source = store.get_source(source_id)
            if source is None or source["state"] != "active":
                return
            if relpaths is None:  # too many changes at once: a full scan is cheaper than a long list
                store.queue_scan(source["profile_id"], source_id)
                self._wake.set()
                return
            with locks.source_lock(source_id):
                source = store.get_source(source_id)
                if source is not None and source["state"] == "active":
                    scan_source(self._host, store, source, only=frozenset(relpaths))
        finally:
            media.close()

    def _queue_due(self, store: SourceStore) -> None:
        cutoff = utc_stamp(-self._interval_s)
        remote = self._host.remote_on()  # one check per pass
        for source in store.list_sources(states=("active", "offline", "paused")):
            if source["state"] == "paused":
                if not remote:  # remote access is off: resume now, not after the interval
                    store.queue_scan(source["profile_id"], source["source_id"])
                continue
            last = source["last_scan_at"] or ""
            if last <= cutoff:
                store.queue_scan(source["profile_id"], source["source_id"])

    def _run_job(self, store: SourceStore, media: Any, job: dict[str, Any]) -> bool:
        source_id = json.loads(job["payload_json"]).get("source_id", "")
        with locks.source_lock(source_id):  # a removal waits for the scan, and the scan stops for it
            source = store.get_source(source_id)
            if source is None or source["state"] == "removed":
                media.finish_job(job["job_id"], self._owner, "cancelled")
                return True
            return self._scan(store, media, job, source)

    def _scan(self, store: SourceStore, media: Any, job: dict[str, Any], source: dict[str, Any]) -> bool:
        def progress(done: int, total: int) -> None:
            if self._stop.is_set():
                raise InterruptedError
            media.progress_job(job["job_id"], self._owner, done, total)
            media.renew_lease(job["job_id"], self._owner, _LEASE_S)

        try:
            stats = scan_source(self._host, store, source, progress=progress)
        except InterruptedError:
            media.release_job(job["job_id"], self._owner)
            return False
        if stats.waiting:
            media.release_job(job["job_id"], self._owner)
            self._stop.wait(self._poll_s)
            return False
        media.finish_job(job["job_id"], self._owner, "cancelled" if stats.removed else "done")
        return not stats.paused  # a paused source is not work: let the loop sleep


__all__ = ["RESCAN_INTERVAL_S", "SERVICE_NAME", "SourceScanService"]
