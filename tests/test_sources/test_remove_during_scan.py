"""Removing a folder while it is being scanned: it stays removed and nothing more is saved."""

from __future__ import annotations

import json
import threading
import time

from superlocalmemory import sources
from superlocalmemory.sources import locks
from superlocalmemory.sources.reconcile import scan_source
from superlocalmemory.sources.scanner import SourceScanService
from superlocalmemory.sources.store import SourceStore


def test_remove_during_a_scan_stays_removed_and_saves_nothing_more(env):
    for i in range(3):
        env.write(f"n{i}.md", f"note {i}")
    sid = env.add_and_confirm()
    media = env.store()
    store = SourceStore(media)
    calls = []

    def progress(done, total):
        if not calls:
            calls.append(1)
            sources.remove_source(sid)

    try:
        stats = scan_source(env.host, store, store.get_source(sid), progress=progress)
        state = store.get_source(sid)["state"]
        rows = {r["relpath"]: r["state"] for r in store.files(sid)}
    finally:
        media.close()
    assert state == "removed" and stats.removed is True
    assert len(env.runtime.saved) == 1
    assert set(rows.values()) <= {"tombstoned"}


def test_a_removed_source_cannot_be_set_back(env):
    sid = env.add_and_confirm()
    sources.remove_source(sid)
    media = env.store()
    try:
        store = SourceStore(media)
        store.set_state(sid, "active", scanned=True)
        store.set_state(sid, "paused")
        assert store.get_source(sid)["state"] == "removed"
    finally:
        media.close()


def test_removing_cancels_the_queued_scan(env):
    env.write("a.md", "x")
    sid = env.add_and_confirm()
    sources.remove_source(sid)
    media = env.store()
    try:
        jobs = media.list_jobs("default", ["queued", "running", "cancelled"])
    finally:
        media.close()
    assert [j["state"] for j in jobs if json.loads(j["payload_json"])["source_id"] == sid] == ["cancelled"]


def test_remove_waits_for_the_scan_that_holds_the_source(env):
    env.write("a.md", "x")
    sid = env.add_and_confirm()
    done = threading.Event()
    lock = locks.source_lock(sid)
    lock.acquire()
    worker = threading.Thread(target=lambda: (sources.remove_source(sid), done.set()))
    try:
        worker.start()
        time.sleep(0.3)
        assert not done.is_set()
        assert locks.is_removing(sid)  # the running scan is told to stop
    finally:
        lock.release()
    worker.join(5)
    assert done.is_set() and not locks.is_removing(sid)


def test_a_queued_scan_of_a_removed_source_is_cancelled(env):
    env.write("a.md", "x")
    sid = env.add_and_confirm()
    media = env.store()
    try:
        store = SourceStore(media)
        store.delete_source_rows(sid)  # removed behind the queue's back
        svc = SourceScanService(env.host)
        svc._queue_due(store)
        job = media.claim_job(svc._owner, 60, kinds=("source_scan",))
        assert svc._run_job(store, media, job) is True
        assert media.get_job(job["job_id"])["state"] == "cancelled"
    finally:
        media.close()
    assert env.runtime.saved == []
