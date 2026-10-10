"""The scan service: idle while the feature is off, working through jobs when on."""

from __future__ import annotations

import threading
import time

from superlocalmemory import sources
from superlocalmemory.sources.scanner import SourceScanService


def threads_named():
    return [t for t in threading.enumerate() if t.name == "slm-source-scan"]


def test_feature_off_means_no_thread_and_no_media_db(env):
    service = SourceScanService(env.host, poll_s=0.01)
    service.start()
    service.wake()
    assert threads_named() == []
    assert not (env.data / "media.db").exists() and not (env.data / "features.json").exists()
    assert service.stop(1.0) is True
    assert service.health()["state"] == "stopped"


def wait_for(cond, seconds=10.0):
    end = time.time() + seconds
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.05)
    return False


def test_feature_on_runs_a_queued_scan(env):
    env.write("a.md", "hello")
    sid = env.add_and_confirm()
    service = SourceScanService(env.host, poll_s=0.01, interval_s=3600)
    env.host.sleep = lambda s: None
    service.start()
    try:
        assert wait_for(lambda: len(env.runtime.saved) == 1)
        assert wait_for(lambda: _job_state(env) == "done")
        assert service.health()["state"] == "running"
    finally:
        assert service.stop(5.0) is True
    assert threads_named() == []


def _job_state(env):
    media = env.store()
    try:
        jobs = media.list_jobs("default")
        return jobs[0]["state"] if jobs else None
    finally:
        media.close()


def test_job_progress_is_recorded(env):
    for i in range(3):
        env.write(f"n{i}.md", f"note {i}")
    env.add_and_confirm()
    service = SourceScanService(env.host, poll_s=0.01, interval_s=3600)
    env.host.sleep = lambda s: None
    service.start()
    try:
        assert wait_for(lambda: _job_state(env) == "done")
        media = env.store()
        try:
            job = media.list_jobs("default")[0]
            assert job["done"] == job["total"] == 3
        finally:
            media.close()
    finally:
        service.stop(5.0)


def test_due_sources_are_rescanned_on_the_interval(env):
    env.write("a.md", "hello")
    sid = env.add_and_confirm()
    service = SourceScanService(env.host, poll_s=0.01, interval_s=0.0)
    env.host.sleep = lambda s: None
    service.start()
    try:
        assert wait_for(lambda: _job_count(env) >= 2)
    finally:
        service.stop(5.0)


def _job_count(env):
    media = env.store()
    try:
        return len(media.list_jobs("default"))
    finally:
        media.close()
