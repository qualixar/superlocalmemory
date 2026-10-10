"""A source paused for remote access does not keep the scanner busy."""

from __future__ import annotations

from superlocalmemory.sources.scanner import SourceScanService
from superlocalmemory.sources.store import SourceStore


def job_count(env):
    media = env.store()
    try:
        return len(media.list_jobs("default", ["queued", "running", "done", "cancelled", "failed"]))
    finally:
        media.close()


def test_paused_source_is_not_queued_again_while_remote_is_on(env):
    env.write("n.md", "a note")
    sid = env.add_and_confirm()
    env.remote = True
    svc = SourceScanService(env.host, poll_s=2.0, interval_s=0.0)
    results = [svc._pass() for _ in range(6)]
    assert results[0] is False and not any(results)  # the loop sleeps instead of spinning
    assert job_count(env) <= 1
    media = env.store()
    try:
        source = SourceStore(media).get_source(sid)
    finally:
        media.close()
    assert source["state"] == "paused" and source["last_scan_at"]
    assert env.runtime.saved == []


def test_paused_source_resumes_when_remote_is_off_again(env):
    env.write("n.md", "a note")
    sid = env.add_and_confirm()
    env.remote = True
    svc = SourceScanService(env.host, poll_s=2.0, interval_s=0.0)
    svc._pass()
    env.remote = False
    svc._pass()
    svc._pass()
    assert env.runtime.contents() == ["a note"]
    assert env.files(sid)["n.md"]["state"] == "indexed"
