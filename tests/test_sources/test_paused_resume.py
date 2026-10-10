"""A paused folder resumes on the next pass once remote access is off."""

from __future__ import annotations

from superlocalmemory.sources.scanner import SourceScanService
from superlocalmemory.sources.store import SourceStore


def queued(env) -> int:
    media = env.store()
    try:
        return len(media.list_jobs("default", ["queued"]))
    finally:
        media.close()


def due(env, svc) -> None:
    media = env.store()
    try:
        svc._queue_due(SourceStore(media))
    finally:
        media.close()


def state(env, sid) -> dict:
    media = env.store()
    try:
        return SourceStore(media).get_source(sid)
    finally:
        media.close()


def paused_source(env):
    env.write("n.md", "a note")
    sid = env.add_and_confirm()
    env.remote = True
    svc = SourceScanService(env.host, poll_s=2.0, interval_s=900.0)
    while svc._pass():  # drain the first scan, which pauses it a moment ago
        pass
    assert state(env, sid)["state"] == "paused" and queued(env) == 0
    return sid, svc


def test_paused_source_is_queued_at_once_when_remote_is_off(env):
    sid, svc = paused_source(env)
    env.remote = False
    due(env, svc)
    assert queued(env) == 1
    due(env, svc)
    assert queued(env) == 1  # no duplicate


def test_paused_source_is_not_queued_while_remote_is_on(env):
    sid, svc = paused_source(env)
    due(env, svc)
    assert queued(env) == 0


def test_recently_scanned_active_source_is_not_queued(env):
    env.write("n.md", "a note")
    sid = env.add_and_confirm()
    svc = SourceScanService(env.host, poll_s=2.0, interval_s=900.0)
    env.scan(sid)
    assert state(env, sid)["state"] == "active"
    before = queued(env)
    due(env, svc)
    assert queued(env) == before


def test_queued_resume_scan_makes_the_source_active(env):
    sid, svc = paused_source(env)
    env.remote = False
    svc._pass()
    assert state(env, sid)["state"] == "active"
