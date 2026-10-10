"""Folder calls wait for a running scan only so long, and a release cannot be lost to one."""

from __future__ import annotations

import threading

import pytest

from superlocalmemory import sources
from superlocalmemory.sources import api, locks
from superlocalmemory.sources.store import SourceStore

SECRET = "AKIA" + "ABCDEFGHIJKLMNOP"


@pytest.fixture(autouse=True)
def short_waits(monkeypatch):
    monkeypatch.setattr(api, "_REMOVE_WAIT_S", 0.1)
    monkeypatch.setattr(api, "_QUICK_WAIT_S", 0.1)


class Held:
    """Holds a source's lock on another thread until closed."""

    def __init__(self, sid):
        self.lock = locks.source_lock(sid)
        self.lock.acquire()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.lock.release()


def row(env, sid, relpath=None):
    media = env.store()
    try:
        store = SourceStore(media)
        return store.get_source(sid) if relpath is None else store.get_file(sid, relpath)
    finally:
        media.close()


def emptied(env):
    env.write("a.md", "x")
    sid = env.add_and_confirm()
    env.scan(sid)
    (env.root / "a.md").unlink()
    assert env.scan(sid).offline_reason == "empty_folder"
    return sid


def call_in_thread(fn, *args):
    box = {}

    def run():
        try:
            box["result"] = fn(*args)
        except Exception as exc:  # noqa: BLE001 - handed back to the test
            box["error"] = exc

    t = threading.Thread(target=run)
    t.start()
    t.join(5)
    assert not t.is_alive()
    return box


def test_held_times_out_and_releases():
    sid = "lock-wait-1"
    with Held(sid):
        with pytest.raises(locks.SourceBusy):
            with locks.held(sid, 0.05):
                pass
    with locks.held(sid, 0.05):
        assert locks.source_lock(sid).locked()
    assert not locks.source_lock(sid).locked()


def test_remove_gives_up_when_a_scan_holds_the_folder(env):
    env.write("a.md", "x")
    sid = env.add_and_confirm()
    env.scan(sid)
    with Held(sid):
        box = call_in_thread(sources.remove_source, sid)
    err = box["error"]
    assert isinstance(err, sources.SourceRefused) and err.code == "source_busy"
    assert not locks.is_removing(sid)
    assert row(env, sid)["state"] == "active" and env.runtime.archived == []
    assert {r["state"] for r in env.files(sid).values()} == {"indexed"}


def test_forget_empty_gives_up_when_a_scan_holds_the_folder(env):
    sid = emptied(env)
    with Held(sid):
        box = call_in_thread(sources.forget_empty, sid)
    assert box["error"].code == "source_busy"
    assert row(env, sid)["state"] == "offline" and env.runtime.archived == []
    assert {r["state"] for r in env.files(sid).values()} == {"indexed"}


def quarantined(env):
    env.write("keys.md", SECRET)
    sid = env.add_and_confirm()
    env.scan(sid)
    assert env.files(sid)["keys.md"]["state"] == "quarantined"
    return sid


def test_release_gives_up_when_a_scan_holds_the_folder(env):
    sid = quarantined(env)
    woken = env.woken
    with Held(sid):
        box = call_in_thread(sources.release_file, sid, "keys.md")
    assert box["error"].code == "source_busy"
    assert row(env, sid, "keys.md")["state"] == "quarantined"
    assert env.woken == woken


def test_release_after_the_lock_frees_wakes_the_scanner(env):
    sid = quarantined(env)
    woken = env.woken
    assert sources.release_file(sid, "keys.md") is True
    assert row(env, sid, "keys.md")["state"] == "pending" and env.woken == woken + 1


def test_release_sees_the_row_a_scan_wrote_while_it_waited(env, monkeypatch):
    sid = quarantined(env)
    monkeypatch.setattr(api, "_QUICK_WAIT_S", 5.0)
    held, reading = threading.Event(), threading.Event()
    real_get = SourceStore.get_file
    releaser = {}

    def spying_get(self, source_id, relpath):
        if threading.current_thread() is releaser.get("thread"):
            reading.set()
        return real_get(self, source_id, relpath)

    monkeypatch.setattr(SourceStore, "get_file", spying_get)

    def scan():
        with locks.held(sid, 5):
            held.set()
            reading.wait(0.3)  # without the lock the release reads here; with it, this just times out
            media = env.store()
            try:
                SourceStore(media).put_file(sid, "keys.md", state="quarantined", sha256="f" * 64)
            finally:
                media.close()

    scanner = threading.Thread(target=scan)
    scanner.start()
    assert held.wait(5)
    t = threading.Thread(target=lambda: releaser.update(result=sources.release_file(sid, "keys.md")))
    releaser["thread"] = t
    t.start()
    t.join(10)
    scanner.join(10)
    assert releaser["result"] is True
    assert row(env, sid, "keys.md")["reason"] == "released:" + "f" * 64
