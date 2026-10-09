"""The folder watcher: a fake observer stands in for the filesystem."""

from __future__ import annotations

import errno
import threading
import time
from types import SimpleNamespace

from superlocalmemory.sources import folder_watch as watcher_mod
from superlocalmemory.sources.report import build_report
from superlocalmemory.sources.store import SourceStore
from superlocalmemory.sources.folder_watch import SourceWatcher


class FakeObserver:
    made: list = []

    def __init__(self, fail: Exception | None = None) -> None:
        self.fail = fail
        self.handler = None
        self.path = None
        self.started = self.stopped = self.joined = False
        FakeObserver.made.append(self)

    def schedule(self, handler, path, recursive=True):
        self.handler, self.path = handler, path

    def start(self):
        if self.fail:
            raise self.fail
        self.started = True

    def stop(self):
        self.stopped = True

    def join(self, timeout=None):
        self.joined = True

    def emit(self, rel, *, dest=None, directory=False):
        self.handler.dispatch(SimpleNamespace(
            src_path=str(self.root / rel), dest_path=str(self.root / dest) if dest else "",
            is_directory=directory))


def make(env, *, fail=None, debounce=0.05):
    FakeObserver.made = []
    calls: list = []
    w = SourceWatcher(env.host, lambda sid, rels: calls.append((sid, rels)), debounce_s=debounce,
                      observer_factory=lambda: FakeObserver(fail))
    return w, calls


def src(env, sid, state="active"):
    return {"source_id": sid, "root_path": str(env.root), "state": state, "profile_id": "default"}


def wait_for(cond, seconds=5.0):
    end = time.time() + seconds
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.01)
    return False


def test_event_leads_to_a_rescan_of_only_that_path(env):
    w, calls = make(env)
    w.sync([src(env, "s1")])
    obs = FakeObserver.made[0]
    obs.root = env.root
    assert obs.started and obs.path == str(env.root)
    obs.emit("a.md")
    assert wait_for(lambda: calls)
    assert calls == [("s1", ["a.md"])]
    assert w.stop(2.0)


def test_events_are_debounced_into_one_call(env):
    w, calls = make(env, debounce=0.2)
    w.sync([src(env, "s1")])
    obs = FakeObserver.made[0]
    obs.root = env.root
    for rel in ("a.md", "b.md", "a.md", "sub/c.md"):
        obs.emit(rel)
        time.sleep(0.02)
    assert calls == []
    assert wait_for(lambda: calls)
    time.sleep(0.4)
    assert len(calls) == 1 and calls[0][1] == ["a.md", "b.md", "sub/c.md"]
    w.stop(2.0)


def test_a_move_reports_both_paths_and_directories_are_ignored(env):
    w, calls = make(env)
    w.sync([src(env, "s1")])
    obs = FakeObserver.made[0]
    obs.root = env.root
    obs.emit("dir", directory=True)
    obs.emit("old.md", dest="new.md")
    assert wait_for(lambda: calls)
    assert calls[0][1] == ["new.md", "old.md"]
    w.stop(2.0)


def test_paths_outside_the_root_are_dropped(env, tmp_path):
    w, calls = make(env)
    w.sync([src(env, "s1")])
    obs = FakeObserver.made[0]
    obs.root = tmp_path
    obs.emit("elsewhere.md")
    time.sleep(0.3)
    assert calls == []
    w.stop(2.0)


def test_only_active_sources_have_an_observer(env):
    w, _ = make(env)
    w.sync([src(env, "a"), src(env, "p", "paused"), src(env, "o", "offline")])
    assert len(FakeObserver.made) == 1 and w.watching("a") and not w.watching("p")
    w.sync([src(env, "p", "paused")])  # a no longer active
    assert FakeObserver.made[0].stopped and FakeObserver.made[0].joined and not w.watching("a")
    w.stop(2.0)


def test_sync_twice_does_not_start_a_second_observer(env):
    w, _ = make(env)
    w.sync([src(env, "a")])
    w.sync([src(env, "a")])
    assert len(FakeObserver.made) == 1
    w.stop(2.0)


def test_inotify_limit_falls_back_once_and_the_report_says_so(env, caplog):
    w, _ = make(env, fail=OSError(errno.ENOSPC, "inotify watch limit reached"))
    with caplog.at_level("WARNING"):
        w.sync([src(env, "a"), src(env, "b")])
        w.sync([src(env, "a"), src(env, "b")])
    assert not w.watching("a") and not w.watching("b")
    assert len([r for r in caplog.records if "watch" in r.getMessage().lower()]) == 1
    w.stop(2.0)


def test_missing_watchdog_falls_back(env, monkeypatch):
    def boom():
        raise ImportError("watchdog")

    monkeypatch.setattr(watcher_mod, "_default_observer", boom)
    w = SourceWatcher(env.host, lambda *a: None, debounce_s=0.05)
    w.sync([src(env, "a")])
    assert not w.watching("a")
    w.stop(2.0)


def test_report_shows_watch_flag(env):
    sid = env.add_and_confirm()
    w, _ = make(env)
    media = env.store()
    try:
        store = SourceStore(media)
        assert build_report(store, store.get_source(sid)).watch == 0
        w.sync([src(env, sid)])
        assert build_report(store, store.get_source(sid)).watch == 1
    finally:
        media.close()
        w.stop(2.0)


def test_stop_joins_every_thread(env):
    w, _ = make(env)
    w.sync([src(env, "a"), src(env, "b")])
    before = threading.active_count()
    assert w.stop(2.0) is True
    assert all(o.stopped and o.joined for o in FakeObserver.made)
    assert not [t for t in threading.enumerate() if t.name == "slm-source-watch"]
    assert threading.active_count() < before


# -- wired into the scan service ---------------------------------------------------------
def test_service_rescans_only_the_changed_file(env):
    from superlocalmemory.sources.scanner import SourceScanService

    env.write("a.md", "one")
    env.write("b.md", "two")
    sid = env.add_and_confirm()
    env.host.sleep = lambda s: None
    env.scan(sid)
    FakeObserver.made = []
    svc = SourceScanService(env.host, poll_s=0.01, interval_s=3600, watch_debounce_s=0.05,
                            observer_factory=lambda: FakeObserver())
    svc.start()
    try:
        assert wait_for(lambda: FakeObserver.made)
        obs = FakeObserver.made[0]
        obs.root = env.root
        env.write("a.md", "one edited")
        env.write("b.md", "two edited")
        before = len(env.runtime.saved)
        obs.emit("a.md")
        assert wait_for(lambda: len(env.runtime.saved) == before + 1)
        time.sleep(0.3)
        assert env.runtime.saved[-1]["content"] == "one edited"
        assert len(env.runtime.saved) == before + 1
    finally:
        assert svc.stop(5.0) is True
    assert all(o.joined for o in FakeObserver.made)


def test_service_has_no_observer_for_a_paused_source(env):
    from superlocalmemory.sources.scanner import SourceScanService

    env.write("a.md", "one")
    sid = env.add_and_confirm()
    env.host.sleep = lambda s: None
    env.remote = True  # the scan pauses the source
    FakeObserver.made = []
    svc = SourceScanService(env.host, poll_s=0.01, interval_s=3600, watch_debounce_s=0.05,
                            observer_factory=lambda: FakeObserver())
    svc.start()
    try:
        time.sleep(0.5)
        assert FakeObserver.made == []
    finally:
        svc.stop(5.0)
