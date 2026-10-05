# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""The saved side of the Answer Check history: one writer, safe erasure, retention."""

from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path

import pytest

from superlocalmemory.core import answer_check_history as h
from superlocalmemory.core import answer_check_history_store as store
from superlocalmemory.storage import migration_runner as mr

from .test_answer_check_history import make_response


@pytest.fixture()
def learning_db(tmp_path: Path) -> Path:
    from superlocalmemory.storage import schema

    learning, memory = tmp_path / "learning.db", tmp_path / "memory.db"
    with sqlite3.connect(memory) as conn:
        schema.create_all_tables(conn)
    result = mr.apply_all(learning, memory)
    assert "M053_answer_check_history" in result["applied"], result
    return learning


@pytest.fixture(autouse=True)
def fresh():
    store._reset_for_testing()
    h._reset_for_testing()
    yield
    store._reset_for_testing()
    h._reset_for_testing()


def _events(n: int, profile: str = "default", start_ms: int | None = None):
    base = int(time.time() * 1000) if start_ms is None else start_ms
    return [h.event_from_response(make_response(), profile, now_ms=base + i, origin_name="")
            for i in range(n)]


def _count(db: Path, profile: str | None = None) -> int:
    with sqlite3.connect(db) as conn:
        if profile is None:
            return conn.execute("SELECT COUNT(*) FROM answer_check_events").fetchone()[0]
        return conn.execute("SELECT COUNT(*) FROM answer_check_events WHERE profile_id=?",
                            (profile,)).fetchone()[0]


def test_insert_batch_roundtrip(learning_db) -> None:
    events = _events(128)
    conn = store.connect(learning_db, readonly=False)
    assert store.insert_batch(conn, events) == 128
    conn.close()
    rows = store.read_window(learning_db, "default", since_ms=0, include_dashboard=True)
    by_id = {r["event_id"]: r for r in rows}
    for ev in events:
        row = by_id[ev.event_id]
        for name in store.COLUMNS:
            value = getattr(ev, name)
            assert row[name] == (int(value) if isinstance(value, bool) else value), name


def test_insert_is_idempotent(learning_db) -> None:
    events = _events(128)
    conn = store.connect(learning_db, readonly=False)
    store.insert_batch(conn, events)
    assert store.insert_batch(conn, events) == 0
    conn.close()
    assert _count(learning_db) == 128


def test_tombstone_blocks_older_events(learning_db) -> None:
    t = int(time.time() * 1000)
    conn = store.connect(learning_db, readonly=False)
    store.erase_profile_rows(conn, "default", now_ms=t)
    older = _events(1, start_ms=t - 1)
    newer = _events(1, start_ms=t + 1)
    assert store.insert_batch(conn, older + newer) == 1
    conn.close()
    rows = store.read_window(learning_db, "default", since_ms=0, include_dashboard=True)
    assert [r["event_id"] for r in rows] == [newer[0].event_id]


def test_cli_erase_beats_inflight_daemon_batch(learning_db) -> None:
    """The daemon has a batch in flight; `slm gdpr` erases from another process."""
    h.enable(True)
    for _ in range(10):
        h.record_recall_verdict(make_response(), profile_id="victim")
    batch = h.snapshot_unsaved(store.BATCH_MAX)          # daemon thread A, mid-flush
    other = sqlite3.connect(learning_db, isolation_level=None)  # the CLI process
    time.sleep(0.002)
    store.erase_profile_rows(other, "victim", now_ms=int(time.time() * 1000))
    other.close()
    conn = store.connect(learning_db, readonly=False)
    assert store.insert_batch(conn, [ev for _, ev in batch]) == 0
    store._state["tombstones_seen_ms"] = 0
    assert store.apply_remote_tombstones(conn) == 10   # and the ring is purged too
    conn.close()
    assert _count(learning_db, "victim") == 0
    assert h.recent("victim", after_seq=0, limit=50)[0] == []


def test_in_process_erase_waits_for_flush(learning_db) -> None:
    h.enable(True)
    store._state["learning_db"] = learning_db
    for _ in range(5):
        h.record_recall_verdict(make_response(), profile_id="victim")
    assert store.erase_profile_everywhere(learning_db, "victim") == 0
    assert store.flush_once() == 0             # nothing of the erased profile left to save
    assert _count(learning_db, "victim") == 0


def test_prune_by_age(learning_db) -> None:
    now = int(time.time() * 1000)
    old = _events(5, start_ms=now - 31 * 86_400_000)
    fresh_ = _events(5, start_ms=now - 1000)
    conn = store.connect(learning_db, readonly=False)
    store.insert_batch(conn, old + fresh_)
    assert store.prune(conn, now_ms=now, retention_days=30, max_rows=10_000) == 5
    conn.close()
    assert _count(learning_db) == 5


def test_prune_by_count_per_profile(learning_db) -> None:
    now = int(time.time() * 1000)
    conn = store.connect(learning_db, readonly=False)
    many = _events(10_500, "busy", start_ms=now - 20_000)
    for i in range(0, len(many), 128):
        store.insert_batch(conn, many[i:i + 128])
    store.insert_batch(conn, _events(10, "quiet", start_ms=now - 20_000))
    assert store.prune(conn, now_ms=now, retention_days=30, max_rows=10_000) == 500
    oldest = conn.execute("SELECT MIN(occurred_ms) FROM answer_check_events "
                          "WHERE profile_id='busy'").fetchone()[0]
    conn.close()
    assert _count(learning_db, "busy") == 10_000 and _count(learning_db, "quiet") == 10
    assert oldest == many[500].occurred_ms


def test_prune_chunks_hold_lock_briefly(learning_db, monkeypatch) -> None:
    now = int(time.time() * 1000)
    conn = store.connect(learning_db, readonly=False)
    old = _events(1_200, start_ms=now - 40 * 86_400_000)
    for i in range(0, len(old), 128):
        store.insert_batch(conn, old[i:i + 128])
    sizes: list[int] = []
    real = store._in_txn

    def spy(c, fn):
        before = c.total_changes
        out = real(c, fn)
        sizes.append(c.total_changes - before)
        return out
    monkeypatch.setattr(store, "_in_txn", spy)
    assert store.prune(conn, now_ms=now, retention_days=30, max_rows=10_000) == 1_200
    conn.close()
    assert max(sizes) <= store.PRUNE_CHUNK and len(sizes) >= 3


#: Liveness bound for a poll, NOT a performance claim: generous so a loaded
#: host cannot make a flush that does happen look like one that did not.
_FLUSH_SEEN_S = 10.0
#: The interval while only a wake can explain a flush: far past any poll here.
_TIMER_OUT_OF_REACH_S = 3600.0


def _wait_for_count(db: Path, n: int, timeout_s: float) -> int:
    deadline = time.monotonic() + timeout_s
    while _count(db) < n and time.monotonic() < deadline:
        time.sleep(0.05)
    return _count(db)


def test_writer_flushes_on_interval_and_on_wake(learning_db, monkeypatch) -> None:
    # The interval: one unsaved check is below the wake threshold, so only the
    # timer can save it.
    store.start_writer(learning_db)
    h.record_recall_verdict(make_response(), profile_id="default")
    assert _wait_for_count(learning_db, 1, store.FLUSH_INTERVAL_S + _FLUSH_SEEN_S) == 1, (
        "idle flush on the interval")
    # The wake: restart with the timer out of reach and the wake flag clear
    # (stop_writer sets it), so a flush seen now can only be the wake's doing.
    store.stop_writer()
    h.wake_event().clear()
    monkeypatch.setattr(store, "FLUSH_INTERVAL_S", _TIMER_OUT_OF_REACH_S)
    store.start_writer(learning_db)
    for _ in range(h.WAKE_AT_UNSAVED):
        h.record_recall_verdict(make_response(), profile_id="default")
    assert _wait_for_count(learning_db, 129, _FLUSH_SEEN_S) == 129, (
        "woken early, not on the timer")


def test_writer_survives_locked_db(learning_db, monkeypatch) -> None:
    # What is tested is surviving a lock, not how long SQLite waits for one. At
    # the real 5 s busy wait the first failure lands at ~7 s, and SQLite counts
    # the sleeps it asks for, not the time that passes, so on a VM whose sleeps
    # overrun that lands past this test's 10 s deadline (most likely why it
    # failed on every GitHub macOS runner).
    monkeypatch.setattr(store, "BUSY_TIMEOUT_MS", 200)
    monkeypatch.setattr(store, "CONNECT_TIMEOUT_S", 0.2)
    store.start_writer(learning_db)
    h.record_recall_verdict(make_response(), profile_id="default")
    blocker = sqlite3.connect(learning_db, isolation_level=None)
    blocker.execute("BEGIN IMMEDIATE")
    try:
        deadline = time.monotonic() + 10   # first try at <= 2 s, gives up after 0.2 s
        while h.counters()["save_failures"] < 1 and time.monotonic() < deadline:
            time.sleep(0.1)
        assert h.counters()["save_failures"] >= 1
    finally:
        blocker.execute("ROLLBACK")
        blocker.close()
    deadline = time.monotonic() + 6
    while _count(learning_db) < 1 and time.monotonic() < deadline:
        time.sleep(0.1)
    assert _count(learning_db) == 1 and h.counters()["dropped_before_save"] == 0


def test_stop_writer_final_flush(learning_db) -> None:
    store.start_writer(learning_db)
    for _ in range(50):
        h.record_recall_verdict(make_response(), profile_id="default")
    assert store.stop_writer() == 0
    assert _count(learning_db) == 50
    h.record_recall_verdict(make_response(), profile_id="default")
    assert h.counters()["recorded"] == 50, "stopped writer means recording is off"


def _writers() -> list[threading.Thread]:
    return [t for t in threading.enumerate()
            if t.name == "slm-answer-check-history" and t.is_alive()]


def test_a_writer_stopped_mid_save_stays_stopped(learning_db, monkeypatch) -> None:
    """A stop that outlasts its wait must not leave that writer to be revived."""
    in_tick, release = threading.Event(), threading.Event()
    busy: list[threading.Thread] = []
    peak = [0]
    real_tick = store._tick

    def tick(*args, **kwargs):
        busy.append(threading.current_thread())
        peak[0] = max(peak[0], len(busy))
        try:
            if not in_tick.is_set():   # the first save sticks, like a long lock
                in_tick.set()
                release.wait(5)
            return real_tick(*args, **kwargs)
        finally:
            busy.remove(threading.current_thread())

    monkeypatch.setattr(store, "_tick", tick)
    store.start_writer(learning_db)
    h.wake_event().set()
    assert in_tick.wait(3)
    stopped = _writers()
    assert len(stopped) == 1
    store.stop_writer(timeout_s=0.1)          # gives up waiting: the save is stuck
    # Set when the successor starts waiting for the stuck writer to finish.
    successor_waits = threading.Event()
    real_join = stopped[0].join

    def join(timeout=None):
        successor_waits.set()
        return real_join(timeout)
    monkeypatch.setattr(stopped[0], "join", join)
    store.start_writer(learning_db)
    try:
        h.wake_event().set()                  # the new writer would save now
        # Release the stuck save only once the new writer has settled: either
        # waiting on its predecessor (correct) or inside a save of its own (the
        # defect, which `peak` then reports). A bounded poll, not a guessed sleep.
        deadline = time.monotonic() + 5
        while not (successor_waits.is_set() or peak[0] > 1) and time.monotonic() < deadline:
            time.sleep(0.01)
        release.set()
        stopped[0].join(5)
        assert not stopped[0].is_alive(), "the stopped writer was revived"
        h.record_recall_verdict(make_response(), profile_id="default")
        h.wake_event().set()
        deadline = time.monotonic() + 5
        while _count(learning_db) < 1 and time.monotonic() < deadline:
            time.sleep(0.05)
        assert _count(learning_db) == 1, "the new writer saves"
        assert len(_writers()) == 1 and peak[0] == 1, "two writers ran at once"
    finally:
        release.set()
        store.stop_writer()
    assert _writers() == []


def test_history_survives_restart(learning_db) -> None:
    store.start_writer(learning_db)
    for _ in range(3):
        h.record_recall_verdict(make_response(), profile_id="default")
    store.stop_writer()
    h._reset_for_testing()
    store.start_writer(learning_db)
    assert h.recent("default", after_seq=0, limit=50)[0] == []
    rows, cursor = store.read_page(learning_db, "default", cursor=None, limit=50,
                                   status=None, since_ms=0)
    assert len(rows) == 3 and cursor is None


def test_read_page_keyset(learning_db) -> None:
    conn = store.connect(learning_db, readonly=False)
    store.insert_batch(conn, _events(7, start_ms=1_000))
    conn.close()
    seen, cursor = [], None
    while True:
        rows, cursor_next = store.read_page(learning_db, "default", cursor=cursor, limit=3,
                                            status=None, since_ms=0)
        seen += rows
        if cursor_next is None:
            break
        ms, eid = cursor_next.split(":")
        cursor = (int(ms), eid)
    assert [r["occurred_ms"] for r in seen] == list(range(1_006, 999, -1))


def test_retention_clamps() -> None:
    assert store.clamp_settings(0, 5) == (1, 1_000)
    assert store.clamp_settings(9999, 50_000) == (365, 10_000)
    assert store.clamp_settings("x", None) == (30, 10_000)
    assert store.clamp_settings(30, 10_000) == (30, 10_000)


def test_readers_tolerate_missing_tables(tmp_path) -> None:
    db = tmp_path / "learning.db"
    sqlite3.connect(db).close()
    assert store.read_window(db, "p", since_ms=0, include_dashboard=True) == []
    assert store.erase_profile_everywhere(db, "p") == 0
    assert store.erase_profile_everywhere(tmp_path / "absent.db", "p") == 0


def test_concurrent_record_and_flush_lose_nothing(learning_db) -> None:
    store.start_writer(learning_db)
    def burst():
        for _ in range(400):
            h.record_recall_verdict(make_response(), profile_id="default")
    threads = [threading.Thread(target=burst) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    store.stop_writer()
    assert _count(learning_db) == 1600 and h.counters()["dropped_before_save"] == 0


# -- every recorded check is accounted for, and a clock step erases nothing new (A-F3) --

def _accounted(c: dict) -> int:
    return c["saved"] + c["erased_unsaved"] + c["dropped_before_save"] + c["unsaved"]


def _writer(learning_db: Path) -> None:
    h.enable(True)
    store._state["learning_db"] = learning_db


def test_a_row_an_erasure_refuses_is_counted(learning_db, monkeypatch) -> None:
    """The daemon has a batch in flight when another process erases the profile."""
    _writer(learning_db)
    for _ in range(3):
        h.record_recall_verdict(make_response(), profile_id="alice")
    taken = h.snapshot_unsaved

    def erase_meanwhile(n):
        batch = taken(n)
        time.sleep(0.002)
        other = store.connect(learning_db, readonly=False)  # `slm gdpr`, elsewhere
        store.erase_profile_rows(other, "alice", now_ms=int(time.time() * 1000))
        other.close()
        return batch
    monkeypatch.setattr(h, "snapshot_unsaved", erase_meanwhile)
    assert store.flush_once() == 3
    c = h.counters()
    assert _count(learning_db, "alice") == 0
    assert c["recorded"] == 3 and c["saved"] == 0 and c["erased_unsaved"] == 3
    assert _accounted(c) == c["recorded"]


def _record_with_clock_behind(profile: str, seconds: float) -> None:
    real = h.time.time
    h.time.time = lambda: real() - seconds
    try:
        h.record_recall_verdict(make_response(), profile_id=profile)
    finally:
        h.time.time = real


def test_a_recall_after_an_erasure_is_saved_when_the_clock_stepped_back(learning_db) -> None:
    _writer(learning_db)
    store.erase_profile_everywhere(learning_db, "bob")
    _record_with_clock_behind("bob", 1.0)          # NTP pulled the clock back
    conn = store.connect(learning_db, readonly=False)
    store.apply_remote_tombstones(conn)            # the next tick sees our own tombstone
    conn.close()
    assert store.flush_once() == 1
    assert _count(learning_db, "bob") == 1
    c = h.counters()
    assert c["saved"] == 1 and c["erased_unsaved"] == 0 and _accounted(c) == c["recorded"]


def test_a_remote_erasure_once_applied_spares_later_recalls(learning_db) -> None:
    _writer(learning_db)
    other = store.connect(learning_db, readonly=False)
    store.erase_profile_rows(other, "carol", now_ms=int(time.time() * 1000))
    other.close()
    conn = store.connect(learning_db, readonly=False)
    store.apply_remote_tombstones(conn)            # this process now knows of it
    _record_with_clock_behind("carol", 1.0)
    store._state["tombstones_seen_ms"] = 0
    store.apply_remote_tombstones(conn)            # seen again: still not its to erase
    conn.close()
    assert store.flush_once() == 1
    assert _count(learning_db, "carol") == 1


def test_a_recall_older_than_an_unseen_erasure_is_still_refused(learning_db) -> None:
    """The guarantee the tombstone exists for is unchanged."""
    _writer(learning_db)
    h.record_recall_verdict(make_response(), profile_id="dave")
    time.sleep(0.002)
    other = store.connect(learning_db, readonly=False)
    store.erase_profile_rows(other, "dave", now_ms=int(time.time() * 1000))
    other.close()
    assert store.flush_once() == 1
    assert _count(learning_db, "dave") == 0
    assert h.counters()["erased_unsaved"] == 1
