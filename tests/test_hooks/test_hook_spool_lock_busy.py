# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""A busy spool lock never costs an event.

The lock is shared by every hook process and the daemon's claim pass. When it
stays held past a producer's wait (a loaded machine, a slow disk, a holder
descheduled mid-append), the producer used to raise ``TimeoutError`` and the
hook reported ``spool_failed``: the event was gone and ``slm doctor`` never
counted it. Now the producer writes its batch to a file of its own, complete
before it is visible, which the daemon drains like any claimed file.

Each test holds the lock itself, so the contention is deterministic rather
than whatever the test host's load happens to be.
"""

from __future__ import annotations

import json
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import pytest

from superlocalmemory.hooks import event_delivery as ed
from superlocalmemory.hooks import event_spool
from superlocalmemory.server import hook_event_writer as hew
from tests.helpers.hook_event_db import bootstrap, query


def _ev(i: int, **extra) -> dict:
    return {"kind": "tool_event", "event_id": f"{i:032x}", "profile_id": "default",
            "session_id": "s", "tool_name": "Read", **extra}


def _ids(paths: list[Path]) -> list[str]:
    return [json.loads(line)["event_id"]
            for p in paths for line in p.read_text(encoding="utf-8").splitlines() if line]


@contextmanager
def _held_for(root: Path, seconds: float) -> Iterator[None]:
    """Another holder keeps the lock for ``seconds``, then lets go."""
    taken, done = threading.Event(), threading.Event()

    def hold() -> None:
        with event_spool.spool_lock(root):
            taken.set()
            done.wait(seconds)

    thread = threading.Thread(target=hold, daemon=True)
    thread.start()
    assert taken.wait(5)
    try:
        yield
    finally:
        done.set()
        thread.join(5)


def test_a_busy_lock_never_loses_the_event(tmp_path: Path) -> None:
    # Production wait, production defaults: the lock outlasts it.
    with event_spool.spool_lock(tmp_path):
        result = event_spool.append_events([_ev(1)], profile="default", root=tmp_path)
        assert (result.written, result.dropped) == (1, 0)
    assert _ids(event_spool.claim_ready(tmp_path)) == [f"{1:032x}"]


def test_the_hook_spools_inside_its_budget_while_the_lock_is_busy(
        tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "data"
    root.mkdir()
    monkeypatch.setenv("SLM_DATA_DIR", str(root))  # no daemon.json: straight to the spool
    events = [ed.tool_event(session_id="s1", tool_name="T", profile_id="default",
                            tool_input={"i": 1}, response_text="ok")]
    spool = root / "hook_spool"
    with event_spool.spool_lock(spool):
        started = time.perf_counter()
        assert ed.deliver(events, profile="default") == "spooled"
        waited = time.perf_counter() - started
    # The hook does not sit out the daemon's full lock wait on a user's tool call.
    assert waited < event_spool.LOCK_WAIT_S
    assert _ids(event_spool.claim_ready(spool)) == [events[0]["event_id"]]


def test_an_event_written_beside_the_lock_reaches_the_database(tmp_path: Path) -> None:
    db = bootstrap(tmp_path / "memory.db")
    spool = tmp_path / "hook_spool"
    event = ed.tool_event(session_id="s1", tool_name="Beside", profile_id="default",
                          tool_input={"n": 1}, response_text="ok")
    with event_spool.spool_lock(spool):
        assert event_spool.append_events([event], profile="default", root=spool,
                                         wait_s=0.01).written == 1
    report = hew.HookEventWriter(db, root=spool).drain_once()
    assert report.applied == 1
    rows = query(db, "SELECT tool_name, metadata FROM tool_events")
    assert [(r["tool_name"], json.loads(r["metadata"])["event_id"]) for r in rows] == [
        ("Beside", event["event_id"])]
    assert event_spool.read_stats(spool).pending_events == 0


def test_two_producers_in_the_same_instant_never_overwrite_each_other(
        tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(event_spool.time, "time_ns", lambda: 1_700_000_000_000_000_000)
    with event_spool.spool_lock(tmp_path):
        for i in (1, 2, 3):
            assert event_spool.append_events([_ev(i)], profile="default", root=tmp_path,
                                             wait_s=0.01).written == 1
    assert sorted(_ids(event_spool.claim_ready(tmp_path))) == [f"{i:032x}" for i in (1, 2, 3)]


def test_a_half_written_file_is_never_visible_to_the_daemon(
        tmp_path: Path, monkeypatch) -> None:
    seen: list[list[Path]] = []
    real_write_all = event_spool._write_all

    def write_and_look(fd: int, data: bytes) -> None:
        half = len(data) // 2
        real_write_all(fd, data[:half])
        seen.append(sorted(tmp_path.glob("*.draining")) + sorted(tmp_path.glob("*.jsonl")))
        real_write_all(fd, data[half:])

    monkeypatch.setattr(event_spool, "_write_all", write_and_look)
    with event_spool.spool_lock(tmp_path):
        event_spool.append_events([_ev(1)], profile="default", root=tmp_path, wait_s=0.01)
    assert seen == [[]]
    assert event_spool.read_stats(tmp_path).pending_events == 1


def test_a_failed_write_beside_the_lock_raises_and_leaves_nothing(
        tmp_path: Path, monkeypatch) -> None:
    def disk_full(fd: int, data: bytes) -> None:
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(event_spool, "_write_all", disk_full)
    with event_spool.spool_lock(tmp_path):
        with pytest.raises(OSError):  # the hook reports it; never a silent success
            event_spool.append_events([_ev(1)], profile="default", root=tmp_path,
                                      wait_s=0.01)
    assert [p.name for p in tmp_path.iterdir() if p.name != ".lock"] == []


def test_erase_removes_events_written_beside_the_lock(tmp_path: Path) -> None:
    with event_spool.spool_lock(tmp_path):
        event_spool.append_events([_ev(1), _ev(2)], profile="work", root=tmp_path,
                                  wait_s=0.01)
        event_spool.append_events([_ev(3)], profile="default", root=tmp_path, wait_s=0.01)
    (tmp_path / ".work.0-1-dead.part").write_bytes(b'{"orphan": 1}\n')  # a producer that died
    assert event_spool.erase_profile(tmp_path, "work") == 2  # the orphan was never an event
    left = sorted(p.name for p in tmp_path.iterdir() if p.name != ".lock")
    assert len(left) == 1 and left[0].startswith("default.")


def test_the_cap_still_holds_and_counts_beside_the_lock(tmp_path: Path) -> None:
    event_spool.append_events([_ev(0)], profile="default", root=tmp_path, max_bytes=400)
    # Busy past the producer's wait, free again before the count is due.
    with _held_for(tmp_path, 0.3):
        result = event_spool.append_events([_ev(i) for i in range(1, 6)], profile="default",
                                           root=tmp_path, max_bytes=400, wait_s=0.05)
    assert result.dropped >= 1 and result.written + result.dropped == 5
    stats = event_spool.read_stats(tmp_path)
    assert stats.dropped == result.dropped
    assert stats.pending_bytes <= 400


def test_a_dead_producers_part_file_is_swept_a_live_ones_is_not(tmp_path: Path) -> None:
    import os

    stale, fresh = tmp_path / ".default.1-1-aa.part", tmp_path / ".default.2-2-bb.part"
    for path in (stale, fresh):
        path.write_bytes(b'{"x": 1}\n')
    old = time.time() - 2 * 3600
    os.utime(stale, (old, old))
    assert event_spool.claim_ready(tmp_path) == []  # never drained: never acknowledged
    assert not stale.exists() and fresh.exists()
