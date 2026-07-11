# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Tests for core.loop_watchdog."""

from __future__ import annotations

import threading
import time


def _imports():
    from superlocalmemory.core import loop_watchdog as lw
    return lw


def test_fresh_tick_not_stale() -> None:
    lw = _imports()
    w = lw.LoopWatchdog(stale_threshold_s=0.1)
    w.tick()
    assert not w.is_stale()


def test_no_tick_becomes_stale() -> None:
    lw = _imports()
    w = lw.LoopWatchdog(stale_threshold_s=0.05)
    w.tick()
    time.sleep(0.12)
    assert w.is_stale()


def test_repeated_ticks_stay_fresh() -> None:
    lw = _imports()
    # V3.4.41: bumped threshold 0.1s -> 0.5s and intervals 0.02s -> 0.05s.
    # GitHub macos-14 runners exhibit timer skew up to ~80ms under load,
    # which made this test occasionally flake on `not is_stale()`. Wider
    # tolerance preserves the contract (5 ticks well below threshold)
    # while removing CI flake.
    w = lw.LoopWatchdog(stale_threshold_s=0.5)
    for _ in range(5):
        w.tick()
        time.sleep(0.05)
    assert not w.is_stale()


def test_on_stale_fires_once() -> None:
    lw = _imports()
    events = []
    w = lw.LoopWatchdog(stale_threshold_s=0.05, on_stale=lambda age: events.append(age))
    w.tick()
    time.sleep(0.08)
    w.check()
    w.check()  # second check within stale window should not re-fire
    assert len(events) == 1, f"on_stale fired {len(events)} times"


def test_background_thread_detects_stall() -> None:
    lw = _imports()
    events = []
    fired = threading.Event()

    def on_stale(age: float) -> None:
        events.append(age)
        fired.set()

    w = lw.LoopWatchdog(
        stale_threshold_s=0.05,
        on_stale=on_stale,
    )
    w.tick()
    stop = threading.Event()
    thread = threading.Thread(target=w.run_forever, args=(stop, 0.02), daemon=True)
    thread.start()
    try:
        assert fired.wait(timeout=2.0)
        assert len(events) >= 1
    finally:
        stop.set()
        thread.join(timeout=1.0)
