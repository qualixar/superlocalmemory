# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Tests for EngineRWLock.

Part of Qualixar | Author: Varun Pratap Bhardwaj
"""

from __future__ import annotations

import threading
import time

import pytest


def _import_lock():
    from superlocalmemory.core.engine_lock import EngineRWLock
    return EngineRWLock


def test_multiple_readers_concurrent() -> None:
    EngineRWLock = _import_lock()
    lock = EngineRWLock()
    inside_barrier = threading.Barrier(3, timeout=2.0)
    exit_event = threading.Event()
    errors: list[BaseException] = []

    def reader() -> None:
        try:
            with lock.read():
                inside_barrier.wait()
                exit_event.wait(timeout=2.0)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=reader) for _ in range(3)]
    for t in threads:
        t.start()
    time.sleep(0.2)
    exit_event.set()
    for t in threads:
        t.join(timeout=2.0)

    assert not errors, f"Reader thread raised: {errors}"


def test_writer_excludes_readers() -> None:
    EngineRWLock = _import_lock()
    lock = EngineRWLock()
    writer_in = threading.Event()
    reader_in = threading.Event()
    writer_hold = threading.Event()

    def writer() -> None:
        with lock.write():
            writer_in.set()
            writer_hold.wait(timeout=2.0)

    def reader() -> None:
        with lock.read():
            reader_in.set()

    w = threading.Thread(target=writer)
    w.start()
    assert writer_in.wait(timeout=1.0), "Writer did not acquire"
    r = threading.Thread(target=reader)
    r.start()
    assert not reader_in.wait(timeout=0.2), "Reader entered during writer"
    writer_hold.set()
    r.join(timeout=1.0)
    w.join(timeout=1.0)
    assert reader_in.is_set(), "Reader never got in after writer released"


def test_reader_excludes_writer() -> None:
    EngineRWLock = _import_lock()
    lock = EngineRWLock()
    reader_in = threading.Event()
    writer_in = threading.Event()
    reader_hold = threading.Event()

    def reader() -> None:
        with lock.read():
            reader_in.set()
            reader_hold.wait(timeout=2.0)

    def writer() -> None:
        with lock.write():
            writer_in.set()

    r = threading.Thread(target=reader)
    r.start()
    assert reader_in.wait(timeout=1.0)
    w = threading.Thread(target=writer)
    w.start()
    assert not writer_in.wait(timeout=0.2), "Writer entered during reader"
    reader_hold.set()
    w.join(timeout=1.0)
    r.join(timeout=1.0)
    assert writer_in.is_set(), "Writer never got in after reader released"


def test_writers_mutually_exclusive() -> None:
    EngineRWLock = _import_lock()
    lock = EngineRWLock()
    counter = {"active": 0, "peak": 0}
    counter_lock = threading.Lock()
    iterations = 50

    def writer() -> None:
        for _ in range(iterations):
            with lock.write():
                with counter_lock:
                    counter["active"] += 1
                    if counter["active"] > counter["peak"]:
                        counter["peak"] = counter["active"]
                time.sleep(0.001)
                with counter_lock:
                    counter["active"] -= 1

    threads = [threading.Thread(target=writer) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5.0)

    assert counter["peak"] == 1, f"Peak concurrent writers: {counter['peak']}"


def test_writer_priority_blocks_new_readers() -> None:
    EngineRWLock = _import_lock()
    lock = EngineRWLock()
    r1_in = threading.Event()
    r1_hold = threading.Event()
    w_queued = threading.Event()
    r2_in = threading.Event()
    w_done = threading.Event()

    def r1() -> None:
        with lock.read():
            r1_in.set()
            r1_hold.wait(timeout=3.0)

    def w() -> None:
        w_queued.set()
        with lock.write():
            w_done.set()

    def r2() -> None:
        with lock.read():
            r2_in.set()

    t_r1 = threading.Thread(target=r1)
    t_r1.start()
    assert r1_in.wait(timeout=1.0)
    t_w = threading.Thread(target=w)
    t_w.start()
    assert w_queued.wait(timeout=1.0)
    time.sleep(0.1)
    t_r2 = threading.Thread(target=r2)
    t_r2.start()
    assert not r2_in.wait(timeout=0.3), "Reader entered despite waiting writer"
    r1_hold.set()
    assert w_done.wait(timeout=1.0), "Writer never acquired"
    assert r2_in.wait(timeout=1.0), "Reader never acquired after writer released"
    t_r1.join(timeout=1.0)
    t_w.join(timeout=1.0)
    t_r2.join(timeout=1.0)


def test_exception_releases_reader() -> None:
    EngineRWLock = _import_lock()
    lock = EngineRWLock()

    with pytest.raises(RuntimeError, match="boom"):
        with lock.read():
            raise RuntimeError("boom")

    acquired = threading.Event()

    def writer() -> None:
        with lock.write():
            acquired.set()

    t = threading.Thread(target=writer)
    t.start()
    assert acquired.wait(timeout=1.0), "Lock leaked after reader exception"
    t.join(timeout=1.0)


def test_exception_releases_writer() -> None:
    EngineRWLock = _import_lock()
    lock = EngineRWLock()

    with pytest.raises(RuntimeError, match="boom"):
        with lock.write():
            raise RuntimeError("boom")

    acquired = threading.Event()

    def reader() -> None:
        with lock.read():
            acquired.set()

    t = threading.Thread(target=reader)
    t.start()
    assert acquired.wait(timeout=1.0), "Lock leaked after writer exception"
    t.join(timeout=1.0)


def test_stress_many_readers_one_writer() -> None:
    EngineRWLock = _import_lock()
    lock = EngineRWLock()

    reader_ops = [0]
    writer_ops = [0]
    stop = threading.Event()
    errors: list[BaseException] = []

    def reader() -> None:
        try:
            while not stop.is_set():
                with lock.read():
                    reader_ops[0] += 1
                    time.sleep(0.001)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    def writer() -> None:
        try:
            for _ in range(20):
                with lock.write():
                    writer_ops[0] += 1
                    time.sleep(0.002)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    readers = [threading.Thread(target=reader) for _ in range(5)]
    w = threading.Thread(target=writer)
    for r in readers:
        r.start()
    w.start()
    w.join(timeout=5.0)
    stop.set()
    for r in readers:
        r.join(timeout=2.0)

    assert not errors, f"Errors in stress: {errors}"
    assert writer_ops[0] == 20, f"Writer completed only {writer_ops[0]}/20"
    assert reader_ops[0] > 0, f"Readers fully starved: {reader_ops[0]}"
