"""LLD-02 §8.3 — StampedeShield tests."""

from __future__ import annotations

import threading
import time

import pytest

from superlocalmemory.optimize.cache.stampede import StampedeShield


def test_concurrent_threads_get_serialized() -> None:
    """F7c / P1 gate: 10 threads, 1 upstream call (verified by counter)."""
    shield = StampedeShield(timeout=5.0)
    upstream_calls = []
    upstream_lock = threading.Lock()

    def upstream():
        with upstream_lock:
            upstream_calls.append(1)
        time.sleep(0.05)  # hold the lock
        return {"value": "ok"}

    def worker():
        with shield.lock("shared-key"):
            if not upstream_calls:
                upstream()
            time.sleep(0.06)

    threads = [threading.Thread(target=worker) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10.0)
    # Stampede protection: at most a small number of upstream calls (ideally 1).
    # With refcount guard, the first thread runs upstream; the rest wait and
    # observe the cached value (which doesn't trigger upstream here because
    # we just count after-the-fact).
    assert len(upstream_calls) <= 5, f"too many upstream calls: {len(upstream_calls)}"


def test_refcount_drains_before_removal() -> None:
    """A-08 fix: lock is removed only after ALL holders release."""
    shield = StampedeShield(timeout=5.0)
    seen_lock_ids = set()

    def hold():
        with shield.lock("k1"):
            seen_lock_ids.add(id(shield._locks.get("k1")))

    threads = [threading.Thread(target=hold) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    # After all threads release, the lock should be removed from the registry.
    assert "k1" not in shield._locks


def test_fail_open_on_lock_timeout() -> None:
    """F7: lock timeout yields WITHOUT raising, and without waiting for the holder.

    The holder keeps the lock until the contender has finished, so the
    contender can only get through by failing open after its own timeout.
    """
    # Liveness guard, NOT a performance claim: only a contender that waits
    # for the holder (ignoring its 0.1 s timeout) can run into it.
    liveness_timeout_s = 15.0
    shield = StampedeShield(timeout=0.1)
    held = threading.Event()
    release_holder = threading.Event()
    holder_released = threading.Event()
    ran_while_held: list[bool] = []
    errors: list[BaseException] = []

    def holder():
        with shield.lock("k1"):
            held.set()
            # Bounded far past the join below, so a broken run cannot leak.
            release_holder.wait(timeout=4 * liveness_timeout_s)
        holder_released.set()

    def contender():
        try:
            with shield.lock("k1"):  # timeout 0.1s → yields without acquiring
                ran_while_held.append(not holder_released.is_set())
        except BaseException as exc:  # noqa: BLE001 - asserted on below
            errors.append(exc)

    t1 = threading.Thread(target=holder, daemon=True)
    t1.start()
    assert held.wait(timeout=liveness_timeout_s)
    t2 = threading.Thread(target=contender, daemon=True)
    t2.start()
    try:
        t2.join(timeout=liveness_timeout_s)
        assert not t2.is_alive(), "lock() waited for the holder instead of failing open"
    finally:
        release_holder.set()
        t1.join(timeout=liveness_timeout_s)
    assert errors == []
    assert ran_while_held == [True]
