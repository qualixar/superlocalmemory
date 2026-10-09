"""Bounded inbox wait."""

from __future__ import annotations

import threading
import time

import pytest

from superlocalmemory.mesh import broker_inbox
from tests.test_mesh.conftest import make_peer


def test_returns_immediately_when_unread_exists(broker) -> None:
    a, b = make_peer(broker, "a"), make_peer(broker, "b")
    broker.send_message(a, b, "hi")
    t0 = time.monotonic()
    msgs, timed_out = broker.wait_inbox(b, timeout_s=5)
    assert time.monotonic() - t0 < 0.3
    assert [m["content"] for m in msgs] == ["hi"] and timed_out is False


def test_wakes_quickly_on_insert_from_another_thread(broker) -> None:
    a, b = make_peer(broker, "a"), make_peer(broker, "b")
    sent_at: list[float] = []

    def later() -> None:
        time.sleep(0.3)
        sent_at.append(time.monotonic())
        broker.send_message(a, b, "late")

    th = threading.Thread(target=later)
    th.start()
    msgs, timed_out = broker.wait_inbox(b, timeout_s=5)
    woke = time.monotonic()
    th.join()
    assert [m["content"] for m in msgs] == ["late"] and timed_out is False
    assert woke - sent_at[0] < 0.3


def test_times_out_empty(broker, monkeypatch) -> None:
    b = make_peer(broker, "b")
    monkeypatch.setattr(broker_inbox, "WAIT_MIN_S", 0.2)
    t0 = time.monotonic()
    msgs, timed_out = broker.wait_inbox(b, timeout_s=0.2)
    assert msgs == [] and timed_out is True
    assert 0.15 <= time.monotonic() - t0 < 0.8


def test_timeout_is_clamped(broker, monkeypatch) -> None:
    b = make_peer(broker, "b")
    monkeypatch.setattr(broker_inbox, "WAIT_MIN_S", 0.3)
    monkeypatch.setattr(broker_inbox, "WAIT_MAX_S", 0.6)
    t0 = time.monotonic()
    broker.wait_inbox(b, timeout_s=0)
    low = time.monotonic() - t0
    t0 = time.monotonic()
    broker.wait_inbox(b, timeout_s=99)
    high = time.monotonic() - t0
    assert 0.25 <= low < 0.55
    assert 0.55 <= high < 1.0


def test_default_bounds_are_one_and_twenty() -> None:
    assert broker_inbox.WAIT_MIN_S == 1 and broker_inbox.WAIT_MAX_S == 20
    assert broker_inbox.MAX_CONCURRENT_WAITS == 8


def test_ninth_concurrent_wait_refused(broker, monkeypatch) -> None:
    b = make_peer(broker, "b")
    monkeypatch.setattr(broker_inbox, "WAIT_MIN_S", 0.5)
    monkeypatch.setattr(broker_inbox, "WAIT_MAX_S", 1.5)
    started = threading.Barrier(9)
    errors: list[Exception] = []

    def waiter() -> None:
        started.wait()
        broker.wait_inbox(b, timeout_s=1.5)

    threads = [threading.Thread(target=waiter) for _ in range(8)]
    for t in threads:
        t.start()
    started.wait()
    time.sleep(0.3)
    with pytest.raises(RuntimeError, match="too many waits"):
        broker.wait_inbox(b, timeout_s=1)
    for t in threads:
        t.join()
    # slots are released afterwards
    broker.wait_inbox(b, timeout_s=0.5)
