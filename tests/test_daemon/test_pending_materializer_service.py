"""The pending materializer as a background service with injected collaborators."""

from __future__ import annotations

import logging
import threading
import time
from types import SimpleNamespace

import pytest

from superlocalmemory.daemon.materializer import (
    PendingMaterializer,
    PendingProfileMismatchError,
)


class FakeRuntime:
    transitioning = False
    background_paused = False

    def __init__(self, profile_id="default"):
        self.snapshot = SimpleNamespace(profile_id=profile_id)

    def operation(self):
        runtime = self

        class _Lease:
            def __enter__(self_inner):
                return runtime.snapshot

            def __exit__(self_inner, *exc):
                return False

        return _Lease()


class FakeStore:
    def __init__(self, items=()):
        self.items = list(items)
        self.done: list[int] = []
        self.failed: list[tuple[int, str]] = []

    def get_pending(self, limit=50, profile_id=None):
        return [i for i in self.items if i["id"] not in self.done]

    def mark_done(self, item_id):
        self.done.append(item_id)

    def mark_failed(self, item_id, reason):
        self.failed.append((item_id, reason))


def _wait(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return predicate()


def _build(store, *, legacy, events=None, ingestion=None, engine=None):
    engine = engine if engine is not None else SimpleNamespace(_profile_id="default")
    events = events if events is not None else []
    return PendingMaterializer(
        engine_supplier=lambda: engine,
        runtime_supplier=lambda: FakeRuntime(),
        pending_store=store,
        emit_event=lambda *a, **k: events.append((a, k)),
        actor_id_supplier=lambda: "actor",
        recalls_in_flight=lambda: 0,
        ingestion_step=ingestion or (lambda eng, limit: (0, 0)),
        legacy_step=legacy,
    ), events


@pytest.fixture
def started():
    services: list[PendingMaterializer] = []
    yield services
    for svc in services:
        svc.stop(2.0)


def test_start_runs_a_named_thread_and_is_idempotent(started):
    svc, _ = _build(FakeStore(), legacy=lambda e, i: "op")
    started.append(svc)
    svc.start()
    svc.start()
    threads = [t for t in threading.enumerate() if t.name == "pending-materializer"]
    assert len(threads) == 1 and threads[0].is_alive()
    assert svc.health()["state"] == "running"


def test_stop_returns_true_and_the_thread_is_gone(started):
    svc, _ = _build(FakeStore(), legacy=lambda e, i: "op")
    started.append(svc)
    svc.start()
    assert svc.stop(5.0) is True
    assert not [t for t in threading.enumerate()
                if t.name == "pending-materializer" and t.is_alive()]
    assert svc.health()["state"] == "stopped"


def test_stop_before_start_is_clean():
    svc, _ = _build(FakeStore(), legacy=lambda e, i: "op")
    assert svc.stop(1.0) is True


def test_stop_timeout_returns_false_and_warns(started, caplog):
    release = threading.Event()
    entered = threading.Event()

    def blocking(engine, limit):
        entered.set()
        release.wait(10)
        return (0, 0)

    svc, _ = _build(FakeStore(), legacy=lambda e, i: "op", ingestion=blocking)
    started.append(svc)
    svc.start()
    assert entered.wait(5.0)
    try:
        with caplog.at_level(logging.WARNING):
            assert svc.stop(0.2) is False
        assert any("did not stop" in r.getMessage() for r in caplog.records)
    finally:
        release.set()
    assert svc.stop(5.0) is True


def test_backfill_marks_done_and_emits(started):
    store = FakeStore([{"id": 1, "content": "hello", "profile_id": "default"}])
    svc, events = _build(store, legacy=lambda engine, item: "op-1")
    started.append(svc)
    svc.start()
    assert _wait(lambda: store.done == [1])
    assert store.failed == []
    assert events[0][1]["payload"]["path"] == "legacy_pending_backfill"
    assert events[0][1]["source_agent"] == "materializer"


def test_backfill_marks_failed_on_error(started):
    store = FakeStore([{"id": 2, "content": "x", "profile_id": "default"}])

    def broken(engine, item):
        raise RuntimeError("no luck")

    svc, _ = _build(store, legacy=broken)
    started.append(svc)
    svc.start()
    assert _wait(lambda: store.failed)
    assert store.failed[0] == (2, "no luck")
    assert store.done == []


def test_profile_mismatch_leaves_the_row_pending(started):
    store = FakeStore([{"id": 3, "content": "x", "profile_id": "default"}])
    calls = []

    def mismatch(engine, item):
        calls.append(item["id"])
        raise PendingProfileMismatchError("moved")

    svc, _ = _build(store, legacy=mismatch)
    started.append(svc)
    svc.start()
    assert _wait(lambda: calls)
    svc.stop(5.0)
    assert store.failed == [] and store.done == []
