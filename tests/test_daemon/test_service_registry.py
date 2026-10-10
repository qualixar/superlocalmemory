"""The background service registry: ordering, failure isolation, thread safety."""

from __future__ import annotations

import threading

import pytest

from superlocalmemory.daemon.services import ServiceRegistry


class Fake:
    def __init__(self, name, log, *, stop_result=True, stop_raises=False,
                 health_raises=False):
        self.name = name
        self.log = log
        self.stop_result = stop_result
        self.stop_raises = stop_raises
        self.health_raises = health_raises
        self.started = False

    def start(self):
        self.started = True
        self.log.append(("start", self.name))

    def stop(self, timeout_s):
        self.log.append(("stop", self.name))
        if self.stop_raises:
            raise RuntimeError("boom")
        return self.stop_result

    def health(self):
        if self.health_raises:
            raise RuntimeError("sick")
        return {"state": "running", "detail": ""}


def test_register_and_get():
    registry, log = ServiceRegistry(), []
    svc = Fake("a", log)
    registry.register(svc)
    assert registry.get("a") is svc
    assert registry.get("missing") is None


def test_duplicate_name_rejected():
    registry, log = ServiceRegistry(), []
    registry.register(Fake("a", log))
    with pytest.raises(ValueError):
        registry.register(Fake("a", log))


def test_unregister_allows_reregistration():
    registry, log = ServiceRegistry(), []
    registry.register(Fake("a", log))
    registry.unregister("a")
    assert registry.get("a") is None
    registry.register(Fake("a", log))


def test_start_all_follows_after_order():
    registry, log = ServiceRegistry(), []
    registry.register(Fake("c", log), after=("b",))
    registry.register(Fake("b", log), after=("a",))
    registry.register(Fake("a", log))
    registry.start_all()
    assert log == [("start", "a"), ("start", "b"), ("start", "c")]


def test_cycle_raises_before_starting_anything():
    registry, log = ServiceRegistry(), []
    registry.register(Fake("a", log), after=("b",))
    registry.register(Fake("b", log), after=("a",))
    registry.register(Fake("c", log))
    with pytest.raises(ValueError):
        registry.start_all()
    assert log == []


def test_unknown_dependency_raises_before_starting_anything():
    registry, log = ServiceRegistry(), []
    registry.register(Fake("a", log))
    registry.register(Fake("b", log), after=("ghost",))
    with pytest.raises(ValueError):
        registry.start_all()
    assert log == []


def test_stop_all_runs_in_reverse_start_order():
    registry, log = ServiceRegistry(), []
    registry.register(Fake("b", log), after=("a",))
    registry.register(Fake("a", log))
    registry.start_all()
    log.clear()
    result = registry.stop_all()
    assert log == [("stop", "b"), ("stop", "a")]
    assert result == {"b": True, "a": True}


def test_a_raising_stop_does_not_block_the_rest():
    registry, log = ServiceRegistry(), []
    registry.register(Fake("a", log))
    registry.register(Fake("b", log, stop_raises=True), after=("a",))
    registry.start_all()
    result = registry.stop_all()
    assert result == {"b": False, "a": True}
    assert ("stop", "a") in log


def test_stop_reports_a_service_that_did_not_stop():
    registry, log = ServiceRegistry(), []
    registry.register(Fake("a", log, stop_result=False))
    assert registry.stop("a", 1.0) is False
    assert registry.stop("unknown", 1.0) is True


def test_start_one_by_name():
    registry, log = ServiceRegistry(), []
    registry.register(Fake("a", log))
    registry.start("a")
    assert log == [("start", "a")]


def test_snapshot_survives_a_raising_health():
    registry, log = ServiceRegistry(), []
    registry.register(Fake("ok", log))
    registry.register(Fake("bad", log, health_raises=True))
    snap = registry.snapshot()
    assert snap["ok"]["state"] == "running"
    assert snap["bad"]["state"] == "failed"
    assert "sick" in snap["bad"]["detail"]


def test_threads_can_register_and_snapshot_together():
    registry, log = ServiceRegistry(), []
    errors: list[BaseException] = []

    def register(prefix):
        try:
            for i in range(50):
                registry.register(Fake(f"{prefix}{i}", log))
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    def snapshot():
        try:
            for _ in range(50):
                registry.snapshot()
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=register, args=("x",)),
               threading.Thread(target=register, args=("y",)),
               threading.Thread(target=snapshot)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    assert len(registry.snapshot()) == 100
