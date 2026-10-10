"""Folder scans stand aside while a model swap pauses background work."""

from __future__ import annotations

from superlocalmemory.sources.scanner import SourceScanService
from tests.test_sources.test_service import _job_state, wait_for


def test_no_scan_starts_while_background_work_is_paused(env):
    env.write("a.md", "hello")
    env.add_and_confirm()
    paused = {"on": True}
    env.host.background_paused = lambda: paused["on"]
    env.host.sleep = lambda s: None
    service = SourceScanService(env.host, poll_s=0.01, interval_s=3600)
    service.start()
    try:
        assert not wait_for(lambda: len(env.runtime.saved) > 0, seconds=1.0)
        paused["on"] = False
        service.wake()
        assert wait_for(lambda: _job_state(env) == "done")
        assert len(env.runtime.saved) == 1
    finally:
        assert service.stop(5.0) is True


def test_a_scan_in_flight_goes_back_to_the_queue_when_a_swap_starts(env):
    for i in range(4):
        env.write(f"n{i}.md", f"note {i}")
    env.add_and_confirm()
    calls = {"n": 0}

    def paused():
        calls["n"] += 1
        return 2 <= calls["n"] <= 4

    env.host.background_paused = paused
    env.host.sleep = lambda s: None
    service = SourceScanService(env.host, poll_s=0.01, interval_s=3600)
    service.start()
    try:
        assert wait_for(lambda: _job_state(env) == "done")
        assert sorted(env.runtime.contents()) == [f"note {i}" for i in range(4)]
    finally:
        assert service.stop(5.0) is True
