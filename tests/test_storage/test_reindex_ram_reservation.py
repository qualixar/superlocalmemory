"""The re-embed loads its model under the shared RAM reservation: one model load at a time."""

from __future__ import annotations

import threading
import time

import pytest

from superlocalmemory.core import embedding_reindex as er
from superlocalmemory.core import embedding_reindex_steps as steps
from superlocalmemory.core import ram_lock
from superlocalmemory.storage import embedding_spaces as sp
from tests.test_storage.test_embedding_reindex_units import (  # noqa: F401 - fixture + helpers
    FakeEmbedder, _target, store,
)


@pytest.fixture()
def lock_file(tmp_path, monkeypatch):
    monkeypatch.setattr(ram_lock, "RAM_LOCK_PATH", tmp_path / "ram.sem")


def _wait_for(predicate, seconds=20.0):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


def test_the_model_load_waits_for_another_heavy_job_then_runs(store, lock_file, monkeypatch):
    root, db_path = store
    loads = []
    monkeypatch.setattr(steps, "build_embedder", lambda cfg: loads.append(time.monotonic()) or FakeEmbedder(4, "new"))
    held, release = threading.Event(), threading.Event()

    def other_heavy_job():
        with ram_lock.ram_reservation("fake-heavy-job", required_mb=0):
            held.set()
            release.wait(30)

    holder = threading.Thread(target=other_heavy_job)
    holder.start()
    assert held.wait(5)
    runner = er.ReindexRunner(db_path=db_path, data_root=root)
    job = runner.request_switch(_target())
    runner.start()
    try:
        time.sleep(0.5)
        assert loads == [], "the model loaded while another heavy job held the reservation"
        release.set()
        holder.join(5)
        assert _wait_for(lambda: runner.status()["job"]["state"] == "activated"), runner.status()
        assert len(loads) == 1 and runner.status()["job"]["job_id"] == job["job_id"]
    finally:
        release.set()
        runner.stop()


def test_the_model_is_built_while_the_reservation_is_held(store, lock_file, monkeypatch):
    root, db_path = store
    seen = {}

    def build(cfg):
        try:
            with ram_lock.ram_reservation("probe", required_mb=0, timeout_s=0.2):
                seen["free"] = True
        except RuntimeError:
            seen["free"] = False
        return FakeEmbedder(4, "new")

    monkeypatch.setattr(steps, "build_embedder", build)
    runner = er.ReindexRunner(db_path=db_path, data_root=root)
    runner.request_switch(_target())
    runner.start()
    try:
        assert _wait_for(lambda: runner.status()["job"]["state"] == "activated"), runner.status()
    finally:
        runner.stop()
    assert seen == {"free": False}


def test_a_reservation_that_is_refused_fails_the_job_in_plain_words(store, lock_file, monkeypatch):
    root, db_path = store
    monkeypatch.setattr(steps, "build_embedder", lambda cfg: FakeEmbedder(4, "new"))
    monkeypatch.setattr(er, "LOAD_WAIT_S", 0.2)
    release = threading.Event()
    held = threading.Event()

    def other():
        with ram_lock.ram_reservation("fake-heavy-job", required_mb=0):
            held.set()
            release.wait(30)

    t = threading.Thread(target=other)
    t.start()
    assert held.wait(5)
    runner = er.ReindexRunner(db_path=db_path, data_root=root)
    runner.request_switch(_target())
    runner.start()
    try:
        assert _wait_for(lambda: runner.status()["job"]["state"] == "failed"), runner.status()
        assert "memory" in runner.status()["job"]["error"]
    finally:
        release.set()
        t.join(5)
        runner.stop()
