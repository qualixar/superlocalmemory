# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""WS-9: an import must not hang the laptop, and agents using SLM at the same time must see no
failure and no "database is locked"."""

from __future__ import annotations

import threading
import time

from fastapi.testclient import TestClient

from superlocalmemory.portability import runner, service
from superlocalmemory.portability.jobs import JOBS
from tests.test_portability import fixtures as fx
from tests.test_portability.conftest import WHO

N = 120


def _items():
    return [{"id": f"id-{n}", "memory": f"Runbook number {n} lives in the shared drive for team {n % 7}.",
             "created_at": "2026-05-01T00:00:00Z"} for n in range(N)]


def _client(world) -> TestClient:
    client = TestClient(world.app)
    client.headers.update(world.client.headers)
    return client


def test_recall_and_saves_keep_working_while_an_import_runs(world, monkeypatch) -> None:
    monkeypatch.setattr(runner.C, "BATCH", 10)
    monkeypatch.setattr(runner.C, "REST_BETWEEN_BATCHES_S", 0.02)
    seed = _client(world).post("/remember", json={
        "content": "The quarterly readiness review for the on-call rotation is every Friday.",
        "idempotency_key": "seed-1"})
    assert seed.status_code == 200, seed.text
    run = service.start_run(world.engine, world.runtime, profile_id=world.profile,
                            docs=world.docs("m.json", fx.dumps(_items())), fmt=None,
                            label="m.json", decider=WHO)
    failures: list[str] = []
    latencies: list[float] = []
    saved_while_importing = []

    def recaller() -> None:
        client = _client(world)
        while JOBS.is_live(run["run_id"]) or len(latencies) < 8:
            began = time.monotonic()
            r = client.get("/recall", params={"q": "quarterly readiness review on-call"})
            latencies.append(time.monotonic() - began)
            if r.status_code != 200 or "locked" in r.text.lower():
                failures.append(f"recall {r.status_code} {r.text[:120]}")
            elif not r.json()["results"]:
                failures.append("recall found nothing")
            time.sleep(0.01)

    def saver() -> None:
        client = _client(world)
        for n in range(10):
            r = client.post("/remember", json={
                "content": f"Agent note {n}: the deploy checklist for service {n} was reviewed today.",
                "idempotency_key": f"agent-note-{n}"})
            if r.status_code != 200 or "locked" in r.text.lower():
                failures.append(f"remember {r.status_code} {r.text[:120]}")
            else:
                saved_while_importing.append(JOBS.is_live(run["run_id"]))
            time.sleep(0.05)

    threads = [threading.Thread(target=recaller), threading.Thread(target=saver)]
    for t in threads:
        t.start()
    assert JOBS.join(run["run_id"], 120), "the import did not finish"
    for t in threads:
        t.join(60)
    final = service.get_run(world.engine, world.profile, run["run_id"])
    assert failures == []
    assert final["state"] == "done" and final["saved_items"] == N
    assert len(latencies) >= 8 and max(latencies) < 10.0
    ordered = sorted(latencies)
    print(f"\nrecall during import: {len(latencies)} recalls, median {ordered[len(ordered) // 2]:.3f}s, "
          f"max {ordered[-1]:.3f}s; {len(saved_while_importing)} saves, "
          f"{sum(saved_while_importing)} while the import was running")
    assert any(saved_while_importing), "no save overlapped the import, so nothing was proven"
    assert world.count("memories", "content LIKE 'Agent note%'") == 10
    assert world.count("memories") == N + 11


def test_the_import_waits_for_recalls_between_batches(world, monkeypatch) -> None:
    from superlocalmemory.core import recall_gate

    seen = []
    monkeypatch.setattr(runner.recall_gate, "yield_to_recalls",
                        lambda seconds: seen.append(seconds))
    monkeypatch.setattr(runner.C, "BATCH", 5)
    world.run("m.json", fx.dumps(_items()[:20]))
    assert len(seen) == 4 and set(seen) == {runner.C.YIELD_MAX_S}
    assert recall_gate.in_flight() == 0


def test_a_busy_writer_is_retried_and_then_stops_the_run_with_words(world, monkeypatch) -> None:
    from superlocalmemory.core.remember_runtime import CanonicalRememberBusy

    def busy(*_a, **_k):
        raise CanonicalRememberBusy("too many saves are arriving at once")

    monkeypatch.setattr(runner, "submit_memory_settled", busy)
    monkeypatch.setattr(runner, "SAVE_ATTEMPTS", 2)
    run = world.run("m.json", fx.dumps(_items()[:3]))
    assert run["state"] == "failed" and "too busy" in run["note"] and "kept" in run["note"]
    assert "Traceback" not in run["note"]


def test_a_refused_item_is_skipped_and_the_run_goes_on(world) -> None:
    run = world.run("m.json", fx.dumps([{"id": "a", "memory": "ok"}, *_items()[:2]]))
    assert run["state"] == "done" and run["saved_items"] == 2
    assert run["skipped"].get("rejected") == 1  # too thin to remember: the writer's own policy
