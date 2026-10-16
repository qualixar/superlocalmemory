# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""POST /internal/hook-events: the daemon accepts hook events only from a
caller holding its capability, validates them, spools them durably, wakes its
writer and acks. The ack means "on disk", never "in memory"."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from superlocalmemory.hooks import event_delivery as ed
from superlocalmemory.hooks import event_spool
from superlocalmemory.infra.daemon_identity import build_descriptor
from superlocalmemory.server import hook_event_writer as hew
from superlocalmemory.server.routes import hook_events


@pytest.fixture()
def setup(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("SLM_DATA_DIR", str(tmp_path))
    woke: list[int] = []
    monkeypatch.setattr(hew, "wake", lambda: woke.append(1))
    descriptor = build_descriptor(data_root=tmp_path, port=1, version="t", state="ready")
    app = FastAPI()
    app.state.daemon_descriptor = descriptor
    app.include_router(hook_events.router)
    headers = {"X-SLM-Daemon-Capability": descriptor.capability,
               "X-SLM-Target-Instance": descriptor.instance_id}
    return TestClient(app), headers, tmp_path / "hook_spool", woke


def _events(n: int = 2) -> list[dict]:
    return [ed.tool_event(session_id="s", tool_name=f"T{i}", profile_id="work",
                          response_text="ok") for i in range(n)]


def test_ack_means_spooled_and_writer_woken(setup) -> None:
    client, headers, spool, woke = setup
    events = _events()
    resp = client.post("/internal/hook-events", json={"events": events}, headers=headers)
    assert resp.status_code == 200
    assert resp.json() == {"ok": True, "accepted": 2, "dropped": 0}
    lines = (spool / "work.jsonl").read_text().splitlines()
    assert [json.loads(x)["event_id"] for x in lines] == [e["event_id"] for e in events]
    assert woke == [1]


def test_without_the_capability_nothing_is_accepted(setup) -> None:
    client, _headers, spool, _ = setup
    resp = client.post("/internal/hook-events", json={"events": _events()})
    assert resp.status_code == 403
    assert not spool.exists()


def test_a_replaced_daemon_instance_is_refused(setup) -> None:
    client, headers, spool, _ = setup
    resp = client.post("/internal/hook-events", json={"events": _events()},
                       headers={**headers, "X-SLM-Target-Instance": "other"})
    assert resp.status_code == 409 and not spool.exists()


@pytest.mark.parametrize("body", [
    {"events": []},
    {"events": [{"kind": "tool_event"}]},
    {"events": [{**ed.tool_event(session_id="s", tool_name="T", profile_id="p"),
                 "extra": 1}]},
    {"events": [ed.fact_signal(session_id="s", fact_ids=["f"], signal_name="bogus",
                               signal_value=True, profile_id="p")]},
])
def test_malformed_events_are_refused(setup, body) -> None:
    client, headers, spool, _ = setup
    resp = client.post("/internal/hook-events", json=body, headers=headers)
    assert resp.status_code == 422 and not spool.exists()


def test_oversized_body_is_refused(setup) -> None:
    client, headers, spool, _ = setup
    resp = client.post("/internal/hook-events", content=b"x" * (hook_events.MAX_BODY_BYTES + 1),
                       headers={**headers, "Content-Type": "application/json"})
    assert resp.status_code == 413 and not spool.exists()


def test_a_full_spool_reports_dropped_and_counts_it(setup, monkeypatch) -> None:
    client, headers, spool, _ = setup
    monkeypatch.setenv("SLM_HOOK_SPOOL_MAX_BYTES", "4096")
    seen = []
    for _ in range(30):
        seen.append(client.post("/internal/hook-events", json={"events": _events(1)},
                                headers=headers).json())
    assert any(r["dropped"] for r in seen)
    assert event_spool.read_stats(spool).dropped == sum(r["dropped"] for r in seen)
