# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Deleting or correcting a memory is announced once, from every surface.

The dashboard's live event stream offers "Memory deleted" and "Memory
corrected". Before 4.1.21 only the MCP ``delete_memory`` tool announced a
delete (a dashboard or CLI delete was silent), and no correction was announced
at all. The daemon's routes now announce both, so the dashboard, the CLI and
MCP are seen alike, and an MCP delete is announced once, not twice.
"""

from __future__ import annotations

import pytest

from tests.test_security._routed_host import HOST, open_host


@pytest.fixture
def host(tmp_path, monkeypatch, mock_embedder):
    with open_host(tmp_path, monkeypatch, mock_embedder) as opened:
        yield opened


@pytest.fixture
def events(monkeypatch) -> list[tuple[str, dict]]:
    from superlocalmemory.infra.event_bus import VALID_EVENT_TYPES, EventBus

    seen: list[tuple[str, dict]] = []

    def record(self, event_type, payload=None, **_kwargs):
        assert event_type in VALID_EVENT_TYPES, event_type
        seen.append((event_type, dict(payload or {})))

    monkeypatch.setattr(EventBus, "emit", record)
    return seen


def _of(events, event_type: str) -> list[dict]:
    return [payload for kind, payload in events if kind == event_type]


def test_a_dashboard_delete_is_announced(host, events) -> None:
    fact_id = host.save("The personal dentist visit is on Friday at noon.", HOST, "p")
    response = host.client.delete(f"/api/memories/{fact_id}")
    assert response.status_code == 200, response.text
    [deleted] = _of(events, "memory.deleted")
    assert (deleted["fact_id"], deleted["profile_id"]) == (fact_id, HOST)


def test_an_mcp_delete_is_announced_once(host, events) -> None:
    fact_id = host.save("The personal dentist visit is on Friday at noon.", HOST, "p")
    out = host.unrouted("delete_memory", fact_id=fact_id)
    assert out["success"] is True, out
    assert [p["fact_id"] for p in _of(events, "memory.deleted")] == [fact_id]


def test_a_correction_is_announced_when_proposed_and_when_reviewed(host, events) -> None:
    fact_id = host.save("The release train leaves on Tuesday at 14:00 UTC.", HOST, "p")
    proposed = host.client.patch(f"/api/memories/{fact_id}", json={
        "content": "The release train leaves on Thursday at 09:00 UTC."})
    assert proposed.status_code == 202, proposed.text
    case = proposed.json()["correction_case"]
    reviewed = host.client.post(f"/api/corrections/{case['case_id']}/apply",
                                json={"expected_version": case["version"]})
    assert reviewed.status_code == 200, reviewed.text

    updates = _of(events, "memory.updated")
    assert [u["status"] for u in updates] == ["proposed", "apply"], updates
    assert updates[0]["fact_id"] == fact_id and updates[0]["profile_id"] == HOST
    assert "Thursday" in updates[0]["content_preview"]
    assert updates[1]["case_id"] == case["case_id"]


def test_a_refused_delete_announces_nothing(host, events) -> None:
    response = host.client.delete("/api/memories/no-such-fact")
    assert response.status_code == 404
    assert _of(events, "memory.deleted") == []
