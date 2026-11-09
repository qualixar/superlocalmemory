# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""A capture through the MCP ``observe`` tool reaches the live event stream.

It emitted ``memory.created``, which is not an event type, so the event bus
refused it and logged a warning on every capture: the dashboard never saw an
MCP capture. HTTP ``/observe`` emits ``memory.captured`` for the same thing.
"""

from __future__ import annotations

import pytest

from tests.test_security._routed_host import open_host


@pytest.fixture
def host(tmp_path, monkeypatch, mock_embedder):
    with open_host(tmp_path, monkeypatch, mock_embedder) as opened:
        yield opened


def test_an_mcp_capture_is_emitted_as_memory_captured(host, monkeypatch) -> None:
    from superlocalmemory.infra.event_bus import VALID_EVENT_TYPES, EventBus

    emitted: list[str] = []
    refused: list[str] = []
    real_emit = EventBus.emit

    def spy(self, event_type, *args, **kwargs):
        emitted.append(event_type)
        try:
            return real_emit(self, event_type, *args, **kwargs)
        except ValueError:
            refused.append(event_type)
            raise

    monkeypatch.setattr(EventBus, "emit", spy)
    out = host.unrouted("observe", content=(
        "We decided to use PostgreSQL instead of MySQL for the billing service "
        "because we need transactional DDL."))

    assert out.get("captured") is True, out
    assert "memory.captured" in emitted and refused == [], (emitted, refused)
    assert set(emitted) <= VALID_EVENT_TYPES
