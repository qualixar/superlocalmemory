# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Observability: HTTP write paths must emit EventBus events.

Before this change only MCP tool handlers emitted bus events; the HTTP
fast paths (/observe, /remember, AutoCapture pipeline, materializer) were
silent, leaving the dashboard event stream structurally empty. These tests
verify the new emissions without touching SLM core logic or altering any
write behaviour.

All tests mock the EventBus so no database or daemon is required.
"""

from __future__ import annotations

import time
import unittest.mock as mock

from superlocalmemory.infra.event_bus import VALID_EVENT_TYPES
from superlocalmemory.server.unified_daemon import ObserveBuffer, _emit_event


# ---------------------------------------------------------------------------
# VALID_EVENT_TYPES coverage
# ---------------------------------------------------------------------------

def test_new_event_types_registered():
    assert "memory.observed" in VALID_EVENT_TYPES
    assert "memory.captured" in VALID_EVENT_TYPES
    assert "memory.dropped" in VALID_EVENT_TYPES
    assert "memory.queued" in VALID_EVENT_TYPES


# ---------------------------------------------------------------------------
# _emit_event helper — never raises, tags source_protocol="http"
# ---------------------------------------------------------------------------

def test_emit_event_never_raises_on_bus_error():
    with mock.patch(
        "superlocalmemory.infra.event_bus.EventBus.get_instance",
        side_effect=RuntimeError("bus down"),
    ):
        _emit_event("memory.observed")  # must not propagate


def test_emit_event_uses_http_protocol(tmp_path):
    from superlocalmemory.infra.event_bus import EventBus
    EventBus.reset_instance(tmp_path / "test.db")
    bus = EventBus.get_instance(tmp_path / "test.db")
    received = []
    bus.subscribe(lambda e: received.append(e))

    with mock.patch(
        "superlocalmemory.infra.event_bus.EventBus.get_instance",
        return_value=bus,
    ):
        _emit_event("memory.observed", payload={"x": 1})

    assert received, "no event emitted"
    assert received[0]["source_protocol"] == "http"


# ---------------------------------------------------------------------------
# ObserveBuffer.enqueue → memory.observed
# ---------------------------------------------------------------------------

def test_enqueue_emits_memory_observed():
    buf = ObserveBuffer(debounce_sec=60)  # long debounce — no flush in test
    emitted: list[tuple] = []

    with mock.patch(
        "superlocalmemory.server.unified_daemon._emit_event",
        side_effect=lambda et, payload=None, **kw: emitted.append((et, payload)),
    ):
        buf.enqueue("hello world")

    assert any(et == "memory.observed" for et, _ in emitted)


def test_enqueue_duplicate_does_not_emit():
    buf = ObserveBuffer(debounce_sec=60)
    emitted: list[str] = []

    with mock.patch(
        "superlocalmemory.server.unified_daemon._emit_event",
        side_effect=lambda et, **kw: emitted.append(et),
    ):
        buf.enqueue("same content")
        buf.enqueue("same content")  # duplicate — no event

    assert emitted.count("memory.observed") == 1


# ---------------------------------------------------------------------------
# ObserveBuffer._flush → memory.captured / memory.dropped
# ---------------------------------------------------------------------------

def _flush_with_decision(capture: bool) -> list[tuple[str, dict | None]]:
    buf = ObserveBuffer(debounce_sec=0.01)
    emitted: list[tuple[str, dict | None]] = []

    decision = mock.MagicMock()
    decision.capture = capture
    decision.category = "note"
    decision.reason = "test reason"
    decision.confidence = 0.9

    with mock.patch(
        "superlocalmemory.hooks.auto_capture.AutoCapture",
    ) as MockAC, mock.patch(
        "superlocalmemory.server.unified_daemon._emit_event",
        side_effect=lambda et, payload=None, **kw: emitted.append((et, payload)),
    ):
        MockAC.return_value.evaluate.return_value = decision
        buf.set_engine(object())
        buf.enqueue("some content")
        time.sleep(0.15)

    return emitted


def test_flush_captured_emits_memory_captured():
    emitted = _flush_with_decision(capture=True)
    event_types = [et for et, _ in emitted]
    assert "memory.captured" in event_types
    assert "memory.dropped" not in event_types


def test_flush_dropped_emits_memory_dropped():
    emitted = _flush_with_decision(capture=False)
    event_types = [et for et, _ in emitted]
    assert "memory.dropped" in event_types
    assert "memory.captured" not in event_types
