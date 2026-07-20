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
from types import SimpleNamespace
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
    buf.set_engine(SimpleNamespace(_config=SimpleNamespace(scope=None), _profile_id="default"))
    emitted: list[tuple] = []

    with mock.patch(
        "superlocalmemory.server.unified_daemon._emit_event",
        side_effect=lambda et, payload=None, **kw: emitted.append((et, payload)),
    ), mock.patch("superlocalmemory.hooks.auto_capture.AutoCapture") as auto, mock.patch(
        "superlocalmemory.core.engine_ingestion.build_engine_ingestion_command"
    ) as build:
        auto.return_value.evaluate.return_value = SimpleNamespace(
            capture=True, category="decision", confidence=0.75, reason="matched"
        )
        build.return_value.submit.return_value = SimpleNamespace(
            operation_id="op-1", fact_ids=("fact-1",), state=SimpleNamespace(value="queryable")
        )
        result = buf.enqueue("we decided to use sqlite for durable storage")

    assert result["durable"] is True
    assert any(et == "memory.observed" for et, _ in emitted)


def test_enqueue_duplicate_does_not_emit():
    buf = ObserveBuffer(debounce_sec=60)
    buf.set_engine(SimpleNamespace(_config=SimpleNamespace(scope=None), _profile_id="default"))
    emitted: list[str] = []

    with mock.patch(
        "superlocalmemory.server.unified_daemon._emit_event",
        side_effect=lambda et, **kw: emitted.append(et),
    ), mock.patch("superlocalmemory.hooks.auto_capture.AutoCapture") as auto, mock.patch(
        "superlocalmemory.core.engine_ingestion.build_engine_ingestion_command"
    ) as build:
        auto.return_value.evaluate.return_value = SimpleNamespace(
            capture=True, category="decision", confidence=0.75, reason="matched"
        )
        build.return_value.submit.return_value = SimpleNamespace(
            operation_id="op-1", fact_ids=("fact-1",), state=SimpleNamespace(value="queryable")
        )
        buf.enqueue("we decided to use the same durable content")
        buf.enqueue("we decided to use the same durable content")

    assert emitted.count("memory.observed") == 1


# ---------------------------------------------------------------------------
# ObserveBuffer._flush → memory.captured / memory.dropped
# ---------------------------------------------------------------------------

def _flush_with_decision(capture: bool) -> list[tuple[str, dict | None]]:
    buf = ObserveBuffer(debounce_sec=60)
    buf.set_engine(SimpleNamespace(_config=SimpleNamespace(scope=None), _profile_id="default"))
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
        command = mock.MagicMock()
        command.submit.return_value = SimpleNamespace(
            operation_id="op-1", fact_ids=("fact-1",), state=SimpleNamespace(value="queryable")
        )
        with mock.patch(
            "superlocalmemory.core.engine_ingestion.build_engine_ingestion_command",
            return_value=command,
        ):
            buf.enqueue("we decided to use some durable content")

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
