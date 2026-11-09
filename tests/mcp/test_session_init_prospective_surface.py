# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""session_init upcoming-events surface — tests for the scheduled-fact window.

Three tests with distinct failure conditions:

1. A store with a temporal fact dated tomorrow returns that fact in
   ``upcoming_events`` — the window query runs and is wired into the response.

2. A store with zero temporal facts in the 14-day window: the prospective query
   IS executed (proving the code path ran), AND upcoming_events is absent in
   the response — no empty stub.

3. When the DB's execute() raises, the surface degrades silently and
   session_init still returns success=True. The query must have been attempted
   (proven by the spy counter), distinguishing this case from "code never ran".

All three tests FAIL before the implementation and PASS after.
    - Test 1 fails because the key is absent.
    - Tests 2 and 3 fail because the DB execute call counter stays at zero
      (the code that calls execute doesn't exist yet).

Run:
    pytest tests/mcp/test_session_init_prospective_surface.py -v
"""

from __future__ import annotations

import asyncio
import datetime
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from superlocalmemory.mcp.tools_active import register_active_tools

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

class _ToolServer:
    """Minimal tool-registration recorder."""

    def __init__(self) -> None:
        self.tools: dict[str, Any] = {}

    def tool(self, *args: Any, **kwargs: Any):
        del args, kwargs

        def decorate(fn: Any) -> Any:
            self.tools[fn.__name__] = fn
            return fn

        return decorate


class _FakeRecallResponse:
    """Empty pool-recall response — isolates the prospective surface."""

    results: list = []
    score_contract_version: str = "2"
    calibration_status: str = "uncalibrated"
    calibration_id = None
    answer_confidence = None
    abstained: bool = False
    abstention_reason = None


def _make_rules() -> SimpleNamespace:
    return SimpleNamespace(
        should_recall=lambda _event: True,
        get_recall_config=lambda: {"relevance_threshold": 0.1},
    )


def _run(coro: Any) -> Any:
    return asyncio.run(coro)


def _tomorrow_iso() -> str:
    return (
        datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=1)
    ).date().isoformat()


# ---------------------------------------------------------------------------
# Spy DB: records every execute() call so tests can assert the query ran.
# ---------------------------------------------------------------------------

class _SpyDB:
    """Fake DB that counts execute() calls and returns caller-supplied rows."""

    def __init__(self, rows_for_prospective: list) -> None:
        self._rows = rows_for_prospective
        self.execute_call_count = 0
        self.execute_raised = False

    def get_pinned(self, _profile_id: str) -> list:
        return []

    def execute(self, sql: str, params: tuple = ()) -> list:
        if "fact_type" in sql and "referenced_date" in sql:
            self.execute_call_count += 1
            return self._rows
        return []


class _SpyDBThatRaises(_SpyDB):
    """Like _SpyDB but raises on the prospective query."""

    def __init__(self) -> None:
        super().__init__(rows_for_prospective=[])

    def execute(self, sql: str, params: tuple = ()) -> list:
        if "fact_type" in sql and "referenced_date" in sql:
            self.execute_call_count += 1
            self.execute_raised = True
            raise RuntimeError("simulated index corruption")
        return []


def _engine_with_db(db: _SpyDB) -> SimpleNamespace:
    return SimpleNamespace(profile_id="default", mode="B", db=db, _db=None)


def _patches(engine_instance: Any) -> tuple:
    """Return the three patches needed to run session_init without a daemon."""
    return (
        patch(
            "superlocalmemory.hooks.rules_engine.RulesEngine",
            return_value=_make_rules(),
        ),
        patch(
            "superlocalmemory.mcp._pool_adapter.pool_recall",
            return_value=_FakeRecallResponse(),
        ),
        patch(
            "superlocalmemory.mcp.tools_active._canonical_feedback_count",
            return_value=0,
        ),
    )


def _call_session_init(server: _ToolServer) -> dict:
    return _run(server.tools["session_init"](project_path="/test/proj"))


# ---------------------------------------------------------------------------
# Test 1: upcoming event lands in the response for a fact within the window.
# RED state: upcoming_events key is absent (surface not yet wired in).
# ---------------------------------------------------------------------------

def test_upcoming_events_present_for_temporal_fact_within_window() -> None:
    """session_init includes a temporal fact dated within 14 days in upcoming_events."""
    spy_db = _SpyDB(rows_for_prospective=[
        {
            "fact_id": "prosp-test-001",
            "content": "The demo is scheduled for tomorrow morning.",
            "referenced_date": _tomorrow_iso(),
        }
    ])
    engine_instance = _engine_with_db(spy_db)

    server = _ToolServer()
    register_active_tools(server, lambda: engine_instance)

    with _patches(engine_instance)[0], _patches(engine_instance)[1], _patches(engine_instance)[2]:
        result = _call_session_init(server)

    assert result["success"] is True, f"session_init failed: {result}"
    events = result.get("upcoming_events")
    assert events is not None, (
        "upcoming_events key absent; the prospective surface is not wired into session_init"
    )
    assert len(events) >= 1, f"Expected >=1 upcoming event but got {events!r}"
    event = events[0]
    assert event["fact_id"] == "prosp-test-001"
    assert "demo" in event["content"]
    assert "scheduled_at" in event


# ---------------------------------------------------------------------------
# Test 2: upcoming_events is absent (not an empty stub) when the window is
# empty. The query MUST run (spy counter proves it).
# RED state: execute_call_count == 0 (no code calls execute for prospective).
# ---------------------------------------------------------------------------

def test_upcoming_events_absent_and_query_executed_when_window_is_empty() -> None:
    """session_init omits upcoming_events when the window is empty, but the query ran."""
    spy_db = _SpyDB(rows_for_prospective=[])  # empty window
    engine_instance = _engine_with_db(spy_db)

    server = _ToolServer()
    register_active_tools(server, lambda: engine_instance)

    with _patches(engine_instance)[0], _patches(engine_instance)[1], _patches(engine_instance)[2]:
        result = _call_session_init(server)

    assert result["success"] is True, f"session_init failed: {result}"

    # The query must have run — otherwise "no upcoming events" and "code never
    # ran" are indistinguishable. This assertion is RED before the implementation.
    assert spy_db.execute_call_count >= 1, (
        "DB execute() was never called for the prospective-events query; "
        "the surface is not wired in"
    )

    # When the window is empty the key must be absent, not an empty list.
    assert "upcoming_events" not in result, (
        f"upcoming_events should be absent when the window is empty, "
        f"got: {result.get('upcoming_events')!r}"
    )


# ---------------------------------------------------------------------------
# Test 3: a DB error in the prospective query does not fail session_init.
# The query MUST have been attempted (spy counter proves it).
# RED state: execute_call_count == 0 (the code that calls execute doesn't exist).
# ---------------------------------------------------------------------------

def test_session_init_survives_prospective_query_error() -> None:
    """A DB error in the prospective-events query degrades silently; session_init succeeds."""
    spy_db = _SpyDBThatRaises()
    engine_instance = _engine_with_db(spy_db)

    server = _ToolServer()
    register_active_tools(server, lambda: engine_instance)

    with _patches(engine_instance)[0], _patches(engine_instance)[1], _patches(engine_instance)[2]:
        result = _call_session_init(server)

    # session_init must succeed even though the prospective surface raised.
    assert result["success"] is True, (
        f"session_init propagated a DB error from the prospective surface: {result}"
    )

    # The query must have been attempted — confirming the code ran and the
    # try/except caught the error. This is RED before the implementation.
    assert spy_db.execute_call_count >= 1, (
        "DB execute() was never called; the prospective surface is not wired in. "
        "A successful session_init here means the code was never reached, not "
        "that the error was properly suppressed."
    )

    # No error propagated into upcoming_events.
    assert "upcoming_events" not in result or result.get("upcoming_events") is None
