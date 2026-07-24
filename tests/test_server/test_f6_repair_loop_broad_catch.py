# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later

"""F6 regression: _source_quality_repair_loop must not die permanently when
a single repair tick raises a non-storage exception.

Scenario reproduced: a row in action_outcomes has malformed fact_ids_json
(e.g. a plain string instead of a JSON array).  json.JSONDecodeError or
TypeError propagates out of _source_quality_repair_tick and escapes the
inner except (SourceQualityRepairUnavailable, sqlite3.Error) clause.
The outer except Exception logs once and marks state="failed", permanently
stopping repair for the entire session.

After the fix, a second except Exception clause in the inner try catches the
unknown error, logs at WARNING, and continues the loop.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from superlocalmemory.server import unified_daemon


def _make_app() -> object:
    app = SimpleNamespace()
    app.state = SimpleNamespace()
    app.state.source_quality_repair_status = {}
    return app


def test_f6_loop_survives_non_storage_exception() -> None:
    """Loop must continue after ValueError escapes the tick (not die on first error)."""
    app = _make_app()
    tick_count = [0]

    async def mock_discover(_mem, _pending, _completed):
        return ["work"]

    async def mock_tick(_app, _mem, _learn, _pending, *, batch_size):
        tick_count[0] += 1
        if tick_count[0] == 1:
            raise ValueError("malformed json blob — not (SourceQualityRepairUnavailable, sqlite3.Error)")
        return []  # empty pending → loop exits

    async def run():
        with patch.object(unified_daemon, "_discover_source_quality_profiles", mock_discover), \
             patch.object(unified_daemon, "_source_quality_repair_tick", mock_tick):
            await unified_daemon._source_quality_repair_loop(
                app,
                memory_db_path=Path("/nonexistent/memory.db"),
                learning_db_path=Path("/nonexistent/learning.db"),
                batch_size=100,
                tick_seconds=0.0,
            )

    asyncio.run(run())

    assert tick_count[0] == 2, (
        f"F6: loop must run both ticks (ran {tick_count[0]}). "
        "A ValueError in tick 1 must not kill the loop — broad per-iteration "
        "catch not implemented."
    )

    status = getattr(app.state, "source_quality_repair_status", {})
    assert status.get("state") == "complete", (
        f"F6: loop must complete after the bad row is skipped "
        f"(state={status.get('state')!r})"
    )


def test_f6_loop_still_terminates_on_cancelled_error() -> None:
    """asyncio.CancelledError must still propagate (loop cancellation must work)."""
    app = _make_app()

    async def mock_discover(_mem, _pending, _completed):
        return ["work"]

    async def mock_tick_cancel(_app, _mem, _learn, _pending, *, batch_size):
        raise asyncio.CancelledError()

    async def run():
        with patch.object(unified_daemon, "_discover_source_quality_profiles", mock_discover), \
             patch.object(unified_daemon, "_source_quality_repair_tick", mock_tick_cancel):
            try:
                await unified_daemon._source_quality_repair_loop(
                    app,
                    memory_db_path=Path("/nonexistent/memory.db"),
                    learning_db_path=Path("/nonexistent/learning.db"),
                    batch_size=100,
                    tick_seconds=0.0,
                )
            except asyncio.CancelledError:
                pass  # expected

    asyncio.run(run())
    status = getattr(app.state, "source_quality_repair_status", {})
    assert status.get("state") == "cancelled"
