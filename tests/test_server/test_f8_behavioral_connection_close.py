# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later

"""F8 regression: get_assertions must close its SQLite connection even when
conn.execute() raises an exception.

Scenario reproduced: the memory.db is momentarily locked (SQLITE_BUSY) and
conn.execute() raises sqlite3.OperationalError.  The conn.close() call at the
end of get_assertions is NOT inside a try/finally, so the connection is leaked.
Under bursty dashboard requests this exhausts file-descriptor limits.

get_tool_events already uses try/finally — get_assertions must mirror that.
"""

from __future__ import annotations

import sqlite3
from unittest.mock import MagicMock, patch

import pytest

from superlocalmemory.server.routes import behavioral


def test_f8_connection_closed_after_execute_raises() -> None:
    """conn.close() must be called even when conn.execute() raises."""
    closed: list[bool] = []

    mock_conn = MagicMock()
    mock_conn.execute.side_effect = sqlite3.OperationalError("database is locked")
    mock_conn.close = lambda: closed.append(True)

    with patch("superlocalmemory.server.routes.behavioral.get_active_profile",
               return_value="work"), \
         patch("sqlite3.connect", return_value=mock_conn):
        result = behavioral.get_assertions()

    assert closed, (
        "F8: conn.close() was NOT called after conn.execute() raised. "
        "The connection is leaked. Wrap in try/finally: conn.close()."
    )
    # Outer except should return a graceful error dict, not re-raise
    assert "error" in result, "get_assertions must return an error dict on exception"


def test_f8_connection_closed_on_success() -> None:
    """conn.close() must also be called on the happy path (sanity check)."""
    closed: list[bool] = []

    mock_row = MagicMock()
    mock_row.keys.return_value = [
        "id", "trigger_condition", "action", "category", "confidence",
        "evidence_count", "reinforcement_count", "contradiction_count",
        "project_path", "source", "created_at", "updated_at",
    ]
    mock_row.__iter__ = lambda self: iter([])
    # Make dict(row) return an empty dict for simplicity
    mock_conn = MagicMock()
    mock_conn.execute.return_value.fetchall.return_value = []
    mock_conn.close = lambda: closed.append(True)

    with patch("superlocalmemory.server.routes.behavioral.get_active_profile",
               return_value="work"), \
         patch("sqlite3.connect", return_value=mock_conn):
        result = behavioral.get_assertions()

    assert closed, (
        "F8: conn.close() must be called on the happy path too "
        "(currently it IS called, but this test guards future regressions)."
    )
    assert result.get("assertions") == []


def test_f8_get_tool_events_already_uses_try_finally() -> None:
    """Regression guard: get_tool_events must continue to use try/finally."""
    import inspect
    source = inspect.getsource(behavioral.get_tool_events)
    assert "finally" in source, (
        "get_tool_events must use try/finally: conn.close() — regression guard"
    )
