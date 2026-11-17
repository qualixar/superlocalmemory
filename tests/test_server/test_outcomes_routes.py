# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""W3-5: ``GET /api/outcomes/health`` serves the one Memory Health report."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from superlocalmemory.storage.migrations import M075_injection_receipts as M075
from tests.test_server.test_per_request_profile import _daemon


@pytest.fixture
def routed(engine_with_mock_deps):
    with _daemon(engine_with_mock_deps, profiles=("work",)) as pair:
        yield pair


def _deliveries(app, profile: str, count: int) -> None:
    at = (datetime.now(UTC) - timedelta(days=2)).isoformat(timespec="microseconds")
    with app.state.engine._db.raw_connection() as conn:
        M075.apply(conn)
        for i in range(count):
            conn.execute(
                "INSERT INTO injection_deliveries (delivery_id, profile_id, session_id, path,"
                " design, holdout_state, epsilon, eligible_count, candidates, delivered_at)"
                " VALUES (?, ?, 's', 'mcp_recall', 'one-out-v2', 'off', 0, 0, 0, ?)",
                (f"{profile}-{i}", profile, at))
        conn.commit()


def test_the_report_has_every_section_and_plain_lines(routed) -> None:
    client, app = routed
    _deliveries(app, "default", 3)
    body = client.get("/api/outcomes/health").json()
    assert body["success"] and body["profile_id"] == "default"
    assert body["ledger"]["deliveries"] == {"mcp_recall": 3}
    assert {"effects", "use", "certificates", "second_opinions"} <= set(body)
    assert any("Deliveries in the last 30 days: 3" in line for line in body["lines"])


def test_the_report_is_per_profile(routed) -> None:
    client, app = routed
    _deliveries(app, "work", 2)
    _deliveries(app, "default", 5)
    body = client.get("/api/outcomes/health", params={"profile_id": "work"}).json()
    assert body["profile_id"] == "work" and body["ledger"]["deliveries"] == {"mcp_recall": 2}


def test_an_unknown_profile_is_refused(routed) -> None:
    client, _app = routed
    response = client.get("/api/outcomes/health", params={"profile_id": "nobody"})
    assert response.status_code >= 400 or response.json().get("success") is False


def test_a_store_without_the_ledger_still_answers(routed) -> None:
    client, _app = routed
    body = client.get("/api/outcomes/health").json()
    assert body["success"] and body["use"]["rate"] is None


# -- the same report through MCP ``health(memory=true)`` -------------------------------------------

class _Server:
    def __init__(self) -> None:
        self.tools: dict = {}

    def tool(self, **_kw):
        def wrap(fn):
            self.tools[fn.__name__] = fn
            return fn
        return wrap


def test_mcp_health_with_memory_returns_the_same_report(tmp_path) -> None:
    import asyncio
    import sqlite3
    from types import SimpleNamespace

    from superlocalmemory.mcp.tools_v3 import register_v3_tools

    db_path = tmp_path / "memory.db"
    conn = sqlite3.connect(str(db_path))
    M075.apply(conn)
    at = (datetime.now(UTC) - timedelta(days=1)).isoformat(timespec="microseconds")
    conn.execute("INSERT INTO injection_deliveries (delivery_id, profile_id, session_id, path,"
                 " design, holdout_state, epsilon, eligible_count, candidates, delivered_at)"
                 " VALUES ('d1', 'default', 's', 'cli_recall', 'one-out-v2', 'off', 0, 0, 0, ?)",
                 (at,))
    conn.commit()
    conn.close()
    engine = SimpleNamespace(profile_id="default", _db=SimpleNamespace(db_path=db_path))
    server = _Server()
    register_v3_tools(server, lambda: engine)
    out = asyncio.run(server.tools["health"](memory=True))
    assert out["success"] and out["profile"] == "default"
    assert out["memory"]["ledger"]["deliveries"] == {"cli_recall": 1}
    assert out["lines"][0].startswith("Memory Health")
    from superlocalmemory.server.remote_profile_binding import NEUTRAL_ARGUMENTS

    assert "memory" in NEUTRAL_ARGUMENTS
