"""Tests for the /api/agents/memory-activity endpoint (multi-agent memory view).

Seeds a temp ingestion_operations table and verifies per-agent grouping,
profile isolation, the unknown-agent bucket, and recency ordering.
"""

import asyncio
import sqlite3

from superlocalmemory.server.routes import agents as agents_route
from superlocalmemory.server.routes import helpers


def _seed(db_path: str) -> None:
    conn = sqlite3.connect(db_path)
    conn.execute(
        "CREATE TABLE ingestion_operations ("
        "operation_id TEXT, profile_id TEXT, source_type TEXT, "
        "trusted_actor_id TEXT, raw_content TEXT, session_id TEXT, created_at TEXT)"
    )
    rows = [
        ("op1", "default", "mcp", "claude", "first memory by claude", "s1", "2026-07-01T00:00:00Z"),
        ("op2", "default", "mcp", "claude", "second by claude", "s1", "2026-07-02T00:00:00Z"),
        ("op3", "default", "python-api", "gemini", "gemini memory", "s2", "2026-07-03T00:00:00Z"),
        ("op4", "default", "mcp", "", "anonymous memory", "s3", "2026-07-04T00:00:00Z"),
        ("op5", "other", "mcp", "claude", "OTHER profile must be excluded", "s9", "2026-07-05T00:00:00Z"),
    ]
    conn.executemany(
        "INSERT INTO ingestion_operations "
        "(operation_id, profile_id, source_type, trusted_actor_id, raw_content, session_id, created_at) "
        "VALUES (?,?,?,?,?,?,?)",
        rows,
    )
    conn.commit()
    conn.close()


def _call(tmp_path, monkeypatch, limit=20):
    db = tmp_path / "memory.db"
    _seed(str(db))
    monkeypatch.setattr(agents_route, "DB_PATH", db)
    monkeypatch.setattr(helpers, "get_active_profile", lambda: "default")
    return asyncio.run(
        agents_route.get_agent_memory_activity(request=None, limit=limit)
    )


def test_per_agent_grouping_and_totals(tmp_path, monkeypatch):
    res = _call(tmp_path, monkeypatch)
    assert res["ok"] is True
    assert res["total_memories"] == 4  # excludes the 'other' profile row
    by_agent = {a["agent_id"]: a for a in res["agents"]}
    assert by_agent["claude"]["count"] == 2
    assert by_agent["gemini"]["count"] == 1
    assert by_agent["unknown"]["count"] == 1  # empty actor -> unknown bucket
    assert res["agent_count"] == 3


def test_source_types_collected(tmp_path, monkeypatch):
    res = _call(tmp_path, monkeypatch)
    by_agent = {a["agent_id"]: a for a in res["agents"]}
    assert set(by_agent["claude"]["source_types"]) == {"mcp"}
    assert set(by_agent["gemini"]["source_types"]) == {"python-api"}


def test_recent_is_recency_ordered_and_profile_scoped(tmp_path, monkeypatch):
    res = _call(tmp_path, monkeypatch)
    # Newest default-profile op is op4 (2026-07-04) by the unknown agent.
    assert res["recent"][0]["agent_id"] == "unknown"
    assert res["recent"][0]["created_at"] == "2026-07-04T00:00:00Z"
    # The 'other' profile row must never appear.
    assert all("OTHER profile" not in (r["content"] or "") for r in res["recent"])


def test_missing_table_is_graceful(tmp_path, monkeypatch):
    db = tmp_path / "empty.db"
    sqlite3.connect(str(db)).close()  # exists but no ingestion_operations
    monkeypatch.setattr(agents_route, "DB_PATH", db)
    monkeypatch.setattr(helpers, "get_active_profile", lambda: "default")
    res = asyncio.run(agents_route.get_agent_memory_activity(request=None, limit=20))
    assert res["ok"] is True
    assert res["total_memories"] == 0
    assert res["agents"] == [] and res["recent"] == []
