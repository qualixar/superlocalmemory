# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE file
# Part of SuperLocalMemory V3
"""POST /api/lifecycle/compact — real compaction (Wave B).

The endpoint was a stub returning {"status":"not_implemented"}. It now
recomputes each fact's lifecycle zone for the active profile: dry_run previews
proposed transitions without mutating; execute applies them.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


def _daemon_headers(app) -> dict[str, str]:
    d = app.state.daemon_descriptor
    return {
        "X-SLM-Daemon-Capability": d.capability,
        "X-SLM-Target-Instance": d.instance_id,
    }


def _seed_old_fact(engine, profile_id: str, fid: str) -> None:
    """Insert a warm-tier fact with a very old created_at so time-based
    lifecycle recomputation has a transition to propose."""
    mem_id = f"mem-{fid}"
    engine._db.execute(
        "INSERT INTO memories (memory_id, profile_id, content, session_id, "
        " speaker, role, created_at, metadata_json, scope) "
        "VALUES (?,?,?,?,?,?,?,?,?)",
        (mem_id, profile_id, "old memory", "s1", "user", "user",
         "2020-01-01T00:00:00Z", "{}", "personal"),
    )
    engine._db.execute(
        "INSERT INTO atomic_facts (fact_id, memory_id, profile_id, content, "
        " lifecycle, created_at) VALUES (?,?,?,?,?,?)",
        (fid, mem_id, profile_id, "old fact", "active", "2020-01-01T00:00:00Z"),
    )


@pytest.fixture
def client(engine_with_mock_deps):
    from superlocalmemory.server.profile_runtime import bind_profile_runtime
    from superlocalmemory.server.unified_daemon import create_app

    engine = engine_with_mock_deps
    engine.profile_id = "default"
    engine._config.active_profile = "default"
    engine._db.execute(
        "INSERT OR IGNORE INTO profiles (profile_id, name) VALUES ('default','default')"
    )
    _seed_old_fact(engine, "default", "old1")
    _seed_old_fact(engine, "default", "old2")

    app = create_app()
    app.state.engine = engine
    app.state.config = engine._config
    bind_profile_runtime(app.state, engine, engine._config)
    return TestClient(app), _daemon_headers(app)


def test_compact_dry_run_previews_without_mutating(client):
    tc, headers = client
    r = tc.post("/api/lifecycle/compact", json={"dry_run": True}, headers=headers)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["success"] is True
    assert body["dry_run"] is True
    assert body["applied"] == 0, "dry_run must not mutate"
    assert body["total_facts"] >= 2
    assert isinstance(body["transitions"], list)


def test_compact_execute_applies_and_converges(client):
    tc, headers = client
    dry = tc.post("/api/lifecycle/compact", json={"dry_run": True},
                  headers=headers).json()
    ex = tc.post("/api/lifecycle/compact", json={"dry_run": False},
                 headers=headers).json()
    assert ex["success"] is True
    assert ex["applied"] == dry["candidates"], "execute applies all previewed"
    # Re-running finds nothing left → converged + idempotent.
    again = tc.post("/api/lifecycle/compact", json={"dry_run": True},
                    headers=headers).json()
    assert again["candidates"] == 0
