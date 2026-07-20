# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE file

"""V3.7.8 WS1 regression: dashboard routes must never leak cross-profile
memories through a stale ``WorkerPool`` subprocess cache.

Before this fix, ``chat.py``'s ``_recall_memories``, ``memories.py``'s
``get_cluster_detail``/``get_memory_facts``, and ``v3_api.py``'s
consolidation-trigger fallback all read from ``WorkerPool.shared()`` — a
long-lived subprocess that caches its OWN ``MemoryEngine`` + ``profile_id``
at process init and is never recycled on a profile switch. A switch could
therefore serve the OLD profile's memories to these routes for up to 120s.
This module proves the migrated routes read the daemon's resident,
lease-protected engine instead — which ``commit_daemon_profile_switch``
rebinds synchronously — and that ``WorkerPool`` is never touched by them.
"""

from __future__ import annotations

import asyncio
import json

import pytest
from fastapi.testclient import TestClient
from types import SimpleNamespace
from unittest.mock import MagicMock


@pytest.fixture(autouse=True)
def _fresh_worker_pool_mock(monkeypatch):
    """Per-test ``WorkerPool.shared()`` mock with clean call history.

    The session-scoped conftest fixture (``_prevent_heavy_model_loading``)
    patches ``WorkerPool.shared`` to ONE shared mock for the whole session, so
    its ``.recall``/``.get_memory_facts``/``.summarize`` call history accumulates
    across every test. These isolation tests assert that leak-prone WorkerPool
    path is never touched; without a per-test reset that assertion is
    order-dependent (green alone, red in the full suite). A fresh mock per test
    makes "WorkerPool was never used" a deterministic, self-contained check.
    """
    from superlocalmemory.core.worker_pool import WorkerPool

    fresh = MagicMock()
    fresh.recall.return_value = {"ok": True, "results": [], "count": 0}
    monkeypatch.setattr(WorkerPool, "shared", staticmethod(lambda: fresh))
    return fresh


def _add_profile(engine, profile_id: str) -> None:
    engine._db.execute(
        "INSERT OR IGNORE INTO profiles (profile_id, name) VALUES (?, ?)",
        (profile_id, profile_id),
    )
    from superlocalmemory.server.routes.helpers import ensure_profile_in_json

    ensure_profile_in_json(profile_id)


def _daemon_headers(app) -> dict[str, str]:
    descriptor = app.state.daemon_descriptor
    return {
        "X-SLM-Daemon-Capability": descriptor.capability,
        "X-SLM-Target-Instance": descriptor.instance_id,
    }


def _sse_tokens(text: str) -> str:
    """Concatenate every ``token``/``citation`` SSE payload in a response."""
    chunks = []
    for block in text.split("\n\n"):
        if "data: " not in block:
            continue
        _, _, data = block.partition("data: ")
        chunks.append(data)
    return " ".join(chunks)


def test_chat_recall_isolates_across_profile_switch(
    engine_with_mock_deps, monkeypatch,
) -> None:
    """/api/v3/chat/stream must reflect the CURRENT profile, not a stale one."""
    from superlocalmemory.core.worker_pool import WorkerPool
    from superlocalmemory.server.unified_daemon import create_app

    engine = engine_with_mock_deps
    engine.profile_id = "alpha"
    engine._config.active_profile = "alpha"
    _add_profile(engine, "alpha")
    _add_profile(engine, "beta")
    app = create_app()
    app.state.engine = engine
    app.state.config = engine._config
    client = TestClient(app)
    headers = _daemon_headers(app)

    alpha_write = client.post(
        "/remember?wait=true",
        json={"content": "alpha-chat-isolation-token-01 belongs only to alpha."},
        headers=headers,
    )
    assert alpha_write.status_code == 200, alpha_write.text

    switched = client.post("/api/profiles/beta/switch").json()
    assert switched["active_profile"] == "beta"

    beta_write = client.post(
        "/remember?wait=true",
        json={"content": "beta-chat-isolation-token-02 belongs only to beta."},
        headers=headers,
    )
    assert beta_write.status_code == 200, beta_write.text

    stream = client.post(
        "/api/v3/chat/stream",
        json={"query": "isolation-token", "mode": "a", "limit": 10},
    )
    assert stream.status_code == 200
    payload = _sse_tokens(stream.text)

    assert "beta-chat-isolation-token-02" in payload
    assert "alpha-chat-isolation-token-01" not in payload

    # The stale-subprocess cache that caused the leak must never be touched.
    WorkerPool.shared().recall.assert_not_called()


def test_recall_via_resident_engine_fallback_reads_current_profile(
    engine_with_mock_deps,
) -> None:
    """Direct unit coverage of the no-adapter fallback path in chat.py."""
    from superlocalmemory.core.worker_pool import WorkerPool
    from superlocalmemory.server.routes import chat as chat_module
    from superlocalmemory.server.profile_runtime import (
        ProfileSnapshot,
        commit_daemon_profile_switch,
    )

    engine = engine_with_mock_deps
    engine.profile_id = "alpha"
    engine._config.active_profile = "alpha"
    _add_profile(engine, "alpha")
    _add_profile(engine, "beta")

    from superlocalmemory.core.engine_ingestion import canonical_store, local_trusted_actor_id

    canonical_store(
        engine,
        "alpha-fallback-token-11 is alpha-only.",
        source_type="test",
        trusted_actor_id=local_trusted_actor_id("test"),
    )

    app_state = SimpleNamespace(
        engine=engine, config=engine._config, engine_recall_adapter=None,
    )

    # Switch the resident engine to beta the same way the daemon does —
    # commit_daemon_profile_switch rebinds engine.profile_id in place.
    commit_daemon_profile_switch(app_state, ProfileSnapshot("alpha", 0), "beta")
    assert engine.profile_id == "beta"

    canonical_store(
        engine,
        "beta-fallback-token-22 is beta-only.",
        source_type="test",
        trusted_actor_id=local_trusted_actor_id("test"),
    )

    results = chat_module._recall_memories(app_state, "fallback-token", limit=10)
    contents = " ".join(r.get("content", "") for r in results)

    assert "beta-fallback-token-22" in contents
    assert "alpha-fallback-token-11" not in contents
    WorkerPool.shared().recall.assert_not_called()


def test_chat_recall_prefers_engine_recall_adapter_when_present() -> None:
    """When the daemon's queue-consumer adapter exists, it is used verbatim."""
    from superlocalmemory.core.worker_pool import WorkerPool
    from superlocalmemory.server.routes import chat as chat_module

    adapter = MagicMock()
    adapter.recall.return_value = {
        "ok": True,
        "results": [{"fact_id": "f1", "content": "adapter-routed-content"}],
    }
    app_state = SimpleNamespace(engine_recall_adapter=adapter)

    results = chat_module._recall_memories(app_state, "query", limit=5)

    assert results == [{"fact_id": "f1", "content": "adapter-routed-content"}]
    adapter.recall.assert_called_once_with("query", limit=5)
    WorkerPool.shared().recall.assert_not_called()


def test_memory_facts_route_isolates_across_profile_switch(
    engine_with_mock_deps,
) -> None:
    """/api/memories/{id}/facts must never serve a different profile's facts."""
    from superlocalmemory.core.worker_pool import WorkerPool
    from superlocalmemory.server.unified_daemon import create_app

    engine = engine_with_mock_deps
    engine.profile_id = "alpha"
    engine._config.active_profile = "alpha"
    _add_profile(engine, "alpha")
    _add_profile(engine, "beta")
    app = create_app()
    app.state.engine = engine
    app.state.config = engine._config
    client = TestClient(app)
    headers = _daemon_headers(app)

    alpha_write = client.post(
        "/remember?wait=true",
        json={"content": "alpha-facts-isolation-token-33 is alpha-only."},
        headers=headers,
    )
    assert alpha_write.status_code == 200, alpha_write.text
    alpha_fact_id = alpha_write.json()["fact_ids"][0]
    rows = engine._db.execute(
        "SELECT memory_id FROM atomic_facts WHERE fact_id = ?", (alpha_fact_id,),
    )
    memory_id = dict(rows[0])["memory_id"]

    switched = client.post("/api/profiles/beta/switch").json()
    assert switched["active_profile"] == "beta"

    facts_after_switch = client.get(f"/api/memories/{memory_id}/facts")
    assert facts_after_switch.status_code == 200, facts_after_switch.text
    body = facts_after_switch.json()
    contents = " ".join(f.get("content", "") for f in body["facts"])
    assert "alpha-facts-isolation-token-33" not in contents
    assert body["fact_count"] == 0

    switched_back = client.post("/api/profiles/alpha/switch").json()
    assert switched_back["active_profile"] == "alpha"
    facts_alpha = client.get(f"/api/memories/{memory_id}/facts").json()
    alpha_contents = " ".join(f.get("content", "") for f in facts_alpha["facts"])
    assert "alpha-facts-isolation-token-33" in alpha_contents

    WorkerPool.shared().get_memory_facts.assert_not_called()


def test_cluster_summary_never_uses_worker_pool(engine_with_mock_deps) -> None:
    """get_cluster_detail's summary generation must not use the stale worker."""
    from superlocalmemory.core.worker_pool import WorkerPool
    from superlocalmemory.server.routes import memories as memories_module

    engine = engine_with_mock_deps
    engine.profile_id = "alpha"
    engine._config.active_profile = "alpha"
    _add_profile(engine, "alpha")

    from superlocalmemory.core.engine_ingestion import canonical_store, local_trusted_actor_id

    fact_ids = canonical_store(
        engine,
        "cluster-summary-token-44 describes a themed memory.",
        source_type="test",
        trusted_actor_id=local_trusted_actor_id("test"),
    )
    fact_id = fact_ids[0]
    engine._db.execute(
        "INSERT INTO memory_scenes (scene_id, profile_id, theme, fact_ids_json) "
        "VALUES (?, ?, ?, ?)",
        ("scene-1", "alpha", "test-theme", json.dumps([fact_id])),
    )

    request = MagicMock()
    request.app.state = SimpleNamespace(engine=engine, config=engine._config)

    import sqlite3

    original_get_conn = memories_module.get_db_connection
    original_get_profile = memories_module.get_active_profile
    memories_module.get_db_connection = lambda: sqlite3.connect(
        str(engine._db.db_path)
    )
    memories_module.get_active_profile = lambda: "alpha"
    try:
        result = asyncio.run(
            memories_module.get_cluster_detail(request, "scene-1", limit=50)
        )
    finally:
        memories_module.get_db_connection = original_get_conn
        memories_module.get_active_profile = original_get_profile

    assert result["cluster_info"]["total_members"] == 1
    assert isinstance(result["summary"], str)
    WorkerPool.shared().summarize.assert_not_called()


def test_consolidation_trigger_never_calls_dead_send_command(
    engine_with_mock_deps, monkeypatch,
) -> None:
    """SEC-M-01: the removed WorkerPool.send_command dead call must be gone."""
    import inspect

    from superlocalmemory.server.routes import v3_api

    source = inspect.getsource(v3_api.trigger_consolidation)
    code_lines = [
        line for line in source.splitlines()
        if not line.strip().startswith("#")
    ]
    code_only = "\n".join(code_lines)
    assert "pool.send_command" not in code_only
    assert "WorkerPool" not in code_only
    assert "get_profile_runtime(request.app.state)" in code_only
