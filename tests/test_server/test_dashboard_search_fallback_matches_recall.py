# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""The dashboard's over-budget /api/search answer must find what the daemon's
/recall fallback finds (4.1.20 WP10).

It matched the whole question as one substring, so a natural question never
matched the saved sentence, while the CLI and MCP (daemon /recall) did.
"""

from __future__ import annotations

import time

from fastapi.testclient import TestClient

from superlocalmemory.retrieval import channel_status as chstat


def test_dashboard_fallback_finds_a_natural_question(
    engine_with_mock_deps, monkeypatch,
) -> None:
    from superlocalmemory.core.engine_ingestion import (
        canonical_store,
        local_trusted_actor_id,
    )
    from superlocalmemory.server.routes.helpers import ensure_profile_in_json
    from superlocalmemory.server.unified_daemon import (
        _recall_keyword_fallback,
        create_app,
    )

    engine = engine_with_mock_deps
    engine._db.execute(
        "INSERT OR IGNORE INTO profiles (profile_id, name) VALUES (?, ?)",
        ("default", "default"),
    )
    ensure_profile_in_json("default")
    engine.profile_id = "default"
    engine._config.active_profile = "default"
    for text in ("The migration window for Project Halcyon is 14 March",
                 "Lunch order: two coffees"):
        canonical_store(engine, text, source_type="test",
                        trusted_actor_id=local_trusted_actor_id("test"))
    question = "When is the Halcyon migration window?"
    daemon_answer = _recall_keyword_fallback(engine, question, 5)

    monkeypatch.setenv("SLM_SEARCH_RECALL_TIMEOUT_S", "0.1")

    def _blocking_recall(*_a, **_k):
        time.sleep(3.0)
        return []

    engine.recall = _blocking_recall
    app = create_app()
    app.state.engine = engine
    app.state.config = engine._config
    resp = TestClient(app).post("/api/search", json={"query": question, "limit": 5})

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["retrieval_mode"] == "degraded_lexical"
    contents = [r["content"] for r in body["results"]]
    assert contents and "Halcyon" in contents[0], body
    # Same rows, same order as the daemon's fallback (CLI and MCP).
    assert [r["fact_id"] for r in body["results"]] == [
        r["fact_id"] for r in daemon_answer["results"]]
    # And it says every channel was skipped.
    assert body["incomplete_channels"] == sorted(chstat.CHANNEL_NAMES)
