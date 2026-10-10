# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""POST /api/search (the dashboard's search) marks a result that came from a picture
or a document page with a ``media`` block, as GET /recall does, so the dashboard's
"Find a picture" box can show the picture. Text-only results carry no block."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient

from superlocalmemory.server.unified_daemon import create_app
from superlocalmemory.storage.models import AtomicFact, FactType

MEDIA_ID = "a" * 32


def _result(fact_id, content):
    fact = AtomicFact(fact_id=fact_id, memory_id=f"m-{fact_id}", content=content,
                      fact_type=FactType.SEMANTIC)
    return SimpleNamespace(fact=fact, score=0.9, relevance_score=0.9, confidence=0.9,
                           memory_confidence=0.9, ranking_score=0.9, rank_position=1,
                           trust_score=0.5, channel_scores={}, evidence_chain=[])


def _recall(*_args, **_kwargs):
    results = [_result("f-pic", "quarterly numbers slide"), _result("f-txt", "plain note")]
    return SimpleNamespace(results=results, query="q", query_type="lookup",
                           retrieval_time_ms=1.0, channel_weights={},
                           total_candidates=2, no_confident_match=False)


def test_search_results_from_a_picture_carry_a_media_block(engine_with_mock_deps) -> None:
    engine = engine_with_mock_deps
    engine.recall = _recall
    app = create_app()
    app.state.engine = engine
    app.state.config = engine._config
    seen: list = []

    def sources(_db, memory_ids):
        seen.append(set(memory_ids))
        return {"m-f-pic": {"type": "media", "media_id": MEDIA_ID}}

    with patch("superlocalmemory.retrieval.media_channel.memory_sources", sources):
        r = TestClient(app).post("/api/search", json={"query": "quarterly numbers", "limit": 10})
    assert r.status_code == 200, r.text
    by_id = {item["fact_id"]: item for item in r.json()["results"]}
    assert by_id["f-pic"]["media"]["media_id"] == MEDIA_ID
    assert by_id["f-pic"]["media"]["kind"] == "image"
    assert "media" not in by_id["f-txt"]
    assert seen == [{"m-f-pic", "m-f-txt"}]
