# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3

"""Tests for Wave Q2b — community summary surfaced as thematic context.

Engine attaches a precomputed community summary when the top results cluster
in one community (gated, fail-open, no per-query LLM). The serializer passes
it through as `thematic_context`.
"""

from __future__ import annotations

import types
from unittest.mock import MagicMock

from superlocalmemory.core.config import RetrievalConfig
from superlocalmemory.retrieval.engine import RetrievalEngine
from superlocalmemory.server.recall_serializer import recall_response_metadata


def _result(fid: str):
    return types.SimpleNamespace(fact=types.SimpleNamespace(fact_id=fid))


def _engine(db: MagicMock, config: RetrievalConfig | None = None) -> RetrievalEngine:
    return RetrievalEngine(
        db=db, config=config or RetrievalConfig(), channels={},
    )


def _summary_rows():
    return [
        {"community_id": 0, "summary": "Work at Accenture.",
         "keywords": "accenture, varun", "fact_ids_json": '["f1", "f2", "f3"]',
         "fact_count": 3},
        {"community_id": 1, "summary": "Paris vacation.",
         "keywords": "paris, vacation", "fact_ids_json": '["f8", "f9"]',
         "fact_count": 2},
    ]


class TestCommunityContext:
    def test_attached_when_top_results_cluster(self) -> None:
        db = MagicMock()
        db.execute.return_value = _summary_rows()
        eng = _engine(db)
        # 2 of 3 top results belong to community 0 -> coverage 0.67, count 2.
        results = [_result("f1"), _result("f2"), _result("zz")]
        ctx = eng._community_context(results, "default")
        assert ctx is not None
        assert ctx["community_id"] == 0
        assert ctx["summary"] == "Work at Accenture."
        assert ctx["matched_results"] == 2
        assert ctx["member_fact_ids"] == ["f1", "f2", "f3"]  # drill-down handle

    def test_none_below_threshold(self) -> None:
        db = MagicMock()
        db.execute.return_value = _summary_rows()
        eng = _engine(db)
        # Only 1 of 5 top results in a community -> count 1 (<2) -> None.
        results = [_result("f1"), _result("a"), _result("b"), _result("c"), _result("d")]
        assert eng._community_context(results, "default") is None

    def test_disabled_by_config(self) -> None:
        db = MagicMock()
        db.execute.return_value = _summary_rows()
        eng = _engine(db, RetrievalConfig(enable_community_context=False))
        results = [_result("f1"), _result("f2"), _result("f3")]
        assert eng._community_context(results, "default") is None
        db.execute.assert_not_called()

    def test_no_summaries_returns_none(self) -> None:
        db = MagicMock()
        db.execute.return_value = []
        eng = _engine(db)
        assert eng._community_context([_result("f1")], "default") is None

    def test_fail_open_on_db_error(self) -> None:
        db = MagicMock()
        db.execute.side_effect = RuntimeError("db down")
        eng = _engine(db)
        assert eng._community_context([_result("f1"), _result("f2")], "default") is None

    def test_empty_results_returns_none(self) -> None:
        db = MagicMock()
        eng = _engine(db)
        assert eng._community_context([], "default") is None
        db.execute.assert_not_called()


class TestSerializerPassthrough:
    def test_thematic_context_passthrough(self) -> None:
        resp = types.SimpleNamespace(
            results=[], community_context={"community_id": 1, "summary": "x"},
        )
        md = recall_response_metadata(resp)
        assert md["thematic_context"] == {"community_id": 1, "summary": "x"}

    def test_thematic_context_none_by_default(self) -> None:
        resp = types.SimpleNamespace(results=[])
        md = recall_response_metadata(resp)
        assert md["thematic_context"] is None
