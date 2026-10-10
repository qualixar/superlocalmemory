# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3

"""Tests for Wave Q2b — community summary surfaced as thematic context.

Engine attaches a precomputed community summary when the top results cluster
in one community (gated, fail-open, no per-query LLM). The serializer passes
it through as `thematic_context`.

Q9 (2026-10-06): a 3,816-member community made `member_fact_ids` 73% of a
104KB session_init response — the full stored membership was echoed into
every recall. `member_fact_ids` is now a bounded, query-relevant sample;
`member_count` and `member_fact_ids_truncated` say the rest honestly instead
of silently dropping it.
"""

from __future__ import annotations

import json
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
        # Sample is bounded to the ids THIS query actually matched, not the
        # whole stored membership — "f3" never appeared in the results.
        assert ctx["member_fact_ids"] == ["f1", "f2"]
        assert ctx["member_count"] == 3
        assert ctx["member_fact_ids_truncated"] is True

    def test_sample_bounded_on_a_large_community(self) -> None:
        """Q9 regression: a 4,000-member community never floods the response.

        Before the fix, `member_fact_ids` was `json.loads(fact_ids_json)` —
        the entire stored membership. 15 of the top results fall in one
        community of 4,000; the sample must cap at 10, never 15 or 4,000.
        """
        db = MagicMock()
        big_members = [f"m{i}" for i in range(4000)]
        db.execute.return_value = [
            {"community_id": 0, "summary": "A very large community.",
             "keywords": "big", "fact_ids_json": json.dumps(big_members),
             "fact_count": 4000},
        ]
        eng = _engine(db)
        results = [_result(f"m{i}") for i in range(15)]
        ctx = eng._community_context(results, "default", top_k=20)
        assert ctx is not None
        assert len(ctx["member_fact_ids"]) == 10
        assert ctx["member_count"] == 4000
        assert ctx["member_fact_ids_truncated"] is True

    def test_member_count_falls_back_to_a_reparse_without_fact_count(self) -> None:
        """An older row with no `fact_count` column still reports a real count."""
        db = MagicMock()
        db.execute.return_value = [
            {"community_id": 0, "summary": "Work at Accenture.",
             "keywords": "accenture, varun", "fact_ids_json": '["f1", "f2", "f3"]',
             "fact_count": 0},
        ]
        eng = _engine(db)
        results = [_result("f1"), _result("f2")]
        ctx = eng._community_context(results, "default")
        assert ctx is not None
        assert ctx["member_count"] == 3
        assert ctx["member_fact_ids_truncated"] is True

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


class TestBuildCommunityContextModuleDirect:
    """Q9 (2026-10-06): the matching/sample logic lives in its own module
    (``retrieval.community_context``) now, extracted out of engine.py so
    that already-over-the-cap file does not grow. Covered above through
    ``RetrievalEngine._community_context`` (the real caller); these confirm
    the module also works correctly called directly, with no engine needed.
    """

    def test_direct_call_matches_the_engine_delegator(self) -> None:
        from superlocalmemory.retrieval.community_context import (
            MAX_MEMBER_SAMPLE,
            build_community_context,
        )

        db = MagicMock()
        db.execute.return_value = _summary_rows()
        results = [_result("f1"), _result("f2"), _result("zz")]

        direct = build_community_context(db, results, "default")
        via_engine = _engine(db)._community_context(results, "default")

        assert direct == via_engine
        assert MAX_MEMBER_SAMPLE == 10

    def test_direct_call_has_no_engine_config_gate(self) -> None:
        """The module itself does not read ``enable_community_context`` —
        gating on config is the engine wrapper's job, by design."""
        from superlocalmemory.retrieval.community_context import (
            build_community_context,
        )

        db = MagicMock()
        db.execute.return_value = _summary_rows()
        results = [_result("f1"), _result("f2"), _result("zz")]

        assert build_community_context(db, results, "default") is not None


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


class TestCommunityContextUnderAView:
    def _db(self) -> MagicMock:
        db = MagicMock()
        db.execute.return_value = _summary_rows()
        return db

    def test_a_hidden_member_withholds_the_summary(self) -> None:
        from superlocalmemory.retrieval import visibility

        results = [_result("f1"), _result("f2"), _result("zz")]
        ctx = visibility.VisibilityContext(hidden_fact_ids=frozenset({"f3"}))
        with visibility.use(ctx):
            assert _engine(self._db())._community_context(results, "default") is None

    def test_a_view_that_hides_none_of_the_members_keeps_it(self) -> None:
        from superlocalmemory.retrieval import visibility

        results = [_result("f1"), _result("f2"), _result("zz")]
        ctx = visibility.VisibilityContext(hidden_fact_ids=frozenset({"other"}))
        with visibility.use(ctx):
            assert _engine(self._db())._community_context(results, "default")["community_id"] == 0
