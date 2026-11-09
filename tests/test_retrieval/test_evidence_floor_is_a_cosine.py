# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""The evidence floor is a raw cosine; the semantic channel stores (cosine + 1) / 2.

``min_semantic_evidence`` is documented as a minimum cosine similarity (0.60), but
the semantic channel reports every score on the [0.5, 1.0] scale ``(cos + 1) / 2``
(``semantic_channel.py``, M-01). Comparing the two directly made the floor fire at
raw cosine 0.20, which unrelated text passes every time: measured on a 26k-fact
store with the built-in model, no-answer questions reach raw 0.53-0.61 against
random facts while true answers sit at 0.62-0.90. So recall never abstained.

The floor is converted to the channel's scale in one place, ``has_primary_evidence``.
"""

from __future__ import annotations

import pytest

from superlocalmemory.retrieval.fusion import FusionResult
from superlocalmemory.retrieval.time_window import (
    has_primary_evidence,
    semantic_floor_on_channel_scale,
    windowed_candidates,
)
from superlocalmemory.storage.models import Mode
from tests.test_retrieval.test_evidence_floor import _build_engine, _make_fact

FLOOR = 0.60  # raw cosine, the shipped default


def test_the_floor_is_converted_to_the_channel_scale() -> None:
    assert semantic_floor_on_channel_scale(0.60) == pytest.approx(0.80)
    assert semantic_floor_on_channel_scale(0.0) == pytest.approx(0.50)


@pytest.mark.parametrize("stored, kept", [
    (0.76, False),   # raw 0.52: what unrelated text scores
    (0.79, False),   # raw 0.58: just under the floor
    (0.80, True),    # raw 0.60: exactly the floor
    (0.875, True),   # raw 0.75: a true answer
])
def test_semantic_alone_counts_only_from_the_floor_cosine(stored: float, kept: bool) -> None:
    assert has_primary_evidence({"semantic": stored}, FLOOR) is kept


def test_keyword_entity_and_time_evidence_still_count_on_their_own() -> None:
    assert has_primary_evidence({"semantic": 0.70, "bm25": 0.4}, FLOOR)
    assert has_primary_evidence({"semantic": 0.70, "entity_graph": 0.2}, FLOOR)
    assert has_primary_evidence({"semantic": 0.70, "temporal": 0.1}, FLOOR)
    assert not has_primary_evidence({"semantic": 0.70, "hopfield": 0.9, "spreading_activation": 0.9}, FLOOR)


def test_an_engine_drops_a_candidate_that_only_scored_like_unrelated_text() -> None:
    facts = [_make_fact("f1", "the dashboard favicon was replaced")]
    engine = _build_engine(facts=facts, semantic_results=[("f1", 0.76)])
    response = engine.recall("zxqv plorth wimble", "default", Mode.A, limit=5)
    assert response.results == []
    assert response.no_confident_match is True


def test_an_engine_keeps_a_candidate_at_the_floor_cosine() -> None:
    facts = [_make_fact("f1", "my daughter likes pancakes on Sunday mornings")]
    engine = _build_engine(facts=facts, semantic_results=[("f1", 0.80)])
    response = engine.recall("what does my daughter like for breakfast", "default", Mode.A, limit=5)
    assert [r.fact.fact_id for r in response.results] == ["f1"]
    assert response.no_confident_match is False


def test_an_inferred_time_window_uses_the_same_scale() -> None:
    old = FusionResult(fact_id="old", fused_score=1.0, channel_ranks={}, channel_scores={"bm25": 2.0})
    new = FusionResult(fact_id="new", fused_score=1.0, channel_ranks={}, channel_scores={"semantic": 0.72})
    # 0.72 stored is raw 0.44: no evidence inside the window, so the window must not narrow.
    kept = windowed_candidates([old, new], lambda fid: fid == "new", explicit=False, min_semantic=FLOOR)
    assert [fr.fact_id for fr in kept] == ["old", "new"]
