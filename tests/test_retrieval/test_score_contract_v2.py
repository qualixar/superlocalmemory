# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""The public score contract separates relevance, confidence, and rank utility."""

from __future__ import annotations

import inspect

from superlocalmemory.storage.models import AtomicFact, RecallResponse, RetrievalResult


def _result(*, score: float, fact_confidence: float) -> RetrievalResult:
    return RetrievalResult(
        fact=AtomicFact(content="A grounded memory", confidence=fact_confidence),
        score=score,
        confidence=1.0,
    )


def test_memory_confidence_is_fact_confidence_not_relevance() -> None:
    from superlocalmemory.core.score_contract import finalize_score_contract

    response = RecallResponse(results=[_result(score=0.2, fact_confidence=0.37)])
    finalize_score_contract(response)
    result = response.results[0]

    assert result.relevance_score == 0.2
    assert result.score == result.relevance_score
    assert result.memory_confidence == 0.37
    assert result.confidence == result.memory_confidence
    assert result.confidence != 1.0


def test_rank_position_matches_final_order() -> None:
    from superlocalmemory.core.score_contract import finalize_score_contract

    response = RecallResponse(results=[
        _result(score=0.3, fact_confidence=0.4),
        _result(score=0.8, fact_confidence=0.9),
    ])
    finalize_score_contract(response)
    assert [result.rank_position for result in response.results] == [1, 2]


def test_uncalibrated_retrieval_never_claims_answer_confidence() -> None:
    from superlocalmemory.core.score_contract import finalize_score_contract

    response = RecallResponse(results=[_result(score=0.6, fact_confidence=0.8)])
    finalize_score_contract(response)

    assert response.score_contract_version == "2"
    assert response.calibration_status == "uncalibrated"
    assert response.calibration_id is None
    assert response.answer_confidence is None
    assert response.abstained is False
    assert response.abstention_reason is None


def test_empty_candidates_abstain_with_truthful_reason() -> None:
    from superlocalmemory.core.score_contract import finalize_score_contract

    no_candidates = RecallResponse(results=[])
    evidence_floor = RecallResponse(results=[], no_confident_match=True)
    finalize_score_contract(no_candidates)
    finalize_score_contract(evidence_floor)

    assert no_candidates.abstained is True
    assert no_candidates.abstention_reason == "no_candidates"
    assert evidence_floor.abstained is True
    assert evidence_floor.abstention_reason == "evidence_floor"


def test_serializer_exposes_aliases_and_canonical_fields() -> None:
    from superlocalmemory.core.score_contract import finalize_score_contract
    from superlocalmemory.server.recall_serializer import (
        recall_response_metadata,
        serialize_recall_response,
    )

    response = RecallResponse(results=[_result(score=0.42, fact_confidence=0.73)])
    finalize_score_contract(response)
    items, _ = serialize_recall_response(response)
    item = items[0]

    assert item["score"] == item["relevance_score"] == 0.42
    assert item["confidence"] == item["memory_confidence"] == 0.73
    assert item["ranking_score"] is None
    assert item["rank_position"] == 1
    assert recall_response_metadata(response) == {
        "score_contract_version": "2",
        "calibration_status": "uncalibrated",
        "calibration_id": None,
        "answer_confidence": None,
        "abstained": False,
        "abstention_reason": None,
    }


def test_run_recall_no_longer_derives_confidence_from_score() -> None:
    from superlocalmemory.core import recall_pipeline

    source = inspect.getsource(recall_pipeline.run_recall)
    assert "r.confidence = min(1.0, r.score * 2.0)" not in source
    assert "finalize_score_contract" in source
