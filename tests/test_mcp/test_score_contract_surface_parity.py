# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Worker, in-process daemon, and pool adapters retain Score Contract v2."""

from __future__ import annotations

from types import SimpleNamespace

from superlocalmemory.core.score_contract import finalize_score_contract
from superlocalmemory.storage.models import AtomicFact, RecallResponse, RetrievalResult


def _response() -> RecallResponse:
    fact = AtomicFact(
        fact_id="fact-1",
        memory_id="memory-1",
        content="Dana approved the recovery plan.",
        confidence=0.71,
    )
    response = RecallResponse(results=[RetrievalResult(
        fact=fact,
        score=0.43,
        confidence=0.71,
        relevance_score=0.43,
        ranking_score=1.275,
        memory_confidence=0.71,
    )])
    return finalize_score_contract(response)


class _Engine:
    def __init__(self) -> None:
        self.response = _response()
        self.profile_id = "default"
        self._db = SimpleNamespace(
            get_memory_content_batch=lambda _ids, *a, **k: {
                "memory-1": "Source memory text",
            },
        )
        self._config = SimpleNamespace(retrieval=SimpleNamespace(
            recall_per_fact_max_chars=2400,
            recall_total_max_chars=12000,
        ))

    def recall(self, *_args, **_kwargs):
        return self.response


def _contract_subset(envelope: dict) -> dict:
    item = envelope["results"][0]
    return {
        "score": item["score"],
        "relevance_score": item["relevance_score"],
        "confidence": item["confidence"],
        "memory_confidence": item["memory_confidence"],
        "ranking_score": item["ranking_score"],
        "rank_position": item["rank_position"],
        "score_contract_version": envelope["score_contract_version"],
        "calibration_status": envelope["calibration_status"],
        "answer_confidence": envelope["answer_confidence"],
    }


def test_worker_and_in_process_daemon_have_score_surface_parity(monkeypatch) -> None:
    from superlocalmemory.core import recall_worker
    from superlocalmemory.server.unified_daemon import EngineRecallAdapter

    engine = _Engine()
    monkeypatch.setattr(recall_worker, "_get_engine", lambda: engine)

    worker = recall_worker._handle_recall("recovery", 10)
    daemon = EngineRecallAdapter(engine).recall("recovery", 10)

    assert _contract_subset(worker) == _contract_subset(daemon)


def test_pool_adapter_preserves_score_fields_and_response_metadata(
    monkeypatch,
) -> None:
    from superlocalmemory.mcp import _pool_adapter

    envelope = {
        "ok": True,
        "results": [{
            "fact_id": "fact-1",
            "content": "grounded",
            "score": 0.4,
            "relevance_score": 0.4,
            "confidence": 0.6,
            "memory_confidence": 0.6,
            "ranking_score": 1.2,
            "rank_position": 1,
        }],
        "score_contract_version": "2",
        "calibration_status": "uncalibrated",
        "calibration_id": None,
        "answer_confidence": None,
        "abstained": False,
        "abstention_reason": None,
    }
    pool = SimpleNamespace(recall=lambda **_kwargs: envelope)
    monkeypatch.setattr(_pool_adapter, "_pool", lambda: pool)

    response = _pool_adapter.pool_recall("grounded")
    result = response.results[0]

    assert result.score == result.relevance_score == 0.4
    assert result.confidence == result.memory_confidence == 0.6
    assert result.ranking_score == 1.2
    assert result.rank_position == 1
    assert response.score_contract_version == "2"
    assert response.answer_confidence is None
