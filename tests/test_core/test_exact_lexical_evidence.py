"""Exact lexical evidence must survive learned reranking."""

from __future__ import annotations

from superlocalmemory.core.recall_pipeline import _preserve_exact_lexical_evidence
from superlocalmemory.storage.models import (
    AtomicFact,
    Mode,
    RecallResponse,
    RetrievalResult,
)


def _result(
    fact_id: str,
    content: str,
    *,
    semantic: float = 0.0,
    bm25: float = 0.0,
) -> RetrievalResult:
    return RetrievalResult(
        fact=AtomicFact(
            fact_id=fact_id,
            memory_id=f"memory-{fact_id}",
            content=content,
        ),
        score=semantic,
        channel_scores={"semantic": semantic, "bm25": bm25},
        confidence=0.7,
    )


def test_exact_bm25_hit_is_restored_to_rank_one() -> None:
    semantic_noise = _result(
        "noise",
        "A semantically similar but different session identifier.",
        semantic=0.91,
    )
    exact = _result(
        "exact",
        "Release canary amber-lattice-20260727-1048 is durable.",
        bm25=12.0,
    )
    response = RecallResponse(
        query="amber-lattice-20260727-1048",
        mode=Mode.A,
        results=[semantic_noise, exact],
    )

    _preserve_exact_lexical_evidence(
        response,
        "amber-lattice-20260727-1048",
    )

    assert [result.fact.fact_id for result in response.results] == [
        "exact",
        "noise",
    ]


def test_substring_without_bm25_evidence_is_not_promoted() -> None:
    first = _result("first", "ranked first", semantic=0.8)
    unsupported = _result(
        "unsupported",
        "Contains amber-lattice-20260727-1048 but has no lexical evidence.",
        semantic=0.4,
    )
    response = RecallResponse(
        query="amber-lattice-20260727-1048",
        mode=Mode.A,
        results=[first, unsupported],
    )

    _preserve_exact_lexical_evidence(
        response,
        "amber-lattice-20260727-1048",
    )

    assert [result.fact.fact_id for result in response.results] == [
        "first",
        "unsupported",
    ]
