# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Typed behavioral-entity extraction from recall results."""

from superlocalmemory.core.recall_pipeline import _behavioral_entities
from superlocalmemory.storage.models import AtomicFact, RetrievalResult


def test_behavioral_entities_come_from_typed_retrieval_facts() -> None:
    results = [
        RetrievalResult(
            fact=AtomicFact(
                fact_id="fact-1",
                canonical_entities=["entity-alice", "entity-project"],
            )
        ),
        RetrievalResult(
            fact=AtomicFact(
                fact_id="fact-2",
                canonical_entities=["entity-alice", "entity-release"],
            )
        ),
    ]

    assert _behavioral_entities(results) == [
        "entity-alice",
        "entity-project",
        "entity-release",
    ]


def test_behavioral_entities_ignore_empty_or_non_string_values() -> None:
    result = RetrievalResult(
        fact=AtomicFact(
            fact_id="fact-1",
            canonical_entities=["", "entity-valid", None],  # type: ignore[list-item]
        )
    )

    assert _behavioral_entities([result]) == ["entity-valid"]
