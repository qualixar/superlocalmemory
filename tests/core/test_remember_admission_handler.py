# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Immediate admission is a small, deterministic SQLite transaction.

The receipt path must stay free of model, hook, graph, and projection work.
Those belong to the durable materializer after the queryable receipt commits.
"""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from superlocalmemory.core.engine_ingestion import (
    _prebuilt_fact_payload,
    build_engine_ingestion_command,
)
from superlocalmemory.core.ingestion_command import IngestionRequest, IngestionState
from superlocalmemory.storage.migrations import M018_ingestion_operations
from superlocalmemory.storage.models import AtomicFact, MemoryRecord


class _PoisonDependency:
    """Fails if admission accidentally reaches an enrichment dependency."""

    def __getattr__(self, name: str):
        raise AssertionError(f"forbidden admission dependency accessed: {name}")


def _install_m018(engine) -> None:
    with engine._db.raw_connection() as conn:
        M018_ingestion_operations.apply(conn)


def test_immediate_admission_is_plain_sql_and_is_fts_queryable(
    engine_with_mock_deps,
    monkeypatch,
) -> None:
    """The receipt projection writes one raw memory and one embedding-free fact."""
    engine = engine_with_mock_deps
    _install_m018(engine)
    command = build_engine_ingestion_command(engine)

    forbidden = Mock(side_effect=AssertionError("forbidden during admission transaction"))
    monkeypatch.setattr(engine, "store_fast", forbidden)
    monkeypatch.setattr(engine._embedder, "embed", forbidden)
    hook_calls = []

    def validate_before_transaction(_operation, context):
        assert getattr(engine._db._txn_state, "conn", None) is None
        hook_calls.append(context)

    monkeypatch.setattr(engine._hooks, "run_pre", validate_before_transaction)
    monkeypatch.setattr(engine._hooks, "run_post", forbidden)
    monkeypatch.setattr(engine._graph_builder, "add_fact", forbidden, raising=False)
    for attribute in (
        "_fact_extractor", "_entity_resolver", "_provenance", "_ann_index",
        "_vector_store", "_context_generator", "_consolidation_engine",
    ):
        monkeypatch.setattr(engine, attribute, _PoisonDependency())

    receipt = command.submit(IngestionRequest(
        content="Aster owns the deterministic admission release gate.",
        profile_id=engine._profile_id,
        source_type="mcp",
        idempotency_key="admission-handler:plain-sql",
        trusted_actor_id="daemon-capability:admission-test",
        session_id="session-admission",
        session_date="2026-07-27",
        speaker="Varun",
        role="user",
    ))

    assert receipt.state is IngestionState.QUERYABLE
    assert forbidden.call_count == 0
    assert hook_calls[0]["agent_id"] == "daemon-capability:admission-test"
    assert len(receipt.queryable_fact_ids) == 1

    fact = engine._db.get_fact(receipt.queryable_fact_ids[0])
    assert fact is not None
    assert fact.content == "Aster owns the deterministic admission release gate."
    assert fact.embedding is None
    assert fact.fisher_mean is None
    assert fact.fisher_variance is None
    assert fact.session_id == "session-admission"
    assert fact.observation_date == "2026-07-27"

    matches = engine._db.search_facts_fts("deterministic admission", engine._profile_id)
    assert [match.fact_id for match in matches] == [fact.fact_id]


def test_immediate_admission_rejects_wrong_profile_and_missing_actor(
    engine_with_mock_deps,
) -> None:
    engine = engine_with_mock_deps
    _install_m018(engine)
    command = build_engine_ingestion_command(engine)

    with pytest.raises(ValueError, match="profile does not match"):
        command.submit(IngestionRequest(
            content="Profile boundaries are part of admission.",
            profile_id="foreign",
            source_type="mcp",
            idempotency_key="admission-handler:foreign-profile",
            trusted_actor_id="daemon-capability:admission-test",
        ))

    with pytest.raises(ValueError, match="trusted actor"):
        command.submit(IngestionRequest(
            content="Actors are authenticated before scheduling work.",
            profile_id=engine._profile_id,
            source_type="mcp",
            idempotency_key="admission-handler:missing-actor",
        ))

    assert command.repository.list_operations() == []


def test_trust_rejection_happens_before_any_durable_admission(
    engine_with_mock_deps,
    monkeypatch,
) -> None:
    engine = engine_with_mock_deps
    _install_m018(engine)
    command = build_engine_ingestion_command(engine)
    monkeypatch.setattr(
        engine._hooks,
        "run_pre",
        Mock(side_effect=PermissionError("policy rejected")),
    )

    with pytest.raises(PermissionError, match="policy rejected"):
        command.submit(IngestionRequest(
            content="This rejected request must leave no durable trace.",
            profile_id=engine._profile_id,
            source_type="mcp",
            idempotency_key="admission-handler:policy-reject",
            trusted_actor_id="daemon-capability:admission-test",
        ))

    assert command.repository.list_operations() == []
    assert engine._db.search_facts_fts("durable trace", engine._profile_id) == []


def test_prebuilt_fact_reuses_existing_memory_without_cascade_deleting_facts(
    engine_with_mock_deps,
) -> None:
    """Attaching a fact must never replace its source-memory parent row."""
    engine = engine_with_mock_deps
    _install_m018(engine)
    memory_id = "existing-memory"
    engine._db.store_memory(MemoryRecord(
        memory_id=memory_id,
        profile_id=engine._profile_id,
        content="Original source memory remains authoritative.",
    ))
    for fact_id, content in (
        ("existing-fact-a", "First existing child fact."),
        ("existing-fact-b", "Second existing child fact."),
    ):
        engine._db.store_fact(AtomicFact(
            fact_id=fact_id,
            memory_id=memory_id,
            profile_id=engine._profile_id,
            content=content,
        ))

    prebuilt = AtomicFact(
        fact_id="new-prebuilt-fact",
        memory_id=memory_id,
        profile_id=engine._profile_id,
        content="New prebuilt child fact.",
    )
    receipt = build_engine_ingestion_command(engine).submit(IngestionRequest(
        content=prebuilt.content,
        profile_id=engine._profile_id,
        source_type="python-api-prebuilt",
        idempotency_key="prebuilt:existing-memory-regression",
        metadata={"_slm_prebuilt_fact_v1": _prebuilt_fact_payload(prebuilt)},
        trusted_actor_id="daemon-capability:admission-test",
    ))

    assert receipt.state is IngestionState.QUERYABLE
    source = engine._db.execute(
        "SELECT content FROM memories WHERE memory_id = ? AND profile_id = ?",
        (memory_id, engine._profile_id),
    )
    assert source[0]["content"] == "Original source memory remains authoritative."
    children = engine._db.execute(
        "SELECT fact_id FROM atomic_facts WHERE memory_id = ? ORDER BY fact_id",
        (memory_id,),
    )
    assert [row["fact_id"] for row in children] == [
        "existing-fact-a",
        "existing-fact-b",
        "new-prebuilt-fact",
    ]
