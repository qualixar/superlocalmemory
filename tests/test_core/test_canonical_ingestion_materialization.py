# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later

"""Production-adapter contracts for V3.7 canonical ingestion."""

from __future__ import annotations

import threading
from unittest.mock import patch
from types import SimpleNamespace

import pytest

from superlocalmemory.core.engine_ingestion import build_engine_ingestion_command
from superlocalmemory.core.ingestion_command import (
    IngestionOperationRepository,
    IngestionRequest,
    IngestionState,
)
from superlocalmemory.server.unified_daemon import (
    _materialize_ingestion_one_pass,
    _materialize_legacy_pending_item,
    _run_materializer_operation,
)
from superlocalmemory.storage.migrations import M018_ingestion_operations
from superlocalmemory.storage.models import AtomicFact, FactType


def _install_m018(engine) -> None:
    with engine._db.raw_connection() as conn:
        M018_ingestion_operations.apply(conn)


def test_materialization_promotes_queryable_fact_without_duplicate_memory(
    engine_with_mock_deps,
) -> None:
    engine = engine_with_mock_deps
    _install_m018(engine)
    command = build_engine_ingestion_command(engine)
    request = IngestionRequest(
        content=(
            "Alice leads the reliability program and reviews every production "
            "incident with the platform team."
        ),
        profile_id=engine._profile_id,
        source_type="http",
        idempotency_key="session-7:turn-12",
        metadata={"source_agent_id": "caller-claimed"},
        trusted_actor_id="daemon-capability:owned-instance",
        session_id="session-7",
    )

    receipt = command.submit(request)
    assert receipt.state is IngestionState.QUERYABLE
    assert len(receipt.queryable_fact_ids) == 1
    queryable_id = receipt.queryable_fact_ids[0]
    queryable = engine._db.get_fact(queryable_id)
    assert queryable is not None
    memory_id = queryable.memory_id
    assert queryable.canonical_entities == []

    derived = AtomicFact(
        fact_id="derived-reliability",
        content="Alice reviews every production incident with the platform team.",
        entities=["Alice"],
        fact_type=FactType.SEMANTIC,
        confidence=0.95,
    )
    with patch.object(
        engine._fact_extractor,
        "extract_facts",
        return_value=[derived],
    ), patch.object(
        engine._graph_builder,
        "build_edges",
        wraps=engine._graph_builder.build_edges,
    ) as graph_spy:
        completed = command.materialize(receipt.operation_id)

    assert completed.state is IngestionState.COMPLETE
    assert queryable_id in completed.final_fact_ids
    assert "derived-reliability" in completed.final_fact_ids
    assert graph_spy.call_count >= 2

    memories = engine._db.execute(
        "SELECT memory_id FROM memories WHERE profile_id=?",
        (engine._profile_id,),
    )
    assert [dict(row)["memory_id"] for row in memories] == [memory_id]

    facts = engine._db.get_facts_by_memory_id(memory_id, engine._profile_id)
    assert {fact.fact_id for fact in facts} == set(completed.final_fact_ids)
    promoted = engine._db.get_fact(queryable_id)
    assert promoted is not None
    assert promoted.canonical_entities, "queryable fact was never fully promoted"

    provenance = engine._db.execute(
        "SELECT fact_id FROM provenance WHERE profile_id=?",
        (engine._profile_id,),
    )
    assert {dict(row)["fact_id"] for row in provenance} == set(
        completed.final_fact_ids
    )


def test_queryable_admission_never_waits_for_embedding_warmup(
    engine_with_mock_deps,
) -> None:
    """Receipt-first admission must not acquire the enrichment worker lock."""
    engine = engine_with_mock_deps
    _install_m018(engine)
    command = build_engine_ingestion_command(engine)

    with patch.object(
        engine._embedder,
        "embed",
        side_effect=AssertionError("embedding belongs to materialization"),
    ) as embed:
        receipt = command.submit(IngestionRequest(
            content=(
                "Nia owns the Atlas retrieval service and records the release "
                "checkpoint on July 16, 2026."
            ),
            profile_id=engine._profile_id,
            source_type="http",
            idempotency_key="receipt-before-embedding",
            trusted_actor_id="daemon-capability:owned-instance",
        ))

    embed.assert_not_called()

    assert receipt.state is IngestionState.QUERYABLE
    fact = engine._db.get_fact(receipt.queryable_fact_ids[0])
    assert fact is not None
    assert fact.embedding is None


def test_materializer_rejects_queryable_facts_from_another_profile(
    engine_with_mock_deps,
) -> None:
    engine = engine_with_mock_deps
    _install_m018(engine)
    command = build_engine_ingestion_command(engine)
    request = IngestionRequest(
        content="Alice owns the incident review process for the platform team.",
        profile_id=engine._profile_id,
        source_type="http",
        idempotency_key="profile-boundary",
        trusted_actor_id="daemon-capability:owned-instance",
    )
    receipt = command.submit(request)
    engine._db.execute(
        "INSERT INTO profiles(profile_id, name) VALUES ('foreign', 'Foreign')"
    )
    engine._db.execute(
        "UPDATE atomic_facts SET profile_id='foreign' WHERE fact_id=?",
        (receipt.queryable_fact_ids[0],),
    )

    failed = command.materialize(receipt.operation_id)

    assert failed.state is IngestionState.FAILED
    assert "profile" in failed.last_error.lower()


@pytest.mark.parametrize(
    ("component", "stage", "error_text"),
    [
        ("extractor", "extraction", "extractor unavailable"),
        ("consolidator", "consolidation", "consolidator unavailable"),
    ],
)
def test_swallowed_pipeline_fallback_cannot_claim_complete_derivation(
    engine_with_mock_deps,
    component,
    stage,
    error_text,
) -> None:
    engine = engine_with_mock_deps
    _install_m018(engine)
    command = build_engine_ingestion_command(engine)
    receipt = command.submit(IngestionRequest(
        content=(
            "Dana owns the production recovery review and records every "
            "approved corrective action for the platform team."
        ),
        profile_id=engine._profile_id,
        source_type="http",
        idempotency_key=f"failed-{component}",
        trusted_actor_id="daemon-capability:owned-instance",
    ))
    target = (
        engine._fact_extractor.extract_facts
        if component == "extractor"
        else engine._consolidator.consolidate
    )
    owner = engine._fact_extractor if component == "extractor" else engine._consolidator
    method = "extract_facts" if component == "extractor" else "consolidate"

    with patch.object(owner, method, side_effect=RuntimeError(error_text)):
        failed = command.materialize(receipt.operation_id)

    assert target is not None
    assert failed.state is IngestionState.FAILED
    assert stage in failed.last_error
    assert failed.final_fact_ids == ()
    assert engine._db.get_fact(receipt.queryable_fact_ids[0]) is not None


def test_production_adapter_rejects_missing_trusted_actor_before_persistence(
    engine_with_mock_deps,
) -> None:
    engine = engine_with_mock_deps
    _install_m018(engine)
    command = build_engine_ingestion_command(engine)
    request = IngestionRequest(
        content="Alice owns the incident review process for the platform team.",
        profile_id=engine._profile_id,
        source_type="http",
        idempotency_key="missing-actor",
    )

    with pytest.raises(ValueError, match="trusted actor"):
        command.submit(request)

    assert command.repository.list_operations() == []


def test_trust_hook_receives_capability_identity_not_caller_metadata(
    engine_with_mock_deps,
) -> None:
    engine = engine_with_mock_deps
    _install_m018(engine)
    command = build_engine_ingestion_command(engine)
    captured: list[dict] = []
    engine._hooks.register_pre("store", lambda context: captured.append(context))
    request = IngestionRequest(
        content="Alice owns the incident review process for the platform team.",
        profile_id=engine._profile_id,
        source_type="http",
        idempotency_key="trusted-actor",
        metadata={"agent_id": "caller-selected-admin"},
        trusted_actor_id="daemon-capability:owned-instance",
    )

    command.submit(request)

    assert captured[-1]["agent_id"] == "daemon-capability:owned-instance"
    assert captured[-1]["agent_id"] != request.metadata["agent_id"]


def test_background_one_pass_completes_queryable_operation(
    engine_with_mock_deps,
) -> None:
    engine = engine_with_mock_deps
    _install_m018(engine)
    command = build_engine_ingestion_command(engine)
    receipt = command.submit(IngestionRequest(
        content=(
            "Alice owns the incident review process and records every approved "
            "corrective action for the platform team."
        ),
        profile_id=engine._profile_id,
        source_type="http",
        idempotency_key="background-one-pass",
        trusted_actor_id="daemon-capability:owned-instance",
    ))

    completed, failed = _materialize_ingestion_one_pass(
        engine, min_queryable_age_seconds=0,
    )

    assert (completed, failed) == (1, 0)
    assert command.repository.get(receipt.operation_id).state is IngestionState.COMPLETE


def test_materializer_operation_drains_before_profile_switch() -> None:
    """Background enrichment must not use engine components during a rebind."""
    from superlocalmemory.server.profile_runtime import ProfileRuntime

    runtime = ProfileRuntime("alpha")
    operation_entered = threading.Event()
    release_operation = threading.Event()
    switch_committed = threading.Event()
    seen: dict[str, str] = {}
    engine = SimpleNamespace(profile_id="alpha")

    def _materialize(current_engine):
        seen["profile"] = current_engine.profile_id
        operation_entered.set()
        assert release_operation.wait(2)
        return "complete"

    materializer = threading.Thread(
        target=_run_materializer_operation,
        args=(runtime, lambda: engine, _materialize),
    )
    switch = threading.Thread(
        target=runtime.transition,
        args=("beta", lambda previous, target: switch_committed.set()),
    )
    materializer.start()
    assert operation_entered.wait(2)
    switch.start()
    assert not switch_committed.wait(0.05)

    release_operation.set()
    materializer.join(2)
    switch.join(2)

    assert seen == {"profile": "alpha"}
    assert switch_committed.is_set()
    assert runtime.snapshot.profile_id == "beta"


def test_bm25_only_runtime_does_not_require_unconfigured_vector_projectors(
    engine_with_mock_deps,
) -> None:
    engine = engine_with_mock_deps
    _install_m018(engine)
    engine._embedder = None
    engine._ann_index = None
    engine._vector_store = None
    command = build_engine_ingestion_command(engine)
    receipt = command.submit(IngestionRequest(
        content=(
            "The team decided to preserve SQLite and BM25 as the durable "
            "retrieval baseline when vector services are unavailable."
        ),
        profile_id=engine._profile_id,
        source_type="python-api",
        idempotency_key="bm25-only-completion",
        trusted_actor_id="local-capability:python-api:test",
    ))

    completed = command.materialize(receipt.operation_id)

    assert completed.state is IngestionState.COMPLETE
    assert completed.derivation_state["ann"] is True
    assert completed.derivation_state["vector"] is True
    assert completed.derivation_state["bm25"] is True


def test_background_one_pass_isolates_poison_operation_and_continues() -> None:
    operations = [
        SimpleNamespace(operation_id="poison"),
        SimpleNamespace(operation_id="healthy"),
    ]

    class FakeRepository:
        def list_materializable(self, *, limit, min_queryable_age_seconds):
            assert limit == 50
            assert min_queryable_age_seconds == 1.0
            return operations

    class FakeCommand:
        repository = FakeRepository()

        def materialize(self, operation_id):
            if operation_id == "poison":
                raise RuntimeError("corrupt operation")
            return SimpleNamespace(
                operation_id=operation_id,
                state=IngestionState.COMPLETE,
                fact_ids=("fact-healthy",),
                raw_content="Healthy durable evidence",
                last_error="",
            )

    with patch(
        "superlocalmemory.core.engine_ingestion.build_engine_ingestion_command",
        return_value=FakeCommand(),
    ), patch("superlocalmemory.server.unified_daemon._emit_event"):
        completed, failed = _materialize_ingestion_one_pass(object())

    assert (completed, failed) == (1, 1)


def test_background_one_pass_yields_before_claiming_during_recall() -> None:
    """Durable M018 work must obey the same recall-priority gate as legacy work."""
    from superlocalmemory.core.recall_gate import begin_recall, end_recall

    begin_recall()
    try:
        with patch(
            "superlocalmemory.core.engine_ingestion.build_engine_ingestion_command"
        ) as build_command:
            completed, failed = _materialize_ingestion_one_pass(object())
    finally:
        end_recall()

    assert (completed, failed) == (0, 0)
    build_command.assert_not_called()


def test_legacy_pending_row_backfills_through_canonical_operation(
    engine_with_mock_deps,
) -> None:
    engine = engine_with_mock_deps
    _install_m018(engine)
    item = {
        "id": 77,
        "content": (
            "Bob owns database recovery reviews and records the approved "
            "decision for each production incident."
        ),
        "tags": "legacy",
        "metadata": (
            '{"scope":"shared","shared_with":["reviewer"],'
            '"_slm_idempotency_key":"mcp:stable-key",'
            '"_slm_source_type":"mcp-offline"}'
        ),
    }

    operation_id = _materialize_legacy_pending_item(engine, item)

    operation = IngestionOperationRepository(engine._db).get(operation_id)
    assert operation.state is IngestionState.COMPLETE
    assert operation.source_type == "mcp-offline"
    assert operation.idempotency_key == "mcp:stable-key"
    assert operation.scope == "shared"
    assert operation.shared_with == ("reviewer",)


def test_engine_startup_pending_replay_uses_canonical_operation(
    engine_with_mock_deps,
) -> None:
    from superlocalmemory.cli.pending_store import get_pending, store_pending

    engine = engine_with_mock_deps
    _install_m018(engine)
    pending_id = store_pending(
        content=(
            "Carol owns recovery testing and records every approved database "
            "failover decision for the platform team."
        ),
        metadata={"scope": "global", "session_id": "startup-session"},
        base_dir=engine._config.base_dir,
    )

    engine._process_pending_memories()

    assert get_pending(base_dir=engine._config.base_dir) == []
    rows = engine._db.execute(
        "SELECT source_type, idempotency_key, state, scope, session_id "
        "FROM ingestion_operations"
    )
    assert [dict(row) for row in rows] == [{
        "source_type": "legacy-pending",
        "idempotency_key": f"pending:{pending_id}",
        "state": "complete",
        "scope": "global",
        "session_id": "startup-session",
    }]


def test_public_python_store_routes_through_canonical_ingestion_and_preserves_context(
    engine_with_mock_deps,
) -> None:
    engine = engine_with_mock_deps

    with patch(
        "superlocalmemory.core.engine_ingestion.local_trusted_actor_id",
        return_value="local-capability:python-api:test",
    ) as actor, patch(
        "superlocalmemory.core.engine_ingestion.canonical_store",
        return_value=["canonical-fact"],
    ) as canonical:
        result = engine.store(
            "Dana approved the recovery plan after the July incident review.",
            session_id="python-session",
            session_date="2026-07-14",
            speaker="Dana",
            role="assistant",
            metadata={"project": "reliability"},
            scope="shared",
            shared_with=["reviewer"],
        )

    assert result == ["canonical-fact"]
    actor.assert_called_once_with("python-api")
    canonical.assert_called_once_with(
        engine,
        "Dana approved the recovery plan after the July incident review.",
        source_type="python-api",
        trusted_actor_id="local-capability:python-api:test",
        metadata={"project": "reliability"},
        scope="shared",
        shared_with=["reviewer"],
        session_id="python-session",
        session_date="2026-07-14",
        speaker="Dana",
        role="assistant",
        require_complete=False,
    )


def test_production_materialization_preserves_session_speaker_and_role(
    engine_with_mock_deps,
) -> None:
    engine = engine_with_mock_deps
    _install_m018(engine)
    command = build_engine_ingestion_command(engine)
    receipt = command.submit(IngestionRequest(
        content=(
            "Dana approved the recovery plan after the July production "
            "incident review with the platform team."
        ),
        profile_id=engine._profile_id,
        source_type="python-api",
        idempotency_key="python-context",
        trusted_actor_id="local-capability:python-api:test",
        session_id="python-session",
        session_date="2026-07-14",
        speaker="Dana",
        role="assistant",
    ))

    completed = command.materialize(receipt.operation_id)

    assert completed.state is IngestionState.COMPLETE
    queryable = engine._db.get_fact(receipt.queryable_fact_ids[0])
    assert queryable is not None
    rows = engine._db.execute(
        "SELECT session_id, session_date, speaker, role FROM memories "
        "WHERE memory_id=?",
        (queryable.memory_id,),
    )
    assert dict(rows[0]) == {
        "session_id": "python-session",
        "session_date": "2026-07-14",
        "speaker": "Dana",
        "role": "assistant",
    }
