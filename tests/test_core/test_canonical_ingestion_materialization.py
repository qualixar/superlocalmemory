# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later

"""Production-adapter contracts for V3.7 canonical ingestion.

Saga design note (research/LLD for the retry regression):

* Model-heavy extraction/enrichment must run without a live SQLite writer
  transaction, otherwise an Ollama/embedding stall blocks every interactive
  write.
* Once the relational pipeline has emitted fact IDs, that partial result is the
  durable retry boundary.  A later exception must checkpoint those IDs and its
  operation-scoped stage observations before marking the operation failed.
* Retrying a checkpointed operation may repair idempotent projections, but must
  never replay extraction/consolidation.  Those stages mutate evidence/access
  counters and can create provenance, graph, and temporal observations.

The tests below exercise the real ``MemoryEngine`` adapter rather than only the
generic command callback seam.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from superlocalmemory.core.engine_ingestion import (
    build_engine_ingestion_command,
    canonical_store,
    canonical_store_fact,
)
from superlocalmemory.core.ingestion_command import (
    IngestionOperationRepository,
    IngestionRequest,
    IngestionState,
)
from superlocalmemory.core.store_pipeline import (
    _record_fact_entity_association,
    run_store,
)
from superlocalmemory.server.unified_daemon import (
    _materialize_ingestion_one_pass,
    _materialize_legacy_pending_item,
    _run_materializer_operation,
)
from superlocalmemory.storage.migrations import (
    M018_ingestion_operations,
    M028_fact_entity_associations,
)
from superlocalmemory.storage.models import (
    AtomicFact,
    CanonicalEntity,
    FactType,
    MemoryRecord,
)


def _install_m018(engine) -> None:
    with engine._db.raw_connection() as conn:
        M018_ingestion_operations.apply(conn)


def _relational_ingestion_snapshot(engine) -> dict[str, int]:
    fact_totals = dict(engine._db.execute(
        "SELECT COUNT(*) AS fact_count, "
        "COALESCE(SUM(evidence_count), 0) AS evidence_count, "
        "COALESCE(SUM(access_count), 0) AS access_count "
        "FROM atomic_facts WHERE profile_id=?",
        (engine._profile_id,),
    )[0])
    table_counts = {
        name: int(dict(engine._db.execute(
            f"SELECT COUNT(*) AS count FROM {name} WHERE profile_id=?",
            (engine._profile_id,),
        )[0])["count"])
        for name in (
            "provenance",
            "graph_edges",
            "temporal_events",
            "fact_temporal_validity",
        )
    }
    return {
        **{name: int(value) for name, value in fact_totals.items()},
        **table_counts,
    }


def test_entity_fact_count_effect_is_constant_query_and_idempotent(
    engine_with_mock_deps,
    monkeypatch,
) -> None:
    engine = engine_with_mock_deps
    engine._db.store_entity(CanonicalEntity(
        entity_id="entity-alice",
        profile_id=engine._profile_id,
        canonical_name="Alice",
    ))
    memory = MemoryRecord(
        memory_id="memory-1",
        profile_id=engine._profile_id,
        content="Alice owns the reliability review.",
    )
    engine._db.store_memory(memory)
    engine._db.store_fact(AtomicFact(
        fact_id="fact-1",
        memory_id=memory.memory_id,
        profile_id=engine._profile_id,
        content="Alice owns the reliability review.",
    ))
    statements: list[str] = []
    execute = engine._db.execute

    def record_execute(sql, params=()):
        statements.append(" ".join(sql.split()))
        return execute(sql, params)

    monkeypatch.setattr(engine._db, "execute", record_execute)
    for _ in range(2):
        _record_fact_entity_association(
            engine._db,
            operation_id="operation-1",
            profile_id=engine._profile_id,
            fact_id="fact-1",
            entity_id="entity-alice",
        )

    effect_statements = [
        sql for sql in statements
        if "fact_entity_associations" in sql
        or sql.startswith("UPDATE canonical_entities")
    ]
    assert len(effect_statements) == 3
    assert not any(
        " LIKE " in sql or "COUNT(" in sql for sql in effect_statements
    )
    assert engine._db.execute(
        "SELECT fact_count FROM canonical_entities WHERE entity_id=?",
        ("entity-alice",),
    )[0]["fact_count"] == 1
    assert engine._db.execute(
        "SELECT COUNT(*) AS count FROM fact_entity_associations"
    )[0]["count"] == 1

    plan = " ".join(
        str(column)
        for row in engine._db.execute(
            "EXPLAIN QUERY PLAN UPDATE canonical_entities "
            "SET fact_count=fact_count+1 WHERE entity_id=? AND profile_id=?",
            ("entity-alice", engine._profile_id),
        )
        for column in row
    )
    assert "USING INDEX" in plan
    assert "SCAN canonical_entities" not in plan


def test_live_entity_count_survives_backfill_interleaving(
    engine_with_mock_deps,
) -> None:
    """A post-readiness fact must be outside the historical repair boundary."""
    engine = engine_with_mock_deps
    with engine._db.raw_connection() as conn:
        M028_fact_entity_associations.apply(conn)
    engine._db.store_entity(CanonicalEntity(
        entity_id="entity-live",
        profile_id=engine._profile_id,
        canonical_name="Live Entity",
    ))
    memory = MemoryRecord(
        memory_id="memory-live",
        profile_id=engine._profile_id,
        content="Live Entity owns the reliability review.",
    )
    engine._db.store_memory(memory)
    engine._db.store_fact(AtomicFact(
        fact_id="fact-live",
        memory_id=memory.memory_id,
        profile_id=engine._profile_id,
        content="Live Entity owns the reliability review.",
        canonical_entities=["entity-live"],
    ))

    # This ordering reproduces the race: background repair wins its write
    # before the live ingestion path records and counts the association.
    while not M028_fact_entity_associations.repair_fact_entity_associations(
        engine._db.db_path,
        batch_size=1,
        max_batches=1,
    )["complete"]:
        pass
    _record_fact_entity_association(
        engine._db,
        operation_id="operation-live",
        profile_id=engine._profile_id,
        fact_id="fact-live",
        entity_id="entity-live",
    )

    assert engine._db.execute(
        "SELECT fact_count FROM canonical_entities WHERE entity_id=?",
        ("entity-live",),
    )[0]["fact_count"] == 1
    association = dict(engine._db.execute(
        "SELECT first_operation_id,count_applied "
        "FROM fact_entity_associations WHERE fact_id=? AND entity_id=?",
        ("fact-live", "entity-live"),
    )[0])
    assert association == {
        "first_operation_id": "operation-live",
        "count_applied": 1,
    }


def test_entity_association_storage_failure_marks_ingestion_incomplete(
    engine_with_mock_deps,
) -> None:
    engine = engine_with_mock_deps
    _install_m018(engine)
    command = build_engine_ingestion_command(engine)
    receipt = command.submit(IngestionRequest(
        content=(
            "Mira approved the Atlas recovery checkpoint on July 23, 2026 "
            "for the reliability team."
        ),
        profile_id=engine._profile_id,
        source_type="http",
        idempotency_key="association-storage-failure",
        trusted_actor_id="daemon-capability:owned-instance",
        session_date="2026-07-23",
    ))
    derived = AtomicFact(
        fact_id="derived-association-failure",
        content="Mira approved the Atlas recovery checkpoint on July 23, 2026.",
        entities=["Mira", "Atlas"],
        fact_type=FactType.SEMANTIC,
        observation_date="2026-07-23",
    )

    with patch.object(
        engine._fact_extractor,
        "extract_facts",
        return_value=[derived],
    ), patch(
        "superlocalmemory.core.store_pipeline."
        "_record_fact_entity_association",
        side_effect=RuntimeError("association schema unavailable"),
    ):
        failed = command.materialize(receipt.operation_id)

    assert failed.state is IngestionState.FAILED
    assert failed.derivation_state["relational_started"] is True
    assert failed.derivation_state["pipeline"] is False
    assert "association schema unavailable" in failed.last_error


def test_partial_relational_failure_retries_without_replaying_extraction_or_consolidation(
    engine_with_mock_deps,
) -> None:
    """Finish idempotent effects from stored facts after an association failure."""
    engine = engine_with_mock_deps
    _install_m018(engine)
    command = build_engine_ingestion_command(engine)
    receipt = command.submit(IngestionRequest(
        content=(
            "Mira approved the Atlas recovery checkpoint on July 23, 2026 "
            "for the reliability team."
        ),
        profile_id=engine._profile_id,
        source_type="http",
        idempotency_key="partial-relational-retry",
        trusted_actor_id="daemon-capability:owned-instance",
        session_date="2026-07-23",
    ))
    derived = AtomicFact(
        fact_id="derived-partial-relational-retry",
        content="Mira approved the Atlas recovery checkpoint on July 23, 2026.",
        entities=["Mira", "Atlas"],
        fact_type=FactType.SEMANTIC,
        observation_date="2026-07-23",
    )
    original_association = _record_fact_entity_association
    association_calls = 0

    def record_once_then_fail(*args, **kwargs):
        nonlocal association_calls
        association_calls += 1
        original_association(*args, **kwargs)
        if association_calls == 1:
            raise RuntimeError("association acknowledgement lost")

    with patch.object(
        engine._fact_extractor,
        "extract_facts",
        return_value=[derived],
    ) as extract_facts, patch.object(
        engine._consolidator,
        "consolidate",
        wraps=engine._consolidator.consolidate,
    ) as consolidate, patch(
        "superlocalmemory.core.store_pipeline._record_fact_entity_association",
        side_effect=record_once_then_fail,
    ):
        failed = command.materialize(receipt.operation_id)
        extraction_calls = extract_facts.call_count
        consolidation_calls = consolidate.call_count
        partial = _relational_ingestion_snapshot(engine)
        completed = command.retry(receipt.operation_id)
        recovered = _relational_ingestion_snapshot(engine)
        retried = command.materialize(receipt.operation_id)

    assert failed.state is IngestionState.FAILED
    assert failed.derivation_state["relational_started"] is True
    assert failed.derivation_state["pipeline"] is False
    assert partial["fact_count"] > 0
    assert completed.state is IngestionState.COMPLETE, (
        completed.derivation_state, completed.last_error,
    )
    assert retried.state is IngestionState.COMPLETE
    assert extract_facts.call_count == extraction_calls
    assert consolidate.call_count == consolidation_calls
    assert recovered == _relational_ingestion_snapshot(engine)


def test_failed_production_materialization_retries_do_not_replay_relational_pipeline(
    engine_with_mock_deps,
) -> None:
    engine = engine_with_mock_deps
    _install_m018(engine)
    command = build_engine_ingestion_command(engine)
    receipt = command.submit(IngestionRequest(
        content=(
            "Alice approved the Atlas recovery plan on July 23, 2026 and "
            "recorded the decision for the reliability team."
        ),
        profile_id=engine._profile_id,
        source_type="http",
        idempotency_key="bounded-relational-retry",
        trusted_actor_id="daemon-capability:owned-instance",
        session_date="2026-07-23",
    ))
    derived = AtomicFact(
        fact_id="derived-atlas-recovery",
        content="Alice approved the Atlas recovery plan on July 23, 2026.",
        entities=["Alice", "Atlas"],
        fact_type=FactType.SEMANTIC,
        observation_date="2026-07-23",
        confidence=0.95,
    )

    with patch.object(
        engine._fact_extractor,
        "extract_facts",
        return_value=[derived],
    ), patch.object(
        engine._hooks,
        "run_post",
        side_effect=RuntimeError("audit sink unavailable"),
    ):
        failed = command.materialize(receipt.operation_id)
        assert failed.state is IngestionState.FAILED
        assert failed.final_fact_ids
        baseline = _relational_ingestion_snapshot(engine)
        assert baseline["provenance"] > 0
        assert baseline["graph_edges"] > 0
        assert baseline["temporal_events"] > 0

        for _ in range(3):
            failed = command.retry(receipt.operation_id)
            assert failed.state is IngestionState.FAILED
            assert _relational_ingestion_snapshot(engine) == baseline


def test_provenance_repair_observes_partial_write_without_duplicate(
    engine_with_mock_deps,
) -> None:
    engine = engine_with_mock_deps
    _install_m018(engine)
    command = build_engine_ingestion_command(engine)
    receipt = command.submit(IngestionRequest(
        content=(
            "Mira approved the Atlas recovery checkpoint on July 23, 2026 "
            "for the reliability team."
        ),
        profile_id=engine._profile_id,
        source_type="http",
        idempotency_key="provenance-observation-repair",
        trusted_actor_id="daemon-capability:owned-instance",
        session_date="2026-07-23",
    ))
    original_record = engine._provenance.record

    def record_then_fail(**kwargs):
        original_record(**kwargs)
        raise RuntimeError("provenance acknowledgement lost")

    with patch.object(
        engine._provenance,
        "record",
        side_effect=record_then_fail,
    ):
        failed = command.materialize(receipt.operation_id)

    assert failed.state is IngestionState.FAILED
    assert failed.derivation_state["provenance"] is False
    baseline = _relational_ingestion_snapshot(engine)
    completed = command.retry(receipt.operation_id)

    assert completed.state is IngestionState.COMPLETE
    assert completed.derivation_state["provenance"] is True
    assert _relational_ingestion_snapshot(engine) == baseline


def test_transient_post_hook_failure_continues_to_complete_without_replay(
    engine_with_mock_deps,
) -> None:
    engine = engine_with_mock_deps
    _install_m018(engine)
    command = build_engine_ingestion_command(engine)
    receipt = command.submit(IngestionRequest(
        content=(
            "Rhea approved the Atlas recovery checkpoint on July 23, 2026 "
            "for the reliability team."
        ),
        profile_id=engine._profile_id,
        source_type="http",
        idempotency_key="post-hook-stage-continuation",
        trusted_actor_id="daemon-capability:owned-instance",
        session_date="2026-07-23",
    ))
    calls = 0

    def transient_post_hook(_operation, _context):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("audit sink unavailable")

    with patch.object(
        engine._fact_extractor,
        "extract_facts",
        wraps=engine._fact_extractor.extract_facts,
    ) as extract_facts, patch.object(
        engine._consolidator,
        "consolidate",
        wraps=engine._consolidator.consolidate,
    ) as consolidate, patch.object(
        engine._hooks,
        "run_post",
        side_effect=transient_post_hook,
    ):
        failed = command.materialize(receipt.operation_id)
        baseline = _relational_ingestion_snapshot(engine)
        extraction_calls = extract_facts.call_count
        consolidation_calls = consolidate.call_count
        completed = command.retry(receipt.operation_id)

    assert failed.state is IngestionState.FAILED
    assert failed.derivation_state["pipeline"] is True
    # The relational checkpoint is the retry boundary.  A post-hook failure
    # must retain the facts and every already-observed pipeline stage instead
    # of reconstructing a partial state from the exception path.
    assert failed.derivation_state["relational"] is True
    assert failed.derivation_state["post_hooks"] is False
    assert completed.state is IngestionState.COMPLETE
    assert completed.derivation_state["post_hooks"] is True
    assert calls == 2
    assert extract_facts.call_count == extraction_calls
    assert consolidate.call_count == consolidation_calls
    assert _relational_ingestion_snapshot(engine) == baseline


def test_prebuilt_fact_post_hook_retry_keeps_consolidation_complete(
    engine_with_mock_deps,
) -> None:
    engine = engine_with_mock_deps
    _install_m018(engine)
    fact = AtomicFact(
        fact_id="prebuilt-post-hook-retry",
        content=(
            "Rhea approved the Atlas prebuilt recovery checkpoint on "
            "July 23, 2026 for the reliability team."
        ),
        entities=["Rhea", "Atlas"],
        fact_type=FactType.SEMANTIC,
        observation_date="2026-07-23",
    )

    with patch(
        "superlocalmemory.core.store_pipeline.run_store",
        wraps=run_store,
    ) as pipeline, patch.object(
        engine._hooks,
        "run_post",
        side_effect=[RuntimeError("audit sink unavailable"), None],
    ):
        with pytest.raises(RuntimeError, match="audit sink unavailable"):
            canonical_store_fact(
                engine,
                fact,
                trusted_actor_id="daemon-capability:owned-instance",
            )
        result = canonical_store_fact(
            engine,
            fact,
            trusted_actor_id="daemon-capability:owned-instance",
        )

    operation_row = engine._db.execute(
        "SELECT operation_id FROM ingestion_operations "
        "WHERE profile_id=? AND source_type=? AND idempotency_key=?",
        (
            engine._profile_id,
            "python-api-prebuilt",
            "prebuilt:prebuilt-post-hook-retry",
        ),
    )[0]
    operation = IngestionOperationRepository(engine._db).get(
        operation_row["operation_id"],
    )
    assert result == fact.fact_id
    assert operation.state is IngestionState.COMPLETE
    assert operation.derivation_state["consolidation"] is True
    assert operation.derivation_state["post_hooks"] is True
    assert pipeline.call_count == 1


def test_worker_death_before_relational_start_is_retryable(
    engine_with_mock_deps,
) -> None:
    engine = engine_with_mock_deps
    _install_m018(engine)
    command = build_engine_ingestion_command(engine)
    receipt = command.submit(IngestionRequest(
        content=(
            "Tara approved the Atlas recovery checkpoint on July 23, 2026 "
            "for the reliability team."
        ),
        profile_id=engine._profile_id,
        source_type="http",
        idempotency_key="pre-relational-worker-death",
        trusted_actor_id="daemon-capability:owned-instance",
        session_date="2026-07-23",
    ))
    derived = AtomicFact(
        fact_id="derived-pre-relational-retry",
        content="Tara approved the Atlas recovery checkpoint on July 23, 2026.",
        entities=["Tara", "Atlas"],
        fact_type=FactType.SEMANTIC,
        observation_date="2026-07-23",
    )

    with patch.object(
        engine._fact_extractor,
        "extract_facts",
        side_effect=[SystemExit("worker terminated before writes"), [derived]],
    ):
        with pytest.raises(SystemExit, match="before writes"):
            command.materialize(receipt.operation_id)
        interrupted = command.repository.get(receipt.operation_id)
        assert interrupted.state is IngestionState.ENRICHING
        assert interrupted.final_fact_ids == ()
        assert interrupted.derivation_state == {
            "pipeline_started": True,
            "pipeline": False,
        }

        completed = command.materialize(receipt.operation_id)

    assert completed.state is IngestionState.COMPLETE
    assert completed.final_fact_ids
    assert completed.derivation_state["pipeline"] is True


def test_expired_production_attempt_resumes_from_started_stage_ledger(
    engine_with_mock_deps,
) -> None:
    engine = engine_with_mock_deps
    _install_m018(engine)
    command = build_engine_ingestion_command(engine)
    receipt = command.submit(IngestionRequest(
        content=(
            "Isha approved the Atlas recovery checkpoint on July 23, 2026 "
            "for the reliability team."
        ),
        profile_id=engine._profile_id,
        source_type="http",
        idempotency_key="crash-safe-stage-ledger",
        trusted_actor_id="daemon-capability:owned-instance",
        session_date="2026-07-23",
    ))
    derived = AtomicFact(
        fact_id="derived-crash-checkpoint",
        content="Isha approved the Atlas recovery checkpoint on July 23, 2026.",
        entities=["Isha", "Atlas"],
        fact_type=FactType.SEMANTIC,
        observation_date="2026-07-23",
    )

    with patch.object(
        engine._fact_extractor,
        "extract_facts",
        return_value=[derived],
    ), patch.object(
        engine._hooks,
        "run_post",
        side_effect=[SystemExit("worker terminated"), None],
    ):
        with pytest.raises(SystemExit, match="worker terminated"):
            command.materialize(receipt.operation_id)
        baseline = _relational_ingestion_snapshot(engine)
        interrupted = command.repository.get(receipt.operation_id)
        assert interrupted.state is IngestionState.ENRICHING
        assert interrupted.final_fact_ids
        assert interrupted.derivation_state["pipeline_started"] is True
        assert interrupted.derivation_state["relational_started"] is True
        assert interrupted.derivation_state["pipeline"] is True
        assert interrupted.derivation_state["post_hooks"] is False

        completed = command.materialize(receipt.operation_id)

    assert completed.state is IngestionState.COMPLETE
    assert completed.derivation_state["post_hooks"] is True
    assert _relational_ingestion_snapshot(engine) == baseline


def test_production_model_work_does_not_block_interactive_sqlite_writer(
    engine_with_mock_deps,
) -> None:
    engine = engine_with_mock_deps
    _install_m018(engine)
    engine._db.execute(
        "CREATE TABLE ingestion_interactive_probe (value TEXT NOT NULL)"
    )
    command = build_engine_ingestion_command(engine)
    receipt = command.submit(IngestionRequest(
        content=(
            "Nia owns the Atlas retrieval service and records every approved "
            "recovery checkpoint for the reliability team."
        ),
        profile_id=engine._profile_id,
        source_type="http",
        idempotency_key="production-writer-concurrency",
        trusted_actor_id="daemon-capability:owned-instance",
    ))
    entered_model_work = threading.Event()
    release_model_work = threading.Event()

    def slow_extract(**_kwargs):
        entered_model_work.set()
        assert release_model_work.wait(timeout=5)
        return []

    with patch.object(
        engine._fact_extractor,
        "extract_facts",
        side_effect=slow_extract,
    ), ThreadPoolExecutor(max_workers=2) as executor:
        materialization = executor.submit(command.materialize, receipt.operation_id)
        assert entered_model_work.wait(timeout=5)
        interactive_write = executor.submit(
            engine._db.execute,
            "INSERT INTO ingestion_interactive_probe(value) VALUES (?)",
            ("interactive",),
        )
        try:
            interactive_write.result(timeout=1)
        finally:
            release_model_work.set()
        assert materialization.result(timeout=5).state is IngestionState.COMPLETE

    assert [
        dict(row)["value"]
        for row in engine._db.execute(
            "SELECT value FROM ingestion_interactive_probe"
        )
    ] == ["interactive"]


def test_materialization_ignores_admission_gate_for_submitted_operation(
    engine_with_mock_deps,
) -> None:
    """An already-submitted operation must materialize even if the admission
    gate now rejects its content.

    Regression: ``run_store`` applied the entropy / low-quality *admission* gate
    at the materialize step. Content whose queryable projection was already
    committed at submit (e.g. near-identical session-close summaries, which the
    entropy gate scores as >0.95 near-duplicates once the window is warm) was
    re-rejected — ``run_store`` returned ``[]`` — so materialization raised
    "materialization produced no final facts", the operation was marked FAILED,
    and the background worker retried it forever. Admission belongs to submit,
    not to materialization.
    """
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
        idempotency_key="admission-gate-at-materialize",
        trusted_actor_id="daemon-capability:owned-instance",
    ))
    assert receipt.state is IngestionState.QUERYABLE
    queryable_id = receipt.queryable_fact_ids[0]

    # The gate now treats the content as a near-duplicate (its rolling window
    # already holds the committed projection). Pre-fix this discarded the fact.
    with patch.object(engine._entropy_gate, "should_pass", return_value=False):
        completed = command.materialize(receipt.operation_id)

    assert completed.state is IngestionState.COMPLETE
    assert completed.last_error == ""
    assert queryable_id in completed.final_fact_ids


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
    assert failed.final_fact_ids == receipt.queryable_fact_ids
    assert failed.derivation_state[stage] is False
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


def test_nonblocking_canonical_store_returns_queryable_receipt_before_enrichment(
    engine_with_mock_deps,
) -> None:
    """Mode B-style extraction must never run on the interactive receipt path."""
    engine = engine_with_mock_deps
    _install_m018(engine)

    with patch.object(
        engine._fact_extractor,
        "extract_facts",
        side_effect=AssertionError("interactive write invoked model extraction"),
    ):
        fact_ids = canonical_store(
            engine,
            "Rhea approved the durable queryable ingestion boundary.",
            source_type="python-api",
            trusted_actor_id="local-capability:python-api:test",
            idempotency_key="nonblocking-canonical-store-1",
            require_complete=False,
        )

    assert len(fact_ids) == 1
    operation = engine._db.execute(
        "SELECT state, queryable_fact_ids_json FROM ingestion_operations "
        "WHERE idempotency_key=?",
        ("nonblocking-canonical-store-1",),
    )[0]
    assert operation["state"] == "queryable"
    assert fact_ids[0] in operation["queryable_fact_ids_json"]


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
