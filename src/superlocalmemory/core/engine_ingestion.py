# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Production adapter between MemoryEngine and canonical ingestion.

This is the anti-corruption layer for the expand-migrate-contract rollout:
the command owns durability and state, while the existing engine remains the
implementation of queryable projection and complete derivation.
"""

from __future__ import annotations

import hashlib
import logging
import os
import uuid
from typing import TYPE_CHECKING

from superlocalmemory.core.ingestion_command import (
    IngestionCommand,
    IngestionOperation,
    IngestionOperationRepository,
    IngestionRequest,
    MaterializationResult,
)

if TYPE_CHECKING:
    from superlocalmemory.core.engine import MemoryEngine
    from superlocalmemory.storage.models import AtomicFact


logger = logging.getLogger(__name__)


_PREBUILT_FACT_KEY = "_slm_prebuilt_fact_v1"
_DERIVATION_VERSION = "v3.7-ingestion-1"


def _pii_redaction_enabled(engine: "MemoryEngine") -> bool:
    """C4: opt-in PII redaction on ingest.

    On when the engine config sets ``pii_redaction`` truthy OR the
    ``SLM_PII_REDACTION`` env var is set (1/on/true/yes). Default OFF — personal
    use is unchanged; team/company operators opt in.
    """
    cfg = getattr(engine, "_config", None)
    if cfg is not None and getattr(cfg, "pii_redaction", False):
        return True
    return os.environ.get("SLM_PII_REDACTION", "").strip().lower() in (
        "1", "on", "true", "yes",
    )


def content_passes_admission(content: str) -> bool:
    """Return whether raw content is eligible to reach durable ingestion.

    This is deliberately stateless: canonical admission must not populate the
    live entropy gate's duplicate window before the materialization pipeline
    evaluates the same content.
    """
    if not content or not content.strip():
        return False

    from superlocalmemory.core.injection import is_low_quality
    from superlocalmemory.encoding.entropy_gate import EntropyGate

    return not is_low_quality(content) and EntropyGate().should_pass(content)


def _prebuilt_fact_payload(fact: AtomicFact) -> dict:
    """Serialize the stable public AtomicFact contract into M018 metadata."""
    enum_fields = {"fact_type", "lifecycle", "signal_type"}
    fields = (
        "fact_id", "memory_id", "profile_id", "scope", "shared_with",
        "content", "fact_type", "entities", "canonical_entities",
        "observation_date", "referenced_date", "interval_start", "interval_end",
        "confidence", "importance", "evidence_count", "access_count",
        "source_turn_ids", "session_id", "embedding", "fisher_mean",
        "fisher_variance", "lifecycle", "langevin_position",
        "emotional_valence", "emotional_arousal", "signal_type", "pinned",
        "created_at",
    )
    payload = {}
    for name in fields:
        value = getattr(fact, name)
        payload[name] = value.value if name in enum_fields else value
    return payload


def _prebuilt_fact_from_payload(payload: dict):
    from superlocalmemory.storage.models import (
        AtomicFact,
        FactType,
        MemoryLifecycle,
        SignalType,
    )

    values = dict(payload)
    values["fact_type"] = FactType(values.get("fact_type", FactType.SEMANTIC.value))
    values["lifecycle"] = MemoryLifecycle(
        values.get("lifecycle", MemoryLifecycle.ACTIVE.value)
    )
    values["signal_type"] = SignalType(
        values.get("signal_type", SignalType.FACTUAL.value)
    )
    return AtomicFact(**values)


def local_trusted_actor_id(actor_kind: str) -> str:
    """Derive a stable local actor from the private install capability."""
    from superlocalmemory.core.security_primitives import ensure_install_token
    from superlocalmemory.infra.daemon_identity import namespace_id_for, owner_id
    from superlocalmemory.infra.data_root import canonical_data_root

    token = ensure_install_token()
    material = (
        b"superlocalmemory-local-actor-v1\0"
        + token.encode("utf-8")
        + b"\0"
        + actor_kind.encode("utf-8")
    )
    fingerprint = hashlib.sha256(material).hexdigest()
    return (
        f"local-capability:{actor_kind}:{owner_id()}:"
        f"{namespace_id_for(canonical_data_root())}:{fingerprint}"
    )


def canonical_store(
    engine: MemoryEngine,
    content: str,
    *,
    source_type: str,
    trusted_actor_id: str,
    metadata: dict | None = None,
    scope: str = "personal",
    shared_with: list[str] | tuple[str, ...] | None = None,
    session_id: str = "",
    session_date: str | None = None,
    speaker: str = "",
    role: str = "user",
    idempotency_key: str = "",
    require_complete: bool = True,
    return_receipt: bool = False,
) -> list[str] | IngestionOperation:
    """Submit canonical evidence, optionally waiting for enrichment completion.

    ``require_complete=False`` is the interactive/CQRS path: it commits a
    profile-scoped memory and FTS-queryable fact through M018, then returns the
    durable receipt without invoking any LLM, embedding, or graph work.  The
    daemon materializer owns that expensive, retryable enrichment.  Explicit
    complete callers retain the historical synchronous contract.
    """
    import time

    from superlocalmemory.core.ingestion_command import IngestionRequest, IngestionState
    from superlocalmemory.infra.local_diagnostics import record_operation

    started = time.monotonic()

    # Preserve the long-standing Python/API contract for rejected content:
    # low-information input is a no-op and never becomes an M018 operation.
    if not content_passes_admission(content):
        record_operation(
            "remember",
            client=trusted_actor_id,
            duration_ms=(time.monotonic() - started) * 1000.0,
            error=ValueError("content rejected by local admission policy"),
        )
        return []
    # C4: opt-in PII redaction. When enabled (config.pii_redaction or
    # SLM_PII_REDACTION), scrub personal identifiers BEFORE the content is
    # extracted, embedded, or persisted — nothing sensitive ever reaches disk.
    if _pii_redaction_enabled(engine):
        from superlocalmemory.core.pii import redact_pii

        scrubbed, n_pii = redact_pii(content)
        if n_pii:
            content = scrubbed
            logger.info("PII redaction: scrubbed %d identifier(s) on ingest", n_pii)
    try:
        command = build_engine_ingestion_command(engine)
        receipt = command.submit(IngestionRequest(
            content=content,
            profile_id=engine._profile_id,
            source_type=source_type,
            idempotency_key=idempotency_key or uuid.uuid4().hex,
            metadata=dict(metadata or {}),
            scope=scope,
            shared_with=tuple(shared_with or ()),
            trusted_actor_id=trusted_actor_id,
            session_id=session_id,
            session_date=session_date or "",
            speaker=speaker,
            role=role,
        ))
        if not require_complete:
            record_operation(
                "remember",
                client=trusted_actor_id,
                duration_ms=(time.monotonic() - started) * 1000.0,
            )
            return receipt if return_receipt else list(receipt.fact_ids)

        result = command.materialize(receipt.operation_id)
        if result.state is not IngestionState.COMPLETE:
            raise RuntimeError(result.last_error or "canonical materialization failed")
    except Exception as exc:
        record_operation(
            "remember",
            client=trusted_actor_id,
            duration_ms=(time.monotonic() - started) * 1000.0,
            error=exc,
        )
        raise
    record_operation(
        "remember",
        client=trusted_actor_id,
        duration_ms=(time.monotonic() - started) * 1000.0,
    )
    return result if return_receipt else list(result.fact_ids)


def canonical_store_fn(
    engine: MemoryEngine,
    *,
    source_type: str,
    trusted_actor_id: str,
):
    """Return an AutoCapture-compatible synchronous canonical store callable."""
    def store(content: str, metadata: dict | None = None) -> list[str]:
        values = dict(metadata or {})
        return canonical_store(
            engine,
            content,
            source_type=source_type,
            trusted_actor_id=trusted_actor_id,
            metadata=values,
            session_id=str(values.get("session_id") or ""),
        )

    return store


def canonical_store_fact(
    engine: MemoryEngine,
    fact: AtomicFact,
    *,
    trusted_actor_id: str,
) -> str:
    """Durably ingest a caller-built fact without changing its public ID."""
    from superlocalmemory.core.ingestion_command import IngestionRequest, IngestionState
    from superlocalmemory.core.injection import is_low_quality

    if is_low_quality(fact.content):
        return fact.fact_id
    command = build_engine_ingestion_command(engine)
    receipt = command.submit(IngestionRequest(
        content=fact.content,
        profile_id=engine._profile_id,
        source_type="python-api-prebuilt",
        idempotency_key=f"prebuilt:{fact.fact_id}",
        metadata={_PREBUILT_FACT_KEY: _prebuilt_fact_payload(fact)},
        scope=fact.scope or "personal",
        shared_with=tuple(fact.shared_with or ()),
        trusted_actor_id=trusted_actor_id,
        session_id=fact.session_id,
        session_date=fact.observation_date or "",
    ))
    result = command.materialize(receipt.operation_id)
    if result.state is not IngestionState.COMPLETE:
        raise RuntimeError(result.last_error or "prebuilt fact materialization failed")
    if fact.fact_id not in result.fact_ids:
        raise RuntimeError("prebuilt fact ID was not preserved by canonical ingestion")
    stored = engine._db.get_fact(fact.fact_id)
    if stored is not None:
        fact.memory_id = stored.memory_id
    return fact.fact_id


def build_engine_ingestion_command(engine: MemoryEngine) -> IngestionCommand:
    """Bind one initialized engine to the durable ingestion command."""
    engine._require_full("canonical_ingestion")
    engine._ensure_init()
    repository = IngestionOperationRepository(engine._db)

    def write_queryable(request: IngestionRequest, operation_id: str) -> list[str]:
        if request.profile_id != engine._profile_id:
            raise ValueError("ingestion request profile does not match engine")
        if not request.trusted_actor_id:
            raise ValueError("trusted actor identity is required")
        hook_context = {
            "operation": "store",
            "agent_id": request.trusted_actor_id or "unknown",
            "profile_id": request.profile_id,
            "content_preview": request.content[:100],
            "ingestion_operation_id": operation_id,
        }
        # Authorization and trust policy must run before raw evidence reaches
        # durable storage.  Materialization reuses this authorization decision.
        engine._hooks.run_pre("store", hook_context)
        metadata = dict(request.metadata)
        metadata["ingestion_operation_id"] = operation_id
        if request.session_id:
            metadata.setdefault("session_id", request.session_id)
        prebuilt_payload = metadata.get(_PREBUILT_FACT_KEY)
        if isinstance(prebuilt_payload, dict):
            from superlocalmemory.storage.models import MemoryRecord

            fact = _prebuilt_fact_from_payload(prebuilt_payload)
            fact.profile_id = request.profile_id
            fact.scope = request.scope
            fact.shared_with = list(request.shared_with) or None
            fact.session_id = request.session_id or fact.session_id
            if request.session_date:
                fact.observation_date = request.session_date
            memory_id = fact.memory_id
            memory_rows = (
                engine._db.execute(
                    "SELECT memory_id FROM memories WHERE memory_id=? AND profile_id=?",
                    (memory_id, request.profile_id),
                )
                if memory_id else []
            )
            if not memory_rows:
                record = MemoryRecord(
                    memory_id=memory_id or uuid.uuid4().hex,
                    profile_id=request.profile_id,
                    content=request.content,
                    session_id=request.session_id,
                    session_date=request.session_date,
                    metadata=metadata,
                    scope=request.scope,
                    shared_with=list(request.shared_with) or None,
                )
                engine._db.store_memory(record)
                memory_id = record.memory_id
            fact.memory_id = memory_id
            engine._db.store_fact(fact)
            return [fact.fact_id]
        return engine.store_fast(
            request.content,
            metadata=metadata,
            scope=request.scope,
            shared_with=list(request.shared_with) or None,
            session_date=request.session_date or None,
            speaker=request.speaker,
            role=request.role,
            index_external=False,
        )

    def resume_checkpoint(operation: IngestionOperation) -> MaterializationResult:
        """Repair only stages whose writes have an idempotent natural key."""
        state = dict(operation.derivation_state)
        if not state.get("pipeline", False):
            return MaterializationResult(
                operation.final_fact_ids, state, operation.last_error,
            )
        facts = engine._db.get_facts_by_ids(
            list(operation.final_fact_ids), operation.profile_id,
        )
        if len(facts) != len(operation.final_fact_ids):
            state["relational"] = False
            return MaterializationResult(
                operation.final_fact_ids,
                state,
                "checkpointed relational facts are missing",
            )
        if state.get("provenance") is False and engine._provenance is not None:
            provenance_complete = True
            for fact in facts:
                existing = engine._db.execute(
                    "SELECT 1 FROM provenance WHERE fact_id=? AND profile_id=? "
                    "AND source_type=? AND source_id=? AND created_by=? LIMIT 1",
                    (
                        fact.fact_id,
                        operation.profile_id,
                        operation.source_type,
                        operation.operation_id,
                        operation.trusted_actor_id,
                    ),
                )
                if existing:
                    continue
                try:
                    engine._provenance.record(
                        fact_id=fact.fact_id,
                        profile_id=operation.profile_id,
                        source_type=operation.source_type,
                        source_id=operation.operation_id,
                        created_by=operation.trusted_actor_id,
                    )
                except Exception:
                    provenance_complete = False
            state["provenance"] = provenance_complete
        if state.get("post_hooks") is False:
            try:
                engine._hooks.run_post("store", {
                    "operation": "store",
                    "agent_id": operation.trusted_actor_id,
                    "profile_id": operation.profile_id,
                    "content_preview": operation.raw_content[:100],
                    "ingestion_operation_id": operation.operation_id,
                    "fact_ids": list(operation.final_fact_ids),
                    "fact_count": len(operation.final_fact_ids),
                })
            except Exception as exc:
                return MaterializationResult(
                    operation.final_fact_ids, state, str(exc),
                )
            state["post_hooks"] = True
        incomplete = [name for name, complete in state.items() if not complete]
        return MaterializationResult(
            operation.final_fact_ids,
            state,
            "" if not incomplete else operation.last_error,
        )

    def resume_partial_relational(
        operation: IngestionOperation,
        memory_id: str,
    ) -> MaterializationResult:
        """Finish idempotent relational effects from facts committed before a crash.

        Extraction and consolidation mutate evidence and access counters, so a
        relational-start checkpoint is an at-most-once boundary for those
        stages.  The facts already committed to this operation's dedicated
        memory can safely be promoted again: they use stable fact IDs, graph
        edge logical keys, temporal event IDs, and fact/entity association
        keys.  Deliberately omit the other best-effort enrichers here; they
        have no operation-scoped idempotency contract.
        """
        from superlocalmemory.core.store_pipeline import run_store

        # Consolidation may replace a submitted projection with an existing
        # canonical fact from another memory.  The operation checkpoint is the
        # authoritative recovery set in that case; falling back to the source
        # memory is only for a fault before the checkpoint captured final IDs.
        if operation.final_fact_ids:
            facts = engine._db.get_facts_by_ids(
                list(operation.final_fact_ids), operation.profile_id,
            )
            if len(facts) != len(operation.final_fact_ids):
                return MaterializationResult(
                    operation.final_fact_ids,
                    dict(operation.derivation_state),
                    "checkpointed relational facts are missing",
                )
        else:
            facts = engine._db.get_facts_by_memory_id(
                memory_id, operation.profile_id,
            )
        fact_ids = tuple(fact.fact_id for fact in facts)
        if not fact_ids:
            return MaterializationResult(
                (),
                dict(operation.derivation_state),
                "no committed facts available for relational recovery",
            )

        class _CommittedFactsExtractor:
            @staticmethod
            def extract_facts(**_kwargs):
                return []

        pipeline_state: dict[str, bool] = {}
        progress: dict[str, object] = {}

        def checkpoint_materialization(
            _phase: str,
            checkpoint_fact_ids: tuple[str, ...],
            state: dict[str, bool],
        ) -> None:
            repository.checkpoint_enriching(
                operation.operation_id,
                final_fact_ids=checkpoint_fact_ids,
                derivation_version=_DERIVATION_VERSION,
                derivation_state=state,
                lease_owner=operation.lease_owner,
                lease_seconds=900.0,
            )

        recovered_ids = run_store(
            operation.raw_content,
            operation.profile_id,
            session_id=operation.session_id,
            session_date=operation.session_date or None,
            speaker=operation.speaker,
            role=operation.role,
            metadata=dict(operation.metadata),
            scope=operation.scope,
            shared_with=list(operation.shared_with) or None,
            config=engine._config,
            db=engine._db,
            embedder=engine._embedder,
            fact_extractor=_CommittedFactsExtractor(),
            entity_resolver=engine._entity_resolver,
            temporal_parser=engine._temporal_parser,
            type_router=None,
            graph_builder=engine._graph_builder,
            consolidator=None,
            observation_builder=None,
            scene_builder=None,
            entropy_gate=engine._entropy_gate,
            ann_index=None,
            sheaf_checker=None,
            retrieval_engine=None,
            provenance=engine._provenance,
            hooks=engine._hooks,
            vector_store=None,
            context_generator=None,
            temporal_validator=engine._temporal_validator,
            auto_linker=None,
            consolidation_engine=None,
            existing_memory_id=memory_id,
            queryable_fact_ids=fact_ids,
            trusted_actor_id=operation.trusted_actor_id,
            pre_authorized=True,
            ingestion_source_type=operation.source_type,
            ingestion_operation_id=operation.operation_id,
            derivation_report=pipeline_state,
            precompleted_derivation_stages=frozenset({
                "extraction", "consolidation",
            }),
            materialization_progress=progress,
            materialization_checkpoint=checkpoint_materialization,
        )
        state = {
            "pipeline_started": True,
            "relational_started": True,
            "pipeline": True,
            "post_hooks": True,
            "relational": True,
            **pipeline_state,
        }
        return MaterializationResult(tuple(recovered_ids), state)

    def materialize(operation: IngestionOperation) -> MaterializationResult:
        # A checkpointed relational result is an at-most-once boundary.  The
        # extraction/consolidation pipeline mutates evidence/access counters,
        # so replaying it is not a valid repair strategy.  Idempotent external
        # projections are resumed by IngestionCommand after every relational
        # stage is complete; otherwise the durable failure remains inspectable.
        if operation.final_fact_ids and operation.derivation_state.get("pipeline", False):
            return resume_checkpoint(operation)

        facts = engine._db.get_facts_by_ids(
            list(operation.queryable_fact_ids),
            operation.profile_id,
        )
        if len(facts) != len(operation.queryable_fact_ids):
            raise ValueError("queryable fact profile mismatch or missing fact")
        memory_ids = {fact.memory_id for fact in facts}
        if len(memory_ids) != 1:
            raise ValueError("queryable facts do not share one source memory")
        memory_id = next(iter(memory_ids))

        from superlocalmemory.core.store_pipeline import run_store

        is_prebuilt = isinstance(operation.metadata.get(_PREBUILT_FACT_KEY), dict)
        if (
            operation.derivation_state.get("pipeline_started", False)
            and not operation.derivation_state.get("pipeline", False)
            and operation.derivation_state.get("relational_started", False)
        ):
            return resume_partial_relational(operation, memory_id)
        repository.checkpoint_enriching(
            operation.operation_id,
            final_fact_ids=(),
            derivation_version=_DERIVATION_VERSION,
            derivation_state={
                "pipeline_started": True,
                "pipeline": False,
            },
            lease_owner=operation.lease_owner,
            lease_seconds=900.0,
        )

        class _QueryableProjectionExtractor:
            @staticmethod
            def extract_facts(**_kwargs):
                return []

        pipeline_state: dict[str, bool] = {}
        progress: dict[str, object] = {}

        def checkpoint_materialization(
            _phase: str,
            fact_ids: tuple[str, ...],
            state: dict[str, bool],
        ) -> None:
            repository.checkpoint_enriching(
                operation.operation_id,
                final_fact_ids=fact_ids,
                derivation_version=_DERIVATION_VERSION,
                derivation_state=state,
                lease_owner=operation.lease_owner,
                lease_seconds=900.0,
            )

        try:
            fact_ids = run_store(
                operation.raw_content,
                operation.profile_id,
                session_id=operation.session_id,
                session_date=operation.session_date or None,
                speaker=operation.speaker,
                role=operation.role,
                metadata=dict(operation.metadata),
                scope=operation.scope,
                shared_with=list(operation.shared_with) or None,
                config=engine._config,
                db=engine._db,
                embedder=engine._embedder,
                fact_extractor=(
                    _QueryableProjectionExtractor()
                    if is_prebuilt else engine._fact_extractor
                ),
                entity_resolver=engine._entity_resolver,
                temporal_parser=engine._temporal_parser,
                type_router=None if is_prebuilt else engine._type_router,
                graph_builder=engine._graph_builder,
                consolidator=None if is_prebuilt else engine._consolidator,
                observation_builder=engine._observation_builder,
                scene_builder=engine._scene_builder,
                entropy_gate=engine._entropy_gate,
                # External indexes are projected only after the relational unit
                # of work commits; sqlite-vec uses a separate connection.
                ann_index=None,
                sheaf_checker=engine._sheaf_checker,
                retrieval_engine=None,
                provenance=engine._provenance,
                hooks=engine._hooks,
                vector_store=None,
                context_generator=engine._context_generator,
                temporal_validator=engine._temporal_validator,
                auto_linker=engine._auto_linker,
                consolidation_engine=engine._consolidation_engine,
                existing_memory_id=memory_id,
                queryable_fact_ids=operation.queryable_fact_ids,
                trusted_actor_id=operation.trusted_actor_id,
                pre_authorized=True,
                ingestion_source_type=operation.source_type,
                ingestion_operation_id=operation.operation_id,
                derivation_report=pipeline_state,
                precompleted_derivation_stages=(
                    frozenset({"extraction", "consolidation"})
                    if is_prebuilt else frozenset()
                ),
                materialization_progress=progress,
                materialization_checkpoint=checkpoint_materialization,
            )
        except Exception as exc:
            # ``run_store`` checkpoints the completed relational pipeline
            # immediately before post-hooks run.  Prefer that durable ledger
            # over rebuilding state from local variables: a one-time hook
            # failure must not make committed extraction/consolidation look
            # incomplete and trigger a destructive pipeline replay on retry.
            checkpoint = repository.get(operation.operation_id)
            partial_ids = tuple(checkpoint.final_fact_ids)
            failed_state = dict(checkpoint.derivation_state)
            if not partial_ids:
                partial_ids = tuple(progress.get("fact_ids") or ())
            if not partial_ids:
                partial_ids = tuple(
                    fact.fact_id
                    for fact in engine._db.get_facts_by_memory_id(
                        memory_id, operation.profile_id
                    )
                )
            if not failed_state:
                relational_complete = bool(
                    progress.get("relational_complete", False)
                )
                failed_state = {
                    **{
                        name: bool(pipeline_state.get(name, False))
                        for name in (
                            "extraction",
                            "canonicalization",
                            "consolidation",
                            "graph",
                            "temporal",
                            "provenance",
                        )
                    },
                    "pipeline_started": True,
                    "relational_started": bool(
                        progress.get("relational_started", False)
                    ),
                    "pipeline": relational_complete,
                    "post_hooks": False,
                }
            # Reaching this exception handler means hooks did not complete.
            # Keep this false even if a future hook implementation mutates the
            # in-memory progress map before raising.
            failed_state["post_hooks"] = False
            return MaterializationResult(partial_ids, failed_state, str(exc))
        if not fact_ids:
            return MaterializationResult((), {"relational": False})

        placeholders = ",".join("?" for _ in fact_ids)
        relational_count = engine._db.execute(
            f"SELECT COUNT(*) AS count FROM atomic_facts "
            f"WHERE fact_id IN ({placeholders}) AND profile_id=?",
            (*fact_ids, operation.profile_id),
        )
        fts_count = engine._db.execute(
            f"SELECT COUNT(*) AS count FROM atomic_facts_fts "
            f"WHERE fact_id IN ({placeholders})",
            tuple(fact_ids),
        )
        provenance_count = engine._db.execute(
            f"SELECT COUNT(DISTINCT fact_id) AS count FROM provenance "
            f"WHERE fact_id IN ({placeholders}) AND profile_id=? "
            "AND source_type=? AND source_id=? AND created_by=?",
            (
                *fact_ids,
                operation.profile_id,
                operation.source_type,
                operation.operation_id,
                operation.trusted_actor_id,
            ),
        )
        temporal_required = engine._temporal_validator is not None
        temporal_complete = True
        if temporal_required:
            temporal_count = engine._db.execute(
                f"SELECT COUNT(DISTINCT fact_id) AS count FROM fact_temporal_validity "
                f"WHERE fact_id IN ({placeholders}) AND profile_id=?",
                (*fact_ids, operation.profile_id),
            )
            temporal_complete = int(dict(temporal_count[0])["count"]) == len(fact_ids)

        embeddings_complete = True
        if engine._embedder is not None:
            embedding_count = engine._db.execute(
                f"SELECT COUNT(*) AS count FROM atomic_facts "
                f"WHERE fact_id IN ({placeholders}) AND embedding IS NOT NULL",
                tuple(fact_ids),
            )
            embeddings_complete = (
                int(dict(embedding_count[0])["count"]) == len(fact_ids)
            )

        derivation_state = {
            "pipeline_started": True,
            "relational_started": True,
            "pipeline": True,
            "post_hooks": True,
            "relational": int(dict(relational_count[0])["count"]) == len(fact_ids),
            "fts": int(dict(fts_count[0])["count"]) == len(fact_ids),
            "extraction": pipeline_state.get("extraction", False),
            "canonicalization": pipeline_state.get("canonicalization", False),
            "consolidation": pipeline_state.get("consolidation", False),
            "graph": pipeline_state.get("graph", False),
            "temporal": (
                temporal_complete and pipeline_state.get("temporal", False)
            ),
            "provenance": (
                int(dict(provenance_count[0])["count"]) == len(fact_ids)
                and pipeline_state.get("provenance", False)
            ),
            "trust_policy": bool(operation.trusted_actor_id),
            "embeddings": embeddings_complete,
        }
        return MaterializationResult(tuple(fact_ids), derivation_state)

    def project(operation: IngestionOperation) -> dict[str, bool]:
        facts = engine._db.get_facts_by_ids(
            list(operation.final_fact_ids),
            operation.profile_id,
        )
        if len(facts) != len(operation.final_fact_ids):
            raise RuntimeError("external projection facts are missing")

        ann_required = engine._ann_index is not None
        vector_store = engine._vector_store
        vector_required = bool(
            vector_store is not None and getattr(vector_store, "available", False)
        )
        ann_complete = True
        vector_complete = True
        bm25_complete = True
        bm25 = getattr(engine._retrieval_engine, "_bm25", None)
        for fact in facts:
            if not fact.embedding:
                if ann_required:
                    ann_complete = False
                if vector_required:
                    vector_complete = False
            if fact.embedding and ann_required:
                try:
                    engine._ann_index.add(fact.fact_id, fact.embedding)
                except Exception:
                    ann_complete = False
            if fact.embedding and vector_required:
                if not vector_store.upsert(
                    fact_id=fact.fact_id,
                    profile_id=operation.profile_id,
                    embedding=fact.embedding,
                ):
                    vector_complete = False
            if bm25 is not None:
                try:
                    bm25.add(fact.fact_id, fact.content, operation.profile_id)
                except Exception:
                    bm25_complete = False
        return {
            "ann": ann_complete,
            "vector": vector_complete,
            "bm25": bm25_complete,
        }

    return IngestionCommand(
        repository,
        write_queryable=write_queryable,
        materialize=materialize,
        project=project,
        derivation_version=_DERIVATION_VERSION,
    )


__all__ = [
    "build_engine_ingestion_command",
    "canonical_store",
    "canonical_store_fn",
    "local_trusted_actor_id",
]
