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

logger = logging.getLogger(__name__)

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


_PREBUILT_FACT_KEY = "_slm_prebuilt_fact_v1"


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
    """Synchronously submit and completely materialize one canonical write."""
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
        result = command.materialize(receipt.operation_id)
        if result.state is not IngestionState.COMPLETE:
            if not require_complete and receipt.fact_ids:
                import logging
                logging.getLogger(__name__).warning(
                    "Canonical operation %s remains %s and will retry: %s",
                    result.operation_id,
                    result.state.value,
                    result.last_error,
                )
                record_operation(
                    "remember", client=trusted_actor_id,
                    duration_ms=(time.monotonic() - started) * 1000.0,
                )
                return list(receipt.fact_ids)
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

    def materialize(operation: IngestionOperation) -> MaterializationResult:
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

        class _QueryableProjectionExtractor:
            @staticmethod
            def extract_facts(**_kwargs):
                return []

        pipeline_state: dict[str, bool] = {}
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
                _QueryableProjectionExtractor() if is_prebuilt else engine._fact_extractor
            ),
            entity_resolver=engine._entity_resolver,
            temporal_parser=engine._temporal_parser,
            type_router=None if is_prebuilt else engine._type_router,
            graph_builder=engine._graph_builder,
            consolidator=None if is_prebuilt else engine._consolidator,
            observation_builder=engine._observation_builder,
            scene_builder=engine._scene_builder,
            entropy_gate=engine._entropy_gate,
            # External indexes are projected only after the relational unit of
            # work commits; sqlite-vec uses a separate connection.
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
        )
        if not fact_ids:
            return MaterializationResult((), {"relational": False})
        if is_prebuilt:
            pipeline_state["extraction"] = True
            pipeline_state["consolidation"] = True

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
    )


__all__ = [
    "build_engine_ingestion_command",
    "canonical_store",
    "canonical_store_fn",
    "local_trusted_actor_id",
]
