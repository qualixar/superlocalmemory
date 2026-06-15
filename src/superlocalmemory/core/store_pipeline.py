# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Store pipeline — extracted free functions for MemoryEngine.store().

Direction: engine.py imports this module. This module NEVER imports engine.py.

Part of Qualixar | Author: Varun Pratap Bhardwaj
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from superlocalmemory.core.config import SLMConfig
    from superlocalmemory.core.hooks import HookRegistry
    from superlocalmemory.storage.database import DatabaseManager

from superlocalmemory.storage.models import (
    AtomicFact, FactType, MemoryRecord,
)

logger = logging.getLogger(__name__)

# Langevin initialization radius for new facts (ACTIVE zone < 0.3)
_INIT_LANGEVIN_RADIUS = 0.05


def _init_langevin_position(dim: int = 8) -> list[float]:
    """Initialize Langevin position near origin for a new fact.

    Small random perturbation ensures each fact gets a unique position
    while staying deep in the ACTIVE zone (radius < 0.3).
    """
    import numpy as np
    rng = np.random.default_rng()
    direction = rng.standard_normal(dim)
    norm = float(np.linalg.norm(direction))
    if norm < 1e-8:
        direction = np.ones(dim)
        norm = float(np.linalg.norm(direction))
    return (direction / norm * _INIT_LANGEVIN_RADIUS).tolist()


# ---------------------------------------------------------------------------
# enrich_fact  (was MemoryEngine._enrich_fact)
# ---------------------------------------------------------------------------

def enrich_fact(
    fact: AtomicFact,
    record: MemoryRecord,
    profile_id: str,
    *,
    embedder: Any,
    entity_resolver: Any,
    temporal_parser: Any,
) -> AtomicFact:
    """Enrich fact with embeddings, entities, temporal, emotional data."""
    from superlocalmemory.encoding.emotional import tag_emotion, emotional_importance_boost
    from superlocalmemory.encoding.signal_inference import infer_signal

    embedding = embedder.embed(fact.content) if embedder else None
    fisher_mean, fisher_variance = (None, None)
    if embedder and embedding:
        fisher_mean, fisher_variance = embedder.compute_fisher_params(embedding)

    canonical = {}
    if entity_resolver and fact.entities:
        canonical = entity_resolver.resolve(fact.entities, profile_id)

    temporal = {}
    if temporal_parser:
        temporal = temporal_parser.extract_dates_from_text(fact.content)

    emotion = tag_emotion(fact.content)
    signal = infer_signal(fact.content)

    # Strategy A: initialize Langevin position near origin (ACTIVE zone).
    # New facts start as ACTIVE; dynamics will evolve them based on access patterns.
    langevin_pos = _init_langevin_position(dim=8)

    return AtomicFact(
        fact_id=fact.fact_id, memory_id=record.memory_id,
        profile_id=profile_id, content=fact.content,
        fact_type=fact.fact_type, entities=fact.entities,
        canonical_entities=list(canonical.values()),
        observation_date=fact.observation_date or record.session_date,
        referenced_date=fact.referenced_date or temporal.get("referenced_date"),
        interval_start=fact.interval_start or temporal.get("interval_start"),
        interval_end=fact.interval_end or temporal.get("interval_end"),
        confidence=fact.confidence,
        importance=min(1.0, fact.importance + emotional_importance_boost(emotion)),
        evidence_count=fact.evidence_count,
        source_turn_ids=fact.source_turn_ids, session_id=record.session_id,
        embedding=embedding, fisher_mean=fisher_mean, fisher_variance=fisher_variance,
        langevin_position=langevin_pos,
        emotional_valence=emotion.valence, emotional_arousal=emotion.arousal,
        signal_type=signal, created_at=fact.created_at,
        pinned=getattr(fact, 'pinned', False),
        scope=getattr(fact, 'scope', 'personal'),
        shared_with=getattr(fact, 'shared_with', None),
    )


# ---------------------------------------------------------------------------
# Vector dual-write helper (P1-2 / embeddings-vector-01)
# ---------------------------------------------------------------------------

def _upsert_fact_vectors(fact, profile_id, ann_index, vector_store, embedder=None):
    """Dual-write a fact's embedding to the ANN index + sqlite-vec store.

    Embeds on-demand when the fact has no embedding (e.g. consolidated
    summary facts created without one), so UPDATE/SUPERSEDE and consolidated
    facts remain visible to the semantic channel instead of having a row in
    ``atomic_facts`` but none in the vector store.
    """
    if not getattr(fact, "embedding", None) and embedder is not None and fact.content:
        try:
            fact.embedding = embedder.embed(fact.content)
        except Exception as _emb_exc:  # pragma: no cover - defensive
            logger.debug("on-demand embed failed for %s: %s", fact.fact_id, _emb_exc)
            return
    if not getattr(fact, "embedding", None):
        return
    if ann_index:
        ann_index.add(fact.fact_id, fact.embedding)
    # V3.2: VectorStore upsert (sqlite-vec) -- dual-write (Rule 12)
    if vector_store and getattr(vector_store, "available", False):
        vector_store.upsert(
            fact_id=fact.fact_id,
            profile_id=profile_id,
            embedding=fact.embedding,
        )


# ---------------------------------------------------------------------------
# run_store  (was MemoryEngine.store)
# ---------------------------------------------------------------------------

def run_store(
    content: str,
    profile_id: str,
    session_id: str = "",
    session_date: str | None = None,
    speaker: str = "",
    role: str = "user",
    metadata: dict[str, Any] | None = None,
    *,
    scope: str = "personal",
    shared_with: list[str] | None = None,
    config: SLMConfig,
    db: DatabaseManager,
    embedder: Any,
    fact_extractor: Any,
    entity_resolver: Any,
    temporal_parser: Any,
    type_router: Any,
    graph_builder: Any,
    consolidator: Any,
    observation_builder: Any,
    scene_builder: Any,
    entropy_gate: Any,
    ann_index: Any,
    sheaf_checker: Any,
    retrieval_engine: Any,
    provenance: Any,
    hooks: HookRegistry,
    vector_store: Any = None,
    temporal_validator: Any = None,
    auto_linker: Any = None,
    context_generator: Any = None,
    consolidation_engine: Any = None,
) -> list[str]:
    """Store content and extract structured facts. Returns fact_ids.

    Multi-scope: ``scope`` sets visibility (personal/shared/global).
    ``shared_with`` is a list of profile_ids for shared scope.
    """
    # Pre-operation hooks (trust gate, ABAC, rate limiter)
    hook_ctx = {
        "operation": "store",
        "agent_id": metadata.get("agent_id", "unknown") if metadata else "unknown",
        "profile_id": profile_id,
        "content_preview": content[:100],
    }
    hooks.run_pre("store", hook_ctx)

    if entropy_gate and not entropy_gate.should_pass(content):
        return []

    # v3.5.0: store-side quality gate (H3). Reject prompt-template leakage,
    # empty placeholders, and other non-memory content BEFORE it enters the
    # DB. Uses the shared is_low_quality from core/injection so both store
    # AND injection filter by identical rules. Saves DB IO + recall pollution.
    try:
        from superlocalmemory.core.injection import is_low_quality
        if is_low_quality(content):
            logger.debug("Store rejected (low-quality content): %s...",
                         content[:80].replace("\n", " "))
            return []
    except Exception:
        pass  # Best-effort gate; store succeeds if import fails

    from superlocalmemory.encoding.temporal_parser import TemporalParser
    parser = temporal_parser or TemporalParser()
    parsed_date = parser.parse_session_date(session_date) if session_date else None

    record = MemoryRecord(
        profile_id=profile_id, content=content,
        session_id=session_id, speaker=speaker, role=role,
        session_date=parsed_date, metadata=metadata or {},
        scope=scope, shared_with=shared_with,
    )
    db.store_memory(record)

    try:
        facts = fact_extractor.extract_facts(
            turns=[content], session_id=session_id,
            session_date=parsed_date, speaker_a=speaker,
        )
    except Exception as _extract_exc:
        # P0-1 (remember-write-04): an extractor EXCEPTION (transient LLM/embed
        # backend error) must NOT orphan the already-committed memory. The None
        # guard below only handled a None *return*, not a raise. Treat a raise
        # as "no facts" so the verbatim/raw fallback persists the content.
        logger.warning(
            "extract_facts() raised — falling back to raw fact: %s", _extract_exc,
        )
        facts = None

    # v3.4.38: Defensive None guard. extract_facts() returns None on transient
    # failures (embedding worker timeout, LLM call fail). Without this guard,
    # line 201's `{f.content for f in facts}` raises 'NoneType' object is not
    # iterable, causing the caller to mark_failed permanently — even though
    # the content is still recoverable. 18 memories were lost to this between
    # April 15-26, 2026.
    if facts is None:
        facts = []

    # V3.3.11: Also store raw content as a verbatim fact to preserve details
    # that fact extraction may abstract away (dates, names, specifics).
    # This ensures BM25 and semantic search can always find the original text.
    # V3.3.12: Extract entities from verbatim content so entity channel + temporal
    # channel can find it (was entities=[] which made 4/6 channels blind).
    # V3.3.20: Stronger verbatim filter — skip greetings, filler, short phrases.
    # Verbatim facts with just "Hey! How are you?" dilute embeddings and add noise.
    _MIN_VERBATIM_WORDS = 8
    if (content.strip()
            and len(content.strip()) >= 40
            and len(content.strip().split()) >= _MIN_VERBATIM_WORDS):
        import uuid
        import re as _re
        _verbatim_text = content.strip()
        # Extract entities using the same regex as fact_extractor
        _ent_re = _re.compile(r"\b([A-Z][a-z]+(?:\s[A-Z][a-z]+){0,3})\b")
        _entity_set = {m.group(1) for m in _ent_re.finditer(_verbatim_text)}
        # Also extract all-caps abbreviations (NYU, MIT, etc.) — dedup with first set
        _entity_set |= {m.group(1) for m in _re.finditer(r'\b([A-Z]{2,})\b', _verbatim_text)}
        _verbatim_entities = sorted(_entity_set)
        verbatim = AtomicFact(
            fact_id=uuid.uuid4().hex[:16],
            content=_verbatim_text,
            fact_type=FactType.EPISODIC,
            entities=_verbatim_entities,
            session_id=session_id,
            observation_date=parsed_date,
            confidence=0.9,
            importance=0.5,
            scope=scope,
            shared_with=shared_with,
        )
        # Avoid duplicate if extraction already produced the exact same text
        extracted_texts = {f.content.strip().lower() for f in facts}
        if verbatim.content.strip().lower() not in extracted_texts:
            facts.append(verbatim)

    # V3.3.21: If fact extraction produced nothing (short input like "this is test"),
    # store the raw content as a minimal fact. User explicitly called `slm remember` —
    # their data should NEVER be silently dropped. The min-length and min-word filters
    # are designed for automatic conversation extraction, not explicit user storage.
    if not facts and content.strip():
        import uuid
        facts = [AtomicFact(
            fact_id=uuid.uuid4().hex[:16],
            content=content.strip(),
            fact_type=FactType.SEMANTIC,
            entities=[],
            session_id=session_id,
            observation_date=parsed_date,
            confidence=0.7,
            importance=0.3,
            scope=scope,
            shared_with=shared_with,
        )]

    if not facts:
        return []

    if type_router:
        facts = type_router.route_facts(facts)

    stored_ids: list[str] = []
    for fact in facts:
        fact = enrich_fact(
            fact, record, profile_id,
            embedder=embedder,
            entity_resolver=entity_resolver,
            temporal_parser=temporal_parser,
        )

        if consolidator:
            try:
                action = consolidator.consolidate(fact, profile_id)
            except Exception as _consolidate_exc:
                # P0-1 (remember-write-03): a consolidate failure (e.g. LLM
                # timeout) must NOT orphan the already-committed memory. Fall
                # back to storing the raw enriched fact so the content stays
                # retrievable across all channels.
                logger.warning(
                    "consolidate() failed for fact %s — storing raw fact as "
                    "fallback: %s", fact.fact_id, _consolidate_exc,
                )
                action = None

            if action is not None:
                if action.action_type.value == "noop":
                    continue

                # Opinion confidence tracking: reinforce or decay
                if fact.fact_type == FactType.OPINION and action.action_type.value == "update":
                    try:
                        existing = db.get_fact(action.new_fact_id)
                        if existing and existing.fact_type == FactType.OPINION:
                            new_conf = min(1.0, existing.confidence + 0.1)
                            db.update_fact(action.new_fact_id, {"confidence": new_conf})
                    except Exception:
                        pass
                elif fact.fact_type == FactType.OPINION and action.action_type.value == "supersede":
                    try:
                        old_id = getattr(action, "old_fact_id", None)
                        if old_id:
                            old_fact = db.get_fact(old_id)
                            if old_fact:
                                new_conf = max(0.0, old_fact.confidence - 0.2)
                                db.update_fact(old_id, {"confidence": new_conf})
                    except Exception:
                        pass

                if action.action_type.value in ("update", "supersede"):
                    updated_fact = db.get_fact(action.new_fact_id)
                    if updated_fact:
                        # P1-2 (embeddings-vector-01): the merged/superseding
                        # fact must reach the vector store (embed on-demand if
                        # it has none) — otherwise it is invisible to the
                        # semantic channel despite living in atomic_facts.
                        _upsert_fact_vectors(
                            updated_fact, profile_id, ann_index, vector_store, embedder,
                        )
                        if graph_builder:
                            graph_builder.build_edges(updated_fact, profile_id)
                        if observation_builder:
                            for eid in updated_fact.canonical_entities:
                                observation_builder.update_profile(
                                    eid, updated_fact, profile_id,
                                )
                    stored_ids.append(action.new_fact_id)
                    continue
                # ADD case: consolidator already stored the fact (F8 fix)
                # Fall through to post-processing below
            else:
                # Consolidate failed → store the raw fact ourselves so the
                # memory is never left without a retrievable fact, then fall
                # through to post-processing (embeddings, graph, context).
                db.store_fact(fact)
        else:
            db.store_fact(fact)

        stored_ids.append(fact.fact_id)

        # Dual-write embedding to ANN index + vector store (embed on-demand if
        # a consolidated ADD fact arrived without one). See _upsert_fact_vectors.
        _upsert_fact_vectors(fact, profile_id, ann_index, vector_store, embedder)
        # Phase 2: Generate contextual description (after consolidator, before graph_builder)
        if context_generator:
            try:
                import json as _json
                ctx_result = context_generator.generate(fact, config.mode.value)
                db.store_fact_context(
                    fact_id=fact.fact_id,
                    profile_id=profile_id,
                    contextual_description=ctx_result.description,
                    keywords=_json.dumps(ctx_result.keywords),
                    generated_by=ctx_result.generated_by,
                )
            except Exception as _ctx_exc:
                logger.debug("Context generation skipped for %s: %s", fact.fact_id, _ctx_exc)

        if graph_builder:
            graph_builder.build_edges(fact, profile_id)

        # Phase 3: AutoLinker creates association_edges (AFTER GraphBuilder)
        if auto_linker is not None:
            try:
                auto_linker.link_new_fact(fact, profile_id)
            except Exception as exc:
                logger.debug("AutoLinker.link_new_fact: %s", exc)

        # Sheaf consistency check (runs after edges exist)
        if (sheaf_checker
                and fact.embedding
                and fact.canonical_entities):
            from superlocalmemory.storage.models import EdgeType, GraphEdge
            try:
                edges_for_fact = db.get_edges_for_node(
                    fact.fact_id, profile_id,
                )
                if len(edges_for_fact) < config.math.sheaf_max_edges_per_check:
                    contradictions = sheaf_checker.check_consistency(
                        fact, profile_id,
                    )
                    for c in contradictions:
                        if c.severity > 0.45:
                            edge = GraphEdge(
                                profile_id=profile_id,
                                source_id=fact.fact_id,
                                target_id=c.fact_id_b,
                                edge_type=EdgeType.SUPERSEDES,
                                weight=c.severity,
                            )
                            db.store_edge(edge)
            except Exception as exc:
                logger.debug("Sheaf check skipped: %s", exc)

        # Phase 4: Temporal validation and contradiction detection
        if temporal_validator:
            try:
                db.store_temporal_validity(
                    fact_id=fact.fact_id,
                    profile_id=profile_id,
                    valid_from=fact.observation_date,
                    valid_until=None,
                )
                invalidations = temporal_validator.validate_and_invalidate(
                    new_fact=fact,
                    profile_id=profile_id,
                )
                if invalidations:
                    logger.info(
                        "Temporal: %d facts invalidated by new fact %s",
                        len(invalidations), fact.fact_id,
                    )
            except Exception as exc:
                logger.debug(
                    "Temporal validation skipped for fact %s: %s",
                    fact.fact_id, exc,
                )

        if observation_builder:
            for eid in fact.canonical_entities:
                observation_builder.update_profile(eid, fact, profile_id)

        # Increment fact_count for each linked canonical entity
        for eid in fact.canonical_entities:
            try:
                db.increment_entity_fact_count(eid)
            except Exception:
                pass  # Non-critical — entity may have been deleted
        if scene_builder:
            scene_builder.assign_to_scene(fact, profile_id)

        # Populate temporal_events for temporal retrieval
        has_dates = (fact.observation_date or fact.referenced_date
                     or fact.interval_start)
        if fact.canonical_entities and has_dates:
            from superlocalmemory.storage.models import TemporalEvent
            for eid in fact.canonical_entities:
                event = TemporalEvent(
                    profile_id=profile_id, entity_id=eid,
                    fact_id=fact.fact_id,
                    observation_date=fact.observation_date,
                    referenced_date=fact.referenced_date,
                    interval_start=fact.interval_start,
                    interval_end=fact.interval_end,
                    description=fact.content[:200],
                )
                db.store_temporal_event(event)

        # Foresight: extract time-bounded predictions
        try:
            from superlocalmemory.encoding.foresight import extract_foresight_signals
            from superlocalmemory.storage.models import TemporalEvent as _TE
            foresight_signals = extract_foresight_signals(fact)
            for sig in foresight_signals:
                f_event = _TE(
                    profile_id=profile_id,
                    entity_id=sig.get("entity_id", ""),
                    fact_id=fact.fact_id,
                    interval_start=sig.get("start_time"),
                    interval_end=sig.get("end_time"),
                    description=sig.get("description", ""),
                )
                db.store_temporal_event(f_event)
        except Exception as exc:
            logger.debug("Foresight extraction: %s", exc)

        # Persist BM25 tokens at ingestion
        bm25 = getattr(retrieval_engine, '_bm25', None) if retrieval_engine else None
        if bm25:
            bm25.add(fact.fact_id, fact.content, profile_id)

        # Record provenance for data lineage (EU AI Act Art. 10)
        if provenance:
            try:
                provenance.record(
                    fact_id=fact.fact_id,
                    profile_id=profile_id,
                    source_type="store",
                    source_id=session_id,
                    created_by=speaker or "unknown",
                )
            except Exception:
                pass

    logger.info("Stored %d facts (session=%s)", len(stored_ids), session_id)

    # Post-operation hooks (audit, trust signal, event bus)
    hook_ctx["fact_ids"] = stored_ids
    hook_ctx["fact_count"] = len(stored_ids)
    hooks.run_post("store", hook_ctx)

    # Phase 5: Step-count trigger for lightweight consolidation (L7)
    if consolidation_engine is not None:
        try:
            consolidation_engine.increment_store_count(profile_id)
        except Exception as _cons_exc:
            logger.debug("Consolidation step-count trigger: %s", _cons_exc)

    return stored_ids


# ---------------------------------------------------------------------------
# run_store_fact_direct  (was MemoryEngine.store_fact_direct)
# ---------------------------------------------------------------------------

def run_store_fact_direct(
    fact: AtomicFact,
    profile_id: str,
    *,
    db: DatabaseManager,
    embedder: Any,
    entity_resolver: Any,
    ann_index: Any,
    graph_builder: Any,
    retrieval_engine: Any,
    vector_store: Any = None,
) -> str:
    """Store a pre-built fact with full enrichment.

    Ensures embedding, Fisher params, canonical entities, BM25 tokens,
    and graph edges are all populated — even for auxiliary data.
    Creates a parent memory record to satisfy FK constraint.
    """
    # remember-write-02: gate low-quality content (empty, bare category tags,
    # placeholder/template leakage) at the WRITE boundary, matching run_store's
    # gate. Previously this direct path had no filter, so junk entered the KB
    # and polluted evidence/stats/embeddings (the read-side filter only hid it).
    from superlocalmemory.core.injection import is_low_quality
    if is_low_quality(fact.content):
        logger.debug("run_store_fact_direct: skipping low-quality content")
        return fact.fact_id

    # Create parent memory record (FK: atomic_facts.memory_id → memories.memory_id)
    if not fact.memory_id:
        record = MemoryRecord(
            profile_id=profile_id,
            content=fact.content[:500],
            session_id=fact.session_id,
        )
        db.store_memory(record)
        fact.memory_id = record.memory_id

    if not fact.embedding and embedder:
        fact.embedding = embedder.embed(fact.content)
        if fact.embedding:
            fact.fisher_mean, fact.fisher_variance = (
                embedder.compute_fisher_params(fact.embedding)
            )
    if entity_resolver and fact.entities:
        canonical = entity_resolver.resolve(
            fact.entities, profile_id,
        )
        fact.canonical_entities = list(canonical.values())
    db.store_fact(fact)
    # v3.4.5: Incremental sync to CozoDB/LanceDB (F-04)
    _sync_to_graph_backends(fact)
    if fact.embedding and ann_index:
        ann_index.add(fact.fact_id, fact.embedding)
    # V3.2: VectorStore upsert (dual-write)
    if fact.embedding and vector_store and vector_store.available:
        vector_store.upsert(
            fact_id=fact.fact_id,
            profile_id=profile_id,
            embedding=fact.embedding,
        )
    if graph_builder:
        graph_builder.build_edges(fact, profile_id)
    # BM25 indexing
    bm25 = getattr(retrieval_engine, '_bm25', None) if retrieval_engine else None
    if bm25:
        bm25.add(fact.fact_id, fact.content, profile_id)
    return fact.fact_id


# ---------------------------------------------------------------------------
# run_close_session  (was MemoryEngine.close_session)
# ---------------------------------------------------------------------------

def run_close_session(
    session_id: str,
    profile_id: str,
    *,
    db: DatabaseManager,
) -> int:
    """Create session-level temporal summary for session-level retrieval.

    Aggregates facts from a completed session into temporal_events
    with session scope. Enables temporal queries like "What happened
    in session 3?"

    Returns number of session summary events created.
    """
    from superlocalmemory.storage.models import TemporalEvent

    facts = db.get_all_facts(profile_id)
    session_facts = [f for f in facts if f.session_id == session_id]
    if not session_facts:
        return 0

    # Group by entity for session-level summaries
    entity_facts: dict[str, list[AtomicFact]] = {}
    for f in session_facts:
        for eid in f.canonical_entities:
            entity_facts.setdefault(eid, []).append(f)

    count = 0
    session_date = session_facts[0].observation_date or ""
    for eid, efacts in entity_facts.items():
        summary_parts = [f.content[:80] for f in efacts[:5]]
        summary = f"Session {session_id}: " + "; ".join(summary_parts)
        event = TemporalEvent(
            profile_id=profile_id,
            entity_id=eid,
            fact_id=efacts[0].fact_id,
            observation_date=session_date,
            description=summary[:500],
        )
        db.store_temporal_event(event)
        count += 1

    logger.info(
        "Session %s closed: %d summary events for %d facts",
        session_id, count, len(session_facts),
    )
    return count


# ---------------------------------------------------------------------------
# v3.4.5: Incremental sync to CozoDB/LanceDB (F-04)
# ---------------------------------------------------------------------------

def _sync_to_graph_backends(fact: Any) -> None:
    """Sync a newly stored fact to CozoDB/LanceDB.

    Non-blocking, best-effort. Called after SQLite write.
    Failures are logged, not raised — SQLite is already committed.
    """
    try:
        from superlocalmemory.core.backend_orchestrator import get_orchestrator
        orch = get_orchestrator()
        if orch is not None:
            orch.sync_new_fact(fact)
    except Exception:
        pass  # Best-effort — daemon may not have initialized yet
