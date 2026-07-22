# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""SuperLocalMemory V3 — retrieval orchestration.

Five parallel candidate producers (semantic, BM25, temporal, spreading
activation, and Hopfield) feed single-pass RRF fusion; optional profile hits
can join that fusion input. The entity graph may then score and boost fused
candidates when enabled and within the recall time budget. It is not a sixth
parallel candidate producer. Optional cross-encoder reranking follows fusion.
Replaces V1's broken 10-channel triple-re-fusion pipeline.

Part of Qualixar | Author: Varun Pratap Bhardwaj
License: AGPL-3.0-or-later
"""
from __future__ import annotations

import concurrent.futures
import functools
import logging
import math
import re
import threading
import time
from typing import TYPE_CHECKING, Any, Protocol

from superlocalmemory.core.config import ChannelWeights, RetrievalConfig
from superlocalmemory.retrieval.fusion import FusionResult, weighted_rrf
from superlocalmemory.retrieval.time_window import in_window, parse_window
from superlocalmemory.retrieval.strategy import QueryStrategy, QueryStrategyClassifier
from superlocalmemory.storage.models import (
    AtomicFact,
    Mode,
    RecallResponse,
    RetrievalResult,
)

if TYPE_CHECKING:
    from superlocalmemory.retrieval.bm25_channel import BM25Channel
    from superlocalmemory.retrieval.entity_channel import EntityGraphChannel
    from superlocalmemory.retrieval.hopfield_channel import HopfieldChannel
    from superlocalmemory.retrieval.semantic_channel import SemanticChannel
    from superlocalmemory.retrieval.temporal_channel import TemporalChannel
    from superlocalmemory.storage.database import DatabaseManager
    from superlocalmemory.trust.scorer import TrustScorer

logger = logging.getLogger(__name__)


class CrossEncoderProtocol(Protocol):
    """Duck-typed cross-encoder interface."""
    def rerank(self, query: str, candidates: list[tuple[str, str]]) -> list[tuple[str, float]]: ...


class EmbeddingProvider(Protocol):
    """Duck-typed embedding provider."""
    def embed(self, text: str) -> list[float]: ...


class RetrievalEngine:
    """Retrieval orchestrator: five candidate producers -> RRF fusion.

    Five parallel candidate producers (semantic, BM25, temporal,
    spreading_activation, hopfield) feed single-pass RRF fusion, followed by
    optional cross-encoder rerank and an optional entity-graph post-fusion
    score enhancement. Entity graph is not a sixth parallel candidate producer.

    Usage::
        engine = RetrievalEngine(db, config, channels, embedder)
        response = engine.recall("What did Alice do?", "default", Mode.A)
    """

    def __init__(
        self, db: DatabaseManager, config: RetrievalConfig,
        channels: dict[str, Any],
        embedder: EmbeddingProvider | None = None,
        reranker: CrossEncoderProtocol | None = None,
        strategy: QueryStrategyClassifier | None = None,
        base_weights: ChannelWeights | None = None,
        profile_channel: Any | None = None,
        bridge_discovery: Any | None = None,
        trust_scorer: TrustScorer | None = None,
    ) -> None:
        self._db = db
        self._config = config
        self._semantic: SemanticChannel | None = channels.get("semantic")
        self._bm25: BM25Channel | None = channels.get("bm25")
        self._entity: EntityGraphChannel | None = channels.get("entity_graph")
        self._temporal: TemporalChannel | None = channels.get("temporal")
        # Phase G: Hopfield channel (6th)
        self._hopfield: HopfieldChannel | None = channels.get("hopfield")
        # Phase 3: Spreading Activation channel
        self._spreading_activation = channels.get("spreading_activation")
        self._embedder = embedder
        self._reranker = reranker
        self._strategy = strategy or QueryStrategyClassifier()
        self._base_weights = (base_weights or ChannelWeights()).as_dict()
        self._profile_channel = profile_channel
        self._bridge = bridge_discovery
        self._trust_scorer = trust_scorer
        # v3.7.9: scope flags (include_global / include_shared) are now threaded
        # as explicit call parameters into every channel's search() method, so
        # concurrent recalls each carry their own flags — no shared mutable state,
        # no lock needed. The _scope_lock and per-recall attribute-set loop have
        # been removed. See defect S01 in the fix/3.7.9 branch notes.
        # One executor belongs to one retrieval engine. Creating/destroying six
        # worker threads on every recall caused allocator/thread-stack RSS churn
        # under sustained sessions. The scope lock already serializes channel
        # execution, so one six-worker pool preserves the existing concurrency
        # semantics while making ownership and shutdown deterministic.
        self._channel_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=6,
            thread_name_prefix="slm-recall-channel",
        )
        self._close_lock = threading.Lock()
        self._closed = False

        # V3.3.4: LRU cache for query embeddings (avoids redundant Ollama API calls)
        # V3.4.40 (2026-05-09): bumped 64 -> 512. Each cached embedding is ~3KB
        # (768 floats × 4 bytes). 512 entries ~1.5MB — trivial memory cost,
        # massive latency win on repeated queries (sub-ms vs 200-2000ms ollama).
        self._query_embedding_cache: dict[str, list[float]] = {}
        self._cache_max_size = 512

        # V3.2: ChannelRegistry for self-registration (Phase 0.5)
        from superlocalmemory.retrieval.channel_registry import ChannelRegistry
        self._registry = ChannelRegistry()
        if self._semantic is not None:
            self._registry.register_channel("semantic", self._semantic, needs_embedding=True)
        if self._bm25 is not None:
            self._registry.register_channel("bm25", self._bm25)
        if self._entity is not None:
            self._registry.register_channel("entity_graph", self._entity)
        if self._temporal is not None:
            self._registry.register_channel("temporal", self._temporal)
        # Phase G: Hopfield channel (6th) — needs embedding input
        if self._hopfield is not None:
            self._registry.register_channel("hopfield", self._hopfield, needs_embedding=True)
        # Phase 3: Spreading Activation (5th channel) — needs embedding input
        if self._spreading_activation is not None:
            self._registry.register_channel(
                "spreading_activation", self._spreading_activation, needs_embedding=True,
            )

    def recall(
        self, query: str, profile_id: str,
        mode: Mode = Mode.A, limit: int = 20,
        *,
        extra_disabled_channels: set[str] | None = None,
        include_global: bool = False,
        include_shared: bool = False,
        window: str | tuple[str, str] | None = None,
    ) -> RecallResponse:
        """Full retrieval pipeline: strategy -> channels -> RRF -> rerank.

        Multi-scope: ``include_global`` / ``include_shared`` control which
        scopes participate in retrieval. Both default to False so direct
        retrieval-engine callers are private unless they explicitly opt in.

        V3.4.40 (2026-05-09): ``extra_disabled_channels`` allows callers to
        skip specific channels for a single recall (e.g. SpreadingActivation
        for the ``--fast`` CLI flag) without mutating shared config.
        """
        t0 = time.monotonic()
        # NOTE: extra_disabled_channels is passed as an explicit local argument
        # to _run_channels() — it is NOT stored on self.  Storing it as a shared
        # mutable instance attribute (the old self._extra_disabled = ...) caused
        # a race condition where two concurrent recalls could overwrite each
        # other's channel-disable set (v3.4.64 fix).

        # v3.5.0 diagnostic: stage timing inside retrieval (SLM_RECALL_TIMING=1).
        import os as _os_e
        import time as _time_e
        _et = bool(_os_e.environ.get("SLM_RECALL_TIMING"))
        _e0 = _time_e.monotonic()

        def _em(_l: str) -> None:
            if _et:
                logger.warning("[RECALL-TIMING]   engine.%-16s %.0f ms",
                               _l, (_time_e.monotonic() - _e0) * 1000.0)

        # 1. Classify query, get adaptive weights
        strat = self._strategy.classify(query, self._base_weights)
        _em("classify")

        # Profile shortcut (runs before channel search)
        if self._profile_channel is not None:
            try:
                profile_hits = self._profile_channel.search(
                    query, profile_id, top_k=10,
                )
                if profile_hits:
                    strat.weights["profile"] = 2.0
            except Exception as exc:
                logger.warning("Profile channel: %s", exc)
                profile_hits = []
        else:
            profile_hits = []

        # Dynamic top-k for aggregation queries
        effective_limit = 100 if strat.query_type == "aggregation" else limit

        # 3. Run channels. Both scope flags AND extra_disabled_channels travel as
        # explicit call parameters so concurrent recalls with different flags
        # cannot corrupt each other.  No lock needed — no shared mutable state.
        ch_results = self._run_channels(
            query, profile_id, strat,
            extra_disabled_channels=extra_disabled_channels,
            include_global=include_global, include_shared=include_shared,
        )
        _em("run_channels")
        if profile_hits:
            ch_results["profile"] = profile_hits
        total = sum(len(v) for v in ch_results.values())

        # 3. Single-pass RRF fusion
        fused = weighted_rrf(ch_results, strat.weights, k=self._config.rrf_k)
        _em("rrf_fusion")

        # V3.3.21: Cross-channel intersection boost for multi-hop/temporal queries.
        # Problem: channels work in ISOLATION. "When did Caroline go to X?" needs
        # entity(Caroline) ∩ temporal(date). RRF averages scores but doesn't enforce
        # the intersection constraint. Fix: boost facts that appear in 2+ signal-type
        # channels (entity+temporal, entity+semantic, temporal+semantic).
        if strat.query_type == "multi_hop" and len(ch_results) >= 2:
            fused = self._apply_cross_channel_intersection(fused, ch_results, strat)

        # Bridge discovery for multi-hop queries
        # V3.3.19: Only bridge.discover() (86ms). Removed bridge.spreading_activation()
        # which did per-node SQL queries across 254K edges → 78s latency.
        # The SYNAPSE SA channel already provides proper SA with in-memory caching.
        # recall-retrieval-01: O(1) membership/score lookups instead of repeated
        # O(N) `any(...)`/`next(...)` scans inside the bridge + scene loops
        # (was O(N^2) per recall, ~400 ms on large sessions). Kept in sync as
        # `fused` grows so behaviour is identical.
        fused_ids = {fr.fact_id for fr in fused}
        fused_scores = {fr.fact_id: fr.fused_score for fr in fused}

        bridge_query_types = ("multi_hop", "entity", "factual", "general")
        if self._bridge is not None and strat.query_type in bridge_query_types:
            try:
                seed_ids = [fr.fact_id for fr in fused[:10]]
                bridges = self._bridge.discover(
                    seed_ids,
                    profile_id,
                    max_bridges=10,
                    include_global=include_global,
                    include_shared=include_shared,
                )
                for fid, score in bridges:
                    if fid not in fused_ids:
                        new_score = score * 0.8
                        fused.append(FusionResult(
                            fact_id=fid, fused_score=new_score,
                            channel_ranks={}, channel_scores={},
                        ))
                        fused_ids.add(fid)
                        fused_scores[fid] = new_score
            except Exception as exc:
                logger.warning("Bridge discovery: %s", exc)

        # Scene expansion (v3.5.0: batch + time-budgeted).
        # Skip if channels already exceeded the per-recall time budget;
        # the scene signal is nice-to-have, never worth delaying response.
        if fused and (_time_e.monotonic() - _e0) < 0.8:
            try:
                top_ids = [fr.fact_id for fr in fused[:20]]
                scenes_map = self._db.get_scenes_for_facts_batch(top_ids, profile_id)
                expanded_ids: set[str] = set()
                for fid in top_ids:
                    for scene in scenes_map.get(fid, [])[:2]:
                        for sfid in scene.fact_ids:
                            if sfid not in fused_ids and sfid not in expanded_ids:
                                expanded_ids.add(sfid)
                                new_score = fused_scores.get(fid, 0.5) * 0.8
                                fused.append(FusionResult(
                                    fact_id=sfid, fused_score=new_score,
                                    channel_ranks={}, channel_scores={},
                                ))
                                fused_ids.add(sfid)
                                fused_scores[sfid] = new_score
            except Exception as exc:
                logger.warning("Scene expansion: %s", exc)

        # V3.4.11: Entity graph signal enhancement (post-RRF boost)
        # Instead of competing as independent channel, entity_graph SCORES
        # the candidates from other channels by graph proximity to query entities.
        # Research: Microsoft GraphRAG DRIFT, Pistis-RAG cascaded architecture.
        if (self._entity is not None
                and "entity_graph" not in set(self._config.disabled_channels)
                and fused
                and (_time_e.monotonic() - _e0) < 0.9):
            try:
                candidate_ids = [fr.fact_id for fr in fused[:100]]
                eg_scores = self._entity.score_candidates(
                    query,
                    candidate_ids,
                    profile_id,
                    include_global=include_global,
                    include_shared=include_shared,
                )
                if eg_scores:
                    boosted = []
                    for fr in fused:
                        eg_sc = eg_scores.get(fr.fact_id, 0.0)
                        if eg_sc > 0:
                            eg_weight = strat.weights.get("entity_graph", 1.0)
                            boost = 1.0 + eg_sc * eg_weight * 0.3
                            boosted.append(FusionResult(
                                fact_id=fr.fact_id,
                                fused_score=fr.fused_score * boost,
                                channel_ranks=fr.channel_ranks,
                                channel_scores={**fr.channel_scores, "entity_graph": eg_sc},
                            ))
                        else:
                            boosted.append(fr)
                    fused = sorted(boosted, key=lambda r: r.fused_score, reverse=True)
            except Exception as exc:
                logger.warning("Entity graph signal enhancement: %s", exc)

        _em("expand+entity_enh")

        # T-window: prune candidates to the requested event-time range.
        # Additive — a None or unparseable window applies no filter. Event
        # times are fetched for the bounded candidate set only (indexed), then
        # in-range facts are kept. Runs before fact load so out-of-window facts
        # are never materialized.
        if window is not None and fused:
            bounds = parse_window(window)
            if bounds is not None:
                etimes = self._db.get_fact_event_times(
                    [fr.fact_id for fr in fused], profile_id,
                )
                fused = [
                    fr for fr in fused
                    if in_window(etimes.get(fr.fact_id), bounds)
                ]
                _em("time_window")

        # 4. Load facts for rerank pool
        pool = min(len(fused), max(effective_limit * 3, 30))
        top = fused[:pool]
        facts = self._load_facts(
            top,
            profile_id,
            include_global=include_global,
            include_shared=include_shared,
        )
        _em("load_facts")

        # V3.3.21: Session diversity for aggregation queries.
        if strat.query_type == "aggregation" and facts:
            top = self._enforce_session_diversity(top, facts, min_sessions=3, top_k=20)

        # 5. Cross-encoder rerank (optional)
        # Bug 4 fix: reduced alpha for multi-hop/temporal to preserve diversity
        # V3.3.21: Skip reranker if worker isn't ready yet (cold start).
        # Returns results without CE reranking (~5-10pp lower quality) but instant
        # instead of blocking 15-19s on first recall. Worker warms up in background.
        reranker_ready = (
            self._reranker is not None
            and getattr(self._reranker, '_worker_ready', False)
        )
        reranker_applied = False
        reranker_status = (
            "fallback_not_ready" if self._reranker is not None
            else "not_configured"
        )
        if reranker_ready and facts:
            ce_alpha = 0.5 if strat.query_type in ("multi_hop", "temporal") else 0.75
            top, reranker_applied, reranker_status = self._apply_reranker(
                query, top, facts, alpha=ce_alpha,
            )
        elif reranker_ready:
            reranker_status = "no_candidates"
        _em(f"rerank(ready={reranker_ready})")

        # v3.6.6: Evidence floor — gate on per-channel scores (NOT fused/RRF score).
        # Nonsense queries fuse at 0.75-0.78 because RRF is rank-derived and
        # uncalibrated. The discriminator is EARNED CHANNEL EVIDENCE:
        #   semantic >= min_semantic_evidence (0.60) OR bm25 > 0
        #   OR entity_graph > 0 OR temporal > 0 OR fact is pinned.
        # spreading_activation and hopfield do NOT count — they are associative
        # amplifiers that fabricated the nonsense results in calibration tests.
        # Kill-switch: SLM_RECALL_NO_FLOOR=1 bypasses the floor.
        import os as _os_floor
        floor_enabled = (
            getattr(self._config, "evidence_floor_enabled", True)
            and _os_floor.environ.get("SLM_RECALL_NO_FLOOR", "0") != "1"
        )
        if floor_enabled:
            min_sem = getattr(self._config, "min_semantic_evidence", 0.60)
            # Qualify the rerank pool BEFORE applying the caller's limit.  RRF
            # can rank associative-only hits above an exact BM25 match; slicing
            # first allowed those hits to occupy every output slot and then be
            # removed by the floor, producing a false abstention even though a
            # qualified candidate was immediately below the slice.
            top = self._apply_evidence_floor(top, facts, min_sem)

        # V3.4.11: Channel diversity — guarantee entity_graph results appear in
        # the final output. Applied AFTER reranking and evidence qualification
        # so an associative-only candidate cannot be reintroduced after the gate.
        final_top = top[:effective_limit]
        final_top = self._enforce_channel_diversity(
            final_top, fused, ch_results, effective_limit,
        )

        # A channel-diversity promotion may come from outside the rerank pool.
        # Load only when that happens; ordinary recalls reuse the existing map.
        if any(fr.fact_id not in facts for fr in final_top):
            facts.update(self._load_facts(
                final_top,
                profile_id,
                include_global=include_global,
                include_shared=include_shared,
            ))

        # Trim facts to the selected, qualified result set.
        selected_ids = {fr.fact_id for fr in final_top}
        facts = {fid: f for fid, f in facts.items() if fid in selected_ids}

        # 6. Build response
        results = self._build_results(final_top, facts, strat)
        ms = (time.monotonic() - t0) * 1000.0
        no_match = floor_enabled and len(results) == 0
        return RecallResponse(
            query=query, mode=mode, results=results,
            query_type=strat.query_type, channel_weights=strat.weights,
            total_candidates=total, retrieval_time_ms=ms,
            no_confident_match=no_match,
            reranker_applied=reranker_applied,
            reranker_status=reranker_status,
        )

    # -- Evidence floor (v3.6.6) -------------------------------------------

    @staticmethod
    def _apply_evidence_floor(
        final_top: list[FusionResult],
        facts: dict[str, AtomicFact],
        min_semantic: float,
    ) -> list[FusionResult]:
        """Filter results that earned no channel evidence.

        Keep a result only if it earned:
          - semantic cosine >= min_semantic (default 0.60), OR
          - bm25 > 0, OR entity_graph > 0, OR temporal > 0, OR
          - the underlying fact is pinned.

        spreading_activation and hopfield do NOT count as primary evidence.
        Empty result after filtering is a success (no_confident_match=True).
        """
        kept: list[FusionResult] = []
        for fr in final_top:
            cs = fr.channel_scores or {}
            # Primary channel evidence check
            if (
                cs.get("semantic", 0.0) >= min_semantic
                or cs.get("bm25", 0.0) > 0.0
                or cs.get("entity_graph", 0.0) > 0.0
                or cs.get("temporal", 0.0) > 0.0
            ):
                kept.append(fr)
                continue
            # Pinned fact bypass — always pass regardless of channel scores
            fact = facts.get(fr.fact_id)
            if fact is not None and getattr(fact, "pinned", False):
                kept.append(fr)
        return kept

    # -- Cross-channel intersection boost -----------------------------------

    @staticmethod
    def _apply_cross_channel_intersection(
        fused: list[FusionResult],
        ch_results: dict[str, list[tuple[str, float]]],
        strat: QueryStrategy,
    ) -> list[FusionResult]:
        """Boost facts that appear across multiple signal-type channels.

        V3.3.21: Solves the channel isolation problem. When a query has both
        entity and temporal signals (e.g., "When did Caroline go to X?"), facts
        matching BOTH dimensions should rank higher than facts matching only one.

        Channel groups:
          - content: semantic, bm25 (text similarity)
          - structure: entity_graph, spreading_activation (graph structure)
          - temporal: temporal (date proximity)
          - associative: hopfield (pattern completion)

        Boost: facts in 2+ groups get 1.5x, facts in 3+ groups get 2.0x.
        """
        # Map channels to signal groups
        _CHANNEL_GROUPS = {
            "semantic": "content", "bm25": "content",
            "entity_graph": "structure", "spreading_activation": "structure",
            "temporal": "temporal",
            "hopfield": "associative",
            "profile": "content",
        }

        # Build fact_id -> set of signal groups it appears in
        fact_groups: dict[str, set[str]] = {}
        for ch_name, results in ch_results.items():
            group = _CHANNEL_GROUPS.get(ch_name, ch_name)
            for fid, _score in results:
                if fid not in fact_groups:
                    fact_groups[fid] = set()
                fact_groups[fid].add(group)

        # Apply boost based on cross-group coverage
        boosted: list[FusionResult] = []
        for fr in fused:
            groups = fact_groups.get(fr.fact_id, set())
            n_groups = len(groups)
            if n_groups >= 3:
                boost = 2.0
            elif n_groups >= 2:
                # Extra boost for temporal+structure intersection (the exact gap)
                if "temporal" in groups and "structure" in groups:
                    boost = 1.8
                else:
                    boost = 1.5
            else:
                boost = 1.0
            boosted.append(FusionResult(
                fact_id=fr.fact_id,
                fused_score=fr.fused_score * boost,
                channel_ranks=fr.channel_ranks,
                channel_scores=fr.channel_scores,
            ))
        boosted.sort(key=lambda r: r.fused_score, reverse=True)
        return boosted

    # -- Session diversity enforcement ----------------------------------------

    @staticmethod
    def _enforce_session_diversity(
        fused: list[FusionResult],
        fact_map: dict[str, AtomicFact],
        min_sessions: int = 3,
        top_k: int = 20,
    ) -> list[FusionResult]:
        """Ensure top-k results span at least min_sessions different session_ids.

        V3.3.21: Category 1 (aggregation) needs facts from MULTIPLE sessions —
        95.7% of cat 1 questions require cross-session evidence. Without this,
        top-20 may cluster around 1-2 sessions, missing scattered mentions.

        Algorithm: if top-k has < min_sessions, promote the highest-scored facts
        from underrepresented sessions into the top-k window.
        """
        if len(fused) <= top_k:
            return fused

        top = fused[:top_k]
        rest = fused[top_k:]

        sessions_in_top: set[str] = set()
        for fr in top:
            fact = fact_map.get(fr.fact_id)
            if fact and fact.session_id:
                sessions_in_top.add(fact.session_id)

        if len(sessions_in_top) >= min_sessions:
            return fused

        promoted: list[FusionResult] = []
        for fr in rest:
            fact = fact_map.get(fr.fact_id)
            if fact and fact.session_id and fact.session_id not in sessions_in_top:
                sessions_in_top.add(fact.session_id)
                promoted.append(fr)
                if len(sessions_in_top) >= min_sessions:
                    break

        if not promoted:
            return fused

        promoted_ids = {fr.fact_id for fr in promoted}
        remaining = [fr for fr in rest if fr.fact_id not in promoted_ids]
        return top + promoted + remaining

    # -- Channel diversity enforcement ----------------------------------------

    @staticmethod
    def _enforce_channel_diversity(
        top: list,
        fused: list,
        ch_results: dict[str, list[tuple[str, float]]],
        effective_limit: int,
        min_per_channel: int = 2,
    ) -> list:
        """Ensure structure channels (entity_graph) get representation.

        V3.4.11: entity_graph finds valid results but RRF scores them low
        because they don't overlap with semantic/bm25 results. This interleaves
        top entity_graph facts into positions 3-4 of the final output instead
        of appending at the end where they'd never be seen.
        """
        structure_channels = ["entity_graph"]
        top_ids = {fr.fact_id for fr in top}

        promoted = []
        for ch_name in structure_channels:
            ch_items = ch_results.get(ch_name, [])
            if not ch_items:
                continue

            present = sum(1 for fid, _ in ch_items if fid in top_ids)
            if present >= min_per_channel:
                continue

            needed = min_per_channel - present
            ch_fids = {fid for fid, _ in ch_items}
            for fr in fused:
                if fr.fact_id in ch_fids and fr.fact_id not in top_ids:
                    promoted.append(fr)
                    top_ids.add(fr.fact_id)
                    needed -= 1
                    if needed <= 0:
                        break

        if not promoted:
            return top

        # Append as safety net — with proper RRF weights (strategy.py),
        # entity_graph facts should already rank naturally in the top-k.
        # This only fires when they're still missing despite weight boost.
        return list(top) + promoted

    # -- Channel execution --------------------------------------------------

    def _embed_query(self, query: str) -> list[float] | None:
        """Embed query with LRU cache. Avoids redundant Ollama/API calls."""
        if self._embedder is None:
            return None
        cached = self._query_embedding_cache.get(query)
        if cached is not None:
            return cached
        emb = self._embedder.embed(query)
        # Evict oldest if cache full
        if len(self._query_embedding_cache) >= self._cache_max_size:
            oldest = next(iter(self._query_embedding_cache))
            del self._query_embedding_cache[oldest]
        self._query_embedding_cache[query] = emb
        return emb

    def _run_channels(
        self,
        query: str,
        profile_id: str,
        strat: QueryStrategy,
        *,
        extra_disabled_channels: set[str] | None = None,
        include_global: bool = False,
        include_shared: bool = False,
    ) -> dict[str, list[tuple[str, float]]]:
        """Run active retrieval channels.

        v3.4.53: channels run in PARALLEL via ThreadPoolExecutor. Industry
        standard (EverMemOS, szl-recall, ContentPilot 2026): all channels
        are independent after embedding; running them serially wastes time
        equal to the sum of all producer latencies. When multiple producers are
        enabled and healthy, parallel dispatch generally bounds the producer
        phase by the slowest submitted producer, plus serial embedding and
        result-collection overhead.
        """
        import os as _os_e
        import time as _time_e
        _et = bool(_os_e.environ.get("SLM_RECALL_TIMING"))
        out: dict[str, list[tuple[str, float]]] = {}
        # Skip channels listed in disabled_channels (ablation support)
        # V3.4.40: union with per-recall extra_disabled set (e.g. --fast skip)
        # V3.4.64: extra_disabled is now a local parameter, not a shared instance
        # attribute — eliminates the concurrent-recall race condition.
        disabled = set(self._config.disabled_channels) | set(extra_disabled_channels or ())

        # V3.3.4: Embed query ONCE, reuse for semantic + hopfield channels
        q_emb: list[float] | None = None
        needs_embedding = (
            (self._semantic is not None and "semantic" not in disabled)
            or (self._hopfield is not None and "hopfield" not in disabled)
            or (self._spreading_activation is not None and "spreading_activation" not in disabled)
        )
        if needs_embedding:
            try:
                q_emb = self._embed_query(query)
                if q_emb is None:
                    logger.warning(
                        "Query embedding returned None — semantic, hopfield, "
                        "spreading_activation channels will be skipped this recall"
                    )
            except Exception as exc:
                logger.warning("Query embedding failed: %s", exc)

        # v3.4.53: collect channel callables and run in parallel.
        # Each channel is a standalone search — no shared mutable state,
        # no ordering dependencies. SQLite WAL mode permits concurrent reads.
        futures: dict[str, concurrent.futures.Future] = {}

        def _safe_channel(name: str, fn, *args):
            """Run a single channel, returning (name, result_or_None)."""
            _cs = _time_e.monotonic() if _et else 0.0
            try:
                res = fn(*args)
                if _et:
                    logger.warning("[RECALL-TIMING]     channel.%-16s %.0f ms",
                                   name, (_time_e.monotonic() - _cs) * 1000.0)
                return (name, res if res else None)
            except Exception as exc:
                logger.warning("%s channel: %s", name, exc)
                return (name, None)

        executor = self._channel_executor
        if self._semantic is not None and q_emb is not None and "semantic" not in disabled:
            futures["semantic"] = executor.submit(
                _safe_channel, "semantic",
                functools.partial(
                    self._semantic.search,
                    include_global=include_global, include_shared=include_shared,
                ),
                q_emb, profile_id, self._config.semantic_top_k,
            )
        if self._bm25 is not None and "bm25" not in disabled:
            futures["bm25"] = executor.submit(
                _safe_channel, "bm25",
                functools.partial(
                    self._bm25.search,
                    include_global=include_global, include_shared=include_shared,
                ),
                query, profile_id, self._config.bm25_top_k,
            )
        if self._temporal is not None and "temporal" not in disabled:
            futures["temporal"] = executor.submit(
                _safe_channel, "temporal",
                functools.partial(
                    self._temporal.search,
                    include_global=include_global, include_shared=include_shared,
                ),
                query, profile_id, self._config.bm25_top_k,
            )
        if self._hopfield is not None and q_emb is not None and "hopfield" not in disabled:
            futures["hopfield"] = executor.submit(
                _safe_channel, "hopfield",
                functools.partial(
                    self._hopfield.search,
                    include_global=include_global, include_shared=include_shared,
                ),
                q_emb, profile_id, self._config.hopfield_top_k,
            )
        if (
            self._spreading_activation is not None
            and q_emb is not None
            and "spreading_activation" not in disabled
        ):
            futures["spreading_activation"] = executor.submit(
                _safe_channel, "spreading_activation",
                functools.partial(
                    self._spreading_activation.search,
                    include_global=include_global, include_shared=include_shared,
                ),
                q_emb, profile_id, self._config.bm25_top_k,
            )

        # Collect results as channels complete.
        for name, fut in futures.items():
            try:
                ch_name, result = fut.result(timeout=30)
                if result:
                    out[ch_name] = result
            except Exception as exc:
                logger.warning("Channel %s timed out or failed: %s", name, exc)

        # Apply registered post-retrieval filters (forgetting filter, etc.)
        if hasattr(self, '_registry') and self._registry._filters:
            for fn in self._registry._filters:
                try:
                    out = fn(out, profile_id, None)
                except Exception as exc:
                    logger.warning("Post-retrieval filter failed: %s", exc)

        return out

    def close(self) -> None:
        """Release the channel workers owned by this retrieval engine."""
        with self._close_lock:
            if self._closed:
                return
            self._closed = True
        self._channel_executor.shutdown(wait=True, cancel_futures=True)

    # -- Fact loading -------------------------------------------------------

    def _load_facts(
        self,
        fused: list[FusionResult],
        profile_id: str,
        *,
        include_global: bool = False,
        include_shared: bool = False,
    ) -> dict[str, AtomicFact]:
        """Load facts by ID — targeted query, not full-table scan.

        V3.3.13: Was loading ALL facts (O(n) memory) then filtering.
        Now uses get_facts_by_ids() for O(k) where k = pool size (~60).
        """
        needed = [fr.fact_id for fr in fused]
        if not needed:
            return {}
        facts = self._db.get_facts_by_ids(
            needed, profile_id,
            include_global=include_global,
            include_shared=include_shared,
        )
        return {f.fact_id: f for f in facts}

    # -- Cross-encoder rerank -----------------------------------------------

    @staticmethod
    def _sigmoid(x: float) -> float:
        """Numerically stable sigmoid."""
        x = max(-500.0, min(500.0, x))
        return 1.0 / (1.0 + math.exp(-x))

    def _apply_reranker(
        self, query: str, fused: list[FusionResult],
        fact_map: dict[str, AtomicFact],
        alpha: float = 0.75,
    ) -> tuple[list[FusionResult], bool, str]:
        """Rerank with blended CE + RRF scores (Bug 1 fix).

        Blended: alpha * sigmoid(CE_score) + (1 - alpha) * rrf_score.
        Speaker tags stripped before scoring (Bug 3 fix).
        """
        # Bug 2 fix: score ALL candidates, not just top_k
        candidates = [
            (fact_map[fr.fact_id], fr.fused_score)
            for fr in fused if fr.fact_id in fact_map
        ]
        if not candidates:
            return fused, False, "no_candidates"

        # V3.3.16: Strip speaker tags WITHOUT copying full AtomicFact objects.
        # Previously created full copies including 768-dim embeddings (~6KB each),
        # which over 304 recalls caused pymalloc arena fragmentation → 25GB.
        # Now: temporarily patch .content on originals, rerank, then restore.
        originals: list[tuple[AtomicFact, str]] = []  # (fact, original_content)
        for fact, _ in candidates:
            orig = fact.content
            fact.content = re.sub(r'^\[[A-Za-z]+\]:\s*', '', orig)
            originals.append((fact, orig))

        try:
            rerank_with_status = getattr(
                self._reranker, "rerank_with_status", None,
            )
            # MagicMock fabricates arbitrary attributes; only use the richer
            # contract when it is defined by the reranker type itself.
            if callable(rerank_with_status) and hasattr(
                type(self._reranker), "rerank_with_status",
            ):
                scored, applied, status = rerank_with_status(
                    query, candidates, top_k=len(candidates),
                )
            else:
                scored = self._reranker.rerank(  # type: ignore[union-attr]
                    query, candidates, top_k=len(candidates),
                )
                applied, status = True, "applied"
        except Exception as exc:
            logger.warning("Cross-encoder rerank failed: %s", exc)
            return fused, False, "error"
        finally:
            # Restore original content (with speaker tags)
            for fact, orig_content in originals:
                fact.content = orig_content

        if not applied:
            return fused, False, status

        score_map = {fact.fact_id: score for fact, score in scored}

        # Min-max normalize CE scores to [0, 1] within the batch instead of
        # sigmoid (which compresses the useful discrimination range).
        ce_values = list(score_map.values())
        ce_min = min(ce_values) if ce_values else 0.0
        ce_max = max(ce_values) if ce_values else 1.0
        ce_range = ce_max - ce_min if ce_max > ce_min else 1.0

        # Also normalize RRF scores so both terms contribute meaningfully
        rrf_values = [fr.fused_score for fr in fused]
        rrf_max = max(rrf_values) if rrf_values else 1.0
        rrf_max = rrf_max if rrf_max > 0 else 1.0

        updated = [
            FusionResult(
                fact_id=fr.fact_id,
                fused_score=(
                    alpha * ((score_map.get(fr.fact_id, ce_min) - ce_min) / ce_range)
                    + (1.0 - alpha) * (fr.fused_score / rrf_max)
                ),
                channel_ranks=fr.channel_ranks,
                channel_scores=fr.channel_scores,
            )
            for fr in fused
        ]
        updated.sort(key=lambda r: r.fused_score, reverse=True)
        return updated, True, "applied"

    # -- Agentic adapter -----------------------------------

    def recall_facts(
        self, query: str, profile_id: str,
        top_k: int = 20, skip_agentic: bool = True,
    ) -> list[tuple[AtomicFact, float]]:
        """Simplified recall returning (fact, score) tuples.

        Used by AgenticRetriever for round-2 re-retrieval.
        skip_agentic is always True here to prevent infinite recursion.
        """
        response = self.recall(query, profile_id, limit=top_k)
        return [(r.fact, r.score) for r in response.results]

    # -- Trust weighting ----------------------------------------------------

    def _get_trust_weight(self, fact: AtomicFact, profile_id: str) -> tuple[float, float]:
        """Look up Bayesian trust score and convert to a multiplicative weight.

        Returns (trust_weight, raw_trust_score).
        trust_weight is clamped to [0.5, 1.5]:
          - trust=0.0 -> weight=0.5  (demote untrusted facts)
          - trust=0.5 -> weight=1.0  (neutral, default prior)
          - trust=1.0 -> weight=1.5  (promote highly trusted facts)
        If trust scoring is disabled or unavailable, returns (1.0, 0.5).
        """
        if not self._config.use_trust_weighting or self._trust_scorer is None:
            return 1.0, 0.5

        try:
            raw = self._trust_scorer.get_fact_trust(fact.fact_id, profile_id)
        except Exception:
            return 1.0, 0.5

        # Linear map: trust 0.0->0.5, 0.5->1.0, 1.0->1.5
        weight = 0.5 + raw  # raw in [0, 1] -> weight in [0.5, 1.5]
        return weight, raw

    # -- Response building --------------------------------------------------

    def _build_results(
        self, fused: list[FusionResult], fact_map: dict[str, AtomicFact],
        strat: QueryStrategy,
    ) -> list[RetrievalResult]:
        from datetime import UTC, datetime
        now = datetime.now(UTC)
        results: list[RetrievalResult] = []
        profile_id = next(
            (f.profile_id for f in fact_map.values()), "default",
        )
        for fr in fused:
            fact = fact_map.get(fr.fact_id)
            if fact is None:
                continue
            evidence = [
                f"{ch}(rank={rk}, score={fr.channel_scores.get(ch, 0.0):.4f})"
                for ch, rk in sorted(fr.channel_ranks.items(), key=lambda x: x[1])
                if rk < 1000
            ]
            # Recency decay: Ebbinghaus exponential + FSRS stability strengthening (v3.4.51).
            #
            # Base: R = e^(-λt),  λ = ln(2)/S,  S = effective half-life in days.
            # FSRS v5 (Dae & Jarrett 2024): S grows with successful recall frequency.
            #   S_effective = S_base × min(2.0, 1 + 0.1 × access_count)
            #   → 0 recalls: S=30d  5 recalls: S=45d  10+ recalls: S=60d (max)
            # Effect: frequently-recalled architectural decisions resist decay naturally;
            # one-off session handoffs and debug notes decay at full rate.
            #
            # Boost range: [0.80×, 1.10×]
            #   0d, 0acc → 1.10×   45d, 0acc → 0.91×   90d, 0acc → 0.84×
            #   45d, 5acc → 0.95×  90d, 10acc → 0.90×  (frequently used memories stay relevant)
            age_days = 0.0
            if fact.created_at:
                try:
                    created = datetime.fromisoformat(fact.created_at.replace("Z", "+00:00"))
                    age_days = max(0.0, (now - created).total_seconds() / 86400.0)
                except (ValueError, TypeError):
                    pass
            _access = max(0, getattr(fact, "access_count", 0) or 0)
            _S = 30.0 * min(2.0, 1.0 + 0.1 * _access)
            recency_boost = 0.8 + 0.3 * math.exp(-(math.log(2) / _S) * age_days)

            # Content quality: penalize short/low-info facts that rank high
            # due to BM25 name-matching (greetings like "Hey Caroline!" score high
            # on BM25 but have zero retrieval value)
            content_len = len(fact.content.strip())
            if content_len < 10:
                quality = 0.3
            elif content_len < 25:
                quality = 0.7
            else:
                quality = 1.0

            # Trust weighting: Bayesian trust modulates final ranking
            trust_weight, raw_trust = self._get_trust_weight(fact, profile_id)

            boosted_score = fr.fused_score * recency_boost * quality * trust_weight
            # v3.5.0 (M2): soft-normalize to [0,1]. RRF weights + scene/entity
            # boosts push raw scores well above 1 (observed: 27.97). A sigmoid
            # preserves rank (monotonic) while giving users a readable 0-1 range.
            normalized_score = 1.0 / (1.0 + math.exp(-boosted_score * 0.5))
            results.append(RetrievalResult(
                fact=fact, score=round(normalized_score, 4),
                channel_scores=fr.channel_scores,
                confidence=fact.confidence,
                relevance_score=round(normalized_score, 4),
                ranking_score=boosted_score,
                memory_confidence=fact.confidence,
                evidence_chain=evidence,
                trust_score=raw_trust,
            ))
        return results


# ---------------------------------------------------------------------------
# apply_channel_weights (LLD-03 §5.5 — module-level pure helper)
# ---------------------------------------------------------------------------


_CHANNEL_KEYS: tuple[str, ...] = (
    "semantic", "bm25", "entity_graph", "temporal",
    # hopfield + spreading_activation are real retrieval channels (score
    # contract v2) with bandit-chosen weights; omitting them here silently
    # discarded adaptive reranking for multi-hop relational recall.
    "spreading_activation", "hopfield",
)


def apply_channel_weights(
    candidates: list[RetrievalResult],
    weights: dict[str, float] | None,
) -> list[RetrievalResult]:
    """Re-score candidates under a bandit-chosen weight bundle.

    Multiplies each candidate's ``channel_scores[ch]`` by ``weights[ch]``
    and applies ``cross_encoder_bias`` to the final score. Preserves order;
    callers reorder via ensemble_rerank.

    Returns a NEW list with new ``RetrievalResult`` instances — never mutates
    input. Unknown / missing weights default to 1.0.

    Safe against ``weights=None`` (returns input unchanged) and empty lists.
    """
    if not candidates or not weights:
        return list(candidates)

    ce_bias = float(weights.get("cross_encoder_bias", 1.0))
    out: list[RetrievalResult] = []
    for c in candidates:
        original_cs = c.channel_scores or {}
        new_cs: dict[str, float] = dict(original_cs)
        base = 0.0
        for ch in _CHANNEL_KEYS:
            raw = float(original_cs.get(ch, 0.0))
            w = float(weights.get(ch, 1.0))
            scaled = raw * w
            new_cs[ch] = scaled
            base += scaled
        new_score = (base if base > 0.0 else float(c.score)) * ce_bias
        out.append(RetrievalResult(
            fact=c.fact,
            score=c.score,
            channel_scores=new_cs,
            confidence=c.confidence,
            relevance_score=c.relevance_score,
            ranking_score=new_score,
            memory_confidence=c.memory_confidence,
            rank_position=c.rank_position,
            evidence_chain=c.evidence_chain,
            trust_score=c.trust_score,
            marker=c.marker,
        ))
    return out
