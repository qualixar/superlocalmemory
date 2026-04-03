# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under the MIT License - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""SuperLocalMemory V3 — Retrieval Engine (6-Channel Orchestrator).

6 channels -> single RRF fusion -> optional cross-encoder rerank.
Channels: semantic, BM25, entity_graph, temporal, spreading_activation, hopfield.
Replaces V1's broken 10-channel triple-re-fusion pipeline.

Part of Qualixar | Author: Varun Pratap Bhardwaj
License: MIT
"""
from __future__ import annotations

import logging
import math
import re
import time
from typing import TYPE_CHECKING, Any, Protocol

from superlocalmemory.core.config import ChannelWeights, RetrievalConfig
from superlocalmemory.retrieval.fusion import FusionResult, weighted_rrf
from superlocalmemory.retrieval.strategy import QueryStrategy, QueryStrategyClassifier
from superlocalmemory.storage.models import (
    AtomicFact, Mode, RecallResponse, RetrievalResult,
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
    """6-channel retrieval: semantic + BM25 + entity_graph + temporal + spreading_activation + hopfield.

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

        # V3.3.4: LRU cache for query embeddings (avoids redundant Ollama API calls)
        self._query_embedding_cache: dict[str, list[float]] = {}
        self._cache_max_size = 64

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
    ) -> RecallResponse:
        """Full retrieval pipeline: strategy -> channels -> RRF -> rerank."""
        t0 = time.monotonic()

        # 1. Classify query, get adaptive weights
        strat = self._strategy.classify(query, self._base_weights)

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
        effective_limit = 50 if strat.query_type == "aggregation" else limit

        # 3. Run 4 channels
        ch_results = self._run_channels(query, profile_id, strat)
        if profile_hits:
            ch_results["profile"] = profile_hits
        total = sum(len(v) for v in ch_results.values())

        # 3. Single-pass RRF fusion
        fused = weighted_rrf(ch_results, strat.weights, k=self._config.rrf_k)

        # Bridge discovery for multi-hop queries
        if self._bridge is not None and strat.query_type in ("multi_hop", "entity", "factual", "general"):
            try:
                seed_ids = [fr.fact_id for fr in fused[:10]]
                bridges = self._bridge.discover(seed_ids, profile_id, max_bridges=10)
                spread = self._bridge.spreading_activation(seed_ids, profile_id)
                extra = bridges + spread
                for fid, score in extra:
                    if not any(fr.fact_id == fid for fr in fused):
                        fused.append(FusionResult(
                            fact_id=fid, fused_score=score * 0.8,
                            channel_ranks={}, channel_scores={},
                        ))
            except Exception as exc:
                logger.warning("Bridge discovery: %s", exc)

        # Scene expansion
        if fused:
            try:
                expanded_ids: set[str] = set()
                for fr in fused[:20]:
                    scenes = self._db.get_scenes_for_fact(fr.fact_id, profile_id)
                    for scene in scenes[:2]:
                        for sfid in scene.fact_ids:
                            if not any(f.fact_id == sfid for f in fused) and sfid not in expanded_ids:
                                expanded_ids.add(sfid)
                                fused.append(FusionResult(
                                    fact_id=sfid, fused_score=fr.fused_score * 0.8,
                                    channel_ranks={}, channel_scores={},
                                ))
            except Exception as exc:
                logger.warning("Scene expansion: %s", exc)

        # 4. Load facts for rerank pool
        pool = min(len(fused), max(effective_limit * 3, 30))
        top = fused[:pool]
        facts = self._load_facts(top, profile_id)

        # 5. Cross-encoder rerank (optional)
        # Bug 4 fix: reduced alpha for multi-hop/temporal to preserve diversity
        if self._reranker is not None and facts:
            ce_alpha = 0.5 if strat.query_type in ("multi_hop", "temporal") else 0.75
            top = self._apply_reranker(query, top, facts, alpha=ce_alpha)

        # 6. Build response
        results = self._build_results(top[:effective_limit], facts, strat)
        ms = (time.monotonic() - t0) * 1000.0
        return RecallResponse(
            query=query, mode=mode, results=results,
            query_type=strat.query_type, channel_weights=strat.weights,
            total_candidates=total, retrieval_time_ms=ms,
        )

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
        self, query: str, profile_id: str, strat: QueryStrategy,
    ) -> dict[str, list[tuple[str, float]]]:
        """Run active retrieval channels. Respects disabled_channels config for ablation."""
        out: dict[str, list[tuple[str, float]]] = {}
        # Skip channels listed in disabled_channels (ablation support)
        disabled = set(self._config.disabled_channels)

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
            except Exception as exc:
                logger.warning("Query embedding failed: %s", exc)

        if self._semantic is not None and q_emb is not None and "semantic" not in disabled:
            try:
                r = self._semantic.search(q_emb, profile_id, self._config.semantic_top_k)
                if r:
                    out["semantic"] = r
            except Exception as exc:
                logger.warning("Semantic channel: %s", exc)

        if self._bm25 is not None and "bm25" not in disabled:
            try:
                r = self._bm25.search(query, profile_id, self._config.bm25_top_k)
                if r:
                    out["bm25"] = r
            except Exception as exc:
                logger.warning("BM25 channel: %s", exc)

        if self._entity is not None and "entity_graph" not in disabled:
            try:
                r = self._entity.search(query, profile_id, top_k=self._config.bm25_top_k)
                if r:
                    out["entity_graph"] = r
            except Exception as exc:
                logger.warning("Entity channel: %s", exc)

        if self._temporal is not None and "temporal" not in disabled:
            try:
                r = self._temporal.search(query, profile_id, top_k=self._config.bm25_top_k)
                if r:
                    out["temporal"] = r
            except Exception as exc:
                logger.warning("Temporal channel: %s", exc)

        # Phase G: Hopfield channel (6th) — energy-based pattern completion
        if self._hopfield is not None and q_emb is not None and "hopfield" not in disabled:
            try:
                r = self._hopfield.search(q_emb, profile_id, self._config.hopfield_top_k)
                if r:
                    out["hopfield"] = r
            except Exception as exc:
                logger.warning("Hopfield channel: %s", exc)

        # Phase 3: Spreading Activation channel (5th) — graph-based associative recall
        if self._spreading_activation is not None and q_emb is not None and "spreading_activation" not in disabled:
            try:
                r = self._spreading_activation.search(q_emb, profile_id, self._config.bm25_top_k)
                if r:
                    out["spreading_activation"] = r
            except Exception as exc:
                logger.warning("Spreading activation channel: %s", exc)

        # Apply registered post-retrieval filters (forgetting filter, etc.)
        if hasattr(self, '_registry') and self._registry._filters:
            for fn in self._registry._filters:
                try:
                    out = fn(out, profile_id, None)
                except Exception as exc:
                    logger.warning("Post-retrieval filter failed: %s", exc)

        return out

    # -- Fact loading -------------------------------------------------------

    def _load_facts(
        self, fused: list[FusionResult], profile_id: str,
    ) -> dict[str, AtomicFact]:
        """Load facts by ID — targeted query, not full-table scan.

        V3.3.13: Was loading ALL facts (O(n) memory) then filtering.
        Now uses get_facts_by_ids() for O(k) where k = pool size (~60).
        """
        needed = [fr.fact_id for fr in fused]
        if not needed:
            return {}
        facts = self._db.get_facts_by_ids(needed, profile_id)
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
    ) -> list[FusionResult]:
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
            return fused

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
            scored = self._reranker.rerank(  # type: ignore[union-attr]
                query, candidates, top_k=len(candidates),
            )
        except Exception as exc:
            logger.warning("Cross-encoder rerank failed: %s", exc)
            return fused
        finally:
            # Restore original content (with speaker tags)
            for fact, orig_content in originals:
                fact.content = orig_content

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
        return updated

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
            # Recency boost: recent facts get up to 1.1x, old facts 0.9x
            age_days = 0.0
            if fact.created_at:
                try:
                    created = datetime.fromisoformat(fact.created_at.replace("Z", "+00:00"))
                    age_days = max(0.0, (now - created).total_seconds() / 86400.0)
                except (ValueError, TypeError):
                    pass
            recency = max(0.1, 1.0 - age_days / 365.0)
            recency_boost = 1.0 + 0.2 * (recency - 0.5)

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
            confidence = min(1.0, boosted_score * 10.0) * fact.confidence
            results.append(RetrievalResult(
                fact=fact, score=boosted_score,
                channel_scores=fr.channel_scores,
                confidence=confidence, evidence_chain=evidence,
                trust_score=raw_trust,
            ))
        return results
