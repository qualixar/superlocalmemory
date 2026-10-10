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
import os
import re
import threading
import time
from typing import TYPE_CHECKING, Any, Protocol

from superlocalmemory.core.config import (
    CANONICAL_RECALL_LIMIT,
    ChannelWeights,
    RetrievalConfig,
)
from superlocalmemory.retrieval import (channel_status as chstat, entity_graph_warmup,
                                      kind_scope, project_search)
from superlocalmemory.retrieval.fusion import FusionResult, weighted_rrf
from superlocalmemory.retrieval.rerank_pool import rerank_pool
from superlocalmemory.retrieval.strategy import QueryStrategy, QueryStrategyClassifier
from superlocalmemory.retrieval.temporal_validity_filter import (
    CorrectionAdmissionCache,
    admit_correction_candidates,
    admit_correction_fusion_results,
)
from superlocalmemory.retrieval.time_window import (
    has_primary_evidence,
    in_window,
    infer_window_from_query,
    parse_window,
    windowed_candidates,
)
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


# How long the parallel channel phase may run before a channel is abandoned.
#
# This is a guard against a genuinely wedged channel, NOT a speed cutoff, and
# the distinction is the whole point. A channel that misses this limit is
# cancelled and contributes NOTHING to fusion: its candidates are not reordered,
# they are absent. So whenever this limit binds, the answer is decided partly by
# what else the machine happened to be doing — the same question returns a
# different answer under load, which is a correctness failure, not a slow one.
#
# It replaced a 1.4 s cutoff that was chosen to keep the recall p95 low. Six
# runs of identical code against the same 0.95 GB store logged 0, 0, 25, 2, 0
# and 0 abandoned channels; in the third run `hopfield` was cut off on 13 of
# 140 queries and `temporal` on 9, while the first run lost nothing on those
# same queries. That spread was the last remaining source of unrepeatable
# recall, and it is why fixing tie-breaks everywhere else moved top-10 churn
# from 40.7% to 22.9% and left rank-1 disagreement sitting at ~15%: a
# tie-break cannot repair a missing input.
#
# The value comes from the measured cost of the channels themselves, on that
# same store, 140 queries, with the limit raised out of the way so nothing was
# truncated (p95 / max, ms):
#
#     temporal  580 / 1983      hopfield  492 / 945      bm25  230 / 1093
#     semantic  264 /  662      spreading_activation  238 / 571
#
# The slowest channel's p95 is 580 ms and its worst single run was 1,983 ms, so
# 8 s is roughly four times the worst observed cost — it should never bind on a
# machine that is merely busy. It also stays well inside the daemon's own
# last-resort recall budget (25 s, `_recall_budget_s`), which is the layer that
# exists to catch a true hang and which already tells the caller when it fires
# (`retrieval_mode=degraded_lexical`). Before this change the inner 1.4 s cutoff
# silently overrode that outer promise of "quality recall under load".
#
# Lowering this to improve a latency percentile means buying that percentile
# with missing answers. Per HARD-RULES RULE 6 the ordering is Correct, then
# Complete, then Repeatable, and only then Fast — so if this needs to move,
# measure what it costs in answer quality first and record the number.
CHANNEL_HANG_GUARD_SECONDS = 8.0

# How long a recall waits for a query vector from a model that has NEVER loaded
# in this process (a fresh daemon). Only that case: a model that loaded before
# is waited for unbounded, exactly as in 4.1.19 (retrieval/query_embedding.py).
#
# The recall ceiling is 3.0 s total (RECALL_CEILING_S, answer check included).
# Waiting the full hang guard here made the first recalls after a start take
# 8.06 s while the model loaded. One second gives a nearly loaded model its
# chance and leaves the rest of the ceiling for the channels that need no
# vector and for the answer check, which budgets itself against the recall's
# start and is skipped rather than overrun. What it costs: on a cold daemon the
# vector channels report ``warming`` and the recall is marked incomplete -- the
# same answer the 8 s wait gave whenever the load took longer than 8 s, and the
# embed is not cancelled, so the next recall of the question gets its vector.
COLD_QUERY_EMBED_WAIT_SECONDS = 1.0


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
        sufficiency_judge: Any | None = None,
        display_min_confidence: float = 0.20,
    ) -> None:
        self._db = db
        # 4.1.18: decides whether a recall answers its question; read by
        # run_recall at the contract boundary. None means today's behaviour.
        self._sufficiency_judge = sufficiency_judge
        self._config = config
        # M3: the kind facet's confidence threshold. RetrievalConfig (this
        # class's own ``config``) has no memory_kinds field — that lives on
        # the TOP-LEVEL SLMConfig — so the wiring layer that HAS the full
        # config passes the live value through here, rather than this class
        # reaching for a config section it was never given.
        self._display_min_confidence = display_min_confidence
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
        self._strategy = strategy or QueryStrategyClassifier(config=config)
        self._base_weights = (base_weights or ChannelWeights()).as_dict()
        self._media_weight = (base_weights or ChannelWeights()).media
        self._media_channel = None  # built on first use, only if pictures are on
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

        # Query embeddings: 512-entry cache (~1.5 MB), single-flight, and a
        # bounded wait so a loading model cannot hold a whole recall
        # (see retrieval/query_embedding.py).
        from superlocalmemory.retrieval.query_embedding import QueryEmbedder
        self._query_embedder = QueryEmbedder(lambda: self._embedder, cache_max_size=512)
        self._query_embedding_cache = self._query_embedder.cache

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
        mode: Mode = Mode.A, limit: int = CANONICAL_RECALL_LIMIT,
        *,
        extra_disabled_channels: set[str] | None = None,
        include_global: bool = False,
        include_shared: bool = False,
        window: str | tuple[str, str] | None = None,
        as_of: str | None = None,
        known_as_of: str | None = None,
        valid_at: str | None = None,
        include_unknown: bool = False,
        facets: Any = None,
    ) -> RecallResponse:
        """Full retrieval pipeline: strategy -> channels -> RRF -> rerank.

        Multi-scope: ``include_global`` / ``include_shared`` control which
        scopes participate in retrieval. Both default to False so direct
        retrieval-engine callers are private unless they explicitly opt in.

        V3.4.40 (2026-05-09): ``extra_disabled_channels`` allows callers to
        skip specific channels for a single recall (e.g. SpreadingActivation
        for the ``--fast`` CLI flag) without mutating shared config.

        ``as_of``: Optional ISO 8601 datetime string. When set, the bi-temporal
        validity filter treats facts as seen from that point in time —
        not-yet-valid and already-expired facts are demoted. Default ``None``
        leaves all existing behaviour unchanged.
        """
        from superlocalmemory.retrieval.temporal_utils import normalize_strict_boundary
        known_as_of = normalize_strict_boundary(known_as_of, "known_as_of")
        valid_at = normalize_strict_boundary(valid_at, "valid_at")
        # 4.1.20 (R5): an explicit window that cannot be read is refused here,
        # the root every door reaches, instead of quietly applying no filter.
        # Blank means "no window", exactly as before.
        from superlocalmemory.retrieval.time_filter import check_window
        window = check_window(window)
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

        # What each channel did, so the caller can tell an answer apart from
        # an outage. Owned by this call for the same reason the dropped set is:
        # a dict on the engine would have two concurrent recalls overwriting
        # each other's report.
        channel_status: dict[str, str] = {}

        # Profile shortcut (runs before channel search).
        #
        # The ablation flag is honoured here too. It was not, so an operator who
        # switched this channel off still had it searched, still had its weight
        # doubled on a hit, and read a status of "ok" or "empty" — which reports
        # their own configuration as a live channel's finding.
        _profile_disabled = "profile" in set(self._config.disabled_channels) | set(
            extra_disabled_channels or (),
        )
        if _profile_disabled:
            profile_hits = []
            channel_status["profile"] = chstat.DISABLED
        elif self._profile_channel is not None:
            try:
                profile_hits = self._profile_channel.search(
                    query, profile_id, top_k=10,
                )
                channel_status["profile"] = (
                    chstat.OK if profile_hits else chstat.EMPTY
                )
                if profile_hits:
                    strat.weights["profile"] = 2.0
            except Exception as exc:
                logger.warning("Profile channel: %s", exc)
                profile_hits = []
                channel_status["profile"] = chstat.ERROR
        else:
            profile_hits = []
            channel_status["profile"] = chstat.NOT_CONFIGURED

        # Dynamic top-k for aggregation queries
        effective_limit = 100 if strat.query_type == "aggregation" else limit

        # 3. Run channels; flags travel as parameters (no shared state). Owned by
        # this call: non-empty means this answer is incomplete, not just slow.
        dropped_channels: set[str] = set()
        # Where this recall's time went (``RecallResponse.stage_ms``); per call.
        stage_ms: dict[str, float] = {}
        _t_channels = time.monotonic()
        ch_results = self._run_channels(
            query, profile_id, strat,
            extra_disabled_channels=extra_disabled_channels,
            include_global=include_global, include_shared=include_shared,
            as_of=as_of, known_as_of=known_as_of, valid_at=valid_at,
            include_unknown=include_unknown,
            dropped_channels=dropped_channels,
            channel_status=channel_status,
            stage_ms=stage_ms,
        )
        stage_ms["channels"] = round((time.monotonic() - _t_channels) * 1000.0, 1)
        if getattr(facets, "kind", None) and getattr(self, "_kind_membership", None):
            ch_results = kind_scope.supplement(  # search inside the kind (kind_scope)
                self, ch_results, query=query, query_embedding=self._embed_query(query)[0],
                profile_id=profile_id, kind=facets.kind, stage_ms=stage_ms)
        if getattr(facets, "project", None):
            ch_results = project_search.supplement(  # search inside the project
                self, ch_results, query=query, query_embedding=self._embed_query(query)[0],
                profile_id=profile_id, project=facets.project, stage_ms=stage_ms)
        if getattr(facets, "tags", None):
            from superlocalmemory.retrieval import tag_search
            ch_results = tag_search.supplement(  # search inside the tag set
                self, ch_results, query=query, query_embedding=self._embed_query(query)[0],
                profile_id=profile_id, tags=facets.tags,
                match=getattr(facets, "tags_match", "all"), stage_ms=stage_ms)
        _em("run_channels")
        # One request may need admission before fusion and again after optional
        # bridge/scene expansion.  Cache only the IDs checked during this one
        # request; every newly expanded candidate remains a hard DB lookup.
        correction_admission = CorrectionAdmissionCache()
        if profile_hits:
            ch_results["profile"] = profile_hits
        # The profile shortcut bypasses _run_channels(), so it needs the same
        # admission before it can influence fusion or seed graph expansion.
        ch_results = admit_correction_candidates(
            ch_results, profile_id, self._db, as_of=as_of,
            known_as_of=known_as_of, valid_at=valid_at,
            include_unknown=include_unknown,
            include_global=include_global, include_shared=include_shared,
            lifecycle_cache=correction_admission,
        )
        total = sum(len(v) for v in ch_results.values())

        # 3. Single-pass RRF fusion
        ch_results = self._semantic_rank_for_unenriched(ch_results)
        if ch_results.get("media"):
            strat.weights["media"] = self._media_weight
        fused = weighted_rrf(ch_results, strat.weights, k=self._config.rrf_k)
        _em("rrf_fusion")

        # V3.3.21: Cross-channel intersection boost for multi-hop/temporal queries.
        # Problem: channels work in ISOLATION. "When did Caroline go to X?" needs
        # entity(Caroline) ∩ temporal(date). RRF averages scores but doesn't enforce
        # the intersection constraint. Fix: boost facts that appear in 2+ signal-type
        # channels (entity+temporal, entity+semantic, temporal+semantic).
        if strat.query_type == "multi_hop" and len(ch_results) >= 2:
            fused = self._apply_cross_channel_intersection(fused, ch_results, strat)
        # What the channels themselves found, before bridges and scenes add
        # neighbours. The rerank pool keeps room for these (rerank_pool).
        found_ids = frozenset(fr.fact_id for fr in fused)

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

        # Scene expansion (v3.5.0: batch).
        #
        # This used to be skipped when more than 0.8 s of the recall had already
        # elapsed, on the reasoning that the scene signal is nice-to-have and
        # never worth delaying a response. The reasoning was wrong, because the
        # stage does not merely decorate the answer — it appends candidates that
        # can outrank what fusion produced. Gating it on a stopwatch therefore
        # made the ANSWER depend on how busy the machine was, and 0.8 s sits on
        # top of recall's own median (~1,044 ms on the 0.95 GB archive), so it
        # was not a rare safety valve: measured over two runs of 60 queries, the
        # two clock gates flipped their decision on 22 of 60, and of the 19
        # queries whose answer changed, every one had a flipped gate.
        #
        # Removing both gates moved rank-1 disagreement between two runs from
        # 20.0% to 3.3% and top-10 from 31.7% to 10.0%, for about 100-180 ms of
        # p95 (1,191-1,230 ms -> 1,256-1,375 ms, ceiling 2,000 ms). Per
        # HARD-RULES RULE 6 that is the correct direction: Correct, Complete,
        # Repeatable, and only then Fast.
        #
        # So do not reintroduce a time condition here. If this stage ever needs
        # bounding, bound it by DATA — a candidate count, a scene cap — so the
        # same input always takes the same path.
        if fused:
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
        # The 0.9 s clock gate that used to guard this stage is gone for the
        # reason given above the scene expansion, and it mattered more here:
        # this stage re-scores every fused candidate and then re-sorts them, so
        # whether it ran decided the top answer outright rather than adding to
        # it. Bound by data if it ever needs bounding, never by elapsed time.
        if self._entity is None:
            channel_status["entity_graph"] = chstat.NOT_CONFIGURED
        elif "entity_graph" in set(self._config.disabled_channels) | set(
            extra_disabled_channels or (),
        ):
            # Both the configured set AND the per-call one, the same way the
            # profile channel resolves it. Reading only the configured set meant
            # a caller that switched this channel off for one recall still had
            # it run, and still saw it reported as though it were the caller's
            # own setting that was being honoured.
            channel_status["entity_graph"] = chstat.DISABLED
        elif not fused:
            # It did not run, and saying "found nothing" would be a different
            # claim. This channel re-scores other channels' candidates rather
            # than producing its own, so with nothing fused there is nothing for
            # it to do — and if the reason nothing fused is that the other five
            # failed, reporting this one as having searched and come back empty
            # hides that.
            channel_status["entity_graph"] = chstat.NO_CANDIDATES
        else:
            # One chain, evaluated once. Repeating the three conditions to guard
            # the work separately is how a status starts describing a decision
            # the code no longer makes.
            try:
                candidate_ids = [fr.fact_id for fr in fused[:100]]
                # None: another thread builds this scope's graph, none usable cached.
                eg_scores = entity_graph_warmup.score_candidates_unless_warming(
                    self._entity, query, candidate_ids, profile_id,
                    include_global=include_global, include_shared=include_shared)
                if eg_scores is None:  # incomplete, never "found nothing"
                    dropped_channels.add("entity_graph")
                channel_status["entity_graph"] = (chstat.WARMING if eg_scores is None
                                                  else chstat.OK if eg_scores else chstat.EMPTY)
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
                    fused = sorted(boosted, key=lambda r: (-r.fused_score, r.fact_id))
            except Exception as exc:
                logger.warning("Entity graph signal enhancement: %s", exc)
                channel_status["entity_graph"] = chstat.ERROR

        # Brain Core S402: bridge and scene expansion append candidates after
        # the channel boundary. Reapply the same hard correction-admission rule
        # immediately before any candidate can be materialized or reranked.
        fused = admit_correction_fusion_results(
            fused, profile_id, self._db, as_of=as_of,
            known_as_of=known_as_of, valid_at=valid_at,
            include_unknown=include_unknown,
            include_global=include_global, include_shared=include_shared,
            lifecycle_cache=correction_admission,
        )

        _em("expand+entity_enh")

        # T-window: prune candidates to the requested event-time range.
        # Event times are fetched for the bounded candidate set only (indexed),
        # then in-range facts are kept — before fact load, so out-of-window facts
        # are never materialized. T3: when the caller passes no explicit window,
        # infer one from natural-language scope in the query ("last week").
        # Safety: an EXPLICIT window is authoritative (honoured even if it empties
        # the set — the user asked for that scope), but an INFERRED window is
        # additive and never makes recall worse — if nothing inside it has
        # primary evidence, fall back to the unwindowed set (time_window.py).
        _explicit_window = window is not None
        _window = window if _explicit_window else infer_window_from_query(query)
        if _window is not None and fused:
            bounds = parse_window(_window)
            if bounds is not None:
                etimes = self._db.get_fact_event_times(
                    [fr.fact_id for fr in fused], profile_id,
                )
                fused = windowed_candidates(
                    fused, lambda fid: in_window(etimes.get(fid), bounds),
                    explicit=_explicit_window,
                    min_semantic=getattr(self._config, "min_semantic_evidence", 0.60),
                    min_media=self._media_floor(),
                )
                _em("time_window")

        # Facets (saved_by / about / kind): explicit, so hard filters -
        # honoured even when they leave nothing (retrieval/facets.py).
        # ``project`` filters but falls back to unfiltered, and says so, when
        # nothing found was saved under it; ``prefer_project`` only marks
        # memories for the bounded boost below (retrieval/project_scope.py).
        preferred: frozenset[str] = frozenset()
        project_scope = None
        tag_scope = None
        if facets is not None and not getattr(facets, "empty", True):
            from superlocalmemory.retrieval.project_scope import narrow

            scoped = narrow(
                self._db, [fr.fact_id for fr in fused], profile_id, facets,
                resolver=getattr(self._entity, "_resolver", None),
                # M3: the configured threshold, not kind_fields' 0.20 default —
                # so a kind facet agrees with what the recall response itself
                # displays for the SAME row.
                display_min_confidence=self._display_min_confidence,
            )
            keep = set(scoped.kept)
            fused = [fr for fr in fused if fr.fact_id in keep]
            preferred, project_scope = scoped.preferred, scoped.report
            if getattr(facets, "tags", None):
                # ``scoped.kept`` already has every facet (incl. tags) ANDed
                # in (retrieval.facets.matching_fact_ids), so its count IS
                # the tag-filtered match count — nothing to recompute here.
                from superlocalmemory.retrieval.tag_scope import build_report
                tag_scope = build_report(self._db, profile_id, facets, len(scoped.kept))
            _em("facets")

        # 4. Load facts for rerank pool
        # Search results and added neighbours each get ``pool`` slots, so a
        # large scene cannot push a found memory out before the cross-encoder
        # reads it (retrieval/rerank_pool.py).
        pool = max(effective_limit * 3, 30)
        top = rerank_pool(fused, found=found_ids, size=pool)
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

        # v3.6.6: Evidence floor — gate on per-channel scores (NOT fused/RRF score).
        # Nonsense queries fuse at 0.75-0.78 because RRF is rank-derived and
        # uncalibrated. The discriminator is EARNED CHANNEL EVIDENCE:
        #   semantic >= min_semantic_evidence (0.60) OR bm25 > 0
        #   OR entity_graph > 0 OR temporal > 0 OR fact is pinned.
        # spreading_activation and hopfield do NOT count — they are associative
        # amplifiers that fabricated the nonsense results in calibration tests.
        # Kill-switch: SLM_RECALL_NO_FLOOR=1 bypasses the floor.
        # Runs BEFORE the cross-encoder so the CE batch contains only
        # evidence-qualified candidates. The floor gates on channel_scores
        # (semantic, bm25, entity_graph, temporal) which are assigned during
        # channel execution and are not affected by CE reranking. Moving the
        # floor here does not change which queries abstain; it reduces the CE
        # batch from ~180 candidates to the qualified subset (~30–60).
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
            top = self._apply_evidence_floor(
                top, facts, min_sem, self._media_floor())

        # 5. Cross-encoder rerank (optional, on the evidence-qualified pool)
        # Bug 4 fix: reduced alpha for multi-hop/temporal to preserve diversity
        # The reranker decides its own readiness. ``rerank_with_status`` is
        # non-blocking: with no model loaded it returns the fused order as
        # "fallback_not_ready" and starts a background re-warm, so a cold
        # start never blocks a recall.
        #
        # 4.1.18: this used to gate on the private ``_worker_ready`` flag.
        # ``_kill_worker`` clears it on every worker death — the routine
        # 500-request recycle, the 30-minute idle kill, a crash — and the
        # re-warm lives inside the call this gate then refused to make, so
        # reranking stayed off until the daemon restarted. The live daemon
        # ran unranked for seven days that way. A reranker type without that
        # private attribute (RemoteReranker) was never called at all.
        reranker_ready = self._reranker is not None
        reranker_applied = False
        reranker_status = (
            "fallback_not_ready" if self._reranker is not None
            else "not_configured"
        )
        _t_rerank = time.monotonic()
        if reranker_ready and facts:
            ce_alpha = 0.5 if strat.query_type in ("multi_hop", "temporal") else 0.75
            top, reranker_applied, reranker_status = self._apply_reranker(
                query, top, facts, alpha=ce_alpha,
            )
        elif reranker_ready:
            reranker_status = "no_candidates"
        stage_ms["rerank"] = round((time.monotonic() - _t_rerank) * 1000.0, 1)
        _em(f"rerank(ready={reranker_ready})")

        if preferred:
            # #150: before the cut, so a same-project memory just below it can
            # be chosen; bounded by project_scope.BOOST, so only one of similar
            # relevance ever passes another memory.
            from superlocalmemory.retrieval.project_scope import boost_order
            top = boost_order(top, preferred)

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
        # #150: the ORDER of these results is preferred once, at the end of
        # recall (core.recall_pipeline -> project_scope.prefer_in_final_order),
        # after learned ranking has rewritten every score. Here it only chose
        # which candidates made the cut (boost_order above).
        results = self._build_results(final_top, facts, strat,
                                      reranked=reranker_applied)
        ms = (time.monotonic() - t0) * 1000.0
        stage_ms["retrieval_total"] = round(ms, 1)
        no_match = floor_enabled and len(results) == 0
        return RecallResponse(
            query=query, mode=mode, results=results,
            query_type=strat.query_type, channel_weights=strat.weights,
            total_candidates=total, retrieval_time_ms=ms,
            no_confident_match=no_match,
            reranker_applied=reranker_applied,
            reranker_status=reranker_status,
            # Q2b: thematic context when the top results cluster in one
            # community. Precomputed summary lookup only — no per-query LLM.
            community_context=self._community_context(results, profile_id),
            incomplete_channels=tuple(sorted(dropped_channels)),
            channel_status=dict(channel_status),
            stage_ms=dict(stage_ms),
            project_scope=project_scope,
            tag_scope=tag_scope,
        )

    # -- Community context (Wave Q2b) --------------------------------------

    def _community_context(
        self, results: list[Any], profile_id: str, top_k: int = 8,
    ) -> dict | None:
        """Attach the precomputed community summary the top results fall into.

        Fail-open: any error returns None (recall is never affected). The
        matching/gating/sample-bounding logic lives in
        ``retrieval.community_context`` (Q9, 2026-10-06: extracted rather
        than grown in place — this module is already over the 800-line cap).
        """
        if not results or not getattr(
            self._config, "enable_community_context", True,
        ):
            return None
        try:
            from superlocalmemory.retrieval.community_context import (
                build_community_context,
            )

            return build_community_context(
                self._db, results, profile_id, top_k=top_k,
            )
        except Exception as exc:
            logger.debug("community context skipped (fail-open): %s", exc)
            return None

    # -- Evidence floor (v3.6.6) -------------------------------------------

    def _media_floor(self) -> float:
        """Picture evidence floor: the live paired plan's, else the configured one."""
        default = getattr(self._config, "media_min_score", 0.30)
        channel = getattr(self, "_media_channel", None)
        return default if channel is None else channel.min_score(default)

    @staticmethod
    def _apply_evidence_floor(
        final_top: list[FusionResult],
        facts: dict[str, AtomicFact],
        min_semantic: float,
        min_media: float = 0.30,
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
            if has_primary_evidence(fr.channel_scores, min_semantic, min_media):
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
        boosted.sort(key=lambda r: (-r.fused_score, r.fact_id))
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
    ) -> list:
        """Keep strong lexical and structure evidence visible in the result cap.

        A semantic channel with a larger weight can fill a small result limit
        even when BM25 has an exact, high-signal hit. That broke the
        ``queryable now`` ingestion contract: a freshly inserted FTS row could
        exist durably but remain invisible to immediate recall. Reserve one
        capped slot for a strong BM25 hit and two for a structure channel when
        such candidates exist, without returning more than ``effective_limit``.
        """
        channel_minimums = (
            ("bm25", 1, 0.0),
            ("entity_graph", 2, 0.0),
        )
        top_ids = {fr.fact_id for fr in top}

        promoted = []
        for ch_name, minimum, score_floor in channel_minimums:
            ch_items = ch_results.get(ch_name, [])
            if not ch_items:
                continue

            eligible_ids = {
                fid
                for fid, score in ch_items
                if (
                    float(score) > score_floor
                    if ch_name == "bm25"
                    else float(score) >= score_floor
                )
            }
            if not eligible_ids:
                continue

            present = sum(1 for fid in eligible_ids if fid in top_ids)
            if present >= minimum:
                continue

            needed = minimum - present
            for fr in fused:
                if fr.fact_id in eligible_ids and fr.fact_id not in top_ids:
                    promoted.append(fr)
                    top_ids.add(fr.fact_id)
                    needed -= 1
                    if needed <= 0:
                        break

        if not promoted:
            return top

        selected = promoted[:effective_limit]
        result = list(top[:effective_limit])
        free_slots = max(0, effective_limit - len(result))
        result.extend(selected[:free_slots])
        remaining = selected[free_slots:]
        if remaining:
            keep = max(0, effective_limit - len(remaining))
            result = result[:keep] + remaining
        return result[:effective_limit]

    # -- Channel execution --------------------------------------------------

    def _embed_query(self, query: str) -> tuple[list[float] | None, str | None]:
        """(vector, status). Bounded by ``COLD_QUERY_EMBED_WAIT_SECONDS`` ONLY
        while the embedder has never been ready (status WARMING: vector
        channels not run); one that has loaded before is waited for exactly
        as in 4.1.19. See retrieval/query_embedding.py.
        """
        return self._query_embedder.embed(query, COLD_QUERY_EMBED_WAIT_SECONDS)

    def _semantic_rank_for_unenriched(
        self, ch_results: dict[str, list[tuple[str, float]]],
    ) -> dict[str, list[tuple[str, float]]]:
        """Give a candidate whose vector does not exist yet a fair semantic rank.

        Fusion here is rank-based, so a fact the semantic channel did not return
        forfeits that channel's entire contribution — the most heavily weighted
        one. When the reason for that absence is simply that the vector has not
        been computed yet, the absence describes the ingest pipeline and says
        nothing about the fact. Left alone, a memory written seconds ago is the
        hardest thing in the store to find, which is the worst possible failure
        for this product.

        Such candidates are placed at the MEDIAN of the semantic ranking, never
        near the top: enough to compete on their other evidence, not enough to
        win on freshness alone. A candidate that HAS a vector and still was not
        returned is left exactly as it is — that absence is real evidence of
        irrelevance, and the two must not be confused.

        Returns a new mapping; the input is not modified.
        """
        sem = ch_results.get("semantic") or []
        if not sem:
            return ch_results
        if not getattr(self._config, "write_recency_floor_enabled", True):
            return ch_results
        if os.environ.get("SLM_WRITE_RECENCY_NO_FLOOR", "0") == "1":
            return ch_results

        have = {fid for fid, _ in sem}
        elsewhere = {
            fid
            for name, rows in ch_results.items()
            if name != "semantic"
            for fid, _ in rows
        }
        candidates = sorted(elsewhere - have)
        if not candidates:
            return ch_results

        from datetime import UTC, datetime, timedelta

        minutes = float(getattr(self._config, "write_recency_floor_minutes", 60.0))
        cutoff = (datetime.now(UTC) - timedelta(minutes=minutes)).isoformat()
        placeholders = ",".join("?" for _ in candidates)
        try:
            # A missing embedding_metadata row means no vector projection exists,
            # which is what makes the semantic channel's silence uninformative.
            unenriched = [
                dict(r)["fact_id"]
                for r in self._db.execute(
                    f"SELECT af.fact_id FROM atomic_facts AS af "
                    f"LEFT JOIN embedding_metadata AS em ON em.fact_id = af.fact_id "
                    f"WHERE af.fact_id IN ({placeholders}) "
                    f"  AND em.fact_id IS NULL "
                    f"  AND af.created_at >= ?",
                    (*candidates, cutoff),
                )
            ]
        except (NameError, AttributeError, TypeError):
            # These mean this code is wrong, not that the data is unusual. A bare
            # `except Exception` here hid a missing import and left the whole
            # feature silently inert while every test still passed.
            raise
        except Exception as exc:
            # A store without this table, or a locked database: ranking must still
            # return. Logged at warning, because "silently did nothing" is the
            # failure mode this task exists to fix.
            logger.warning("recent-unenriched admission skipped: %s: %s",
                           type(exc).__name__, exc)
            return ch_results
        if not unenriched:
            return ch_results

        scores = sorted(s for _, s in sem)
        mid = len(scores) // 2
        median = (
            scores[mid] if len(scores) % 2 == 1
            else (scores[mid - 1] + scores[mid]) / 2.0
        )
        insert_at = len(sem) // 2
        merged = list(sem[:insert_at]) + [(fid, median) for fid in unenriched] + list(sem[insert_at:])
        logger.debug(
            "admitted %d recent un-enriched candidate(s) at semantic rank %d of %d",
            len(unenriched), insert_at + 1, len(merged),
        )
        return {**ch_results, "semantic": merged}

    def _run_channels(
        self,
        query: str,
        profile_id: str,
        strat: QueryStrategy,
        *,
        extra_disabled_channels: set[str] | None = None,
        include_global: bool = False,
        include_shared: bool = False,
        as_of: str | None = None,
        known_as_of: str | None = None,
        valid_at: str | None = None,
        include_unknown: bool = False,
        dropped_channels: set[str] | None = None,
        channel_status: dict[str, str] | None = None,
        stage_ms: dict[str, float] | None = None,
    ) -> dict[str, list[tuple[str, float]]]:
        """Run active retrieval channels.

        v3.4.53: channels run in PARALLEL via ThreadPoolExecutor. Industry
        standard (EverMemOS, szl-recall, ContentPilot 2026): all channels
        are independent after embedding; running them serially wastes time
        equal to the sum of all producer latencies. When multiple producers are
        enabled and healthy, parallel dispatch generally bounds the producer
        phase by the slowest submitted producer, plus serial embedding and
        result-collection overhead.

        ``dropped_channels``, when given, receives the name of every channel
        abandoned at ``CHANNEL_HANG_GUARD_SECONDS``. Those channels contributed
        nothing, so the caller needs to know the answer is incomplete rather
        than merely late. It is a caller-owned set passed down per recall and
        deliberately not an attribute of self — two concurrent recalls sharing
        one would report each other's losses (the v3.4.64 race).

        ``channel_status``, likewise caller-owned, receives one entry per
        channel saying what became of it. Every channel gets exactly one:
        those that cannot run are recorded before dispatch with the reason, and
        every dispatched channel is recorded by the collection loop below —
        which iterates the futures, so it cannot skip one.
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
        emb_wait: str | None = None  # WARMING: model not ready, recall stopped waiting
        needs_embedding = (
            (self._semantic is not None and "semantic" not in disabled)
            or (self._hopfield is not None and "hopfield" not in disabled)
            or (self._spreading_activation is not None and "spreading_activation" not in disabled)
        )
        if needs_embedding:
            _t_embed = _time_e.monotonic()
            try:
                q_emb, emb_wait = self._embed_query(query)
                if q_emb is None and emb_wait is None:
                    logger.warning(
                        "Query embedding returned None — semantic, hopfield, "
                        "spreading_activation channels will be skipped this recall"
                    )
            except Exception as exc:
                logger.warning("Query embedding failed: %s", exc)
            if stage_ms is not None:
                stage_ms["query_embedding"] = round(
                    (_time_e.monotonic() - _t_embed) * 1000.0, 1)

        media_vec = None  # pictures: one bounded embed, before dispatch (media_channel)
        if "media" not in disabled:
            from superlocalmemory.retrieval import media_channel
            self._media_channel = self._media_channel or media_channel.for_engine(
                self._db, text_query_vector=lambda q: self._embed_query(q)[0])
            media_vec, _media_state = self._media_channel.prepare(query, profile_id)
            if _media_state and channel_status is not None:
                channel_status["media"] = _media_state

        # Why a channel will not run, recorded BEFORE dispatch. An embedding
        # failure silently takes three of the five channels down together, and
        # the answer never said so: it looked exactly like a store with nothing
        # relevant in it. Configuration and ablation are recorded too, so an
        # operator reading a list of absent channels can tell their own choices
        # apart from a fault.
        if channel_status is not None:
            for _name, _obj, _needs_emb in (
                ("semantic", self._semantic, True),
                ("bm25", self._bm25, False),
                ("temporal", self._temporal, False),
                ("hopfield", self._hopfield, True),
                ("spreading_activation", self._spreading_activation, True),
            ):
                if _obj is None:
                    channel_status[_name] = chstat.NOT_CONFIGURED
                elif _name in disabled:
                    channel_status[_name] = chstat.DISABLED
                elif _needs_emb and q_emb is None:
                    channel_status[_name] = emb_wait or chstat.NO_EMBEDDING
        if emb_wait and dropped_channels is not None and channel_status is not None:
            # Not run because the vector was late: the answer is incomplete.
            dropped_channels.update(
                n for n, st in channel_status.items() if st == emb_wait)

        # v3.4.53: collect channel callables and run in parallel.
        # Each channel is a standalone search — no shared mutable state,
        # no ordering dependencies. SQLite WAL mode permits concurrent reads.
        futures: dict[str, concurrent.futures.Future] = {}

        def _safe_channel(name: str, fn, *args):
            """Run a single channel, returning (name, result_or_None, status).

            Returning the status alongside the result is what separates "found
            nothing" from "raised": both used to come back as ``None``.
            """
            _cs = _time_e.monotonic() if _et else 0.0
            try:
                res = fn(*args)
                if _et:
                    logger.warning("[RECALL-TIMING]     channel.%-16s %.0f ms",
                                   name, (_time_e.monotonic() - _cs) * 1000.0)
                if res:
                    return (name, res, chstat.OK)
                return (name, None, chstat.EMPTY)
            except Exception as exc:
                logger.warning("%s channel: %s", name, exc)
                return (name, None, chstat.ERROR)

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
                    query_type=strat.query_type,
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

        if media_vec is not None:
            futures["media"] = executor.submit(
                _safe_channel, "media", self._media_channel.search,
                media_vec, profile_id, self._config.semantic_top_k,
            )

        # One shared limit keeps parallel dispatch genuinely bounded.  A
        # per-future timeout here would serialise the wait and turn five slow
        # channels into five seconds of UI latency.
        done, pending = concurrent.futures.wait(
            futures.values(), timeout=CHANNEL_HANG_GUARD_SECONDS,
        )
        for name, fut in futures.items():
            if fut in pending:
                # Not a latency notice: this answer is missing whatever this
                # channel alone could see, so it is logged at the level that
                # says so and recorded for the caller.
                logger.error(
                    "Channel %s did not finish within %.1fs; this recall is "
                    "answering without it",
                    name, CHANNEL_HANG_GUARD_SECONDS,
                )
                if dropped_channels is not None:
                    dropped_channels.add(name)
                # Same branch as the dropped set on purpose: two writes in one
                # place cannot disagree about which channels timed out.
                if channel_status is not None:
                    channel_status[name] = chstat.TIMEOUT
                fut.cancel()  # no-op if already running; prevents queued jobs from starting
                continue
            try:
                ch_name, result, status = fut.result()
                if channel_status is not None:
                    channel_status[ch_name] = status
                if result:
                    out[ch_name] = result
            except Exception as exc:
                logger.warning("Channel %s failed: %s", name, exc)
                if channel_status is not None:
                    channel_status[name] = chstat.ERROR

        # Apply registered post-retrieval filters (forgetting filter, etc.).
        # Pass as_of in context dict when set so the bi-temporal validity filter
        # can perform point-in-time demotion. None context preserves the existing
        # behaviour for all callers that don't use time-travel recall.
        _filter_context = {"as_of": as_of} if as_of is not None else None
        if hasattr(self, '_registry') and self._registry._filters:
            for fn in self._registry._filters:
                try:
                    out = fn(out, profile_id, _filter_context)
                except Exception as exc:
                    logger.warning("Post-retrieval filter failed: %s", exc)

        return out

    def close(self, *, wait: bool = False) -> None:
        """Release owned channel workers without blocking daemon shutdown.

        Active channel calls have their own response deadline.  Waiting here
        can still deadlock shutdown when an extension ignores that deadline,
        so the daemon uses the executor's non-blocking cancellation path.
        """
        with self._close_lock:
            if self._closed:
                return
            self._closed = True
        self._channel_executor.shutdown(wait=wait, cancel_futures=True)
        self._query_embedder.close()

    def worker_threads(self) -> list:
        """The pool threads this engine owns, for a bounded join on close.

        Read BEFORE ``close``: closing detaches the query-embed pool.
        """
        from superlocalmemory.core.thread_join import executor_threads

        return (
            executor_threads(getattr(self, "_channel_executor", None))
            + self._query_embedder.worker_threads()
        )

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
        # Bug 2 fix: score ALL candidates, not just top_k. v3.8.5: verified on
        # the real DB that bounding the CE to the top-N fusion candidates both
        # (a) gave NO latency win (the cross-encoder batches all pairs in one
        # forward pass, so 60 vs 184 pairs is within noise) and (b) CHANGED the
        # top-5 on 4/8 queries — the CE legitimately promotes items ranked below
        # the fusion top-N into the answer. So exhaustive reranking stays: it is
        # a quality feature, not the latency bottleneck.
        from superlocalmemory.retrieval import media_rerank
        neutral = media_rerank.neutral_ids(fused, fact_map)  # pictures without words keep their rank
        candidates = [
            (fact_map[fr.fact_id], fr.fused_score)
            for fr in fused if fr.fact_id in fact_map and fr.fact_id not in neutral
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

        # The worker can report applied=True while returning scores=null — the
        # subprocess answers, so the call "succeeded", but there is nothing to
        # score with.  Iterating None here raised TypeError from OUTSIDE the
        # try/except above (which only wraps the rerank call itself), so the
        # error escaped into the recall path rather than degrading to the fused
        # ordering.  Fail soft: reranking is a quality improvement on top of a
        # correct result set, never a correctness requirement.
        # `not scored` covers None AND an empty sequence. An empty list is the
        # same defect wearing different clothes: the worker says applied=True but
        # supplied nothing to rank with. Guarding only None would let [] through
        # to build an empty score_map, and every candidate would then be scored
        # against a degenerate min/max — silently shrinking the fused component
        # by (1 - alpha) while still reporting the rerank as applied.
        if not scored:
            logger.warning(
                "Cross-encoder worker reported applied=True with %s scores; "
                "falling back to fused ranking for this query.",
                "null" if scored is None else "empty",
            )
            return fused, False, "worker_null_scores"

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
        updated.sort(key=lambda r: (-r.fused_score, r.fact_id))
        return media_rerank.restore_ranks(fused, updated, neutral), True, "applied"

    # -- Agentic adapter -----------------------------------

    def recall_facts(
        self, query: str, profile_id: str,
        top_k: int = 20, skip_agentic: bool = True,
        facets: Any = None,
    ) -> list[tuple[AtomicFact, float]]:
        """Simplified recall returning (fact, score) tuples.

        Used by AgenticRetriever for every round (including round 2's
        re-retrieval). ``skip_agentic`` is always True here to prevent
        infinite recursion. ``facets`` (4.1.19 L2-08): forwarded to the real
        ``recall()`` so the caller's hard filter (project / agent / about /
        kind) still applies when ``fast=False`` triggers the internal
        verification round — before this fix it was silently dropped here,
        so a ``fast=False`` recall could return results outside a facet the
        caller explicitly asked for.
        """
        response = self.recall(query, profile_id, limit=top_k, facets=facets)
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
        strat: QueryStrategy, *, reranked: bool = False,
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
                # Channel name breaks a tie, so the evidence string a caller
                # sees is the same on two runs when two channels agree on rank.
                for ch, rk in sorted(fr.channel_ranks.items(), key=lambda x: (x[1], x[0]))
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
            age_known = False
            if fact.created_at:
                try:
                    created = datetime.fromisoformat(fact.created_at.replace("Z", "+00:00"))
                    age_days = max(0.0, (now - created).total_seconds() / 86400.0)
                    age_known = True
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

            # Query-type-conditioned recency amplifier.
            # Applied only to "recency" and "temporal" queries; factual, entity,
            # and all other types receive a factor of exactly 1.0 (no change).
            # The amplitude scalar is read from RetrievalConfig so it can be tuned
            # or zeroed at runtime.  strength=0.0 is a strict no-op — the if-guard
            # ensures the previous ranking is reproduced byte-for-byte.
            #
            # recency  — 7-day half-life, 1.5× maximum (present-activity queries)
            # temporal — 30-day half-life, 1.2× maximum (past-event queries)
            #
            # Hook for the follow-on embedding-lag adjustment (task 2.6): that
            # adjustment also multiplies boosted_score and belongs immediately after
            # this block, conditioned on channel_scores["semantic"] == 0.0 AND
            # age_days < 1.0. Add it as an independent if-block here so the two
            # factors compose cleanly without restructuring what is above or below.
            _prior_strength = getattr(self._config, "recency_prior_strength", 0.5)
            # age_known matters here: the fallback above leaves age_days at 0.0
            # when a fact carries no usable timestamp, which reads as "written
            # moments ago" and would hand an undated fact the largest possible
            # boost for being new. Not knowing when something was written is not
            # evidence that it is fresh.
            if (_prior_strength > 0.0 and age_known
                    and strat.query_type in ("recency", "temporal")):
                _half_life = 7.0 if strat.query_type == "recency" else 30.0
                # Both query types use max_amp=1.5 so the decay is visible.
                # With max_amp=1.2 and half_life=30, the raw value at age 0d
                # is 1.5 and at age 30d is 1.25 — both clamp to 1.2. The prior
                # was inert over the first ~39 days, which is the range it was
                # built to discriminate. Raising the cap to 1.5 lets the
                # formula vary from 1.5 (fresh) through 1.25 (30d) toward 1.0
                # (old). This changes ranking: facts from 2 days ago and 30
                # days ago now receive different boosts. The change is a
                # correction to a clamp that made the prior inert, not a
                # measured gain.
                _max_amp = 1.5
                _cond_boost = 1.0 + _prior_strength * math.exp(
                    -(math.log(2) / _half_life) * age_days
                )
                _cond_boost = min(_cond_boost, _max_amp)
                boosted_score = boosted_score * _cond_boost

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
                rerank_score=fr.fused_score if reranked else None,
            ))
        # ranking_score incorporates every modifier computed in this loop
        # (Ebbinghaus decay, quality, trust, and the query-type-conditioned
        # recency amplifier). Sort here so RecallResponse.results[0] is
        # always the highest-ranked fact — callers that rely on the returned
        # order get the amplified ranking, not the pre-amplifier fused order.
        # Tie-break on fact_id keeps two runs over an unchanged store stable.
        results.sort(key=lambda r: (-(r.ranking_score or 0.0), r.fact.fact_id))
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
    "media",  # only scaled when a recall actually carries it (below)
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
        for ch in (k for k in _CHANNEL_KEYS if k != "media" or k in original_cs):
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
