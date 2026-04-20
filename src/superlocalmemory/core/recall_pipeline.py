# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Recall pipeline — extracted free functions for MemoryEngine.recall().

Direction: engine.py imports this module. This module NEVER imports engine.py.

Part of Qualixar | Author: Varun Pratap Bhardwaj
"""

from __future__ import annotations

import hashlib
import hmac
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from superlocalmemory.core.config import SLMConfig
    from superlocalmemory.core.hooks import HookRegistry
    from superlocalmemory.storage.database import DatabaseManager

from superlocalmemory.core.security_primitives import ensure_install_token
from superlocalmemory.storage.models import Mode, RecallResponse

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LLD-00 §3 — HMAC fact-id markers (P0.4, SEC-C-01 fix)
# ---------------------------------------------------------------------------
#
# Every fact surfaced in a recall response is tagged with
#   slm:fact:<fact_id>:<hmac8>
# where hmac8 is the first 8 hex chars of HMAC-SHA256(install_token, fact_id).
#
# post_tool_outcome_hook (LLD-09) scans only for this prefix and validates
# the HMAC. Unverified markers are ignored — this closes the tool-output
# injection attack where attacker-controlled output could forge engagement
# signals by spelling a known fact_id.

_HMAC_MARKER_PREFIX = "slm:fact:"
_HMAC_LEN = 8


def _emit_marker(fact_id: str) -> str:
    """Tag ``fact_id`` with its HMAC so downstream hooks can validate.

    Deterministic per install: a given (install_token, fact_id) pair always
    produces the same marker. Token rotation invalidates old markers.
    """
    token = ensure_install_token()
    digest = hmac.new(
        token.encode("utf-8"), fact_id.encode("utf-8"), hashlib.sha256
    ).hexdigest()[:_HMAC_LEN]
    return f"{_HMAC_MARKER_PREFIX}{fact_id}:{digest}"


def _validate_marker(marker: str) -> str | None:
    """Return ``fact_id`` if ``marker`` is a valid HMAC marker, else None.

    Uses constant-time compare. Never raises.
    """
    if not isinstance(marker, str) or not marker.startswith(_HMAC_MARKER_PREFIX):
        return None
    rest = marker[len(_HMAC_MARKER_PREFIX):]
    fact_id, sep, presented = rest.rpartition(":")
    if not sep or not fact_id or len(presented) != _HMAC_LEN:
        return None
    try:
        token = ensure_install_token()
    except Exception:  # pragma: no cover — install-token I/O failure
        return None
    expected = hmac.new(
        token.encode("utf-8"), fact_id.encode("utf-8"), hashlib.sha256
    ).hexdigest()[:_HMAC_LEN]
    if hmac.compare_digest(presented, expected):
        return fact_id
    return None


def _apply_markers_to_response(response: RecallResponse) -> None:
    """Populate ``result.marker`` on every result in ``response``, in place.

    Called as the last step of :func:`run_recall` before returning. Empty
    responses pass through untouched.

    # L-P-06: audit flagged ``dataclasses.replace`` as a cheaper path.
    # Verified: ``RecallResult`` is NOT frozen, so the direct in-place
    # attribute assignment below is the O(1) mutation path — no dataclass
    # reconstruction happens. ``replace`` would ALLOCATE a fresh instance
    # per result (strictly slower). Keep the in-place mutation.
    """
    for r in response.results:
        r.marker = _emit_marker(r.fact.fact_id)


# ---------------------------------------------------------------------------
# Stage 8 SB-1 — feed shadow_router from recall-settled signals.
#
# LLD-10 Track A.3 needs live-recall A/B observations to feed ShadowTest
# (pre-promotion) and ModelRollback (post-promotion). The ndcg_at_10
# signal materialises when ``EngagementRewardModel.finalize_outcome``
# settles a row — that is the natural call site for this helper.
#
# This is a THIN wrapper over ``core.shadow_router.get_shadow_router``
# so the finalize-outcome path does not need to import shadow_router
# directly. Fail-soft on every error — recall pipeline integrity comes
# first.
# ---------------------------------------------------------------------------


def feed_recall_settled(
    *,
    memory_db: str,
    learning_db: str,
    profile_id: str,
    query_id: str,
    ndcg_at_10: float,
) -> None:
    """Route a settled recall's NDCG@10 into the shadow router.

    The arm is recomputed from ``query_id`` so callers don't need to
    persist arm assignment anywhere — the router's determinism
    guarantees the same arm decision at settle-time that was used at
    recall-time.

    Called from ``EngagementRewardModel.finalize_outcome`` (LLD-08 §4.2)
    after the reward row is committed. Cheap on the hot path: one
    singleton-cache read + one paired-list append.
    """
    try:
        from superlocalmemory.core import shadow_router as _sr
        router = _sr.get_shadow_router(
            memory_db=memory_db,
            learning_db=learning_db,
            profile_id=profile_id,
        )
        arm = router.route_query(query_id)
        router.on_recall_settled(
            query_id=query_id, arm=arm, ndcg_at_10=float(ndcg_at_10),
        )
    except Exception as exc:  # pragma: no cover — defence in depth
        logger.debug("feed_recall_settled error: %s", exc)


# ---------------------------------------------------------------------------
# V3.3.16: Module-level singletons for recall hot-path objects.
# Prevents creating new BehavioralTracker / ForgettingScheduler per recall
# (304 recalls = 304 objects that fragment pymalloc arenas → 25GB).
# ---------------------------------------------------------------------------

_behavioral_tracker_cache: dict[int, object] = {}
_forgetting_scheduler_cache: dict[int, object] = {}


def _get_behavioral_tracker(db: Any) -> Any:
    """Get or create a cached BehavioralTracker for this DB instance."""
    key = id(db)
    if key not in _behavioral_tracker_cache:
        from superlocalmemory.learning.behavioral import BehavioralTracker
        _behavioral_tracker_cache[key] = BehavioralTracker(db)
    return _behavioral_tracker_cache[key]


def _get_forgetting_scheduler(db: Any, config: Any) -> Any:
    """Get or create a cached ForgettingScheduler for this DB instance."""
    key = id(db)
    if key not in _forgetting_scheduler_cache:
        from superlocalmemory.learning.forgetting_scheduler import ForgettingScheduler
        from superlocalmemory.math.ebbinghaus import EbbinghausCurve
        ebbinghaus = EbbinghausCurve(config.forgetting)
        _forgetting_scheduler_cache[key] = ForgettingScheduler(db, ebbinghaus, config.forgetting)
    return _forgetting_scheduler_cache[key]


# ---------------------------------------------------------------------------
# S8-ARC-04 (v3.4.22): unified ranking entry point.
# ---------------------------------------------------------------------------

_RANKING_MODES: frozenset[str] = frozenset({"off", "v1", "v2", "v2-ensemble"})


def _resolve_ranking_mode(env: "dict[str, str] | os._Environ[str]") -> str:
    """Map the ``SLM_RANKING`` env var to a canonical mode.

    Legacy ``SLM_V2_PIPELINE_DISABLED=1`` and ``SLM_BANDIT_DISABLED=1``
    are honoured for one-release back-compat. Explicit ``SLM_RANKING``
    wins if both are set.
    """
    raw = (env.get("SLM_RANKING", "") or "").strip().lower()
    if raw in _RANKING_MODES:
        return raw
    if (env.get("SLM_V2_PIPELINE_DISABLED", "0") or "0").strip() == "1":
        # v2 disabled → fall back to v1 adaptive only.
        return "v1"
    if (env.get("SLM_BANDIT_DISABLED", "0") or "0").strip() == "1":
        # Bandit disabled → v2 without ensemble.
        return "v2"
    return "v2-ensemble"


def apply_ranking(
    response: "RecallResponse",
    query: str,
    profile_id: str,
    query_id: str,
    *,
    config: Any = None,
    pipeline_version: str = "v2-ensemble",
) -> "RecallResponse":
    """Run the ranking pipeline at the requested version.

    Modes:
      - ``off``: identity — no ranking passes run at all.
      - ``v1``: v3.1 Active-Memory adaptive rerank only.
      - ``v2``: v1 + v3.4.22 lambdarank rerank + signal enqueue.
      - ``v2-ensemble`` (default): v2 + v3.4.22 contextual-bandit ensemble.

    Each underlying pass is already defensive (catches its own exceptions),
    so this wrapper adds an outer try/except to guarantee the caller
    always gets a response back. Previously three separate call sites in
    run_recall chained these; collapsing keeps precedence explicit.
    """
    if pipeline_version == "off":
        return response
    try:
        response = apply_adaptive_ranking(response, query, profile_id,
                                          config=config)
    except Exception as exc:  # pragma: no cover — defensive
        logger.debug("apply_ranking v1 step skipped: %s", exc)
    if pipeline_version == "v1":
        return response
    try:
        response = apply_v2_adaptive_ranking(
            response, query, profile_id, query_id,
        )
    except Exception as exc:  # pragma: no cover — defensive
        logger.debug("apply_ranking v2 step skipped: %s", exc)
    if pipeline_version == "v2":
        return response
    try:
        response = apply_v2_bandit_ensemble(
            response, query, profile_id, query_id,
        )
    except Exception as exc:  # pragma: no cover — defensive
        logger.debug("apply_ranking ensemble step skipped: %s", exc)
    return response


# ---------------------------------------------------------------------------
# apply_adaptive_ranking  (was MemoryEngine._apply_adaptive_ranking)
# ---------------------------------------------------------------------------

def apply_adaptive_ranking(
    response: RecallResponse,
    query: str,
    pid: str,
    *,
    config: SLMConfig,
) -> RecallResponse:
    """Apply adaptive re-ranking if enough learning signals exist.

    Phase 1 (< 50 signals): returns response unchanged (backward compat).
    Phase 2 (50+): heuristic boosts from recency, access count, trust.
    Phase 3 (200+): LightGBM ML-based reranking.
    """
    from superlocalmemory.learning.feedback import FeedbackCollector
    from pathlib import Path

    learning_db = Path.home() / ".superlocalmemory" / "learning.db"
    if not learning_db.exists():
        return response

    collector = FeedbackCollector(learning_db)
    signal_count = collector.get_feedback_count(pid)

    if signal_count < 50:
        return response  # Phase 1: no change

    from superlocalmemory.learning.ranker import AdaptiveRanker
    ranker = AdaptiveRanker(signal_count=signal_count)

    result_dicts = []
    for r in response.results:
        result_dicts.append({
            "score": r.score,
            "cross_encoder_score": r.score,
            "trust_score": r.trust_score,
            "channel_scores": r.channel_scores or {},
            "fact": {
                "age_days": 0,
                "access_count": r.fact.access_count,
            },
            "_original": r,
        })

    query_context = {"query_type": response.query_type}
    reranked = ranker.rerank(result_dicts, query_context)

    # Rebuild response with new ordering
    new_results = [d["_original"] for d in reranked]

    return RecallResponse(
        query=response.query,
        mode=response.mode,
        results=new_results,
        query_type=response.query_type,
        channel_weights=response.channel_weights,
        total_candidates=response.total_candidates,
        retrieval_time_ms=response.retrieval_time_ms,
    )


# ---------------------------------------------------------------------------
# apply_v2_adaptive_ranking (LLD-02 §4.3)
# ---------------------------------------------------------------------------
#
# Opt-in v3.4.22 path: load active model from learning.db with SHA-256
# verification, re-rank via native Booster, enqueue signals async. The
# existing ``apply_adaptive_ranking`` above stays for 3.4.20 callers.
# ---------------------------------------------------------------------------


def apply_v2_adaptive_ranking(
    response: RecallResponse,
    query: str,
    profile_id: str,
    query_id: str,
    *,
    learning_db_path: Any = None,
) -> RecallResponse:
    """LLD-02 §4.3 — load verified model, rerank, enqueue signals.

    Never raises. On any error, returns ``response`` unchanged.
    """
    try:
        from pathlib import Path as _P

        from superlocalmemory.learning.database import LearningDatabase
        from superlocalmemory.learning.model_cache import load_active
        from superlocalmemory.learning.ranker import AdaptiveRanker
        from superlocalmemory.learning.signals import (
            SignalBatch,
            SignalCandidate,
            enqueue,
        )

        db_path = (_P(learning_db_path) if learning_db_path
                   else _P.home() / ".superlocalmemory" / "learning.db")
        if not db_path.exists():
            return response

        db = LearningDatabase(db_path)
        signal_count = db.count_signals(profile_id)
        active = load_active(db, profile_id)

        ranker = AdaptiveRanker(
            signal_count=signal_count,
            active_model=active,
        )

        # Build result-dict shape expected by the ranker's rerank() path.
        result_dicts: list[dict] = []
        for r in response.results:
            result_dicts.append({
                "fact_id": r.fact.fact_id,
                "score": r.score,
                "cross_encoder_score": r.score,
                "trust_score": r.trust_score,
                "channel_scores": r.channel_scores or {},
                "fact": {
                    "age_days": 0,
                    "access_count": r.fact.access_count,
                },
                "_original": r,
            })

        query_context = {
            "query_type": response.query_type,
            "profile_id": profile_id,
        }
        reranked_dicts = ranker.rerank(result_dicts, query_context)
        new_results = [d["_original"] for d in reranked_dicts
                       if "_original" in d]

        # S8-SK-04 fix: signal enqueue is OWNED by ``apply_v2_bandit_ensemble``
        # (see below), not this function. Previously both emitted a batch
        # under the same query_id which doubled ``learning_signals`` and
        # tripped the phase-transition threshold at half the intended
        # signal count. This function now just re-ranks; the ensemble path
        # is the single source of signal events.

        return RecallResponse(
            query=response.query,
            mode=response.mode,
            results=new_results,
            query_type=response.query_type,
            channel_weights=response.channel_weights,
            total_candidates=response.total_candidates,
            retrieval_time_ms=response.retrieval_time_ms,
        )
    except Exception as exc:  # pragma: no cover — defensive
        logger.debug("apply_v2_adaptive_ranking skipped: %s", exc)
        return response


# ---------------------------------------------------------------------------
# apply_v2_bandit_ensemble (LLD-03 §5.5)
# ---------------------------------------------------------------------------
#
# Contextual Thompson bandit chooses channel weights. If an LGBM model is
# active, a D8-blended ensemble re-ranks the reweighted candidates. Never
# raises; honours ``SLM_BANDIT_DISABLED=1`` as a kill switch.
# ---------------------------------------------------------------------------


def apply_v2_bandit_ensemble(
    response: RecallResponse,
    query: str,
    profile_id: str,
    query_id: str,
    *,
    learning_db_path: Any = None,
) -> RecallResponse:
    """Apply contextual bandit + optional LGBM ensemble rerank. Safe on error."""
    import os as _os

    if _os.environ.get("SLM_BANDIT_DISABLED", "0") == "1":
        return response
    if not response.results:
        return response

    try:
        from datetime import datetime as _dt
        from pathlib import Path as _P

        from superlocalmemory.learning.bandit import ContextualBandit
        from superlocalmemory.learning.ensemble import (
            choose_ensemble,
            ensemble_rerank,
        )
        from superlocalmemory.learning.signals import (
            SignalBatch,
            SignalCandidate,
            enqueue,
        )
        from superlocalmemory.retrieval.engine import apply_channel_weights

        db_path = (_P(learning_db_path) if learning_db_path
                   else _P.home() / ".superlocalmemory" / "learning.db")
        if not db_path.exists():
            return response

        # --- 1. bandit.choose ---------------------------------------------
        entity_count = 0
        # Use query_context hints if available on the engine — cheap fallback.
        bandit = ContextualBandit(db_path, profile_id)
        choice = bandit.choose(
            {
                "query_type": response.query_type,
                "entity_count": entity_count,
            },
            query_id,
        )

        # --- 2. apply channel weights -------------------------------------
        weighted = apply_channel_weights(list(response.results), choice.weights)

        # --- 3. choose ensemble + load model (optional) -------------------
        active_model = None
        signal_count = 0
        try:
            from superlocalmemory.learning.database import LearningDatabase
            from superlocalmemory.learning.model_cache import load_active
            db = LearningDatabase(db_path)
            signal_count = db.count_signals(profile_id)
            active_model = load_active(db, profile_id)
        except Exception as exc:
            logger.debug("v2 bandit: model/signal load skipped: %s", exc)

        weights = choose_ensemble(signal_count, active_model)

        # --- 4. ensemble rerank -------------------------------------------
        query_context = {
            "query_type": response.query_type,
            "profile_id": profile_id,
            "query_id": query_id,
            "bandit_play_id": choice.play_id,
        }
        try:
            final_results = ensemble_rerank(
                weighted, choice, active_model, weights, query_context,
            )
        except Exception as exc:
            logger.debug("v2 bandit ensemble_rerank skipped: %s", exc)
            final_results = weighted

        # --- 5. enqueue signals (non-blocking) ----------------------------
        try:
            top20 = final_results[:20]
            candidates = tuple(
                SignalCandidate(
                    fact_id=r.fact.fact_id,
                    channel_scores=dict(r.channel_scores or {}),
                    cross_encoder_score=None,
                    result_dict={"fact_id": r.fact.fact_id,
                                 "score": r.score},
                )
                for r in top20
            )
            enqueue(SignalBatch(
                profile_id=profile_id,
                query_id=query_id,
                query_text=query,
                candidates=candidates,
                query_context=query_context,
            ))
        except Exception as exc:
            logger.debug("v2 bandit signal enqueue skipped: %s", exc)

        return RecallResponse(
            query=response.query,
            mode=response.mode,
            results=final_results,
            query_type=response.query_type,
            channel_weights=response.channel_weights,
            total_candidates=response.total_candidates,
            retrieval_time_ms=response.retrieval_time_ms,
        )
    except Exception as exc:  # pragma: no cover — defensive top-level
        logger.debug("apply_v2_bandit_ensemble skipped: %s", exc)
        return response


# ---------------------------------------------------------------------------
# run_recall  (was MemoryEngine.recall)
# ---------------------------------------------------------------------------

def run_recall(
    query: str,
    profile_id: str,
    mode: Mode | None = None,
    limit: int = 20,
    agent_id: str = "unknown",
    *,
    config: SLMConfig,
    retrieval_engine: Any,
    trust_scorer: Any,
    embedder: Any,
    db: DatabaseManager,
    llm: Any,
    hooks: HookRegistry,
    access_log: Any = None,
    auto_linker: Any = None,
) -> RecallResponse:
    """Recall relevant facts for a query.

    Pipeline: retrieval -> agentic sufficiency (if configured) -> post-recall updates.
    """
    # Pre-operation hooks
    hook_ctx = {
        "operation": "recall",
        "agent_id": agent_id,
        "profile_id": profile_id,
        "query_preview": query[:100],
    }
    hooks.run_pre("recall", hook_ctx)

    m = mode or config.mode

    response = retrieval_engine.recall(query, profile_id, m, limit)

    # Agentic sufficiency verification
    # V3.3.19: Only trigger for multi_hop queries in Mode A (rule-based).
    # Single-hop/factual/temporal queries get WORSE with decomposition —
    # sub-query noise dilutes precision. Mode C (LLM) can trigger broadly.
    agentic_rounds = config.retrieval.agentic_max_rounds
    if agentic_rounds > 0 and response.results:
        max_score = max((r.score for r in response.results), default=0.0)
        has_llm = llm is not None and getattr(llm, "is_available", False)
        should_trigger = (
            response.query_type == "multi_hop"
            or (has_llm and max_score < config.retrieval.agentic_confidence_threshold)
            or (has_llm and len(response.results) < 3)
        )
        if should_trigger:
            try:
                from superlocalmemory.retrieval.agentic import AgenticRetriever
                agentic = AgenticRetriever(
                    confidence_threshold=config.retrieval.agentic_confidence_threshold,
                    db=db,
                )
                enhanced_facts = agentic.retrieve(
                    query=query, profile_id=profile_id,
                    retrieval_engine=retrieval_engine,
                    llm=llm,
                    top_k=limit,
                    query_type=response.query_type,
                )
                # Replace response results with enhanced facts if we got more
                if len(enhanced_facts) > len(response.results):
                    from superlocalmemory.storage.models import RetrievalResult
                    enhanced_results = []
                    for i, f in enumerate(enhanced_facts):
                        # Look up real trust score for agentic results
                        fact_trust = 0.5
                        if trust_scorer:
                            try:
                                fact_trust = trust_scorer.get_fact_trust(
                                    f.fact_id, profile_id,
                                )
                            except Exception:
                                pass
                        enhanced_results.append(RetrievalResult(
                            fact=f, score=1.0 / (i + 1),
                            channel_scores={"agentic": 1.0},
                            confidence=f.confidence,
                            evidence_chain=["agentic_round_2"],
                            trust_score=fact_trust,
                        ))
                    response = RecallResponse(
                        query=query, mode=m, results=enhanced_results[:limit],
                        query_type=response.query_type,
                        channel_weights=response.channel_weights,
                        total_candidates=response.total_candidates + len(enhanced_facts),
                        retrieval_time_ms=response.retrieval_time_ms,
                    )
            except Exception as exc:
                logger.debug("Agentic sufficiency skipped: %s", exc)

    # V3.2: Log access for recalled facts (Phase 1)
    if access_log and response.results:
        try:
            fact_ids = [r.fact.fact_id for r in response.results]
            access_log.store_access_batch(
                fact_ids=fact_ids,
                profile_id=profile_id,
                access_type="recall",
            )
        except Exception as exc:
            logger.debug("Access log batch store failed: %s", exc)

    # V3.4.11: Promote recalled facts back to active tier (single batch UPDATE)
    if response.results:
        try:
            from superlocalmemory.core.tier_manager import promote_on_access_batch
            fact_ids = [r.fact.fact_id for r in response.results[:10]]
            promote_on_access_batch(db, fact_ids)
        except Exception:
            pass  # tier_manager not available yet — graceful

    # V3.3.16: Behavioral tracking + spaced repetition use module-level
    # singletons to avoid creating new objects per recall (was causing
    # object accumulation across 304 benchmark recalls).
    try:
        # v3.4.7: Extract entities from results for behavioral tracking.
        # Was passing wrong param (result_count instead of entities) → TypeError.
        entities: list[str] = []
        for r in response.results[:10]:
            rd = r if isinstance(r, dict) else (dict(r) if hasattr(r, "keys") else {})
            ents_json = rd.get("canonical_entities_json", "")
            if ents_json:
                try:
                    import json as _json
                    entities.extend(_json.loads(ents_json))
                except (ValueError, TypeError):
                    pass
        _get_behavioral_tracker(db).record_query(
            query=query,
            query_type=response.query_type,
            entities=entities[:20],
            profile_id=profile_id,
        )
    except Exception as exc:
        logger.debug("Behavioral tracking: %s", exc)

    if response.results:
        try:
            fsched = _get_forgetting_scheduler(db, config)
            for r in response.results[:10]:
                fsched.on_access_event(r.fact.fact_id, profile_id)
        except Exception as exc:
            logger.debug("Spaced repetition update: %s", exc)

    # Phase 3: Hebbian strengthening for co-accessed facts
    if auto_linker and response.results:
        try:
            recalled_ids = [
                r.fact.fact_id for r in response.results[:10]
            ]
            auto_linker.strengthen_co_access(recalled_ids, profile_id)
        except Exception as exc:
            logger.debug("Hebbian strengthening: %s", exc)

    # S8-ARC-04 (v3.4.22): unified ranking entry point. Single env-var
    # (SLM_RANKING=off|v1|v2|v2-ensemble) controls the pipeline. Legacy
    # SLM_V2_PIPELINE_DISABLED + SLM_BANDIT_DISABLED still honoured for
    # one-release back-compat. Identity when no active model.
    try:
        import os as _os
        import uuid as _uuid
        query_id = _uuid.uuid4().hex
        mode = _resolve_ranking_mode(_os.environ)
        response = apply_ranking(
            response, query, profile_id, query_id,
            config=config, pipeline_version=mode,
        )
    except Exception as exc:
        logger.debug("Ranking pipeline skipped: %s", exc)

    # Reconsolidation: access updates trust + count (neuroscience principle)
    if trust_scorer:
        for r in response.results:
            trust_scorer.update_on_access("fact", r.fact.fact_id, profile_id)

    # Fisher Bayesian update on recall — narrows variance on accessed facts
    # so they score higher on subsequent recalls (critical for benchmark: +24pp).
    # V3.3.16: Reuse query embedding from retrieval engine cache instead of
    # calling embedder.embed() again (which was the memory leak source).
    q_var_arr = None
    if embedder and hasattr(retrieval_engine, '_query_embedding_cache'):
        cached_emb = retrieval_engine._query_embedding_cache.get(query)
        if cached_emb is not None:
            import numpy as _np
            _, q_var_list = embedder.compute_fisher_params(cached_emb)
            q_var_arr = _np.array(q_var_list, dtype=_np.float64)

    for r in response.results:
        updates: dict[str, object] = {
            "access_count": r.fact.access_count + 1,
        }
        if (q_var_arr is not None
                and r.fact.fisher_variance
                and len(r.fact.fisher_variance) == len(q_var_arr)
                and r.fact.access_count >= 3):
            import numpy as _np
            f_var = _np.array(r.fact.fisher_variance, dtype=_np.float64)
            new_var = 1.0 / (1.0 / _np.maximum(f_var, 0.05) + 1.0 / _np.maximum(q_var_arr, 0.05))
            new_var = _np.clip(new_var, 0.05, 2.0)
            updates["fisher_variance"] = new_var.tolist()
        db.update_fact(r.fact.fact_id, updates)

    # Post-operation hooks (audit, trust signal, learning)
    hook_ctx["result_count"] = len(response.results)
    hook_ctx["query_type"] = response.query_type
    hooks.run_post("recall", hook_ctx)

    # LLD-00 §3 — stamp HMAC markers on every result so post_tool_outcome_hook
    # can validate fact_ids observed in downstream tool output.
    _apply_markers_to_response(response)

    return response
