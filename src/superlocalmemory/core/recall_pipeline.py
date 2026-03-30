# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under the MIT License - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Recall pipeline — extracted free functions for MemoryEngine.recall().

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

from superlocalmemory.storage.models import Mode, RecallResponse

logger = logging.getLogger(__name__)


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
    agentic_rounds = config.retrieval.agentic_max_rounds
    if agentic_rounds > 0 and response.results:
        max_score = max((r.score for r in response.results), default=0.0)
        should_trigger = (
            max_score < config.retrieval.agentic_confidence_threshold
            or response.query_type == "multi_hop"
            or len(response.results) < 3
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

    # Phase 3: Hebbian strengthening for co-accessed facts
    if auto_linker and response.results:
        try:
            recalled_ids = [
                r.fact.fact_id for r in response.results[:10]
            ]
            auto_linker.strengthen_co_access(recalled_ids, profile_id)
        except Exception as exc:
            logger.debug("Hebbian strengthening: %s", exc)

    # Adaptive re-ranking (V3.1 Active Memory)
    try:
        response = apply_adaptive_ranking(response, query, profile_id, config=config)
    except Exception as exc:
        logger.debug("Adaptive ranking skipped: %s", exc)

    # Reconsolidation: access updates trust + count (neuroscience principle)
    if trust_scorer:
        for r in response.results:
            trust_scorer.update_on_access("fact", r.fact.fact_id, profile_id)

    # Fisher Bayesian update on recall
    q_emb = embedder.embed(query) if embedder else None
    q_var_arr = None
    if embedder and q_emb:
        _, q_var_list = embedder.compute_fisher_params(q_emb)
        import numpy as _np
        q_var_arr = _np.array(q_var_list, dtype=_np.float64)

    for r in response.results:
        updates: dict[str, object] = {
            "access_count": r.fact.access_count + 1,
        }
        # Bayesian variance narrowing after 3+ accesses
        if (q_var_arr is not None
                and r.fact.fisher_variance
                and len(r.fact.fisher_variance) == len(q_var_arr)
                and r.fact.access_count >= 3):
            import numpy as _np
            f_var = _np.array(r.fact.fisher_variance, dtype=_np.float64)
            # Conjugate Gaussian update: 1/new_var = 1/f_var + 1/q_var
            new_var = 1.0 / (1.0 / _np.maximum(f_var, 0.05) + 1.0 / _np.maximum(q_var_arr, 0.05))
            new_var = _np.clip(new_var, 0.05, 2.0)
            updates["fisher_variance"] = new_var.tolist()

        db.update_fact(r.fact.fact_id, updates)

    # Post-operation hooks (audit, trust signal, learning)
    hook_ctx["result_count"] = len(response.results)
    hook_ctx["query_type"] = response.query_type
    hooks.run_post("recall", hook_ctx)

    return response
