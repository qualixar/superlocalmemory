# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Exposure rows recorded in shadow while ranking is off (W3-6c, D-45).

The ranker learns from what agents used of what a recall offered. With
``SLM_RANKING`` off (the default) nothing recorded what was offered, so no
training row could exist (F-48). Recall now queues its top 20 as exposure rows
whenever the store measures (``outcomes.receipts``), without changing the
order. The rows are the same ones the ensemble logs, keyed by the recall's
``query_id``, which its pull receipt also carries.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def enqueue_exposures(response: Any, query: str, profile_id: str,
                      query_id: str) -> bool:
    """Queue the top 20 results as exposure rows for the ranker's training data.

    The same rows the ensemble logs (``SignalCandidate`` per result, features
    computed later by the signal worker from the same inputs), keyed by the
    recall's ``query_id``, the id its pull receipt carries (W3-6c, D-45). A
    ``put_nowait`` onto a bounded queue: a full queue drops and counts, and a
    recall never waits. Never raises."""
    if not query_id or not response.results:
        return False
    try:
        from superlocalmemory.learning.signals import SignalBatch, SignalCandidate, enqueue

        enqueue(SignalBatch(
            profile_id=profile_id, query_id=query_id, query_text=query,
            candidates=tuple(
                SignalCandidate(fact_id=r.fact.fact_id,
                                channel_scores=dict(r.channel_scores or {}),
                                cross_encoder_score=None,
                                result_dict={"fact_id": r.fact.fact_id, "score": r.score})
                for r in response.results[:20]),
            query_context={"query_type": response.query_type, "profile_id": profile_id,
                           "query_id": query_id}))
        return True
    except Exception as exc:  # a training row is never worth a failed recall
        logger.debug("exposure enqueue skipped: %s", exc)
        return False


def measuring(config: Any) -> bool:
    """Whether this store records measurement (``outcomes.receipts`` shadow or
    on, and no kill switch): the umbrella for receipts and exposure rows."""
    try:
        from superlocalmemory.core.feature_flags import FeatureFlags

        home = getattr(config, "base_dir", None)
        return bool(home) and FeatureFlags(home).observing("outcomes.receipts")
    except Exception:  # noqa: BLE001 - an unreadable flag file measures nothing
        return False


__all__ = ["enqueue_exposures", "measuring"]
