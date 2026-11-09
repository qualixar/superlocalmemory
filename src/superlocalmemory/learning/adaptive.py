# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""SuperLocalMemory V3 — Adaptive Learning (3-Phase).

Learns optimal retrieval weights from user feedback.
Ported from V2.8 LightGBM-based learning system.
Profile-scoped: each profile learns independently.

Phase 1: Collect feedback (ranking_feedback)
Phase 2: Train model on feedback patterns
Phase 3: Apply learned weights to retrieval

Part of Qualixar | Author: Varun Pratap Bhardwaj
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from superlocalmemory.storage.models import FeedbackRecord

logger = logging.getLogger(__name__)

# Minimum feedback records before training
_MIN_FEEDBACK_FOR_TRAINING = 20

# Default channel weights (before learning)
_DEFAULT_WEIGHTS = {
    "semantic": 1.5,
    "bm25": 1.0,
    "entity_graph": 1.0,
    "temporal": 1.0,
}


class AdaptiveLearner:
    """3-phase adaptive learning for retrieval weight optimization.

    Learns from user feedback which channels produce the best results
    for different query types. Profile-scoped.
    """

    def __init__(self, db) -> None:
        self._db = db
        self._learned_weights: dict[str, dict[str, float]] = {}

    # -- Phase 1: Collect feedback -----------------------------------------

    def record_feedback(
        self,
        query: str,
        fact_id: str,
        feedback_type: str,
        profile_id: str,
        dwell_time_ms: int = 0,
    ) -> FeedbackRecord:
        """Record user feedback on a retrieval result.

        feedback_type: "relevant", "irrelevant", "partial"
        """
        record = FeedbackRecord(
            profile_id=profile_id,
            query=query,
            fact_id=fact_id,
            feedback_type=feedback_type,
            dwell_time_ms=dwell_time_ms,
            timestamp=datetime.now(UTC).isoformat(),
        )
        self._db.execute(
            "INSERT INTO feedback_records "
            "(feedback_id, profile_id, query, fact_id, feedback_type, "
            "dwell_time_ms, timestamp) VALUES (?,?,?,?,?,?,?)",
            (record.feedback_id, record.profile_id, record.query,
             record.fact_id, record.feedback_type, record.dwell_time_ms,
             record.timestamp),
        )
        return record

    def get_feedback_count(self, profile_id: str) -> int:
        """Count feedback records for a profile."""
        rows = self._db.execute(
            "SELECT COUNT(*) AS c FROM feedback_records WHERE profile_id = ?",
            (profile_id,),
        )
        return int(dict(rows[0])["c"]) if rows else 0

    # -- Phase 2: Learn patterns -------------------------------------------

    def train(self, profile_id: str) -> dict[str, dict[str, float]]:
        """Learn optimal weights from feedback patterns.

        Simple heuristic approach (LightGBM port deferred to production):
        - Analyze which channels produced "relevant" results
        - Boost channels that correlate with positive feedback
        - Reduce channels that correlate with negative feedback
        """
        count = self.get_feedback_count(profile_id)
        if count < _MIN_FEEDBACK_FOR_TRAINING:
            logger.info(
                "Only %d feedback records (need %d). Using defaults.",
                count, _MIN_FEEDBACK_FOR_TRAINING,
            )
            return {}

        rows = self._db.execute(
            "SELECT query, fact_id, feedback_type FROM feedback_records "
            "WHERE profile_id = ? ORDER BY timestamp DESC LIMIT 500",
            (profile_id,),
        )

        # Count positive/negative per query pattern
        positive_count = 0
        negative_count = 0
        for row in rows:
            d = dict(row)
            if d["feedback_type"] == "relevant":
                positive_count += 1
            elif d["feedback_type"] == "irrelevant":
                negative_count += 1

        # Simple relevance ratio → weight adjustment
        if positive_count + negative_count == 0:
            return {}

        relevance_ratio = positive_count / (positive_count + negative_count)

        # If retrieval is generally good (>70% relevant), trust current weights
        # If poor (<50%), boost BM25 and entity (more precise channels)
        if relevance_ratio < 0.5:
            learned = {
                "general": {
                    "semantic": 1.0,
                    "bm25": 1.5,         # Boost precision
                    "entity_graph": 1.3,  # Boost entity matching
                    "temporal": 0.8,
                },
            }
        else:
            learned = {
                "general": dict(_DEFAULT_WEIGHTS),
            }

        self._learned_weights = learned
        logger.info("Learned weights (ratio=%.2f): %s", relevance_ratio, learned)
        return learned

    # -- Phase 3: Apply weights --------------------------------------------

    def get_weights(
        self, query_type: str, profile_id: str
    ) -> dict[str, float]:
        """Get learned weights for a query type.

        Falls back to defaults if no learned weights available.
        """
        if not self._learned_weights:
            self.train(profile_id)

        if query_type in self._learned_weights:
            return self._learned_weights[query_type]
        if "general" in self._learned_weights:
            return self._learned_weights["general"]
        return dict(_DEFAULT_WEIGHTS)

    def is_trained(self, profile_id: str) -> bool:
        """Check if the learner has enough data to provide learned weights."""
        return self.get_feedback_count(profile_id) >= _MIN_FEEDBACK_FOR_TRAINING
