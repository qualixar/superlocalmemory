# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""V3-native feature extractor for adaptive ranking.

Extracts features from retrieval results for LightGBM training.
Features are grouped by source: channel scores, fusion, reranker,
math layers, query analysis, memory metadata, user history.

Each feature vector has a fixed dimension (FEATURE_DIM).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

# Feature dimension — total number of features extracted
FEATURE_DIM = 20


@dataclass(frozen=True)
class FeatureVector:
    """Immutable feature vector for one retrieval result."""
    fact_id: str
    query_id: str
    features: dict[str, float]
    label: float | None = None  # None = unlabeled (inference), float = labeled (training)

    def to_list(self) -> list[float]:
        """Convert to ordered list for LightGBM input."""
        return [self.features.get(name, 0.0) for name in FEATURE_NAMES]


# Ordered feature names — must match FEATURE_DIM
FEATURE_NAMES: list[str] = [
    # Channel scores (4)
    "semantic_score",
    "bm25_score",
    "entity_score",
    "temporal_score",
    # Fusion (2)
    "rrf_rank",
    "rrf_score",
    # Reranker (1)
    "cross_encoder_score",
    # Math layers (3)
    "fisher_distance",
    "fisher_confidence",
    "sheaf_consistent",
    # Query features (4)
    "query_type_sh",  # one-hot: single-hop
    "query_type_mh",  # one-hot: multi-hop
    "query_type_temp",  # one-hot: temporal
    "query_type_od",  # one-hot: open-domain
    # Memory metadata (4)
    "fact_age_days",
    "access_count",
    "fact_trust_score",
    "fact_confidence",
    # User/profile (2)
    "profile_recall_count",
    "topic_affinity",
]

assert len(FEATURE_NAMES) == FEATURE_DIM


class FeatureExtractor:
    """Extract features from V3 retrieval results for LightGBM ranking."""

    @staticmethod
    def extract(result: dict[str, Any], query_context: dict[str, Any]) -> FeatureVector:
        """Extract features from a single retrieval result.

        Args:
            result: dict with keys from RetrievalResult (score, channel_scores,
                    fact metadata, etc.)
            query_context: dict with query_type, query_length, profile stats, etc.

        Returns:
            FeatureVector with FEATURE_DIM features.
        """
        channel = result.get("channel_scores", {})
        fact = result.get("fact", {})

        features = {
            # Channel scores
            "semantic_score": channel.get("semantic", 0.0),
            "bm25_score": channel.get("bm25", 0.0),
            "entity_score": channel.get("entity_graph", 0.0),
            "temporal_score": channel.get("temporal", 0.0),
            # Fusion
            "rrf_rank": result.get("rrf_rank", 0.0),
            "rrf_score": result.get("rrf_score", 0.0),
            # Reranker
            "cross_encoder_score": result.get("cross_encoder_score", 0.0),
            # Math
            "fisher_distance": result.get("fisher_distance", 0.0),
            "fisher_confidence": result.get("fisher_confidence", 0.0),
            "sheaf_consistent": 1.0 if result.get("sheaf_consistent", True) else 0.0,
            # Query (one-hot)
            "query_type_sh": 1.0 if query_context.get("query_type") == "single_hop" else 0.0,
            "query_type_mh": 1.0 if query_context.get("query_type") == "multi_hop" else 0.0,
            "query_type_temp": 1.0 if query_context.get("query_type") == "temporal" else 0.0,
            "query_type_od": 1.0 if query_context.get("query_type") == "open_domain" else 0.0,
            # Memory metadata
            "fact_age_days": _safe_float(fact.get("age_days", 0)),
            "access_count": _safe_float(fact.get("access_count", 0)),
            "fact_trust_score": _safe_float(result.get("trust_score", 0.5)),
            "fact_confidence": _safe_float(fact.get("confidence", 0.7)),
            # User/profile
            "profile_recall_count": _safe_float(query_context.get("profile_recall_count", 0)),
            "topic_affinity": _safe_float(query_context.get("topic_affinity", 0.0)),
        }

        return FeatureVector(
            fact_id=result.get("fact_id", ""),
            query_id=query_context.get("query_id", ""),
            features=features,
        )

    @staticmethod
    def extract_batch(results: list[dict], query_context: dict) -> list[FeatureVector]:
        """Extract features from a batch of retrieval results."""
        return [FeatureExtractor.extract(r, query_context) for r in results]


def _safe_float(value: Any) -> float:
    """Convert to float safely, defaulting to 0.0."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
