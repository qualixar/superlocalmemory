#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 SuperLocalMemory (superlocalmemory.com)
"""
Helper functions for AdaptiveRanker.

Extracted from adaptive_ranker.py to reduce file size while maintaining
backward compatibility.
"""

import logging
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional

from .constants import RULE_BOOST

logger = logging.getLogger("superlocalmemory.learning.ranking.helpers")

# NumPy is optional — used for feature matrix construction
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    np = None
    HAS_NUMPY = False


def calculate_rule_boost(features: List[float]) -> float:
    """
    Calculate rule-based boost multiplier from extracted features.

    This function encapsulates the rule-based boosting logic from Phase 1,
    making the main rerank method more readable.

    Args:
        features: Feature vector extracted for a memory.

    Returns:
        Boost multiplier (typically 0.5 to 2.0).
    """
    boost = 1.0

    # Feature [2]: tech_match
    tech_match = features[2]
    if tech_match >= 0.8:
        boost *= RULE_BOOST['tech_match_strong']
    elif tech_match >= 0.4:
        boost *= RULE_BOOST['tech_match_weak']

    # Feature [3]: project_match
    project_match = features[3]
    if project_match >= 0.9:
        boost *= RULE_BOOST['project_match']
    elif project_match <= 0.35:
        boost *= RULE_BOOST['project_mismatch']

    # Feature [5]: source_quality
    source_quality = features[5]
    if source_quality >= 0.7:
        boost *= RULE_BOOST['source_quality_high']
    elif source_quality < 0.3:
        boost *= RULE_BOOST['source_quality_low']

    # Feature [7]: recency_score (exponential decay)
    recency = features[7]
    # Linear interpolation between penalty and boost
    recency_factor = (
        RULE_BOOST['recency_penalty_max']
        + recency * (
            RULE_BOOST['recency_boost_max']
            - RULE_BOOST['recency_penalty_max']
        )
    )
    boost *= recency_factor

    # Feature [6]: importance_norm
    importance_norm = features[6]
    if importance_norm >= 0.8:
        boost *= RULE_BOOST['high_importance']

    # Feature [8]: access_frequency
    access_freq = features[8]
    if access_freq >= 0.5:
        boost *= RULE_BOOST['high_access']

    # Feature [10]: signal_count (v2.7.4 — feedback volume)
    if len(features) > 10:
        signal_count = features[10]
        if signal_count >= 0.3:  # 3+ signals
            boost *= 1.1  # Mild boost for well-known memories

    # Feature [11]: avg_signal_value (v2.7.4 — feedback quality)
    if len(features) > 11:
        avg_signal = features[11]
        if avg_signal >= 0.7:
            boost *= 1.15  # Boost memories with positive feedback
        elif avg_signal < 0.3 and avg_signal > 0.0:
            boost *= 0.85  # Penalize memories with negative feedback

    # Feature [12]: lifecycle_state (v2.8)
    if len(features) > 12:
        lifecycle_state = features[12]
        if lifecycle_state >= 0.9:
            boost *= RULE_BOOST.get('lifecycle_active', 1.0)
        elif lifecycle_state >= 0.6:
            boost *= RULE_BOOST.get('lifecycle_warm', 0.85)
        elif lifecycle_state >= 0.3:
            boost *= RULE_BOOST.get('lifecycle_cold', 0.6)

    # Feature [13]: outcome_success_rate (v2.8)
    if len(features) > 13:
        success_rate = features[13]
        if success_rate >= 0.8:
            boost *= RULE_BOOST.get('outcome_success_high', 1.3)
        elif success_rate <= 0.2:
            boost *= RULE_BOOST.get('outcome_failure_high', 0.7)

    # Feature [15]: behavioral_match (v2.8)
    if len(features) > 15:
        behavioral = features[15]
        if behavioral >= 0.7:
            boost *= RULE_BOOST.get('behavioral_match_strong', 1.25)

    # Feature [16]: cross_project_score (v2.8)
    if len(features) > 16:
        cross_project = features[16]
        if cross_project >= 0.5:
            boost *= RULE_BOOST.get('cross_project_boost', 1.15)

    # Feature [18]: trust_at_creation (v2.8)
    if len(features) > 18:
        trust = features[18]
        if trust >= 0.9:
            boost *= RULE_BOOST.get('high_trust_creator', 1.1)
        elif trust <= 0.3:
            boost *= RULE_BOOST.get('low_trust_creator', 0.8)

    return boost


def prepare_training_data_internal(
    feedback: List[dict],
    feature_extractor,
) -> Optional[tuple]:
    """
    Prepare training data from feedback records.

    For each unique query (grouped by query_hash):
        - Fetch all feedback entries for that query
        - Look up the corresponding memory from memory.db
        - Extract features for each memory
        - Use signal_value as the relevance label

    Args:
        feedback: List of feedback records from LearningDB.
        feature_extractor: FeatureExtractor instance with context set.

    Returns:
        Tuple of (X, y, groups) for LGBMRanker, or None if insufficient.
        X: numpy array (n_samples, NUM_FEATURES)
        y: numpy array (n_samples,) — relevance labels
        groups: list of ints — samples per query group
    """
    if not HAS_NUMPY:
        logger.warning("NumPy not available for training data preparation")
        return None

    if not feedback:
        return None

    # Group feedback by query_hash
    query_groups: Dict[str, List[dict]] = {}
    for entry in feedback:
        qh = entry['query_hash']
        if qh not in query_groups:
            query_groups[qh] = []
        query_groups[qh].append(entry)

    # Filter: only keep groups with 2+ items (ranking requires pairs)
    query_groups = {
        qh: entries for qh, entries in query_groups.items()
        if len(entries) >= 2
    }

    if not query_groups:
        logger.info("No query groups with 2+ feedback entries")
        return None

    # Collect memory IDs we need to look up
    memory_ids_needed = set()
    for entries in query_groups.values():
        for entry in entries:
            memory_ids_needed.add(entry['memory_id'])

    # Fetch memories from memory.db
    memory_db_path = Path.home() / ".claude-memory" / "memory.db"
    if not memory_db_path.exists():
        logger.warning("memory.db not found at %s", memory_db_path)
        return None

    memories_by_id = {}
    try:
        conn = sqlite3.connect(str(memory_db_path), timeout=5)
        try:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            # Batch fetch memories (in chunks to avoid SQLite variable limit)
            id_list = list(memory_ids_needed)
            chunk_size = 500
            for i in range(0, len(id_list), chunk_size):
                chunk = id_list[i:i + chunk_size]
                placeholders = ','.join('?' for _ in chunk)
                cursor.execute(f'''
                    SELECT id, content, summary, project_path, project_name,
                           tags, category, memory_type, importance, created_at,
                           last_accessed, access_count
                    FROM memories
                    WHERE id IN ({placeholders})
                ''', chunk)
                for row in cursor.fetchall():
                    memories_by_id[row['id']] = dict(row)
        finally:
            conn.close()
    except Exception as e:
        logger.error("Failed to fetch memories for training: %s", e)
        return None

    # Build feature matrix and labels
    all_features = []
    all_labels = []
    groups = []

    # Set a neutral context for training (we don't have query-time context)
    feature_extractor.set_context()

    for qh, entries in query_groups.items():
        group_features = []
        group_labels = []

        for entry in entries:
            mid = entry['memory_id']
            memory = memories_by_id.get(mid)
            if memory is None:
                continue  # Memory may have been deleted

            # Use query_keywords as proxy for query text
            query_text = entry.get('query_keywords', '') or ''

            features = feature_extractor.extract_features(
                memory, query_text
            )
            group_features.append(features)
            group_labels.append(float(entry['signal_value']))

        # Only include groups with 2+ valid entries
        if len(group_features) >= 2:
            all_features.extend(group_features)
            all_labels.extend(group_labels)
            groups.append(len(group_features))

    if not groups or len(all_features) < 4:
        logger.info(
            "Insufficient valid training data: %d features, %d groups",
            len(all_features), len(groups)
        )
        return None

    X = np.array(all_features, dtype=np.float64)
    y = np.array(all_labels, dtype=np.float64)

    logger.info(
        "Prepared training data: %d samples, %d groups, %d features",
        X.shape[0], len(groups), X.shape[1]
    )

    return X, y, groups
