#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 SuperLocalMemory (superlocalmemory.com)
"""
SyntheticBootstrapper — Bootstrap ML model from existing data patterns.

PROBLEM: LightGBM needs 200+ feedback signals across 50+ unique queries
to activate ML ranking (Phase 2). A new user has zero feedback. Without
bootstrap, users must endure ~200 recalls before getting personalization.
That's weeks of usage with no benefit. Users abandon before reaching Phase 2.

SOLUTION: Generate synthetic (query, memory, relevance_label) tuples from
EXISTING data patterns in memory.db. These aren't real user feedback, but
they encode reasonable assumptions:
    - Frequently accessed memories are probably relevant to their keywords
    - High-importance memories should rank higher for their topics
    - Learned patterns (from pattern_learner.py) encode real preferences
    - Recent memories should generally outrank older ones

Four Strategies:
    1. Access-based: Memories accessed 5+ times -> positive for their keywords
    2. Importance-based: Importance >= 8 -> positive for their tags
    3. Pattern-based: Learned identity_patterns -> positive for matching memories
    4. Recency decay: For any synthetic query, recent memories rank higher

The bootstrap model uses MORE aggressive regularization than the real model
(fewer trees, smaller depth, higher reg_lambda) to prevent overfitting
on synthetic data. Once real feedback accumulates, the model is retrained
with continued learning (init_model), gradually replacing synthetic signal
with real signal.

Research Backing:
    - FCS LREC 2024: Cold-start mitigation via synthetic bootstrap
    - eKNOW 2025: BM25 -> re-ranker pipeline effectiveness
"""

import hashlib
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

# LightGBM is OPTIONAL — bootstrap only works when LightGBM is installed
try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    lgb = None
    HAS_LIGHTGBM = False

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    np = None
    HAS_NUMPY = False

from .feature_extractor import FeatureExtractor, FEATURE_NAMES, NUM_FEATURES
from .bootstrap import (
    MEMORY_DB_PATH,
    MODELS_DIR,
    MODEL_PATH,
    MIN_MEMORIES_FOR_BOOTSTRAP,
    BOOTSTRAP_CONFIG,
    BOOTSTRAP_PARAMS,
    extract_keywords,
    get_memory_count,
    get_memories_by_access,
    get_memories_by_importance,
    get_recent_memories,
    get_learned_patterns,
    search_memories,
    find_negative_memories,
    diverse_sample,
    count_sources,
)

logger = logging.getLogger("superlocalmemory.learning.synthetic_bootstrap")


class SyntheticBootstrapper:
    """
    Generates synthetic training data and bootstraps the ML ranking model.

    Usage:
        bootstrapper = SyntheticBootstrapper()
        if bootstrapper.should_bootstrap():
            result = bootstrapper.bootstrap_model()
            if result:
                print(f"Bootstrapped with {result['training_samples']} samples")

    The bootstrapped model is saved to the same path as the real model.
    When real feedback accumulates, AdaptiveRanker.train() uses
    continued learning (init_model) to incrementally replace synthetic
    signal with real signal.
    """

    MIN_MEMORIES_FOR_BOOTSTRAP = MIN_MEMORIES_FOR_BOOTSTRAP
    BOOTSTRAP_CONFIG = BOOTSTRAP_CONFIG

    def __init__(
        self,
        memory_db_path: Optional[Path] = None,
        learning_db=None,
    ):
        """
        Initialize SyntheticBootstrapper.

        Args:
            memory_db_path: Path to memory.db (defaults to ~/.claude-memory/memory.db).
            learning_db: Optional LearningDB instance for recording metadata.
        """
        self._memory_db = Path(memory_db_path) if memory_db_path else MEMORY_DB_PATH
        self._learning_db = learning_db
        self._feature_extractor = FeatureExtractor()

    # ========================================================================
    # LearningDB Access
    # ========================================================================

    def _get_learning_db(self):
        """Get or create the LearningDB instance."""
        if self._learning_db is None:
            try:
                from .learning_db import LearningDB
                self._learning_db = LearningDB()
            except Exception as e:
                logger.warning("Cannot access LearningDB: %s", e)
                return None
        return self._learning_db

    # ========================================================================
    # Pre-flight Checks
    # ========================================================================

    def should_bootstrap(self) -> bool:
        """
        Check if synthetic bootstrap is needed and possible.

        Returns True if:
            1. LightGBM + NumPy are available
            2. No existing model file (or forced rebuild)
            3. At least MIN_MEMORIES_FOR_BOOTSTRAP memories exist in memory.db
        """
        if not HAS_LIGHTGBM or not HAS_NUMPY:
            logger.debug("Bootstrap unavailable: LightGBM=%s, NumPy=%s",
                         HAS_LIGHTGBM, HAS_NUMPY)
            return False

        if MODEL_PATH.exists():
            logger.debug("Model already exists at %s — skipping bootstrap",
                         MODEL_PATH)
            return False

        memory_count = self._get_memory_count()
        if memory_count < MIN_MEMORIES_FOR_BOOTSTRAP:
            logger.debug(
                "Not enough memories for bootstrap: %d (need %d)",
                memory_count, MIN_MEMORIES_FOR_BOOTSTRAP
            )
            return False

        return True

    def get_tier(self) -> Optional[str]:
        """
        Determine bootstrap tier based on memory count.

        Returns:
            'small', 'medium', 'large', or None if < MIN_MEMORIES.
        """
        count = self._get_memory_count()
        for tier_name, config in BOOTSTRAP_CONFIG.items():
            if config['min_memories'] <= count <= config['max_memories']:
                return tier_name
        return None

    def _get_memory_count(self) -> int:
        """Count total memories in memory.db."""
        return get_memory_count(self._memory_db)

    # ========================================================================
    # Synthetic Data Generation
    # ========================================================================

    def generate_synthetic_training_data(self) -> List[dict]:
        """
        Generate synthetic (query, memory, label, features) records.

        Combines four strategies to produce training data from existing
        memory patterns. Each record contains:
            - query: Synthetic query string (extracted keywords)
            - memory_id: ID of the memory in memory.db
            - label: Relevance label (0.0 = irrelevant, 1.0 = highly relevant)
            - source: Which strategy generated this record
            - features: 9-dimensional feature vector

        Returns:
            List of training record dicts. May be empty if insufficient data.
        """
        records = []

        # Strategy 1: Access-based pseudo-labels
        access_records = self._generate_access_based()
        records.extend(access_records)
        logger.info("Strategy 1 (access): %d records", len(access_records))

        # Strategy 2: Importance-based pseudo-labels
        importance_records = self._generate_importance_based()
        records.extend(importance_records)
        logger.info("Strategy 2 (importance): %d records",
                     len(importance_records))

        # Strategy 3: Pattern-based synthetic queries
        pattern_records = self._generate_pattern_based()
        records.extend(pattern_records)
        logger.info("Strategy 3 (patterns): %d records", len(pattern_records))

        # Strategy 4: Recency decay pseudo-labels
        recency_records = self._generate_recency_based()
        records.extend(recency_records)
        logger.info("Strategy 4 (recency): %d records", len(recency_records))

        logger.info("Total synthetic records: %d", len(records))
        return records

    def _generate_access_based(self) -> List[dict]:
        """
        Strategy 1: Memories accessed 5+ times are relevant for their keywords.

        Logic: If a user keeps coming back to a memory via certain searches,
        the keywords in that memory are relevant queries for it.
        """
        records = []
        high_access_memories = self._get_memories_by_access(min_access=5)

        for memory in high_access_memories:
            keywords = self._extract_keywords(memory.get('content', ''))
            if not keywords:
                continue

            query = ' '.join(keywords)

            # Positive: This memory is relevant to its own keywords
            records.append(self._build_record(
                query=query,
                memory=memory,
                label=1.0,
                source='access_positive',
            ))

            # Find some non-matching memories as negatives
            negatives = self._find_negative_memories(
                memory, exclude_ids={memory['id']}, limit=2
            )
            for neg_memory in negatives:
                records.append(self._build_record(
                    query=query,
                    memory=neg_memory,
                    label=0.0,
                    source='access_negative',
                ))

        return records

    def _generate_importance_based(self) -> List[dict]:
        """
        Strategy 2: High-importance memories (>= 8) are positive for their tags.

        Logic: User explicitly rated these memories as important. Their tags
        represent topics the user cares about.
        """
        records = []
        important_memories = self._get_memories_by_importance(min_importance=8)

        for memory in important_memories:
            # Use tags as synthetic query, fall back to content keywords
            tags = memory.get('tags', '')
            if isinstance(tags, str):
                try:
                    import json
                    tags_list = json.loads(tags)
                except (ValueError, TypeError):
                    tags_list = [t.strip() for t in tags.split(',') if t.strip()]
            elif isinstance(tags, list):
                tags_list = tags
            else:
                tags_list = []

            if tags_list:
                query = ' '.join(tags_list[:5])
            else:
                keywords = self._extract_keywords(memory.get('content', ''))
                query = ' '.join(keywords) if keywords else ''

            if not query:
                continue

            # Positive: High-importance memory matches its tags
            records.append(self._build_record(
                query=query,
                memory=memory,
                label=1.0,
                source='importance_positive',
            ))

            # Find some negatives
            negatives = self._find_negative_memories(
                memory, exclude_ids={memory['id']}, limit=2
            )
            for neg_memory in negatives:
                records.append(self._build_record(
                    query=query,
                    memory=neg_memory,
                    label=0.0,
                    source='importance_negative',
                ))

        return records

    def _generate_pattern_based(self) -> List[dict]:
        """
        Strategy 3: Use learned identity_patterns to create synthetic queries.

        Logic: Pattern learner has already identified user's tech preferences,
        coding style, etc. Use these as queries and find matching memories.
        """
        records = []
        patterns = self._get_learned_patterns(min_confidence=0.7)

        if not patterns:
            return records

        for pattern in patterns:
            # Build query from pattern key + value
            query_parts = []
            key = pattern.get('key', '')
            value = pattern.get('value', '')
            if key:
                query_parts.append(key)
            if value and value != key:
                query_parts.append(value)

            query = ' '.join(query_parts)
            if not query or len(query) < 3:
                continue

            # Search for memories matching this pattern
            matching = self._search_memories(query, limit=10)

            if len(matching) < 2:
                continue

            # Top results are positive, bottom results are weak negatives
            for i, memory in enumerate(matching):
                if i < 3:
                    label = 1.0  # Top matches are relevant
                elif i < 6:
                    label = 0.5  # Middle matches are weakly relevant
                else:
                    label = 0.1  # Bottom matches are marginal

                records.append(self._build_record(
                    query=query,
                    memory=memory,
                    label=label,
                    source='pattern',
                ))

        return records

    def _generate_recency_based(self) -> List[dict]:
        """
        Strategy 4: Recency decay — for shared-topic queries, recent wins.

        Logic: For memories about the same topic, more recent memories
        should generally rank higher (fresher context, more current).
        Generates pairs where newer = positive, older = weak negative.
        """
        records = []

        # Get a sample of recent and old memories
        recent = self._get_recent_memories(limit=30)
        if len(recent) < 4:
            return records

        # Take pairs: for each recent memory's keywords, create a query
        # then the recent memory is positive and older memories are negative
        processed_queries: Set[str] = set()

        for memory in recent[:15]:
            keywords = self._extract_keywords(memory.get('content', ''))
            query = ' '.join(keywords) if keywords else ''
            if not query or query in processed_queries:
                continue
            processed_queries.add(query)

            # This recent memory is positive
            records.append(self._build_record(
                query=query,
                memory=memory,
                label=0.8,  # Good but not perfect (it's synthetic)
                source='recency_positive',
            ))

            # Find older memories about similar topic
            similar_old = self._search_memories(query, limit=5)
            for old_mem in similar_old:
                if old_mem['id'] == memory['id']:
                    continue
                # Older memories get lower label
                records.append(self._build_record(
                    query=query,
                    memory=old_mem,
                    label=0.3,
                    source='recency_negative',
                ))

        return records

    # ========================================================================
    # Record Building
    # ========================================================================

    def _build_record(
        self,
        query: str,
        memory: dict,
        label: float,
        source: str,
    ) -> dict:
        """
        Build a training record with features.

        For synthetic data, we use simplified context:
        - No tech preferences (unknown at bootstrap time)
        - No current project
        - No workflow phase
        Focus on measurable features: importance, recency, access_frequency.
        """
        # Set neutral context (no query-time info for synthetic data)
        # Context is already set externally or defaults to neutral
        features = self._feature_extractor.extract_features(memory, query)

        return {
            'query': query,
            'query_hash': hashlib.sha256(query.encode()).hexdigest()[:16],
            'memory_id': memory.get('id', 0),
            'label': label,
            'source': source,
            'features': features,
        }

    # ========================================================================
    # Model Training
    # ========================================================================

    def bootstrap_model(self) -> Optional[Dict[str, Any]]:
        """
        Generate synthetic data and train the bootstrap model.

        Steps:
            1. Generate synthetic training data
            2. Build feature matrix and label vectors
            3. Train LightGBM with aggressive regularization
            4. Save model to ~/.claude-memory/models/ranker.txt
            5. Record metadata in learning_db
            6. Return metadata

        Returns:
            Training metadata dict, or None if bootstrap not possible.
        """
        if not HAS_LIGHTGBM or not HAS_NUMPY:
            logger.warning("Bootstrap requires LightGBM and NumPy")
            return None

        tier = self.get_tier()
        if tier is None:
            logger.info("Not enough memories for bootstrap")
            return None

        config = BOOTSTRAP_CONFIG[tier]
        logger.info(
            "Starting bootstrap (tier=%s, target=%d samples)",
            tier, config['target_samples']
        )

        # Set neutral context for feature extraction
        self._feature_extractor.set_context()

        # Generate synthetic data
        records = self.generate_synthetic_training_data()
        if not records:
            logger.warning("No synthetic records generated")
            return None

        # Trim to target sample count if needed
        if len(records) > config['target_samples']:
            # Keep a diverse sample across sources
            records = self._diverse_sample(records, config['target_samples'])

        # Group by query_hash for LGBMRanker
        query_groups: Dict[str, List[dict]] = {}
        for record in records:
            qh = record['query_hash']
            if qh not in query_groups:
                query_groups[qh] = []
            query_groups[qh].append(record)

        # Filter: only keep groups with 2+ items
        query_groups = {
            qh: recs for qh, recs in query_groups.items()
            if len(recs) >= 2
        }

        if not query_groups:
            logger.warning("No valid query groups (need 2+ records per group)")
            return None

        # Build matrices
        all_features = []
        all_labels = []
        groups = []

        for qh, group_records in query_groups.items():
            group_size = 0
            for record in group_records:
                all_features.append(record['features'])
                all_labels.append(record['label'])
                group_size += 1
            groups.append(group_size)

        X = np.array(all_features, dtype=np.float64)
        y = np.array(all_labels, dtype=np.float64)
        total_samples = X.shape[0]

        if total_samples < 10:
            logger.warning("Too few samples after grouping: %d", total_samples)
            return None

        logger.info(
            "Training bootstrap model: %d samples, %d groups, tier=%s",
            total_samples, len(groups), tier
        )

        # Create LightGBM dataset
        train_dataset = lgb.Dataset(
            X, label=y, group=groups,
            feature_name=list(FEATURE_NAMES),
            free_raw_data=False,
        )

        # Use tiered n_estimators and max_depth
        params = dict(BOOTSTRAP_PARAMS)
        params['max_depth'] = config['max_depth']
        n_estimators = config['n_estimators']

        # Train
        try:
            booster = lgb.train(
                params,
                train_dataset,
                num_boost_round=n_estimators,
                valid_sets=[train_dataset],
                valid_names=['train'],
                callbacks=[lgb.log_evaluation(period=0)],  # Silent
            )
        except Exception as e:
            logger.error("Bootstrap training failed: %s", e)
            return None

        # Save model
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        try:
            booster.save_model(str(MODEL_PATH))
            logger.info("Bootstrap model saved to %s", MODEL_PATH)
        except Exception as e:
            logger.error("Failed to save bootstrap model: %s", e)
            return None

        # Extract NDCG@10 from training evaluation
        ndcg_at_10 = None
        try:
            eval_results = booster.eval_train(
                lgb.Dataset(X, label=y, group=groups)
            )
            for name, _dataset_name, value, _is_higher_better in eval_results:
                if 'ndcg@10' in name:
                    ndcg_at_10 = value
                    break
        except Exception:
            pass

        # Record metadata in learning_db
        model_version = f"bootstrap_{tier}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        ldb = self._get_learning_db()
        if ldb:
            try:
                ldb.record_model_training(
                    model_version=model_version,
                    training_samples=total_samples,
                    synthetic_samples=total_samples,
                    real_samples=0,
                    ndcg_at_10=ndcg_at_10,
                    model_path=str(MODEL_PATH),
                )
            except Exception as e:
                logger.warning("Failed to record bootstrap metadata: %s", e)

        metadata = {
            'model_version': model_version,
            'tier': tier,
            'training_samples': total_samples,
            'synthetic_samples': total_samples,
            'query_groups': len(groups),
            'n_estimators': n_estimators,
            'max_depth': config['max_depth'],
            'ndcg_at_10': ndcg_at_10,
            'model_path': str(MODEL_PATH),
            'source_breakdown': self._count_sources(records),
            'created_at': datetime.now().isoformat(),
        }
        logger.info("Bootstrap complete: %s", metadata)
        return metadata

    # ========================================================================
    # Memory Database Queries (READ-ONLY on memory.db)
    # ========================================================================

    def _get_memories_by_access(self, min_access: int = 5) -> List[dict]:
        """
        Fetch memories with access_count >= min_access from memory.db.

        These are memories the user keeps coming back to — strong positive signal.
        """
        return get_memories_by_access(self._memory_db, min_access)

    def _get_memories_by_importance(self, min_importance: int = 8) -> List[dict]:
        """
        Fetch memories with importance >= min_importance from memory.db.

        High importance = user explicitly rated these as valuable.
        """
        return get_memories_by_importance(self._memory_db, min_importance)

    def _get_recent_memories(self, limit: int = 30) -> List[dict]:
        """Fetch the N most recently created memories."""
        return get_recent_memories(self._memory_db, limit)

    def _get_learned_patterns(
        self,
        min_confidence: float = 0.7,
    ) -> List[dict]:
        """
        Fetch high-confidence identity_patterns from memory.db.

        These are patterns detected by pattern_learner.py (Layer 4) —
        tech preferences, coding style, terminology, etc.

        Returns empty list if identity_patterns table doesn't exist
        (backward compatible with pre-v2.3 databases).
        """
        return get_learned_patterns(self._memory_db, min_confidence)

    def _search_memories(self, query: str, limit: int = 20) -> List[dict]:
        """
        Simple FTS5 search in memory.db.

        Used to find memories matching synthetic query terms.
        This is a lightweight search — no TF-IDF, no HNSW, just FTS5.
        """
        return search_memories(self._memory_db, query, limit)

    def _find_negative_memories(
        self,
        anchor_memory: dict,
        exclude_ids: Optional[Set[int]] = None,
        limit: int = 2,
    ) -> List[dict]:
        """
        Find memories dissimilar to the anchor (for negative examples).

        Simple heuristic: pick memories from a different category or project.
        Falls back to random sample if no structured differences available.
        """
        return find_negative_memories(self._memory_db, anchor_memory, exclude_ids, limit)

    # ========================================================================
    # Text Processing
    # ========================================================================

    def _extract_keywords(self, content: str, top_n: int = 3) -> List[str]:
        """
        Extract meaningful keywords from memory content.

        Simple frequency-based extraction:
        1. Tokenize (alphanumeric words)
        2. Remove stopwords and short words
        3. Return top N by frequency

        No external NLP dependencies — just regex + counter.
        """
        return extract_keywords(content, top_n)

    # ========================================================================
    # Utility
    # ========================================================================

    def _diverse_sample(
        self,
        records: List[dict],
        target: int,
    ) -> List[dict]:
        """
        Sample records while maintaining source diversity.

        Takes proportional samples from each source strategy to ensure
        the training data isn't dominated by one strategy.
        """
        return diverse_sample(records, target)

    def _count_sources(self, records: List[dict]) -> Dict[str, int]:
        """Count records by source strategy."""
        return count_sources(records)


# ============================================================================
# Module-level convenience
# ============================================================================

def should_bootstrap(memory_db_path: Optional[Path] = None) -> bool:
    """Quick check if bootstrap is needed (creates temporary bootstrapper)."""
    try:
        bootstrapper = SyntheticBootstrapper(memory_db_path=memory_db_path)
        return bootstrapper.should_bootstrap()
    except Exception:
        return False


def run_bootstrap(
    memory_db_path: Optional[Path] = None,
    learning_db=None,
) -> Optional[Dict[str, Any]]:
    """Run bootstrap and return metadata (convenience function)."""
    try:
        bootstrapper = SyntheticBootstrapper(
            memory_db_path=memory_db_path,
            learning_db=learning_db,
        )
        if bootstrapper.should_bootstrap():
            return bootstrapper.bootstrap_model()
        return None
    except Exception as e:
        logger.error("Bootstrap failed: %s", e)
        return None
