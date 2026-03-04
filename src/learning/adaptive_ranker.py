#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 SuperLocalMemory (superlocalmemory.com)
"""
AdaptiveRanker — Three-phase adaptive re-ranking engine.

This is the core ranking engine for v2.7 "Your AI Learns You". It sits
between the existing search methods (FTS5 + TF-IDF + HNSW) and the final
result list, re-ordering candidates based on learned user preferences.

Three Phases (progressive adaptation):

    Phase 0 — Baseline (< 20 feedback signals):
        Pure v2.6 behavior. No re-ranking applied. Results returned as-is
        from the existing search pipeline. Zero risk of degradation.

    Phase 1 — Rule-Based (20-199 signals):
        Applies learned-pattern boosting to search results. Uses feature
        extraction to compute boost multipliers for tech match, project
        match, recency, and source quality. Deterministic and interpretable.

    Phase 2 — ML Model (200+ signals across 50+ unique queries):
        LightGBM LambdaRank re-ranker. Trained on real feedback data
        (and optionally bootstrapped from synthetic data). Produces ML
        scores that replace the original ranking order.

Design Principles:
    - LightGBM is OPTIONAL. If not installed, falls back to rule-based.
    - Any exception in re-ranking falls back to original v2.6 results.
    - Model is loaded lazily and cached in memory.
    - Training is explicit (called by user or scheduled), never implicit.
    - Original scores are preserved as 'base_score' for diagnostics.

Research Backing:
    - eKNOW 2025: BM25 -> re-ranker pipeline for personal collections
    - MACLA (arXiv:2512.18950): Bayesian confidence scoring
    - FCS LREC 2024: Cold-start mitigation via synthetic bootstrap
"""

import logging
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# LightGBM is OPTIONAL — graceful fallback to rule-based ranking
try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    lgb = None
    HAS_LIGHTGBM = False

# NumPy is used for feature matrix construction (comes with sklearn)
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    np = None
    HAS_NUMPY = False

from .feature_extractor import FeatureExtractor, FEATURE_NAMES, NUM_FEATURES

logger = logging.getLogger("superlocalmemory.learning.adaptive_ranker")

# Import constants and helpers from ranking subpackage
from .ranking import (
    MODELS_DIR,
    MODEL_PATH,
    PHASE_THRESHOLDS,
    MIN_UNIQUE_QUERIES_FOR_ML,
    RULE_BOOST,
    TRAINING_PARAMS,
    calculate_rule_boost,
    prepare_training_data_internal,
)


class AdaptiveRanker:
    """
    Three-phase adaptive re-ranking engine.

    Usage (called by memory_store_v2.search or mcp_server recall):
        ranker = AdaptiveRanker()
        results = ranker.rerank(search_results, query, context={
            'tech_preferences': {...},
            'current_project': 'MyProject',
            'source_scores': {...},
            'workflow_phase': 'testing',
        })

    The caller wraps this in try/except — any exception here causes
    fallback to original v2.6 results. Zero risk of degradation.
    """

    PHASE_THRESHOLDS = PHASE_THRESHOLDS
    MODEL_PATH = MODEL_PATH

    def __init__(self, learning_db=None):
        """
        Initialize AdaptiveRanker.

        Args:
            learning_db: Optional LearningDB instance. If None, imports
                         and creates one lazily.
        """
        self._learning_db = learning_db
        self._feature_extractor = FeatureExtractor()
        self._model = None  # Loaded lazily on first ML rerank
        self._model_load_attempted = False
        self._lock = threading.Lock()

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
    # Phase Detection
    # ========================================================================

    def get_phase(self) -> str:
        """
        Determine the current ranking phase based on feedback data.

        Returns:
            'baseline' — Not enough data for personalization
            'rule_based' — Enough data for rule-based boosting
            'ml_model' — Enough data for ML ranking (if LightGBM available)
        """
        ldb = self._get_learning_db()
        if ldb is None:
            return 'baseline'

        try:
            feedback_count = ldb.get_feedback_count()
            unique_queries = ldb.get_unique_query_count()
        except Exception as e:
            logger.warning("Failed to check feedback counts: %s", e)
            return 'baseline'

        # Phase 2: ML model — requires enough data AND LightGBM AND numpy
        if (
            feedback_count >= PHASE_THRESHOLDS['ml_model']
            and unique_queries >= MIN_UNIQUE_QUERIES_FOR_ML
            and HAS_LIGHTGBM
            and HAS_NUMPY
        ):
            return 'ml_model'

        # Phase 1: Rule-based — just needs minimum feedback
        if feedback_count >= PHASE_THRESHOLDS['rule_based']:
            return 'rule_based'

        # Phase 0: Not enough data yet
        return 'baseline'

    def get_phase_info(self) -> Dict[str, Any]:
        """
        Return detailed phase information for diagnostics.

        Returns:
            Dict with phase, feedback_count, unique_queries, thresholds,
            model_loaded, lightgbm_available.
        """
        ldb = self._get_learning_db()
        feedback_count = 0
        unique_queries = 0

        if ldb is not None:
            try:
                feedback_count = ldb.get_feedback_count()
                unique_queries = ldb.get_unique_query_count()
            except Exception:
                pass

        phase = self.get_phase()

        return {
            'phase': phase,
            'feedback_count': feedback_count,
            'unique_queries': unique_queries,
            'thresholds': dict(PHASE_THRESHOLDS),
            'min_unique_queries_for_ml': MIN_UNIQUE_QUERIES_FOR_ML,
            'model_loaded': self._model is not None,
            'model_path_exists': MODEL_PATH.exists(),
            'lightgbm_available': HAS_LIGHTGBM,
            'numpy_available': HAS_NUMPY,
        }

    # ========================================================================
    # Main Re-ranking Entry Point
    # ========================================================================

    def rerank(
        self,
        results: List[dict],
        query: str,
        context: Optional[dict] = None,
    ) -> List[dict]:
        """
        Re-rank search results based on learned user preferences.

        This is the main entry point, called after the search pipeline
        produces initial results. It determines the current phase and
        routes to the appropriate ranking strategy.

        Args:
            results: List of memory dicts from search (with 'score' field).
            query: The recall query string.
            context: Optional context dict with:
                     - tech_preferences: Dict[str, dict] — user's tech prefs
                     - current_project: str — active project name
                     - source_scores: Dict[str, float] — source quality scores
                     - workflow_phase: str — current workflow phase

        Returns:
            Re-ranked list of memory dicts. Each memory gets:
            - 'base_score': Original score from search pipeline
            - 'ranking_phase': Which phase was used
            - 'score': Updated score (may differ from base_score)

        CRITICAL: The caller wraps this in try/except. Any exception
        causes fallback to original v2.6 results. This method must
        never corrupt the results list.
        """
        if not results:
            return results

        # Short-circuit: don't re-rank trivially small result sets
        if len(results) <= 1:
            for r in results:
                r['base_score'] = r.get('score', 0.0)
                r['ranking_phase'] = 'baseline'
            return results

        context = context or {}

        # Fetch signal stats for features [10-11] (v2.7.4)
        signal_stats = {}
        ldb = self._get_learning_db()
        if ldb:
            try:
                memory_ids = [r.get('id') for r in results if r.get('id')]
                if memory_ids:
                    signal_stats = ldb.get_signal_stats_for_memories(memory_ids)
            except Exception:
                pass  # Signal stats failure is not critical

        # Set up feature extraction context (once per query)
        self._feature_extractor.set_context(
            source_scores=context.get('source_scores'),
            tech_preferences=context.get('tech_preferences'),
            current_project=context.get('current_project'),
            workflow_phase=context.get('workflow_phase'),
            signal_stats=signal_stats,
        )

        # Determine phase and route
        phase = self.get_phase()

        if phase == 'baseline':
            # Phase 0: No re-ranking — preserve original order
            for r in results:
                r['base_score'] = r.get('score', 0.0)
                r['ranking_phase'] = 'baseline'
            return results

        elif phase == 'rule_based':
            return self._rerank_rule_based(results, query, context)

        elif phase == 'ml_model':
            # Try ML first, fall back to rule-based if model fails
            try:
                return self._rerank_ml(results, query, context)
            except Exception as e:
                logger.warning(
                    "ML re-ranking failed, falling back to rule-based: %s", e
                )
                return self._rerank_rule_based(results, query, context)

        # Defensive: unknown phase -> no re-ranking
        for r in results:
            r['base_score'] = r.get('score', 0.0)
            r['ranking_phase'] = 'unknown'
        return results

    # ========================================================================
    # Phase 1: Rule-Based Re-ranking
    # ========================================================================

    def _rerank_rule_based(
        self,
        results: List[dict],
        query: str,
        context: dict,
    ) -> List[dict]:
        """
        Phase 1: Apply rule-based boosting using extracted features.

        Each result's score is multiplied by boost factors derived from
        feature values. The boosts are conservative — they nudge the
        ranking order without dramatically flipping results.
        """
        feature_vectors = self._feature_extractor.extract_batch(results, query)

        for i, result in enumerate(results):
            base_score = result.get('score', 0.0)
            result['base_score'] = base_score
            result['ranking_phase'] = 'rule_based'

            if i >= len(feature_vectors):
                continue

            features = feature_vectors[i]
            boost = calculate_rule_boost(features)

            # Apply boost to score
            result['score'] = base_score * boost

        # Re-sort by boosted score (highest first)
        results.sort(key=lambda x: x.get('score', 0.0), reverse=True)
        return results

    # ========================================================================
    # Phase 2: ML Re-ranking (LightGBM)
    # ========================================================================

    def _rerank_ml(
        self,
        results: List[dict],
        query: str,
        context: dict,
    ) -> List[dict]:
        """
        Phase 2: LightGBM LambdaRank re-ranking.

        Extracts features, runs the trained model, and sorts by ML scores.
        Preserves original score as 'base_score' and adds 'ml_score'.
        """
        if not HAS_LIGHTGBM or not HAS_NUMPY:
            raise RuntimeError("LightGBM or NumPy not available for ML ranking")

        # Load model if not cached
        model = self._load_model()
        if model is None:
            raise RuntimeError("No trained ranking model available")

        # Extract features
        feature_vectors = self._feature_extractor.extract_batch(results, query)
        if not feature_vectors:
            raise ValueError("Feature extraction returned empty results")

        # Build feature matrix
        X = np.array(feature_vectors, dtype=np.float64)

        # Validate shape
        if X.shape[1] != NUM_FEATURES:
            raise ValueError(
                f"Feature dimension mismatch: expected {NUM_FEATURES}, "
                f"got {X.shape[1]}"
            )

        # Predict scores
        ml_scores = model.predict(X)

        # Annotate results with ML scores
        for i, result in enumerate(results):
            result['base_score'] = result.get('score', 0.0)
            result['ranking_phase'] = 'ml_model'
            if i < len(ml_scores):
                result['ml_score'] = float(ml_scores[i])
                result['score'] = float(ml_scores[i])
            else:
                result['ml_score'] = 0.0

        # Re-sort by ML score (highest first)
        results.sort(key=lambda x: x.get('score', 0.0), reverse=True)
        return results

    # ========================================================================
    # Model Management
    # ========================================================================

    def _load_model(self):
        """
        Load LightGBM model from disk (lazy, cached).

        Returns:
            lgb.Booster instance or None if unavailable.
        """
        # Return cached model if already loaded
        if self._model is not None:
            return self._model

        # Avoid repeated failed load attempts
        if self._model_load_attempted:
            return None

        with self._lock:
            # Double-check after acquiring lock
            if self._model is not None:
                return self._model
            if self._model_load_attempted:
                return None

            self._model_load_attempted = True

            if not HAS_LIGHTGBM:
                logger.info("LightGBM not installed — ML ranking unavailable")
                return None

            if not MODEL_PATH.exists():
                logger.info(
                    "No ranking model at %s — ML ranking unavailable",
                    MODEL_PATH
                )
                return None

            try:
                model = lgb.Booster(model_file=str(MODEL_PATH))

                # v2.7.4: Check for feature dimension mismatch (10→12 upgrade)
                model_num_features = model.num_feature()
                if model_num_features != NUM_FEATURES:
                    logger.info(
                        "Feature mismatch: model has %d features, expected %d. "
                        "Triggering auto-retrain in background.",
                        model_num_features, NUM_FEATURES,
                    )
                    # Delete old model and trigger re-bootstrap
                    MODEL_PATH.unlink(missing_ok=True)
                    self._trigger_retrain_background()
                    return None

                self._model = model
                logger.info("Loaded ranking model from %s", MODEL_PATH)
                return self._model
            except Exception as e:
                logger.warning("Failed to load ranking model: %s", e)
                return None

    def _trigger_retrain_background(self):
        """Trigger model re-bootstrap in a background thread (v2.7.4)."""
        try:
            import threading

            def _retrain():
                try:
                    from .synthetic_bootstrap import SyntheticBootstrapper
                    bootstrapper = SyntheticBootstrapper()
                    if bootstrapper.should_bootstrap():
                        result = bootstrapper.bootstrap_model()
                        if result:
                            logger.info(
                                "Auto-retrain complete with %d-feature model",
                                NUM_FEATURES,
                            )
                            # Reload the new model
                            with self._lock:
                                self._model = None
                                self._model_load_attempted = False
                except Exception as e:
                    logger.warning("Auto-retrain failed: %s", e)

            thread = threading.Thread(target=_retrain, daemon=True)
            thread.start()
        except Exception:
            pass

    def reload_model(self):
        """
        Force reload of the ranking model from disk.

        Call this after training a new model to pick up the updated weights.
        """
        with self._lock:
            self._model = None
            self._model_load_attempted = False
        # Trigger fresh load
        return self._load_model()

    # ========================================================================
    # Model Training
    # ========================================================================

    def train(self, force: bool = False) -> Optional[Dict[str, Any]]:
        """
        Train or retrain the LightGBM ranking model.

        Uses continued training (init_model) if a model already exists,
        incorporating new feedback data incrementally.

        Args:
            force: If True, train even if below ML threshold.
                   Useful for synthetic bootstrap training.

        Returns:
            Training metadata dict, or None if training not possible.
            Metadata includes: model_version, training_samples, ndcg_at_10,
            model_path, created_at.
        """
        if not HAS_LIGHTGBM or not HAS_NUMPY:
            logger.warning(
                "Cannot train: LightGBM=%s, NumPy=%s",
                HAS_LIGHTGBM, HAS_NUMPY
            )
            return None

        ldb = self._get_learning_db()
        if ldb is None:
            logger.warning("Cannot train: LearningDB unavailable")
            return None

        # Check if we have enough data (unless forced)
        if not force:
            feedback_count = ldb.get_feedback_count()
            unique_queries = ldb.get_unique_query_count()
            if (
                feedback_count < PHASE_THRESHOLDS['ml_model']
                or unique_queries < MIN_UNIQUE_QUERIES_FOR_ML
            ):
                logger.info(
                    "Insufficient data for training: %d feedback / %d queries "
                    "(need %d / %d)",
                    feedback_count, unique_queries,
                    PHASE_THRESHOLDS['ml_model'], MIN_UNIQUE_QUERIES_FOR_ML,
                )
                return None

        # Prepare training data
        training_data = self._prepare_training_data()
        if training_data is None:
            logger.warning("No usable training data available")
            return None

        X, y, groups = training_data
        total_samples = X.shape[0]

        if total_samples < 10:
            logger.warning("Too few training samples: %d", total_samples)
            return None

        logger.info(
            "Training ranking model: %d samples, %d groups",
            total_samples, len(groups)
        )

        # Create LightGBM dataset
        train_dataset = lgb.Dataset(
            X, label=y, group=groups,
            feature_name=list(FEATURE_NAMES),
            free_raw_data=False,
        )

        # Training parameters
        params = dict(TRAINING_PARAMS)
        n_estimators = params.pop('n_estimators', 50)

        # Check for existing model (continued training)
        init_model = None
        if MODEL_PATH.exists():
            try:
                init_model = lgb.Booster(model_file=str(MODEL_PATH))
                logger.info("Continuing training from existing model")
            except Exception:
                logger.info("Starting fresh training (existing model unreadable)")
                init_model = None

        # Train
        try:
            booster = lgb.train(
                params,
                train_dataset,
                num_boost_round=n_estimators,
                init_model=init_model,
                valid_sets=[train_dataset],
                valid_names=['train'],
                callbacks=[lgb.log_evaluation(period=0)],  # Silent training
            )
        except Exception as e:
            logger.error("LightGBM training failed: %s", e)
            return None

        # Save model
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        try:
            booster.save_model(str(MODEL_PATH))
            logger.info("Ranking model saved to %s", MODEL_PATH)
        except Exception as e:
            logger.error("Failed to save ranking model: %s", e)
            return None

        # Extract NDCG@10 from training evaluation (if available)
        ndcg_at_10 = None
        try:
            eval_results = booster.eval_train(lgb.Dataset(X, label=y, group=groups))
            for name, _dataset_name, value, _is_higher_better in eval_results:
                if 'ndcg@10' in name:
                    ndcg_at_10 = value
                    break
        except Exception:
            pass

        # Record metadata in learning_db
        model_version = datetime.now().strftime("v%Y%m%d_%H%M%S")
        try:
            ldb.record_model_training(
                model_version=model_version,
                training_samples=total_samples,
                real_samples=total_samples,
                synthetic_samples=0,
                ndcg_at_10=ndcg_at_10,
                model_path=str(MODEL_PATH),
            )
        except Exception as e:
            logger.warning("Failed to record training metadata: %s", e)

        # Reload model into cache
        self.reload_model()

        metadata = {
            'model_version': model_version,
            'training_samples': total_samples,
            'query_groups': len(groups),
            'n_estimators': n_estimators,
            'ndcg_at_10': ndcg_at_10,
            'model_path': str(MODEL_PATH),
            'continued_from': init_model is not None,
            'created_at': datetime.now().isoformat(),
        }
        logger.info("Training complete: %s", metadata)
        return metadata

    def _prepare_training_data(self) -> Optional[tuple]:
        """
        Prepare training data from feedback records.

        For each unique query (grouped by query_hash):
            - Fetch all feedback entries for that query
            - Look up the corresponding memory from memory.db
            - Extract features for each memory
            - Use signal_value as the relevance label

        Returns:
            Tuple of (X, y, groups) for LGBMRanker, or None if insufficient.
            X: numpy array (n_samples, NUM_FEATURES)
            y: numpy array (n_samples,) — relevance labels
            groups: list of ints — samples per query group
        """
        ldb = self._get_learning_db()
        if ldb is None:
            return None

        feedback = ldb.get_feedback_for_training()
        if not feedback:
            return None

        return prepare_training_data_internal(feedback, self._feature_extractor)


# ============================================================================
# Module-level convenience
# ============================================================================

def get_phase() -> str:
    """Quick check of current ranking phase (creates temporary ranker)."""
    try:
        ranker = AdaptiveRanker()
        return ranker.get_phase()
    except Exception:
        return 'baseline'
