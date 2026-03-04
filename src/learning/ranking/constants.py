#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 SuperLocalMemory (superlocalmemory.com)
"""
Constants for AdaptiveRanker.

Includes phase thresholds, rule-based boost multipliers, and LightGBM
training parameters.
"""

from pathlib import Path

# ============================================================================
# Paths
# ============================================================================

MODELS_DIR = Path.home() / ".claude-memory" / "models"
MODEL_PATH = MODELS_DIR / "ranker.txt"

# ============================================================================
# Phase Thresholds
# ============================================================================

# Phase thresholds — how many feedback signals to trigger each phase
PHASE_THRESHOLDS = {
    'baseline': 0,       # 0 feedback samples -> no re-ranking
    'rule_based': 20,    # 20+ feedback -> rule-based boosting
    'ml_model': 200,     # 200+ feedback across 50+ unique queries -> ML
}

# Minimum unique queries required for ML phase (prevents overfitting
# to a small number of repeated queries)
MIN_UNIQUE_QUERIES_FOR_ML = 50

# ============================================================================
# Rule-Based Boost Multipliers (Phase 1)
# ============================================================================

# These are conservative — they nudge the ranking without flipping order
RULE_BOOST = {
    'tech_match_strong': 1.3,      # Memory matches 2+ preferred techs
    'tech_match_weak': 1.1,        # Memory matches 1 preferred tech
    'project_match': 1.5,          # Memory from current project
    'project_unknown': 1.0,        # No project context — no boost
    'project_mismatch': 0.9,       # Memory from different project
    'source_quality_high': 1.2,    # Source quality > 0.7
    'source_quality_low': 0.85,    # Source quality < 0.3
    'recency_boost_max': 1.2,      # Recent memory (< 7 days)
    'recency_penalty_max': 0.8,    # Old memory (> 365 days)
    'high_importance': 1.15,       # Importance >= 8
    'high_access': 1.1,            # Accessed 5+ times
    # v2.8: Lifecycle + behavioral boosts
    'lifecycle_active': 1.0,
    'lifecycle_warm': 0.85,
    'lifecycle_cold': 0.6,
    'outcome_success_high': 1.3,
    'outcome_failure_high': 0.7,
    'behavioral_match_strong': 1.25,
    'cross_project_boost': 1.15,
    'high_trust_creator': 1.1,
    'low_trust_creator': 0.8,
}

# ============================================================================
# LightGBM Training Parameters
# ============================================================================

# LightGBM training parameters — tuned for small, personal datasets
# Aggressive regularization prevents overfitting on < 10K samples
TRAINING_PARAMS = {
    'objective': 'lambdarank',
    'metric': 'ndcg',
    'ndcg_eval_at': [5, 10],
    'learning_rate': 0.05,
    'num_leaves': 16,
    'max_depth': 4,
    'min_child_samples': 10,
    'subsample': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'boosting_type': 'dart',
    'n_estimators': 50,
    'verbose': -1,
}
