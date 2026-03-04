#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 SuperLocalMemory (superlocalmemory.com)
"""
Bootstrap constants and configuration.

All constant values, configuration dicts, and static data used
by SyntheticBootstrapper are defined here.
"""

from pathlib import Path

# ============================================================================
# Paths
# ============================================================================

MEMORY_DB_PATH = Path.home() / ".claude-memory" / "memory.db"
MODELS_DIR = Path.home() / ".claude-memory" / "models"
MODEL_PATH = MODELS_DIR / "ranker.txt"

# ============================================================================
# Bootstrap Configuration
# ============================================================================

# Minimum memories needed before bootstrap makes sense
MIN_MEMORIES_FOR_BOOTSTRAP = 50

# Tiered config — bootstrap model complexity scales with data size
BOOTSTRAP_CONFIG = {
    'small': {
        'min_memories': 50,
        'max_memories': 499,
        'target_samples': 200,
        'n_estimators': 30,
        'max_depth': 3,
    },
    'medium': {
        'min_memories': 500,
        'max_memories': 4999,
        'target_samples': 1000,
        'n_estimators': 50,
        'max_depth': 4,
    },
    'large': {
        'min_memories': 5000,
        'max_memories': float('inf'),
        'target_samples': 2000,
        'n_estimators': 100,
        'max_depth': 6,
    },
}

# ============================================================================
# LightGBM Parameters
# ============================================================================

# LightGBM bootstrap parameters — MORE aggressive regularization than
# real training because synthetic data has systematic biases
BOOTSTRAP_PARAMS = {
    'objective': 'lambdarank',
    'metric': 'ndcg',
    'ndcg_eval_at': [5, 10],
    'learning_rate': 0.1,
    'num_leaves': 8,
    'max_depth': 3,
    'min_child_samples': 5,
    'subsample': 0.7,
    'reg_alpha': 0.5,
    'reg_lambda': 2.0,
    'boosting_type': 'dart',
    'verbose': -1,
}

# ============================================================================
# Text Processing
# ============================================================================

# English stopwords for keyword extraction (no external deps)
STOPWORDS = frozenset({
    'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
    'of', 'with', 'by', 'from', 'is', 'it', 'this', 'that', 'was', 'are',
    'be', 'has', 'have', 'had', 'do', 'does', 'did', 'will', 'would',
    'could', 'should', 'may', 'might', 'can', 'not', 'no', 'if', 'then',
    'so', 'as', 'up', 'out', 'about', 'into', 'over', 'after', 'before',
    'when', 'where', 'how', 'what', 'which', 'who', 'whom', 'why',
    'all', 'each', 'every', 'both', 'few', 'more', 'most', 'other',
    'some', 'such', 'than', 'too', 'very', 'just', 'also', 'now',
    'here', 'there', 'use', 'used', 'using', 'make', 'made',
    'need', 'needed', 'get', 'got', 'set', 'new', 'old', 'one', 'two',
})

# Minimum word length for keyword extraction
MIN_KEYWORD_LENGTH = 3
