#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 SuperLocalMemory (superlocalmemory.com)
"""
Ranking utilities package for AdaptiveRanker.

Provides constants, helpers, and feature utilities extracted from
the main adaptive_ranker.py module for better maintainability.
"""

from .constants import (
    MODELS_DIR,
    MODEL_PATH,
    PHASE_THRESHOLDS,
    MIN_UNIQUE_QUERIES_FOR_ML,
    RULE_BOOST,
    TRAINING_PARAMS,
)
from .helpers import (
    calculate_rule_boost,
    prepare_training_data_internal,
)

__all__ = [
    'MODELS_DIR',
    'MODEL_PATH',
    'PHASE_THRESHOLDS',
    'MIN_UNIQUE_QUERIES_FOR_ML',
    'RULE_BOOST',
    'TRAINING_PARAMS',
    'calculate_rule_boost',
    'prepare_training_data_internal',
]
