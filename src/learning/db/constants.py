#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 SuperLocalMemory (superlocalmemory.com)
"""
Constants and defaults for learning.db.

This module contains database paths, configuration defaults, and other
constants used by the learning database system.
"""

from pathlib import Path

# Database paths
MEMORY_DIR = Path.home() / ".claude-memory"
LEARNING_DB_PATH = MEMORY_DIR / "learning.db"

# Default values
DEFAULT_PROFILE = "default"
DEFAULT_CONFIDENCE = 1.0
DEFAULT_LIMIT = 100

# Table names (for reference and testing)
TABLE_TRANSFERABLE_PATTERNS = "transferable_patterns"
TABLE_WORKFLOW_PATTERNS = "workflow_patterns"
TABLE_RANKING_FEEDBACK = "ranking_feedback"
TABLE_RANKING_MODELS = "ranking_models"
TABLE_SOURCE_QUALITY = "source_quality"
TABLE_ENGAGEMENT_METRICS = "engagement_metrics"
TABLE_ACTION_OUTCOMES = "action_outcomes"
TABLE_BEHAVIORAL_PATTERNS = "behavioral_patterns"
TABLE_CROSS_PROJECT_BEHAVIORS = "cross_project_behaviors"

# All table names for iteration
ALL_TABLES = [
    TABLE_TRANSFERABLE_PATTERNS,
    TABLE_WORKFLOW_PATTERNS,
    TABLE_RANKING_FEEDBACK,
    TABLE_RANKING_MODELS,
    TABLE_SOURCE_QUALITY,
    TABLE_ENGAGEMENT_METRICS,
    TABLE_ACTION_OUTCOMES,
    TABLE_BEHAVIORAL_PATTERNS,
    TABLE_CROSS_PROJECT_BEHAVIORS,
]
