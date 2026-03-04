#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 SuperLocalMemory (superlocalmemory.com)
"""
Learning System — Feature detection and graceful import.

This module detects available dependencies and exposes feature flags
that the rest of the system uses to enable/disable learning features.

Design principle: If ANY import fails, the entire learning system
degrades gracefully to v2.6 behavior. Core memory operations are
NEVER affected by learning system failures.

Dependencies (all optional):
    lightgbm>=4.0.0  — Learning-to-Rank re-ranker
    scipy>=1.9.0     — Statistical functions (temporal decay, KDE)
"""

import logging
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("superlocalmemory.learning")

# ============================================================================
# Feature Detection
# ============================================================================

# Check LightGBM availability (required for ML ranking)
try:
    import lightgbm  # noqa: F401
    HAS_LIGHTGBM = True
    LIGHTGBM_VERSION = lightgbm.__version__
except ImportError:
    HAS_LIGHTGBM = False
    LIGHTGBM_VERSION = None

# Check SciPy availability (required for statistical functions)
try:
    import scipy  # noqa: F401
    HAS_SCIPY = True
    SCIPY_VERSION = scipy.__version__
except ImportError:
    HAS_SCIPY = False
    SCIPY_VERSION = None

# Check scikit-learn availability (already in core, but verify)
try:
    import sklearn  # noqa: F401
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

# ============================================================================
# Composite Feature Flags
# ============================================================================

# ML ranking requires LightGBM
ML_RANKING_AVAILABLE = HAS_LIGHTGBM

# Full learning requires LightGBM + SciPy
FULL_LEARNING_AVAILABLE = HAS_LIGHTGBM and HAS_SCIPY

# Rule-based ranking works with zero optional deps
RULE_BASED_RANKING_AVAILABLE = True

# ============================================================================
# Paths
# ============================================================================

MEMORY_DIR = Path.home() / ".claude-memory"
LEARNING_DB_PATH = MEMORY_DIR / "learning.db"
MEMORY_DB_PATH = MEMORY_DIR / "memory.db"
MODELS_DIR = MEMORY_DIR / "models"

# ============================================================================
# Module-level lazy imports
# ============================================================================

# These are imported lazily to avoid circular imports and to allow
# individual module failures without breaking the whole system.

_learning_db = None
_adaptive_ranker = None
_feedback_collector = None
_engagement_tracker = None


def get_learning_db() -> Any:
    """Get or create the LearningDB singleton."""
    global _learning_db
    if _learning_db is None:
        try:
            from .learning_db import LearningDB
            _learning_db = LearningDB()
        except Exception as e:
            logger.warning("Failed to initialize LearningDB: %s", e)
            return None
    return _learning_db


def get_adaptive_ranker() -> Any:
    """Get or create the AdaptiveRanker singleton."""
    global _adaptive_ranker
    if _adaptive_ranker is None:
        try:
            from .adaptive_ranker import AdaptiveRanker
            _adaptive_ranker = AdaptiveRanker()
        except Exception as e:
            logger.warning("Failed to initialize AdaptiveRanker: %s", e)
            return None
    return _adaptive_ranker


def get_feedback_collector() -> Any:
    """Get or create the FeedbackCollector singleton."""
    global _feedback_collector
    if _feedback_collector is None:
        try:
            from .feedback_collector import FeedbackCollector
            _feedback_collector = FeedbackCollector()
        except Exception as e:
            logger.warning("Failed to initialize FeedbackCollector: %s", e)
            return None
    return _feedback_collector


def get_engagement_tracker() -> Any:
    """Get or create the EngagementTracker singleton."""
    global _engagement_tracker
    if _engagement_tracker is None:
        try:
            from .engagement_tracker import EngagementTracker
            _engagement_tracker = EngagementTracker()
        except Exception as e:
            logger.warning("Failed to initialize EngagementTracker: %s", e)
            return None
    return _engagement_tracker


def get_status() -> dict:
    """Return learning system status for diagnostics."""
    status = {
        "learning_available": FULL_LEARNING_AVAILABLE,
        "ml_ranking_available": ML_RANKING_AVAILABLE,
        "rule_based_available": RULE_BASED_RANKING_AVAILABLE,
        "dependencies": {
            "lightgbm": {
                "installed": HAS_LIGHTGBM,
                "version": LIGHTGBM_VERSION,
            },
            "scipy": {
                "installed": HAS_SCIPY,
                "version": SCIPY_VERSION,
            },
            "sklearn": {
                "installed": HAS_SKLEARN,
            },
        },
        "paths": {
            "learning_db": str(LEARNING_DB_PATH),
            "models_dir": str(MODELS_DIR),
        },
    }

    # Add learning DB stats if available
    ldb = get_learning_db()
    if ldb:
        try:
            status["learning_db_stats"] = ldb.get_stats()
        except Exception:
            status["learning_db_stats"] = None

    # v2.8 engine status (lifecycle, behavioral, compliance)
    v28_engines = {}
    try:
        from lifecycle import get_status as lifecycle_status
        ls = lifecycle_status()
        v28_engines["lifecycle"] = {"available": ls.get("lifecycle_available", False), "detail": ls}
    except (ImportError, Exception):
        v28_engines["lifecycle"] = {"available": False, "detail": None}

    try:
        from behavioral import get_status as behavioral_status
        bs = behavioral_status()
        v28_engines["behavioral"] = {"available": bs.get("behavioral_available", False), "detail": bs}
    except (ImportError, Exception):
        v28_engines["behavioral"] = {"available": False, "detail": None}

    try:
        from compliance import get_status as compliance_status
        cs = compliance_status()
        v28_engines["compliance"] = {"available": cs.get("compliance_available", False), "detail": cs}
    except (ImportError, Exception):
        v28_engines["compliance"] = {"available": False, "detail": None}

    status["v28_engines"] = v28_engines

    return status


# Log feature availability on import
if FULL_LEARNING_AVAILABLE:
    logger.info(
        "Learning system available: LightGBM %s, SciPy %s",
        LIGHTGBM_VERSION, SCIPY_VERSION
    )
elif ML_RANKING_AVAILABLE:
    logger.info(
        "Partial learning: LightGBM %s available, SciPy missing",
        LIGHTGBM_VERSION
    )
else:
    logger.info(
        "Learning dependencies not installed. "
        "Install with: pip3 install -r requirements-learning.txt"
    )
