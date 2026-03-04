#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 SuperLocalMemory (superlocalmemory.com)
"""
Bootstrap utilities package.

Re-exports all constants, functions, and utilities used by SyntheticBootstrapper.
"""

# Constants
from .constants import (
    MEMORY_DB_PATH,
    MODELS_DIR,
    MODEL_PATH,
    MIN_MEMORIES_FOR_BOOTSTRAP,
    BOOTSTRAP_CONFIG,
    BOOTSTRAP_PARAMS,
    STOPWORDS,
    MIN_KEYWORD_LENGTH,
)

# Text utilities
from .text_utils import (
    extract_keywords,
    clean_fts_query,
)

# Database queries
from .db_queries import (
    get_memory_count,
    get_memories_by_access,
    get_memories_by_importance,
    get_recent_memories,
    get_learned_patterns,
    search_memories,
    find_negative_memories,
)

# Sampling utilities
from .sampling import (
    diverse_sample,
    count_sources,
)

__all__ = [
    # Constants
    'MEMORY_DB_PATH',
    'MODELS_DIR',
    'MODEL_PATH',
    'MIN_MEMORIES_FOR_BOOTSTRAP',
    'BOOTSTRAP_CONFIG',
    'BOOTSTRAP_PARAMS',
    'STOPWORDS',
    'MIN_KEYWORD_LENGTH',
    # Text utilities
    'extract_keywords',
    'clean_fts_query',
    # Database queries
    'get_memory_count',
    'get_memories_by_access',
    'get_memories_by_importance',
    'get_recent_memories',
    'get_learned_patterns',
    'search_memories',
    'find_negative_memories',
    # Sampling
    'diverse_sample',
    'count_sources',
]
