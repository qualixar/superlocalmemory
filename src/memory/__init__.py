#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 SuperLocalMemory (superlocalmemory.com)
"""
Memory package - Constants, schema, and utilities for MemoryStoreV2.

This package contains extracted utilities from memory_store_v2.py to keep
the main class focused and under the 800-line target.
"""

from .constants import (
    MEMORY_DIR,
    DB_PATH,
    VECTORS_PATH,
    MAX_CONTENT_SIZE,
    MAX_SUMMARY_SIZE,
    MAX_TAG_LENGTH,
    MAX_TAGS,
    CREATOR_METADATA,
)

from .helpers import format_content
from .cli import run_cli

__all__ = [
    'MEMORY_DIR',
    'DB_PATH',
    'VECTORS_PATH',
    'MAX_CONTENT_SIZE',
    'MAX_SUMMARY_SIZE',
    'MAX_TAG_LENGTH',
    'MAX_TAGS',
    'CREATOR_METADATA',
    'format_content',
    'run_cli',
]
