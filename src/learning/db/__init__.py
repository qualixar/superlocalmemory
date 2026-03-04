#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 SuperLocalMemory (superlocalmemory.com)
"""
Learning database utilities and schema management.

This package contains extracted modules from learning_db.py to improve
code organization and maintainability.
"""

from .constants import (
    MEMORY_DIR,
    LEARNING_DB_PATH,
    DEFAULT_PROFILE,
    DEFAULT_CONFIDENCE,
    DEFAULT_LIMIT,
    ALL_TABLES,
)

from .schema import (
    initialize_schema,
    create_all_tables,
    add_profile_columns,
    create_indexes,
)

__all__ = [
    # Constants
    "MEMORY_DIR",
    "LEARNING_DB_PATH",
    "DEFAULT_PROFILE",
    "DEFAULT_CONFIDENCE",
    "DEFAULT_LIMIT",
    "ALL_TABLES",
    # Schema functions
    "initialize_schema",
    "create_all_tables",
    "add_profile_columns",
    "create_indexes",
]
