#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 SuperLocalMemory (superlocalmemory.com)
"""
Constants and configuration values for MemoryStoreV2.

This module contains all module-level constants extracted from memory_store_v2.py
to reduce file size and improve maintainability.
"""

from pathlib import Path

# Database paths
MEMORY_DIR = Path.home() / ".claude-memory"
DB_PATH = MEMORY_DIR / "memory.db"
VECTORS_PATH = MEMORY_DIR / "vectors"

# Security: Input validation limits
MAX_CONTENT_SIZE = 1_000_000    # 1MB max content
MAX_SUMMARY_SIZE = 10_000       # 10KB max summary
MAX_TAG_LENGTH = 50             # 50 chars per tag
MAX_TAGS = 20                   # 20 tags max

# Creator Attribution Metadata (REQUIRED by MIT License)
# This data is embedded in the database creator_metadata table
CREATOR_METADATA = {
    'creator_name': 'Varun Pratap Bhardwaj',
    'creator_role': 'Solution Architect & Original Creator',
    'creator_github': 'varun369',
    'project_name': 'SuperLocalMemory V2',
    'project_url': 'https://github.com/varun369/SuperLocalMemoryV2',
    'license': 'MIT',
    'attribution_required': 'yes',
    'version': '2.5.0',
    'architecture_date': '2026-01-15',
    'release_date': '2026-02-07',
    'signature': 'VBPB-SLM-V2-2026-ARCHITECT',
    'verification_hash': 'sha256:c9f3d1a8b5e2f4c6d8a9b3e7f1c4d6a8b9c3e7f2d5a8c1b4e6f9d2a7c5b8e1'
}
