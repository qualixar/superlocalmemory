#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 SuperLocalMemory (superlocalmemory.com)
"""
Compression sub-package for SuperLocalMemory.
Provides tier-based progressive summarization and archival.

Note: Uses relative imports to avoid collision with Python stdlib.
"""

from .config import CompressionConfig
from .tier_classifier import TierClassifier
from .tier2_compressor import Tier2Compressor
from .tier3_compressor import Tier3Compressor
from .cold_storage import ColdStorageManager
from .orchestrator import CompressionOrchestrator

__all__ = [
    'CompressionConfig',
    'TierClassifier',
    'Tier2Compressor',
    'Tier3Compressor',
    'ColdStorageManager',
    'CompressionOrchestrator',
]
