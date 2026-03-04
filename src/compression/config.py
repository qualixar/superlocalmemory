#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 SuperLocalMemory (superlocalmemory.com)
"""
Compression configuration management.
Handles loading and saving compression settings from config.json.
"""

import json
from pathlib import Path
from typing import Dict, Any


MEMORY_DIR = Path.home() / ".claude-memory"
CONFIG_PATH = MEMORY_DIR / "config.json"


class CompressionConfig:
    """Configuration for compression behavior."""

    def __init__(self):
        self.config = self._load_config()
        self.compression_settings = self.config.get('compression', {})

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from config.json."""
        if CONFIG_PATH.exists():
            with open(CONFIG_PATH, 'r') as f:
                return json.load(f)
        return {}

    def save(self):
        """Save configuration back to config.json."""
        with open(CONFIG_PATH, 'w') as f:
            json.dump(self.config, f, indent=2)

    @property
    def enabled(self) -> bool:
        return self.compression_settings.get('enabled', True)

    @property
    def tier2_threshold_days(self) -> int:
        return self.compression_settings.get('tier2_threshold_days', 30)

    @property
    def tier3_threshold_days(self) -> int:
        return self.compression_settings.get('tier3_threshold_days', 90)

    @property
    def cold_storage_threshold_days(self) -> int:
        return self.compression_settings.get('cold_storage_threshold_days', 365)

    @property
    def preserve_high_importance(self) -> bool:
        return self.compression_settings.get('preserve_high_importance', True)

    @property
    def preserve_recently_accessed(self) -> bool:
        return self.compression_settings.get('preserve_recently_accessed', True)

    def initialize_defaults(self):
        """Initialize compression settings in config if not present."""
        if 'compression' not in self.config:
            self.config['compression'] = {
                'enabled': True,
                'tier2_threshold_days': 30,
                'tier3_threshold_days': 90,
                'cold_storage_threshold_days': 365,
                'preserve_high_importance': True,
                'preserve_recently_accessed': True
            }
            self.save()
