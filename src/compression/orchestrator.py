#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 SuperLocalMemory (superlocalmemory.com)
"""
Compression orchestrator.
Coordinates classification, compression, and archival operations.
"""

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

from compression.config import CompressionConfig
from compression.tier_classifier import TierClassifier
from compression.tier2_compressor import Tier2Compressor
from compression.tier3_compressor import Tier3Compressor
from compression.cold_storage import ColdStorageManager


MEMORY_DIR = Path.home() / ".claude-memory"
DB_PATH = MEMORY_DIR / "memory.db"


class CompressionOrchestrator:
    """Main orchestrator for compression operations."""

    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self.config = CompressionConfig()
        self.classifier = TierClassifier(db_path)
        self.tier2_compressor = Tier2Compressor(db_path)
        self.tier3_compressor = Tier3Compressor(db_path)
        self.cold_storage = ColdStorageManager(db_path)

    def run_full_compression(self) -> Dict[str, Any]:
        """
        Run full compression cycle: classify, compress, and archive.

        Returns:
            Statistics about compression operation
        """
        if not self.config.enabled:
            return {'status': 'disabled', 'message': 'Compression is disabled in config'}

        stats = {
            'started_at': datetime.now().isoformat(),
            'tier_updates': 0,
            'tier2_compressed': 0,
            'tier3_compressed': 0,
            'cold_stored': 0,
            'errors': []
        }

        try:
            # Step 1: Classify memories into tiers
            tier_updates = self.classifier.classify_memories()
            stats['tier_updates'] = len(tier_updates)

            # Step 2: Compress Tier 2 memories
            stats['tier2_compressed'] = self.tier2_compressor.compress_all_tier2()

            # Step 3: Compress Tier 3 memories
            stats['tier3_compressed'] = self.tier3_compressor.compress_all_tier3()

            # Step 4: Move old memories to cold storage
            candidates = self.cold_storage.get_cold_storage_candidates()
            if candidates:
                stats['cold_stored'] = self.cold_storage.move_to_cold_storage(candidates)

            # Get final tier stats
            stats['tier_stats'] = self.classifier.get_tier_stats()

            # Calculate space savings
            stats['space_savings'] = self._calculate_space_savings()

        except Exception as e:
            stats['errors'].append(str(e))

        stats['completed_at'] = datetime.now().isoformat()
        return stats

    def _calculate_space_savings(self) -> Dict[str, Any]:
        """Calculate estimated space savings from compression."""
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()

            # Get size of compressed content
            cursor.execute('''
                SELECT
                    tier,
                    COUNT(*) as count,
                    SUM(LENGTH(content)) as total_size
                FROM memories
                GROUP BY tier
            ''')

            tier_sizes = {}
            for tier, count, total_size in cursor.fetchall():
                tier_sizes[tier] = {
                    'count': count,
                    'size_bytes': total_size or 0
                }

            # Get size of archived content
            cursor.execute('''
                SELECT
                    COUNT(*) as count,
                    SUM(LENGTH(full_content)) as total_size
                FROM memory_archive
            ''')
            archive_count, archive_size = cursor.fetchone()

        finally:
            conn.close()

        # Estimate original size if all were Tier 1
        tier1_avg = tier_sizes.get(1, {}).get('size_bytes', 50000) / max(tier_sizes.get(1, {}).get('count', 1), 1)
        total_memories = sum(t.get('count', 0) for t in tier_sizes.values())
        estimated_original = int(tier1_avg * total_memories)

        current_size = sum(t.get('size_bytes', 0) for t in tier_sizes.values())

        return {
            'estimated_original_bytes': estimated_original,
            'current_size_bytes': current_size,
            'savings_bytes': estimated_original - current_size,
            'savings_percent': round((1 - current_size / max(estimated_original, 1)) * 100, 1),
            'tier_breakdown': tier_sizes,
            'archive_count': archive_count or 0,
            'archive_size_bytes': archive_size or 0
        }
