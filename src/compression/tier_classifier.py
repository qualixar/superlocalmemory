#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 SuperLocalMemory (superlocalmemory.com)
"""
Tier classification logic for memory compression.
Classifies memories into tiers based on age and access patterns.
"""

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple

from compression.config import CompressionConfig


MEMORY_DIR = Path.home() / ".claude-memory"
DB_PATH = MEMORY_DIR / "memory.db"


class TierClassifier:
    """Classify memories into compression tiers based on age and access patterns."""

    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self.config = CompressionConfig()
        self._ensure_schema()

    def _ensure_schema(self):
        """Add tier and access tracking columns if not present."""
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()

            # Check if tier column exists
            cursor.execute("PRAGMA table_info(memories)")
            columns = [row[1] for row in cursor.fetchall()]

            if 'tier' not in columns:
                cursor.execute('ALTER TABLE memories ADD COLUMN tier INTEGER DEFAULT 1')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_tier ON memories(tier)')

            if 'last_accessed' not in columns:
                cursor.execute('ALTER TABLE memories ADD COLUMN last_accessed TIMESTAMP')

            if 'access_count' not in columns:
                cursor.execute('ALTER TABLE memories ADD COLUMN access_count INTEGER DEFAULT 0')

            # Create memory_archive table if not exists
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS memory_archive (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    memory_id INTEGER UNIQUE NOT NULL,
                    full_content TEXT NOT NULL,
                    archived_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (memory_id) REFERENCES memories(id) ON DELETE CASCADE
                )
            ''')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_archive_memory ON memory_archive(memory_id)')

            conn.commit()
        finally:
            conn.close()

    def classify_memories(self) -> List[Tuple[int, int]]:
        """
        Classify all memories into tiers based on age and access.

        Returns:
            List of (tier, memory_id) tuples
        """
        if not self.config.enabled:
            return []

        now = datetime.now()
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()

            # Get all memories with access tracking
            cursor.execute('''
                SELECT id, created_at, last_accessed, access_count, importance, tier
                FROM memories
            ''')
            memories = cursor.fetchall()

            tier_updates = []

            for memory_id, created_at, last_accessed, access_count, importance, current_tier in memories:
                created = datetime.fromisoformat(created_at)
                age_days = (now - created).days

                # Override: High-importance memories stay in Tier 1
                if self.config.preserve_high_importance and importance and importance >= 8:
                    tier = 1
                # Recently accessed stays in Tier 1
                elif self.config.preserve_recently_accessed and last_accessed:
                    last_access = datetime.fromisoformat(last_accessed)
                    if (now - last_access).days < 7:
                        tier = 1
                    else:
                        tier = self._classify_by_age(age_days)
                # Age-based classification
                else:
                    tier = self._classify_by_age(age_days)

                # Only update if tier changed
                if tier != current_tier:
                    tier_updates.append((tier, memory_id))

            # Update tier field
            if tier_updates:
                cursor.executemany('''
                    UPDATE memories SET tier = ? WHERE id = ?
                ''', tier_updates)
                conn.commit()

        finally:
            conn.close()
        return tier_updates

    def _classify_by_age(self, age_days: int) -> int:
        """Classify memory tier based on age."""
        if age_days < self.config.tier2_threshold_days:
            return 1  # Recent
        elif age_days < self.config.tier3_threshold_days:
            return 2  # Active
        else:
            return 3  # Archived

    def get_tier_stats(self) -> Dict[str, int]:
        """Get count of memories in each tier."""
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()

            cursor.execute('''
                SELECT tier, COUNT(*) FROM memories GROUP BY tier
            ''')
            stats = dict(cursor.fetchall())
        finally:
            conn.close()

        return {
            'tier1': stats.get(1, 0),
            'tier2': stats.get(2, 0),
            'tier3': stats.get(3, 0)
        }
