#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 SuperLocalMemory (superlocalmemory.com)
"""
Tier 3 compression logic.
Compresses memories to bullet points only format.
"""

import sqlite3
import json
import re
from datetime import datetime
from pathlib import Path
from typing import List


MEMORY_DIR = Path.home() / ".claude-memory"
DB_PATH = MEMORY_DIR / "memory.db"


class Tier3Compressor:
    """Compress memories to bullet points only (Tier 3)."""

    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path

    def compress_to_tier3(self, memory_id: int) -> bool:
        """
        Compress memory to bullet points only.

        Args:
            memory_id: ID of memory to compress

        Returns:
            True if compression succeeded, False otherwise
        """
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()

            # Get Tier 2 compressed content
            cursor.execute('''
                SELECT content, tier FROM memories WHERE id = ?
            ''', (memory_id,))
            result = cursor.fetchone()

            if not result:
                return False

            content, current_tier = result

            # Skip if in wrong tier
            if current_tier != 3:
                return False

            # Try to parse as Tier 2 compressed content
            try:
                compressed_content = json.loads(content)

                # Check if already Tier 3
                if isinstance(compressed_content, dict) and 'bullets' in compressed_content:
                    return True  # Already Tier 3

                # Get summary from Tier 2
                if isinstance(compressed_content, dict) and 'summary' in compressed_content:
                    summary = compressed_content.get('summary', '')
                    tier2_archived_at = compressed_content.get('compressed_at')
                    original_length = compressed_content.get('original_length', 0)
                else:
                    # Not Tier 2 format, treat as plain text
                    summary = content
                    tier2_archived_at = None
                    original_length = len(content)

            except (json.JSONDecodeError, TypeError):
                # Not JSON, treat as plain text
                summary = content
                tier2_archived_at = None
                original_length = len(content)

            # Convert summary to bullet points (max 5)
            bullet_points = self._summarize_to_bullets(summary)

            # Ultra-compressed version
            ultra_compressed = {
                'bullets': bullet_points,
                'tier2_archived_at': tier2_archived_at,
                'original_length': original_length,
                'compressed_to_tier3_at': datetime.now().isoformat()
            }

            # Update memory
            cursor.execute('''
                UPDATE memories
                SET content = ?, tier = 3, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            ''', (json.dumps(ultra_compressed), memory_id))

            conn.commit()
        finally:
            conn.close()
        return True

    def _summarize_to_bullets(self, summary: str, max_bullets: int = 5) -> List[str]:
        """
        Convert summary to bullet points.

        Args:
            summary: Summary text
            max_bullets: Maximum number of bullets

        Returns:
            List of bullet point strings
        """
        # Split into sentences
        sentences = re.split(r'[.!?]+', summary)

        bullets = []

        for sent in sentences:
            sent = sent.strip()

            if len(sent) < 10:
                continue

            # Truncate long sentences
            if len(sent) > 80:
                sent = sent[:77] + '...'

            bullets.append(sent)

            if len(bullets) >= max_bullets:
                break

        return bullets if bullets else ['[No summary available]']

    def compress_all_tier3(self) -> int:
        """Compress all memories that are in Tier 3."""
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()

            cursor.execute('SELECT id FROM memories WHERE tier = 3')
            memory_ids = [row[0] for row in cursor.fetchall()]
        finally:
            conn.close()

        compressed_count = 0
        for memory_id in memory_ids:
            if self.compress_to_tier3(memory_id):
                compressed_count += 1

        return compressed_count
