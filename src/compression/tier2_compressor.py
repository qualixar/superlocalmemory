#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 SuperLocalMemory (superlocalmemory.com)
"""
Tier 2 compression logic.
Compresses memories to summary + key excerpts format.
"""

import sqlite3
import json
import re
from datetime import datetime
from pathlib import Path
from typing import List


MEMORY_DIR = Path.home() / ".claude-memory"
DB_PATH = MEMORY_DIR / "memory.db"


class Tier2Compressor:
    """Compress memories to summary + key excerpts (Tier 2)."""

    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path

    def compress_to_tier2(self, memory_id: int) -> bool:
        """
        Compress memory to summary + excerpts.

        Args:
            memory_id: ID of memory to compress

        Returns:
            True if compression succeeded, False otherwise
        """
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()

            # Get full content
            cursor.execute('''
                SELECT content, summary, tier FROM memories WHERE id = ?
            ''', (memory_id,))
            result = cursor.fetchone()

            if not result:
                return False

            content, existing_summary, current_tier = result

            # Skip if already compressed or in wrong tier
            if current_tier != 2:
                return False

            # Check if already archived (don't re-compress)
            cursor.execute('''
                SELECT full_content FROM memory_archive WHERE memory_id = ?
            ''', (memory_id,))
            if cursor.fetchone():
                return True  # Already compressed

            # Try to parse as JSON (might already be compressed)
            try:
                parsed = json.loads(content)
                if isinstance(parsed, dict) and 'summary' in parsed:
                    return True  # Already compressed
            except (json.JSONDecodeError, TypeError):
                pass  # Not compressed yet

            # Generate/enhance summary if needed
            if not existing_summary or len(existing_summary) < 100:
                summary = self._generate_summary(content)
            else:
                summary = existing_summary

            # Extract key excerpts (important sentences, code blocks, lists)
            excerpts = self._extract_key_excerpts(content)

            # Store compressed version
            compressed_content = {
                'summary': summary,
                'excerpts': excerpts,
                'original_length': len(content),
                'compressed_at': datetime.now().isoformat()
            }

            # Move full content to archive table
            cursor.execute('''
                INSERT INTO memory_archive (memory_id, full_content, archived_at)
                VALUES (?, ?, CURRENT_TIMESTAMP)
            ''', (memory_id, content))

            # Update memory with compressed version
            cursor.execute('''
                UPDATE memories
                SET content = ?, tier = 2, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            ''', (json.dumps(compressed_content), memory_id))

            conn.commit()
            return True
        finally:
            conn.close()

    def _generate_summary(self, content: str, max_length: int = 300) -> str:
        """
        Generate extractive summary from content.
        Uses sentence scoring based on heuristics (no external LLM).

        Args:
            content: Full content text
            max_length: Maximum summary length in characters

        Returns:
            Extracted summary
        """
        # Split into sentences
        sentences = re.split(r'[.!?]+', content)

        # Score sentences by importance (simple heuristic)
        scored_sentences = []

        for i, sent in enumerate(sentences):
            sent = sent.strip()
            if len(sent) < 10:
                continue

            score = 0

            # Boost if contains tech terms
            tech_terms = ['api', 'database', 'auth', 'component', 'function',
                         'class', 'method', 'variable', 'error', 'bug', 'fix',
                         'implement', 'refactor', 'test', 'deploy']
            score += sum(1 for term in tech_terms if term in sent.lower())

            # Boost if at start or end (thesis/conclusion)
            if i == 0 or i == len(sentences) - 1:
                score += 2

            # Boost if contains numbers/specifics
            if re.search(r'\d+', sent):
                score += 1

            # Boost if contains important keywords
            important_keywords = ['important', 'critical', 'note', 'remember',
                                 'key', 'main', 'primary', 'must', 'should']
            score += sum(2 for kw in important_keywords if kw in sent.lower())

            scored_sentences.append((score, sent))

        # Take top sentences up to max_length
        scored_sentences.sort(reverse=True, key=lambda x: x[0])

        summary_parts = []
        current_length = 0

        for score, sent in scored_sentences:
            if current_length + len(sent) > max_length:
                break

            summary_parts.append(sent)
            current_length += len(sent)

        if not summary_parts:
            # Fallback: take first sentence
            return sentences[0][:max_length] if sentences else content[:max_length]

        return '. '.join(summary_parts) + '.'

    def _extract_key_excerpts(self, content: str, max_excerpts: int = 3) -> List[str]:
        """
        Extract key excerpts (code blocks, lists, important paragraphs).

        Args:
            content: Full content text
            max_excerpts: Maximum number of excerpts to extract

        Returns:
            List of excerpt strings
        """
        excerpts = []

        # Extract code blocks (markdown or indented)
        code_blocks = re.findall(r'```[\s\S]*?```', content)
        excerpts.extend(code_blocks[:2])  # Max 2 code blocks

        # Extract bullet lists
        list_pattern = r'(?:^|\n)(?:[-*•]|\d+\.)\s+.+(?:\n(?:[-*•]|\d+\.)\s+.+)*'
        lists = re.findall(list_pattern, content, re.MULTILINE)
        if lists and len(excerpts) < max_excerpts:
            excerpts.extend(lists[:1])  # Max 1 list

        # Extract paragraphs with important keywords if we need more
        if len(excerpts) < max_excerpts:
            paragraphs = content.split('\n\n')
            important_keywords = ['important', 'critical', 'note', 'remember', 'key']

            for para in paragraphs:
                if len(excerpts) >= max_excerpts:
                    break

                if any(kw in para.lower() for kw in important_keywords):
                    # Truncate long paragraphs
                    if len(para) > 200:
                        para = para[:197] + '...'
                    excerpts.append(para)

        # Truncate if too many
        return excerpts[:max_excerpts]

    def compress_all_tier2(self) -> int:
        """Compress all memories that are in Tier 2."""
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()

            cursor.execute('SELECT id FROM memories WHERE tier = 2')
            memory_ids = [row[0] for row in cursor.fetchall()]
        finally:
            conn.close()

        compressed_count = 0
        for memory_id in memory_ids:
            if self.compress_to_tier2(memory_id):
                compressed_count += 1

        return compressed_count
