#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 SuperLocalMemory (superlocalmemory.com)
"""
Pattern Store - SQLite-backed pattern storage and retrieval.

Handles identity_patterns and pattern_examples tables,
including schema migration, CRUD operations, and profile-scoped queries.
"""

import sqlite3
import json
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path

logger = logging.getLogger(__name__)


class PatternStore:
    """Handles pattern storage and retrieval."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._init_tables()

    def _init_tables(self):
        """Initialize pattern tables if they don't exist, or recreate if schema is incomplete."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Check if existing tables have correct schema
        for table_name, required_cols in [
            ('identity_patterns', {'pattern_type', 'key', 'value', 'confidence'}),
            ('pattern_examples', {'pattern_id', 'memory_id'}),
        ]:
            cursor.execute(f"PRAGMA table_info({table_name})")
            existing_cols = {row[1] for row in cursor.fetchall()}
            if existing_cols and not required_cols.issubset(existing_cols):
                logger.warning(f"Dropping incomplete {table_name} table (missing: {required_cols - existing_cols})")
                cursor.execute(f'DROP TABLE IF EXISTS {table_name}')

        # Identity patterns table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS identity_patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pattern_type TEXT NOT NULL,
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                confidence REAL DEFAULT 0.5,
                evidence_count INTEGER DEFAULT 1,
                memory_ids TEXT,
                category TEXT,
                profile TEXT DEFAULT 'default',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(pattern_type, key, category, profile)
            )
        ''')

        # Add profile column if upgrading from older schema
        try:
            cursor.execute('ALTER TABLE identity_patterns ADD COLUMN profile TEXT DEFAULT "default"')
        except sqlite3.OperationalError:
            pass  # Column already exists

        # Pattern examples table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS pattern_examples (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pattern_id INTEGER NOT NULL,
                memory_id INTEGER NOT NULL,
                example_text TEXT,
                FOREIGN KEY (pattern_id) REFERENCES identity_patterns(id) ON DELETE CASCADE,
                FOREIGN KEY (memory_id) REFERENCES memories(id) ON DELETE CASCADE
            )
        ''')

        # Indexes
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_pattern_type ON identity_patterns(pattern_type)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_pattern_confidence ON identity_patterns(confidence)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_pattern_profile ON identity_patterns(profile)')

        conn.commit()
        conn.close()

    def save_pattern(self, pattern: Dict[str, Any]) -> int:
        """Save or update a pattern (scoped by profile)."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        profile = pattern.get('profile', 'default')

        try:
            # Check if pattern exists for this profile
            cursor.execute('''
                SELECT id FROM identity_patterns
                WHERE pattern_type = ? AND key = ? AND category = ? AND profile = ?
            ''', (pattern['pattern_type'], pattern['key'], pattern['category'], profile))

            existing = cursor.fetchone()

            memory_ids_json = json.dumps(pattern['memory_ids'])

            if existing:
                # Update existing pattern
                pattern_id = existing[0]
                cursor.execute('''
                    UPDATE identity_patterns
                    SET value = ?, confidence = ?, evidence_count = ?,
                        memory_ids = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                ''', (
                    pattern['value'],
                    pattern['confidence'],
                    pattern['evidence_count'],
                    memory_ids_json,
                    pattern_id
                ))
            else:
                # Insert new pattern
                cursor.execute('''
                    INSERT INTO identity_patterns
                    (pattern_type, key, value, confidence, evidence_count, memory_ids, category, profile)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    pattern['pattern_type'],
                    pattern['key'],
                    pattern['value'],
                    pattern['confidence'],
                    pattern['evidence_count'],
                    memory_ids_json,
                    pattern['category'],
                    profile
                ))
                pattern_id = cursor.lastrowid

            # Save examples
            self._save_pattern_examples(cursor, pattern_id, pattern['memory_ids'], pattern['key'])

            conn.commit()
            return pattern_id

        finally:
            conn.close()

    def _save_pattern_examples(self, cursor, pattern_id: int, memory_ids: List[int], key: str):
        """Save representative examples for pattern."""
        # Clear old examples
        cursor.execute('DELETE FROM pattern_examples WHERE pattern_id = ?', (pattern_id,))

        # Save top 3 examples
        for memory_id in memory_ids[:3]:
            cursor.execute('SELECT content FROM memories WHERE id = ?', (memory_id,))
            row = cursor.fetchone()

            if row:
                content = row[0]
                excerpt = self._extract_relevant_excerpt(content, key)

                cursor.execute('''
                    INSERT INTO pattern_examples (pattern_id, memory_id, example_text)
                    VALUES (?, ?, ?)
                ''', (pattern_id, memory_id, excerpt))

    def _extract_relevant_excerpt(self, content: str, key: str) -> str:
        """Extract 150-char excerpt showing pattern."""
        # Find first mention of key term
        key_lower = key.lower().replace('_', ' ')
        idx = content.lower().find(key_lower)

        if idx >= 0:
            start = max(0, idx - 50)
            end = min(len(content), idx + 100)
            excerpt = content[start:end]
            return excerpt if len(excerpt) <= 150 else excerpt[:150] + '...'

        # Fallback: first 150 chars
        return content[:150] + ('...' if len(content) > 150 else '')

    def get_patterns(self, min_confidence: float = 0.7, pattern_type: Optional[str] = None,
                     profile: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get patterns above confidence threshold, optionally filtered by profile."""
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()

            # Build query with optional filters
            conditions = ['confidence >= ?']
            params = [min_confidence]

            if pattern_type:
                conditions.append('pattern_type = ?')
                params.append(pattern_type)

            if profile:
                conditions.append('profile = ?')
                params.append(profile)

            where_clause = ' AND '.join(conditions)
            cursor.execute(f'''
                SELECT id, pattern_type, key, value, confidence, evidence_count,
                       updated_at, created_at, category
                FROM identity_patterns
                WHERE {where_clause}
                ORDER BY confidence DESC, evidence_count DESC
            ''', params)

            patterns = []
            for row in cursor.fetchall():
                patterns.append({
                    'id': row[0],
                    'pattern_type': row[1],
                    'key': row[2],
                    'value': row[3],
                    'confidence': row[4],
                    'evidence_count': row[5],
                    'frequency': row[5],
                    'last_seen': row[6],
                    'created_at': row[7],
                    'category': row[8]
                })

        finally:
            conn.close()
        return patterns
