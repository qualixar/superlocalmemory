#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 SuperLocalMemory (superlocalmemory.com)
"""
Database query utilities for synthetic bootstrap.

All read-only queries against memory.db used by SyntheticBootstrapper.
These functions are stateless and take db_path as parameter.
"""

import logging
import sqlite3
from pathlib import Path
from typing import List, Optional, Set

from .text_utils import clean_fts_query

logger = logging.getLogger("superlocalmemory.learning.bootstrap.db_queries")


def get_memory_count(db_path: Path) -> int:
    """
    Count total memories in memory.db.

    Args:
        db_path: Path to memory.db.

    Returns:
        Total number of memories, or 0 if error.
    """
    if not db_path.exists():
        return 0
    try:
        conn = sqlite3.connect(str(db_path), timeout=5)
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM memories')
        count = cursor.fetchone()[0]
        conn.close()
        return count
    except Exception as e:
        logger.warning("Failed to count memories: %s", e)
        return 0


def get_memories_by_access(db_path: Path, min_access: int = 5) -> List[dict]:
    """
    Fetch memories with access_count >= min_access from memory.db.

    These are memories the user keeps coming back to — strong positive signal.

    Args:
        db_path: Path to memory.db.
        min_access: Minimum access_count threshold.

    Returns:
        List of memory dicts.
    """
    if not db_path.exists():
        return []
    try:
        conn = sqlite3.connect(str(db_path), timeout=5)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute('''
            SELECT id, content, summary, project_name, tags,
                   category, importance, created_at, access_count
            FROM memories
            WHERE access_count >= ?
            ORDER BY access_count DESC
            LIMIT 100
        ''', (min_access,))
        results = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return results
    except Exception as e:
        logger.warning("Failed to fetch high-access memories: %s", e)
        return []


def get_memories_by_importance(db_path: Path, min_importance: int = 8) -> List[dict]:
    """
    Fetch memories with importance >= min_importance from memory.db.

    High importance = user explicitly rated these as valuable.

    Args:
        db_path: Path to memory.db.
        min_importance: Minimum importance threshold.

    Returns:
        List of memory dicts.
    """
    if not db_path.exists():
        return []
    try:
        conn = sqlite3.connect(str(db_path), timeout=5)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute('''
            SELECT id, content, summary, project_name, tags,
                   category, importance, created_at, access_count
            FROM memories
            WHERE importance >= ?
            ORDER BY importance DESC
            LIMIT 100
        ''', (min_importance,))
        results = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return results
    except Exception as e:
        logger.warning("Failed to fetch high-importance memories: %s", e)
        return []


def get_recent_memories(db_path: Path, limit: int = 30) -> List[dict]:
    """
    Fetch the N most recently created memories.

    Args:
        db_path: Path to memory.db.
        limit: Maximum number of memories to return.

    Returns:
        List of memory dicts, sorted by created_at DESC.
    """
    if not db_path.exists():
        return []
    try:
        conn = sqlite3.connect(str(db_path), timeout=5)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute('''
            SELECT id, content, summary, project_name, tags,
                   category, importance, created_at, access_count
            FROM memories
            ORDER BY created_at DESC
            LIMIT ?
        ''', (limit,))
        results = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return results
    except Exception as e:
        logger.warning("Failed to fetch recent memories: %s", e)
        return []


def get_learned_patterns(
    db_path: Path,
    min_confidence: float = 0.7,
) -> List[dict]:
    """
    Fetch high-confidence identity_patterns from memory.db.

    These are patterns detected by pattern_learner.py (Layer 4) —
    tech preferences, coding style, terminology, etc.

    Returns empty list if identity_patterns table doesn't exist
    (backward compatible with pre-v2.3 databases).

    Args:
        db_path: Path to memory.db.
        min_confidence: Minimum confidence threshold.

    Returns:
        List of pattern dicts.
    """
    if not db_path.exists():
        return []
    try:
        conn = sqlite3.connect(str(db_path), timeout=5)
        try:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            # Check if table exists (backward compatibility)
            cursor.execute('''
                SELECT name FROM sqlite_master
                WHERE type='table' AND name='identity_patterns'
            ''')
            if cursor.fetchone() is None:
                return []

            cursor.execute('''
                SELECT id, pattern_type, key, value, confidence,
                       evidence_count, category
                FROM identity_patterns
                WHERE confidence >= ?
                ORDER BY confidence DESC
                LIMIT 50
            ''', (min_confidence,))
            results = [dict(row) for row in cursor.fetchall()]
            return results
        finally:
            conn.close()
    except Exception as e:
        logger.warning("Failed to fetch learned patterns: %s", e)
        return []


def search_memories(db_path: Path, query: str, limit: int = 20) -> List[dict]:
    """
    Simple FTS5 search in memory.db.

    Used to find memories matching synthetic query terms.
    This is a lightweight search — no TF-IDF, no HNSW, just FTS5.

    Args:
        db_path: Path to memory.db.
        query: Search query string.
        limit: Maximum results to return.

    Returns:
        List of memory dicts matching the query.
    """
    if not db_path.exists():
        return []
    if not query or not query.strip():
        return []

    try:
        conn = sqlite3.connect(str(db_path), timeout=5)
        try:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            # Clean query for FTS5 (same approach as memory_store_v2.search)
            fts_query = clean_fts_query(query)
            if not fts_query:
                return []

            cursor.execute('''
                SELECT m.id, m.content, m.summary, m.project_name, m.tags,
                       m.category, m.importance, m.created_at, m.access_count
                FROM memories m
                JOIN memories_fts fts ON m.id = fts.rowid
                WHERE memories_fts MATCH ?
                ORDER BY rank
                LIMIT ?
            ''', (fts_query, limit))
            results = [dict(row) for row in cursor.fetchall()]
            return results
        finally:
            conn.close()
    except Exception as e:
        logger.debug("FTS5 search failed (may not exist yet): %s", e)
        return []


def find_negative_memories(
    db_path: Path,
    anchor_memory: dict,
    exclude_ids: Optional[Set[int]] = None,
    limit: int = 2,
) -> List[dict]:
    """
    Find memories dissimilar to the anchor (for negative examples).

    Simple heuristic: pick memories from a different category or project.
    Falls back to random sample if no structured differences available.

    Args:
        db_path: Path to memory.db.
        anchor_memory: The reference memory to find negatives for.
        exclude_ids: Set of memory IDs to exclude from results.
        limit: Maximum number of negatives to return.

    Returns:
        List of negative example memory dicts.
    """
    if not db_path.exists():
        return []
    exclude_ids = exclude_ids or set()

    try:
        conn = sqlite3.connect(str(db_path), timeout=5)
        try:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            anchor_project = anchor_memory.get('project_name', '')
            anchor_category = anchor_memory.get('category', '')

            # Try to find memories from different project or category
            conditions = []
            params: list = []

            if anchor_project:
                conditions.append('project_name != ?')
                params.append(anchor_project)
            if anchor_category:
                conditions.append('category != ?')
                params.append(anchor_category)

            # Exclude specified IDs
            if exclude_ids:
                placeholders = ','.join('?' for _ in exclude_ids)
                conditions.append(f'id NOT IN ({placeholders})')
                params.extend(exclude_ids)

            where_clause = ' AND '.join(conditions) if conditions else '1=1'

            cursor.execute(f'''
                SELECT id, content, summary, project_name, tags,
                       category, importance, created_at, access_count
                FROM memories
                WHERE {where_clause}
                ORDER BY RANDOM()
                LIMIT ?
            ''', (*params, limit))
            results = [dict(row) for row in cursor.fetchall()]
            return results
        finally:
            conn.close()
    except Exception as e:
        logger.debug("Failed to find negative memories: %s", e)
        return []
