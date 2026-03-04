#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 SuperLocalMemory (superlocalmemory.com)
"""
Database schema definitions for MemoryStoreV2.

This module contains SQL schema definitions and migration logic extracted from
memory_store_v2.py to reduce file size and improve maintainability.
"""

from typing import Dict, List, Tuple


# V2 column definitions for migration support
V2_COLUMNS: Dict[str, str] = {
    'summary': 'TEXT',
    'project_path': 'TEXT',
    'project_name': 'TEXT',
    'category': 'TEXT',
    'parent_id': 'INTEGER',
    'tree_path': 'TEXT',
    'depth': 'INTEGER DEFAULT 0',
    'memory_type': 'TEXT DEFAULT "session"',
    'importance': 'INTEGER DEFAULT 5',
    'updated_at': 'TIMESTAMP DEFAULT CURRENT_TIMESTAMP',
    'last_accessed': 'TIMESTAMP',
    'access_count': 'INTEGER DEFAULT 0',
    'content_hash': 'TEXT',
    'cluster_id': 'INTEGER',
    'profile': 'TEXT DEFAULT "default"'
}

# v2.8.0 schema migrations - lifecycle + access control columns
V28_MIGRATIONS: List[Tuple[str, str]] = [
    ("lifecycle_state", "TEXT DEFAULT 'active'"),
    ("lifecycle_updated_at", "TIMESTAMP"),
    ("lifecycle_history", "TEXT DEFAULT '[]'"),
    ("access_level", "TEXT DEFAULT 'public'"),
]

# Index definitions for V2 fields
V2_INDEXES: List[Tuple[str, str]] = [
    ('idx_project', 'project_path'),
    ('idx_tags', 'tags'),
    ('idx_category', 'category'),
    ('idx_tree_path', 'tree_path'),
    ('idx_cluster', 'cluster_id'),
    ('idx_last_accessed', 'last_accessed'),
    ('idx_parent_id', 'parent_id'),
    ('idx_profile', 'profile')
]


def get_memories_table_sql() -> str:
    """
    Returns the CREATE TABLE SQL for the main memories table.
    V1 compatible + V2 extensions.
    """
    return '''
        CREATE TABLE IF NOT EXISTS memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content TEXT NOT NULL,
            summary TEXT,

            -- Organization
            project_path TEXT,
            project_name TEXT,
            tags TEXT,
            category TEXT,

            -- Hierarchy (Layer 2 link)
            parent_id INTEGER,
            tree_path TEXT,
            depth INTEGER DEFAULT 0,

            -- Metadata
            memory_type TEXT DEFAULT 'session',
            importance INTEGER DEFAULT 5,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_accessed TIMESTAMP,
            access_count INTEGER DEFAULT 0,

            -- Deduplication
            content_hash TEXT UNIQUE,

            -- Graph (Layer 3 link)
            cluster_id INTEGER,

            FOREIGN KEY (parent_id) REFERENCES memories(id) ON DELETE CASCADE
        )
    '''


def get_sessions_table_sql() -> str:
    """
    Returns the CREATE TABLE SQL for the sessions table.
    V1 compatible.
    """
    return '''
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT UNIQUE,
            project_path TEXT,
            started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            ended_at TIMESTAMP,
            summary TEXT
        )
    '''


def get_fts_table_sql() -> str:
    """
    Returns the CREATE VIRTUAL TABLE SQL for full-text search.
    V1 compatible.
    """
    return '''
        CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts
        USING fts5(content, summary, tags, content='memories', content_rowid='id')
    '''


def get_fts_trigger_insert_sql() -> str:
    """Returns the FTS INSERT trigger SQL."""
    return '''
        CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
            INSERT INTO memories_fts(rowid, content, summary, tags)
            VALUES (new.id, new.content, new.summary, new.tags);
        END
    '''


def get_fts_trigger_delete_sql() -> str:
    """Returns the FTS DELETE trigger SQL."""
    return '''
        CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
            INSERT INTO memories_fts(memories_fts, rowid, content, summary, tags)
            VALUES('delete', old.id, old.content, old.summary, old.tags);
        END
    '''


def get_fts_trigger_update_sql() -> str:
    """Returns the FTS UPDATE trigger SQL."""
    return '''
        CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
            INSERT INTO memories_fts(memories_fts, rowid, content, summary, tags)
            VALUES('delete', old.id, old.content, old.summary, old.tags);
            INSERT INTO memories_fts(rowid, content, summary, tags)
            VALUES (new.id, new.content, new.summary, new.tags);
        END
    '''


def get_creator_metadata_table_sql() -> str:
    """
    Returns the CREATE TABLE SQL for creator attribution metadata.
    REQUIRED by MIT License.
    """
    return '''
        CREATE TABLE IF NOT EXISTS creator_metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    '''
