#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 SuperLocalMemory (superlocalmemory.com)
"""
Database schema definitions for learning.db.

This module contains all CREATE TABLE statements and schema migration logic
for the learning database. Extracted from learning_db.py to improve modularity.
"""

import logging
import sqlite3
from typing import List

logger = logging.getLogger("superlocalmemory.learning.db.schema")


# SQL table definitions (as constants for reuse and testing)

SCHEMA_TRANSFERABLE_PATTERNS = '''
    CREATE TABLE IF NOT EXISTS transferable_patterns (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        pattern_type TEXT NOT NULL,
        key TEXT NOT NULL,
        value TEXT NOT NULL,
        confidence REAL DEFAULT 0.0,
        evidence_count INTEGER DEFAULT 0,
        profiles_seen INTEGER DEFAULT 0,
        first_seen TIMESTAMP,
        last_seen TIMESTAMP,
        decay_factor REAL DEFAULT 1.0,
        contradictions TEXT DEFAULT '[]',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(pattern_type, key)
    )
'''

SCHEMA_WORKFLOW_PATTERNS = '''
    CREATE TABLE IF NOT EXISTS workflow_patterns (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        pattern_type TEXT NOT NULL,
        pattern_key TEXT NOT NULL,
        pattern_value TEXT NOT NULL,
        confidence REAL DEFAULT 0.0,
        evidence_count INTEGER DEFAULT 0,
        last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        metadata TEXT DEFAULT '{}'
    )
'''

SCHEMA_RANKING_FEEDBACK = '''
    CREATE TABLE IF NOT EXISTS ranking_feedback (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        query_hash TEXT NOT NULL,
        query_keywords TEXT,
        memory_id INTEGER NOT NULL,
        rank_position INTEGER,
        signal_type TEXT NOT NULL,
        signal_value REAL DEFAULT 1.0,
        channel TEXT NOT NULL,
        source_tool TEXT,
        dwell_time REAL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
'''

SCHEMA_RANKING_MODELS = '''
    CREATE TABLE IF NOT EXISTS ranking_models (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        model_version TEXT NOT NULL,
        training_samples INTEGER,
        synthetic_samples INTEGER DEFAULT 0,
        real_samples INTEGER DEFAULT 0,
        ndcg_at_10 REAL,
        model_path TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
'''

SCHEMA_SOURCE_QUALITY = '''
    CREATE TABLE IF NOT EXISTS source_quality (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        source_id TEXT NOT NULL UNIQUE,
        positive_signals INTEGER DEFAULT 0,
        total_memories INTEGER DEFAULT 0,
        quality_score REAL DEFAULT 0.5,
        last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
'''

SCHEMA_ENGAGEMENT_METRICS = '''
    CREATE TABLE IF NOT EXISTS engagement_metrics (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        metric_date DATE NOT NULL UNIQUE,
        memories_created INTEGER DEFAULT 0,
        recalls_performed INTEGER DEFAULT 0,
        feedback_signals INTEGER DEFAULT 0,
        patterns_updated INTEGER DEFAULT 0,
        active_sources TEXT DEFAULT '[]',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
'''

SCHEMA_ACTION_OUTCOMES = '''
    CREATE TABLE IF NOT EXISTS action_outcomes (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        memory_ids TEXT NOT NULL,
        outcome TEXT NOT NULL,
        action_type TEXT DEFAULT 'other',
        context TEXT DEFAULT '{}',
        confidence REAL DEFAULT 1.0,
        agent_id TEXT DEFAULT 'user',
        project TEXT,
        profile TEXT DEFAULT 'default',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
'''

SCHEMA_BEHAVIORAL_PATTERNS = '''
    CREATE TABLE IF NOT EXISTS behavioral_patterns (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        pattern_type TEXT NOT NULL,
        pattern_key TEXT NOT NULL,
        success_rate REAL DEFAULT 0.0,
        evidence_count INTEGER DEFAULT 0,
        confidence REAL DEFAULT 0.0,
        metadata TEXT DEFAULT '{}',
        project TEXT,
        profile TEXT DEFAULT 'default',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
'''

SCHEMA_CROSS_PROJECT_BEHAVIORS = '''
    CREATE TABLE IF NOT EXISTS cross_project_behaviors (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        source_project TEXT NOT NULL,
        target_project TEXT NOT NULL,
        pattern_id INTEGER NOT NULL,
        transfer_type TEXT DEFAULT 'metadata',
        confidence REAL DEFAULT 0.0,
        profile TEXT DEFAULT 'default',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (pattern_id) REFERENCES behavioral_patterns(id)
    )
'''


def create_all_tables(conn: sqlite3.Connection) -> None:
    """
    Create all learning database tables.

    Args:
        conn: Open SQLite connection

    Raises:
        sqlite3.Error: If table creation fails
    """
    cursor = conn.cursor()

    try:
        # Layer 1: Cross-project transferable patterns
        cursor.execute(SCHEMA_TRANSFERABLE_PATTERNS)

        # Layer 3: Workflow patterns
        cursor.execute(SCHEMA_WORKFLOW_PATTERNS)

        # Feedback from all channels
        cursor.execute(SCHEMA_RANKING_FEEDBACK)

        # Model metadata
        cursor.execute(SCHEMA_RANKING_MODELS)

        # Source quality scores
        cursor.execute(SCHEMA_SOURCE_QUALITY)

        # Engagement metrics
        cursor.execute(SCHEMA_ENGAGEMENT_METRICS)

        # v2.8.0: Behavioral learning tables
        cursor.execute(SCHEMA_ACTION_OUTCOMES)
        cursor.execute(SCHEMA_BEHAVIORAL_PATTERNS)
        cursor.execute(SCHEMA_CROSS_PROJECT_BEHAVIORS)

        conn.commit()
        logger.debug("All tables created successfully")

    except Exception as e:
        logger.error("Failed to create tables: %s", e)
        conn.rollback()
        raise


def add_profile_columns(conn: sqlite3.Connection) -> None:
    """
    Add profile columns to tables (migration for v2.7.4+).

    Args:
        conn: Open SQLite connection
    """
    cursor = conn.cursor()
    tables = ['ranking_feedback', 'transferable_patterns', 'workflow_patterns', 'source_quality']

    for table in tables:
        try:
            cursor.execute(f'ALTER TABLE {table} ADD COLUMN profile TEXT DEFAULT "default"')
            logger.debug(f"Added profile column to {table}")
        except Exception:
            # Column already exists, ignore
            pass

    conn.commit()


def create_indexes(conn: sqlite3.Connection) -> None:
    """
    Create all performance indexes.

    Args:
        conn: Open SQLite connection
    """
    cursor = conn.cursor()

    indexes = [
        # Profile indexes
        'CREATE INDEX IF NOT EXISTS idx_feedback_profile ON ranking_feedback(profile)',
        'CREATE INDEX IF NOT EXISTS idx_patterns_profile ON transferable_patterns(profile)',
        'CREATE INDEX IF NOT EXISTS idx_workflow_profile ON workflow_patterns(profile)',

        # Feedback indexes
        'CREATE INDEX IF NOT EXISTS idx_feedback_query ON ranking_feedback(query_hash)',
        'CREATE INDEX IF NOT EXISTS idx_feedback_memory ON ranking_feedback(memory_id)',
        'CREATE INDEX IF NOT EXISTS idx_feedback_channel ON ranking_feedback(channel)',
        'CREATE INDEX IF NOT EXISTS idx_feedback_created ON ranking_feedback(created_at)',

        # Pattern indexes
        'CREATE INDEX IF NOT EXISTS idx_patterns_type ON transferable_patterns(pattern_type)',
        'CREATE INDEX IF NOT EXISTS idx_workflow_type ON workflow_patterns(pattern_type)',

        # Engagement index
        'CREATE INDEX IF NOT EXISTS idx_engagement_date ON engagement_metrics(metric_date)',

        # v2.8.0 behavioral indexes
        'CREATE INDEX IF NOT EXISTS idx_outcomes_memory ON action_outcomes(memory_ids)',
        'CREATE INDEX IF NOT EXISTS idx_outcomes_project ON action_outcomes(project)',
        'CREATE INDEX IF NOT EXISTS idx_outcomes_profile ON action_outcomes(profile)',
        'CREATE INDEX IF NOT EXISTS idx_bpatterns_type ON behavioral_patterns(pattern_type)',
        'CREATE INDEX IF NOT EXISTS idx_bpatterns_project ON behavioral_patterns(project)',
        'CREATE INDEX IF NOT EXISTS idx_xproject_source ON cross_project_behaviors(source_project)',
        'CREATE INDEX IF NOT EXISTS idx_xproject_target ON cross_project_behaviors(target_project)',
    ]

    for index_sql in indexes:
        cursor.execute(index_sql)

    conn.commit()
    logger.debug("All indexes created successfully")


def initialize_schema(conn: sqlite3.Connection) -> None:
    """
    Full schema initialization: tables + migrations + indexes.

    Args:
        conn: Open SQLite connection

    Raises:
        sqlite3.Error: If schema initialization fails
    """
    try:
        create_all_tables(conn)
        add_profile_columns(conn)
        create_indexes(conn)
        logger.info("Learning schema initialized successfully")
    except Exception as e:
        logger.error("Failed to initialize learning schema: %s", e)
        raise
