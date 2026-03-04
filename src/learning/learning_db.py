#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 SuperLocalMemory (superlocalmemory.com)
"""
LearningDB — Manages the separate learning.db for behavioral data.

CRITICAL DESIGN DECISIONS:
    1. learning.db is SEPARATE from memory.db (GDPR erasable, security isolation)
    2. All tables use CREATE TABLE IF NOT EXISTS (safe for re-runs)
    3. WAL mode for concurrent read/write from multiple agents
    4. Singleton pattern matches existing DbConnectionManager approach
    5. Thread-safe via threading.Lock on write operations

Tables (6):
    transferable_patterns  — Layer 1: Cross-project tech preferences
    workflow_patterns      — Layer 3: Sequence + temporal patterns
    ranking_feedback       — Feedback from all channels (MCP, CLI, dashboard)
    ranking_models         — Model metadata and training history
    source_quality         — Per-source learning (which tools produce better memories)
    engagement_metrics     — Local-only engagement stats (never transmitted)
"""

import json
import logging
import sqlite3
import threading
from datetime import datetime, date
from pathlib import Path
from typing import Optional, Dict, List, Any

from .db.constants import MEMORY_DIR, LEARNING_DB_PATH, DEFAULT_PROFILE
from .db.schema import initialize_schema

logger = logging.getLogger("superlocalmemory.learning.db")


class LearningDB:
    """
    Manages the learning.db database for behavioral data.

    Singleton per database path. Thread-safe writes.
    Separate from memory.db for GDPR compliance and security isolation.

    Usage:
        db = LearningDB()
        db.store_feedback(query_hash="abc123", memory_id=42, signal_type="mcp_used")
        stats = db.get_stats()
    """

    _instances: Dict[str, "LearningDB"] = {}
    _instances_lock = threading.Lock()

    @classmethod
    def get_instance(cls, db_path: Optional[Path] = None) -> "LearningDB":
        """Get or create the singleton LearningDB."""
        if db_path is None:
            db_path = LEARNING_DB_PATH
        key = str(db_path)
        with cls._instances_lock:
            if key not in cls._instances:
                cls._instances[key] = cls(db_path)
            return cls._instances[key]

    @classmethod
    def reset_instance(cls, db_path: Optional[Path] = None):
        """Remove singleton. Used for testing."""
        with cls._instances_lock:
            if db_path is None:
                cls._instances.clear()
            else:
                key = str(db_path)
                if key in cls._instances:
                    del cls._instances[key]

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = Path(db_path) if db_path else LEARNING_DB_PATH
        self._write_lock = threading.Lock()
        self._ensure_directory()
        self._init_schema()

    def _get_active_profile(self) -> str:
        """Get the active profile name from profiles.json. Returns 'default' if unavailable."""
        try:
            import json
            profiles_path = self.db_path.parent / "profiles.json"
            if profiles_path.exists():
                with open(profiles_path, 'r') as f:
                    config = json.load(f)
                return config.get('active_profile', 'default')
        except Exception:
            pass
        return "default"
        logger.info("LearningDB initialized: %s", self.db_path)

    def _ensure_directory(self):
        """Ensure the parent directory exists."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def _get_connection(self) -> sqlite3.Connection:
        """Get a new database connection with standard pragmas."""
        conn = sqlite3.connect(str(self.db_path), timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _init_schema(self):
        """Create all learning tables if they don't exist."""
        conn = self._get_connection()
        try:
            initialize_schema(conn)
        finally:
            conn.close()

    # ======================================================================
    # Feedback Operations
    # ======================================================================

    def store_feedback(
        self,
        query_hash: str,
        memory_id: int,
        signal_type: str,
        signal_value: float = 1.0,
        channel: str = "mcp",
        query_keywords: Optional[str] = None,
        rank_position: Optional[int] = None,
        source_tool: Optional[str] = None,
        dwell_time: Optional[float] = None,
        profile: Optional[str] = None,
    ) -> int:
        """
        Store a ranking feedback signal.

        Args:
            query_hash: SHA256[:16] of the query (privacy-preserving)
            memory_id: ID of the memory in memory.db
            signal_type: One of 'mcp_used', 'cli_useful', 'dashboard_click', 'passive_decay'
            signal_value: 1.0=strong positive, 0.5=weak, 0.0=negative
            channel: 'mcp', 'cli', or 'dashboard'
            query_keywords: Top keywords for grouping (optional)
            rank_position: Where it appeared in results (1-50)
            source_tool: Tool that originated the query (e.g., 'claude-desktop')
            dwell_time: Seconds spent viewing (dashboard only)
            profile: Active profile name (v2.7.4 — per-profile learning)

        Returns:
            Row ID of the inserted feedback record.
        """
        # v2.7.4: Detect active profile if not provided
        if not profile:
            profile = self._get_active_profile()

        with self._write_lock:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO ranking_feedback
                        (query_hash, memory_id, signal_type, signal_value,
                         channel, query_keywords, rank_position, source_tool,
                         dwell_time, profile)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    query_hash, memory_id, signal_type, signal_value,
                    channel, query_keywords, rank_position, source_tool,
                    dwell_time, profile,
                ))
                conn.commit()
                row_id = cursor.lastrowid
                logger.debug(
                    "Feedback stored: memory=%d, type=%s, value=%.1f",
                    memory_id, signal_type, signal_value
                )
                return row_id
            except Exception as e:
                conn.rollback()
                logger.error("Failed to store feedback: %s", e)
                raise
            finally:
                conn.close()

    def get_feedback_count(self, profile_scoped: bool = False) -> int:
        """Get total number of feedback signals.

        Args:
            profile_scoped: If True, count only signals for the active profile.
        """
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            if profile_scoped:
                profile = self._get_active_profile()
                cursor.execute(
                    'SELECT COUNT(*) FROM ranking_feedback WHERE profile = ?',
                    (profile,)
                )
            else:
                cursor.execute('SELECT COUNT(*) FROM ranking_feedback')
            return cursor.fetchone()[0]
        finally:
            conn.close()

    def get_signal_stats_for_memories(self, memory_ids: Optional[List[int]] = None) -> Dict[str, Dict[str, float]]:
        """
        Get aggregate feedback signal stats per memory (v2.7.4).

        Returns a dict mapping str(memory_id) to {count, avg_value}.
        Used by FeatureExtractor for features [10] and [11].

        Args:
            memory_ids: If provided, only fetch stats for these IDs.
                        If None, fetch stats for all memories with signals.

        Returns:
            {'42': {'count': 5, 'avg_value': 0.72}, ...}
        """
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            if memory_ids:
                placeholders = ','.join('?' for _ in memory_ids)
                cursor.execute(
                    f'SELECT memory_id, COUNT(*) as cnt, AVG(signal_value) as avg_val '
                    f'FROM ranking_feedback WHERE memory_id IN ({placeholders}) '
                    f'GROUP BY memory_id',
                    tuple(memory_ids),
                )
            else:
                cursor.execute(
                    'SELECT memory_id, COUNT(*) as cnt, AVG(signal_value) as avg_val '
                    'FROM ranking_feedback GROUP BY memory_id'
                )
            result = {}
            for row in cursor.fetchall():
                result[str(row['memory_id'])] = {
                    'count': row['cnt'],
                    'avg_value': round(float(row['avg_val']), 3),
                }
            return result
        except Exception as e:
            logger.error("Failed to get signal stats: %s", e)
            return {}
        finally:
            conn.close()

    def get_unique_query_count(self, profile_scoped: bool = False) -> int:
        """Get number of unique queries with feedback."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            if profile_scoped:
                profile = self._get_active_profile()
                cursor.execute(
                    'SELECT COUNT(DISTINCT query_hash) FROM ranking_feedback WHERE profile = ?',
                    (profile,)
                )
            else:
                cursor.execute(
                    'SELECT COUNT(DISTINCT query_hash) FROM ranking_feedback'
                )
            return cursor.fetchone()[0]
        finally:
            conn.close()

    def get_feedback_for_training(
        self,
        limit: int = 10000,
    ) -> List[Dict[str, Any]]:
        """
        Get feedback records suitable for model training.

        Returns list of dicts with query_hash, memory_id, signal_value, etc.
        Ordered by created_at DESC (newest first).
        """
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT query_hash, query_keywords, memory_id, rank_position,
                       signal_type, signal_value, channel, source_tool,
                       created_at
                FROM ranking_feedback
                ORDER BY created_at DESC
                LIMIT ?
            ''', (limit,))
            return [dict(row) for row in cursor.fetchall()]
        finally:
            conn.close()

    # ======================================================================
    # Transferable Pattern Operations
    # ======================================================================

    def upsert_transferable_pattern(
        self,
        pattern_type: str,
        key: str,
        value: str,
        confidence: float,
        evidence_count: int,
        profiles_seen: int = 1,
        decay_factor: float = 1.0,
        contradictions: Optional[List[str]] = None,
    ) -> int:
        """Insert or update a transferable pattern."""
        now = datetime.now().isoformat()
        contradictions_json = json.dumps(contradictions or [])

        with self._write_lock:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()

                # Check if pattern exists
                cursor.execute(
                    'SELECT id, first_seen FROM transferable_patterns '
                    'WHERE pattern_type = ? AND key = ?',
                    (pattern_type, key)
                )
                existing = cursor.fetchone()

                if existing:
                    cursor.execute('''
                        UPDATE transferable_patterns
                        SET value = ?, confidence = ?, evidence_count = ?,
                            profiles_seen = ?, last_seen = ?, decay_factor = ?,
                            contradictions = ?, updated_at = ?
                        WHERE id = ?
                    ''', (
                        value, confidence, evidence_count,
                        profiles_seen, now, decay_factor,
                        contradictions_json, now, existing['id']
                    ))
                    row_id = existing['id']
                else:
                    cursor.execute('''
                        INSERT INTO transferable_patterns
                            (pattern_type, key, value, confidence, evidence_count,
                             profiles_seen, first_seen, last_seen, decay_factor,
                             contradictions, created_at, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        pattern_type, key, value, confidence, evidence_count,
                        profiles_seen, now, now, decay_factor,
                        contradictions_json, now, now
                    ))
                    row_id = cursor.lastrowid

                conn.commit()
                return row_id
            except Exception as e:
                conn.rollback()
                logger.error("Failed to upsert pattern: %s", e)
                raise
            finally:
                conn.close()

    def get_transferable_patterns(
        self,
        min_confidence: float = 0.0,
        pattern_type: Optional[str] = None,
        profile_scoped: bool = False,
    ) -> List[Dict[str, Any]]:
        """Get transferable patterns filtered by confidence, type, and profile."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            profile_filter = ""
            params = [min_confidence]
            if profile_scoped:
                profile = self._get_active_profile()
                profile_filter = " AND profile = ?"
                params.append(profile)
            if pattern_type:
                cursor.execute(
                    'SELECT * FROM transferable_patterns '
                    'WHERE confidence >= ? AND pattern_type = ?' + profile_filter +
                    ' ORDER BY confidence DESC',
                    tuple(params[:1]) + (pattern_type,) + tuple(params[1:])
                )
            else:
                cursor.execute(
                    'SELECT * FROM transferable_patterns '
                    'WHERE confidence >= ?' + profile_filter +
                    ' ORDER BY confidence DESC',
                    tuple(params)
                )
            return [dict(row) for row in cursor.fetchall()]
        finally:
            conn.close()

    # ======================================================================
    # Workflow Pattern Operations
    # ======================================================================

    def store_workflow_pattern(
        self,
        pattern_type: str,
        pattern_key: str,
        pattern_value: str,
        confidence: float = 0.0,
        evidence_count: int = 0,
        metadata: Optional[Dict] = None,
    ) -> int:
        """Store a workflow pattern (sequence, temporal, or style)."""
        metadata_json = json.dumps(metadata or {})

        with self._write_lock:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO workflow_patterns
                        (pattern_type, pattern_key, pattern_value,
                         confidence, evidence_count, metadata)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    pattern_type, pattern_key, pattern_value,
                    confidence, evidence_count, metadata_json
                ))
                conn.commit()
                return cursor.lastrowid
            except Exception as e:
                conn.rollback()
                logger.error("Failed to store workflow pattern: %s", e)
                raise
            finally:
                conn.close()

    def get_workflow_patterns(
        self,
        pattern_type: Optional[str] = None,
        min_confidence: float = 0.0,
        profile_scoped: bool = False,
    ) -> List[Dict[str, Any]]:
        """Get workflow patterns filtered by type, confidence, and profile."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            profile_filter = ""
            extra_params = []
            if profile_scoped:
                profile = self._get_active_profile()
                profile_filter = " AND profile = ?"
                extra_params.append(profile)
            if pattern_type:
                cursor.execute(
                    'SELECT * FROM workflow_patterns '
                    'WHERE pattern_type = ? AND confidence >= ?' + profile_filter +
                    ' ORDER BY confidence DESC',
                    (pattern_type, min_confidence) + tuple(extra_params)
                )
            else:
                cursor.execute(
                    'SELECT * FROM workflow_patterns '
                    'WHERE confidence >= ?' + profile_filter +
                    ' ORDER BY confidence DESC',
                    (min_confidence,) + tuple(extra_params)
                )
            return [dict(row) for row in cursor.fetchall()]
        finally:
            conn.close()

    def clear_workflow_patterns(self, pattern_type: Optional[str] = None) -> None:
        """Clear workflow patterns (used before re-mining)."""
        with self._write_lock:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                if pattern_type:
                    cursor.execute(
                        'DELETE FROM workflow_patterns WHERE pattern_type = ?',
                        (pattern_type,)
                    )
                else:
                    cursor.execute('DELETE FROM workflow_patterns')
                conn.commit()
            except Exception as e:
                conn.rollback()
                logger.error("Failed to clear workflow patterns: %s", e)
                raise
            finally:
                conn.close()

    # ======================================================================
    # Source Quality Operations
    # ======================================================================

    def update_source_quality(
        self,
        source_id: str,
        positive_signals: int,
        total_memories: int,
    ) -> None:
        """Update quality score for a memory source."""
        # Beta-Binomial smoothing: (alpha + pos) / (alpha + beta + total)
        quality_score = (1.0 + positive_signals) / (2.0 + total_memories)

        with self._write_lock:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO source_quality
                        (source_id, positive_signals, total_memories,
                         quality_score, last_updated)
                    VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
                    ON CONFLICT(source_id) DO UPDATE SET
                        positive_signals = ?,
                        total_memories = ?,
                        quality_score = ?,
                        last_updated = CURRENT_TIMESTAMP
                ''', (
                    source_id, positive_signals, total_memories, quality_score,
                    positive_signals, total_memories, quality_score,
                ))
                conn.commit()
            except Exception as e:
                conn.rollback()
                logger.error("Failed to update source quality: %s", e)
                raise
            finally:
                conn.close()

    def get_source_scores(self, profile_scoped: bool = False) -> Dict[str, float]:
        """Get quality scores for all known sources."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            if profile_scoped:
                profile = self._get_active_profile()
                cursor.execute(
                    'SELECT source_id, quality_score FROM source_quality WHERE profile = ?',
                    (profile,)
                )
            else:
                cursor.execute('SELECT source_id, quality_score FROM source_quality')
            return {row['source_id']: row['quality_score'] for row in cursor.fetchall()}
        finally:
            conn.close()

    # ======================================================================
    # Model Metadata Operations
    # ======================================================================

    def record_model_training(
        self,
        model_version: str,
        training_samples: int,
        synthetic_samples: int = 0,
        real_samples: int = 0,
        ndcg_at_10: Optional[float] = None,
        model_path: Optional[str] = None,
    ) -> int:
        """Record metadata about a trained ranking model."""
        with self._write_lock:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO ranking_models
                        (model_version, training_samples, synthetic_samples,
                         real_samples, ndcg_at_10, model_path)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    model_version, training_samples, synthetic_samples,
                    real_samples, ndcg_at_10, model_path,
                ))
                conn.commit()
                return cursor.lastrowid
            except Exception as e:
                conn.rollback()
                logger.error("Failed to record model training: %s", e)
                raise
            finally:
                conn.close()

    def get_latest_model(self) -> Optional[Dict[str, Any]]:
        """Get metadata for the most recently trained model."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM ranking_models
                ORDER BY created_at DESC
                LIMIT 1
            ''')
            row = cursor.fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    # ======================================================================
    # Engagement Metrics Operations
    # ======================================================================

    def increment_engagement(
        self,
        metric_type: str,
        count: int = 1,
        source: Optional[str] = None,
    ) -> None:
        """
        Increment a daily engagement metric.

        Args:
            metric_type: One of 'memories_created', 'recalls_performed',
                        'feedback_signals', 'patterns_updated'
            count: Increment amount (default 1)
            source: Source tool identifier to track in active_sources
        """
        today = date.today().isoformat()
        valid_metrics = {
            'memories_created', 'recalls_performed',
            'feedback_signals', 'patterns_updated',
        }
        if metric_type not in valid_metrics:
            logger.warning("Invalid metric type: %s", metric_type)
            return

        with self._write_lock:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()

                # Ensure today's row exists
                cursor.execute('''
                    INSERT OR IGNORE INTO engagement_metrics (metric_date)
                    VALUES (?)
                ''', (today,))

                # Increment the specific metric
                cursor.execute(f'''
                    UPDATE engagement_metrics
                    SET {metric_type} = {metric_type} + ?
                    WHERE metric_date = ?
                ''', (count, today))

                # Update active sources if provided
                if source:
                    cursor.execute('''
                        SELECT active_sources FROM engagement_metrics
                        WHERE metric_date = ?
                    ''', (today,))
                    row = cursor.fetchone()
                    if row:
                        sources = json.loads(row['active_sources'] or '[]')
                        if source not in sources:
                            sources.append(source)
                            cursor.execute('''
                                UPDATE engagement_metrics
                                SET active_sources = ?
                                WHERE metric_date = ?
                            ''', (json.dumps(sources), today))

                conn.commit()
            except Exception as e:
                conn.rollback()
                logger.error("Failed to update engagement: %s", e)
            finally:
                conn.close()

    def get_engagement_history(
        self,
        days: int = 30,
    ) -> List[Dict[str, Any]]:
        """Get engagement metrics for the last N days."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM engagement_metrics
                ORDER BY metric_date DESC
                LIMIT ?
            ''', (days,))
            return [dict(row) for row in cursor.fetchall()]
        finally:
            conn.close()

    # ======================================================================
    # Statistics & Diagnostics
    # ======================================================================

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive learning database statistics."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            stats = {}

            # Feedback stats
            cursor.execute('SELECT COUNT(*) FROM ranking_feedback')
            stats['feedback_count'] = cursor.fetchone()[0]

            cursor.execute(
                'SELECT COUNT(DISTINCT query_hash) FROM ranking_feedback'
            )
            stats['unique_queries'] = cursor.fetchone()[0]

            # Pattern stats
            cursor.execute('SELECT COUNT(*) FROM transferable_patterns')
            stats['transferable_patterns'] = cursor.fetchone()[0]

            cursor.execute(
                'SELECT COUNT(*) FROM transferable_patterns '
                'WHERE confidence >= 0.6'
            )
            stats['high_confidence_patterns'] = cursor.fetchone()[0]

            # Workflow stats
            cursor.execute('SELECT COUNT(*) FROM workflow_patterns')
            stats['workflow_patterns'] = cursor.fetchone()[0]

            # Source quality stats
            cursor.execute('SELECT COUNT(*) FROM source_quality')
            stats['tracked_sources'] = cursor.fetchone()[0]

            # Model stats
            cursor.execute(
                'SELECT COUNT(*) FROM ranking_models'
            )
            stats['models_trained'] = cursor.fetchone()[0]

            latest_model = self.get_latest_model()
            if latest_model:
                stats['latest_model_version'] = latest_model['model_version']
                stats['latest_model_ndcg'] = latest_model['ndcg_at_10']
            else:
                stats['latest_model_version'] = None
                stats['latest_model_ndcg'] = None

            # DB file size
            if self.db_path.exists():
                stats['db_size_bytes'] = self.db_path.stat().st_size
                stats['db_size_kb'] = round(stats['db_size_bytes'] / 1024, 1)
            else:
                stats['db_size_bytes'] = 0
                stats['db_size_kb'] = 0

            return stats
        finally:
            conn.close()

    # ======================================================================
    # v2.8.0: Action Outcomes CRUD
    # ======================================================================

    def store_outcome(self, memory_ids: Any, outcome: str, action_type: str = "other", context: Optional[Dict] = None, confidence: float = 1.0, agent_id: str = "user", project: Optional[str] = None, profile: str = "default") -> int:
        """Store an action outcome for behavioral learning."""
        memory_ids_str = json.dumps(memory_ids if isinstance(memory_ids, list) else [memory_ids])
        context_str = json.dumps(context or {})
        conn = self._get_connection()
        try:
            with self._write_lock:
                cursor = conn.execute(
                    "INSERT INTO action_outcomes (memory_ids, outcome, action_type, context, confidence, agent_id, project, profile) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (memory_ids_str, outcome, action_type, context_str, confidence, agent_id, project, profile),
                )
                conn.commit()
                return cursor.lastrowid
        finally:
            conn.close()

    def get_outcomes(self, memory_id: Optional[int] = None, project: Optional[str] = None, profile: str = "default", limit: int = 100) -> List[Dict[str, Any]]:
        """Get action outcomes, optionally filtered."""
        conn = self._get_connection()
        try:
            query = "SELECT * FROM action_outcomes WHERE profile = ?"
            params = [profile]
            if project:
                query += " AND project = ?"
                params.append(project)
            query += " ORDER BY created_at DESC LIMIT ?"
            params.append(limit)
            rows = conn.execute(query, params).fetchall()
            results = []
            for row in rows:
                d = dict(row)
                d["memory_ids"] = json.loads(d["memory_ids"])
                d["context"] = json.loads(d["context"])
                if memory_id and memory_id not in d["memory_ids"]:
                    continue
                results.append(d)
            return results
        finally:
            conn.close()

    # ======================================================================
    # v2.8.0: Behavioral Patterns CRUD
    # ======================================================================

    def store_behavioral_pattern(self, pattern_type: str, pattern_key: str, success_rate: float = 0.0, evidence_count: int = 0, confidence: float = 0.0, metadata: Optional[Dict] = None, project: Optional[str] = None, profile: str = "default") -> int:
        """Store or update a behavioral pattern."""
        metadata_str = json.dumps(metadata or {})
        conn = self._get_connection()
        try:
            with self._write_lock:
                cursor = conn.execute(
                    "INSERT INTO behavioral_patterns (pattern_type, pattern_key, success_rate, evidence_count, confidence, metadata, project, profile) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (pattern_type, pattern_key, success_rate, evidence_count, confidence, metadata_str, project, profile),
                )
                conn.commit()
                return cursor.lastrowid
        finally:
            conn.close()

    def get_behavioral_patterns(self, pattern_type: Optional[str] = None, project: Optional[str] = None, min_confidence: float = 0.0, profile: str = "default") -> List[Dict[str, Any]]:
        """Get behavioral patterns, optionally filtered."""
        conn = self._get_connection()
        try:
            query = "SELECT * FROM behavioral_patterns WHERE profile = ? AND confidence >= ?"
            params = [profile, min_confidence]
            if pattern_type:
                query += " AND pattern_type = ?"
                params.append(pattern_type)
            if project:
                query += " AND project = ?"
                params.append(project)
            query += " ORDER BY confidence DESC"
            rows = conn.execute(query, params).fetchall()
            results = []
            for row in rows:
                d = dict(row)
                d["metadata"] = json.loads(d["metadata"])
                results.append(d)
            return results
        finally:
            conn.close()

    # ======================================================================
    # v2.8.0: Cross-Project CRUD
    # ======================================================================

    def store_cross_project(self, source_project: str, target_project: str, pattern_id: int, transfer_type: str = "metadata", confidence: float = 0.0, profile: str = "default") -> int:
        """Record a cross-project behavioral transfer."""
        conn = self._get_connection()
        try:
            with self._write_lock:
                cursor = conn.execute(
                    "INSERT INTO cross_project_behaviors (source_project, target_project, pattern_id, transfer_type, confidence, profile) VALUES (?, ?, ?, ?, ?, ?)",
                    (source_project, target_project, pattern_id, transfer_type, confidence, profile),
                )
                conn.commit()
                return cursor.lastrowid
        finally:
            conn.close()

    def get_cross_project_transfers(self, source_project: Optional[str] = None, target_project: Optional[str] = None, profile: str = "default") -> List[Dict[str, Any]]:
        """Get cross-project transfer records."""
        conn = self._get_connection()
        try:
            query = "SELECT * FROM cross_project_behaviors WHERE profile = ?"
            params = [profile]
            if source_project:
                query += " AND source_project = ?"
                params.append(source_project)
            if target_project:
                query += " AND target_project = ?"
                params.append(target_project)
            query += " ORDER BY created_at DESC"
            return [dict(row) for row in conn.execute(query, params).fetchall()]
        finally:
            conn.close()

    # ======================================================================
    # Reset / Cleanup
    # ======================================================================

    def reset(self) -> None:
        """
        Delete all learning data. Memories in memory.db are preserved.

        This is the GDPR Article 17 "Right to Erasure" handler for
        behavioral data.
        """
        with self._write_lock:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute('DELETE FROM ranking_feedback')
                cursor.execute('DELETE FROM transferable_patterns')
                cursor.execute('DELETE FROM workflow_patterns')
                cursor.execute('DELETE FROM ranking_models')
                cursor.execute('DELETE FROM source_quality')
                cursor.execute('DELETE FROM engagement_metrics')
                conn.commit()
                logger.info(
                    "Learning data reset. Memories in memory.db preserved."
                )
            except Exception as e:
                conn.rollback()
                logger.error("Failed to reset learning data: %s", e)
                raise
            finally:
                conn.close()

    def delete_database(self) -> None:
        """
        Completely delete learning.db file.
        More aggressive than reset() — removes the file entirely.
        """
        with self._write_lock:
            LearningDB.reset_instance(self.db_path)
            if self.db_path.exists():
                self.db_path.unlink()
                logger.info("Learning database deleted: %s", self.db_path)
                # Also clean WAL/SHM files
                wal = self.db_path.with_suffix('.db-wal')
                shm = self.db_path.with_suffix('.db-shm')
                if wal.exists():
                    wal.unlink()
                if shm.exists():
                    shm.unlink()
