#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 SuperLocalMemory (superlocalmemory.com)
"""
MemoryStore V2 - Extended Memory System with Tree and Graph Support
Maintains backward compatibility with V1 API while adding:
- Tree hierarchy (parent_id, tree_path, depth)
- Categories and clusters
- Tier-based progressive summarization
- Enhanced search with tier filtering
"""

import sqlite3
import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from contextlib import contextmanager

# Connection Manager (v2.5+) — fixes "database is locked" with multiple agents
try:
    from db_connection_manager import DbConnectionManager
    USE_CONNECTION_MANAGER = True
except ImportError:
    USE_CONNECTION_MANAGER = False

# Event Bus (v2.5+) — real-time event broadcasting
try:
    from event_bus import EventBus
    USE_EVENT_BUS = True
except ImportError:
    USE_EVENT_BUS = False

# Agent Registry + Provenance (v2.5+) — tracks who writes what
try:
    from agent_registry import AgentRegistry
    from provenance_tracker import ProvenanceTracker
    USE_PROVENANCE = True
except ImportError:
    USE_PROVENANCE = False

# Trust Scorer (v2.5+) — silent signal collection, no enforcement
try:
    from trust_scorer import TrustScorer
    USE_TRUST = True
except ImportError:
    USE_TRUST = False

# TF-IDF for local semantic search (no external APIs)
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

import logging
logger = logging.getLogger(__name__)

# Import constants and utilities from memory package
from memory.constants import (
    MEMORY_DIR, DB_PATH, VECTORS_PATH,
    MAX_CONTENT_SIZE, MAX_SUMMARY_SIZE, MAX_TAG_LENGTH, MAX_TAGS,
    CREATOR_METADATA
)
from memory.schema import (
    V2_COLUMNS, V28_MIGRATIONS, V2_INDEXES,
    get_memories_table_sql, get_sessions_table_sql, get_fts_table_sql,
    get_fts_trigger_insert_sql, get_fts_trigger_delete_sql, get_fts_trigger_update_sql,
    get_creator_metadata_table_sql
)
from memory.helpers import format_content


class MemoryStoreV2:
    """
    Extended memory store with hierarchical tree and graph integration.

    Key Features:
    - Tree hierarchy via parent_id and materialized paths
    - Category-based organization
    - GraphRAG cluster integration
    - Tier-based access tracking
    - Backward compatible with V1 API
    """

    def __init__(self, db_path: Optional[Path] = None, profile: Optional[str] = None):
        """
        Initialize MemoryStore V2.

        Args:
            db_path: Optional custom database path (defaults to ~/.claude-memory/memory.db)
            profile: Optional profile override. If None, reads from profiles.json config.
        """
        self.db_path = db_path or DB_PATH
        self.vectors_path = VECTORS_PATH
        self._profile_override = profile

        # Connection Manager (v2.5+) — thread-safe WAL + write queue
        # Falls back to direct sqlite3.connect() if unavailable
        self._db_mgr = None
        if USE_CONNECTION_MANAGER:
            try:
                self._db_mgr = DbConnectionManager.get_instance(self.db_path)
            except Exception:
                pass  # Fall back to direct connections

        # Event Bus (v2.5+) — real-time event broadcasting
        # If unavailable, events simply don't fire (core ops unaffected)
        self._event_bus = None
        if USE_EVENT_BUS:
            try:
                self._event_bus = EventBus.get_instance(self.db_path)
            except Exception:
                pass

        self._init_db()

        # Agent Registry + Provenance (v2.5+)
        # MUST run AFTER _init_db() — ProvenanceTracker ALTER TABLEs the memories table
        self._agent_registry = None
        self._provenance_tracker = None
        if USE_PROVENANCE:
            try:
                self._agent_registry = AgentRegistry.get_instance(self.db_path)
                self._provenance_tracker = ProvenanceTracker.get_instance(self.db_path)
            except Exception:
                pass

        # Trust Scorer (v2.5+) — silent signal collection
        self._trust_scorer = None
        if USE_TRUST:
            try:
                self._trust_scorer = TrustScorer.get_instance(self.db_path)
            except Exception:
                pass

        self.vectorizer = None
        self.vectors = None
        self.memory_ids = []
        self._last_vector_count = 0
        self._load_vectors()

        # HNSW index for O(log n) search (v2.6, optional)
        self._hnsw_index = None
        try:
            from hnsw_index import HNSWIndex
            if self.vectors is not None and len(self.memory_ids) > 0:
                dim = self.vectors.shape[1]
                self._hnsw_index = HNSWIndex(dimension=dim, max_elements=max(len(self.memory_ids) * 2, 1000))
                self._hnsw_index.build(self.vectors.toarray() if hasattr(self.vectors, 'toarray') else self.vectors, self.memory_ids)
                logger.info("HNSW index built with %d vectors", len(self.memory_ids))
        except (ImportError, Exception) as e:
            logger.debug("HNSW index not available: %s", e)
            self._hnsw_index = None

    # =========================================================================
    # Connection helpers — abstract ConnectionManager vs direct sqlite3
    # =========================================================================

    @contextmanager
    def _read_connection(self):
        """
        Context manager for read operations.
        Uses ConnectionManager pool if available, else direct sqlite3.connect().
        """
        if self._db_mgr:
            with self._db_mgr.read_connection() as conn:
                yield conn
        else:
            conn = sqlite3.connect(self.db_path)
            try:
                yield conn
            finally:
                conn.close()

    def _execute_write(self, callback):
        """
        Execute a write operation (INSERT/UPDATE/DELETE).
        Uses ConnectionManager write queue if available, else direct sqlite3.connect().

        Args:
            callback: Function(conn) that performs writes and calls conn.commit()

        Returns:
            Whatever the callback returns
        """
        if self._db_mgr:
            return self._db_mgr.execute_write(callback)
        else:
            conn = sqlite3.connect(self.db_path)
            try:
                result = callback(conn)
                return result
            finally:
                conn.close()

    def _emit_event(self, event_type: str, memory_id: Optional[int] = None, **kwargs):
        """
        Emit an event to the Event Bus (v2.5+).

        Progressive enhancement: if Event Bus is unavailable, this is a no-op.
        Event emission failure must NEVER break core memory operations.

        Args:
            event_type: Event type (e.g., "memory.created")
            memory_id: Associated memory ID (if applicable)
            **kwargs: Additional payload fields
        """
        if not self._event_bus:
            return
        try:
            self._event_bus.emit(
                event_type=event_type,
                memory_id=memory_id,
                payload=kwargs,
                importance=kwargs.get("importance", 5),
            )
        except Exception:
            pass  # Event bus failure must never break core operations

    def _get_active_profile(self) -> str:
        """
        Get the currently active profile name.
        Reads from profiles.json config file. Falls back to 'default'.
        """
        if self._profile_override:
            return self._profile_override

        config_file = MEMORY_DIR / "profiles.json"
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    config = json.load(f)
                return config.get('active_profile', 'default')
            except (json.JSONDecodeError, IOError):
                pass
        return 'default'

    def _init_db(self):
        """Initialize SQLite database with V2 schema extensions."""
        def _do_init(conn):
            cursor = conn.cursor()

            # Database integrity check (v2.6: detect corruption early)
            try:
                result = cursor.execute('PRAGMA quick_check').fetchone()
                if result[0] != 'ok':
                    logger.warning("Database integrity issue detected: %s", result[0])
            except Exception:
                logger.warning("Could not run database integrity check")

            # Check if we need to add V2 columns to existing table
            cursor.execute("PRAGMA table_info(memories)")
            existing_columns = {row[1] for row in cursor.fetchall()}

            # Main memories table (V1 compatible + V2 extensions)
            cursor.execute(get_memories_table_sql())

            # Add missing V2 columns to existing table (migration support)
            # This handles upgrades from very old databases that might be missing columns
            for col_name, col_type in V2_COLUMNS.items():
                if col_name not in existing_columns:
                    try:
                        cursor.execute(f'ALTER TABLE memories ADD COLUMN {col_name} {col_type}')
                    except sqlite3.OperationalError:
                        # Column might already exist from concurrent migration
                        pass

            # v2.8.0 schema migration — lifecycle + access control columns
            for col_name, col_type in V28_MIGRATIONS:
                try:
                    cursor.execute(f"ALTER TABLE memories ADD COLUMN {col_name} {col_type}")
                except sqlite3.OperationalError:
                    pass  # Column already exists

            # Sessions table (V1 compatible)
            cursor.execute(get_sessions_table_sql())

            # Full-text search index (V1 compatible)
            cursor.execute(get_fts_table_sql())

            # FTS Triggers (V1 compatible)
            cursor.execute(get_fts_trigger_insert_sql())
            cursor.execute(get_fts_trigger_delete_sql())
            cursor.execute(get_fts_trigger_update_sql())

            # Create indexes for V2 fields (safe for old databases without V2 columns)
            for idx_name, col_name in V2_INDEXES:
                try:
                    cursor.execute(f'CREATE INDEX IF NOT EXISTS {idx_name} ON memories({col_name})')
                except sqlite3.OperationalError:
                    # Column doesn't exist yet (old database) - skip index creation
                    # Index will be created automatically on next schema upgrade
                    pass

            # v2.8.0 indexes for lifecycle + access control
            try:
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_lifecycle_state ON memories(lifecycle_state)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_access_level ON memories(access_level)")
            except sqlite3.OperationalError:
                pass

            # Creator Attribution Metadata Table (REQUIRED by MIT License)
            # This table embeds creator information directly in the database
            cursor.execute(get_creator_metadata_table_sql())

            # Insert creator attribution (embedded in database body)
            for key, value in CREATOR_METADATA.items():
                cursor.execute('''
                    INSERT OR IGNORE INTO creator_metadata (key, value)
                    VALUES (?, ?)
                ''', (key, value))

            conn.commit()

        self._execute_write(_do_init)

    def _content_hash(self, content: str) -> str:
        """Generate hash for deduplication."""
        return hashlib.sha256(content.encode()).hexdigest()[:32]

    def add_memory(
        self,
        content: str,
        summary: Optional[str] = None,
        project_path: Optional[str] = None,
        project_name: Optional[str] = None,
        tags: Optional[List[str]] = None,
        category: Optional[str] = None,
        parent_id: Optional[int] = None,
        memory_type: str = "session",
        importance: int = 5
    ) -> int:
        """
        Add a new memory with V2 enhancements.

        Args:
            content: Memory content (required, max 1MB)
            summary: Optional summary (max 10KB)
            project_path: Project absolute path
            project_name: Human-readable project name
            tags: List of tags (max 20 tags, 50 chars each)
            category: High-level category (e.g., "frontend", "backend")
            parent_id: Parent memory ID for hierarchical nesting
            memory_type: Type of memory ('session', 'long-term', 'reference')

        Raises:
            TypeError: If content is not a string
            ValueError: If content is empty or exceeds size limits

        Returns:
            Memory ID (int), or existing ID if duplicate detected
        """
        # SECURITY: Input validation
        if not isinstance(content, str):
            raise TypeError("Content must be a string")

        content = content.strip()
        if not content:
            raise ValueError("Content cannot be empty")

        if len(content) > MAX_CONTENT_SIZE:
            raise ValueError(f"Content exceeds maximum size of {MAX_CONTENT_SIZE} bytes")

        if summary and len(summary) > MAX_SUMMARY_SIZE:
            raise ValueError(f"Summary exceeds maximum size of {MAX_SUMMARY_SIZE} bytes")

        if tags:
            if len(tags) > MAX_TAGS:
                raise ValueError(f"Too many tags (max {MAX_TAGS})")
            for tag in tags:
                if len(tag) > MAX_TAG_LENGTH:
                    raise ValueError(f"Tag '{tag[:20]}...' exceeds max length of {MAX_TAG_LENGTH}")

        if importance < 1 or importance > 10:
            importance = max(1, min(10, importance))  # Clamp to valid range

        content_hash = self._content_hash(content)
        active_profile = self._get_active_profile()

        def _do_add(conn):
            cursor = conn.cursor()

            try:
                # Calculate tree_path and depth
                tree_path, depth = self._calculate_tree_position(cursor, parent_id)

                cursor.execute('''
                    INSERT INTO memories (
                        content, summary, project_path, project_name, tags, category,
                        parent_id, tree_path, depth,
                        memory_type, importance, content_hash,
                        last_accessed, access_count, profile
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    content,
                    summary,
                    project_path,
                    project_name,
                    json.dumps(tags) if tags else None,
                    category,
                    parent_id,
                    tree_path,
                    depth,
                    memory_type,
                    importance,
                    content_hash,
                    datetime.now().isoformat(),
                    0,
                    active_profile
                ))
                memory_id = cursor.lastrowid

                # Update tree_path with actual memory_id
                if tree_path:
                    tree_path = f"{tree_path}.{memory_id}"
                else:
                    tree_path = str(memory_id)

                cursor.execute('UPDATE memories SET tree_path = ? WHERE id = ?', (tree_path, memory_id))

                conn.commit()
                return memory_id

            except sqlite3.IntegrityError:
                # Duplicate content
                cursor.execute('SELECT id FROM memories WHERE content_hash = ?', (content_hash,))
                result = cursor.fetchone()
                return result[0] if result else -1

        memory_id = self._execute_write(_do_add)

        # Rebuild vectors after adding (reads only — outside write callback)
        self._rebuild_vectors()

        # Emit event (v2.5 — Event Bus)
        self._emit_event("memory.created", memory_id=memory_id,
                         content_preview="[redacted]", tags=tags,
                         project=project_name, importance=importance)

        # Record provenance (v2.5 — who created this memory)
        if self._provenance_tracker:
            try:
                self._provenance_tracker.record_provenance(memory_id)
            except Exception:
                pass  # Provenance failure must never break core

        # Trust signal (v2.5 — silent collection)
        if self._trust_scorer:
            try:
                self._trust_scorer.on_memory_created("user", memory_id, importance)
            except Exception:
                pass  # Trust failure must never break core

        # Auto-backup check (non-blocking)
        try:
            from auto_backup import AutoBackup
            backup = AutoBackup()
            backup.check_and_backup()
        except Exception:
            pass  # Backup failure must never break memory operations

        return memory_id

    def _calculate_tree_position(self, cursor: sqlite3.Cursor, parent_id: Optional[int]) -> Tuple[str, int]:
        """
        Calculate tree_path and depth for a new memory.

        Args:
            cursor: Database cursor
            parent_id: Parent memory ID (None for root level)

        Returns:
            Tuple of (tree_path, depth)
        """
        if parent_id is None:
            return ("", 0)

        cursor.execute('SELECT tree_path, depth FROM memories WHERE id = ?', (parent_id,))
        result = cursor.fetchone()

        if result:
            parent_path, parent_depth = result
            return (parent_path, parent_depth + 1)
        else:
            # Parent not found, treat as root
            return ("", 0)

    def search(
        self,
        query: str,
        limit: int = 5,
        project_path: Optional[str] = None,
        memory_type: Optional[str] = None,
        category: Optional[str] = None,
        cluster_id: Optional[int] = None,
        min_importance: Optional[int] = None,
        lifecycle_states: Optional[tuple] = None,
        agent_context: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search memories with enhanced V2 filtering.

        Args:
            query: Search query string
            limit: Maximum results to return
            project_path: Filter by project path
            memory_type: Filter by memory type
            category: Filter by category
            cluster_id: Filter by graph cluster
            min_importance: Minimum importance score
            lifecycle_states: Tuple of lifecycle states to include (default: active, warm)

        Returns:
            List of memory dictionaries with scores
        """
        if lifecycle_states is None:
            lifecycle_states = ("active", "warm")

        results = []
        active_profile = self._get_active_profile()

        with self._read_connection() as conn:
            # Method 0: HNSW accelerated search (O(log n), v2.6)
            _hnsw_used = False
            if SKLEARN_AVAILABLE and self.vectorizer is not None and self.vectors is not None:
                try:
                    from hnsw_index import HNSWIndex
                    if hasattr(self, '_hnsw_index') and self._hnsw_index is not None:
                        query_vec = self.vectorizer.transform([query]).toarray().flatten()
                        hnsw_results = self._hnsw_index.search(query_vec, k=limit * 2)
                        cursor = conn.cursor()
                        for memory_id, score in hnsw_results:
                            if score > 0.05:
                                cursor.execute('''
                                    SELECT id, content, summary, project_path, project_name, tags,
                                           category, parent_id, tree_path, depth,
                                           memory_type, importance, created_at, cluster_id,
                                           last_accessed, access_count, lifecycle_state
                                    FROM memories WHERE id = ? AND profile = ?
                                ''', (memory_id, active_profile))
                                row = cursor.fetchone()
                                if row and self._apply_filters(row, project_path, memory_type,
                                                              category, cluster_id, min_importance, lifecycle_states):
                                    results.append(self._row_to_dict(row, score, 'hnsw'))
                        _hnsw_used = len(results) > 0
                except (ImportError, Exception):
                    pass  # HNSW not available, fall through to TF-IDF

            # Method 1: TF-IDF semantic search (fallback if HNSW unavailable or returned no results)
            if not _hnsw_used and SKLEARN_AVAILABLE and self.vectorizer is not None and self.vectors is not None:
                try:
                    query_vec = self.vectorizer.transform([query])
                    similarities = cosine_similarity(query_vec, self.vectors).flatten()
                    top_indices = np.argsort(similarities)[::-1][:limit * 2]

                    cursor = conn.cursor()

                    for idx in top_indices:
                        if idx < len(self.memory_ids):
                            memory_id = self.memory_ids[idx]
                            score = float(similarities[idx])

                            if score > 0.05:  # Minimum relevance threshold
                                cursor.execute('''
                                    SELECT id, content, summary, project_path, project_name, tags,
                                           category, parent_id, tree_path, depth,
                                           memory_type, importance, created_at, cluster_id,
                                           last_accessed, access_count, lifecycle_state
                                    FROM memories WHERE id = ? AND profile = ?
                                ''', (memory_id, active_profile))
                                row = cursor.fetchone()

                                if row and self._apply_filters(row, project_path, memory_type,
                                                              category, cluster_id, min_importance, lifecycle_states):
                                    results.append(self._row_to_dict(row, score, 'semantic'))

                except Exception as e:
                    print(f"Semantic search error: {e}")

            # Method 2: FTS fallback/supplement
            cursor = conn.cursor()

            # Clean query for FTS
            import re
            fts_query = ' OR '.join(re.findall(r'\w+', query))

            if fts_query:
                cursor.execute('''
                    SELECT m.id, m.content, m.summary, m.project_path, m.project_name,
                           m.tags, m.category, m.parent_id, m.tree_path, m.depth,
                           m.memory_type, m.importance, m.created_at, m.cluster_id,
                           m.last_accessed, m.access_count, m.lifecycle_state
                    FROM memories m
                    JOIN memories_fts fts ON m.id = fts.rowid
                    WHERE memories_fts MATCH ? AND m.profile = ?
                    ORDER BY rank
                    LIMIT ?
                ''', (fts_query, active_profile, limit))

                existing_ids = {r['id'] for r in results}

                for row in cursor.fetchall():
                    if row[0] not in existing_ids:
                        if self._apply_filters(row, project_path, memory_type,
                                              category, cluster_id, min_importance, lifecycle_states):
                            results.append(self._row_to_dict(row, 0.5, 'keyword'))

        # Update access tracking for returned results
        self._update_access_tracking([r['id'] for r in results])

        # Reactivate warm memories that were recalled (lifecycle v2.8)
        warm_ids = [r['id'] for r in results if r.get('lifecycle_state') == 'warm']
        if warm_ids:
            try:
                from lifecycle.lifecycle_engine import LifecycleEngine
                engine = LifecycleEngine(self.db_path)
                for mem_id in warm_ids:
                    engine.reactivate_memory(mem_id, trigger="recall")
            except (ImportError, Exception):
                pass  # Lifecycle engine not available

        # Sort by score and limit
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:limit]

    def _apply_filters(
        self,
        row: tuple,
        project_path: Optional[str],
        memory_type: Optional[str],
        category: Optional[str],
        cluster_id: Optional[int],
        min_importance: Optional[int],
        lifecycle_states: Optional[tuple] = None,
    ) -> bool:
        """Apply filter criteria to a database row."""
        # Row indices: project_path=3, category=6, memory_type=10, importance=11, cluster_id=13
        if project_path and row[3] != project_path:
            return False
        if memory_type and row[10] != memory_type:
            return False
        if category and row[6] != category:
            return False
        if cluster_id is not None and row[13] != cluster_id:
            return False
        if min_importance is not None and (row[11] or 0) < min_importance:
            return False
        # Lifecycle state filter (v2.8) — index 16 if present
        if lifecycle_states and len(row) > 16:
            state = row[16] or "active"
            if state not in lifecycle_states:
                return False
        return True

    def _check_abac(
        self,
        subject: Dict[str, Any],
        resource: Dict[str, Any],
        action: str,
        policy_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Check ABAC policy for an access request.

        Returns {"allowed": True/False, "reason": str}.
        When ABAC engine is unavailable (import error, missing file),
        defaults to allow for backward compatibility with v2.7.
        """
        try:
            from compliance.abac_engine import ABACEngine
            if policy_path is None:
                policy_path = str(Path(self.db_path).parent / "abac_policies.json")
            engine = ABACEngine(config_path=policy_path)
            return engine.evaluate(subject=subject, resource=resource, action=action)
        except (ImportError, Exception):
            return {"allowed": True, "reason": "ABAC unavailable — default allow"}

    def _row_to_dict(self, row: tuple, score: float, match_type: str) -> Dict[str, Any]:
        """Convert database row to memory dictionary."""
        # Backward compatibility: Handle both JSON array and comma-separated string tags
        tags_raw = row[5]
        if tags_raw:
            try:
                # Try parsing as JSON (v2.1.0+ format)
                tags = json.loads(tags_raw)
            except (json.JSONDecodeError, TypeError):
                # Fall back to comma-separated string (v2.0.0 format)
                tags = [t.strip() for t in str(tags_raw).split(',') if t.strip()]
        else:
            tags = []

        return {
            'id': row[0],
            'content': row[1],
            'summary': row[2],
            'project_path': row[3],
            'project_name': row[4],
            'tags': tags,
            'category': row[6],
            'parent_id': row[7],
            'tree_path': row[8],
            'depth': row[9],
            'memory_type': row[10],
            'importance': row[11],
            'created_at': row[12],
            'cluster_id': row[13],
            'last_accessed': row[14],
            'access_count': row[15],
            'lifecycle_state': row[16] if len(row) > 16 else 'active',
            'score': score,
            'match_type': match_type
        }

    def _update_access_tracking(self, memory_ids: List[int]):
        """Update last_accessed and access_count for retrieved memories."""
        if not memory_ids:
            return

        def _do_update(conn):
            cursor = conn.cursor()
            now = datetime.now().isoformat()
            for mem_id in memory_ids:
                cursor.execute('''
                    UPDATE memories
                    SET last_accessed = ?, access_count = access_count + 1
                    WHERE id = ?
                ''', (now, mem_id))
            conn.commit()

        self._execute_write(_do_update)

    def get_tree(self, parent_id: Optional[int] = None, max_depth: int = 3) -> List[Dict[str, Any]]:
        """
        Get hierarchical tree structure of memories.

        Args:
            parent_id: Root parent ID (None for top-level)
            max_depth: Maximum depth to retrieve

        Returns:
            List of memories with tree structure
        """
        active_profile = self._get_active_profile()

        with self._read_connection() as conn:
            cursor = conn.cursor()

            if parent_id is None:
                # Get root level memories
                cursor.execute('''
                    SELECT id, content, summary, project_path, project_name, tags,
                           category, parent_id, tree_path, depth, memory_type, importance,
                           created_at, cluster_id, last_accessed, access_count
                    FROM memories
                    WHERE parent_id IS NULL AND depth <= ? AND profile = ?
                    ORDER BY tree_path
                ''', (max_depth, active_profile))
            else:
                # Get subtree under specific parent
                cursor.execute('''
                    SELECT tree_path FROM memories WHERE id = ?
                ''', (parent_id,))
                result = cursor.fetchone()

                if not result:
                    return []

                parent_path = result[0]
                cursor.execute('''
                    SELECT id, content, summary, project_path, project_name, tags,
                           category, parent_id, tree_path, depth, memory_type, importance,
                           created_at, cluster_id, last_accessed, access_count
                    FROM memories
                    WHERE tree_path LIKE ? AND depth <= ?
                    ORDER BY tree_path
                ''', (f"{parent_path}.%", max_depth))

            results = []
            for row in cursor.fetchall():
                results.append(self._row_to_dict(row, 1.0, 'tree'))

        return results

    def update_tier(self, memory_id: int, new_tier: str, compressed_summary: Optional[str] = None):
        """
        Update memory tier for progressive summarization.

        Args:
            memory_id: Memory ID to update
            new_tier: New tier level ('hot', 'warm', 'cold', 'archived')
            compressed_summary: Optional compressed summary for higher tiers
        """
        def _do_update(conn):
            cursor = conn.cursor()
            if compressed_summary:
                cursor.execute('''
                    UPDATE memories
                    SET memory_type = ?, summary = ?, updated_at = ?
                    WHERE id = ?
                ''', (new_tier, compressed_summary, datetime.now().isoformat(), memory_id))
            else:
                cursor.execute('''
                    UPDATE memories
                    SET memory_type = ?, updated_at = ?
                    WHERE id = ?
                ''', (new_tier, datetime.now().isoformat(), memory_id))
            conn.commit()

        self._execute_write(_do_update)

        # Emit event (v2.5)
        self._emit_event("memory.updated", memory_id=memory_id, new_tier=new_tier)

    def get_by_cluster(self, cluster_id: int) -> List[Dict[str, Any]]:
        """
        Get all memories in a specific graph cluster.

        Args:
            cluster_id: Graph cluster ID

        Returns:
            List of memories in the cluster
        """
        active_profile = self._get_active_profile()

        with self._read_connection() as conn:
            cursor = conn.cursor()

            cursor.execute('''
                SELECT id, content, summary, project_path, project_name, tags,
                       category, parent_id, tree_path, depth, memory_type, importance,
                       created_at, cluster_id, last_accessed, access_count
                FROM memories
                WHERE cluster_id = ? AND profile = ?
                ORDER BY importance DESC, created_at DESC
            ''', (cluster_id, active_profile))

            results = []
            for row in cursor.fetchall():
                results.append(self._row_to_dict(row, 1.0, 'cluster'))

        return results

    # ========== V1 Backward Compatible Methods ==========

    def _load_vectors(self):
        """Load vectors by rebuilding from database (V1 compatible)."""
        self._rebuild_vectors()

    def _rebuild_vectors(self):
        """Rebuild TF-IDF vectors from active profile memories (V1 compatible, backward compatible)."""
        if not SKLEARN_AVAILABLE:
            return

        # Incremental optimization: skip rebuild if memory count hasn't changed much (v2.6)
        if hasattr(self, '_last_vector_count') and self._last_vector_count > 0:
            with self._read_connection() as conn:
                cursor = conn.cursor()
                active_profile = self._get_active_profile()
                cursor.execute("PRAGMA table_info(memories)")
                columns = {row[1] for row in cursor.fetchall()}
                if 'profile' in columns:
                    cursor.execute('SELECT COUNT(*) FROM memories WHERE profile = ?', (active_profile,))
                else:
                    cursor.execute('SELECT COUNT(*) FROM memories')
                current_count = cursor.fetchone()[0]

            # Only rebuild if count changed by more than 5% or is the first few memories
            if self._last_vector_count > 10:
                change_ratio = abs(current_count - self._last_vector_count) / self._last_vector_count
                if change_ratio < 0.05:
                    return  # Skip rebuild — vectors are still accurate enough

        active_profile = self._get_active_profile()

        with self._read_connection() as conn:
            cursor = conn.cursor()

            # Check which columns exist (backward compatibility for old databases)
            cursor.execute("PRAGMA table_info(memories)")
            columns = {row[1] for row in cursor.fetchall()}

            # Build SELECT query based on available columns, filtered by profile
            has_profile = 'profile' in columns
            if 'summary' in columns:
                if has_profile:
                    cursor.execute('SELECT id, content, summary FROM memories WHERE profile = ?', (active_profile,))
                else:
                    cursor.execute('SELECT id, content, summary FROM memories')
                rows = cursor.fetchall()
                texts = [f"{row[1]} {row[2] or ''}" for row in rows]
            else:
                # Old database without summary column
                cursor.execute('SELECT id, content FROM memories')
                rows = cursor.fetchall()
                texts = [row[1] for row in rows]

        if not rows:
            self.vectorizer = None
            self.vectors = None
            self.memory_ids = []
            return

        self.memory_ids = [row[0] for row in rows]

        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.vectors = self.vectorizer.fit_transform(texts)
        self._last_vector_count = len(self.memory_ids)

        # Save memory IDs as JSON (safe serialization)
        self.vectors_path.mkdir(exist_ok=True)
        with open(self.vectors_path / "memory_ids.json", 'w') as f:
            json.dump(self.memory_ids, f)

    def get_recent(self, limit: int = 10, project_path: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get most recent memories (V1 compatible, profile-aware)."""
        active_profile = self._get_active_profile()

        with self._read_connection() as conn:
            cursor = conn.cursor()

            if project_path:
                cursor.execute('''
                    SELECT id, content, summary, project_path, project_name, tags,
                           category, parent_id, tree_path, depth, memory_type, importance,
                           created_at, cluster_id, last_accessed, access_count
                    FROM memories
                    WHERE project_path = ? AND profile = ?
                    ORDER BY created_at DESC
                    LIMIT ?
                ''', (project_path, active_profile, limit))
            else:
                cursor.execute('''
                    SELECT id, content, summary, project_path, project_name, tags,
                           category, parent_id, tree_path, depth, memory_type, importance,
                           created_at, cluster_id, last_accessed, access_count
                    FROM memories
                    WHERE profile = ?
                    ORDER BY created_at DESC
                    LIMIT ?
                ''', (active_profile, limit))

            results = []
            for row in cursor.fetchall():
                results.append(self._row_to_dict(row, 1.0, 'recent'))

        return results

    def get_by_id(self, memory_id: int) -> Optional[Dict[str, Any]]:
        """Get a specific memory by ID (V1 compatible, profile-aware)."""
        active_profile = self._get_active_profile()
        with self._read_connection() as conn:
            cursor = conn.cursor()

            cursor.execute('''
                SELECT id, content, summary, project_path, project_name, tags,
                       category, parent_id, tree_path, depth, memory_type, importance,
                       created_at, cluster_id, last_accessed, access_count
                FROM memories WHERE id = ? AND profile = ?
            ''', (memory_id, active_profile))

            row = cursor.fetchone()

        if not row:
            return None

        # Update access tracking
        self._update_access_tracking([memory_id])

        return self._row_to_dict(row, 1.0, 'direct')

    def delete_memory(self, memory_id: int) -> bool:
        """Delete a specific memory (V1 compatible, profile-aware)."""
        active_profile = self._get_active_profile()
        def _do_delete(conn):
            cursor = conn.cursor()
            cursor.execute('DELETE FROM memories WHERE id = ? AND profile = ?', (memory_id, active_profile))
            deleted = cursor.rowcount > 0
            conn.commit()
            return deleted

        deleted = self._execute_write(_do_delete)

        if deleted:
            self._rebuild_vectors()
            # Emit event (v2.5)
            self._emit_event("memory.deleted", memory_id=memory_id)
            # Trust signal (v2.5 — silent)
            if self._trust_scorer:
                try:
                    self._trust_scorer.on_memory_deleted("user", memory_id)
                except Exception:
                    pass

        return deleted

    def list_all(self, limit: int = 50) -> List[Dict[str, Any]]:
        """List all memories with short previews (V1 compatible, profile-aware)."""
        active_profile = self._get_active_profile()

        with self._read_connection() as conn:
            cursor = conn.cursor()

            cursor.execute('''
                SELECT id, content, summary, project_path, project_name, tags,
                       category, parent_id, tree_path, depth, memory_type, importance,
                       created_at, cluster_id, last_accessed, access_count
                FROM memories
                WHERE profile = ?
                ORDER BY created_at DESC
                LIMIT ?
            ''', (active_profile, limit))

            results = []
            for row in cursor.fetchall():
                mem_dict = self._row_to_dict(row, 1.0, 'list')

                # Add title field for V1 compatibility
                content = row[1]
                first_line = content.split('\n')[0][:60]
                mem_dict['title'] = first_line + ('...' if len(content) > 60 else '')

                results.append(mem_dict)

        return results

    def get_stats(self) -> Dict[str, Any]:
        """Get memory store statistics (V1 compatible with V2 extensions, profile-aware)."""
        active_profile = self._get_active_profile()

        with self._read_connection() as conn:
            cursor = conn.cursor()

            cursor.execute('SELECT COUNT(*) FROM memories WHERE profile = ?', (active_profile,))
            total_memories = cursor.fetchone()[0]

            cursor.execute('SELECT COUNT(DISTINCT project_path) FROM memories WHERE project_path IS NOT NULL AND profile = ?', (active_profile,))
            total_projects = cursor.fetchone()[0]

            cursor.execute('SELECT memory_type, COUNT(*) FROM memories WHERE profile = ? GROUP BY memory_type', (active_profile,))
            by_type = dict(cursor.fetchall())

            cursor.execute('SELECT category, COUNT(*) FROM memories WHERE category IS NOT NULL AND profile = ? GROUP BY category', (active_profile,))
            by_category = dict(cursor.fetchall())

            cursor.execute('SELECT MIN(created_at), MAX(created_at) FROM memories WHERE profile = ?', (active_profile,))
            date_range = cursor.fetchone()

            cursor.execute('SELECT COUNT(DISTINCT cluster_id) FROM memories WHERE cluster_id IS NOT NULL AND profile = ?', (active_profile,))
            total_clusters = cursor.fetchone()[0]

            cursor.execute('SELECT MAX(depth) FROM memories WHERE profile = ?', (active_profile,))
            max_depth = cursor.fetchone()[0] or 0

            # Total across all profiles
            cursor.execute('SELECT COUNT(*) FROM memories')
            total_all_profiles = cursor.fetchone()[0]

        return {
            'total_memories': total_memories,
            'total_all_profiles': total_all_profiles,
            'active_profile': active_profile,
            'total_projects': total_projects,
            'total_clusters': total_clusters,
            'max_tree_depth': max_depth,
            'by_type': by_type,
            'by_category': by_category,
            'date_range': {'earliest': date_range[0], 'latest': date_range[1]},
            'sklearn_available': SKLEARN_AVAILABLE
        }

    def get_attribution(self) -> Dict[str, str]:
        """
        Get creator attribution information embedded in the database.

        This information is REQUIRED by MIT License and must be preserved.
        Removing or obscuring this attribution violates the license terms.

        Returns:
            Dictionary with creator information and attribution requirements,
            including Qualixar platform provenance.
        """
        with self._read_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT key, value FROM creator_metadata')
            attribution = dict(cursor.fetchall())

        # Fallback if table doesn't exist yet (old databases)
        if not attribution:
            attribution = {
                'creator_name': 'Varun Pratap Bhardwaj',
                'creator_role': 'Solution Architect & Original Creator',
                'project_name': 'SuperLocalMemory V2',
                'license': 'MIT',
                'attribution_required': 'yes'
            }

        # Qualixar platform provenance (non-breaking additions)
        attribution['platform'] = 'Qualixar'
        attribution['verify_url'] = 'https://qualixar.com'

        return attribution

    def export_for_context(self, query: str, max_tokens: int = 4000) -> str:
        """Export relevant memories formatted for Claude context injection (V1 compatible)."""
        memories = self.search(query, limit=10)

        if not memories:
            return "No relevant memories found."

        output = ["## Relevant Memory Context\n"]
        char_count = 0
        max_chars = max_tokens * 4  # Rough token to char conversion

        for mem in memories:
            entry = f"\n### Memory (Score: {mem['score']:.2f})\n"
            if mem.get('project_name'):
                entry += f"**Project:** {mem['project_name']}\n"
            if mem.get('category'):
                entry += f"**Category:** {mem['category']}\n"
            if mem.get('summary'):
                entry += f"**Summary:** {mem['summary']}\n"
            entry += f"**Content:**\n{mem['content'][:1000]}...\n" if len(mem['content']) > 1000 else f"**Content:**\n{mem['content']}\n"

            if char_count + len(entry) > max_chars:
                break

            output.append(entry)
            char_count += len(entry)

        return ''.join(output)


# CLI interface (V1 compatible + V2 extensions)
if __name__ == "__main__":
    from memory.cli import run_cli
    run_cli()
