#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 SuperLocalMemory (superlocalmemory.com)
"""GraphEngine - Main orchestrator for the knowledge graph.

Coordinates entity extraction, edge building, community detection,
and graph traversal operations. All processing is local.
"""
import sqlite3
import json
import time
from pathlib import Path
from typing import List, Dict

import numpy as np

from graph.constants import (
    logger, MEMORY_DIR, DB_PATH, IGRAPH_AVAILABLE, cosine_similarity
)
from graph.entity_extractor import EntityExtractor
from graph.edge_builder import EdgeBuilder
from graph.cluster_builder import ClusterBuilder
from graph.schema import ensure_graph_tables
from graph.build_helpers import apply_sampling, clear_profile_graph_data
from graph.graph_search import (
    get_related as _get_related,
    get_cluster_members as _get_cluster_members,
    get_stats as _get_stats,
)


class GraphEngine:
    """Main graph engine coordinating all graph operations."""

    def __init__(self, db_path: Path = DB_PATH):
        """Initialize graph engine."""
        self.db_path = db_path
        self.entity_extractor = EntityExtractor(max_features=20)
        self.edge_builder = EdgeBuilder(db_path)
        self.cluster_builder = ClusterBuilder(db_path)
        self._ensure_graph_tables()

    def _get_active_profile(self) -> str:
        """Get the currently active profile name from config."""
        config_file = MEMORY_DIR / "profiles.json"
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    config = json.load(f)
                return config.get('active_profile', 'default')
            except (json.JSONDecodeError, IOError):
                pass
        return 'default'

    def _ensure_graph_tables(self):
        """Create graph tables if they don't exist, or recreate if schema is incomplete."""
        ensure_graph_tables(self.db_path)

    def build_graph(self, min_similarity: float = 0.3) -> Dict[str, any]:
        """
        Build complete knowledge graph from all memories.

        Args:
            min_similarity: Minimum cosine similarity for edge creation

        Returns:
            Dictionary with build statistics
        """
        start_time = time.time()
        logger.info("Starting full graph build...")
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            # Check required tables
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            existing_tables = {row[0] for row in cursor.fetchall()}
            missing = {'memories', 'graph_edges', 'graph_nodes', 'graph_clusters'} - existing_tables
            if missing:
                logger.error(f"Missing required tables: {missing}")
                return {'success': False, 'error': 'database_not_initialized',
                        'message': f"Database not initialized. Missing tables: {', '.join(missing)}",
                        'fix': "Run 'superlocalmemoryv2-status' first to initialize the database, or add some memories."}

            active_profile = self._get_active_profile()
            logger.info(f"Building graph for profile: {active_profile}")
            memories = cursor.execute(
                'SELECT id, content, summary FROM memories WHERE profile = ? ORDER BY id',
                (active_profile,)).fetchall()

            if len(memories) == 0:
                return {'success': False, 'error': 'no_memories',
                        'message': 'No memories found in database.',
                        'fix': "Add some memories first: superlocalmemoryv2-remember 'Your content here'"}
            if len(memories) < 2:
                return {'success': False, 'error': 'insufficient_memories',
                        'message': 'Need at least 2 memories to build knowledge graph.',
                        'memories': len(memories),
                        'fix': "Add more memories: superlocalmemoryv2-remember 'Your content here'"}

            memories = apply_sampling(cursor, memories, active_profile)
            clear_profile_graph_data(cursor, conn, memories, active_profile)

            logger.info(f"Processing {len(memories)} memories")
            memory_ids = [m[0] for m in memories]
            contents = [f"{m[1]} {m[2] or ''}" for m in memories]
            entities_list, vectors = self.entity_extractor.extract_entities(contents)

            for memory_id, entities, vector in zip(memory_ids, entities_list, vectors):
                cursor.execute('''
                    INSERT INTO graph_nodes (memory_id, entities, embedding_vector)
                    VALUES (?, ?, ?)
                ''', (memory_id, json.dumps(entities), json.dumps(vector.tolist())))
            conn.commit()
            logger.info(f"Stored {len(memory_ids)} graph nodes")

            edges_count = self.edge_builder.build_edges(memory_ids, vectors, entities_list)
            clusters_count = self.cluster_builder.detect_communities()
            hierarchical_stats = self.cluster_builder.hierarchical_cluster()
            subclusters = hierarchical_stats.get('subclusters_created', 0)
            summaries = self.cluster_builder.generate_cluster_summaries()
            elapsed = time.time() - start_time

            stats = {
                'success': True, 'memories': len(memories), 'nodes': len(memory_ids),
                'edges': edges_count, 'clusters': clusters_count, 'subclusters': subclusters,
                'max_depth': hierarchical_stats.get('depth_reached', 0),
                'summaries_generated': summaries, 'time_seconds': round(elapsed, 2)
            }
            if not IGRAPH_AVAILABLE:
                stats['warning'] = 'igraph/leidenalg not installed — graph built without clustering. Install with: pip3 install python-igraph leidenalg'
            logger.info(f"Graph build complete: {stats}")
            return stats

        except Exception as e:
            logger.error(f"Graph build failed: {e}")
            conn.rollback()
            return {'success': False, 'error': str(e)}
        finally:
            conn.close()

    def extract_entities(self, memory_id: int) -> List[str]:
        """Extract entities for a single memory."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        try:
            memory = cursor.execute(
                'SELECT content, summary FROM memories WHERE id = ?', (memory_id,)
            ).fetchone()
            if not memory:
                return []
            content = f"{memory[0]} {memory[1] or ''}"
            entities_list, _ = self.entity_extractor.extract_entities([content])
            return entities_list[0] if entities_list else []
        finally:
            conn.close()

    def get_related(self, memory_id: int, max_hops: int = 2) -> List[Dict]:
        """Get memories connected to this memory via graph edges (active profile only)."""
        return _get_related(self.db_path, memory_id, max_hops)

    def get_cluster_members(self, cluster_id: int) -> List[Dict]:
        """Get all memories in a cluster (filtered by active profile)."""
        return _get_cluster_members(self.db_path, cluster_id)

    def add_memory_incremental(self, memory_id: int) -> bool:
        """Add single memory to existing graph (incremental update)."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        try:
            memory = cursor.execute(
                'SELECT content, summary FROM memories WHERE id = ?', (memory_id,)
            ).fetchone()
            if not memory:
                return False

            content = f"{memory[0]} {memory[1] or ''}"
            entities_list, vector = self.entity_extractor.extract_entities([content])
            if not entities_list:
                return False

            new_entities, new_vector = entities_list[0], vector[0]
            cursor.execute('''
                INSERT OR REPLACE INTO graph_nodes (memory_id, entities, embedding_vector)
                VALUES (?, ?, ?)
            ''', (memory_id, json.dumps(new_entities), json.dumps(new_vector.tolist())))

            active_profile = self._get_active_profile()
            existing = cursor.execute('''
                SELECT gn.memory_id, gn.embedding_vector, gn.entities
                FROM graph_nodes gn JOIN memories m ON gn.memory_id = m.id
                WHERE gn.memory_id != ? AND m.profile = ?
            ''', (memory_id, active_profile)).fetchall()

            edges_added = 0
            for existing_id, ev_json, ee_json in existing:
                ev = np.array(json.loads(ev_json))
                sim = cosine_similarity([new_vector], [ev])[0][0]
                if sim >= self.edge_builder.min_similarity:
                    ee = json.loads(ee_json)
                    shared = list(set(new_entities) & set(ee))
                    rel_type = self.edge_builder._classify_relationship(sim, shared)
                    cursor.execute('''
                        INSERT OR REPLACE INTO graph_edges
                        (source_memory_id, target_memory_id, relationship_type,
                         weight, shared_entities, similarity_score)
                        VALUES (?, ?, ?, ?, ?, ?)
                    ''', (memory_id, existing_id, rel_type,
                          float(sim), json.dumps(shared), float(sim)))
                    edges_added += 1

            conn.commit()
            logger.info(f"Added memory {memory_id} to graph with {edges_added} edges")
            if edges_added > 5:
                logger.info("Significant graph change - consider re-clustering")
            return True
        except Exception as e:
            logger.error(f"Incremental add failed: {e}")
            conn.rollback()
            return False
        finally:
            conn.close()

    def get_stats(self) -> Dict[str, any]:
        """Get graph statistics for the active profile."""
        return _get_stats(self.db_path)
