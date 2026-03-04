#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 SuperLocalMemory (superlocalmemory.com)
"""HybridSearchEngine - Main orchestrator for multi-method retrieval fusion.
"""
import time
import json
import sqlite3
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any

from search_engine_v2 import BM25SearchEngine
from query_optimizer import QueryOptimizer
from cache_manager import CacheManager

from search.index_loader import IndexLoaderMixin
from search.methods import SearchMethodsMixin
from search.fusion import FusionMixin


class HybridSearchEngine(IndexLoaderMixin, SearchMethodsMixin, FusionMixin):
    """
    Hybrid search combining BM25, graph traversal, and semantic search.

    Provides flexible retrieval strategies based on query type and
    available resources.
    """

    def __init__(
        self,
        db_path: Path,
        bm25_engine: Optional[BM25SearchEngine] = None,
        query_optimizer: Optional[QueryOptimizer] = None,
        cache_manager: Optional[CacheManager] = None,
        enable_cache: bool = True
    ):
        """
        Initialize hybrid search engine.

        Args:
            db_path: Path to memory database
            bm25_engine: Pre-configured BM25 engine (will create if None)
            query_optimizer: Query optimizer instance (will create if None)
            cache_manager: Cache manager instance (will create if None)
            enable_cache: Enable result caching
        """
        self.db_path = db_path

        # Initialize components
        self.bm25 = bm25_engine or BM25SearchEngine()
        self.optimizer = query_optimizer or QueryOptimizer()
        self.cache = cache_manager if enable_cache else None

        # Graph engine (lazy load to avoid circular dependencies)
        self._graph_engine = None

        # TF-IDF fallback (from memory_store_v2)
        self._tfidf_vectorizer = None
        self._tfidf_vectors = None
        self._memory_ids = []

        # Performance tracking
        self.last_search_time = 0.0
        self.last_fusion_time = 0.0

        # Load index
        self._load_index()

    def search(
        self,
        query: str,
        limit: int = 10,
        method: str = "hybrid",
        weights: Optional[Dict[str, float]] = None,
        use_cache: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Hybrid search with multiple retrieval methods.

        Args:
            query: Search query
            limit: Maximum results
            method: Fusion method ("hybrid", "weighted", "rrf", "bm25", "semantic", "graph")
            weights: Custom weights for weighted fusion (default: balanced)
            use_cache: Use cache for results

        Returns:
            List of memory dictionaries with scores and match details
        """
        start_time = time.time()

        # Check cache
        if use_cache and self.cache:
            cached = self.cache.get(query, limit=limit, method=method)
            if cached is not None:
                self.last_search_time = time.time() - start_time
                return cached

        # Default weights
        if weights is None:
            weights = {
                'bm25': 0.4,
                'semantic': 0.3,
                'graph': 0.3
            }

        # Single method search
        if method == "bm25":
            raw_results = self.search_bm25(query, limit)
        elif method == "semantic":
            raw_results = self.search_semantic(query, limit)
        elif method == "graph":
            raw_results = self.search_graph(query, limit)

        # Multi-method fusion
        else:
            fusion_start = time.time()

            # Get results from all methods
            results_dict = {}

            if weights.get('bm25', 0) > 0:
                results_dict['bm25'] = self.search_bm25(query, limit=limit*2)

            if weights.get('semantic', 0) > 0:
                results_dict['semantic'] = self.search_semantic(query, limit=limit*2)

            if weights.get('graph', 0) > 0:
                results_dict['graph'] = self.search_graph(query, limit=limit*2)

            # Fusion
            if method == "rrf":
                raw_results = self._reciprocal_rank_fusion(list(results_dict.values()))
            else:  # weighted or hybrid
                raw_results = self._weighted_fusion(results_dict, weights)

            self.last_fusion_time = time.time() - fusion_start

        # Limit results
        raw_results = raw_results[:limit]

        # Fetch full memory details
        results = self._fetch_memory_details(raw_results, query)

        # Cache results
        if use_cache and self.cache:
            self.cache.put(query, results, limit=limit, method=method)

        self.last_search_time = time.time() - start_time

        return results

    def _fetch_memory_details(
        self,
        raw_results: List[Tuple[int, float]],
        query: str
    ) -> List[Dict[str, Any]]:
        """
        Fetch full memory details for result IDs.

        Args:
            raw_results: List of (memory_id, score) tuples
            query: Original query (for context)

        Returns:
            List of memory dictionaries with full details
        """
        if not raw_results:
            return []

        memory_ids = [mem_id for mem_id, _ in raw_results]
        id_to_score = {mem_id: score for mem_id, score in raw_results}

        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()

            # Fetch memories
            placeholders = ','.join(['?'] * len(memory_ids))
            cursor.execute(f'''
                SELECT id, content, summary, project_path, project_name, tags,
                       category, parent_id, tree_path, depth, memory_type,
                       importance, created_at, cluster_id, last_accessed, access_count
                FROM memories
                WHERE id IN ({placeholders})
            ''', memory_ids)

            rows = cursor.fetchall()
        finally:
            conn.close()

        # Build result dictionaries
        results = []
        for row in rows:
            mem_id = row[0]
            results.append({
                'id': mem_id,
                'content': row[1],
                'summary': row[2],
                'project_path': row[3],
                'project_name': row[4],
                'tags': json.loads(row[5]) if row[5] else [],
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
                'score': id_to_score.get(mem_id, 0.0),
                'match_type': 'hybrid'
            })

        # Sort by score
        results.sort(key=lambda x: x['score'], reverse=True)

        return results

    def get_stats(self) -> Dict[str, Any]:
        """
        Get hybrid search statistics.

        Returns:
            Dictionary with performance stats
        """
        stats = {
            'bm25': self.bm25.get_stats(),
            'optimizer': self.optimizer.get_stats(),
            'last_search_time_ms': self.last_search_time * 1000,
            'last_fusion_time_ms': self.last_fusion_time * 1000,
            'tfidf_available': self._tfidf_vectorizer is not None,
            'graph_available': self._graph_engine is not None
        }

        if self.cache:
            stats['cache'] = self.cache.get_stats()

        return stats
