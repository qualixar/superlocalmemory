#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 SuperLocalMemory (superlocalmemory.com)
"""HNSWIndex - Fast Approximate Nearest Neighbor Search for SuperLocalMemory V2

Implements HNSW (Hierarchical Navigable Small World) algorithm for:
- Sub-10ms vector similarity search for 10K+ memories
- Incremental updates without full rebuild
- Disk persistence for instant startup
- Graceful fallback to linear search if hnswlib unavailable

All processing is local - no external APIs.

LIMITS:
- MAX_MEMORIES_FOR_HNSW: 100,000 (prevents memory exhaustion)
- MAX_DIMENSION: 5000 (typical: 384 for sentence embeddings)
- Performance target: <10ms for 10K memories, <50ms for 100K memories
"""
# SECURITY: HNSW index limits to prevent resource exhaustion
MAX_MEMORIES_FOR_HNSW = 100_000
MAX_DIMENSION = 5000
DEFAULT_M = 16              # HNSW parameter: number of connections per layer
DEFAULT_EF_CONSTRUCTION = 200  # HNSW parameter: size of dynamic candidate list
DEFAULT_EF_SEARCH = 50      # HNSW parameter: search-time candidate list size

import sqlite3
import json
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
import numpy as np

# Core dependencies for fallback
try:
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Optional HNSW dependency
HNSW_AVAILABLE = False
try:
    import hnswlib
    HNSW_AVAILABLE = True
except ImportError:
    HNSW_AVAILABLE = False
    # Graceful degradation - will use linear search fallback

MEMORY_DIR = Path.home() / ".claude-memory"
HNSW_INDEX_PATH = MEMORY_DIR / "hnsw_index.bin"
HNSW_METADATA_PATH = MEMORY_DIR / "hnsw_metadata.json"

logger = logging.getLogger(__name__)


class HNSWIndex:
    """
    Fast approximate nearest neighbor search using HNSW algorithm.

    Features:
    - Sub-10ms search for 10K memories
    - Incremental updates (no full rebuild needed)
    - Disk persistence with automatic loading
    - Graceful fallback to linear search if hnswlib unavailable

    Performance:
    - 10K memories: <10ms search time
    - 100K memories: <50ms search time
    - Memory overhead: ~200 bytes per vector (configurable)
    """

    def __init__(
        self,
        dimension: int = 384,
        max_elements: int = MAX_MEMORIES_FOR_HNSW,
        m: int = DEFAULT_M,
        ef_construction: int = DEFAULT_EF_CONSTRUCTION,
        ef_search: int = DEFAULT_EF_SEARCH,
        index_path: Optional[Path] = None,
        metadata_path: Optional[Path] = None
    ):
        """
        Initialize HNSW index.

        Args:
            dimension: Vector dimension (e.g., 384 for all-MiniLM-L6-v2)
            max_elements: Maximum number of vectors to index
            m: HNSW M parameter (connections per layer, typical: 16)
            ef_construction: HNSW ef_construction (candidate list size, typical: 200)
            ef_search: HNSW ef_search (search candidate list size, typical: 50)
            index_path: Custom path for index file
            metadata_path: Custom path for metadata file

        Raises:
            ValueError: If parameters exceed security limits
        """
        # SECURITY: Input validation
        if dimension > MAX_DIMENSION:
            raise ValueError(f"Dimension {dimension} exceeds maximum {MAX_DIMENSION}")

        if max_elements > MAX_MEMORIES_FOR_HNSW:
            raise ValueError(f"Max elements {max_elements} exceeds limit {MAX_MEMORIES_FOR_HNSW}")

        self.dimension = dimension
        self.max_elements = max_elements
        self.m = m
        self.ef_construction = ef_construction
        self.ef_search = ef_search

        self.index_path = index_path or HNSW_INDEX_PATH
        self.metadata_path = metadata_path or HNSW_METADATA_PATH

        # Initialize index and metadata
        self.index = None
        self.memory_ids = []  # Maps index position to memory ID
        self.id_to_idx = {}   # Maps memory ID to index position
        self.use_hnsw = HNSW_AVAILABLE

        # Fallback: store vectors for linear search
        self.vectors = None

        # Load existing index if available
        self._load()

    def _load(self):
        """Load existing index and metadata from disk."""
        if not self.use_hnsw:
            logger.info("HNSW unavailable - will use linear search fallback")
            return

        if not self.index_path.exists() or not self.metadata_path.exists():
            logger.info("No existing HNSW index found - will create new index")
            return

        try:
            # Load metadata
            with open(self.metadata_path, 'r') as f:
                metadata = json.load(f)

            # Validate metadata — auto-rebuild on dimension mismatch
            if metadata.get('dimension') != self.dimension:
                logger.info(
                    "Index dimension changed: %s -> %s. "
                    "Deleting old index files and rebuilding.",
                    metadata.get('dimension'), self.dimension,
                )
                try:
                    self.index_path.unlink(missing_ok=True)
                    self.metadata_path.unlink(missing_ok=True)
                except OSError as del_err:
                    logger.warning("Could not delete old index files: %s", del_err)
                return

            # Load HNSW index
            self.index = hnswlib.Index(space='cosine', dim=self.dimension)
            self.index.load_index(str(self.index_path))
            self.index.set_ef(self.ef_search)

            # Load memory ID mapping
            self.memory_ids = metadata.get('memory_ids', [])
            self.id_to_idx = {mem_id: idx for idx, mem_id in enumerate(self.memory_ids)}

            logger.info(f"Loaded HNSW index with {len(self.memory_ids)} vectors")

        except Exception as e:
            logger.error(f"Failed to load HNSW index: {e}. Will rebuild.")
            self.index = None
            self.memory_ids = []
            self.id_to_idx = {}

    def _save(self):
        """Save index and metadata to disk."""
        if not self.use_hnsw or self.index is None:
            return

        try:
            # Create directory if needed
            self.index_path.parent.mkdir(parents=True, exist_ok=True)

            # Save HNSW index
            self.index.save_index(str(self.index_path))

            # Save metadata
            metadata = {
                'dimension': self.dimension,
                'max_elements': self.max_elements,
                'm': self.m,
                'ef_construction': self.ef_construction,
                'ef_search': self.ef_search,
                'memory_ids': self.memory_ids,
                'created_at': datetime.now().isoformat(),
                'version': '2.2.0'
            }

            with open(self.metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)

            logger.info(f"Saved HNSW index with {len(self.memory_ids)} vectors")

        except Exception as e:
            logger.error(f"Failed to save HNSW index: {e}")

    def build(self, vectors: np.ndarray, memory_ids: List[int]) -> None:
        """
        Build HNSW index from vectors.

        Args:
            vectors: Array of shape (n_memories, dimension)
            memory_ids: List of memory IDs corresponding to vectors

        Raises:
            ValueError: If input validation fails
        """
        # SECURITY: Input validation
        if len(vectors) > self.max_elements:
            raise ValueError(
                f"Cannot index {len(vectors)} vectors (max: {self.max_elements}). "
                "Use incremental updates or increase max_elements."
            )

        if vectors.shape[1] != self.dimension:
            raise ValueError(
                f"Vector dimension {vectors.shape[1]} does not match index dimension {self.dimension}"
            )

        if len(vectors) != len(memory_ids):
            raise ValueError("Number of vectors must match number of memory IDs")

        # Convert to float32 for efficiency
        vectors = vectors.astype('float32')

        if self.use_hnsw:
            # Build HNSW index
            try:
                start_time = time.time()

                self.index = hnswlib.Index(space='cosine', dim=self.dimension)
                self.index.init_index(
                    max_elements=self.max_elements,
                    M=self.m,
                    ef_construction=self.ef_construction,
                    random_seed=42
                )
                self.index.set_ef(self.ef_search)

                # Add vectors in batch
                self.index.add_items(vectors, list(range(len(vectors))))

                self.memory_ids = list(memory_ids)
                self.id_to_idx = {mem_id: idx for idx, mem_id in enumerate(memory_ids)}

                # Save to disk
                self._save()

                elapsed = time.time() - start_time
                logger.info(f"Built HNSW index with {len(vectors)} vectors in {elapsed:.2f}s")

            except Exception as e:
                logger.error(f"HNSW build failed: {e}. Falling back to linear search.")
                self.use_hnsw = False
                self._build_fallback(vectors, memory_ids)
        else:
            # Fallback: store vectors for linear search
            self._build_fallback(vectors, memory_ids)

    def _build_fallback(self, vectors: np.ndarray, memory_ids: List[int]):
        """Build fallback index using linear search."""
        if not SKLEARN_AVAILABLE:
            logger.warning("sklearn unavailable - search functionality disabled")
            return

        self.vectors = vectors.astype('float32')
        self.memory_ids = list(memory_ids)
        self.id_to_idx = {mem_id: idx for idx, mem_id in enumerate(memory_ids)}
        logger.info(f"Built fallback index with {len(vectors)} vectors (linear search)")

    def add(self, vector: np.ndarray, memory_id: int) -> None:
        """
        Add single vector to index (incremental update).

        Args:
            vector: Vector of shape (dimension,)
            memory_id: Memory ID for this vector

        Raises:
            ValueError: If index is full or vector invalid
        """
        # SECURITY: Input validation
        if len(self.memory_ids) >= self.max_elements:
            raise ValueError(f"Index is full (max: {self.max_elements})")

        if len(vector) != self.dimension:
            raise ValueError(f"Vector dimension {len(vector)} does not match {self.dimension}")

        vector = vector.astype('float32').reshape(1, -1)

        if self.use_hnsw and self.index is not None:
            try:
                # Add to HNSW index
                idx = len(self.memory_ids)
                self.index.add_items(vector, [idx])
                self.memory_ids.append(memory_id)
                self.id_to_idx[memory_id] = idx

                # Save updated index
                self._save()

            except Exception as e:
                logger.error(f"Failed to add vector to HNSW: {e}")
                # Continue with best effort - don't crash
        else:
            # Fallback: append to vectors array
            if self.vectors is None:
                self.vectors = vector
            else:
                self.vectors = np.vstack([self.vectors, vector])

            idx = len(self.memory_ids)
            self.memory_ids.append(memory_id)
            self.id_to_idx[memory_id] = idx

    def search(
        self,
        query_vector: np.ndarray,
        k: int = 5,
        filter_ids: Optional[List[int]] = None
    ) -> List[Tuple[int, float]]:
        """
        Search for k nearest neighbors.

        Args:
            query_vector: Query vector of shape (dimension,)
            k: Number of results to return
            filter_ids: Optional list of memory IDs to restrict search

        Returns:
            List of (memory_id, distance) tuples, sorted by distance (lower = more similar)

        Performance:
        - HNSW: <10ms for 10K vectors, <50ms for 100K vectors
        - Fallback: O(n) linear search, ~100ms for 10K vectors
        """
        if len(self.memory_ids) == 0:
            return []

        # SECURITY: Input validation
        if len(query_vector) != self.dimension:
            raise ValueError(f"Query dimension {len(query_vector)} does not match {self.dimension}")

        query_vector = query_vector.astype('float32').reshape(1, -1)
        k = min(k, len(self.memory_ids))  # Don't request more than available

        if self.use_hnsw and self.index is not None:
            # HNSW search
            try:
                start_time = time.time()

                # Get more candidates if filtering is needed
                search_k = k * 3 if filter_ids else k
                search_k = min(search_k, len(self.memory_ids))

                labels, distances = self.index.knn_query(query_vector, k=search_k)

                # Convert to results
                results = []
                for idx, dist in zip(labels[0], distances[0]):
                    mem_id = self.memory_ids[idx]

                    # Apply filter if provided
                    if filter_ids is None or mem_id in filter_ids:
                        # Convert cosine distance to similarity score (1 - distance)
                        similarity = 1.0 - dist
                        results.append((mem_id, float(similarity)))

                    if len(results) >= k:
                        break

                elapsed = time.time() - start_time
                logger.debug(f"HNSW search took {elapsed*1000:.2f}ms for {len(self.memory_ids)} vectors")

                return results

            except Exception as e:
                logger.error(f"HNSW search failed: {e}. Falling back to linear search.")
                # Fall through to fallback

        # Fallback: linear search with sklearn
        if SKLEARN_AVAILABLE and self.vectors is not None:
            start_time = time.time()

            # Compute similarities
            similarities = cosine_similarity(query_vector, self.vectors)[0]

            # Get top k indices
            if filter_ids:
                # Filter first, then sort
                filtered_indices = [idx for idx, mem_id in enumerate(self.memory_ids) if mem_id in filter_ids]
                if not filtered_indices:
                    return []
                filtered_similarities = similarities[filtered_indices]
                top_indices = np.argsort(filtered_similarities)[::-1][:k]
                results = [(self.memory_ids[filtered_indices[idx]], float(filtered_similarities[idx]))
                          for idx in top_indices]
            else:
                # Direct sorting
                top_indices = np.argsort(similarities)[::-1][:k]
                results = [(self.memory_ids[idx], float(similarities[idx])) for idx in top_indices]

            elapsed = time.time() - start_time
            logger.debug(f"Linear search took {elapsed*1000:.2f}ms for {len(self.memory_ids)} vectors")

            return results

        logger.warning("No search method available (HNSW and sklearn both unavailable)")
        return []

    def update(self, memory_id: int, vector: np.ndarray) -> None:
        """
        Update vector for existing memory.

        Note: HNSW doesn't support in-place updates efficiently.
        This marks the item for rebuild or uses a workaround.

        Args:
            memory_id: Memory ID to update
            vector: New vector of shape (dimension,)
        """
        if memory_id not in self.id_to_idx:
            logger.warning(f"Memory ID {memory_id} not in index - adding as new")
            self.add(vector, memory_id)
            return

        # For HNSW, mark as dirty and suggest rebuild
        # HNSW doesn't support efficient updates - best practice is periodic rebuild
        logger.warning(
            f"Updated memory {memory_id} - HNSW index is now stale. "
            "Consider calling rebuild() periodically for optimal performance."
        )

        # Update fallback index if available
        if self.vectors is not None:
            idx = self.id_to_idx[memory_id]
            self.vectors[idx] = vector.astype('float32')

    def delete(self, memory_id: int):
        """
        Delete memory from index.

        Note: HNSW doesn't support efficient deletion.
        This marks the item for rebuild.

        Args:
            memory_id: Memory ID to delete
        """
        if memory_id not in self.id_to_idx:
            logger.warning(f"Memory ID {memory_id} not in index")
            return

        # For now, just remove from mapping (soft delete)
        # Physical removal requires full rebuild
        idx = self.id_to_idx[memory_id]
        del self.id_to_idx[memory_id]

        logger.info(
            f"Soft-deleted memory {memory_id} from index. "
            "Call rebuild() to physically remove."
        )

    def rebuild_from_db(self, db_path: Path, embedding_column: str = 'embedding'):
        """
        Rebuild index from database.

        Args:
            db_path: Path to SQLite database
            embedding_column: Name of column containing embeddings (JSON array)
        """
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        try:
            # Check if embedding column exists
            cursor.execute("PRAGMA table_info(memories)")
            columns = {row[1] for row in cursor.fetchall()}

            if embedding_column not in columns:
                logger.warning(f"Column '{embedding_column}' not found in database")
                conn.close()
                return

            # Load embeddings
            cursor.execute(f'SELECT id, {embedding_column} FROM memories WHERE {embedding_column} IS NOT NULL')
            rows = cursor.fetchall()

            if not rows:
                logger.info("No embeddings found in database")
                conn.close()
                return

            # Parse embeddings
            memory_ids = []
            vectors = []

            for mem_id, embedding_json in rows:
                try:
                    embedding = json.loads(embedding_json)
                    memory_ids.append(mem_id)
                    vectors.append(embedding)
                except json.JSONDecodeError:
                    logger.warning(f"Invalid embedding JSON for memory {mem_id}")

            if not vectors:
                logger.info("No valid embeddings to index")
                conn.close()
                return

            vectors = np.array(vectors, dtype='float32')

            # Build index
            self.build(vectors, memory_ids)
            logger.info(f"Rebuilt HNSW index from database with {len(memory_ids)} vectors")

        finally:
            conn.close()

    def get_stats(self) -> Dict[str, Any]:
        """
        Get index statistics.

        Returns:
            Dictionary with index stats
        """
        return {
            'hnsw_available': HNSW_AVAILABLE,
            'use_hnsw': self.use_hnsw,
            'sklearn_available': SKLEARN_AVAILABLE,
            'dimension': self.dimension,
            'max_elements': self.max_elements,
            'indexed_count': len(self.memory_ids),
            'capacity_used_pct': (len(self.memory_ids) / self.max_elements * 100) if self.max_elements > 0 else 0,
            'm': self.m,
            'ef_construction': self.ef_construction,
            'ef_search': self.ef_search,
            'index_exists': self.index is not None,
            'fallback_active': self.vectors is not None
        }


# CLI interface for testing
if __name__ == "__main__":
    import sys

    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    if len(sys.argv) < 2:
        print("HNSWIndex CLI - Fast Approximate Nearest Neighbor Search")
        print("\nCommands:")
        print("  python hnsw_index.py stats                    # Show index statistics")
        print("  python hnsw_index.py rebuild                  # Rebuild from database")
        print("  python hnsw_index.py test                     # Run performance test")
        sys.exit(0)

    command = sys.argv[1]

    if command == "stats":
        index = HNSWIndex()
        stats = index.get_stats()
        print(json.dumps(stats, indent=2))

    elif command == "rebuild":
        db_path = MEMORY_DIR / "memory.db"
        if not db_path.exists():
            print(f"Database not found at {db_path}")
            sys.exit(1)

        print("Rebuilding HNSW index from database...")
        index = HNSWIndex()
        index.rebuild_from_db(db_path)
        print("Rebuild complete!")
        print(json.dumps(index.get_stats(), indent=2))

    elif command == "test":
        print("Running HNSW performance test...")

        # Generate random test data
        n_vectors = 10000
        dimension = 384

        print(f"Generating {n_vectors} random {dimension}-dim vectors...")
        vectors = np.random.randn(n_vectors, dimension).astype('float32')
        memory_ids = list(range(n_vectors))

        # Build index
        print("Building index...")
        index = HNSWIndex(dimension=dimension)
        start = time.time()
        index.build(vectors, memory_ids)
        build_time = time.time() - start
        print(f"Build time: {build_time:.2f}s ({n_vectors/build_time:.0f} vectors/sec)")

        # Test search performance
        print("\nTesting search performance...")
        query = np.random.randn(dimension).astype('float32')

        # Warm-up
        for _ in range(10):
            index.search(query, k=5)

        # Benchmark
        n_queries = 100
        start = time.time()
        for _ in range(n_queries):
            results = index.search(query, k=5)
        search_time = (time.time() - start) / n_queries

        print(f"Average search time: {search_time*1000:.2f}ms")
        print(f"Queries per second: {1/search_time:.0f}")
        print(f"\nSample results: {results[:3]}")

        # Print stats
        print("\nIndex statistics:")
        print(json.dumps(index.get_stats(), indent=2))

    else:
        print(f"Unknown command: {command}")
        print("Run without arguments to see available commands.")
