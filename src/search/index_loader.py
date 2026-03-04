#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 SuperLocalMemory (superlocalmemory.com)
"""Index loading and graph engine lazy-loading for hybrid search.
"""
import json
import sqlite3
from pathlib import Path
from typing import Optional

from search_engine_v2 import BM25SearchEngine
from query_optimizer import QueryOptimizer


class IndexLoaderMixin:
    """
    Mixin that provides index loading and graph engine lazy-loading.

    Expects the host class to have:
        - self.db_path: Path
        - self.bm25: BM25SearchEngine
        - self.optimizer: QueryOptimizer
        - self._graph_engine: Optional[GraphEngine]
        - self._tfidf_vectorizer
        - self._tfidf_vectors
        - self._memory_ids: list
    """

    def _load_index(self):
        """
        Load documents from database and build search indexes.
        """
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()

            # Fetch all memories
            cursor.execute('''
                SELECT id, content, summary, tags
                FROM memories
                ORDER BY id
            ''')

            rows = cursor.fetchall()
        finally:
            conn.close()

        if not rows:
            return

        # Build BM25 index
        doc_ids = [row[0] for row in rows]
        documents = []
        vocabulary = set()

        for row in rows:
            # Combine content + summary + tags for indexing
            text_parts = [row[1]]  # content

            if row[2]:  # summary
                text_parts.append(row[2])

            if row[3]:  # tags (JSON)
                try:
                    tags = json.loads(row[3])
                    text_parts.extend(tags)
                except Exception:
                    pass

            doc_text = ' '.join(text_parts)
            documents.append(doc_text)

            # Build vocabulary for spell correction
            tokens = self.bm25._tokenize(doc_text)
            vocabulary.update(tokens)

        # Index with BM25
        self.bm25.index_documents(documents, doc_ids)
        self._memory_ids = doc_ids

        # Initialize optimizer with vocabulary
        self.optimizer.vocabulary = vocabulary

        # Build co-occurrence for query expansion
        tokenized_docs = [self.bm25._tokenize(doc) for doc in documents]
        self.optimizer.build_cooccurrence_matrix(tokenized_docs)

        # Try to load TF-IDF (optional semantic search)
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
            import numpy as np

            self._tfidf_vectorizer = TfidfVectorizer(
                max_features=5000,
                stop_words='english',
                ngram_range=(1, 2)
            )
            self._tfidf_vectors = self._tfidf_vectorizer.fit_transform(documents)

        except ImportError:
            # sklearn not available - skip semantic search
            pass

    def _load_graph_engine(self):
        """Lazy load graph engine to avoid circular imports."""
        if self._graph_engine is None:
            try:
                from graph_engine import GraphEngine
                self._graph_engine = GraphEngine(self.db_path)
            except ImportError:
                # Graph engine not available
                pass
        return self._graph_engine
