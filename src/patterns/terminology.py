#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 SuperLocalMemory (superlocalmemory.com)
"""
Terminology Learner - User-specific term definition extraction.

Learns how the user defines ambiguous terms like 'optimize', 'refactor', etc.
by analyzing contextual co-occurrence patterns across memories.
"""

import sqlite3
import re
import logging
from typing import Dict, List, Optional, Any
from collections import Counter
from pathlib import Path

logger = logging.getLogger(__name__)


class TerminologyLearner:
    """Learns user-specific definitions of common terms."""

    def __init__(self, db_path: Path):
        self.db_path = db_path

        # Common ambiguous terms to learn
        self.ambiguous_terms = [
            'optimize', 'refactor', 'clean', 'simple',
            'mvp', 'prototype', 'scale', 'production-ready',
            'fix', 'improve', 'update', 'enhance'
        ]

    def learn_terminology(self, memory_ids: List[int]) -> Dict[str, Dict[str, Any]]:
        """Learn user-specific term definitions."""
        patterns = {}

        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()

            for term in self.ambiguous_terms:
                contexts = []

                # Find all contexts where term appears
                for memory_id in memory_ids:
                    cursor.execute('SELECT content FROM memories WHERE id = ?', (memory_id,))
                    row = cursor.fetchone()

                    if not row:
                        continue

                    content = row[0]

                    # Find term in content (case-insensitive)
                    pattern = r'\b' + re.escape(term) + r'\b'
                    for match in re.finditer(pattern, content, re.IGNORECASE):
                        term_idx = match.start()

                        # Extract 100-char window around term
                        start = max(0, term_idx - 100)
                        end = min(len(content), term_idx + len(term) + 100)
                        context_window = content[start:end]

                        contexts.append({
                            'memory_id': memory_id,
                            'context': context_window
                        })

                # Analyze contexts to extract meaning (need at least 3 examples)
                if len(contexts) >= 3:
                    definition = self._extract_definition(term, contexts)

                    if definition:
                        evidence_list = list(set([ctx['memory_id'] for ctx in contexts]))

                        # Confidence increases with more examples, capped at 0.95
                        confidence = min(0.95, 0.6 + (len(contexts) * 0.05))

                        patterns[term] = {
                            'pattern_type': 'terminology',
                            'key': term,
                            'value': definition,
                            'confidence': round(confidence, 2),
                            'evidence_count': len(evidence_list),
                            'memory_ids': evidence_list,
                            'category': 'general'
                        }

        finally:
            conn.close()
        return patterns

    def _extract_definition(self, term: str, contexts: List[Dict]) -> Optional[str]:
        """Extract definition from contexts using pattern matching."""
        # Collect words near the term across all contexts
        nearby_words = []

        for ctx in contexts:
            words = re.findall(r'\b\w+\b', ctx['context'].lower())
            nearby_words.extend(words)

        # Count word frequencies
        word_counts = Counter(nearby_words)

        # Remove the term itself and common stopwords
        stopwords = {'the', 'a', 'an', 'is', 'to', 'for', 'of', 'in', 'on', 'at',
                     'and', 'or', 'but', 'with', 'from', 'by', 'this', 'that'}
        word_counts = Counter({w: c for w, c in word_counts.items()
                              if w not in stopwords and w != term.lower()})

        # Get top co-occurring words
        top_words = [w for w, _ in word_counts.most_common(8)]

        # Apply heuristic rules based on term and context
        if term == 'optimize':
            if any(w in top_words for w in ['performance', 'speed', 'faster', 'latency']):
                return "Performance optimization (speed/latency)"
            elif any(w in top_words for w in ['code', 'clean', 'refactor']):
                return "Code quality optimization"

        elif term == 'refactor':
            if any(w in top_words for w in ['architecture', 'structure', 'design']):
                return "Architecture change, not just renaming"
            elif any(w in top_words for w in ['clean', 'organize', 'simplify']):
                return "Code organization improvement"

        elif term == 'mvp':
            if any(w in top_words for w in ['core', 'basic', 'essential', 'minimal']):
                return "Core features only, no polish"

        elif term == 'production-ready':
            if any(w in top_words for w in ['test', 'error', 'monitoring', 'deploy']):
                return "Fully tested and monitored for deployment"

        # Generic definition if specific pattern not matched
        if len(top_words) >= 3:
            return f"Commonly used with: {', '.join(top_words[:3])}"

        return None
