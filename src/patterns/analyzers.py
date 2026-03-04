#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 SuperLocalMemory (superlocalmemory.com)
"""
Pattern Analyzers - Frequency and Context Analysis

Technology preference detection via frequency counting,
and coding style pattern detection from context.
"""

import sqlite3
import re
import logging
from typing import Dict, List, Any
from collections import Counter
from pathlib import Path

logger = logging.getLogger(__name__)


class FrequencyAnalyzer:
    """Analyzes technology and tool preferences via frequency counting."""

    def __init__(self, db_path: Path):
        self.db_path = db_path

        # Predefined technology categories
        self.tech_categories = {
            'frontend_framework': ['react', 'nextjs', 'next.js', 'vue', 'angular', 'svelte', 'solid'],
            'backend_framework': ['express', 'fastapi', 'django', 'flask', 'nestjs', 'spring', 'rails'],
            'database': ['postgres', 'postgresql', 'mysql', 'mongodb', 'redis', 'dynamodb', 'sqlite'],
            'state_management': ['redux', 'context', 'zustand', 'mobx', 'recoil', 'jotai'],
            'styling': ['tailwind', 'css modules', 'styled-components', 'emotion', 'sass', 'less'],
            'language': ['python', 'javascript', 'typescript', 'go', 'rust', 'java', 'c++'],
            'deployment': ['docker', 'kubernetes', 'vercel', 'netlify', 'aws', 'gcp', 'azure'],
            'testing': ['jest', 'pytest', 'vitest', 'mocha', 'cypress', 'playwright'],
        }

    def analyze_preferences(self, memory_ids: List[int]) -> Dict[str, Dict[str, Any]]:
        """Analyze technology preferences across memories."""
        patterns = {}

        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()

            for category, keywords in self.tech_categories.items():
                keyword_counts = Counter()
                evidence_memories = {}  # {keyword: [memory_ids]}

                for memory_id in memory_ids:
                    cursor.execute('SELECT content FROM memories WHERE id = ?', (memory_id,))
                    row = cursor.fetchone()

                    if not row:
                        continue

                    content = row[0].lower()

                    for keyword in keywords:
                        # Count occurrences with word boundaries
                        pattern = r'\b' + re.escape(keyword.replace('.', r'\.')) + r'\b'
                        matches = re.findall(pattern, content, re.IGNORECASE)
                        count = len(matches)

                        if count > 0:
                            keyword_counts[keyword] += count

                            if keyword not in evidence_memories:
                                evidence_memories[keyword] = []
                            evidence_memories[keyword].append(memory_id)

                # Determine preference (most mentioned)
                if keyword_counts:
                    top_keyword = keyword_counts.most_common(1)[0][0]
                    total_mentions = sum(keyword_counts.values())
                    top_count = keyword_counts[top_keyword]

                    # Calculate confidence (% of mentions)
                    confidence = top_count / total_mentions if total_mentions > 0 else 0

                    # Only create pattern if confidence > 0.6 and at least 3 mentions
                    if confidence > 0.6 and top_count >= 3:
                        value = self._format_preference(top_keyword, keyword_counts)
                        evidence_list = list(set(evidence_memories.get(top_keyword, [])))

                        patterns[category] = {
                            'pattern_type': 'preference',
                            'key': category,
                            'value': value,
                            'confidence': round(confidence, 2),
                            'evidence_count': len(evidence_list),
                            'memory_ids': evidence_list,
                            'category': self._categorize_pattern(category)
                        }

        finally:
            conn.close()
        return patterns

    def _format_preference(self, top_keyword: str, all_counts: Counter) -> str:
        """Format preference value (e.g., 'Next.js over React')."""
        # Normalize keyword for display
        display_map = {
            'nextjs': 'Next.js',
            'next.js': 'Next.js',
            'postgres': 'PostgreSQL',
            'postgresql': 'PostgreSQL',
            'fastapi': 'FastAPI',
            'nestjs': 'NestJS',
            'mongodb': 'MongoDB',
            'redis': 'Redis',
            'dynamodb': 'DynamoDB',
            'tailwind': 'Tailwind CSS',
        }

        top_display = display_map.get(top_keyword.lower(), top_keyword.title())

        if len(all_counts) > 1:
            second = all_counts.most_common(2)[1]
            second_keyword = second[0]
            second_display = display_map.get(second_keyword.lower(), second_keyword.title())

            # Only show comparison if second choice has significant mentions
            if second[1] / all_counts[top_keyword] > 0.3:
                return f"{top_display} over {second_display}"

        return top_display

    def _categorize_pattern(self, tech_category: str) -> str:
        """Map tech category to high-level category."""
        category_map = {
            'frontend_framework': 'frontend',
            'state_management': 'frontend',
            'styling': 'frontend',
            'backend_framework': 'backend',
            'database': 'backend',
            'language': 'general',
            'deployment': 'devops',
            'testing': 'general',
        }
        return category_map.get(tech_category, 'general')


class ContextAnalyzer:
    """Analyzes coding style patterns from context."""

    def __init__(self, db_path: Path):
        self.db_path = db_path

        # Style pattern detection rules
        self.style_indicators = {
            'optimization_priority': {
                'performance': ['optimize', 'faster', 'performance', 'speed', 'latency', 'efficient', 'cache'],
                'readability': ['readable', 'clean', 'maintainable', 'clear', 'simple', 'understandable']
            },
            'error_handling': {
                'explicit': ['error boundary', 'explicit', 'throw', 'handle error', 'try catch', 'error handling'],
                'permissive': ['ignore', 'suppress', 'skip error', 'optional']
            },
            'testing_approach': {
                'comprehensive': ['test coverage', 'unit test', 'integration test', 'e2e test', 'test suite'],
                'minimal': ['manual test', 'skip test', 'no tests']
            },
            'code_organization': {
                'modular': ['separate', 'module', 'component', 'split', 'refactor', 'extract'],
                'monolithic': ['single file', 'one place', 'combined']
            }
        }

    def analyze_style(self, memory_ids: List[int]) -> Dict[str, Dict[str, Any]]:
        """Detect stylistic patterns from context."""
        patterns = {}

        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()

            for pattern_key, indicators in self.style_indicators.items():
                indicator_counts = Counter()
                evidence_memories = {}  # {style_type: [memory_ids]}

                for memory_id in memory_ids:
                    cursor.execute('SELECT content FROM memories WHERE id = ?', (memory_id,))
                    row = cursor.fetchone()

                    if not row:
                        continue

                    content = row[0].lower()

                    for style_type, keywords in indicators.items():
                        for keyword in keywords:
                            if keyword in content:
                                indicator_counts[style_type] += 1

                                if style_type not in evidence_memories:
                                    evidence_memories[style_type] = []
                                evidence_memories[style_type].append(memory_id)

                # Determine dominant style
                if indicator_counts:
                    top_style = indicator_counts.most_common(1)[0][0]
                    total = sum(indicator_counts.values())
                    top_count = indicator_counts[top_style]
                    confidence = top_count / total if total > 0 else 0

                    # Only create pattern if confidence > 0.65 and at least 3 mentions
                    if confidence > 0.65 and top_count >= 3:
                        value = self._format_style_value(pattern_key, top_style, indicator_counts)
                        evidence_list = list(set(evidence_memories.get(top_style, [])))

                        patterns[pattern_key] = {
                            'pattern_type': 'style',
                            'key': pattern_key,
                            'value': value,
                            'confidence': round(confidence, 2),
                            'evidence_count': len(evidence_list),
                            'memory_ids': evidence_list,
                            'category': 'general'
                        }

        finally:
            conn.close()
        return patterns

    def _format_style_value(self, pattern_key: str, top_style: str, all_counts: Counter) -> str:
        """Format style value as comparison or preference."""
        style_formats = {
            'optimization_priority': {
                'performance': 'Performance over readability',
                'readability': 'Readability over performance'
            },
            'error_handling': {
                'explicit': 'Explicit error boundaries',
                'permissive': 'Permissive error handling'
            },
            'testing_approach': {
                'comprehensive': 'Comprehensive testing',
                'minimal': 'Minimal testing'
            },
            'code_organization': {
                'modular': 'Modular organization',
                'monolithic': 'Monolithic organization'
            }
        }

        if pattern_key in style_formats and top_style in style_formats[pattern_key]:
            return style_formats[pattern_key][top_style]

        return top_style.replace('_', ' ').title()
