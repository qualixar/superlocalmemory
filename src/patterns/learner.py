#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 SuperLocalMemory (superlocalmemory.com)
"""
Pattern Learner - Main orchestrator and CLI.

Coordinates frequency analysis, context analysis, terminology learning,
confidence scoring, and pattern storage into a unified learning pipeline.
"""

import sqlite3
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
from collections import Counter

from .analyzers import FrequencyAnalyzer, ContextAnalyzer
from .terminology import TerminologyLearner
from .scoring import ConfidenceScorer
from .store import PatternStore

logger = logging.getLogger(__name__)

# Local NLP tools (no external APIs)
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

MEMORY_DIR = Path.home() / ".claude-memory"
DB_PATH = MEMORY_DIR / "memory.db"


class PatternLearner:
    """Main pattern learning orchestrator."""

    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self.frequency_analyzer = FrequencyAnalyzer(db_path)
        self.context_analyzer = ContextAnalyzer(db_path)
        self.terminology_learner = TerminologyLearner(db_path)
        self.confidence_scorer = ConfidenceScorer(db_path)
        self.pattern_store = PatternStore(db_path)

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

    def weekly_pattern_update(self) -> Dict[str, int]:
        """Full pattern analysis of all memories for active profile. Run this weekly."""
        active_profile = self._get_active_profile()
        print(f"Starting weekly pattern update for profile: {active_profile}...")

        # Get memory IDs for active profile only
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute('SELECT id FROM memories WHERE profile = ? ORDER BY created_at',
                           (active_profile,))
            all_memory_ids = [row[0] for row in cursor.fetchall()]
            total_memories = len(all_memory_ids)
        finally:
            conn.close()

        if total_memories == 0:
            print(f"No memories found for profile '{active_profile}'. Add memories first.")
            return {'preferences': 0, 'styles': 0, 'terminology': 0}

        print(f"Analyzing {total_memories} memories for profile '{active_profile}'...")

        # Run all analyzers
        preferences = self.frequency_analyzer.analyze_preferences(all_memory_ids)
        print(f"  Found {len(preferences)} preference patterns")

        styles = self.context_analyzer.analyze_style(all_memory_ids)
        print(f"  Found {len(styles)} style patterns")

        terms = self.terminology_learner.learn_terminology(all_memory_ids)
        print(f"  Found {len(terms)} terminology patterns")

        # Recalculate confidence scores and save all patterns (tagged with profile)
        counts = {'preferences': 0, 'styles': 0, 'terminology': 0}

        for pattern in preferences.values():
            confidence = self.confidence_scorer.calculate_confidence(
                pattern['pattern_type'],
                pattern['key'],
                pattern['value'],
                pattern['memory_ids'],
                total_memories
            )
            pattern['confidence'] = round(confidence, 2)
            pattern['profile'] = active_profile
            self.pattern_store.save_pattern(pattern)
            counts['preferences'] += 1

        for pattern in styles.values():
            confidence = self.confidence_scorer.calculate_confidence(
                pattern['pattern_type'],
                pattern['key'],
                pattern['value'],
                pattern['memory_ids'],
                total_memories
            )
            pattern['confidence'] = round(confidence, 2)
            pattern['profile'] = active_profile
            self.pattern_store.save_pattern(pattern)
            counts['styles'] += 1

        for pattern in terms.values():
            confidence = self.confidence_scorer.calculate_confidence(
                pattern['pattern_type'],
                pattern['key'],
                pattern['value'],
                pattern['memory_ids'],
                total_memories
            )
            pattern['confidence'] = round(confidence, 2)
            pattern['profile'] = active_profile
            self.pattern_store.save_pattern(pattern)
            counts['terminology'] += 1

        print(f"\nPattern update complete:")
        print(f"  {counts['preferences']} preferences")
        print(f"  {counts['styles']} styles")
        print(f"  {counts['terminology']} terminology")

        return counts

    def on_new_memory(self, memory_id: int):
        """Incremental update when new memory is added."""
        active_profile = self._get_active_profile()
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute('SELECT COUNT(*) FROM memories WHERE profile = ?',
                           (active_profile,))
            total = cursor.fetchone()[0]
        finally:
            conn.close()

        # Only do incremental updates if we have many memories (>50)
        if total > 50:
            # Deferred to batch update for efficiency (see weekly_pattern_update)
            pass
        else:
            # For small memory counts, just do full update
            self.weekly_pattern_update()

    def get_patterns(self, min_confidence: float = 0.7) -> List[Dict[str, Any]]:
        """Query patterns above confidence threshold for active profile."""
        active_profile = self._get_active_profile()
        return self.pattern_store.get_patterns(min_confidence, profile=active_profile)

    def get_identity_context(self, min_confidence: float = 0.7) -> str:
        """Format patterns for Claude context injection."""
        patterns = self.get_patterns(min_confidence)

        if not patterns:
            return "## Working with User - Learned Patterns\n\nNo patterns learned yet. Add more memories to build your profile."

        # Group by pattern type
        sections = {
            'preference': [],
            'style': [],
            'terminology': []
        }

        for p in patterns:
            sections[p['pattern_type']].append(
                f"- **{p['key'].replace('_', ' ').title()}:** {p['value']} "
                f"(confidence: {p['confidence']:.0%}, {p['evidence_count']} examples)"
            )

        output = "## Working with User - Learned Patterns\n\n"

        if sections['preference']:
            output += "**Technology Preferences:**\n" + '\n'.join(sections['preference']) + '\n\n'

        if sections['style']:
            output += "**Coding Style:**\n" + '\n'.join(sections['style']) + '\n\n'

        if sections['terminology']:
            output += "**Terminology:**\n" + '\n'.join(sections['terminology']) + '\n'

        return output


# CLI Interface
if __name__ == "__main__":
    import sys

    learner = PatternLearner()

    if len(sys.argv) < 2:
        print("Pattern Learner - Identity Profile Extraction")
        print("\nUsage:")
        print("  python pattern_learner.py update           # Full pattern update (weekly)")
        print("  python pattern_learner.py list [min_conf]  # List learned patterns (default: 0.7)")
        print("  python pattern_learner.py context [min]    # Get context for Claude")
        print("  python pattern_learner.py stats            # Pattern statistics")
        sys.exit(0)

    command = sys.argv[1]

    if command == "update":
        counts = learner.weekly_pattern_update()
        print(f"\nTotal patterns learned: {sum(counts.values())}")

    elif command == "list":
        min_conf = float(sys.argv[2]) if len(sys.argv) > 2 else 0.7
        patterns = learner.get_patterns(min_conf)

        if not patterns:
            print(f"No patterns found with confidence >= {min_conf:.0%}")
        else:
            print(f"\n{'Type':<15} {'Category':<12} {'Pattern':<30} {'Confidence':<12} {'Evidence':<10}")
            print("-" * 95)

            for p in patterns:
                pattern_display = f"{p['key'].replace('_', ' ').title()}: {p['value']}"
                if len(pattern_display) > 28:
                    pattern_display = pattern_display[:28] + "..."

                print(f"{p['pattern_type']:<15} {p['category']:<12} {pattern_display:<30} "
                      f"{p['confidence']:>6.0%}        {p['evidence_count']:<10}")

    elif command == "context":
        min_conf = float(sys.argv[2]) if len(sys.argv) > 2 else 0.7
        context = learner.get_identity_context(min_conf)
        print(context)

    elif command == "stats":
        patterns = learner.get_patterns(0.5)  # Include all patterns

        if not patterns:
            print("No patterns learned yet.")
        else:
            by_type = Counter([p['pattern_type'] for p in patterns])
            by_category = Counter([p['category'] for p in patterns])

            avg_confidence = sum(p['confidence'] for p in patterns) / len(patterns)
            high_conf = len([p for p in patterns if p['confidence'] >= 0.8])

            print(f"\nPattern Statistics:")
            print(f"  Total patterns: {len(patterns)}")
            print(f"  Average confidence: {avg_confidence:.0%}")
            print(f"  High confidence (>=80%): {high_conf}")
            print(f"\nBy Type:")
            for ptype, count in by_type.most_common():
                print(f"  {ptype}: {count}")
            print(f"\nBy Category:")
            for cat, count in by_category.most_common():
                print(f"  {cat}: {count}")

    else:
        print(f"Unknown command: {command}")
        sys.exit(1)
