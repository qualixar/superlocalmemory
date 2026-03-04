#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 SuperLocalMemory (superlocalmemory.com)
"""
CrossProjectAggregator — Layer 1: Transferable Tech Preferences.

Aggregates technology preferences across ALL user profiles by wrapping
the existing FrequencyAnalyzer from pattern_learner.py. This module
does NOT replace pattern_learner.py — it sits on top, reading its
per-profile results and merging them into cross-project patterns stored
in learning.db's `transferable_patterns` table.

Key behaviors:
    - Reads memories from memory.db across all profiles (READ-ONLY)
    - Wraps FrequencyAnalyzer.analyze_preferences() for per-profile analysis
    - Merges profile results with exponential temporal decay (1-year half-life)
    - Detects contradictions when preferences change across profiles or time
    - Stores merged patterns in learning.db via LearningDB.upsert_transferable_pattern()

Temporal Decay:
    weight = exp(-age_days / 365)
    This gives a 1-year half-life: memories from 365 days ago contribute ~37%
    of their original weight. Recent profiles dominate, but old preferences
    are not forgotten unless contradicted.

Contradiction Detection:
    If the preferred value for a category changed within the last 90 days
    (comparing the current top choice against previous stored value),
    a contradiction is logged. This signals preference evolution — not an
    error. The adaptive ranker can use contradictions to weight recent
    preferences higher.

Research Backing:
    - MACLA (arXiv:2512.18950): Bayesian confidence with temporal priors
    - MemoryBank (AAAI 2024): Cross-session preference persistence
    - Pattern originally from pattern_learner.py Layer 4

Thread Safety:
    Write operations to learning.db are protected by LearningDB's internal
    write lock. Read operations to memory.db use per-call connections (SQLite
    WAL mode supports concurrent reads).
"""

import json
import logging
import math
import sqlite3
import sys
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

logger = logging.getLogger("superlocalmemory.learning.aggregator")

# ---------------------------------------------------------------------------
# Import FrequencyAnalyzer from pattern_learner.py (lives in ~/.claude-memory/)
# ---------------------------------------------------------------------------
MEMORY_DIR = Path.home() / ".claude-memory"
DEFAULT_MEMORY_DB = MEMORY_DIR / "memory.db"

if str(MEMORY_DIR) not in sys.path:
    sys.path.insert(0, str(MEMORY_DIR))

try:
    from pattern_learner import FrequencyAnalyzer
    HAS_FREQ_ANALYZER = True
except ImportError:
    HAS_FREQ_ANALYZER = False
    logger.warning(
        "FrequencyAnalyzer not available. "
        "Ensure pattern_learner.py is in %s",
        MEMORY_DIR,
    )

# ---------------------------------------------------------------------------
# Import LearningDB (sibling module in src/learning/)
# ---------------------------------------------------------------------------
try:
    from .learning_db import LearningDB
except ImportError:
    try:
        from learning_db import LearningDB
    except ImportError:
        LearningDB = None
        logger.warning("LearningDB not available — aggregator results will not persist.")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Temporal decay half-life: 365 days (1 year)
DECAY_HALF_LIFE_DAYS = 365.0

# Contradiction detection window: 90 days
CONTRADICTION_WINDOW_DAYS = 90

# Minimum evidence to consider a pattern valid for merging
MIN_EVIDENCE_FOR_MERGE = 2

# Minimum confidence for a merged pattern to be stored
MIN_MERGE_CONFIDENCE = 0.3


class CrossProjectAggregator:
    """
    Aggregates tech preferences across all user profiles.

    Wraps FrequencyAnalyzer to analyze per-profile memories, then merges
    results with temporal decay into transferable patterns stored in
    learning.db.

    Usage:
        aggregator = CrossProjectAggregator()
        results = aggregator.aggregate_all_profiles()
        prefs = aggregator.get_tech_preferences(min_confidence=0.6)
    """

    def __init__(
        self,
        memory_db_path: Optional[Path] = None,
        learning_db: Optional[Any] = None,
    ):
        """
        Initialize the cross-project aggregator.

        Args:
            memory_db_path: Path to memory.db. Defaults to ~/.claude-memory/memory.db.
                            This database is READ-ONLY from this module's perspective.
            learning_db: A LearningDB instance for storing results. If None, one is
                         created using the default path.
        """
        self.memory_db_path = Path(memory_db_path) if memory_db_path else DEFAULT_MEMORY_DB
        self._lock = threading.Lock()

        # Initialize LearningDB for storing aggregated patterns
        if learning_db is not None:
            self._learning_db = learning_db
        elif LearningDB is not None:
            try:
                self._learning_db = LearningDB.get_instance()
            except Exception as e:
                logger.error("Failed to initialize LearningDB: %s", e)
                self._learning_db = None
        else:
            self._learning_db = None

        # Initialize FrequencyAnalyzer if available
        if HAS_FREQ_ANALYZER:
            self._analyzer = FrequencyAnalyzer(self.memory_db_path)
        else:
            self._analyzer = None

        logger.info(
            "CrossProjectAggregator initialized: memory_db=%s, "
            "freq_analyzer=%s, learning_db=%s",
            self.memory_db_path,
            "available" if self._analyzer else "unavailable",
            "available" if self._learning_db else "unavailable",
        )

    # ======================================================================
    # Core Aggregation
    # ======================================================================

    def aggregate_all_profiles(self) -> Dict[str, dict]:
        """
        Aggregate tech preferences across ALL profiles in memory.db.

        Workflow:
            1. List all distinct profiles from memory.db
            2. For each profile, collect memory IDs and timestamps
            3. Run FrequencyAnalyzer.analyze_preferences() per profile
            4. Merge results with exponential temporal decay
            5. Detect contradictions against previously stored patterns
            6. Store merged patterns in learning.db

        Returns:
            Dict mapping pattern_key -> {value, confidence, evidence_count,
            profiles_seen, contradictions, decay_factor}
        """
        if not self._analyzer:
            logger.warning("FrequencyAnalyzer unavailable — cannot aggregate.")
            return {}

        # Step 1: List all profiles and their memory data
        profile_data = self._get_all_profile_data()
        if not profile_data:
            logger.info("No profiles found in memory.db — nothing to aggregate.")
            return {}

        logger.info(
            "Aggregating preferences across %d profile(s): %s",
            len(profile_data),
            ", ".join(p["profile"] for p in profile_data),
        )

        # Step 2-3: Analyze each profile
        profile_patterns = []
        for pdata in profile_data:
            profile_name = pdata["profile"]
            memory_ids = pdata["memory_ids"]

            if not memory_ids:
                logger.debug("Profile '%s' has no memories — skipping.", profile_name)
                continue

            try:
                patterns = self._analyzer.analyze_preferences(memory_ids)
                if patterns:
                    profile_patterns.append({
                        "profile": profile_name,
                        "patterns": patterns,
                        "latest_timestamp": pdata["latest_timestamp"],
                        "memory_count": len(memory_ids),
                    })
                    logger.debug(
                        "Profile '%s': %d patterns from %d memories",
                        profile_name, len(patterns), len(memory_ids),
                    )
            except Exception as e:
                logger.error(
                    "Failed to analyze profile '%s': %s",
                    profile_name, e,
                )
                continue

        if not profile_patterns:
            logger.info("No patterns found across any profile.")
            return {}

        # Step 4: Merge with temporal decay
        merged = self._merge_with_decay(profile_patterns)

        # Step 5: Detect contradictions
        for key, pattern_data in merged.items():
            contradictions = self._detect_contradictions(key, pattern_data)
            pattern_data["contradictions"] = contradictions

        # Step 6: Store in learning.db
        self._store_merged_patterns(merged)

        logger.info(
            "Aggregation complete: %d transferable patterns stored.",
            len(merged),
        )
        return merged

    # ======================================================================
    # Profile Data Extraction (READ-ONLY on memory.db)
    # ======================================================================

    def _get_all_profile_data(self) -> List[dict]:
        """
        Get all profiles and their memory IDs from memory.db.

        Returns list of {profile, memory_ids, latest_timestamp, memory_count}.
        """
        results = []

        try:
            conn = sqlite3.connect(str(self.memory_db_path), timeout=10)
            try:
                conn.execute("PRAGMA busy_timeout=5000")
                cursor = conn.cursor()

                # Get distinct profiles
                cursor.execute(
                    "SELECT DISTINCT profile FROM memories "
                    "WHERE profile IS NOT NULL ORDER BY profile"
                )
                profiles = [row[0] for row in cursor.fetchall()]

                if not profiles:
                    # Fallback: if no profile column or all NULL, treat as 'default'
                    cursor.execute("SELECT id FROM memories ORDER BY created_at")
                    all_ids = [row[0] for row in cursor.fetchall()]
                    if all_ids:
                        # Get the latest timestamp
                        cursor.execute(
                            "SELECT MAX(created_at) FROM memories"
                        )
                        latest = cursor.fetchone()[0] or datetime.now().isoformat()
                        results.append({
                            "profile": "default",
                            "memory_ids": all_ids,
                            "latest_timestamp": latest,
                        })

                # For each profile, get memory IDs and latest timestamp
                for profile in profiles:
                    cursor.execute(
                        "SELECT id FROM memories WHERE profile = ? ORDER BY created_at",
                        (profile,),
                    )
                    memory_ids = [row[0] for row in cursor.fetchall()]

                    cursor.execute(
                        "SELECT MAX(created_at) FROM memories WHERE profile = ?",
                        (profile,),
                    )
                    latest = cursor.fetchone()[0] or datetime.now().isoformat()

                    if memory_ids:
                        results.append({
                            "profile": profile,
                            "memory_ids": memory_ids,
                            "latest_timestamp": latest,
                        })
            finally:
                conn.close()

        except sqlite3.OperationalError as e:
            # Handle case where 'profile' column doesn't exist
            logger.warning(
                "Could not query profiles from memory.db: %s. "
                "Falling back to all memories as 'default' profile.",
                e,
            )
            try:
                conn = sqlite3.connect(str(self.memory_db_path), timeout=10)
                try:
                    cursor = conn.cursor()
                    cursor.execute("SELECT id FROM memories ORDER BY created_at")
                    all_ids = [row[0] for row in cursor.fetchall()]
                    if all_ids:
                        cursor.execute("SELECT MAX(created_at) FROM memories")
                        latest = cursor.fetchone()[0] or datetime.now().isoformat()
                        results.append({
                            "profile": "default",
                            "memory_ids": all_ids,
                            "latest_timestamp": latest,
                        })
                finally:
                    conn.close()
            except Exception as inner_e:
                logger.error("Failed to read memory.db: %s", inner_e)

        except Exception as e:
            logger.error("Unexpected error reading profiles: %s", e)

        return results

    # ======================================================================
    # Temporal Decay Merging
    # ======================================================================

    def _merge_with_decay(
        self,
        profile_patterns: List[dict],
    ) -> Dict[str, dict]:
        """
        Merge per-profile patterns with exponential temporal decay.

        Each profile's contribution is weighted by:
            weight = exp(-age_days / DECAY_HALF_LIFE_DAYS)

        where age_days is the number of days since the profile's most
        recent memory was created. This ensures recent profiles dominate
        while old preferences decay gracefully.

        Args:
            profile_patterns: List of {profile, patterns, latest_timestamp, memory_count}

        Returns:
            Dict[category_key, {value, confidence, evidence_count, profiles_seen,
            decay_factor, profile_history}]
        """
        now = datetime.now()

        # Collect all contributions per category key
        # key -> list of {value, confidence, evidence_count, weight, profile}
        contributions: Dict[str, List[dict]] = {}

        for pdata in profile_patterns:
            # Calculate temporal weight for this profile
            age_days = self._days_since(pdata["latest_timestamp"], now)
            weight = math.exp(-age_days / DECAY_HALF_LIFE_DAYS)

            for category_key, pattern in pdata["patterns"].items():
                if category_key not in contributions:
                    contributions[category_key] = []

                contributions[category_key].append({
                    "value": pattern.get("value", ""),
                    "confidence": pattern.get("confidence", 0.0),
                    "evidence_count": pattern.get("evidence_count", 0),
                    "weight": weight,
                    "profile": pdata["profile"],
                    "latest_timestamp": pdata["latest_timestamp"],
                })

        # Merge contributions per category
        merged = {}
        for category_key, contribs in contributions.items():
            merged_pattern = self._merge_category_contributions(
                category_key, contribs
            )
            if merged_pattern is not None:
                merged[category_key] = merged_pattern

        return merged

    def _merge_category_contributions(
        self,
        category_key: str,
        contributions: List[dict],
    ) -> Optional[dict]:
        """
        Merge contributions for a single category across profiles.

        Strategy:
            1. Group contributions by value (the preferred tech)
            2. For each value, sum weighted evidence
            3. The value with highest weighted evidence wins
            4. Confidence = weighted_evidence / total_weighted_evidence
        """
        if not contributions:
            return None

        # Group by value
        value_scores: Dict[str, float] = {}
        value_evidence: Dict[str, int] = {}
        value_profiles: Dict[str, set] = {}
        value_weights: Dict[str, float] = {}

        total_weighted_evidence = 0.0

        for contrib in contributions:
            value = contrib["value"]
            weighted_ev = contrib["evidence_count"] * contrib["weight"]

            if value not in value_scores:
                value_scores[value] = 0.0
                value_evidence[value] = 0
                value_profiles[value] = set()
                value_weights[value] = 0.0

            value_scores[value] += weighted_ev
            value_evidence[value] += contrib["evidence_count"]
            value_profiles[value].add(contrib["profile"])
            value_weights[value] = max(value_weights[value], contrib["weight"])
            total_weighted_evidence += weighted_ev

        if total_weighted_evidence == 0:
            return None

        # Find the winning value
        winning_value = max(value_scores, key=value_scores.get)
        winning_score = value_scores[winning_value]

        # Calculate merged confidence
        confidence = winning_score / total_weighted_evidence if total_weighted_evidence > 0 else 0.0

        total_evidence = sum(value_evidence.values())
        winning_evidence = value_evidence[winning_value]

        if winning_evidence < MIN_EVIDENCE_FOR_MERGE:
            return None

        if confidence < MIN_MERGE_CONFIDENCE:
            return None

        # Average decay factor across contributing profiles for the winner
        winning_decay = value_weights[winning_value]

        # Build profile history for contradiction detection
        profile_history = []
        for contrib in contributions:
            profile_history.append({
                "profile": contrib["profile"],
                "value": contrib["value"],
                "confidence": round(contrib["confidence"], 3),
                "weight": round(contrib["weight"], 3),
                "timestamp": contrib["latest_timestamp"],
            })

        return {
            "value": winning_value,
            "confidence": round(min(0.95, confidence), 3),
            "evidence_count": winning_evidence,
            "profiles_seen": len(value_profiles[winning_value]),
            "total_profiles": len(set(c["profile"] for c in contributions)),
            "decay_factor": round(winning_decay, 4),
            "profile_history": profile_history,
            "contradictions": [],  # Filled in by _detect_contradictions
        }

    # ======================================================================
    # Contradiction Detection
    # ======================================================================

    def _detect_contradictions(
        self,
        pattern_key: str,
        pattern_data: dict,
    ) -> List[str]:
        """
        Detect if the preferred value changed recently.

        A contradiction is logged when:
            1. The current winning value differs from the previously stored value
            2. The change happened within the last CONTRADICTION_WINDOW_DAYS
            3. Multiple profiles disagree on the preferred value

        Contradictions are informational — they signal preference evolution,
        not errors. The adaptive ranker uses them to weight recent preferences.

        Args:
            pattern_key: Category key (e.g., 'frontend_framework')
            pattern_data: Merged pattern data with profile_history

        Returns:
            List of contradiction description strings.
        """
        contradictions = []
        current_value = pattern_data["value"]

        # Check 1: Cross-profile disagreement
        profile_history = pattern_data.get("profile_history", [])
        distinct_values = set(h["value"] for h in profile_history)

        if len(distinct_values) > 1:
            other_values = distinct_values - {current_value}
            for other_val in other_values:
                disagreeing_profiles = [
                    h["profile"] for h in profile_history
                    if h["value"] == other_val
                ]
                contradictions.append(
                    "Profile(s) %s prefer '%s' instead of '%s'" % (
                        ", ".join(disagreeing_profiles),
                        other_val,
                        current_value,
                    )
                )

        # Check 2: Change from previously stored value (in learning.db)
        if self._learning_db is not None:
            try:
                stored = self._learning_db.get_transferable_patterns(
                    min_confidence=0.0,
                    pattern_type="preference",
                )
                for row in stored:
                    if row.get("key") == pattern_key:
                        old_value = row.get("value", "")
                        old_updated = row.get("updated_at") or row.get("last_seen")
                        if old_value and old_value != current_value:
                            # Check if the old pattern was updated recently
                            if old_updated and self._is_within_window(
                                old_updated, CONTRADICTION_WINDOW_DAYS
                            ):
                                contradictions.append(
                                    "Preference changed from '%s' to '%s' "
                                    "within last %d days" % (
                                        old_value,
                                        current_value,
                                        CONTRADICTION_WINDOW_DAYS,
                                    )
                                )
                        break
            except Exception as e:
                logger.debug(
                    "Could not check stored patterns for contradictions: %s", e
                )

        if contradictions:
            logger.info(
                "Contradictions for '%s': %s",
                pattern_key,
                "; ".join(contradictions),
            )

        return contradictions

    # ======================================================================
    # Storage (learning.db)
    # ======================================================================

    def _store_merged_patterns(self, merged: Dict[str, dict]):
        """
        Store merged patterns in learning.db's transferable_patterns table.

        Uses LearningDB.upsert_transferable_pattern() which handles
        INSERT ON CONFLICT UPDATE internally with its own write lock.
        """
        if self._learning_db is None:
            logger.warning(
                "LearningDB unavailable — %d patterns computed but not stored.",
                len(merged),
            )
            return

        stored_count = 0
        for key, data in merged.items():
            try:
                self._learning_db.upsert_transferable_pattern(
                    pattern_type="preference",
                    key=key,
                    value=data["value"],
                    confidence=data["confidence"],
                    evidence_count=data["evidence_count"],
                    profiles_seen=data.get("profiles_seen", 1),
                    decay_factor=data.get("decay_factor", 1.0),
                    contradictions=data.get("contradictions"),
                )
                stored_count += 1
            except Exception as e:
                logger.error(
                    "Failed to store pattern '%s': %s", key, e
                )

        logger.info(
            "Stored %d/%d merged patterns in learning.db.",
            stored_count, len(merged),
        )

    # ======================================================================
    # Query Interface
    # ======================================================================

    def get_tech_preferences(
        self,
        min_confidence: float = 0.6,
    ) -> Dict[str, dict]:
        """
        Retrieve aggregated tech preferences from learning.db.

        This reads from the `transferable_patterns` table — the stored
        results of a previous aggregate_all_profiles() call.

        Args:
            min_confidence: Minimum confidence threshold (0.0 to 1.0).
                            Default 0.6 matches FrequencyAnalyzer's threshold.

        Returns:
            Dict mapping category_key -> {value, confidence, evidence_count,
            profiles_seen, decay_factor, contradictions}
        """
        if self._learning_db is None:
            logger.warning("LearningDB unavailable — cannot read preferences.")
            return {}

        try:
            rows = self._learning_db.get_transferable_patterns(
                min_confidence=min_confidence,
                pattern_type="preference",
            )

            preferences = {}
            for row in rows:
                key = row.get("key", "")
                if not key:
                    continue

                # Parse contradictions from JSON
                contradictions = []
                raw_contradictions = row.get("contradictions", "[]")
                if isinstance(raw_contradictions, str):
                    try:
                        contradictions = json.loads(raw_contradictions)
                    except (json.JSONDecodeError, TypeError):
                        contradictions = []
                elif isinstance(raw_contradictions, list):
                    contradictions = raw_contradictions

                preferences[key] = {
                    "value": row.get("value", ""),
                    "confidence": row.get("confidence", 0.0),
                    "evidence_count": row.get("evidence_count", 0),
                    "profiles_seen": row.get("profiles_seen", 1),
                    "decay_factor": row.get("decay_factor", 1.0),
                    "contradictions": contradictions,
                    "first_seen": row.get("first_seen"),
                    "last_seen": row.get("last_seen"),
                }

            return preferences

        except Exception as e:
            logger.error("Failed to read tech preferences: %s", e)
            return {}

    def get_preference_context(self, min_confidence: float = 0.6) -> str:
        """
        Format transferable preferences for injection into AI context.

        Returns a human-readable markdown string suitable for CLAUDE.md
        or system prompt injection.

        Args:
            min_confidence: Minimum confidence threshold.

        Returns:
            Formatted markdown string.
        """
        prefs = self.get_tech_preferences(min_confidence)

        if not prefs:
            return (
                "## Cross-Project Tech Preferences\n\n"
                "No transferable preferences learned yet. "
                "Use more profiles and add memories to build your tech profile."
            )

        lines = ["## Cross-Project Tech Preferences\n"]

        for key, data in sorted(prefs.items(), key=lambda x: -x[1]["confidence"]):
            display_key = key.replace("_", " ").title()
            conf_pct = data["confidence"] * 100
            evidence = data["evidence_count"]
            profiles = data["profiles_seen"]
            line = (
                "- **%s:** %s (%.0f%% confidence, %d evidence, %d profile%s)"
                % (
                    display_key,
                    data["value"],
                    conf_pct,
                    evidence,
                    profiles,
                    "s" if profiles != 1 else "",
                )
            )

            # Flag contradictions
            if data.get("contradictions"):
                line += " [EVOLVING]"

            lines.append(line)

        return "\n".join(lines)

    # ======================================================================
    # Utility Methods
    # ======================================================================

    @staticmethod
    def _days_since(timestamp_str: str, now: Optional[datetime] = None) -> float:
        """
        Calculate days between a timestamp string and now.

        Handles multiple timestamp formats from SQLite (ISO 8601, space-separated).
        Returns 0.0 on parse failure (treat as recent).
        """
        if now is None:
            now = datetime.now()

        if not timestamp_str:
            return 0.0

        try:
            ts = datetime.fromisoformat(timestamp_str.replace(" ", "T"))
            delta = now - ts
            return max(0.0, delta.total_seconds() / 86400.0)
        except (ValueError, AttributeError, TypeError):
            pass

        # Fallback: try common formats
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%Y-%m-%dT%H:%M:%S.%f"):
            try:
                ts = datetime.strptime(str(timestamp_str), fmt)
                delta = now - ts
                return max(0.0, delta.total_seconds() / 86400.0)
            except (ValueError, TypeError):
                continue

        logger.debug("Could not parse timestamp: %s", timestamp_str)
        return 0.0

    @staticmethod
    def _is_within_window(timestamp_str: str, window_days: int) -> bool:
        """Check if a timestamp is within the given window (in days)."""
        if not timestamp_str:
            return False
        try:
            ts = datetime.fromisoformat(
                str(timestamp_str).replace(" ", "T")
            )
            return (datetime.now() - ts).days <= window_days
        except (ValueError, AttributeError, TypeError):
            return False


# ===========================================================================
# CLI Interface
# ===========================================================================

if __name__ == "__main__":
    import sys as _sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    aggregator = CrossProjectAggregator()

    if len(_sys.argv) < 2:
        print("CrossProjectAggregator — Layer 1: Transferable Tech Preferences")
        print()
        print("Usage:")
        print("  python cross_project_aggregator.py aggregate       # Run full aggregation")
        print("  python cross_project_aggregator.py preferences     # Show stored preferences")
        print("  python cross_project_aggregator.py context [min]   # Get context for AI injection")
        _sys.exit(0)

    command = _sys.argv[1]

    if command == "aggregate":
        results = aggregator.aggregate_all_profiles()
        if results:
            print("\nAggregated %d transferable patterns:" % len(results))
            for key, data in sorted(results.items()):
                print(
                    "  %-25s %-30s  conf=%.2f  evidence=%d  profiles=%d%s"
                    % (
                        key,
                        data["value"],
                        data["confidence"],
                        data["evidence_count"],
                        data.get("profiles_seen", 1),
                        "  [CONTRADICTIONS]" if data.get("contradictions") else "",
                    )
                )
        else:
            print("No patterns found. Add memories across profiles first.")

    elif command == "preferences":
        min_conf = float(_sys.argv[2]) if len(_sys.argv) > 2 else 0.6
        prefs = aggregator.get_tech_preferences(min_confidence=min_conf)
        if prefs:
            print("\nTransferable Tech Preferences (min confidence: %.0f%%):" % (min_conf * 100))
            for key, data in sorted(prefs.items(), key=lambda x: -x[1]["confidence"]):
                print(
                    "  %-25s %-30s  conf=%.2f  evidence=%d  profiles=%d"
                    % (
                        key,
                        data["value"],
                        data["confidence"],
                        data["evidence_count"],
                        data.get("profiles_seen", 1),
                    )
                )
                if data.get("contradictions"):
                    for c in data["contradictions"]:
                        print("    ^-- %s" % c)
        else:
            print("No preferences stored. Run 'aggregate' first.")

    elif command == "context":
        min_conf = float(_sys.argv[2]) if len(_sys.argv) > 2 else 0.6
        print(aggregator.get_preference_context(min_conf))

    else:
        print("Unknown command: %s" % command)
        _sys.exit(1)
