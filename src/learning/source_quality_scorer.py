#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 SuperLocalMemory (superlocalmemory.com)
"""
SourceQualityScorer — Per-source quality learning.

Learns which memory sources (tools/agents) produce memories that users
actually find useful. If memories from 'mcp:claude-desktop' get positive
feedback (via memory_used) 3x more often than memories from 'cli:terminal',
then Claude Desktop memories receive a quality boost in the adaptive ranker.

Data Sources:
    - memory.db `created_by` column (set by ProvenanceTracker in v2.5)
        Values like: 'mcp:claude-desktop', 'mcp:cursor', 'cli:terminal',
        'rest:api', 'user', etc.
    - learning.db `ranking_feedback` table (positive signals from FeedbackCollector)
        Signal types: 'mcp_used', 'cli_useful', 'dashboard_click'

Scoring Algorithm (Beta-Binomial Smoothing):
    quality_score = (alpha + positive_signals) / (alpha + beta + total_memories)

    With alpha=1, beta=1 (Laplace smoothing / uniform prior):
        - Unknown source with 0 feedback: 1/(2+0) = 0.50 (neutral)
        - Source with 5 positives out of 10 total: 6/12 = 0.50 (average)
        - Source with 8 positives out of 10 total: 9/12 = 0.75 (good)
        - Source with 1 positive out of 10 total: 2/12 = 0.17 (poor)

    This naturally handles:
        - Cold start: new sources get 0.5 (neutral) until evidence accumulates
        - Low sample: smoothing prevents extreme scores from few observations
        - Convergence: scores stabilize as evidence grows

Storage:
    Results stored in learning.db `source_quality` table via LearningDB.
    The adaptive ranker reads source_quality at query time to boost/penalize
    memories based on their source.

Thread Safety:
    - All writes protected by LearningDB's internal write lock
    - Reads to memory.db use per-call connections (safe with WAL mode)
    - compute_source_scores() is idempotent — safe to call concurrently

Graceful Degradation:
    - If memory.db lacks `created_by` column: all memories grouped as 'unknown'
    - If learning.db unavailable: scores computed but not persisted
    - If ranking_feedback is empty: all sources get 0.5 (neutral)

Research Backing:
    - Beta-Binomial smoothing: Standard Bayesian approach (matches trust_scorer.py)
    - Source reliability learning: ADPMF (IPM 2024) privacy-preserving feedback
    - FCS LREC 2024: cold-start handling via smoothing priors
"""

import json
import logging
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

logger = logging.getLogger("superlocalmemory.learning.source_quality")

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
        logger.warning(
            "LearningDB not available — source quality scores will not persist."
        )

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MEMORY_DIR = Path.home() / ".claude-memory"
DEFAULT_MEMORY_DB = MEMORY_DIR / "memory.db"

# Beta-Binomial prior parameters (Laplace smoothing)
ALPHA = 1.0  # Prior successes
BETA = 1.0   # Prior failures

# Default score for unknown sources (= alpha / (alpha + beta))
DEFAULT_QUALITY_SCORE = ALPHA / (ALPHA + BETA)

# Minimum total memories from a source before we trust its score
# Below this, the score is blended toward the default
MIN_EVIDENCE_THRESHOLD = 5

# Positive feedback signal types from ranking_feedback table
POSITIVE_SIGNAL_TYPES = ("mcp_used", "cli_useful", "dashboard_click")


class SourceQualityScorer:
    """
    Learns which memory sources produce higher-quality memories.

    Computes a quality score per source using Beta-Binomial smoothing
    over positive feedback signals. Stores results in learning.db for
    use by the adaptive ranker.

    Usage:
        scorer = SourceQualityScorer()
        scores = scorer.compute_source_scores()
        # scores = {'mcp:claude-desktop': 0.72, 'cli:terminal': 0.45, ...}

        boost = scorer.get_source_boost(memory_dict)
        # boost = 0.72 (for a memory from claude-desktop)
    """

    def __init__(
        self,
        memory_db_path: Optional[Path] = None,
        learning_db: Optional[Any] = None,
    ):
        """
        Initialize the source quality scorer.

        Args:
            memory_db_path: Path to memory.db (READ-ONLY). Defaults to
                            ~/.claude-memory/memory.db.
            learning_db: A LearningDB instance for reading feedback and
                         storing scores. If None, one is created.
        """
        self.memory_db_path = Path(memory_db_path) if memory_db_path else DEFAULT_MEMORY_DB
        self._lock = threading.Lock()

        # In-memory cache of source scores (refreshed by compute_source_scores)
        self._cached_scores: Dict[str, float] = {}

        # Initialize LearningDB
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

        # Pre-load cached scores from learning.db if available
        self._load_cached_scores()

        logger.info(
            "SourceQualityScorer initialized: memory_db=%s, learning_db=%s, "
            "cached_sources=%d",
            self.memory_db_path,
            "available" if self._learning_db else "unavailable",
            len(self._cached_scores),
        )

    # ======================================================================
    # Core Scoring
    # ======================================================================

    def compute_source_scores(self) -> Dict[str, float]:
        """
        Compute quality scores for all memory sources.

        Workflow:
            1. Get total memories per source from memory.db (created_by column)
            2. Get positive feedback count per source by joining
               learning.db ranking_feedback with memory.db memories
            3. Compute Beta-Binomial smoothed score per source
            4. Store results in learning.db source_quality table
            5. Update in-memory cache

        Returns:
            Dict mapping source_id -> quality_score (0.0 to 1.0)
        """
        # Step 1: Count total memories per source from memory.db
        source_totals = self._get_memory_counts_by_source()

        if not source_totals:
            logger.info("No source data found in memory.db.")
            return {}

        # Step 2: Count positive signals per source
        source_positives = self._get_positive_signal_counts(
            set(source_totals.keys())
        )

        # Step 3: Compute Beta-Binomial scores
        scores = {}
        for source_id, total in source_totals.items():
            positives = source_positives.get(source_id, 0)
            score = self._beta_binomial_score(positives, total)
            scores[source_id] = round(score, 4)

        # Step 4: Store in learning.db
        self._store_scores(scores, source_totals, source_positives)

        # Step 5: Update cache
        with self._lock:
            self._cached_scores = dict(scores)

        logger.info(
            "Source quality scores computed for %d sources: %s",
            len(scores),
            ", ".join(
                "%s=%.3f" % (s, sc) for s, sc in sorted(
                    scores.items(), key=lambda x: -x[1]
                )[:5]
            ) + ("..." if len(scores) > 5 else ""),
        )

        return scores

    def get_source_boost(
        self,
        memory: dict,
        source_scores: Optional[Dict[str, float]] = None,
    ) -> float:
        """
        Get the ranking boost for a memory based on its source quality.

        This is called by the adaptive ranker at query time for each
        candidate memory. The boost is a float in [0.0, 1.0] that
        represents how trustworthy/useful this source tends to be.

        Args:
            memory: A memory dict. Must have 'created_by' key, or will
                    fall back to DEFAULT_QUALITY_SCORE.
            source_scores: Optional pre-computed scores dict. If None,
                           uses the internal cache. Pass this to avoid
                           repeated cache reads in a tight loop.

        Returns:
            Quality score (0.0 to 1.0). 0.5 for unknown sources.
        """
        scores = source_scores if source_scores is not None else self._cached_scores

        # Extract source identifier from the memory
        source_id = self._extract_source_id(memory)

        if not source_id or source_id not in scores:
            return DEFAULT_QUALITY_SCORE

        return scores[source_id]

    def refresh(self):
        """
        Recompute all source scores.

        Convenience wrapper for compute_source_scores(). Called periodically
        by the engagement tracker or on explicit user request.
        """
        return self.compute_source_scores()

    # ======================================================================
    # Data Extraction (memory.db — READ-ONLY)
    # ======================================================================

    def _get_memory_counts_by_source(self) -> Dict[str, int]:
        """
        Count total memories per source from memory.db's `created_by` column.

        Handles the case where the `created_by` column does not exist
        (older databases pre-v2.5). In that case, all memories are
        grouped under 'unknown'.

        Returns:
            Dict mapping source_id -> total memory count.
        """
        counts: Dict[str, int] = {}

        try:
            conn = sqlite3.connect(str(self.memory_db_path), timeout=10)
            try:
                conn.execute("PRAGMA busy_timeout=5000")
                cursor = conn.cursor()

                # Check if created_by column exists
                cursor.execute("PRAGMA table_info(memories)")
                columns = {row[1] for row in cursor.fetchall()}

                if "created_by" in columns:
                    cursor.execute("""
                        SELECT
                            COALESCE(created_by, 'unknown') AS source,
                            COUNT(*) AS cnt
                        FROM memories
                        GROUP BY source
                        ORDER BY cnt DESC
                    """)
                    for row in cursor.fetchall():
                        source_id = row[0] if row[0] else "unknown"
                        counts[source_id] = row[1]
                else:
                    # Column doesn't exist — count all as 'unknown'
                    cursor.execute("SELECT COUNT(*) FROM memories")
                    total = cursor.fetchone()[0]
                    if total > 0:
                        counts["unknown"] = total
                    logger.debug(
                        "created_by column not in memory.db — "
                        "all %d memories grouped as 'unknown'.",
                        total,
                    )
            finally:
                conn.close()

        except sqlite3.OperationalError as e:
            logger.warning("Error reading memory counts by source: %s", e)
        except Exception as e:
            logger.error("Unexpected error reading memory.db: %s", e)

        return counts

    def _get_positive_signal_counts(
        self,
        known_sources: set,
    ) -> Dict[str, int]:
        """
        Count positive feedback signals per source.

        Joins learning.db's ranking_feedback (positive signals) with
        memory.db's memories (to get created_by) on memory_id.

        This requires reading from BOTH databases. We do a two-step approach:
            1. Get all memory_ids with positive feedback from learning.db
            2. Look up their created_by from memory.db

        This avoids ATTACH DATABASE which can have locking issues.

        Returns:
            Dict mapping source_id -> positive signal count.
        """
        positives: Dict[str, int] = {}

        if self._learning_db is None:
            return positives

        # Step 1: Get memory_ids with positive feedback from learning.db
        feedback_memory_ids: Dict[int, int] = {}  # memory_id -> count

        try:
            feedback_rows = self._learning_db.get_feedback_for_training(limit=50000)
            for row in feedback_rows:
                signal_type = row.get("signal_type", "")
                if signal_type in POSITIVE_SIGNAL_TYPES:
                    mem_id = row.get("memory_id")
                    if mem_id is not None:
                        feedback_memory_ids[mem_id] = (
                            feedback_memory_ids.get(mem_id, 0) + 1
                        )
        except Exception as e:
            logger.warning("Could not read feedback from learning.db: %s", e)
            return positives

        if not feedback_memory_ids:
            return positives

        # Step 2: Look up created_by for each feedback memory_id in memory.db
        try:
            conn = sqlite3.connect(str(self.memory_db_path), timeout=10)
            try:
                conn.execute("PRAGMA busy_timeout=5000")
                cursor = conn.cursor()

                # Check if created_by column exists
                cursor.execute("PRAGMA table_info(memories)")
                columns = {row[1] for row in cursor.fetchall()}

                if "created_by" not in columns:
                    # All positives go to 'unknown'
                    total_positives = sum(feedback_memory_ids.values())
                    if total_positives > 0:
                        positives["unknown"] = total_positives
                    return positives

                # Batch lookup in chunks to avoid SQLite variable limit
                mem_ids = list(feedback_memory_ids.keys())
                chunk_size = 500  # SQLite max variables is 999

                for i in range(0, len(mem_ids), chunk_size):
                    chunk = mem_ids[i:i + chunk_size]
                    placeholders = ",".join("?" * len(chunk))
                    cursor.execute(
                        "SELECT id, COALESCE(created_by, 'unknown') "
                        "FROM memories WHERE id IN (%s)" % placeholders,
                        chunk,
                    )
                    for row in cursor.fetchall():
                        mem_id = row[0]
                        source_id = row[1] if row[1] else "unknown"
                        count = feedback_memory_ids.get(mem_id, 0)
                        positives[source_id] = positives.get(source_id, 0) + count
            finally:
                conn.close()

        except sqlite3.OperationalError as e:
            logger.warning("Error looking up memory sources: %s", e)
        except Exception as e:
            logger.error("Unexpected error in positive signal lookup: %s", e)

        return positives

    # ======================================================================
    # Scoring Math
    # ======================================================================

    @staticmethod
    def _beta_binomial_score(positive_count: int, total_count: int) -> float:
        """
        Compute Beta-Binomial smoothed quality score.

        Formula: (alpha + positive) / (alpha + beta + total)

        With alpha=1, beta=1 (uniform prior / Laplace smoothing):
            - 0 positives, 0 total = 0.50 (neutral)
            - 5 positives, 10 total = 0.50
            - 8 positives, 10 total = 0.75
            - 1 positive, 10 total = 0.17
            - 50 positives, 100 total = 0.50

        This converges to the true rate as evidence grows, while being
        conservative (pulled toward 0.5) with limited data.

        Args:
            positive_count: Number of positive feedback signals.
            total_count: Total number of memories from this source.

        Returns:
            Quality score in [0.0, 1.0].
        """
        score = (ALPHA + positive_count) / (ALPHA + BETA + total_count)
        return max(0.0, min(1.0, score))

    # ======================================================================
    # Storage (learning.db)
    # ======================================================================

    def _store_scores(
        self,
        scores: Dict[str, float],
        totals: Dict[str, int],
        positives: Dict[str, int],
    ):
        """
        Store computed scores in learning.db's source_quality table.

        Uses LearningDB.update_source_quality() which handles UPSERT
        internally with its own write lock.
        """
        if self._learning_db is None:
            logger.debug(
                "LearningDB unavailable — scores computed but not stored."
            )
            return

        stored = 0
        for source_id, score in scores.items():
            try:
                self._learning_db.update_source_quality(
                    source_id=source_id,
                    positive_signals=positives.get(source_id, 0),
                    total_memories=totals.get(source_id, 0),
                )
                stored += 1
            except Exception as e:
                logger.error(
                    "Failed to store score for source '%s': %s",
                    source_id, e,
                )

        logger.debug("Stored %d/%d source quality scores.", stored, len(scores))

    def _load_cached_scores(self):
        """
        Load source quality scores from learning.db into the in-memory cache.

        Called on initialization so that get_source_boost() works immediately
        without requiring a compute_source_scores() call first.
        """
        if self._learning_db is None:
            return

        try:
            db_scores = self._learning_db.get_source_scores()
            with self._lock:
                self._cached_scores = dict(db_scores)
            if db_scores:
                logger.debug(
                    "Loaded %d cached source scores from learning.db.",
                    len(db_scores),
                )
        except Exception as e:
            logger.debug("Could not load cached source scores: %s", e)

    # ======================================================================
    # Utility Methods
    # ======================================================================

    @staticmethod
    def _extract_source_id(memory: dict) -> Optional[str]:
        """
        Extract the source identifier from a memory dict.

        Checks 'created_by' first (set by ProvenanceTracker), then
        falls back to 'source_protocol' if available.

        Args:
            memory: A memory dict (from search results or direct DB query).

        Returns:
            Source identifier string, or None if not available.
        """
        # Primary: created_by (e.g., 'mcp:claude-desktop', 'cli:terminal')
        source = memory.get("created_by")
        if source and source != "user":
            return source

        # Fallback: source_protocol (e.g., 'mcp', 'cli', 'rest')
        protocol = memory.get("source_protocol")
        if protocol:
            return protocol

        # Last resort: the 'user' default from provenance_tracker
        if source == "user":
            return "user"

        return None

    def get_all_scores(self) -> Dict[str, dict]:
        """
        Get detailed quality information for all tracked sources.

        Returns full details including positive signals, total memories,
        and computed score for diagnostic/dashboard display.

        Returns:
            Dict mapping source_id -> {quality_score, positive_signals,
            total_memories, last_updated}
        """
        if self._learning_db is None:
            # Return from cache with minimal info
            with self._lock:
                return {
                    source_id: {
                        "quality_score": score,
                        "positive_signals": None,
                        "total_memories": None,
                        "last_updated": None,
                    }
                    for source_id, score in self._cached_scores.items()
                }

        try:
            conn = self._learning_db._get_connection()
            cursor = conn.cursor()
            cursor.execute("""
                SELECT source_id, quality_score, positive_signals,
                       total_memories, last_updated
                FROM source_quality
                ORDER BY quality_score DESC
            """)
            results = {}
            for row in cursor.fetchall():
                results[row["source_id"]] = {
                    "quality_score": row["quality_score"],
                    "positive_signals": row["positive_signals"],
                    "total_memories": row["total_memories"],
                    "last_updated": row["last_updated"],
                }
            conn.close()
            return results
        except Exception as e:
            logger.error("Failed to read detailed source scores: %s", e)
            return {}

    def get_source_summary(self) -> str:
        """
        Get a human-readable summary of source quality scores.

        Returns:
            Formatted multi-line string for diagnostics or dashboard.
        """
        all_scores = self.get_all_scores()

        if not all_scores:
            return "No source quality data available. Run refresh() first."

        lines = ["Source Quality Scores:", ""]
        lines.append(
            "  %-30s  %8s  %8s  %8s"
            % ("Source", "Score", "Positive", "Total")
        )
        lines.append("  " + "-" * 62)

        for source_id, data in sorted(
            all_scores.items(), key=lambda x: -x[1]["quality_score"]
        ):
            pos = data["positive_signals"]
            tot = data["total_memories"]
            lines.append(
                "  %-30s  %8.3f  %8s  %8s"
                % (
                    source_id,
                    data["quality_score"],
                    str(pos) if pos is not None else "?",
                    str(tot) if tot is not None else "?",
                )
            )

        return "\n".join(lines)


# ===========================================================================
# CLI Interface
# ===========================================================================

if __name__ == "__main__":
    import sys as _sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    scorer = SourceQualityScorer()

    if len(_sys.argv) < 2:
        print("SourceQualityScorer — Per-Source Quality Learning")
        print()
        print("Usage:")
        print("  python source_quality_scorer.py compute    # Compute all source scores")
        print("  python source_quality_scorer.py show       # Show current scores")
        print("  python source_quality_scorer.py summary    # Human-readable summary")
        _sys.exit(0)

    command = _sys.argv[1]

    if command == "compute":
        scores = scorer.compute_source_scores()
        if scores:
            print("\nComputed quality scores for %d sources:" % len(scores))
            for source_id, score in sorted(scores.items(), key=lambda x: -x[1]):
                bar = "#" * int(score * 20)
                print("  %-30s  %.3f  [%-20s]" % (source_id, score, bar))
        else:
            print("No sources found. Add memories with provenance tracking first.")

    elif command == "show":
        all_scores = scorer.get_all_scores()
        if all_scores:
            print("\nStored source quality scores:")
            for source_id, data in sorted(
                all_scores.items(), key=lambda x: -x[1]["quality_score"]
            ):
                print(
                    "  %-30s  score=%.3f  positives=%s  total=%s"
                    % (
                        source_id,
                        data["quality_score"],
                        data["positive_signals"],
                        data["total_memories"],
                    )
                )
        else:
            print("No scores stored. Run 'compute' first.")

    elif command == "summary":
        print(scorer.get_source_summary())

    else:
        print("Unknown command: %s" % command)
        _sys.exit(1)
