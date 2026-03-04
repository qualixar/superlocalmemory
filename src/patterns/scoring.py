#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 SuperLocalMemory (superlocalmemory.com)
"""
Confidence Scoring - Bayesian pattern confidence calculation.

Uses Beta-Binomial posterior with log-scaled competition,
recency bonuses, and temporal distribution factors.
"""

import sqlite3
import logging
from datetime import datetime, timedelta
from typing import List
from pathlib import Path

logger = logging.getLogger(__name__)


class ConfidenceScorer:
    """Calculates and tracks pattern confidence scores."""

    def __init__(self, db_path: Path):
        self.db_path = db_path

    def calculate_confidence(
        self,
        pattern_type: str,
        key: str,
        value: str,
        evidence_memory_ids: List[int],
        total_memories: int
    ) -> float:
        """
        Calculate confidence using Beta-Binomial Bayesian posterior.

        Based on MACLA (arXiv:2512.18950, Forouzandeh et al., Dec 2025):
          posterior_mean = (alpha + evidence) / (alpha + beta + evidence + competition)

        Adaptation: MACLA's Beta-Binomial uses pairwise interaction counts.
        Our corpus has sparse signals (most memories are irrelevant to any
        single pattern). We use log-scaled competition instead of raw total
        to avoid over-dilution: competition = log2(total_memories).

        Pattern-specific priors (alpha, beta):
        - preference (1, 4): prior mean 0.20, ~8 items to reach 0.5
        - style (1, 5): prior mean 0.17, subtler signals need more evidence
        - terminology (2, 3): prior mean 0.40, direct usage signal
        """
        if total_memories == 0 or not evidence_memory_ids:
            return 0.0

        import math
        evidence_count = len(evidence_memory_ids)

        # Pattern-specific Beta priors (alpha, beta)
        PRIORS = {
            'preference': (1.0, 4.0),
            'style':      (1.0, 5.0),
            'terminology': (2.0, 3.0),
        }
        alpha, beta = PRIORS.get(pattern_type, (1.0, 4.0))

        # Log-scaled competition: grows slowly with corpus size
        # 10 memories -> 3.3, 60 -> 5.9, 500 -> 9.0, 5000 -> 12.3
        competition = math.log2(max(2, total_memories))

        # MACLA-inspired Beta posterior with log competition
        posterior_mean = (alpha + evidence_count) / (alpha + beta + evidence_count + competition)

        # Recency adjustment (mild: 1.0 to 1.15)
        recency_bonus = self._calculate_recency_bonus(evidence_memory_ids)
        recency_factor = 1.0 + min(0.15, 0.075 * (recency_bonus - 1.0) / 0.2) if recency_bonus > 1.0 else 1.0

        # Temporal spread adjustment (0.9 to 1.1)
        distribution_factor = self._calculate_distribution_factor(evidence_memory_ids)

        # Final confidence
        confidence = posterior_mean * recency_factor * distribution_factor

        return min(0.95, round(confidence, 3))

    def _calculate_recency_bonus(self, memory_ids: List[int]) -> float:
        """Give bonus to patterns with recent evidence."""
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()

            # Get timestamps
            placeholders = ','.join('?' * len(memory_ids))
            cursor.execute(f'''
                SELECT created_at FROM memories
                WHERE id IN ({placeholders})
                ORDER BY created_at DESC
            ''', memory_ids)

            timestamps = cursor.fetchall()
        finally:
            conn.close()

        if not timestamps:
            return 1.0

        # Check if any memories are from last 30 days
        recent_count = 0
        cutoff = datetime.now() - timedelta(days=30)

        for ts_tuple in timestamps:
            ts_str = ts_tuple[0]
            try:
                ts = datetime.fromisoformat(ts_str.replace(' ', 'T'))
                if ts > cutoff:
                    recent_count += 1
            except (ValueError, AttributeError):
                pass

        # Bonus if >50% are recent
        if len(timestamps) > 0 and recent_count / len(timestamps) > 0.5:
            return 1.2
        else:
            return 1.0

    def _calculate_distribution_factor(self, memory_ids: List[int]) -> float:
        """Better confidence if memories are distributed over time, not just one session."""
        if len(memory_ids) < 3:
            return 0.8  # Penalize low sample size

        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()

            placeholders = ','.join('?' * len(memory_ids))
            cursor.execute(f'''
                SELECT created_at FROM memories
                WHERE id IN ({placeholders})
                ORDER BY created_at
            ''', memory_ids)

            timestamps = [row[0] for row in cursor.fetchall()]
        finally:
            conn.close()

        if len(timestamps) < 2:
            return 0.8

        try:
            # Parse timestamps
            dates = []
            for ts_str in timestamps:
                try:
                    ts = datetime.fromisoformat(ts_str.replace(' ', 'T'))
                    dates.append(ts)
                except (ValueError, AttributeError):
                    pass

            if len(dates) < 2:
                return 0.8

            # Calculate time span
            time_span = (dates[-1] - dates[0]).days

            # If memories span multiple days, higher confidence
            if time_span > 7:
                return 1.1
            elif time_span > 1:
                return 1.0
            else:
                return 0.9  # All on same day = might be one-off

        except Exception:
            return 1.0
