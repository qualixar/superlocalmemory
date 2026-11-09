# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Workflow pattern miner -- sliding-window sequence and temporal mining.

Detects repeating workflow sequences and time-of-day activity patterns
from memory creation timestamps and content.  Uses n-gram sliding
windows (length 2-5) over a classified activity stream.

Seven activity types: docs, architecture, code, test, debug, deploy, config.

Ported from V2 WorkflowPatternMiner with V2 LearningDB deps removed.
Direct sqlite3 for storage.

Part of Qualixar | Author: Varun Pratap Bhardwaj
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Activity taxonomy (7 categories)
# ---------------------------------------------------------------------------

ACTIVITY_TYPES: dict[str, list[str]] = {
    "docs": [
        "documentation", "readme", "wiki", "spec", "prd",
        "design doc", "changelog", "api doc",
    ],
    "architecture": [
        "architecture", "diagram", "system design", "schema",
        "api design", "data model", "erd",
    ],
    "code": [
        "implement", "function", "class", "module", "refactor",
        "code", "feature", "component",
    ],
    "test": [
        "test", "pytest", "jest", "coverage", "assertion",
        "mock", "spec", "unit test",
    ],
    "debug": [
        "bug", "fix", "error", "stack trace", "debug",
        "issue", "exception", "traceback",
    ],
    "deploy": [
        "deploy", "docker", "ci/cd", "pipeline", "release",
        "production", "staging", "build",
    ],
    "config": [
        "config", "env", "settings", "setup", "install",
        "dependency", "package", "requirements",
    ],
}

# Pre-compiled regex per keyword for word-boundary matching
_KEYWORD_PATTERNS: list[tuple[str, re.Pattern]] = []
for _act, _kws in ACTIVITY_TYPES.items():
    for _kw in _kws:
        _KEYWORD_PATTERNS.append(
            (_act, re.compile(r"\b" + re.escape(_kw) + r"\b", re.IGNORECASE))
        )

# ---------------------------------------------------------------------------
# Schema for local action log
# ---------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS workflow_actions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    profile_id TEXT NOT NULL,
    action TEXT NOT NULL,
    metadata TEXT DEFAULT '{}',
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_wf_profile
    ON workflow_actions(profile_id, created_at);

CREATE TABLE IF NOT EXISTS workflow_patterns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    profile_id TEXT NOT NULL,
    pattern_type TEXT NOT NULL,
    pattern_key TEXT NOT NULL,
    pattern_value TEXT DEFAULT '{}',
    confidence REAL DEFAULT 0.0,
    evidence_count INTEGER DEFAULT 0,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_wp_profile
    ON workflow_patterns(profile_id, pattern_type);
"""


class WorkflowMiner:
    """Mine workflow sequences and temporal patterns from memory content.

    Args:
        db_path: Path to a sqlite database.  If the file does not exist
                 it is created with the required schema.
    """

    def __init__(self, db_path: Path | str) -> None:
        self._db_path = Path(db_path)
        self._ensure_schema()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_action(
        self, profile_id: str, action: str, metadata: dict[str, Any] | None = None
    ) -> None:
        """Record a user action for later mining."""
        now = datetime.now(UTC).isoformat()
        meta_json = json.dumps(metadata or {})
        conn = sqlite3.connect(str(self._db_path))
        try:
            conn.execute(
                "INSERT INTO workflow_actions (profile_id, action, metadata, created_at) "
                "VALUES (?, ?, ?, ?)",
                (profile_id, action, meta_json, now),
            )
            conn.commit()
        finally:
            conn.close()

    def mine(self, profile_id: str, min_support: float = 0.3) -> list[dict]:
        """Mine workflow sequence patterns for a profile.

        Returns a list of pattern dicts sorted by support descending.
        """
        actions = self._fetch_actions(profile_id)
        if len(actions) < 2:
            return []

        activity_stream = [a["action"] for a in actions]
        return self._mine_sequences(activity_stream, min_support)

    def mine_from_memories(
        self,
        memories: list[dict],
        min_support: float = 0.3,
    ) -> list[dict]:
        """Mine sequences from a pre-fetched list of memory dicts.

        Each dict should have a ``content`` key.
        """
        stream: list[str] = []
        for mem in memories:
            activity = classify_activity(mem.get("content", ""))
            if activity != "unknown":
                stream.append(activity)
        if len(stream) < 2:
            return []
        return self._mine_sequences(stream, min_support)

    def mine_temporal(self, memories: list[dict]) -> dict[str, dict]:
        """Detect time-of-day activity preferences.

        Returns dict keyed by bucket (morning/afternoon/evening/night).
        Buckets with < 5 evidence memories are omitted.
        """
        buckets: dict[str, Counter] = {
            "morning": Counter(),
            "afternoon": Counter(),
            "evening": Counter(),
            "night": Counter(),
        }

        for mem in memories:
            activity = classify_activity(mem.get("content", ""))
            if activity == "unknown":
                continue
            hour = _parse_hour(mem.get("created_at"))
            if hour is None:
                continue
            bucket = _hour_to_bucket(hour)
            buckets[bucket][activity] += 1

        result: dict[str, dict] = {}
        for bucket_name, counter in buckets.items():
            total = sum(counter.values())
            if total < 5:
                continue
            dominant, dom_count = counter.most_common(1)[0]
            result[bucket_name] = {
                "dominant_activity": dominant,
                "confidence": round(dom_count / total, 4),
                "evidence_count": total,
                "distribution": dict(counter),
            }
        return result

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _ensure_schema(self) -> None:
        conn = sqlite3.connect(str(self._db_path))
        try:
            conn.executescript(_SCHEMA)
        finally:
            conn.close()

    def _fetch_actions(self, profile_id: str) -> list[dict]:
        conn = sqlite3.connect(str(self._db_path))
        conn.row_factory = sqlite3.Row
        try:
            cur = conn.execute(
                "SELECT action, created_at FROM workflow_actions "
                "WHERE profile_id = ? ORDER BY created_at ASC LIMIT 500",
                (profile_id,),
            )
            return [dict(r) for r in cur.fetchall()]
        except sqlite3.OperationalError:
            return []
        finally:
            conn.close()

    @staticmethod
    def _mine_sequences(
        activity_stream: list[str], min_support: float
    ) -> list[dict]:
        """Extract n-gram sequences and filter by support."""
        all_patterns: list[dict] = []

        for n in range(2, 6):
            if len(activity_stream) < n:
                continue
            ngram_counts: Counter = Counter()
            total_windows = len(activity_stream) - n + 1

            for i in range(total_windows):
                ngram = tuple(activity_stream[i : i + n])
                # Skip consecutive identical activities (noise)
                if any(ngram[j] == ngram[j + 1] for j in range(len(ngram) - 1)):
                    continue
                ngram_counts[ngram] += 1

            for ngram, count in ngram_counts.items():
                support = count / total_windows if total_windows > 0 else 0.0
                if support >= min_support:
                    all_patterns.append({
                        "sequence": list(ngram),
                        "support": round(support, 4),
                        "count": count,
                        "length": n,
                    })

        all_patterns.sort(key=lambda p: (-p["support"], -p["length"]))
        return all_patterns[:20]


# ----------------------------------------------------------------------
# Module-level helpers
# ----------------------------------------------------------------------


def classify_activity(content: str) -> str:
    """Classify content into one of 7 activity types or 'unknown'."""
    if not content:
        return "unknown"
    scores: Counter = Counter()
    for act_type, pattern in _KEYWORD_PATTERNS:
        if pattern.search(content):
            scores[act_type] += 1
    if not scores:
        return "unknown"
    return scores.most_common(1)[0][0]


def _hour_to_bucket(hour: int) -> str:
    if 6 <= hour <= 11:
        return "morning"
    if 12 <= hour <= 17:
        return "afternoon"
    if 18 <= hour <= 23:
        return "evening"
    return "night"


def _parse_hour(timestamp: str | None) -> int | None:
    if not timestamp:
        return None
    for fmt in (
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S.%f",
    ):
        try:
            return datetime.strptime(timestamp, fmt).hour
        except (ValueError, TypeError):
            continue
    try:
        return datetime.fromisoformat(timestamp).hour
    except (ValueError, TypeError):
        return None
