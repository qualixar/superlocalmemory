# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Cross-project knowledge aggregator.

Transfers learned preferences and patterns from completed projects to
new ones via temporal-decay merging and contradiction detection.
Ported from V2 CrossProjectAggregator with inline sqlite3 queries.

Part of Qualixar | Author: Varun Pratap Bhardwaj
"""

from __future__ import annotations

import json
import logging
import math
import re
import sqlite3
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Temporal decay half-life in days (1 year)
DECAY_HALF_LIFE_DAYS = 365.0

# Contradiction detection window in days
CONTRADICTION_WINDOW_DAYS = 90

# Minimum evidence to consider a pattern valid
MIN_EVIDENCE_FOR_MERGE = 2

# Minimum confidence for a merged pattern to be stored
MIN_MERGE_CONFIDENCE = 0.3

# Simple keyword regex for frequency analysis
_WORD_RE = re.compile(r"[a-zA-Z0-9_]+")

# Technology categories detected from content
TECH_CATEGORIES: dict[str, list[str]] = {
    "frontend_framework": [
        "react", "vue", "angular", "svelte", "next", "nuxt", "solid",
    ],
    "backend_framework": [
        "express", "fastapi", "django", "flask", "spring", "rails", "nest",
    ],
    "language": [
        "python", "typescript", "javascript", "rust", "go", "java", "kotlin",
    ],
    "database": [
        "postgres", "mysql", "sqlite", "mongo", "redis", "dynamodb", "supabase",
    ],
    "cloud": ["aws", "azure", "gcp", "vercel", "cloudflare", "netlify"],
    "testing": ["pytest", "jest", "vitest", "cypress", "playwright"],
}

# ---------------------------------------------------------------------------
# Schema for transferable patterns
# ---------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS transferable_patterns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    profile_id TEXT NOT NULL DEFAULT 'default',
    pattern_type TEXT NOT NULL DEFAULT 'preference',
    key TEXT NOT NULL,
    value TEXT NOT NULL,
    confidence REAL DEFAULT 0.0,
    evidence_count INTEGER DEFAULT 0,
    profiles_seen INTEGER DEFAULT 1,
    decay_factor REAL DEFAULT 1.0,
    contradictions TEXT DEFAULT '[]',
    first_seen TEXT,
    last_seen TEXT,
    UNIQUE(profile_id, pattern_type, key)
);

CREATE TABLE IF NOT EXISTS memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    profile_id TEXT DEFAULT 'default',
    content TEXT,
    project_name TEXT,
    created_at TEXT
);
"""


class CrossProjectAggregator:
    """Aggregate technology preferences across profiles."""

    def __init__(self, db_path: Path | str) -> None:
        self._db_path = Path(db_path)
        self._ensure_schema()

    def aggregate(
        self, source_profiles: list[str], target_profile: str
    ) -> list[dict]:
        """Transfer patterns from *source_profiles* to *target_profile*."""
        if not source_profiles:
            return []

        # Step 1: Analyse each source profile
        profile_patterns: list[dict] = []
        for profile_id in source_profiles:
            pdata = self._analyse_profile(profile_id)
            if pdata:
                profile_patterns.append(pdata)

        if not profile_patterns:
            return []

        # Step 2: Merge with temporal decay
        merged = self._merge_with_decay(profile_patterns)

        # Step 3: Detect contradictions (scoped to the target profile)
        for key, pattern_data in merged.items():
            pattern_data["contradictions"] = self._detect_contradictions(
                key, pattern_data, target_profile
            )

        # Step 4: Store and return (scoped to the target profile)
        self._store_patterns(merged, target_profile)
        return [{"key": k, **v} for k, v in merged.items()]

    def get_preferences(
        self,
        profile_id: str = "default",
        min_confidence: float = 0.6,
    ) -> dict[str, dict]:
        """Retrieve stored transferable preferences for *profile_id* above *min_confidence*."""
        conn = sqlite3.connect(str(self._db_path))
        conn.row_factory = sqlite3.Row
        try:
            cur = conn.execute(
                "SELECT * FROM transferable_patterns "
                "WHERE profile_id = ? AND confidence >= ? ORDER BY confidence DESC",
                (profile_id, min_confidence),
            )
            result: dict[str, dict] = {}
            for row in cur.fetchall():
                d = dict(row)
                contradictions = _parse_json_list(d.get("contradictions", "[]"))
                result[d["key"]] = {
                    "value": d["value"],
                    "confidence": d["confidence"],
                    "evidence_count": d["evidence_count"],
                    "profiles_seen": d.get("profiles_seen", 1),
                    "decay_factor": d.get("decay_factor", 1.0),
                    "contradictions": contradictions,
                }
            return result
        except sqlite3.OperationalError:
            return {}
        finally:
            conn.close()

    def _analyse_profile(self, profile_id: str) -> Optional[dict]:
        """Detect tech category preferences in a single profile."""
        conn = sqlite3.connect(str(self._db_path))
        conn.row_factory = sqlite3.Row
        try:
            cur = conn.execute(
                "SELECT content, created_at FROM memories "
                "WHERE profile_id = ? ORDER BY created_at DESC LIMIT 500",
                (profile_id,),
            )
            rows = cur.fetchall()
        except sqlite3.OperationalError:
            return None
        finally:
            conn.close()

        if not rows:
            return None

        # Detect tech keywords per category
        category_counts: dict[str, Counter] = {
            cat: Counter() for cat in TECH_CATEGORIES
        }
        for row in rows:
            content = (dict(row).get("content") or "").lower()
            tokens = set(_WORD_RE.findall(content))
            for category, keywords in TECH_CATEGORIES.items():
                for kw in keywords:
                    if kw in tokens:
                        category_counts[category][kw] += 1

        # Build patterns: winner per category with enough evidence
        patterns: dict[str, dict] = {}
        for category, counter in category_counts.items():
            if not counter:
                continue
            winner, win_count = counter.most_common(1)[0]
            total = sum(counter.values())
            if win_count >= MIN_EVIDENCE_FOR_MERGE:
                patterns[category] = {
                    "value": winner,
                    "confidence": round(win_count / total, 3) if total else 0.0,
                    "evidence_count": win_count,
                }

        if not patterns:
            return None

        # Latest timestamp for decay calculation
        latest_ts = dict(rows[0]).get("created_at", datetime.now(UTC).isoformat())

        return {
            "profile": profile_id,
            "patterns": patterns,
            "latest_timestamp": latest_ts,
            "memory_count": len(rows),
        }

    def _merge_with_decay(
        self, profile_patterns: list[dict]
    ) -> dict[str, dict]:
        now = datetime.now(UTC)
        contributions: dict[str, list[dict]] = {}

        for pdata in profile_patterns:
            age_days = _days_since(pdata["latest_timestamp"], now)
            weight = math.exp(-age_days / DECAY_HALF_LIFE_DAYS)

            for cat_key, pattern in pdata["patterns"].items():
                contributions.setdefault(cat_key, []).append({
                    "value": pattern["value"],
                    "confidence": pattern["confidence"],
                    "evidence_count": pattern["evidence_count"],
                    "weight": weight,
                    "profile": pdata["profile"],
                    "latest_timestamp": pdata["latest_timestamp"],
                })

        merged: dict[str, dict] = {}
        for cat_key, contribs in contributions.items():
            result = self._merge_single(contribs)
            if result is not None:
                merged[cat_key] = result
        return merged

    @staticmethod
    def _merge_single(contributions: list[dict]) -> Optional[dict]:
        """Merge contributions for one category across profiles."""
        value_scores: dict[str, float] = {}
        value_evidence: dict[str, int] = {}
        value_profiles: dict[str, set] = {}
        total_weighted = 0.0

        for c in contributions:
            v = c["value"]
            w_ev = c["evidence_count"] * c["weight"]
            value_scores[v] = value_scores.get(v, 0.0) + w_ev
            value_evidence[v] = value_evidence.get(v, 0) + c["evidence_count"]
            value_profiles.setdefault(v, set()).add(c["profile"])
            total_weighted += w_ev

        if total_weighted == 0:
            return None

        winner = max(value_scores, key=lambda k: value_scores[k])
        confidence = value_scores[winner] / total_weighted
        ev = value_evidence[winner]

        if ev < MIN_EVIDENCE_FOR_MERGE or confidence < MIN_MERGE_CONFIDENCE:
            return None

        return {
            "value": winner,
            "confidence": round(min(0.95, confidence), 3),
            "evidence_count": ev,
            "profiles_seen": len(value_profiles[winner]),
            "decay_factor": round(
                max(c["weight"] for c in contributions if c["value"] == winner),
                4,
            ),
            "contradictions": [],
        }

    def _detect_contradictions(
        self, pattern_key: str, pattern_data: dict, profile_id: str = "default"
    ) -> list[str]:
        contradictions: list[str] = []

        # Check stored value vs new value, scoped to this profile.
        conn = sqlite3.connect(str(self._db_path))
        conn.row_factory = sqlite3.Row
        try:
            cur = conn.execute(
                "SELECT value, last_seen FROM transferable_patterns "
                "WHERE profile_id = ? AND key = ? AND pattern_type = 'preference'",
                (profile_id, pattern_key),
            )
            row = cur.fetchone()
            if row:
                d = dict(row)
                old_val = d.get("value", "")
                if old_val and old_val != pattern_data["value"]:
                    old_ts = d.get("last_seen", "")
                    if old_ts and _is_within_window(old_ts, CONTRADICTION_WINDOW_DAYS):
                        contradictions.append(
                            f"Changed from '{old_val}' to "
                            f"'{pattern_data['value']}' within "
                            f"{CONTRADICTION_WINDOW_DAYS} days"
                        )
        except sqlite3.OperationalError:
            pass
        finally:
            conn.close()

        return contradictions

    def _ensure_schema(self) -> None:
        conn = sqlite3.connect(str(self._db_path))
        try:
            conn.executescript(_SCHEMA)
        finally:
            conn.close()

    def _store_patterns(
        self, merged: dict[str, dict], profile_id: str = "default"
    ) -> None:
        """Persist merged patterns for *profile_id*. Each profile has isolated rows."""
        conn = sqlite3.connect(str(self._db_path))
        now = datetime.now(UTC).isoformat()
        try:
            for key, data in merged.items():
                conn.execute(
                    """INSERT INTO transferable_patterns
                       (profile_id, pattern_type, key, value, confidence,
                        evidence_count, profiles_seen, decay_factor,
                        contradictions, first_seen, last_seen)
                       VALUES (?, 'preference', ?, ?, ?, ?, ?, ?, ?, ?, ?)
                       ON CONFLICT(profile_id, pattern_type, key) DO UPDATE SET
                           value = excluded.value,
                           confidence = excluded.confidence,
                           evidence_count = excluded.evidence_count,
                           profiles_seen = excluded.profiles_seen,
                           decay_factor = excluded.decay_factor,
                           contradictions = excluded.contradictions,
                           last_seen = excluded.last_seen
                    """,
                    (
                        profile_id,
                        key,
                        data["value"],
                        data["confidence"],
                        data["evidence_count"],
                        data.get("profiles_seen", 1),
                        data.get("decay_factor", 1.0),
                        json.dumps(data.get("contradictions", [])),
                        now,
                        now,
                    ),
                )
            conn.commit()
        finally:
            conn.close()


# ----------------------------------------------------------------------
# Module-level helpers
# ----------------------------------------------------------------------


def _parse_json_list(raw: Any) -> list:
    """Parse a JSON string into a list, returning [] on failure."""
    if isinstance(raw, list):
        return raw
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            pass
    return []


def _days_since(timestamp_str: str, now: datetime | None = None) -> float:
    if now is None:
        now = datetime.now(UTC)
    if not timestamp_str:
        return 0.0
    try:
        ts = datetime.fromisoformat(str(timestamp_str).replace(" ", "T"))
        return max(0.0, (now - ts).total_seconds() / 86400.0)
    except (ValueError, AttributeError, TypeError):
        pass
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%Y-%m-%dT%H:%M:%S.%f"):
        try:
            ts = datetime.strptime(str(timestamp_str), fmt)
            return max(0.0, (now - ts).total_seconds() / 86400.0)
        except (ValueError, TypeError):
            continue
    return 0.0


def _is_within_window(timestamp_str: str, window_days: int) -> bool:
    if not timestamp_str:
        return False
    try:
        ts = datetime.fromisoformat(str(timestamp_str).replace(" ", "T"))
        return (datetime.now(UTC) - ts).days <= window_days
    except (ValueError, AttributeError, TypeError):
        return False
