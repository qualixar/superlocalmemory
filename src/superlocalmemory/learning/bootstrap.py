# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Synthetic bootstrap for cold-start ML ranking.

Generates (query, fact, label) triples from existing memory patterns
so the adaptive ranker can operate before real feedback accumulates.

Four strategies:
  1. Access-based  -- frequently accessed memories are positive for their keywords
  2. Importance-based -- high-importance memories are positive for their tags
  3. Pattern-based -- learned identity patterns generate synthetic queries
  4. Recency-based -- recent memories rank higher for shared-topic queries

Ported from V2 SyntheticBootstrapper with V2-specific deps removed.

Part of Qualixar | Author: Varun Pratap Bhardwaj
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import sqlite3
from collections import Counter
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Stopwords for keyword extraction (no NLP dep)
# ---------------------------------------------------------------------------
_STOPWORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "shall",
    "should", "may", "might", "can", "could", "to", "of", "in", "for",
    "on", "with", "at", "by", "from", "as", "into", "through", "during",
    "before", "after", "above", "below", "between", "under", "again",
    "further", "then", "once", "here", "there", "when", "where", "why",
    "how", "all", "each", "every", "both", "few", "more", "most", "other",
    "some", "such", "no", "not", "only", "own", "same", "so", "than",
    "too", "very", "just", "because", "but", "and", "or", "if", "while",
    "about", "up", "out", "off", "over", "also", "it", "its", "this",
    "that", "i", "me", "my", "we", "our", "you", "your", "he", "him",
    "she", "her", "they", "them", "what", "which", "who", "whom",
})

_WORD_RE = re.compile(r"[a-zA-Z0-9_]+")


class SyntheticBootstrap:
    """Generate synthetic training data for cold-start ML ranking.

    Args:
        db_path: Path to the sqlite database containing a ``memories`` table.
                 A fresh database is created if the file does not exist.
    """

    def __init__(self, db_path: Path | str) -> None:
        self._db_path = Path(db_path)
        self._ensure_schema()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, profile_id: str, count: int = 200) -> list[dict]:
        """Generate synthetic (query, fact_id, label) triples.

        Returns up to *count* records.  Returns an empty list when
        the profile has no memories to mine from.
        """
        memories = self._fetch_memories(profile_id)
        if not memories:
            return []

        records: list[dict] = []
        records.extend(self._access_based(memories))
        records.extend(self._importance_based(memories))
        records.extend(self._recency_based(memories))

        # Trim to target count with source diversity
        if len(records) > count:
            records = self._diverse_sample(records, count)

        return records

    # ------------------------------------------------------------------
    # Strategy 1: access-based
    # ------------------------------------------------------------------

    def _access_based(self, memories: list[dict]) -> list[dict]:
        """Memories accessed 5+ times are positive for their keywords."""
        records: list[dict] = []
        high_access = [m for m in memories if m.get("access_count", 0) >= 5]

        for mem in high_access:
            keywords = _extract_keywords(mem.get("content", ""))
            if not keywords:
                continue
            query = " ".join(keywords)

            records.append(_build_record(query, mem, label=1.0, source="access_pos"))

            # Pick negatives from memories with different tags
            for neg in self._find_negatives(memories, mem, limit=2):
                records.append(_build_record(query, neg, label=0.0, source="access_neg"))

        return records

    # ------------------------------------------------------------------
    # Strategy 2: importance-based
    # ------------------------------------------------------------------

    def _importance_based(self, memories: list[dict]) -> list[dict]:
        """High-importance memories (>=8) are positive for their tags."""
        records: list[dict] = []
        important = [m for m in memories if (m.get("importance") or 0) >= 8]

        for mem in important:
            query = self._tags_to_query(mem)
            if not query:
                keywords = _extract_keywords(mem.get("content", ""))
                query = " ".join(keywords) if keywords else ""
            if not query:
                continue

            records.append(_build_record(query, mem, label=1.0, source="importance_pos"))

            for neg in self._find_negatives(memories, mem, limit=2):
                records.append(_build_record(query, neg, label=0.0, source="importance_neg"))

        return records

    # ------------------------------------------------------------------
    # Strategy 3: recency-based
    # ------------------------------------------------------------------

    def _recency_based(self, memories: list[dict]) -> list[dict]:
        """Recent memories rank higher for shared-topic queries."""
        records: list[dict] = []
        recent = memories[:30]  # Already sorted by created_at DESC
        if len(recent) < 4:
            return records

        seen_queries: set[str] = set()
        for mem in recent[:15]:
            keywords = _extract_keywords(mem.get("content", ""))
            query = " ".join(keywords) if keywords else ""
            if not query or query in seen_queries:
                continue
            seen_queries.add(query)

            records.append(_build_record(query, mem, label=0.8, source="recency_pos"))

            # Older memories about the same topic are weak negatives
            for older in memories[30:]:
                content = (older.get("content") or "").lower()
                if any(kw in content for kw in keywords):
                    records.append(
                        _build_record(query, older, label=0.3, source="recency_neg")
                    )
                    break

        return records

    # ------------------------------------------------------------------
    # Database
    # ------------------------------------------------------------------

    def _ensure_schema(self) -> None:
        """Create the memories table if it does not exist (for testing)."""
        conn = sqlite3.connect(str(self._db_path))
        try:
            conn.execute(
                """CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    profile_id TEXT DEFAULT 'default',
                    content TEXT,
                    tags TEXT,
                    importance INTEGER DEFAULT 5,
                    access_count INTEGER DEFAULT 0,
                    created_at TEXT
                )"""
            )
            conn.commit()
        finally:
            conn.close()

    def _fetch_memories(self, profile_id: str) -> list[dict]:
        """Fetch memories for a profile, ordered by recency."""
        conn = sqlite3.connect(str(self._db_path))
        conn.row_factory = sqlite3.Row
        try:
            cur = conn.execute(
                "SELECT * FROM memories WHERE profile_id = ? "
                "ORDER BY created_at DESC LIMIT 500",
                (profile_id,),
            )
            return [dict(r) for r in cur.fetchall()]
        except sqlite3.OperationalError:
            return []
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _tags_to_query(memory: dict) -> str:
        """Extract a query string from the memory's tags field."""
        tags = memory.get("tags", "")
        if not tags:
            return ""
        if isinstance(tags, list):
            return " ".join(tags[:5])
        try:
            parsed = json.loads(tags)
            if isinstance(parsed, list):
                return " ".join(str(t) for t in parsed[:5])
        except (json.JSONDecodeError, TypeError):
            pass
        return " ".join(t.strip() for t in str(tags).split(",") if t.strip())[:5]

    @staticmethod
    def _find_negatives(
        memories: list[dict], anchor: dict, limit: int = 2
    ) -> list[dict]:
        """Pick memories dissimilar to anchor (different tags or id)."""
        anchor_id = anchor.get("id")
        anchor_tags = set(
            t.strip().lower()
            for t in str(anchor.get("tags", "")).split(",")
            if t.strip()
        )
        negatives: list[dict] = []
        for mem in memories:
            if mem.get("id") == anchor_id:
                continue
            mem_tags = set(
                t.strip().lower()
                for t in str(mem.get("tags", "")).split(",")
                if t.strip()
            )
            if not mem_tags & anchor_tags:
                negatives.append(mem)
                if len(negatives) >= limit:
                    break
        return negatives

    @staticmethod
    def _diverse_sample(records: list[dict], target: int) -> list[dict]:
        """Sample records proportionally across sources."""
        by_source: dict[str, list[dict]] = {}
        for r in records:
            by_source.setdefault(r["source"], []).append(r)
        n_sources = len(by_source) or 1
        per_source = max(1, target // n_sources)
        sampled: list[dict] = []
        for src_records in by_source.values():
            sampled.extend(src_records[:per_source])
        return sampled[:target]


# ----------------------------------------------------------------------
# Module-level helpers (stateless)
# ----------------------------------------------------------------------


def _extract_keywords(content: str, top_n: int = 3) -> list[str]:
    """Extract top-N keywords from text via frequency (no NLP deps)."""
    if not content:
        return []
    tokens = _WORD_RE.findall(content.lower())
    filtered = [t for t in tokens if t not in _STOPWORDS and len(t) > 2]
    counts = Counter(filtered)
    return [word for word, _ in counts.most_common(top_n)]


def _build_record(
    query: str,
    memory: dict,
    label: float,
    source: str,
) -> dict:
    """Build a single training record."""
    return {
        "query": query,
        "query_hash": hashlib.sha256(query.encode()).hexdigest()[:16],
        "fact_id": memory.get("id", 0),
        "label": label,
        "source": source,
    }
