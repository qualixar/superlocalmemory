# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""SuperLocalMemory V3 — BM25 Keyword Search Channel.

Persistent BM25Plus index over fact content. Catches exact name/date
matches that embedding similarity misses.

V1 bug fix: V1 kept BM25 tokens in-memory only — a restart lost
the entire index. This version persists tokens to the DB via
store_bm25_tokens / get_all_bm25_tokens and cold-loads on init.

Part of Qualixar | Author: Varun Pratap Bhardwaj
License: Elastic-2.0
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

from rank_bm25 import BM25Plus

if TYPE_CHECKING:
    from superlocalmemory.storage.database import DatabaseManager

logger = logging.getLogger(__name__)

# Minimal stopwords — small set to avoid stripping important terms
_STOPWORDS: frozenset[str] = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "shall",
    "should", "may", "might", "must", "can", "could", "of", "in", "to",
    "for", "with", "on", "at", "from", "by", "as", "into", "through",
    "and", "but", "or", "nor", "not", "so", "yet", "if", "then", "than",
    "that", "this", "it", "its", "i", "me", "my", "we", "our", "you",
    "your", "he", "him", "his", "she", "her", "they", "them", "their",
})

# Token pattern: words with letters/digits, keeps hyphens and apostrophes
_TOKEN_RE = re.compile(r"[a-zA-Z0-9][\w'-]*[a-zA-Z0-9]|[a-zA-Z0-9]")


def tokenize(text: str) -> list[str]:
    """Tokenize text: lowercase, split, remove stopwords.

    Exported so encoding pipeline can persist tokens at ingest time.
    """
    tokens = _TOKEN_RE.findall(text.lower())
    return [t for t in tokens if t not in _STOPWORDS]


class BM25Channel:
    """Persistent BM25Plus index for keyword retrieval.

    On cold start, loads all tokens from the DB. After that, new facts
    are added incrementally. The BM25Plus model is rebuilt lazily
    before each search when the corpus has changed.

    Attributes:
        document_count: Number of indexed documents.
    """

    def __init__(self, db: DatabaseManager) -> None:
        self._db = db
        self._corpus: list[list[str]] = []
        self._fact_ids: list[str] = []
        self._fact_id_set: set[str] = set()
        self._raw_texts: list[str] = []  # V3.3.12: raw content for phrase matching
        self._bm25: BM25Plus | None = None
        self._dirty: bool = False
        self._loaded_profiles: set[str] = set()

    @property
    def document_count(self) -> int:
        return len(self._corpus)

    def ensure_loaded(self, profile_id: str) -> None:
        """Cold-load BM25 tokens from DB for a profile (once).

        Idempotent: subsequent calls for the same profile are no-ops.
        """
        if profile_id in self._loaded_profiles:
            return

        token_map = self._db.get_all_bm25_tokens(profile_id)
        _inc_global = getattr(self, 'include_global', True)
        _inc_shared = getattr(self, 'include_shared', True)
        if not token_map:
            # Fallback: tokenize facts directly if no pre-stored tokens
            facts = self._db.get_all_facts(
                profile_id,
                include_global=_inc_global,
                include_shared=_inc_shared,
            )
            for fact in facts:
                if fact.fact_id in self._fact_id_set:
                    continue
                tokens = tokenize(fact.content)
                if tokens:
                    self._corpus.append(tokens)
                    self._fact_ids.append(fact.fact_id)
                    self._fact_id_set.add(fact.fact_id)
                    self._raw_texts.append(fact.content)
                    # Persist for next cold start
                    self._db.store_bm25_tokens(fact.fact_id, profile_id, tokens)
        else:
            # Load raw texts for phrase matching (V3.3.12)
            fact_content_map = {}
            try:
                facts = self._db.get_all_facts(
                    profile_id,
                    include_global=_inc_global,
                    include_shared=_inc_shared,
                )
                fact_content_map = {f.fact_id: f.content for f in facts}
            except Exception:
                pass
            for fid, tokens in token_map.items():
                if fid in self._fact_id_set:
                    continue
                self._corpus.append(tokens)
                self._fact_ids.append(fid)
                self._fact_id_set.add(fid)
                self._raw_texts.append(fact_content_map.get(fid, ""))

        self._dirty = True
        self._loaded_profiles.add(profile_id)
        logger.debug(
            "BM25 cold-loaded %d documents for profile=%s",
            len(token_map) if token_map else 0, profile_id,
        )

    def add(self, fact_id: str, content: str, profile_id: str) -> None:
        """Add a single fact to the index and persist tokens.

        Args:
            fact_id: Unique fact identifier.
            content: Raw text content to index.
            profile_id: Owner profile.
        """
        tokens = tokenize(content)
        if not tokens:
            return

        self._corpus.append(tokens)
        self._fact_ids.append(fact_id)
        self._fact_id_set.add(fact_id)
        if not hasattr(self, '_raw_texts'):
            self._raw_texts = []
        self._raw_texts.append(content)
        self._dirty = True

        # Persist for cold start
        self._db.store_bm25_tokens(fact_id, profile_id, tokens)

    def _fts5_search(
        self, query: str, profile_id: str, top_k: int = 30,
    ) -> list[tuple[str, float]]:
        """v3.5.0: SQLite FTS5 keyword search (C-level indexed, scales to millions).

        Uses the ``atomic_facts_fts`` external-content FTS5 table (kept in sync
        by INSERT/DELETE/UPDATE triggers). Joins ``atomic_facts`` for profile
        scoping (FTS table has no profile_id). ``bm25()`` returns a negative
        score (lower = better match); we negate it to the channel's
        "higher = better" convention. Returns [] on no matches.

        Raises (OperationalError) if the FTS5 table is absent — the caller
        then falls back to the legacy in-memory rank_bm25 path.
        """
        tokens = tokenize(query)
        if not tokens:
            return []
        # Quote each token so query punctuation can't break FTS5 MATCH syntax;
        # OR-join for high recall (any token may match).
        match_expr = " OR ".join('"' + t.replace('"', "") + '"' for t in tokens)
        sql = (
            "SELECT af.fact_id AS fact_id, bm25(atomic_facts_fts) AS rank "
            "FROM atomic_facts_fts "
            "JOIN atomic_facts af ON af.rowid = atomic_facts_fts.rowid "
            "WHERE atomic_facts_fts MATCH ? AND af.profile_id = ? "
            "ORDER BY rank LIMIT ?"
        )
        rows = self._db.execute(sql, (match_expr, profile_id, int(top_k)))
        out: list[tuple[str, float]] = []
        for r in rows:
            d = dict(r)
            fid = d.get("fact_id")
            if not fid:
                continue
            out.append((fid, -float(d.get("rank", 0.0))))
        return out

    def search(
        self,
        query: str,
        profile_id: str,
        top_k: int = 30,
    ) -> list[tuple[str, float]]:
        """Search BM25 index for matching facts.

        Auto-loads from DB on first call for this profile.

        Args:
            query: Search query text.
            profile_id: Scope to this profile.
            top_k: Maximum results.

        Returns:
            List of (fact_id, bm25_score) sorted by score descending.
        """
        # v3.5.0: FTS5 fast path — C-level indexed, ~ms, scales to millions.
        # The legacy in-memory rank_bm25 path rebuilt the whole index over the
        # entire corpus on every corpus change (11s+ at 17.5k facts, does not
        # scale). FTS5 (atomic_facts_fts, kept in sync by triggers) replaces it.
        # Falls back to rank_bm25 ONLY if the FTS5 table is genuinely
        # unavailable (raises) — e.g. a pre-FTS legacy DB.
        try:
            return self._fts5_search(query, profile_id, top_k)
        except Exception as exc:  # pragma: no cover — legacy/missing FTS table
            logger.debug(
                "BM25 FTS5 path unavailable, using rank_bm25 fallback: %s", exc,
            )

        self.ensure_loaded(profile_id)

        if not self._corpus:
            return []

        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        # Rebuild BM25 model if corpus changed
        if self._dirty or self._bm25 is None:
            self._bm25 = BM25Plus(self._corpus, k1=1.2, b=0.75)
            self._dirty = False

        scores = self._bm25.get_scores(query_tokens)

        scored: list[tuple[str, float]] = []
        # V3.3.12: Exact phrase bonus — boost facts containing the full query phrase
        query_lower = query.lower().strip()
        for i, score in enumerate(scores):
            if score > 0.0:
                bonus = score
                # Exact phrase match bonus: if the query appears as a substring in the document
                if len(query_lower) >= 5 and i < len(self._raw_texts):
                    if query_lower in self._raw_texts[i].lower():
                        bonus *= 1.5  # 50% boost for exact phrase match
                scored.append((self._fact_ids[i], bonus))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def clear(self) -> None:
        """Clear the in-memory index (does NOT delete DB tokens)."""
        self._corpus = []
        self._fact_ids = []
        self._fact_id_set = set()
        self._bm25 = None
        self._dirty = False
        self._loaded_profiles = set()
