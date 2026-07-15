# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3

"""A-MEM-inspired automatic edge creation between related facts.

Creates association_edges ONLY (never writes to graph_edges -- Rule 13).
Disabled in Mode A without embedding provider.

Responsibilities:
1. link_new_fact(): Find similar facts via VectorStore, create edges
2. evolve_linked_facts(): Update contextual_description of linked old facts
3. strengthen_co_access(): Hebbian +0.05 for co-accessed pairs
4. decay_unused(): Weight decay for stale edges

Part of Qualixar | Author: Varun Pratap Bhardwaj
License: AGPL-3.0-or-later
"""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np

from superlocalmemory.storage.models import AtomicFact, _new_id

logger = logging.getLogger(__name__)


class AutoLinker:
    """A-MEM automatic edge creation between related facts.

    Creates association_edges ONLY (never graph_edges -- Rule 13).
    """

    SIMILARITY_THRESHOLD: float = 0.7    # A-MEM auto-link threshold
    HEBBIAN_INCREMENT: float = 0.05      # Per co-access weight boost
    MAX_LINKS_PER_FACT: int = 10         # Cap auto-links per new fact
    DECAY_RATE: float = 0.01             # Per-day weight decay
    DECAY_MIN_WEIGHT: float = 0.05       # Minimum weight before deletion

    def __init__(
        self,
        db: Any,
        vector_store: Any,
        context_generator: Any | None = None,
        config: Any | None = None,
    ) -> None:
        self._db = db
        self._vector_store = vector_store
        self._context_gen = context_generator
        self._config = config
        self._mode = config.mode.value if config else "a"

    def link_new_fact(
        self, new_fact: AtomicFact, profile_id: str,
    ) -> list[str]:
        """Find similar facts via VectorStore KNN, create association_edges.

        Called in store_pipeline AFTER GraphBuilder (which writes graph_edges).
        Returns: list of linked fact_ids.
        """
        if new_fact.embedding is None:
            logger.debug(
                "AutoLinker: link_new_fact skipped (no embedding) "
                "for fact %s",
                new_fact.fact_id,
            )
            return []

        if not self._vector_store or not self._vector_store.available:
            return []

        try:
            query_vec = np.asarray(new_fact.embedding, dtype=np.float32)
            candidates = self._vector_store.search(
                query_vec.tolist(),
                top_k=self.MAX_LINKS_PER_FACT + 1,
            )
        except Exception as exc:
            logger.debug("AutoLinker: VectorStore search failed: %s", exc)
            return []

        linked_ids: list[str] = []
        for fact_id, score in candidates:
            if fact_id == new_fact.fact_id:
                continue
            if score < self.SIMILARITY_THRESHOLD:
                continue
            if len(linked_ids) >= self.MAX_LINKS_PER_FACT:
                break

            # Create bidirectional edge (INSERT OR IGNORE for idempotency)
            self._create_edge(
                profile_id,
                new_fact.fact_id,
                fact_id,
                association_type="auto_link",
                weight=round(float(score), 4),
            )
            linked_ids.append(fact_id)

        # Memory evolution: update context of linked old facts
        if linked_ids and self._context_gen:
            self.evolve_linked_facts(
                new_fact.fact_id, linked_ids, profile_id,
            )

        return linked_ids

    def evolve_linked_facts(
        self,
        new_fact_id: str,
        linked_fact_ids: list[str],
        profile_id: str,
    ) -> None:
        """Update contextual_description of linked old facts.

        A-MEM memory evolution: old facts gain new context from new links.
        Updates fact_context table ONLY (immutability preserved -- Rule 17).
        Non-blocking, fire-and-forget. Errors logged, never raised (Rule 19).
        """
        try:
            new_fact = self._db.get_fact(new_fact_id)
            if not new_fact:
                return

            for old_fact_id in linked_fact_ids:
                old_fact = self._db.get_fact(old_fact_id)
                if not old_fact:
                    continue

                if self._mode == "a":
                    self._append_keyword(old_fact_id, new_fact, profile_id)
                else:
                    if self._context_gen:
                        self._regenerate_context(
                            old_fact_id, old_fact, new_fact, profile_id,
                        )
        except Exception as exc:
            logger.debug(
                "evolve_linked_facts failed (non-fatal): %s", exc,
            )

    def strengthen_co_access(
        self, fact_ids: list[str], profile_id: str,
    ) -> int:
        """Hebbian strengthening: +0.05 weight for facts recalled together.

        For each pair (i, j) in the recalled set: if an association_edge
        exists, increment weight by HEBBIAN_INCREMENT (capped at 1.0).
        Returns: number of edges strengthened.
        """
        if len(fact_ids) < 2:
            return 0

        strengthened = 0
        for i in range(len(fact_ids)):
            for j in range(i + 1, len(fact_ids)):
                rows = self._db.execute(
                    "SELECT edge_id, weight, co_access_count "
                    "FROM association_edges "
                    "WHERE profile_id = ? "
                    "AND source_fact_id = ? AND target_fact_id = ? "
                    "UNION ALL "
                    "SELECT edge_id, weight, co_access_count "
                    "FROM association_edges "
                    "WHERE profile_id = ? "
                    "AND source_fact_id = ? AND target_fact_id = ?",
                    (profile_id, fact_ids[i], fact_ids[j],
                     profile_id, fact_ids[j], fact_ids[i]),
                )
                for row in rows:
                    d = dict(row)
                    new_weight = min(
                        1.0, d["weight"] + self.HEBBIAN_INCREMENT,
                    )
                    new_count = d["co_access_count"] + 1
                    self._db.execute(
                        "UPDATE association_edges "
                        "SET weight = ?, co_access_count = ?, "
                        "    last_strengthened = datetime('now') "
                        "WHERE edge_id = ?",
                        (new_weight, new_count, d["edge_id"]),
                    )
                    strengthened += 1

        return strengthened

    def decay_unused(
        self, profile_id: str, days_threshold: int = 30,
    ) -> int:
        """Decay weights of association_edges not used in N days.

        Formula: new_weight = weight * exp(-DECAY_RATE * days)
        Edges below DECAY_MIN_WEIGHT are deleted.
        Returns: number of edges decayed or deleted.
        """
        rows = self._db.execute(
            "SELECT edge_id, weight, last_strengthened, created_at "
            "FROM association_edges "
            "WHERE profile_id = ? AND ("
            "  last_strengthened IS NULL "
            "  AND created_at < datetime('now', ?)"
            "  OR last_strengthened < datetime('now', ?)"
            ")",
            (profile_id,
             f'-{days_threshold} days',
             f'-{days_threshold} days'),
        )

        affected = 0
        for row in rows:
            d = dict(row)
            days_inactive = days_threshold  # Conservative estimate
            new_weight = d["weight"] * math.exp(
                -self.DECAY_RATE * days_inactive,
            )

            if new_weight < self.DECAY_MIN_WEIGHT:
                self._db.execute(
                    "DELETE FROM association_edges WHERE edge_id = ?",
                    (d["edge_id"],),
                )
            else:
                self._db.execute(
                    "UPDATE association_edges SET weight = ? "
                    "WHERE edge_id = ?",
                    (round(new_weight, 4), d["edge_id"]),
                )
            affected += 1

        return affected

    # --- Private helpers ---

    def _create_edge(
        self,
        profile_id: str,
        source_id: str,
        target_id: str,
        association_type: str,
        weight: float,
    ) -> None:
        """Insert a single association_edge. Idempotent via UNIQUE index."""
        self._db.execute(
            "INSERT OR IGNORE INTO association_edges "
            "(edge_id, profile_id, source_fact_id, target_fact_id, "
            " association_type, weight, co_access_count, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, 0, datetime('now'))",
            (_new_id(), profile_id, source_id, target_id,
             association_type, weight),
        )

    def _append_keyword(
        self, fact_id: str, new_fact: AtomicFact, profile_id: str,
    ) -> None:
        """Mode A: append new fact's first entity as keyword."""
        keyword = ""
        if new_fact.canonical_entities:
            keyword = new_fact.canonical_entities[0]
        elif new_fact.content:
            words = new_fact.content.split()
            keyword = words[0] if words else ""
        if not keyword:
            return

        try:
            rows = self._db.execute(
                "SELECT keywords FROM fact_context "
                "WHERE fact_id = ? AND profile_id = ?",
                (fact_id, profile_id),
            )
            if rows:
                existing = dict(rows[0]).get("keywords", "") or ""
                if keyword.lower() not in existing.lower():
                    updated = (
                        (existing + ", " + keyword) if existing else keyword
                    )
                    self._db.execute(
                        "UPDATE fact_context SET keywords = ? "
                        "WHERE fact_id = ? AND profile_id = ?",
                        (updated, fact_id, profile_id),
                    )
        except Exception as exc:
            logger.debug("_append_keyword failed: %s", exc)

    def _regenerate_context(
        self,
        old_fact_id: str,
        old_fact: AtomicFact,
        new_fact: AtomicFact,
        profile_id: str,
    ) -> None:
        """Mode B/C: regenerate contextual_description for old fact."""
        try:
            if self._context_gen:
                new_desc = self._context_gen.generate(old_fact, self._mode)
                self._db.execute(
                    "UPDATE fact_context SET contextual_description = ? "
                    "WHERE fact_id = ? AND profile_id = ?",
                    (new_desc, old_fact_id, profile_id),
                )
        except Exception as exc:
            logger.debug("_regenerate_context failed: %s", exc)
