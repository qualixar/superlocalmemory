# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""SuperLocalMemory V3 — Profile Channel (Entity-Profile Retrieval).

Returns fact IDs from entity profiles — enables direct answers for
"What does Alice do?" style queries without full embedding search.

This is a SHORTCUT channel: if the query mentions a known entity,
the profile's accumulated fact IDs are injected directly into the
retrieval pool with high scores.

Competitor reference: EverMemOS profile synthesis (~+15-20% SH).

Part of Qualixar | Author: Varun Pratap Bhardwaj
License: AGPL-3.0-or-later
"""
from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from superlocalmemory.storage.database import DatabaseManager

logger = logging.getLogger(__name__)

# Pattern for extracting potential entity names (capitalized multi-word)
_ENTITY_PATTERN = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b")

# Common sentence starters to exclude from entity extraction
_SENTENCE_STARTERS = frozenset({
    "What", "Where", "Who", "Which", "How", "When", "Does", "Did",
    "Can", "Could", "Would", "Should", "Are", "Is", "Was", "Were",
    "Has", "Have", "The", "Tell", "Please", "Do", "Why",
})


class ProfileChannel:
    """Entity-profile-based retrieval for direct entity queries.

    If the query mentions a known entity (by canonical name or alias),
    the entity's profile fact IDs are returned with high base score.

    Usage::

        channel = ProfileChannel(db)
        results = channel.search("What is Alice's job?", "default")
        # returns [(fact_id_1, 0.95), (fact_id_2, 0.95), ...]
    """

    def __init__(self, db: DatabaseManager) -> None:
        self._db = db

    def search(
        self,
        query: str,
        profile_id: str,
        top_k: int = 10,
    ) -> list[tuple[str, float]]:
        """Search entity profiles for matching facts.

        Args:
            query: User query text.
            profile_id: Scope to this profile.
            top_k: Maximum results to return.

        Returns:
            List of (fact_id, score) sorted by score descending.
        """
        entities = self._extract_entity_names(query)
        if not entities:
            return []

        results: list[tuple[str, float]] = []
        seen: set[str] = set()

        for name in entities:
            entity = self._db.get_entity_by_name(name, profile_id)
            if not entity:
                continue

            profiles = self._db.get_entity_profiles_by_entity(
                entity.entity_id, profile_id,
            )
            for p in profiles:
                for fid in p.fact_ids:
                    if fid not in seen:
                        seen.add(fid)
                        results.append((fid, 0.95))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    @staticmethod
    def _extract_entity_names(query: str) -> list[str]:
        """Extract potential entity names from query text.

        Returns capitalized words/phrases that aren't common
        sentence starters.
        """
        matches = _ENTITY_PATTERN.findall(query)
        return [m for m in matches if m not in _SENTENCE_STARTERS]
