# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""SuperLocalMemory V3 — Entity Resolution.

Maps variant mentions ("Alice", "Ms. Smith", "she") to canonical entities.
Persists ALL resolution results to DB — the #1 fix from the V1 audit where
entity resolution ran but results were silently discarded.

4-tier strategy:
  a) Exact match in canonical_entities (case-insensitive)
  b) Alias match in entity_aliases (case-insensitive)
  c) Fuzzy match via Jaro-Winkler similarity (threshold 0.85)
  d) LLM disambiguation (Mode B/C only)

Part of Qualixar | Author: Varun Pratap Bhardwaj
License: AGPL-3.0-or-later
"""

from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING

from superlocalmemory.storage.models import CanonicalEntity, EntityAlias, _new_id, _now

if TYPE_CHECKING:
    from superlocalmemory.llm.backbone import LLMBackbone
    from superlocalmemory.storage.database import DatabaseManager

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

JARO_WINKLER_AUTO_MERGE: float = 0.85   # Auto-merge threshold
JARO_WINKLER_LLM_FLOOR: float = 0.70    # Below this, never merge
PRONOUNS: frozenset[str] = frozenset({
    "he", "she", "they", "him", "her", "them", "his", "hers", "their",
    "himself", "herself", "themselves", "it", "its", "i", "me", "my",
    "we", "us", "our", "you", "your",
})

# Heuristic entity type patterns
_ORG_MARKERS = ("Inc", "Corp", "LLC", "Ltd", "University", "Hospital", "Bank",
                "Foundation", "Institute", "Company", "Group", "Agency")
_PLACE_MARKERS = ("City", "State", "County", "Island", "River", "Mountain",
                  "Lake", "Park", "Street", "Avenue", "Road", "District")
_EVENT_MARKERS = ("Festival", "Conference", "Summit", "Workshop", "Meeting",
                  "Election", "War", "Match", "Game", "Concert", "Wedding")
# v3.4.10: Skill entity type — skills, commands, agents, plugins
_SKILL_MARKERS_RE = re.compile(r'\b(?:skill|command|agent|plugin|hook|mcp)\b', re.IGNORECASE)
_SKILL_NAMESPACE_RE = re.compile(r"^[\w-]+:[\w-]+$")  # e.g., "superpowers:brainstorming"


# ---------------------------------------------------------------------------
# Jaro-Winkler similarity — pure Python fallback
# ---------------------------------------------------------------------------

def jaro_winkler(s1: str, s2: str, prefix_weight: float = 0.1) -> float:
    """Jaro-Winkler similarity in [0, 1]. 1 = identical.

    Pure Python implementation. Capped at 200 chars for O(n^2) safety.
    """
    s1, s2 = s1[:200], s2[:200]
    if not s1 or not s2:
        return 0.0
    if s1 == s2:
        return 1.0

    len1, len2 = len(s1), len(s2)
    match_dist = max(len1, len2) // 2 - 1
    if match_dist < 0:
        match_dist = 0

    s1_matched = [False] * len1
    s2_matched = [False] * len2
    matches = transpositions = 0

    for i in range(len1):
        lo = max(0, i - match_dist)
        hi = min(i + match_dist + 1, len2)
        for j in range(lo, hi):
            if s2_matched[j] or s1[i] != s2[j]:
                continue
            s1_matched[i] = s2_matched[j] = True
            matches += 1
            break

    if matches == 0:
        return 0.0

    k = 0
    for i in range(len1):
        if not s1_matched[i]:
            continue
        while not s2_matched[k]:
            k += 1
        if s1[i] != s2[k]:
            transpositions += 1
        k += 1

    jaro = (
        matches / len1 + matches / len2
        + (matches - transpositions / 2) / matches
    ) / 3.0

    prefix = sum(
        1 for i in range(min(4, len1, len2)) if s1[i] == s2[i]
    )
    return jaro + prefix * prefix_weight * (1.0 - jaro)


_COMMON_WORDS = frozenset({
    # Months / time words (biggest source of garbage entities)
    "january", "february", "march", "april", "may", "june", "july",
    "august", "september", "october", "november", "december",
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "today", "tomorrow", "yesterday", "morning", "evening", "night",
    # English stop words that get capitalized at sentence start
    "a", "an", "the", "all", "not", "no", "yes", "and", "or", "but",
    "if", "is", "are", "was", "were", "be", "been", "being", "have",
    "has", "had", "do", "does", "did", "will", "would", "shall", "should",
    "can", "could", "just", "also", "only", "very", "too", "so", "then",
    "than", "that", "this", "these", "those", "here", "there", "where",
    "when", "what", "which", "who", "whom", "how", "why", "each", "every",
    "both", "few", "more", "most", "other", "some", "such", "any", "many",
    "much", "own", "same", "new", "old", "first", "last", "next", "now",
    "still", "already", "always", "never", "often", "sometimes", "about",
    "above", "after", "again", "against", "along", "among", "around",
    "before", "below", "between", "beyond", "during", "except", "from",
    "into", "near", "off", "onto", "out", "over", "past", "since",
    "through", "under", "until", "upon", "with", "within", "without",
    # Technical stop words (common in dev sessions)
    "phase", "test", "gap", "dashboard", "remaining", "session", "results",
    "tools", "projects", "prompts", "integration", "cli", "engagement",
    "mode", "error", "step", "fix", "build", "check", "run", "start",
    "stop", "config", "status", "version", "query", "data", "file", "path",
    "node", "edge", "table", "index", "schema", "model", "type", "class",
    "function", "module", "package", "import", "export", "default", "pattern",
    "memory", "profile", "context", "pipeline", "worker", "daemon",
    "server", "client", "route", "endpoint", "handler", "hook",
    "feature", "release", "update", "upgrade", "deploy", "debug", "log",
    "output", "input", "key", "value", "true", "false", "null", "none",
    "ready", "done", "todo", "complete", "pending", "active", "failed",
    "success", "warning", "critical", "high", "medium", "low",
    "total", "count", "list", "item", "entry", "record", "row", "column",
    "source", "target", "origin", "destination", "backup", "restore",
    "create", "read", "delete", "remove", "add", "set", "get", "put",
    "push", "pull", "fetch", "send", "receive", "request", "response",
    "enable", "disable", "open", "close", "load", "save", "reset",
    # Abstract nouns often misclassified as people
    "completeness", "correctness", "limitations", "requirements",
    "dependencies", "performance", "security", "quality", "coverage",
    "progress", "analysis", "research", "implementation", "verification",
    "overview", "summary", "details", "notes", "changes", "issues",
    "approach", "strategy", "solution", "problem", "question", "answer",
})


def _guess_entity_type(name: str) -> str:
    """Heuristic entity type classification from name string.

    v3.4.10: Aggressive false-positive prevention. "person" is assigned ONLY
    when the name looks like a real human name (2-3 capitalized words, none
    in the stop list). Everything else defaults to "concept".
    """
    # Reject very short or very long names
    if len(name) <= 2 or len(name) > 100:
        return "concept"

    # Reject pure numbers, dates, version strings
    if re.match(r"^[\d.v\-/]+$", name):
        return "concept"

    if any(m in name for m in _ORG_MARKERS):
        return "organization"
    if any(m in name for m in _PLACE_MARKERS):
        return "place"
    if any(m in name for m in _EVENT_MARKERS):
        return "event"

    # v3.4.10: Skill entities — namespaced skills always skill type
    if _SKILL_NAMESPACE_RE.match(name):
        return "skill"

    # Check ALL words against the stop list (not just the full name)
    words = name.lower().split()
    if any(w in _COMMON_WORDS for w in words):
        return "concept"

    # v3.4.10: Skill entities — word-boundary match AFTER common-word filter
    if _SKILL_MARKERS_RE.search(name):
        return "skill"

    # Multi-word entity: "person" only if 2-3 capitalized words, no stop words
    if re.match(r"^[A-Z][a-z]+ [A-Z][a-z]+( [A-Z][a-z]+)?$", name):
        if not any(p in _COMMON_WORDS for p in words):
            return "person"

    # Single capitalized word — almost never a person in our context
    # Only known first names should get "person" but we can't maintain
    # a name dictionary, so default to "concept"
    if re.match(r"^[A-Z][a-z]+$", name):
        return "concept"

    # Contains dots/slashes/hyphens/underscores = technical term
    if re.search(r"[./\-_]", name):
        return "concept"

    # ALL-CAPS or mixed case with numbers = technical/concept
    if re.match(r"^[A-Z]+$", name) or re.search(r"\d", name):
        return "concept"

    return "concept"


# ---------------------------------------------------------------------------
# Entity Resolver
# ---------------------------------------------------------------------------

class EntityResolver:
    """Resolves raw entity mentions to persisted canonical entities.

    Every resolution is persisted — canonical entities and aliases are stored
    immediately, ensuring downstream graph building and retrieval use the
    resolved identities.
    """

    def __init__(
        self,
        db: DatabaseManager,
        llm: LLMBackbone | None = None,
    ) -> None:
        self._db = db
        self._llm = llm

    # -- Public API ---------------------------------------------------------

    def resolve(
        self,
        raw_entities: list[str],
        profile_id: str,
    ) -> dict[str, str]:
        """Resolve raw mentions to canonical entity IDs.

        Returns mapping: raw_name -> canonical entity_id.
        All new entities and aliases are persisted before returning.
        """
        if not raw_entities:
            return {}

        resolution: dict[str, str] = {}
        candidates_for_llm: list[str] = []

        for raw in raw_entities:
            name = raw.strip()
            if not name or name.lower() in PRONOUNS:
                continue

            # Skip very short/long entities
            if len(name) <= 2 or len(name) > 100:
                continue

            # Skip single-word stop words
            words = name.lower().split()
            if len(words) == 1 and name.lower() in _COMMON_WORDS:
                continue

            # Skip multi-word entities where ALL words are stop words or <=2 chars
            if len(words) > 1 and all(w in _COMMON_WORDS or len(w) <= 2 for w in words):
                continue

            # Skip pure numbers/versions
            if re.match(r"^[\d.v\-/]+$", name):
                continue

            # Tier a: exact match on canonical_name
            entity = self._db.get_entity_by_name(name, profile_id)
            if entity is not None:
                resolution[raw] = entity.entity_id
                self._touch_last_seen(entity.entity_id, profile_id)
                continue

            # Tier b: alias match (case-insensitive, indexed)
            entity_id = self._alias_lookup(name, profile_id)
            if entity_id is not None:
                resolution[raw] = entity_id
                self._touch_last_seen(entity_id, profile_id)
                continue

            # Tier c: fuzzy match via Jaro-Winkler
            match_id, score = self._fuzzy_match(name, profile_id)
            if match_id is not None and score >= JARO_WINKLER_AUTO_MERGE:
                resolution[raw] = match_id
                self._persist_alias(match_id, name, score, "jaro_winkler", profile_id)
                self._touch_last_seen(match_id, profile_id)
                continue

            # Candidate zone (0.70–0.85): queue for LLM in Mode B/C
            if match_id is not None and score >= JARO_WINKLER_LLM_FLOOR:
                candidates_for_llm.append(raw)
                continue

            # No match at all — create new entity
            new_id = self._create_entity(name, profile_id)
            resolution[raw] = new_id

        # Tier d: LLM disambiguation for fuzzy candidates (Mode B/C)
        if candidates_for_llm and self._llm is not None:
            llm_resolved = self._llm_disambiguate(
                candidates_for_llm, profile_id,
            )
            for raw_name, entity_id in llm_resolved.items():
                resolution[raw_name] = entity_id

            # Create new entities for candidates LLM couldn't resolve
            for raw_name in candidates_for_llm:
                if raw_name not in resolution:
                    new_id = self._create_entity(raw_name.strip(), profile_id)
                    resolution[raw_name] = new_id
        elif candidates_for_llm:
            # No LLM available — create new entities for all candidates
            for raw_name in candidates_for_llm:
                new_id = self._create_entity(raw_name.strip(), profile_id)
                resolution[raw_name] = new_id

        return resolution

    def create_speaker_entities(
        self,
        speaker_a: str,
        speaker_b: str,
        profile_id: str,
    ) -> None:
        """Pre-create canonical entities for conversation speakers.

        Called at session start so that speaker names are immediately
        resolvable during fact extraction. Fixes B03 (speaker entities
        not available during first-turn encoding).
        """
        for speaker in (speaker_a, speaker_b):
            name = speaker.strip()
            if not name or name.lower() in PRONOUNS:
                continue
            existing = self._db.get_entity_by_name(name, profile_id)
            if existing is None:
                self._create_entity(name, profile_id, entity_type="person")

    def get_canonical_name(self, raw_name: str, profile_id: str) -> str:
        """Quick lookup: returns canonical name or original if not found."""
        name = raw_name.strip()
        if not name:
            return raw_name

        entity = self._db.get_entity_by_name(name, profile_id)
        if entity is not None:
            return entity.canonical_name

        entity_id = self._alias_lookup(name, profile_id)
        if entity_id is not None:
            rows = self._db.execute(
                "SELECT canonical_name FROM canonical_entities WHERE entity_id = ?",
                (entity_id,),
            )
            if rows:
                return str(dict(rows[0])["canonical_name"])

        return raw_name

    def merge_entities(
        self,
        entity_id_keep: str,
        entity_id_merge: str,
        profile_id: str,
    ) -> None:
        """Merge two entities: move all aliases and facts to keep, delete merge.

        Reassigns aliases, updates canonical_entities_json in atomic_facts,
        and removes the merged entity record.
        """
        # Move aliases from merge -> keep
        aliases = self._db.get_aliases_for_entity(entity_id_merge, profile_id)
        for alias in aliases:
            new_alias = EntityAlias(
                alias_id=_new_id(),
                entity_id=entity_id_keep,
                alias=alias.alias,
                confidence=alias.confidence,
                source=f"merge_from:{entity_id_merge}",
            )
            self._db.store_alias(new_alias, profile_id)

        # Also add the merged entity's canonical name as an alias of keep
        merged = self._db.get_entity_by_name("", "")  # placeholder
        rows = self._db.execute(
            "SELECT canonical_name FROM canonical_entities WHERE entity_id = ?",
            (entity_id_merge,),
        )
        if rows:
            merged_name = str(dict(rows[0])["canonical_name"])
            self._persist_alias(entity_id_keep, merged_name, 1.0, "merge", profile_id)

        # Update atomic_facts: replace entity_id_merge with entity_id_keep
        # in canonical_entities_json column
        fact_rows = self._db.execute(
            "SELECT fact_id, canonical_entities_json FROM atomic_facts "
            "WHERE profile_id = ? AND canonical_entities_json LIKE ?",
            (profile_id, f'%"{entity_id_merge}"%'),
        )
        for row in fact_rows:
            d = dict(row)
            try:
                entities = json.loads(d["canonical_entities_json"])
                updated = [
                    entity_id_keep if eid == entity_id_merge else eid
                    for eid in entities
                ]
                # Deduplicate while preserving order
                seen: set[str] = set()
                deduped = []
                for eid in updated:
                    if eid not in seen:
                        seen.add(eid)
                        deduped.append(eid)
                self._db.execute(
                    "UPDATE atomic_facts SET canonical_entities_json = ? "
                    "WHERE fact_id = ?",
                    (json.dumps(deduped), d["fact_id"]),
                )
            except (json.JSONDecodeError, TypeError):
                continue

        # Delete the merged entity (tenant-scoped)
        self._db.execute(
            "DELETE FROM entity_aliases WHERE entity_id = ? AND profile_id = ?",
            (entity_id_merge, profile_id),
        )
        self._db.execute(
            "DELETE FROM canonical_entities WHERE entity_id = ? AND profile_id = ?",
            (entity_id_merge, profile_id),
        )
        logger.info(
            "Merged entity %s into %s (profile=%s)",
            entity_id_merge, entity_id_keep, profile_id,
        )

    # -- Internal: lookups --------------------------------------------------

    def _alias_lookup(self, name: str, profile_id: str) -> str | None:
        """Look up entity_id via alias table (case-insensitive)."""
        rows = self._db.execute(
            "SELECT ea.entity_id FROM entity_aliases ea "
            "JOIN canonical_entities ce ON ce.entity_id = ea.entity_id "
            "WHERE LOWER(ea.alias) = LOWER(?) AND ce.profile_id = ?",
            (name, profile_id),
        )
        if rows:
            return str(dict(rows[0])["entity_id"])
        return None

    def _fuzzy_match(
        self, name: str, profile_id: str,
    ) -> tuple[str | None, float]:
        """Scan all canonical names + aliases for best Jaro-Winkler match.

        Returns (entity_id, score) or (None, 0.0).
        """
        best_id: str | None = None
        best_score: float = 0.0
        name_lower = name.lower()

        # Check canonical names
        rows = self._db.execute(
            "SELECT entity_id, canonical_name FROM canonical_entities "
            "WHERE profile_id = ?",
            (profile_id,),
        )
        for row in rows:
            d = dict(row)
            score = jaro_winkler(name_lower, d["canonical_name"].lower())
            if score > best_score:
                best_score = score
                best_id = d["entity_id"]

        # Check aliases
        alias_rows = self._db.execute(
            "SELECT ea.entity_id, ea.alias FROM entity_aliases ea "
            "JOIN canonical_entities ce ON ce.entity_id = ea.entity_id "
            "WHERE ce.profile_id = ?",
            (profile_id,),
        )
        for row in alias_rows:
            d = dict(row)
            score = jaro_winkler(name_lower, d["alias"].lower())
            if score > best_score:
                best_score = score
                best_id = d["entity_id"]

        return (best_id, best_score)

    # -- Internal: persistence ----------------------------------------------

    def _create_entity(
        self,
        name: str,
        profile_id: str,
        entity_type: str | None = None,
    ) -> str:
        """Create a new canonical entity + self-alias. Returns entity_id."""
        etype = entity_type or _guess_entity_type(name)
        now = _now()
        entity = CanonicalEntity(
            entity_id=_new_id(),
            profile_id=profile_id,
            canonical_name=name,
            entity_type=etype,
            first_seen=now,
            last_seen=now,
            fact_count=0,
        )
        self._db.store_entity(entity)

        # Store name as its own alias for uniform lookup
        self._persist_alias(entity.entity_id, name, 1.0, "canonical", profile_id)

        logger.debug(
            "Created entity '%s' [%s] (type=%s, profile=%s)",
            name, entity.entity_id, etype, profile_id,
        )
        return entity.entity_id

    def _persist_alias(
        self,
        entity_id: str,
        alias_text: str,
        confidence: float,
        source: str,
        profile_id: str,
    ) -> None:
        """Store an alias under a profile, skipping duplicates within it."""
        # Check if alias already exists for this entity IN THIS PROFILE. The
        # dedup must be profile-scoped or one profile's alias suppresses
        # another profile's identical alias for a same-id entity.
        existing = self._db.execute(
            "SELECT alias_id FROM entity_aliases "
            "WHERE entity_id = ? AND profile_id = ? AND LOWER(alias) = LOWER(?)",
            (entity_id, profile_id, alias_text),
        )
        if existing:
            return
        alias = EntityAlias(
            alias_id=_new_id(),
            entity_id=entity_id,
            alias=alias_text,
            confidence=confidence,
            source=source,
        )
        self._db.store_alias(alias, profile_id)

    def _touch_last_seen(self, entity_id: str, profile_id: str = "default") -> None:
        """Update last_seen timestamp on a canonical entity scoped to profile.

        L-01 fix: the original query had no profile_id guard. Since entity_ids are
        UUIDs today the blast radius is theoretical, but the guard is required for
        defense-in-depth against future import/sharing features that could introduce
        UUID collisions across profiles.
        """
        self._db.execute(
            "UPDATE canonical_entities SET last_seen = ? "
            "WHERE entity_id = ? AND profile_id = ?",
            (_now(), entity_id, profile_id),
        )

    # -- Internal: LLM disambiguation (Mode B/C) ---------------------------

    def _llm_disambiguate(
        self,
        raw_names: list[str],
        profile_id: str,
    ) -> dict[str, str]:
        """Ask LLM whether fuzzy candidates match existing entities."""
        if not self._llm or not raw_names:
            return {}

        # Gather known entity names for context
        rows = self._db.execute(
            "SELECT entity_id, canonical_name FROM canonical_entities "
            "WHERE profile_id = ? LIMIT 50",
            (profile_id,),
        )
        known = {
            str(dict(r)["canonical_name"]): str(dict(r)["entity_id"])
            for r in rows
        }

        prompt = (
            "Entity resolution task. For each mention, decide if it refers to "
            "one of the known entities or is a new entity.\n\n"
            f"Mentions to resolve: {raw_names}\n"
            f"Known entities: {list(known.keys())}\n\n"
            "Respond with ONLY a JSON object mapping each mention to a "
            "known entity name if they match, or to itself if it is new.\n"
            'Example: {"Ms. Smith": "Alice Smith", "Bob": "Bob"}'
        )

        try:
            response = self._llm.generate(
                prompt=prompt,
                system="You are a precise entity resolution system.",
                max_tokens=256,
                temperature=0.0,
            )
            match = re.search(r"\{[^}]*\}", response)
            if not match:
                return {}

            result = json.loads(match.group())
            if not isinstance(result, dict):
                return {}

            resolved: dict[str, str] = {}
            for mention, canonical_name in result.items():
                mention_str = str(mention)
                name_str = str(canonical_name)
                if name_str in known:
                    entity_id = known[name_str]
                    resolved[mention_str] = entity_id
                    self._persist_alias(
                        entity_id, mention_str, 0.9, "llm", profile_id,
                    )
                    self._touch_last_seen(entity_id, profile_id)
                # If LLM says it's itself, leave for caller to create
            return resolved

        except (json.JSONDecodeError, TypeError, Exception) as exc:
            logger.warning("LLM entity disambiguation failed: %s", exc)
            return {}
