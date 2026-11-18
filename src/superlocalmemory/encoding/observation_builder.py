# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""SuperLocalMemory V3 — Observation Builder (Entity Profiles).

Builds and updates accumulated knowledge profiles per entity.
When a new fact mentions an entity, the entity's profile is updated.

V1 had this module but NEVER CALLED it. Now wired into the encoding pipeline.

Part of Qualixar | Author: Varun Pratap Bhardwaj
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime

from superlocalmemory.storage.models import AtomicFact, EntityProfile

logger = logging.getLogger(__name__)


class ObservationBuilder:
    """Build and maintain entity knowledge profiles.

    Each canonical entity gets a running profile that accumulates
    all facts known about it. Used for:
    - Entity-centric retrieval (return profile as context)
    - Consolidation (detect when new info conflicts with profile)
    - Answer generation (entity profiles provide rich context)
    """

    def __init__(self, db) -> None:
        self._db = db

    def update_profile(
        self,
        entity_id: str,
        new_fact: AtomicFact,
        profile_id: str,
    ) -> EntityProfile:
        """Update (or create) entity profile with new fact.

        Appends fact to profile's fact list and regenerates summary.
        """
        existing = self._get_profile(entity_id, profile_id)

        if existing is not None:
            fact_ids = existing.fact_ids
            if new_fact.fact_id not in fact_ids:
                fact_ids = [*fact_ids, new_fact.fact_id]
            summary = self._build_summary(entity_id, fact_ids, profile_id)
            updated = EntityProfile(
                profile_entry_id=existing.profile_entry_id,
                entity_id=entity_id,
                profile_id=profile_id,
                knowledge_summary=summary,
                fact_ids=fact_ids,
                last_updated=datetime.now(UTC).isoformat(),
            )
        else:
            summary = self._build_summary(entity_id, [new_fact.fact_id], profile_id)
            updated = EntityProfile(
                entity_id=entity_id,
                profile_id=profile_id,
                knowledge_summary=summary,
                fact_ids=[new_fact.fact_id],
                last_updated=datetime.now(UTC).isoformat(),
            )

        self._save_profile(updated)
        return updated

    def get_profile(self, entity_id: str, profile_id: str) -> EntityProfile | None:
        """Get the current knowledge profile for an entity."""
        return self._get_profile(entity_id, profile_id)

    def build_all_profiles(self, profile_id: str) -> list[EntityProfile]:
        """Rebuild all entity profiles from scratch. Use after migration."""
        rows = self._db.execute(
            "SELECT DISTINCT entity_id FROM canonical_entities WHERE profile_id = ?",
            (profile_id,),
        )
        profiles = []
        for row in rows:
            eid = dict(row)["entity_id"]
            facts = self._db.get_facts_by_entity(eid, profile_id)
            if facts:
                fact_ids = [f.fact_id for f in facts]
                summary = self._build_summary(eid, fact_ids, profile_id)
                ep = EntityProfile(
                    entity_id=eid,
                    profile_id=profile_id,
                    knowledge_summary=summary,
                    fact_ids=fact_ids,
                    last_updated=datetime.now(UTC).isoformat(),
                )
                self._save_profile(ep)
                profiles.append(ep)
        return profiles

    # -- Internal ----------------------------------------------------------

    def _get_profile(self, entity_id: str, profile_id: str) -> EntityProfile | None:
        """Load entity profile from DB."""
        rows = self._db.execute(
            "SELECT * FROM entity_profiles WHERE entity_id = ? AND profile_id = ?",
            (entity_id, profile_id),
        )
        if not rows:
            return None
        d = dict(rows[0])
        return EntityProfile(
            profile_entry_id=d["profile_entry_id"],
            entity_id=d["entity_id"],
            profile_id=d["profile_id"],
            knowledge_summary=d.get("knowledge_summary", ""),
            fact_ids=json.loads(d.get("fact_ids_json", "[]")),
            last_updated=d.get("last_updated", ""),
        )

    def _save_profile(self, profile: EntityProfile) -> None:
        """Upsert entity profile to DB (the profile it replaces is kept as a version, WS-6)."""
        from superlocalmemory.compiler import versions

        with versions.atomic(self._db):
            versions.keep_previous(
                self._db, "entity_profile",
                where="profile_entry_id = ? OR (entity_id = ? AND profile_id = ?)",
                params=(profile.profile_entry_id, profile.entity_id, profile.profile_id))
            self._db.execute(
                """INSERT OR REPLACE INTO entity_profiles
                   (profile_entry_id, entity_id, profile_id,
                    knowledge_summary, fact_ids_json, last_updated)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    profile.profile_entry_id,
                    profile.entity_id,
                    profile.profile_id,
                    profile.knowledge_summary,
                    json.dumps(profile.fact_ids),
                    profile.last_updated,
                ),
            )

    # V3.4.40 (2026-05-09): bounded summary to prevent unbounded growth.
    # Pre-V3.4.40 builds concatenated full fact content. Hubs accumulated
    # 60+ KB summaries that polluted recall and bloated entity_profiles.
    _SUMMARY_MAX_FACTS = 10           # Last N facts (was 20)
    _SUMMARY_MAX_CHARS_PER_FACT = 200  # Truncate each fact (NEW)
    _SUMMARY_MAX_TOTAL_CHARS = 2048    # Hard cap on full summary (NEW)

    def _build_summary(
        self, entity_id: str, fact_ids: list[str], profile_id: str
    ) -> str:
        """Build a knowledge summary from all facts about an entity.

        V3.4.40: bounded — last 10 facts, 200 chars each, 2048 total.
        Older builds produced 60KB summaries on hub entities.
        Mode B/C could use LLM rollup for higher-quality compression.
        """
        facts = []
        for fid in fact_ids[-self._SUMMARY_MAX_FACTS:]:
            rows = self._db.execute(
                "SELECT content FROM atomic_facts WHERE fact_id = ? AND profile_id = ?",
                (fid, profile_id),
            )
            if rows:
                content = dict(rows[0])["content"]
                # Truncate per-fact to bound the join below
                if len(content) > self._SUMMARY_MAX_CHARS_PER_FACT:
                    content = content[: self._SUMMARY_MAX_CHARS_PER_FACT - 1] + "…"
                facts.append(content)

        if not facts:
            return ""
        joined = " | ".join(facts)
        # Hard cap as final safety net
        if len(joined) > self._SUMMARY_MAX_TOTAL_CHARS:
            joined = joined[: self._SUMMARY_MAX_TOTAL_CHARS - 1] + "…"
        return joined
