# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""The derived rows an import's facts leave behind, removed in batches (part of exact undo).

Deleting a fact removes its keyword, vector, graph and scene copies (``core/mutations``). What
the save path and the background passes also write about a fact lives in tables no foreign key
reaches: the consolidation log, the entity records the fact named and the profile summaries
built from them, the summaries that quote it, the derivation lineage of its save, and the trust
score of the actor that wrote it. ``core/erasure_scrub`` removes these for one fact at a time
(scanning the event log for each), which is right for a single erasure and far too slow for a
run of thousands, so this module does the same work once per batch of facts.

Everything here is keyed by ids the run owns. It never reads or writes memory text, and it
rebuilds an entity's summary by the same rule that wrote it (``ObservationBuilder``) from the
facts that are left.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from superlocalmemory.storage.json_columns import load_json_column

logger = logging.getLogger(__name__)
SENTINEL_TYPE = "unresolved"


def _marks(n: int) -> str:
    return ",".join("?" * n)


def _quiet(db: Any, sql: str, params: tuple | list = ()) -> list:
    """Run one statement; a store without the table (an older schema) has nothing to remove."""
    try:
        return list(db.execute(sql, tuple(params)))
    except Exception as exc:  # noqa: BLE001
        if "no such table" in str(exc).lower() or "no such column" in str(exc).lower():
            return []
        raise


def entities_naming(db: Any, profile_id: str, fact_ids: list[str]) -> set[str]:
    """The entities these facts name. Call while the facts still exist."""
    from superlocalmemory.core import erasure_scrub

    found: set[str] = set()
    for fact_id in fact_ids:
        found.update(erasure_scrub.entities_of(db, profile_id, fact_id))
    return found


def after_facts_deleted(db: Any, profile_id: str, fact_ids: list[str], entity_ids: set[str]) -> None:
    """The batch's derived rows. Call after the facts are gone."""
    if not fact_ids:
        return
    for i in range(0, len(fact_ids), 400):
        part = fact_ids[i:i + 400]
        _quiet(db, f"DELETE FROM consolidation_log WHERE profile_id = ? "
                   f"AND new_fact_id IN ({_marks(len(part))})", [profile_id, *part])
    _rebuild_entity_profiles(db, profile_id, set(fact_ids), entity_ids)
    _drop_unnamed(db, profile_id, entity_ids)


def _rebuild_entity_profiles(db: Any, profile_id: str, gone: set[str], entity_ids: set[str]) -> None:
    from superlocalmemory.encoding.observation_builder import ObservationBuilder

    if not entity_ids:
        return
    builder = ObservationBuilder(db)
    ids = sorted(entity_ids)
    for i in range(0, len(ids), 400):
        part = ids[i:i + 400]
        for row in _quiet(db, f"SELECT profile_entry_id, entity_id, fact_ids_json FROM "
                              f"entity_profiles WHERE profile_id = ? AND entity_id IN "
                              f"({_marks(len(part))})", [profile_id, *part]):
            try:
                listed = load_json_column(row["fact_ids_json"], [], column="entity_profiles.fact_ids_json")
            except ValueError:
                continue
            remaining = [f for f in listed if f not in gone]
            if len(remaining) == len(listed):
                continue
            summary = builder._build_summary(row["entity_id"], remaining, profile_id)
            db.execute("UPDATE entity_profiles SET knowledge_summary = ?, fact_ids_json = ? "
                       "WHERE profile_entry_id = ?",
                       (summary, json.dumps(remaining), row["profile_entry_id"]))


def _named(db: Any, profile_id: str, entity_id: str) -> bool:
    for sql, params in (
        ("SELECT 1 FROM fact_entity_associations WHERE entity_id = ? LIMIT 1", (entity_id,)),
        ("SELECT 1 FROM temporal_events WHERE entity_id = ? LIMIT 1", (entity_id,)),
        ("SELECT 1 FROM atomic_facts WHERE profile_id = ? AND canonical_entities_json LIKE ? "
         "LIMIT 1", (profile_id, f'%"{entity_id}"%')),
    ):
        if _quiet(db, sql, params):
            return True
    return False


def _drop_unnamed(db: Any, profile_id: str, entity_ids: set[str]) -> None:
    for entity_id in sorted(entity_ids):
        if _named(db, profile_id, entity_id):
            continue
        for table in ("entity_communities", "consolidated_summaries"):
            _quiet(db, f"DELETE FROM {table} WHERE entity_id = ?", (entity_id,))  # noqa: S608
        _quiet(db, "DELETE FROM canonical_entities WHERE entity_id = ? AND profile_id = ?",
               (entity_id, profile_id))


def at_the_end(db: Any, profile_id: str, run_id: str, fact_ids: set[str], started_at: str,
               operation_ids: list[str]) -> None:
    """What only a look at the whole run can decide, once every fact is gone."""
    _drop_quoting_summaries(db, profile_id, fact_ids)
    for i in range(0, len(operation_ids), 400):
        part = operation_ids[i:i + 400]
        _quiet(db, f"DELETE FROM derivation_lineage WHERE operation_id IN ({_marks(len(part))})",
               part)
    # The run wrote as its own actor, so the score the save path kept for that actor is the run's.
    _quiet(db, "DELETE FROM trust_scores WHERE profile_id = ? AND target_type = 'agent' "
               "AND target_id = ?", (profile_id, f"import:{run_id}"))
    _drop_new_sentinel(db, profile_id, started_at)


def _drop_quoting_summaries(db: Any, profile_id: str, fact_ids: set[str]) -> None:
    """Summaries built from the run's facts are regenerated by their owners from what is left."""
    if not fact_ids:
        return
    for table, key, column in (("core_memory_blocks", "block_id", "source_fact_ids"),
                               ("community_summaries", "summary_id", "fact_ids_json"),
                               ("consolidated_summaries", "summary_id", "source_fact_ids")):
        doomed = []
        for row in _quiet(db, f"SELECT {key}, {column} FROM {table} WHERE profile_id = ?",  # noqa: S608
                          (profile_id,)):
            try:
                listed = json.loads(row[column] or "[]")
            except ValueError:
                continue
            if isinstance(listed, list) and fact_ids.intersection(map(str, listed)):
                doomed.append(row[key])
        for i in range(0, len(doomed), 400):
            part = doomed[i:i + 400]
            _quiet(db, f"DELETE FROM {table} WHERE {key} IN ({_marks(len(part))})",  # noqa: S608
                   part)


def _drop_new_sentinel(db: Any, profile_id: str, started_at: str) -> None:
    """The "no entity yet" placeholder the first enrichment creates, if this run created it."""
    for row in _quiet(db, "SELECT entity_id FROM canonical_entities WHERE profile_id = ? "
                          "AND entity_type = ? AND datetime(first_seen) >= datetime(?)",
                      (profile_id, SENTINEL_TYPE, started_at)):
        if not _named(db, profile_id, row["entity_id"]):
            _quiet(db, "DELETE FROM canonical_entities WHERE entity_id = ?", (row["entity_id"],))


__all__ = ["after_facts_deleted", "at_the_end", "entities_naming"]
