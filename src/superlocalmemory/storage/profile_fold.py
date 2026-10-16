# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Fold a deleted profile's memories into another profile (memory.db side).

WHY
---
Deleting a dashboard profile promises to move its memories to 'default'. It
moved two tables and then deleted the profile row with foreign keys on, so
every table tied to ``profiles`` by ON DELETE CASCADE dropped the moved
memories' keyword tokens, vector rows, entities, edges, temporal rows and
scenes on the spot, and every table without that key (the correction ledger,
entity links, the vectors themselves) stayed filed under a profile that no
longer existed. The text survived; the ways to find, link and correct it did
not.

HOW
---
:func:`fold_profile` runs inside the caller's single memory.db transaction,
before the profile row is deleted. Every table with a ``profile_id`` column is
discovered from the live schema (the GDPR erasure's rule), and each one has a
recorded decision below. A table with no decision stops the fold: a new table
must be classified, never silently left behind (tests/test_server/
test_profile_delete_moves_everything.py checks the table is complete).

DECISIONS
---------
MOVE    the row belongs to a moved memory; its key is a global id (fact,
        memory, entity, edge, case, operation...) so it cannot collide.
MERGE   move, but the table has a unique key that 'default' may already hold
        for the same thing; default's row wins and the duplicate is dropped.
REKEY   move; a caller-supplied idempotency key that 'default' already uses is
        suffixed with ``:folded:<row id>`` so neither row is lost.
VEC     a vector table partitioned by profile: sqlite-vec cannot update a
        partition key, so each vector is deleted and re-inserted, same rowid.
DELETE  per-profile configuration, coordination or derived state that would
        change default's behaviour if inherited (learned patterns, compiled
        blocks, community summaries, mesh state, backup destinations, roles);
        it is rebuilt from the moved memories where it is derived.
KEEP    deliberately still names the deleted profile: immutable receipts and
        audit records of what happened under it, and the ids-only change feed
        that tells cached indexes to drop it.

DECISION TABLE (the code's DECISIONS dict is authoritative; the reason for each
table is recorded there)

MOVE    action_outcomes, association_edges, atomic_facts, bm25_tokens,
        canonical_entities, ccq_audit_log, ccq_consolidated_blocks,
        completion_manifests, consolidation_log, correction_cases_overtaken,
        correction_events, data_repair_undo, dead_letter_operations, derivation_lineage,
        embedding_metadata, embedding_quantization_metadata, enrichment_changes,
        entity_aliases,
        fact_access_log, fact_consolidations, fact_context, fact_hub_baseline,
        fact_importance, fact_outcome_score, fact_retention, fact_temporal_validity,
        feedback_records, memories, memory_archive,
        memory_kind_history, memory_merge_log, memory_scenes, pinned_facts,
        polar_embeddings, projection_obligations, projection_outbox,
        projection_tombstones, provenance, reembed_next_map, reembed_prev_map,
        scene_fact_members, temporal_events, tool_events, vector_row_map
MERGE   consolidated_summaries, entity_profiles, fact_entity_associations,
        graph_edges, ingestion_log, trust_scores
REKEY   correction_cases, ingestion_operations
VEC     fact_embeddings, reembed_next_vec, reembed_prev_vec, reembed_trash_vec
DELETE  activation_cache, backup_destinations, behavioral_assertions,
        behavioral_patterns, community_summaries, compliance_audit,
        core_memory_blocks, cross_platform_sync_log, entity_communities,
        entity_link_repair_undo, graph_generation, memory_kind_runs,
        mesh_events, mesh_locks, mesh_messages, mesh_peers, mesh_state, pending_outcomes,
        persona_summary, rbac_memberships, soft_prompt_templates
KEEP    erasure_receipts, fact_search_changes, profiles, write_commits

Canonical entities first MERGE by name (case-insensitive, the resolver's own
rule): a moved entity that default already knows is folded into default's, and
every reference to it -- links, aliases, temporal rows, edges, summaries, the
facts' entity lists, scenes -- is repointed before the rows move.
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any

MOVE, MERGE, REKEY, VEC, DELETE, KEEP = "move", "merge", "rekey", "vec", "delete", "keep"

#: table -> (decision, why). Every discovered profile-scoped table must be here.
DECISIONS: dict[str, tuple[str, str]] = {
    # the memories themselves
    "memories": (MOVE, "the memory"),
    "atomic_facts": (MOVE, "its facts"),
    "memory_archive": (MOVE, "archived facts of the memories"),
    "pinned_facts": (MOVE, "pins on its facts"),
    "data_repair_undo": (MOVE, "undo record of an automatic data repair; it names the "
                               "moved facts by id, so it still applies after the move"),
    "dead_letter_operations": (MOVE, "writes that failed and can still be retried"),
    # search indexes and vectors
    "bm25_tokens": (MOVE, "keyword index"),
    "fact_embeddings": (VEC, "vectors (partition per profile)"),
    "reembed_next_vec": (VEC, "vectors staged for a model switch"),
    "reembed_prev_vec": (VEC, "vectors of the previous model, kept for rollback"),
    "reembed_trash_vec": (VEC, "a replaced space awaiting drop"),
    "reembed_next_map": (MOVE, "staged vector map"),
    "reembed_prev_map": (MOVE, "previous vector map"),
    "embedding_metadata": (MOVE, "vector row map"),
    "vector_row_map": (MOVE, "vector row map"),
    "embedding_quantization_metadata": (MOVE, "vector compression record"),
    "polar_embeddings": (MOVE, "compressed vectors"),
    "fact_context": (MOVE, "contextual description of a fact"),
    # entities and graph
    "canonical_entities": (MOVE, "entities (same-name ones merged first)"),
    "entity_aliases": (MOVE, "aliases of the entities"),
    "fact_entity_associations": (MERGE, "fact-entity links"),
    "entity_profiles": (MERGE, "per-entity knowledge; default's wins on a clash"),
    "consolidated_summaries": (MERGE, "per-entity summaries; default's wins on a clash"),
    "graph_edges": (MERGE, "graph edges; an edge default already has is not doubled"),
    "association_edges": (MOVE, "fact associations"),
    "temporal_events": (MOVE, "temporal rows"),
    "memory_scenes": (MOVE, "scenes (their member rows follow by trigger)"),
    "scene_fact_members": (MOVE, "scene members"),
    "fact_importance": (MOVE, "per-fact importance"),
    "fact_retention": (MOVE, "per-fact retention"),
    "enrichment_changes": (MOVE, "undo ledger of automatic metadata improvements; it "
                                 "follows the moved facts so a run stays undoable"),
    "fact_temporal_validity": (MOVE, "bitemporal validity"),
    "fact_outcome_score": (MOVE, "per-fact outcome score"),
    "fact_hub_baseline": (MOVE, "per-fact hub baseline"),
    "fact_access_log": (MOVE, "access history of the facts"),
    "feedback_records": (MOVE, "feedback on the facts"),
    "provenance": (MOVE, "where each fact came from"),
    "derivation_lineage": (MOVE, "how each fact was derived"),
    "memory_kind_history": (MOVE, "kind changes of the facts"),
    "memory_merge_log": (MOVE, "merges of the facts"),
    "consolidation_log": (MOVE, "consolidation history"),
    "fact_consolidations": (MOVE, "consolidated facts"),
    "ccq_consolidated_blocks": (MOVE, "consolidated blocks of the facts"),
    "ccq_audit_log": (MOVE, "audit of those blocks"),
    "action_outcomes": (MOVE, "outcomes recorded against the facts"),
    "tool_events": (MOVE, "tool activity history"),
    # corrections (M042: facts referenced ON DELETE RESTRICT, no text)
    "correction_cases": (REKEY, "correction cases of the facts"),
    "correction_events": (MOVE, "the cases' event ledger"),
    "correction_cases_overtaken": (MOVE, "closed cases, restorable"),
    # ingestion and projection bookkeeping
    "ingestion_operations": (REKEY, "the operations that produced the facts"),
    "ingestion_log": (MERGE, "content dedup; default's record wins"),
    "completion_manifests": (MOVE, "completion of those operations"),
    "projection_obligations": (MOVE, "pending projection work"),
    "projection_outbox": (MOVE, "pending projection work"),
    "projection_tombstones": (MOVE, "erased facts must stay out of default's projections too"),
    "trust_scores": (MERGE, "trust in sources; default's own judgement wins on a clash"),
    # per-profile state: deleted
    "activation_cache": (DELETE, "derived cache"),
    "graph_generation": (DELETE, "the deleted profile's graph version counter"),
    "entity_communities": (DELETE, "derived; recomputed for default"),
    "community_summaries": (DELETE, "derived; recomputed for default"),
    "persona_summary": (DELETE, "derived per profile"),
    "core_memory_blocks": (DELETE, "compiled per profile; recompiled"),
    "soft_prompt_templates": (DELETE, "generated per profile"),
    "behavioral_patterns": (DELETE, "learned behaviour of the deleted profile"),
    "behavioral_assertions": (DELETE, "learned behaviour of the deleted profile"),
    "memory_kind_runs": (DELETE, "classifier run state of the deleted profile"),
    "pending_outcomes": (DELETE, "short-lived recall sessions of the deleted profile"),
    "backup_destinations": (DELETE, "configuration; default must not back up there"),
    "cross_platform_sync_log": (DELETE, "sync state of the deleted profile"),
    "rbac_memberships": (DELETE, "roles; a later profile of the same name gets none"),
    "compliance_audit": (DELETE, "in-store audit rows (the audit chain keeps the record)"),
    "mesh_events": (DELETE, "live coordination"),
    "mesh_locks": (DELETE, "live coordination"),
    "mesh_messages": (DELETE, "live coordination"),
    "mesh_peers": (DELETE, "live coordination"),
    "mesh_state": (DELETE, "live coordination"),
    "entity_link_repair_undo": (DELETE, "undo record of an entity link repair; it names "
                                        "the deleted profile's rows and cannot apply after"),
    # deliberately kept
    "write_commits": (KEEP, "immutable write receipts (a trigger forbids changes)"),
    "erasure_receipts": (KEEP, "Art. 17 receipts of erasures made in that profile"),
    "fact_search_changes": (KEEP, "ids-only change feed that drops the profile from cached indexes"),
    "profiles": (KEEP, "deleted by the caller after the fold"),
}

#: REKEY tables: (unique columns besides profile_id, key column, row id column)
_REKEY = {
    "correction_cases": (("idempotency_key",), "idempotency_key", "case_id"),
    "ingestion_operations": (("source_type", "idempotency_key"), "idempotency_key",
                             "operation_id"),
}
#: MERGE tables whose clash no UNIQUE index enforces: the natural key the code
#: reads them by (one row per profile and key).
_NATURAL = {"trust_scores": ("target_type", "target_id"),
            "graph_edges": ("source_id", "target_id", "edge_type")}
#: Moved first: the scene trigger joins atomic_facts under the new profile.
_FIRST = ("memories", "atomic_facts")


class ProfileFoldError(RuntimeError):
    """The fold cannot run safely; nothing has been changed."""


def profile_scoped_tables(conn: Any) -> list[str]:
    names = [r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name").fetchall()]
    out = []
    for name in names:
        if name.startswith("sqlite_"):
            continue
        try:
            cols = {r[1] for r in conn.execute(f'PRAGMA table_info("{name}")')}
        except sqlite3.OperationalError:
            continue  # a virtual table whose module is not loaded: checked by _check
        if "profile_id" in cols:
            out.append(name)
    return out


def _check(conn: Any, tables: list[str]) -> None:
    unknown = [t for t in tables if t not in DECISIONS]
    if unknown:
        raise ProfileFoldError(f"no recorded decision for {', '.join(unknown)}; "
                               "the profile was not deleted")
    vec = [r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE sql LIKE 'CREATE VIRTUAL TABLE%vec0%'")]
    missing = [t for t in vec if t not in tables]
    if missing:
        raise ProfileFoldError("the vector extension is not loaded, so the vectors in "
                               f"{', '.join(missing)} cannot be moved; the profile was not deleted")
    if "embedding_reindex_jobs" in {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}:
        from superlocalmemory.storage.embedding_spaces import ACTIVE_STATES

        marks = ",".join("?" for _ in ACTIVE_STATES)
        if conn.execute(f"SELECT 1 FROM embedding_reindex_jobs WHERE state IN ({marks}) "
                        "LIMIT 1", ACTIVE_STATES).fetchone():
            raise ProfileFoldError("an embedding re-index is running; delete the profile "
                                   "when it has finished")


def _repoint_json(conn: Any, table: str, key: str, column: str, profile: str,
                  old: str, new: str) -> None:
    rows = conn.execute(f'SELECT "{key}", "{column}" FROM "{table}" WHERE profile_id = ? '
                        f'AND "{column}" LIKE ?', (profile, f'%"{old}"%')).fetchall()
    for row_key, raw in rows:
        try:
            ids = json.loads(raw)
        except (TypeError, ValueError):
            continue
        if not isinstance(ids, list):
            continue
        fixed = list(dict.fromkeys(new if i == old else i for i in ids))
        conn.execute(f'UPDATE "{table}" SET "{column}" = ? WHERE "{key}" = ?',
                     (json.dumps(fixed), row_key))


def _merge_entities(conn: Any, tables: set[str], source: str, target: str) -> list[str]:
    """Fold each moved entity default already knows by name into default's."""
    if "canonical_entities" not in tables:
        return []
    pairs = conn.execute(
        "SELECT x.entity_id, MIN(d.entity_id) FROM canonical_entities x "
        "JOIN canonical_entities d ON d.profile_id = ? "
        "AND LOWER(d.canonical_name) = LOWER(x.canonical_name) "
        "WHERE x.profile_id = ? GROUP BY x.entity_id", (target, source)).fetchall()
    simple = (("entity_aliases", "entity_id"), ("temporal_events", "entity_id"),
              ("graph_edges", "source_id"), ("graph_edges", "target_id"))
    unique = ("fact_entity_associations", "entity_profiles", "consolidated_summaries")
    for old, new in pairs:
        for table, col in simple:
            if table in tables:
                conn.execute(f'UPDATE "{table}" SET "{col}" = ? WHERE "{col}" = ?', (new, old))
        for table in unique:
            if table in tables:
                conn.execute(f'UPDATE OR IGNORE "{table}" SET entity_id = ? WHERE entity_id = ?',
                             (new, old))
                conn.execute(f'DELETE FROM "{table}" WHERE entity_id = ?', (old,))
        _repoint_json(conn, "atomic_facts", "fact_id", "canonical_entities_json", source, old, new)
        if "memory_scenes" in tables:
            _repoint_json(conn, "memory_scenes", "scene_id", "entity_ids_json", source, old, new)
        conn.execute(
            "UPDATE canonical_entities SET "
            "fact_count = COALESCE(fact_count, 0) + COALESCE((SELECT fact_count FROM "
            "canonical_entities WHERE entity_id = ?), 0), "
            "first_seen = MIN(first_seen, COALESCE((SELECT first_seen FROM canonical_entities "
            "WHERE entity_id = ?), first_seen)), "
            "last_seen = MAX(last_seen, COALESCE((SELECT last_seen FROM canonical_entities "
            "WHERE entity_id = ?), last_seen)) WHERE entity_id = ?", (old, old, old, new))
        conn.execute("DELETE FROM canonical_entities WHERE entity_id = ?", (old,))
    if "enrichment_changes" in tables:  # keep moved improvement runs exactly undoable
        from superlocalmemory.storage.enrichment_ledger import remap_entity_ids

        remap_entity_ids(conn, source, {old: new for old, new in pairs})
    return [old for old, _new in pairs]


def _rekey(conn: Any, table: str, source: str, target: str) -> None:
    unique_cols, key, row_id = _REKEY[table]
    match = " AND ".join(f'd."{c}" = "{table}"."{c}"' for c in unique_cols)
    conn.execute(
        f'UPDATE "{table}" SET "{key}" = "{key}" || \':folded:\' || "{row_id}" '
        f'WHERE profile_id = ? AND EXISTS (SELECT 1 FROM "{table}" d '
        f'WHERE d.profile_id = ? AND {match})', (source, target))


def _move_vectors(conn: Any, table: str, source: str, target: str) -> int:
    rows = conn.execute(f'SELECT rowid, embedding FROM "{table}" WHERE profile_id = ?',
                        (source,)).fetchall()
    if rows:  # one bulk delete, then the same rowids back (20k vectors: 6.7 s, was 12.9 s)
        conn.execute(f'DELETE FROM "{table}" WHERE profile_id = ?', (source,))
        conn.executemany(f'INSERT INTO "{table}" (rowid, profile_id, embedding) VALUES (?, ?, ?)',
                         [(rowid, target, embedding) for rowid, embedding in rows])
    return len(rows)


def _apply(conn: Any, table: str, action: str, source: str, target: str) -> int:
    if action == VEC:
        return _move_vectors(conn, table, source, target)
    if action == DELETE:
        return conn.execute(f'DELETE FROM "{table}" WHERE profile_id = ?', (source,)).rowcount
    if action == REKEY:
        _rekey(conn, table, source, target)
    if action == MERGE and table in _NATURAL:
        match = " AND ".join(f'd."{c}" IS "{table}"."{c}"' for c in _NATURAL[table])
        conn.execute(f'DELETE FROM "{table}" WHERE profile_id = ? AND EXISTS (SELECT 1 FROM '
                     f'"{table}" d WHERE d.profile_id = ? AND {match})', (source, target))
    verb = "UPDATE OR IGNORE" if action == MERGE else "UPDATE"
    moved = conn.execute(f'{verb} "{table}" SET profile_id = ? WHERE profile_id = ?',
                         (target, source)).rowcount
    if action == MERGE:  # what is left collided with a row default already has
        conn.execute(f'DELETE FROM "{table}" WHERE profile_id = ?', (source,))
    return moved


def _requeue_projection(conn: Any, tables: set[str], target: str) -> None:
    """The graph/vector projection re-reads each moved fact and files it under target."""
    if "projection_outbox" not in tables or "atomic_facts" not in tables:
        return
    from superlocalmemory.storage.projection_outbox import OP_UPSERT, _now

    conn.execute(
        "INSERT INTO projection_outbox (fact_id, profile_id, op, revision, enqueued_at, "
        "attempts, last_error) SELECT fact_id, ?, ?, 1, ?, 0, NULL FROM temp._fold_facts "
        "WHERE true ON CONFLICT(fact_id) DO UPDATE SET op = excluded.op, "
        "profile_id = excluded.profile_id, revision = projection_outbox.revision + 1, "
        "enqueued_at = excluded.enqueued_at, attempts = 0, last_error = NULL",
        (target, OP_UPSERT, _now()))


def fold_profile(conn: Any, source: str, target: str = "default") -> dict[str, Any]:
    """Move ``source``'s memories to ``target`` in the caller's transaction.

    Returns per-table counts plus ``merged_entities`` (ids folded away, for the
    caller to drop from an external graph projection after COMMIT). Raises
    :class:`ProfileFoldError` before changing anything when it cannot run safely.
    """
    if not source or source == target:
        raise ProfileFoldError("a profile cannot be folded into itself")
    tables = profile_scoped_tables(conn)
    _check(conn, tables)
    present = set(tables)
    conn.execute("INSERT OR IGNORE INTO profiles (profile_id, name) VALUES (?, ?)",
                 (target, target))
    conn.execute("DROP TABLE IF EXISTS temp._fold_facts")
    if "atomic_facts" in present:  # a bare store (profiles only) has no memories to move
        conn.execute("CREATE TEMP TABLE _fold_facts AS SELECT fact_id FROM atomic_facts "
                     "WHERE profile_id = ?", (source,))
    merged = _merge_entities(conn, present, source, target)
    order = [t for t in _FIRST if t in present] + [t for t in tables if t not in _FIRST]
    counts: dict[str, Any] = {}
    for table in order:  # every move before any delete: move triggers write state rows
        action = DECISIONS[table][0]
        if action not in (KEEP, DELETE):
            counts[table] = _apply(conn, table, action, source, target)
    for table in order:
        if DECISIONS[table][0] == DELETE:
            counts[table] = _apply(conn, table, DELETE, source, target)
    _requeue_projection(conn, present, target)
    conn.execute("DROP TABLE IF EXISTS temp._fold_facts")
    counts["merged_entities"] = merged
    return counts


__all__ = ["DECISIONS", "DELETE", "KEEP", "MERGE", "MOVE", "REKEY", "VEC", "ProfileFoldError",
           "fold_profile", "profile_scoped_tables"]
