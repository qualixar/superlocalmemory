# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Capture derivation lineage without inventing source spans."""

from __future__ import annotations

import hashlib
import json
from typing import Any


def _lineage_id(profile_id: str, object_type: str, object_id: str, operation_id: str) -> str:
    value = f"{profile_id}\0{object_type}\0{object_id}\0{operation_id}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _table_exists(db: Any, table: str) -> bool:
    return bool(db.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,),
    ))


def _record(
    db: Any,
    *,
    profile_id: str,
    object_type: str,
    object_id: str,
    operation_id: str,
    derivation_version: str,
    source_status: str,
    source_start: int | None = None,
    source_end: int | None = None,
    source_text_sha256: str = "",
    source_fact_ids: tuple[str, ...] = (),
    unresolved_reason: str = "",
) -> None:
    db.execute(
        "INSERT OR REPLACE INTO derivation_lineage "
        "(lineage_id,profile_id,object_type,object_id,operation_id,"
        "derivation_version,source_status,source_start,source_end,"
        "source_text_sha256,source_fact_ids_json,unresolved_reason) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            _lineage_id(profile_id, object_type, object_id, operation_id),
            profile_id, object_type, object_id, operation_id,
            derivation_version, source_status, source_start, source_end,
            source_text_sha256,
            json.dumps(list(source_fact_ids), separators=(",", ":")),
            unresolved_reason,
        ),
    )


def _fact_ids(value: Any) -> tuple[str, ...]:
    try:
        decoded = json.loads(value or "[]")
    except (TypeError, ValueError):
        return ()
    if not isinstance(decoded, list):
        return ()
    return tuple(str(item) for item in decoded)


def _capture_derived_rows(
    db: Any,
    *,
    table: str,
    id_column: str,
    object_type: str,
    profile_id: str,
    operation_id: str,
    derivation_version: str,
    operation_fact_ids: frozenset[str],
) -> None:
    if not _table_exists(db, table):
        return
    rows = db.execute(
        f'SELECT "{id_column}",fact_ids_json FROM "{table}" WHERE profile_id=?',
        (profile_id,),
    )
    for row in rows:
        source_ids = tuple(
            fact_id
            for fact_id in _fact_ids(row["fact_ids_json"])
            if fact_id in operation_fact_ids
        )
        if not source_ids:
            continue
        _record(
            db,
            profile_id=profile_id,
            object_type=object_type,
            object_id=str(row[id_column]),
            operation_id=operation_id,
            derivation_version=derivation_version,
            source_status="derived_from_facts",
            source_fact_ids=source_ids,
            unresolved_reason="direct_span_not_applicable",
        )


def capture_operation_lineage(
    db: Any,
    *,
    operation_id: str,
    profile_id: str,
    raw_content: str,
    fact_ids: tuple[str, ...],
    derivation_version: str,
) -> None:
    """Persist operation lineage while distinguishing spans from derivations."""
    if not _table_exists(db, "derivation_lineage"):
        return
    fact_lineage: dict[str, dict[str, Any]] = {}
    for fact_id in fact_ids:
        rows = db.execute(
            "SELECT content FROM atomic_facts WHERE fact_id=? AND profile_id=?",
            (fact_id, profile_id),
        )
        if not rows:
            _record(
                db, profile_id=profile_id, object_type="fact", object_id=fact_id,
                operation_id=operation_id, derivation_version=derivation_version,
                source_status="unresolved", unresolved_reason="final_fact_missing",
            )
            fact_lineage[fact_id] = {
                "source_status": "unresolved",
                "unresolved_reason": "final_fact_missing",
            }
            continue
        content = str(rows[0]["content"])
        start = raw_content.find(content)
        if start >= 0:
            _record(
                db, profile_id=profile_id, object_type="fact", object_id=fact_id,
                operation_id=operation_id, derivation_version=derivation_version,
                source_status="exact", source_start=start,
                source_end=start + len(content),
                source_text_sha256=hashlib.sha256(content.encode("utf-8")).hexdigest(),
            )
            fact_lineage[fact_id] = {
                "source_status": "exact",
                "source_start": start,
                "source_end": start + len(content),
                "source_text_sha256": hashlib.sha256(
                    content.encode("utf-8")
                ).hexdigest(),
            }
        else:
            _record(
                db, profile_id=profile_id, object_type="fact", object_id=fact_id,
                operation_id=operation_id, derivation_version=derivation_version,
                source_status="unresolved",
                unresolved_reason="no_exact_span_in_raw_source",
            )
            fact_lineage[fact_id] = {
                "source_status": "unresolved",
                "unresolved_reason": "no_exact_span_in_raw_source",
            }

    operation_fact_ids = frozenset(fact_ids)
    _record(
        db,
        profile_id=profile_id,
        object_type="profile",
        object_id=profile_id,
        operation_id=operation_id,
        derivation_version=derivation_version,
        source_status="not_applicable",
        unresolved_reason="profile_scope_not_derived_from_source_span",
    )
    _capture_derived_rows(
        db,
        table="entity_profiles",
        id_column="profile_entry_id",
        object_type="entity_summary",
        profile_id=profile_id,
        operation_id=operation_id,
        derivation_version=derivation_version,
        operation_fact_ids=operation_fact_ids,
    )
    _capture_derived_rows(
        db,
        table="memory_scenes",
        id_column="scene_id",
        object_type="memory_scene",
        profile_id=profile_id,
        operation_id=operation_id,
        derivation_version=derivation_version,
        operation_fact_ids=operation_fact_ids,
    )

    if _table_exists(db, "graph_edges"):
        rows = db.execute(
            "SELECT edge_id,source_id,target_id FROM graph_edges WHERE profile_id=?",
            (profile_id,),
        )
        for row in rows:
            source_ids = tuple(
                value
                for value in (str(row["source_id"]), str(row["target_id"]))
                if value in operation_fact_ids
            )
            if source_ids:
                _record(
                    db,
                    profile_id=profile_id,
                    object_type="graph_edge",
                    object_id=str(row["edge_id"]),
                    operation_id=operation_id,
                    derivation_version=derivation_version,
                    source_status="derived_from_facts",
                    source_fact_ids=source_ids,
                    unresolved_reason="direct_span_not_applicable",
                )

    if _table_exists(db, "bm25_tokens"):
        indexed_ids = {
            str(row["fact_id"])
            for row in db.execute(
                "SELECT fact_id FROM bm25_tokens WHERE profile_id=?", (profile_id,)
            )
        }
        for fact_id in operation_fact_ids & indexed_ids:
            lineage = fact_lineage.get(fact_id, {})
            _record(
                db,
                profile_id=profile_id,
                object_type="index_bm25",
                object_id=fact_id,
                operation_id=operation_id,
                derivation_version=derivation_version,
                source_status=str(lineage.get("source_status", "unresolved")),
                source_start=lineage.get("source_start"),
                source_end=lineage.get("source_end"),
                source_text_sha256=str(lineage.get("source_text_sha256", "")),
                source_fact_ids=(fact_id,),
                unresolved_reason=str(
                    lineage.get("unresolved_reason", "source_fact_lineage_missing")
                ),
            )


__all__ = ["capture_operation_lineage"]
