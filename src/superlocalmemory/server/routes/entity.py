# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Entity compilation API routes — view and recompile entity summaries."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request, Query

from .helpers import require_engine, get_active_profile

router = APIRouter(prefix="/api/entity", tags=["entity"])


@router.get("/list")
async def list_entities(
    request: Request,
    profile: str | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
):
    """List all entities with basic info (canonical name, type, fact count)."""
    engine = require_engine(request)
    # Default to the ACTIVE profile (request runtime truth), never the literal
    # "default" — otherwise every profile sees the default profile's entities.
    profile = profile or get_active_profile()

    import sqlite3
    import json
    conn = sqlite3.connect(str(engine._config.db_path))
    conn.row_factory = sqlite3.Row
    try:
        total = conn.execute(
            "SELECT COUNT(*) FROM canonical_entities WHERE profile_id = ?",
            (profile,),
        ).fetchone()[0]

        rows = conn.execute("""
            SELECT ce.entity_id, ce.canonical_name, ce.entity_type,
                   ce.fact_count, ce.first_seen, ce.last_seen,
                   ep.knowledge_summary, ep.compiled_truth,
                   ep.compilation_confidence, ep.last_compiled_at
            FROM canonical_entities ce
            LEFT JOIN entity_profiles ep
              ON ce.entity_id = ep.entity_id AND ep.profile_id = ce.profile_id
            WHERE ce.profile_id = ?
            ORDER BY ce.fact_count DESC
            LIMIT ? OFFSET ?
        """, (profile, limit, offset)).fetchall()

        entities = []
        for r in rows:
            summary = r["knowledge_summary"] or ""
            entities.append({
                "entity_id": r["entity_id"],
                "name": r["canonical_name"],
                "type": r["entity_type"] or "unknown",
                "fact_count": r["fact_count"] or 0,
                "first_seen": r["first_seen"],
                "last_seen": r["last_seen"],
                "summary_preview": summary[:200] if summary else "",
                "has_compiled_truth": bool(r["compiled_truth"]),
                "confidence": r["compilation_confidence"] or 0.5,
                "last_compiled_at": r["last_compiled_at"],
            })

        return {"entities": entities, "total": total, "limit": limit, "offset": offset}
    finally:
        conn.close()


@router.get("/{entity_name}")
async def get_entity(
    entity_name: str,
    request: Request,
    profile: str | None = Query(default=None),
    project: str = Query(default=""),
):
    """Get compiled truth + timeline for an entity."""
    engine = require_engine(request)
    profile = profile or get_active_profile()

    import sqlite3
    import json
    conn = sqlite3.connect(str(engine._config.db_path))
    conn.row_factory = sqlite3.Row
    try:
        # Search by canonical_name (case-insensitive)
        row = conn.execute("""
            SELECT ep.compiled_truth, ep.timeline, ep.fact_ids_json,
                   ep.last_compiled_at, ep.compilation_confidence,
                   ep.knowledge_summary, ce.entity_type
            FROM entity_profiles ep
            JOIN canonical_entities ce ON ep.entity_id = ce.entity_id
            WHERE LOWER(ce.canonical_name) = LOWER(?)
              AND ep.profile_id = ?
              AND ep.project_name = ?
        """, (entity_name, profile, project)).fetchone()

        if not row:
            raise HTTPException(404, detail=f"Entity '{entity_name}' not found")

        return {
            "entity_name": entity_name,
            "entity_type": row["entity_type"],
            "compiled_truth": row["compiled_truth"] or "",
            "knowledge_summary": row["knowledge_summary"] or "",
            "timeline": json.loads(row["timeline"]) if row["timeline"] else [],
            "source_fact_ids": json.loads(row["fact_ids_json"]) if row["fact_ids_json"] else [],
            "last_compiled_at": row["last_compiled_at"],
            "confidence": row["compilation_confidence"],
        }
    finally:
        conn.close()


@router.post("/{entity_name}/recompile")
async def recompile_entity(
    entity_name: str,
    request: Request,
    profile: str | None = Query(default=None),
    project: str = Query(default=""),
):
    """Force immediate recompilation of an entity."""
    engine = require_engine(request)
    profile = profile or get_active_profile()

    import sqlite3
    conn = sqlite3.connect(str(engine._config.db_path))
    conn.row_factory = sqlite3.Row
    try:
        entity = conn.execute(
            "SELECT entity_id, canonical_name, entity_type FROM canonical_entities "
            "WHERE LOWER(canonical_name) = LOWER(?) AND profile_id = ?",
            (entity_name, profile),
        ).fetchone()

        if not entity:
            raise HTTPException(404, detail=f"Entity '{entity_name}' not found")

        from superlocalmemory.learning.entity_compiler import EntityCompiler
        compiler = EntityCompiler(str(engine._config.db_path), engine._config)
        result = compiler.compile_entity(
            profile, project, entity["entity_id"], entity["canonical_name"],
        )

        if result:
            return {"ok": True, **result}
        return {"ok": False, "reason": "no facts to compile"}
    finally:
        conn.close()
