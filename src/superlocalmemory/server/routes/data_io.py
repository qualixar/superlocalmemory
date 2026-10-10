# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com
"""SuperLocalMemory V3 - Import/Export Routes
 - AGPL-3.0-or-later

Routes: /api/export, /api/import
"""
import asyncio
import io
import gzip
import hashlib
import json
import logging
from typing import Any, Optional
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Query, Request, UploadFile, File
from fastapi.responses import StreamingResponse

from .helpers import (
    DB_PATH,
    dict_factory,
    get_active_profile,
    get_db_connection,
    require_engine,
)

logger = logging.getLogger("superlocalmemory.routes.data_io")

# Hard cap on the total decompressed byte count for gzip imports.
# Bounding the compressed upload size alone does not prevent a decompression
# bomb: a few kilobytes of input can expand to gigabytes.  This cap is checked
# incrementally during streaming decompression so the full expanded content is
# never materialized before the guard fires.
_MAX_DECOMPRESSED_BYTES: int = 200 * 1024 * 1024  # 200 MB


def _internal_error(detail: str = "Internal server error") -> HTTPException:
    """SEC-H-02: log full traceback server-side; return a generic message to the client."""
    logger.exception("data_io route error")
    return HTTPException(status_code=500, detail=detail)

# WebSocket manager reference (set by ui_server.py at startup)
ws_manager = None

router = APIRouter()


@router.get("/api/export")
async def export_memories(
    request: Request,
    format: str = Query("json", pattern="^(json|jsonl|csv)$"),
    category: Optional[str] = None,
    project_name: Optional[str] = None,
):
    """Export memories as JSON, JSONL, or CSV."""
    # Bulk data export. This GET is not covered by the mutation middleware and
    # is reached both by a plain fetch and a top-level navigation, neither of
    # which carries a credential header — so gate on the loopback-trusted
    # mutation boundary: local owner allowed, remote uncredentialed fails closed.
    from superlocalmemory.server.write_identity import require_http_mutation_actor
    require_http_mutation_actor(request, getattr(request.app.state, "daemon_descriptor", None),
                                actor_kind="data-export")
    # The actor check above admits any loopback process. The export is every
    # memory of the active workspace, so it also needs READ on it: a signed-in
    # user with that role, or the owner when the workspace does not require
    # login. Company mode with no session is refused here (401).
    from superlocalmemory.access.rbac import Permission
    from superlocalmemory.server.rbac_enforce import require_permission
    active_profile = get_active_profile()
    require_permission(request, Permission.READ, profile=active_profile)
    try:
        conn = get_db_connection()
        conn.row_factory = dict_factory
        cursor = conn.cursor()

        # Detect schema
        try:
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='atomic_facts'",
            )
            use_v3 = cursor.fetchone() is not None
        except Exception:
            use_v3 = False

        if use_v3:
            # Withheld summaries are excluded, and this is about IMPORT, not
            # tidiness. Import re-ingests every record through the normal
            # pipeline, which mints a fresh memory with quarantined = 0 — so an
            # export taken here and imported anywhere would resurrect all 1,195
            # model-written rows as though the owner had written them, on a
            # machine where nothing had gone wrong. The repair would then have
            # to run again there.
            #
            # Nothing the owner wrote is lost: these are derived artefacts, the
            # consolidator regenerates them from the facts that ARE exported,
            # and their text is kept in consolidated_summaries. (That table is
            # not in this export either — it is a view, not a memory. Worth
            # revisiting when export covers derived state.)
            query = "SELECT * FROM atomic_facts WHERE profile_id = ?"
            if _has_column(cursor, "atomic_facts", "quarantined"):
                query += " AND COALESCE(quarantined, 0) = 0"
            params = [active_profile]
            if category:
                query += " AND fact_type = ?"
                params.append(category)
            if project_name:
                query += " AND session_id = ?"
                params.append(project_name)
            query += " ORDER BY created_at"
        else:
            query = "SELECT * FROM memories WHERE profile = ?"
            params = [active_profile]
            if category:
                query += " AND category = ?"
                params.append(category)
            if project_name:
                query += " AND project_name = ?"
                params.append(project_name)
            query += " ORDER BY created_at"

        cursor.execute(query, params)
        memories = cursor.fetchall()
        if use_v3:
            _mark_sources(cursor, memories, active_profile)
        conn.close()

        if format == "jsonl":
            content = "\n".join(json.dumps(m) for m in memories)
            media_type = "application/x-ndjson"
        elif format == "csv":
            import csv
            import io as _io
            if memories:
                buf = _io.StringIO()
                fieldnames = list(memories[0].keys())
                writer = csv.DictWriter(
                    buf, fieldnames=fieldnames, extrasaction="ignore",
                )
                writer.writeheader()
                for m in memories:
                    writer.writerow({
                        k: (json.dumps(v) if isinstance(v, (dict, list)) else v)
                        for k, v in m.items()
                    })
                content = buf.getvalue()
            else:
                content = ""
            media_type = "text/csv"
        else:
            content = json.dumps({
                "version": "3.0.0",
                "exported_at": datetime.now(timezone.utc).isoformat(),
                "total_memories": len(memories),
                "filters": {"category": category, "project_name": project_name},
                "memories": memories,
            }, indent=2)
            media_type = "application/json"

        ts = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
        if len(content) > 10000:
            compressed = gzip.compress(content.encode())
            return StreamingResponse(
                io.BytesIO(compressed), media_type="application/gzip",
                headers={
                    "Content-Disposition": f"attachment; filename=memories_export_{ts}.{format}.gz",
                },
            )
        return StreamingResponse(
            io.BytesIO(content.encode()), media_type=media_type,
            headers={
                "Content-Disposition": f"attachment; filename=memories_export_{ts}.{format}",
            },
        )

    except Exception:
        raise _internal_error("Export error")


# What a marker may say in an export: ids only, never a path or a file name.
_SOURCE_KEYS = ("media_id", "document_id", "page", "part", "source_id")
_SOURCE_TYPES = {"media": "media", "document": "document_page", "folder": "folder"}


def _mark_sources(cursor: Any, facts: list, profile_id: str) -> None:
    """Tag the facts that came from a picture, a document page or a connected folder.

    Only their text is exported (never the picture, the document or the folder),
    so the record says where the text came from. Other facts are left untouched.
    """
    try:
        cursor.execute(
            "SELECT memory_id, metadata_json FROM memories WHERE profile_id = ?"
            " AND metadata_json LIKE '%_slm_source%'", (profile_id,))
        marks: dict[str, dict] = {}
        for row in cursor.fetchall():
            try:
                raw = (json.loads(row["metadata_json"] or "{}") or {}).get("_slm_source")
            except ValueError:
                continue
            kind = _SOURCE_TYPES.get(raw.get("type")) if isinstance(raw, dict) else None
            if kind:
                marks[row["memory_id"]] = {
                    "type": kind, **{k: raw[k] for k in _SOURCE_KEYS if k in raw}}
    except Exception:  # noqa: BLE001 -- an export must not fail over a marker
        logger.warning("export: source markers unavailable")
        return
    for fact in facts:
        mark = marks.get(fact.get("memory_id"))
        if mark:
            fact["source"] = dict(mark)


def _has_column(cursor: Any, table: str, column: str) -> bool:
    """Whether ``table`` carries ``column`` in this database.

    Presence-guarded because ``quarantined`` arrives with a migration and this
    route must keep working on a store the engine has not opened.
    """
    try:
        cursor.execute(f"PRAGMA table_info({table})")
        return any(row[1] == column for row in cursor.fetchall())
    except Exception:  # noqa: BLE001 -- an export must not fail over a probe
        return False


@router.post("/api/import")
async def import_memories(request: Request, file: UploadFile = File(...)):
    """Import memories from JSON file using V3 engine."""
    try:
        # Bound the upload so a huge file cannot OOM the daemon (read one byte
        # past the cap to detect oversize without buffering the whole payload).
        _MAX_IMPORT_BYTES = 50 * 1024 * 1024
        content = await file.read(_MAX_IMPORT_BYTES + 1)
        if len(content) > _MAX_IMPORT_BYTES:
            raise HTTPException(status_code=413,
                                detail="Import file exceeds the 50 MB limit")
        if file.filename and file.filename.endswith('.gz'):
            # Stream-decompress with an incremental byte counter so the full
            # expanded payload is never allocated before the guard fires.
            _chunk_size = 65_536
            chunks: list[bytes] = []
            total_decompressed = 0
            with gzip.GzipFile(fileobj=io.BytesIO(content)) as _gz:
                while True:
                    chunk = _gz.read(_chunk_size)
                    if not chunk:
                        break
                    total_decompressed += len(chunk)
                    if total_decompressed > _MAX_DECOMPRESSED_BYTES:
                        raise HTTPException(
                            status_code=413,
                            detail=(
                                f"Decompressed content exceeds the "
                                f"{_MAX_DECOMPRESSED_BYTES // (1024 * 1024)} MB limit"
                            ),
                        )
                    chunks.append(chunk)
            content = b"".join(chunks)

        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            logger.warning("import: invalid JSON payload")
            raise HTTPException(status_code=400, detail="Invalid JSON format")

        if isinstance(data, dict) and 'memories' in data:
            memories = data['memories']
        elif isinstance(data, list):
            memories = data
        else:
            raise HTTPException(
                status_code=400, detail="Invalid format: expected 'memories' array",
            )

        engine = require_engine(request)
        from superlocalmemory.core.engine_ingestion import (
            build_engine_ingestion_command,
        )
        from superlocalmemory.core.ingestion_command import (
            IngestionRequest,
            IngestionState,
        )

        from superlocalmemory.core.ingestion_command import IdempotencyConflict
        from superlocalmemory.core.metadata_guard import strip_reserved_metadata
        from superlocalmemory.memory_core import (
            find_redacted_duplicate,
            pii_redaction_enabled,
            prepare_metadata,
            prepare_user_text,
        )

        redact = pii_redaction_enabled(engine._config)

        command = build_engine_ingestion_command(engine)
        from superlocalmemory.server.write_identity import (
            authenticated_request_actor,
        )
        actor_id = authenticated_request_actor(
            request,
            actor_kind="http-import",
        )
        file_digest = hashlib.sha256(content).hexdigest()
        imported = 0
        skipped = 0
        errors = []
        operation_ids: list[str] = []

        for idx, memory in enumerate(memories):
            try:
                memory_content = memory.get('content')
                if not memory_content:
                    errors.append(f"Memory {idx}: missing 'content' field")
                    continue
                # Imported exactly as exported, credentials included, like any
                # other save: an export and re-import must round-trip unchanged.

                metadata = {
                    "project_name": memory.get('project_name'),
                    "category": memory.get('category'),
                    "tags": memory.get('tags', ''),
                }
                for _field in (
                    "fact_type", "confidence", "importance", "entities",
                    "canonical_entities", "referenced_date", "pinned",
                ):
                    if _field in memory:
                        metadata[_field] = memory[_field]
                prepared = prepare_user_text(engine._config, memory_content)
                metadata, _ = prepare_metadata(
                    strip_reserved_metadata(metadata), pii_redaction=redact,
                )
                request_obj = IngestionRequest(
                    content=prepared.text,
                    profile_id=engine._profile_id,
                    source_type="http-import",
                    idempotency_key=f"import:{file_digest}:{idx}",
                    metadata=metadata,
                    scope=memory.get("scope") or "personal",
                    shared_with=tuple(memory.get("shared_with") or ()),
                    trusted_actor_id=actor_id,
                    session_id=memory.get('session_id', ''),
                    session_date=memory.get('session_date') or "",
                    speaker=memory.get('speaker') or "",
                    role=memory.get('role') or "user",
                )
                try:
                    receipt, created = command.submit_with_status(request_obj)
                except IdempotencyConflict:
                    # Saved raw before redaction was on: the same text once
                    # prepared is already present, not a failure.
                    if find_redacted_duplicate(
                        engine._db, request_obj, redact,
                    ) is None:
                        raise
                    skipped += 1
                    continue
                completed = await asyncio.to_thread(command.materialize, receipt.operation_id)
                if completed.state is not IngestionState.COMPLETE:
                    raise RuntimeError(
                        completed.last_error or "canonical import failed"
                    )
                operation_ids.append(completed.operation_id)
                if created:
                    imported += 1
                else:
                    skipped += 1

                if ws_manager:
                    await ws_manager.broadcast({
                        "type": "memory_added", "memory_id": imported,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    })

            except Exception as e:
                if "UNIQUE constraint failed" in str(e):
                    skipped += 1
                else:
                    logger.warning("import: memory %d failed: %s", idx, e)
                    errors.append(f"Memory {idx}: import failed")

        return {
            "success": True, "imported_count": imported,
            "skipped_count": skipped, "total_processed": len(memories),
            "errors": errors[:10], "operation_ids": operation_ids,
        }

    except HTTPException:
        raise
    except Exception:
        raise _internal_error("Import error")
