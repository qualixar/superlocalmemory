# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com
"""SuperLocalMemory V3 - Import/Export Routes
 - AGPL-3.0-or-later

Routes: /api/export, /api/import
"""
import io
import gzip
import hashlib
import json
import logging
from typing import Optional
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
    try:
        conn = get_db_connection()
        conn.row_factory = dict_factory
        cursor = conn.cursor()
        active_profile = get_active_profile()

        # Detect schema
        try:
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='atomic_facts'",
            )
            use_v3 = cursor.fetchone() is not None
        except Exception:
            use_v3 = False

        if use_v3:
            query = "SELECT * FROM atomic_facts WHERE profile_id = ?"
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
            content = gzip.decompress(content)

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

                metadata = {
                    "project_name": memory.get('project_name'),
                    "category": memory.get('category'),
                    "tags": memory.get('tags', ''),
                }
                receipt, created = command.submit_with_status(IngestionRequest(
                    content=memory_content,
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
                ))
                completed = command.materialize(receipt.operation_id)
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
