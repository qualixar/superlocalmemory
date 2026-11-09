# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com
"""SuperLocalMemory V3 - Import/Export Routes
 - AGPL-3.0-or-later

Routes: /api/export, /api/import
"""
import asyncio
import gzip
import hashlib
import io
import json
import logging
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, File, HTTPException, Query, Request, UploadFile

from .export_delivery import export_response
from .export_memories import write_memories_export
from .helpers import (
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
    include_embeddings: bool = Query(
        False, description="Include embeddings as base64 (omitted by default)"),
):
    """Export memories as JSON, JSONL, or CSV.

    The file is written to a temporary file first and sent only when whole, so a
    failure is an error status and never a download that looks complete.
    Embeddings are left out unless ``include_embeddings`` is set.
    """
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

    def write(fp) -> int:
        conn = get_db_connection()
        try:
            return write_memories_export(
                conn, fp, format, active_profile, category=category,
                project_name=project_name, include_embeddings=include_embeddings)
        finally:
            conn.close()

    ts = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    media = {"jsonl": "application/x-ndjson", "csv": "text/csv"}.get(format, "application/json")
    return await export_response(
        write, media_type=media, filename=f"memories_export_{ts}.{format}", gzip_over=10000)


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
            IdempotencyConflict,
            IngestionRequest,
            IngestionState,
        )
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
