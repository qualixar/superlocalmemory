# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Ingestion endpoint — accepts data from external adapters.

POST /ingest with {content, source_type, dedup_key, metadata}.
Deduplicates by source_type + dedup_key. Stores via MemoryEngine.
Admission control: max 10 concurrent ingestions (HTTP 429 on overflow).

Part of Qualixar | Author: Varun Pratap Bhardwaj
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

router = APIRouter(tags=["ingestion"])

_MAX_CONCURRENT = 10
_active_count = 0
_active_lock = threading.Lock()


class IngestRequest(BaseModel):
    # Cap ingest content (~1 MB) so a single call cannot exhaust memory through
    # the embedding/tokenizer pipeline.
    content: str = Field(..., max_length=1_000_000)
    source_type: str
    dedup_key: str
    metadata: dict = {}


@router.post("/ingest")
async def ingest(req: IngestRequest, request: Request):
    """Ingest content from an external adapter.

    Deduplicates by (source_type, dedup_key). Returns 429 if too many
    concurrent ingestions. Stores via the singleton MemoryEngine.
    """
    global _active_count

    from .helpers import require_engine
    engine = require_engine(request)

    if not req.content:
        raise HTTPException(400, detail="content required")
    if not req.source_type:
        raise HTTPException(400, detail="source_type required")
    if not req.dedup_key:
        raise HTTPException(400, detail="dedup_key required")

    # Admission control
    with _active_lock:
        if _active_count >= _MAX_CONCURRENT:
            raise HTTPException(
                429,
                detail="Too many concurrent ingestions",
                headers={"Retry-After": "5"},
            )
        _active_count += 1

    try:
        from superlocalmemory.core.engine_ingestion import (
            build_engine_ingestion_command,
        )
        from superlocalmemory.core.ingestion_command import (
            IngestionRequest,
            IngestionState,
        )

        from superlocalmemory.server.write_identity import (
            authenticated_request_actor,
        )
        actor_id = authenticated_request_actor(
            request,
            actor_kind="http-ingest",
        )
        command = build_engine_ingestion_command(engine)
        receipt, created = command.submit_with_status(IngestionRequest(
            content=req.content,
            profile_id=engine._profile_id,
            source_type=req.source_type,
            idempotency_key=req.dedup_key,
            metadata=dict(req.metadata),
            trusted_actor_id=actor_id,
        ))
        completed = command.materialize(receipt.operation_id)
        if completed.state is not IngestionState.COMPLETE:
            raise RuntimeError(
                completed.last_error or "canonical adapter ingestion failed"
            )
        fact_ids = list(completed.fact_ids)

        # Compatibility ledger for existing dashboards. M018 is the source of
        # truth; retries repair this row without repeating the canonical write.
        engine._db.execute(
            "INSERT OR IGNORE INTO ingestion_log "
            "(profile_id, source_type, dedup_key, fact_ids, metadata, status, ingested_at) "
            "VALUES (?, ?, ?, ?, ?, 'ingested', ?)",
            (
                engine._profile_id,
                req.source_type,
                req.dedup_key,
                json.dumps(fact_ids),
                json.dumps(req.metadata),
                datetime.now(timezone.utc).isoformat(),
            ),
        )

        result = {
            "ingested": created,
            "operation_id": completed.operation_id,
            "fact_ids": fact_ids,
        }
        if not created:
            result["reason"] = "already_ingested"
        return result

    except Exception as exc:
        raise HTTPException(500, detail=str(exc))
    finally:
        with _active_lock:
            _active_count -= 1
