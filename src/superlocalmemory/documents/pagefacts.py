# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Which facts each saved document's pages produced, for the index and the health checks."""

from __future__ import annotations

import json
import logging
from typing import Any, Iterator, Sequence

logger = logging.getLogger(__name__)

CHUNK = 400


def chunks(values: Sequence[str]) -> Iterator[list[str]]:
    items = list(values)
    for i in range(0, len(items), CHUNK):
        yield items[i:i + CHUNK]


def marks(n: int) -> str:
    return ",".join("?" * n)


def fact_ids_by_document(store: Any, document_ids: Sequence[str]) -> dict[str, list[str]]:
    """document_id -> fact ids of its pages (one read per 400 documents)."""
    out: dict[str, list[str]] = {d: [] for d in document_ids}
    for chunk in chunks(document_ids):
        rows = store._read().execute(
            f"SELECT document_id, fact_ids_json FROM doc_pages WHERE document_id IN ({marks(len(chunk))})", chunk)
        for doc_id, text in rows.fetchall():
            out[doc_id].extend(json.loads(text or "[]"))
    return out


def committed_save_facts(runtime: Any, profile_id: str, document_id: str) -> list[str]:
    """Fact ids of every page and document memory the writer committed under this document's keys.

    A page whose save was still queued when its job paused (Deferred) was never recorded in
    ``doc_pages``, yet its memory exists once the commit lands. The writer's own operation rows
    (keys ``doc:<document_id>:...``) are the only complete list, so removal and erasure read them too.
    """
    db = getattr(runtime, "_db", None)
    if db is None or not document_id:
        return []
    like = "doc:" + document_id.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + ":%"
    try:
        rows = db.execute(
            "SELECT queryable_fact_ids_json, final_fact_ids_json FROM ingestion_operations"
            " WHERE profile_id = ? AND source_type = 'document' AND idempotency_key LIKE ? ESCAPE '\\'",
            (profile_id, like))
        return sorted({str(f) for row in rows for column in (row[0], row[1]) for f in json.loads(column or "[]")})
    except Exception as exc:  # noqa: BLE001 - the recorded pages are still hidden; this only adds to them
        logger.warning("could not look up a document's committed saves (%s)", type(exc).__name__)
        return []
