# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Ask how a document job is going, and take a document back out."""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

_DOC_VIEW = ("state", "page_count", "pages_text_layer", "pages_ocr", "pages_empty", "title")


def _open(store: Any) -> tuple[Any, bool]:
    if store is not None:
        return store, False
    from superlocalmemory.media import open_media_store

    return open_media_store(), True


def job_status(job_id: str, profile_id: str, *, store: Any = None) -> dict[str, Any] | None:
    """Progress of one job of this profile; None when it is unknown or belongs to another profile."""
    store_ref, opened = _open(store)
    if store_ref is None:
        return None
    try:
        job = store_ref.get_job(job_id)
        if not job or job["profile_id"] != profile_id:
            return None
        try:
            document_id = json.loads(job.get("payload_json") or "{}").get("document_id")
        except ValueError:
            document_id = None
        document = store_ref.get_document(document_id) if document_id else None
        return {
            "job_id": job["job_id"], "kind": job["kind"], "state": job["state"], "done": job["done"],
            "total": job["total"], "error": job["error"], "document_id": document_id,
            "document": {k: document[k] for k in _DOC_VIEW} if document else None,
        }
    finally:
        if opened:
            store_ref.close()


def _memory_facts(store: Any, document: dict[str, Any]) -> list[str]:
    facts = list(json.loads(document.get("fact_ids_json") or "[]"))
    for page in store.get_pages(document["document_id"]):
        facts += json.loads(page.get("fact_ids_json") or "[]")
    return facts


def remove_document(document_id: str, profile_id: str, *, hard: bool = False, runtime: Any = None,
                    store: Any = None) -> bool:
    """Take a document out of view: the document and its page pictures are tombstoned and the
    memories made from it are archived (kept, but hidden from recall). False when the document is
    unknown, belongs to another profile, or is already removed. Erasing it for good is not built yet.
    """
    if hard:
        raise NotImplementedError("erasing a document for good is not available yet")
    store_ref, opened = _open(store)
    if store_ref is None:
        return False
    try:
        document = store_ref.get_document(document_id)
        if not document or document["profile_id"] != profile_id or document["state"] == "tombstoned":
            return False
        facts = _memory_facts(store_ref, document)
        store_ref.tombstone_document(document_id)
        _archive(runtime, profile_id, document_id, facts)
        return True
    finally:
        if opened:
            store_ref.close()


def _archive(runtime: Any, profile_id: str, document_id: str, fact_ids: list[str]) -> None:
    for fact_id in fact_ids:
        try:
            runtime.archive_fact(profile_id, fact_id, idempotency_key=f"doc-remove:{document_id}:{fact_id}")
        except Exception as exc:  # noqa: BLE001 - hide what can be hidden; the rest stays visible, not lost
            logger.warning("a document memory could not be hidden (%s)", type(exc).__name__)
