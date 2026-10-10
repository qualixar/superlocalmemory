# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Ask how a document job is going, and take a document back out."""

from __future__ import annotations

import json
import logging
from pathlib import Path
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
                    eraser: Any = None, store: Any = None) -> bool:
    """Take a document out of view, or (``hard=True``) erase it for good.

    Soft: the document and its page pictures are tombstoned and the memories made from it are
    archived (kept, but hidden from recall); False when the document is unknown, belongs to another
    profile, or is already removed.

    Hard: ``eraser(profile_id, fact_ids, document_id)`` erases every memory made from the document
    through the erasure service and returns its counts; then the page pictures, vectors, rows and the
    PDF go (the PDF stays while another document uses it). False when the document is unknown, belongs
    to another profile, or the erasure was not complete (nothing more is dropped then).
    """
    if hard and eraser is None:
        raise ValueError("a hard removal needs an eraser")
    store_ref, opened = _open(store)
    if store_ref is None:
        return False
    try:
        document = store_ref.get_document(document_id)
        if not document or document["profile_id"] != profile_id:
            return False
        if hard:
            return _erase(store_ref, document, eraser)
        if document["state"] == "tombstoned":
            return False
        # Hide first: if any memory cannot be hidden, the document stays listed, so the person
        # can try again (a removed-looking document with recallable pages has no way back).
        facts = _memory_facts(store_ref, document)
        if _archive(runtime, profile_id, document_id, facts):
            return False
        store_ref.tombstone_document(document_id)
        # Pages a running job saved during the removal; anything later is hidden by the job itself
        # when it sees the tombstone (DocumentJob._hide_saved).
        late = [f for f in _memory_facts(store_ref, store_ref.get_document(document_id) or document)
                if f not in set(facts)]
        _archive(runtime, profile_id, document_id, late)
        return True
    finally:
        if opened:
            store_ref.close()


def _erase(store: Any, document: dict[str, Any], eraser: Any) -> bool:
    from superlocalmemory.media.erasure_documents import erase_document_rows

    facts = sorted({f for f in _memory_facts(store, document) if f})
    if facts:
        counts = eraser(document["profile_id"], facts, document["document_id"]) or {}
        if not counts.get("erasure_complete"):
            logger.warning("a document erasure was not complete")
            return False
    out = erase_document_rows(store, Path(store.path).parent, [document["document_id"]])
    return not out["residue"]


def archive_document_facts(store: Any, runtime: Any, document: dict[str, Any]) -> int:
    """Hide every memory a document owns; returns how many could not be hidden."""
    return _archive(runtime, document["profile_id"], document["document_id"], _memory_facts(store, document))


def _archive(runtime: Any, profile_id: str, document_id: str, fact_ids: list[str]) -> int:
    """Hide each memory (idempotent per document and fact); returns how many failed."""
    failed = 0
    for fact_id in fact_ids:
        try:
            runtime.archive_fact(profile_id, fact_id, idempotency_key=f"doc-remove:{document_id}:{fact_id}")
        except Exception as exc:  # noqa: BLE001 - counted; the caller decides
            failed += 1
            logger.warning("a document memory could not be hidden (%s)", type(exc).__name__)
    return failed
