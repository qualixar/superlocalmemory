# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Erasing a chosen set of facts (for example everything saved from one document).

The steps are the ones an entity erasure takes for the facts it removes: the erasure
service with a receipt, the fact-keyed side tables, the memory rows, and the text scrub.
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any, Sequence

logger = logging.getLogger(__name__)

_CHUNK = 400


def _targets(db: Any, profile_id: str, fact_ids: Sequence[str]) -> list[tuple[str, Any]]:
    found: list[tuple[str, Any]] = []
    ids = sorted({str(f) for f in fact_ids if f})
    for i in range(0, len(ids), _CHUNK):
        chunk = ids[i:i + _CHUNK]
        rows = db.execute(
            "SELECT fact_id, memory_id FROM atomic_facts WHERE profile_id = ? "
            f"AND fact_id IN ({','.join('?' * len(chunk))})", (profile_id, *chunk))
        found.extend((dict(r)["fact_id"], dict(r).get("memory_id")) for r in rows)
    return found


def erase_facts(gdpr: Any, fact_ids: Sequence[str], profile_id: str, subject_id: str) -> dict[str, Any]:
    """Erase these facts of one profile; ``counts["erasure_complete"]`` says whether every owner proved it."""
    from superlocalmemory.compliance.erased_cases import delete_erased_facts
    from superlocalmemory.core import erasure_scrub
    from superlocalmemory.core.transactions.concrete_owners import build_erasure_service_for_db
    from superlocalmemory.core.transactions.owners import OperationContext

    db, requested_at = gdpr._db, time.time()
    targets = _targets(db, profile_id, fact_ids)
    counts: dict[str, Any] = {"facts": len(targets)}
    gdpr._audit("delete", "facts", subject_id, f"targeted fact erasure in profile {profile_id}", profile_id=profile_id)
    if not targets:
        counts["erasure_complete"] = 1
        return counts
    ids = [fid for fid, _ in targets]
    named = {fid: erasure_scrub.entities_of(db, profile_id, fid) for fid in ids}
    erasure_scrub.prepare(db)
    op_id = uuid.uuid4().hex
    context = OperationContext(operation_id=op_id, profile_id=profile_id, subject_id=subject_id,
                               fact_ids=tuple(sorted(ids)))
    service = build_erasure_service_for_db(db, gdpr._engine, gdpr._data_root)
    service.remove(db, context)
    receipt = service.finalize(db, context, subject_type="fact", subject_id=subject_id,
                               requested_by="gdpr", requested_at=requested_at)
    if not receipt.persisted:
        counts["receipt_persist_failed"] = 1
    if not receipt.all_erased:
        counts["vector_store_failures"] = sum(1 for p in receipt.proofs if not p.erased)
    gdpr._erase_fact_keyed_tables_for(ids, counts)
    delete_erased_facts(db, targets, profile_id, erasure_id=op_id,
                        has_siblings=gdpr._memory_has_siblings, counts=counts)
    try:
        for fid in ids:
            erasure_scrub.scrub(db, profile_id, fid, named[fid])
    except Exception as exc:  # noqa: BLE001 - reported in the counts
        logger.error("targeted erase: text scrub failed (%s)", type(exc).__name__)
        counts["text_scrub_failed"] = 1
    failed = ("vector_store_failures", "text_scrub_failed", *(f"{t}_failed" for t, _ in gdpr._FACT_KEYED_TABLES))
    counts["erasure_complete"] = 0 if any(counts.get(m) for m in failed) else 1
    return counts
