# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Find shared folder rows whose borrowed memory, picture or document is gone, so they can be saved afresh."""

from __future__ import annotations

import logging
from typing import Any

from superlocalmemory.sources.store import SourceStore, entries_of

logger = logging.getLogger(__name__)

_BATCH = 400


def _living_memories(runtime: Any, ids: list[str]) -> set[str] | None:
    """Memories with at least one fact that is not archived; None when the writer cannot be asked."""
    db = getattr(runtime, "_db", None)
    if db is None:
        return None
    alive: set[str] = set()
    try:
        for i in range(0, len(ids), _BATCH):
            part = ids[i:i + _BATCH]
            rows = db.execute("SELECT DISTINCT memory_id FROM atomic_facts WHERE lifecycle != 'archived'"
                              " AND memory_id IN (" + ",".join("?" * len(part)) + ")", tuple(part))
            alive.update(r[0] for r in rows)
    except Exception as exc:  # noqa: BLE001 - not knowing leaves the row as it is
        logger.warning("could not check borrowed folder memories (%s)", type(exc).__name__)
        return None
    return alive


def _dead_documents(store: SourceStore, ids: list[str]) -> set[str]:
    dead = set()
    for doc_id in ids:
        doc = store._m.get_document(doc_id)
        if doc is None or doc["state"] == "tombstoned":
            dead.add(doc_id)
    return dead


def reset_dead_borrows(store: SourceStore, runtime: Any, source_id: str) -> int:
    """Put back to ``pending`` every shared row whose borrowed item was hidden or erased since."""
    shared = [r for r in store.files(source_id, ("indexed",)) if r.get("reason") == "shared"]
    if not shared:
        return 0
    memories = sorted({e["shared_m"] for r in shared for e in entries_of(r) if e.get("shared_m")})
    docs = sorted({e["shared_doc"] for r in shared for e in entries_of(r) if e.get("shared_doc")})
    alive = _living_memories(runtime, memories) if memories else set()
    dead = _dead_documents(store, docs)
    if alive is not None:
        dead |= set(memories) - alive
    count = 0
    for row in shared:
        borrowed = {e.get("shared_m") or e.get("shared_doc") for e in entries_of(row)}
        if borrowed & dead:
            store.put_file(source_id, row["relpath"], state="pending")
            count += 1
    return count


__all__ = ["reset_dead_borrows"]
