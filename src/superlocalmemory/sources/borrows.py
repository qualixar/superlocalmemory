# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Find shared folder rows whose borrowed memory, picture or document is gone, so they can be saved afresh."""

from __future__ import annotations

from typing import Any

from superlocalmemory.sources.store import SourceStore, entries_of


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
    dead = _dead_documents(store, docs)
    if memories:  # a picture borrow is dead only when its media item is no longer active
        dead |= set(memories) - store._m.active_anchor_ids(memories)
    count = 0
    for row in shared:
        borrowed = {e.get("shared_m") or e.get("shared_doc") for e in entries_of(row)}
        if borrowed & dead:
            store.put_file(source_id, row["relpath"], state="pending")
            count += 1
    return count


__all__ = ["reset_dead_borrows"]
