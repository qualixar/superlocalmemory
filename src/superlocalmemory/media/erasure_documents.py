# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Erasing saved documents: page pictures follow their memories, the document follows its last page.

Nothing here logs document text or file locations.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Iterable

from superlocalmemory.media import files
from superlocalmemory.media.erasure import erase_items

logger = logging.getLogger(__name__)

INDEX_DERIVER = "doc.index"


def forget_cached_index(root: Path) -> None:
    """The cached document list names entities; drop it whenever a page memory is erased."""
    try:
        from superlocalmemory.cache.factory import invalidate_deriver

        invalidate_deriver(INDEX_DERIVER, root)
    except Exception as exc:  # noqa: BLE001 - the list is rebuilt from live rows anyway
        logger.warning("the cached document list could not be cleared (%s)", type(exc).__name__)


def erase_document_rows(store: Any, root: Path, document_ids: Iterable[str]) -> dict[str, Any]:
    """Remove documents with every page picture, then the PDF when no other document or row uses it.

    Returns ``{"documents", "items", "files", "residue"}`` (the residue names ids and short hashes only).
    """
    ids = sorted(set(document_ids))
    out: dict[str, Any] = {"documents": 0, "items": 0, "files": 0, "residue": []}
    if not ids:
        return out
    pages = erase_items(store, root, store.document_item_ids(ids))
    out["items"] = pages["items"]
    out["residue"].extend(pages["residue"])
    removed = store.delete_documents(ids)
    out["documents"] = len(removed)
    for doc in removed:
        rel, sha = doc["source_relpath"], doc["sha256"]
        if not rel or store.file_in_use(sha, rel) or not (files.media_root(root) / rel).exists():
            continue
        files.remove_original(root, rel)
        if (files.media_root(root) / rel).exists():
            out["residue"].append(f"file:{(sha or '')[:12]}")
        else:
            out["files"] += 1
    forget_cached_index(root)
    return out


def erase_pages_of_memories(store: Any, root: Path, profile_id: str, memory_ids: Iterable[str],
                            fact_ids: Iterable[str], anchor_ids: Iterable[str] = ()) -> dict[str, Any]:
    """Erase the page pictures of erased memories, and any document left with no page memory.

    ``anchor_ids`` are picture ids already chosen by the caller (they are erased in the same step).
    Returns the same shape as ``erase_document_rows`` plus ``"touched"``.
    """
    memories = sorted({str(m) for m in memory_ids if m})
    page_ids = store.page_item_ids_for_memories(profile_id, memories)
    picture_ids = sorted({*page_ids, *anchor_ids})
    pictures = erase_items(store, root, picture_ids)
    touched = store.drop_page_memories(profile_id, memories, list(fact_ids))
    out = erase_document_rows(store, root, store.documents_left_empty(touched)) if touched else {
        "documents": 0, "items": 0, "files": 0, "residue": []}
    out["items"] += pictures["items"]
    out["files"] += pictures["files"]
    out["residue"] = [*pictures["residue"], *out["residue"]]
    out["touched"], out["picture_ids"] = touched, picture_ids
    if touched:
        forget_cached_index(root)
    return out
