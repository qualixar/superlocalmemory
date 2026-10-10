# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Health checks for a profile's saved documents.

Four checks, each one query plus a little Python:

1. ``empty_pages``: pages with no readable text (no text layer and nothing from OCR).
2. ``no_entities``: documents whose page facts mention no entity at all.
3. ``duplicate_pages``: pages of different documents whose picture hashes differ by at most 4 bits.
4. ``contradicted``: documents with a page fact on either end of a "contradiction" or "supersedes" edge
   of the memory graph (``None`` when the graph table cannot be read).

Nothing here logs titles or text.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

from superlocalmemory.documents.pagefacts import chunks, fact_ids_by_document, marks

logger = logging.getLogger(__name__)

MAX_ROWS = 500
MAX_HASHED_PAGES = 20000
MAX_DISTANCE = 4
_SLICES, _SLICE_BITS = 5, 13      # 5 slices of 13 bits: two hashes within 4 bits agree on a whole slice


def _live_documents(store: Any, profile_id: str) -> dict[str, str]:
    rows = store._read().execute(
        "SELECT document_id, title FROM documents WHERE profile_id = ? AND state != 'tombstoned'", (profile_id,))
    return {r[0]: r[1] for r in rows.fetchall()}


def _empty_pages(store: Any, profile_id: str) -> list[dict[str, Any]]:
    rows = store._read().execute(
        "SELECT d.document_id, d.title, p.page_no FROM doc_pages p JOIN documents d ON d.document_id = p.document_id"
        " WHERE d.profile_id = ? AND d.state != 'tombstoned' AND p.text_origin = 'none'"
        " ORDER BY d.created_at, d.document_id, p.page_no LIMIT ?", (profile_id, MAX_ROWS))
    return [{"document_id": r[0], "title": r[1], "page_no": r[2]} for r in rows.fetchall()]


def _linked_facts(db: Any, profile_id: str, fact_ids: list[str]) -> set[str]:
    linked: set[str] = set()
    for chunk in chunks(sorted(set(fact_ids))):
        rows = db.execute(
            "SELECT DISTINCT fact_id FROM fact_entity_associations"
            f" WHERE profile_id = ? AND fact_id IN ({marks(len(chunk))})", (profile_id, *chunk))
        linked.update(r["fact_id"] if hasattr(r, "keys") else r[0] for r in rows)
    return linked


def _no_entities(store: Any, db: Any, profile_id: str, titles: dict[str, str]) -> list[dict[str, Any]]:
    facts = fact_ids_by_document(store, list(titles))
    linked = _linked_facts(db, profile_id, [f for fs in facts.values() for f in fs])
    return [{"document_id": d, "title": titles[d]} for d in titles
            if facts[d] and not linked.intersection(facts[d])][:MAX_ROWS]


def _duplicate_pages(store: Any, profile_id: str, titles: dict[str, str]) -> list[dict[str, Any]]:
    rows = store._read().execute(
        "SELECT document_id, page_no, phash FROM media_items WHERE profile_id = ? AND kind = 'page'"
        " AND state = 'active' AND phash IS NOT NULL AND document_id IS NOT NULL LIMIT ?",
        (profile_id, MAX_HASHED_PAGES)).fetchall()
    pages: list[tuple[str, int, int]] = []
    for doc_id, page_no, phash in rows:
        try:
            if doc_id in titles:
                pages.append((doc_id, int(page_no), int(phash, 16)))
        except (TypeError, ValueError):
            continue
    buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index, (_, _, value) in enumerate(pages):
        for s in range(_SLICES):
            buckets[(s, (value >> (_SLICE_BITS * s)) & ((1 << _SLICE_BITS) - 1))].append(index)
    found: dict[tuple[int, int], int] = {}
    for members in buckets.values():
        for n, i in enumerate(members):
            for j in members[n + 1:]:
                a, b = pages[i], pages[j]
                distance = bin(a[2] ^ b[2]).count("1")
                if a[0] != b[0] and distance <= MAX_DISTANCE:
                    found[(min(i, j), max(i, j))] = distance
                if len(found) >= MAX_ROWS:
                    break
    out = []
    for (i, j), distance in found.items():
        first, second = sorted((pages[i], pages[j]), key=lambda p: (p[0], p[1]))
        out.append({"a": {"document_id": first[0], "page_no": first[1], "title": titles[first[0]]},
                    "b": {"document_id": second[0], "page_no": second[1], "title": titles[second[0]]},
                    "distance": distance})
    return sorted(out, key=lambda x: (x["a"]["document_id"], x["a"]["page_no"], x["b"]["document_id"], x["b"]["page_no"]))


def _contradicted(store: Any, db: Any, profile_id: str, titles: dict[str, str]) -> list[dict[str, Any]] | None:
    facts = fact_ids_by_document(store, list(titles))
    owner = {f: d for d, fs in facts.items() for f in fs}
    edges: set[tuple[str, str, str]] = set()
    try:
        for chunk in chunks(sorted(owner)):
            m = marks(len(chunk))
            rows = db.execute(
                "SELECT edge_id, source_id, target_id FROM graph_edges WHERE profile_id = ?"
                f" AND edge_type IN ('contradiction', 'supersedes') AND (source_id IN ({m}) OR target_id IN ({m}))",
                (profile_id, *chunk, *chunk))
            edges.update((r["edge_id"], r["source_id"], r["target_id"]) if hasattr(r, "keys") else tuple(r[:3])
                         for r in rows)
    except Exception as exc:  # noqa: BLE001 - no graph table (or it is unreadable): this check is skipped
        logger.info("contradiction check skipped (%s)", type(exc).__name__)
        return None
    per_doc: dict[str, int] = defaultdict(int)
    for _, source, target in edges:
        for doc_id in {owner.get(source), owner.get(target)} - {None}:
            per_doc[doc_id] += 1
    return [{"document_id": d, "title": titles[d], "edges": n} for d, n in sorted(per_doc.items())][:MAX_ROWS]


def document_lint(profile_id: str, *, store: Any = None, db: Any = None,
                  data_root: str | Path | None = None) -> dict[str, Any]:
    """The four checks for one profile. Without a media store every list is empty."""
    from superlocalmemory.media import open_media_store

    opened = store is None
    store = store if store is not None else open_media_store(data_root=data_root)
    if store is None:
        return {"empty_pages": [], "no_entities": [], "duplicate_pages": [], "contradicted": []}
    try:
        titles = _live_documents(store, profile_id)
        return {
            "empty_pages": _empty_pages(store, profile_id),
            "no_entities": _no_entities(store, db, profile_id, titles) if db is not None else [],
            "duplicate_pages": _duplicate_pages(store, profile_id, titles),
            "contradicted": _contradicted(store, db, profile_id, titles) if db is not None else None,
        }
    finally:
        if opened:
            store.close()
