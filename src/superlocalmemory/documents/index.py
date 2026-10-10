# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""The list of saved documents, with the main entities each one mentions.

The entities come from the facts the pages produced (the entity tables of the memory
database; nothing new is stored). The result is cached; the memory database has no change
counter, so the cache key carries the profile's count of documents and latest document update
plus its count of facts and latest fact ``created_at``. Anything that adds, removes or
re-saves a fact or a document changes the key.

Nothing here logs document titles or text.
"""

from __future__ import annotations

import base64
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from superlocalmemory.cache.keys import CacheKey, params_hash
from superlocalmemory.documents.pagefacts import chunks, fact_ids_by_document, marks

DERIVER_ID = "doc.index"
DERIVER_VERSION = "1"
TOP_ENTITIES = 5
MAX_LIMIT = 200
_VIEW = ("document_id", "title", "state", "page_count", "pages_text_layer", "pages_ocr", "pages_empty", "created_at")


def _decode(cursor: str) -> tuple[str, str] | None:
    try:
        created, doc_id = json.loads(base64.urlsafe_b64decode(cursor.encode("ascii")))
        return str(created), str(doc_id)
    except (ValueError, TypeError, UnicodeError):
        return None


def _encode(created_at: str, document_id: str) -> str:
    return base64.urlsafe_b64encode(json.dumps([created_at, document_id]).encode()).decode("ascii")


def _signature(store: Any, db: Any, profile_id: str) -> str:
    docs = store._read().execute(
        "SELECT COUNT(*), COALESCE(MAX(updated_at), '') FROM documents WHERE profile_id = ? AND state != 'tombstoned'",
        (profile_id,)).fetchone()
    facts: tuple = (0, "")
    if db is not None:
        rows = db.execute("SELECT COUNT(*) AS n, COALESCE(MAX(created_at), '') AS latest FROM atomic_facts "
                          "WHERE profile_id = ?", (profile_id,))
        if rows:
            row = rows[0]
            facts = (row["n"], row["latest"]) if hasattr(row, "keys") else (row[0], row[1])
    return json.dumps([profile_id, list(docs), list(facts)])


def _entities(db: Any, profile_id: str, fact_ids: list[str]) -> list[dict[str, Any]]:
    if db is None or not fact_ids:
        return []
    counts: Counter = Counter()
    for chunk in chunks(sorted(set(fact_ids))):
        rows = db.execute(
            "SELECT e.canonical_name AS name, COUNT(DISTINCT a.fact_id) AS n FROM fact_entity_associations a"
            " JOIN canonical_entities e ON e.entity_id = a.entity_id"
            f" WHERE a.profile_id = ? AND a.fact_id IN ({marks(len(chunk))}) GROUP BY a.entity_id, e.canonical_name",
            (profile_id, *chunk))
        for row in rows:
            name, n = (row["name"], row["n"]) if hasattr(row, "keys") else (row[0], row[1])
            counts[name] += int(n)
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:TOP_ENTITIES]
    return [{"name": name, "facts": n} for name, n in ranked]


def _compute(store: Any, db: Any, profile_id: str, limit: int, cursor: str) -> dict[str, Any]:
    sql, args = "SELECT * FROM documents WHERE profile_id = ? AND state != 'tombstoned'", [profile_id]
    after = _decode(cursor) if cursor else None
    if after:
        sql += " AND (created_at < ? OR (created_at = ? AND document_id < ?))"
        args += [after[0], after[0], after[1]]
    rows = store._read().execute(sql + " ORDER BY created_at DESC, document_id DESC LIMIT ?",
                                 [*args, limit + 1]).fetchall()
    page = [dict(r) for r in rows[:limit]]
    facts = fact_ids_by_document(store, [d["document_id"] for d in page])
    documents = [{**{k: d[k] for k in _VIEW}, "entities": _entities(db, profile_id, facts[d["document_id"]])}
                 for d in page]
    more = len(rows) > limit
    return {"documents": documents, "next_cursor": _encode(page[-1]["created_at"], page[-1]["document_id"]) if more else None}


def document_index(profile_id: str, limit: int = 50, cursor: str = "", *, store: Any = None, db: Any = None,
                   cache: Any = None, data_root: str | Path | None = None) -> dict[str, Any]:
    """One page of the profile's documents, newest first: ``{"documents": [...], "next_cursor": str | None}``."""
    from superlocalmemory.cache.factory import default_cache, get_or_compute
    from superlocalmemory.media import open_media_store

    limit = max(1, min(int(limit), MAX_LIMIT))
    opened = store is None
    store = store if store is not None else open_media_store(data_root=data_root)
    if store is None:
        return {"documents": [], "next_cursor": None}
    try:
        key = CacheKey(hashlib.sha256(_signature(store, db, profile_id).encode()).hexdigest(), DERIVER_ID,
                       DERIVER_VERSION,
                       params_hash=params_hash({"limit": limit, "cursor": cursor}))
        payload = get_or_compute(cache if cache is not None else default_cache(), key, "json", lambda: json.dumps(
            _compute(store, db, profile_id, limit, cursor), sort_keys=True).encode("utf-8"))
        return json.loads(payload)
    finally:
        if opened:
            store.close()
