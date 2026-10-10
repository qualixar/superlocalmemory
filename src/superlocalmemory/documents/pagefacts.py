# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Which facts each saved document's pages produced, for the index and the health checks."""

from __future__ import annotations

import json
from typing import Any, Iterator, Sequence

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
