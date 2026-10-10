# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V4 | https://qualixar.com | https://varunpratap.com

"""The pictures, documents, connected folders and jobs of one profile, for the
right-of-access export.

Read straight from ``media.db`` (opened read-only, never created) and limited to
the one profile. Records only: no picture or file bytes, no thumbnails (the
record says whether one exists), no stored-file locations and no job input.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

_PAGE = 500
_PICTURE = ("media_id, kind, mime, bytes, width, height, captured_at, exif_json, anchor_memory_id,"
            " origin, state, created_at, tombstoned_at, (thumb_webp IS NOT NULL) AS has_thumbnail,"
            " (original_relpath IS NOT NULL AND original_relpath != '') AS has_original")
_DOCUMENT = ("document_id, title, mime, bytes, page_count, pages_text_layer, pages_ocr, pages_empty,"
             " origin, state, memory_id, created_at, updated_at, tombstoned_at")
_FOLDER = ("source_id, kind, root_path, display_name, state, remote_visible, watch, created_at,"
           " last_scan_at")
_JOB = "job_id, kind, state, done, total, error, created_at, updated_at"


def _all(conn: sqlite3.Connection, sql: str, args: tuple) -> list[dict[str, Any]]:
    cursor = conn.execute(sql, args)
    out: list[dict[str, Any]] = []
    while True:
        page = cursor.fetchmany(_PAGE)
        if not page:
            return out
        out.extend(dict(row) for row in page)


def _exif(raw: Any) -> Any:
    try:
        return json.loads(raw) if isinstance(raw, str) else raw
    except ValueError:
        return {}


def _pictures(conn: sqlite3.Connection, profile_id: str,
              scopes: Callable[[list[str]], dict[str, str]]) -> list[dict[str, Any]]:
    rows = _all(conn, f"SELECT {_PICTURE} FROM media_items WHERE profile_id = ? AND kind = 'image'"
                " ORDER BY created_at, media_id", (profile_id,))
    scope_of = scopes([r["anchor_memory_id"] for r in rows if r["anchor_memory_id"]])
    for row in rows:
        row["exif"] = _exif(row.pop("exif_json"))
        row["anchor_scope"] = scope_of.get(row["anchor_memory_id"])
        row["has_thumbnail"] = bool(row["has_thumbnail"])
        row["has_original"] = bool(row["has_original"])
    return rows


def _documents(conn: sqlite3.Connection, profile_id: str) -> list[dict[str, Any]]:
    docs = _all(conn, f"SELECT {_DOCUMENT} FROM documents WHERE profile_id = ?"
                " ORDER BY created_at, document_id", (profile_id,))
    pages = _all(conn, "SELECT p.document_id, p.page_no, p.media_id, p.text_origin, p.char_count"
                 " FROM doc_pages p JOIN documents d ON d.document_id = p.document_id"
                 " WHERE d.profile_id = ? ORDER BY p.document_id, p.page_no", (profile_id,))
    by_doc: dict[str, list[dict[str, Any]]] = {}
    for page in pages:
        by_doc.setdefault(page.pop("document_id"), []).append(page)
    for doc in docs:
        doc["pages"] = by_doc.get(doc["document_id"], [])
    return docs


def _scopes_from(db: Any, profile_id: str) -> Callable[[list[str]], dict[str, str]]:
    def lookup(memory_ids: list[str]) -> dict[str, str]:
        out: dict[str, str] = {}
        for i in range(0, len(memory_ids), _PAGE):
            part = memory_ids[i:i + _PAGE]
            rows = db.execute(
                "SELECT memory_id, scope FROM memories WHERE profile_id = ? AND memory_id IN ("
                + ",".join("?" * len(part)) + ")", (profile_id, *part))
            out.update((r["memory_id"], r["scope"]) for r in rows)
        return out
    return lookup


def export_media(data_root: Path, profile_id: str, db: Any) -> dict[str, list[dict[str, Any]]] | None:
    """The profile's media records, or ``None`` when there is no ``media.db`` (or it cannot be read)."""
    path = Path(data_root) / "media.db"
    if not path.is_file():
        return None
    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=5)
        conn.row_factory = sqlite3.Row
        try:
            args = (profile_id,)
            return {
                "pictures": _pictures(conn, profile_id, _scopes_from(db, profile_id)),
                "documents": _documents(conn, profile_id),
                "folders": _all(conn, f"SELECT {_FOLDER} FROM sources WHERE profile_id = ?"
                                " ORDER BY created_at, source_id", args),
                "jobs": _all(conn, f"SELECT {_JOB} FROM jobs WHERE profile_id = ?"
                             " ORDER BY created_at, job_id", args),
            }
        finally:
            conn.close()
    except Exception as exc:  # noqa: BLE001 - the rest of the export still goes out
        logger.warning("GDPR export: media.db read failed: %s", type(exc).__name__)
        return None


__all__ = ["export_media"]
