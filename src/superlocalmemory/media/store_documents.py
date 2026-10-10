# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Document and page rows of the media store."""

from __future__ import annotations

import json
import uuid
from typing import Any, Sequence

from superlocalmemory.media.store_jobs import utc_stamp

_DOC_FIELDS = frozenset({"title", "state", "page_count", "memory_id", "fact_ids_json", "source_relpath"})
_NEW_DOC = ("document_id", "profile_id", "sha256", "title", "mime", "bytes", "source_relpath")
_ORIGINS = ("text_layer", "ocr", "none")


class DocumentsMixin:
    """Mixed into MediaStore; needs its ``_write()`` and ``_read()`` helpers."""

    def insert_document(self, **fields: Any) -> str:
        origin = fields.pop("origin", "user")
        if set(fields) != set(_NEW_DOC):
            raise ValueError(f"expected exactly {sorted(_NEW_DOC)}")
        now = utc_stamp()
        with self._write() as conn:
            conn.execute(
                f"INSERT INTO documents({','.join(_NEW_DOC)}, origin, state, created_at, updated_at)"
                f" VALUES ({','.join('?' * len(_NEW_DOC))}, ?, 'processing', ?, ?)",
                [fields[k] for k in _NEW_DOC] + [origin, now, now])
        return fields["document_id"]

    def insert_document_with_job(self, payload: dict[str, Any], **fields: Any) -> str:
        """Insert a new document and queue its job in ONE transaction; returns the job id.

        Two separate commits could leave a ``processing`` document with no job when the second
        failed, and every later upload of the same file would be answered "already saved".
        """
        origin = fields.pop("origin", "user")
        if set(fields) != set(_NEW_DOC):
            raise ValueError(f"expected exactly {sorted(_NEW_DOC)}")
        job_id, now = uuid.uuid4().hex, utc_stamp()
        job_input = {**payload, "document_id": fields["document_id"]}
        with self._write() as conn:
            conn.execute(
                f"INSERT INTO documents({','.join(_NEW_DOC)}, origin, state, created_at, updated_at)"
                f" VALUES ({','.join('?' * len(_NEW_DOC))}, ?, 'processing', ?, ?)",
                [fields[k] for k in _NEW_DOC] + [origin, now, now])
            conn.execute(
                "INSERT INTO jobs(job_id, profile_id, kind, state, done, total, payload_json, created_at, updated_at)"
                " VALUES (?, ?, 'document', 'queued', 0, 0, ?, ?, ?)",
                (job_id, fields["profile_id"], json.dumps(job_input), now, now))
        return job_id

    def get_document(self, document_id: str) -> dict[str, Any] | None:
        row = self._read().execute("SELECT * FROM documents WHERE document_id = ?", (document_id,)).fetchone()
        return dict(row) if row else None

    def find_document_by_sha(self, profile_id: str, sha256: str, *,
                             exclude_origin: str | None = None) -> dict[str, Any] | None:
        """The newest document of this profile with this content that has not been removed.

        ``exclude_origin`` skips documents made by that origin (a user's save never joins a folder's).
        """
        row = self._read().execute(
            "SELECT * FROM documents WHERE profile_id = ? AND sha256 = ? AND state != 'tombstoned'"
            " AND origin != ? ORDER BY created_at DESC LIMIT 1",
            (profile_id, sha256, exclude_origin or "")).fetchone()
        return dict(row) if row else None

    def update_document(self, document_id: str, **fields: Any) -> None:
        unknown = set(fields) - _DOC_FIELDS
        if unknown or not fields:
            raise ValueError(f"unknown document fields {sorted(unknown)}")
        sets = ", ".join(f"{k} = ?" for k in fields)
        with self._write() as conn:
            conn.execute(f"UPDATE documents SET {sets}, updated_at = ? WHERE document_id = ?",
                         [*fields.values(), utc_stamp(), document_id])

    def mark_document_ready(self, document_id: str, page_count: int) -> bool:
        """processing -> ready in one statement; False when it was removed meanwhile (stays removed)."""
        with self._write() as conn:
            return conn.execute(
                "UPDATE documents SET state = 'ready', page_count = ?, updated_at = ?"
                " WHERE document_id = ? AND state = 'processing'",
                (page_count, utc_stamp(), document_id)).rowcount == 1

    def document_bytes(self, profile_id: str) -> int:
        row = self._read().execute(
            "SELECT COALESCE(SUM(bytes), 0) FROM documents WHERE profile_id = ? AND state != 'tombstoned'",
            (profile_id,)).fetchone()
        return int(row[0])

    def put_page(self, document_id: str, page_no: int, *, media_id: str | None, memory_ids: Sequence[str],
                 fact_ids: Sequence[str], text_origin: str, char_count: int) -> None:
        if text_origin not in _ORIGINS:
            raise ValueError("unknown text origin")
        with self._write() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO doc_pages(document_id, page_no, media_id, memory_ids_json, fact_ids_json,"
                " text_origin, char_count) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (document_id, int(page_no), media_id, json.dumps(list(memory_ids)), json.dumps(list(fact_ids)),
                 text_origin, int(char_count)))

    def get_pages(self, document_id: str) -> list[dict[str, Any]]:
        rows = self._read().execute("SELECT * FROM doc_pages WHERE document_id = ? ORDER BY page_no",
                                    (document_id,)).fetchall()
        return [dict(r) for r in rows]

    def page_numbers(self, document_id: str) -> set[int]:
        rows = self._read().execute("SELECT page_no FROM doc_pages WHERE document_id = ?", (document_id,)).fetchall()
        return {int(r[0]) for r in rows}

    def refresh_document_counts(self, document_id: str) -> None:
        """Set the per-origin page counts from the page rows (so a resumed run counts every page once)."""
        with self._write() as conn:
            conn.execute(
                "UPDATE documents SET updated_at = ?,"
                " pages_text_layer = (SELECT COUNT(*) FROM doc_pages WHERE document_id = ? AND text_origin = 'text_layer'),"
                " pages_ocr = (SELECT COUNT(*) FROM doc_pages WHERE document_id = ? AND text_origin = 'ocr'),"
                " pages_empty = (SELECT COUNT(*) FROM doc_pages WHERE document_id = ? AND text_origin = 'none')"
                " WHERE document_id = ?", (utc_stamp(), document_id, document_id, document_id, document_id))

    def tombstone_document(self, document_id: str) -> bool:
        """Mark a document and its page pictures removed (kept on disk for later erasure); False if already."""
        now = utc_stamp()
        with self._write() as conn:
            changed = conn.execute(
                "UPDATE documents SET state = 'tombstoned', tombstoned_at = ?, updated_at = ?"
                " WHERE document_id = ? AND state != 'tombstoned'", (now, now, document_id)).rowcount
            if changed:
                conn.execute("UPDATE media_items SET state = 'tombstoned', tombstoned_at = ?"
                             " WHERE document_id = ? AND state = 'active'", (now, document_id))
        return bool(changed)
