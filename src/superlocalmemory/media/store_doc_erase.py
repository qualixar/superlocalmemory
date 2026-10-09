# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Removing documents and their pages for good, after the memories made from them are erased."""

from __future__ import annotations

import json
from typing import Any, Sequence

_CHUNK = 400


def _marks(n: int) -> str:
    return ",".join("?" * n)


def _chunks(values: Sequence[str]):
    ids = sorted({str(v) for v in values if v})
    for i in range(0, len(ids), _CHUNK):
        yield ids[i:i + _CHUNK]


class DocumentEraseMixin:
    """Mixed into MediaStore; needs ``_write()``, ``_read()`` and the item erase mixin."""

    def _pages_of_memories(self, profile_id: str, memory_ids: Sequence[str]) -> list[dict[str, Any]]:
        found: dict[tuple[str, int], dict[str, Any]] = {}
        for chunk in _chunks(memory_ids):
            rows = self._read().execute(
                "SELECT DISTINCT p.document_id, p.page_no, p.media_id, p.memory_ids_json, p.fact_ids_json"
                " FROM doc_pages p JOIN documents d ON d.document_id = p.document_id,"
                f" json_each(p.memory_ids_json) j WHERE d.profile_id = ? AND j.value IN ({_marks(len(chunk))})",
                [profile_id, *chunk]).fetchall()
            for r in rows:
                found[(r[0], r[1])] = dict(r)
        return list(found.values())

    def page_item_ids_for_memories(self, profile_id: str, memory_ids: Sequence[str]) -> list[str]:
        """Page pictures of one profile that show a page whose memory is among ``memory_ids``."""
        return sorted({r["media_id"] for r in self._pages_of_memories(profile_id, memory_ids) if r["media_id"]})

    def drop_page_memories(self, profile_id: str, memory_ids: Sequence[str], fact_ids: Sequence[str]) -> list[str]:
        """Forget erased memories and facts in the page and document rows (a page left with no memory goes); returns the documents touched."""
        erased_m, erased_f = {str(m) for m in memory_ids}, {str(f) for f in fact_ids}
        touched = {r["document_id"] for r in self._pages_of_memories(profile_id, memory_ids)}
        with self._write() as conn:
            for page in self._pages_of_memories(profile_id, memory_ids):
                keep_m = [m for m in json.loads(page["memory_ids_json"]) if m not in erased_m]
                keep_f = [f for f in json.loads(page["fact_ids_json"]) if f not in erased_f]
                where = (page["document_id"], page["page_no"])
                if keep_m:
                    conn.execute("UPDATE doc_pages SET memory_ids_json = ?, fact_ids_json = ? "
                                 "WHERE document_id = ? AND page_no = ?", (json.dumps(keep_m), json.dumps(keep_f), *where))
                else:
                    conn.execute("DELETE FROM doc_pages WHERE document_id = ? AND page_no = ?", where)
            for chunk in _chunks(memory_ids):
                rows = conn.execute(
                    f"SELECT document_id FROM documents WHERE profile_id = ? AND memory_id IN ({_marks(len(chunk))})",
                    [profile_id, *chunk]).fetchall()
                for r in rows:
                    touched.add(r[0])
                    conn.execute("UPDATE documents SET memory_id = NULL, fact_ids_json = '[]' WHERE document_id = ?",
                                 (r[0],))
        return sorted(touched)

    def documents_left_empty(self, document_ids: Sequence[str]) -> list[str]:
        """Of these documents, those no longer processing that keep no page memory at all."""
        out: list[str] = []
        for doc_id in sorted(set(document_ids)):
            doc = self.get_document(doc_id)
            if not doc or doc["state"] == "processing":
                continue
            left = self._read().execute(
                "SELECT 1 FROM doc_pages WHERE document_id = ? AND memory_ids_json != '[]' LIMIT 1",
                (doc_id,)).fetchone()
            if left is None:
                out.append(doc_id)
        return out

    def document_item_ids(self, document_ids: Sequence[str]) -> list[str]:
        found: list[str] = []
        for chunk in _chunks(document_ids):
            rows = self._read().execute(
                f"SELECT media_id FROM media_items WHERE document_id IN ({_marks(len(chunk))})", chunk).fetchall()
            found.extend(r[0] for r in rows)
        return found

    def all_document_ids(self, profile_id: str) -> list[str]:
        rows = self._read().execute("SELECT document_id FROM documents WHERE profile_id = ?", (profile_id,))
        return [r[0] for r in rows.fetchall()]

    def delete_documents(self, document_ids: Sequence[str]) -> list[dict[str, Any]]:
        """Delete the document and page rows; returns each document's file address."""
        removed: list[dict[str, Any]] = []
        with self._wlock:
            self._w.execute("PRAGMA secure_delete=ON")
            with self._write() as conn:
                for chunk in _chunks(document_ids):
                    where = f"document_id IN ({_marks(len(chunk))})"
                    rows = conn.execute(f"SELECT sha256, source_relpath FROM documents WHERE {where}", chunk).fetchall()
                    removed.extend({"sha256": r[0], "source_relpath": r[1]} for r in rows)
                    conn.execute(f"DELETE FROM doc_pages WHERE {where}", chunk)
                    conn.execute(f"DELETE FROM documents WHERE {where}", chunk)
            try:
                self._w.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchall()
            except Exception:  # noqa: BLE001 - a busy reader only delays the truncate
                pass
        return removed

    def document_file_in_use(self, source_relpath: str | None) -> bool:
        return bool(source_relpath and self._read().execute(
            "SELECT 1 FROM documents WHERE source_relpath = ? LIMIT 1", (source_relpath,)).fetchone())
