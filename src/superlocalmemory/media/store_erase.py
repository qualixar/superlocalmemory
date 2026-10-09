# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Removing items for good: rows, thumbnails and vectors, with freed pages overwritten."""

from __future__ import annotations

from typing import Any, Sequence

_CHUNK = 400


def _marks(n: int) -> str:
    return ",".join("?" * n)


class EraseMixin:
    """Mixed into MediaStore; needs its ``_write()``, ``_read()`` and ``_drop_vector_rows``."""

    def item_ids_for_anchors(self, profile_id: str, memory_ids: Sequence[str]) -> list[str]:
        """Every item (any state) of one profile whose anchor memory is among ``memory_ids``."""
        ids = sorted({str(m) for m in memory_ids if m})
        found: list[str] = []
        for i in range(0, len(ids), _CHUNK):
            chunk = ids[i:i + _CHUNK]
            rows = self._read().execute(
                f"SELECT media_id FROM media_items WHERE profile_id = ? AND anchor_memory_id IN ({_marks(len(chunk))})",
                [profile_id, *chunk]).fetchall()
            found.extend(r[0] for r in rows)
        return found

    def active_anchor_ids(self, memory_ids: Sequence[str]) -> set[str]:
        """Those of ``memory_ids`` that anchor at least one item still in the ``active`` state."""
        ids = sorted({str(m) for m in memory_ids if m})
        alive: set[str] = set()
        for i in range(0, len(ids), _CHUNK):
            chunk = ids[i:i + _CHUNK]
            rows = self._read().execute(
                "SELECT DISTINCT anchor_memory_id FROM media_items WHERE state = 'active'"
                f" AND anchor_memory_id IN ({_marks(len(chunk))})", chunk).fetchall()
            alive.update(r[0] for r in rows)
        return alive

    def anchors(self, profile_id: str) -> dict[str, str | None]:
        """media_id -> anchor memory id for every item of the profile, in any state."""
        rows = self._read().execute(
            "SELECT media_id, anchor_memory_id FROM media_items WHERE profile_id = ?", (profile_id,))
        return {r[0]: r[1] for r in rows.fetchall()}

    def all_item_ids(self, profile_id: str) -> list[str]:
        rows = self._read().execute("SELECT media_id FROM media_items WHERE profile_id = ?", (profile_id,))
        return [r[0] for r in rows.fetchall()]

    def erase_items(self, media_ids: Sequence[str]) -> list[dict[str, Any]]:
        """Delete items with their vectors and thumbnails in one transaction.

        Returns ``{"stored_sha256", "original_relpath"}`` of each removed item so the
        caller can decide which files and cached results are no longer needed.
        """
        removed: list[dict[str, Any]] = []
        ids = sorted(set(media_ids))
        if not ids:
            return removed
        with self._wlock:
            self._w.execute("PRAGMA secure_delete=ON")
            with self._write() as conn:
                for i in range(0, len(ids), _CHUNK):
                    chunk = ids[i:i + _CHUNK]
                    where = f"media_id IN ({_marks(len(chunk))})"
                    rows = conn.execute(
                        f"SELECT stored_sha256, original_relpath FROM media_items WHERE {where}", chunk).fetchall()
                    removed.extend({"stored_sha256": r[0], "original_relpath": r[1]} for r in rows)
                    self._drop_vector_rows(conn, where, chunk)
                    conn.execute(f"UPDATE doc_pages SET media_id = NULL WHERE {where}", chunk)
                    conn.execute(f"DELETE FROM media_items WHERE {where}", chunk)
            try:
                self._w.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchall()
            except Exception:  # noqa: BLE001 - a busy reader only delays the truncate
                pass
        return removed

    def file_in_use(self, stored_sha256: str | None, original_relpath: str | None) -> bool:
        """Whether any remaining item or document (any profile, any state) still points at this file."""
        conn = self._read()
        if stored_sha256 and conn.execute(
                "SELECT 1 FROM media_items WHERE stored_sha256 = ? LIMIT 1", (stored_sha256,)).fetchone():
            return True
        if original_relpath and conn.execute(
                "SELECT 1 FROM media_items WHERE original_relpath = ? LIMIT 1", (original_relpath,)).fetchone():
            return True
        return self.document_file_in_use(original_relpath)

    def known_relpaths(self) -> set[str]:
        rows = self._read().execute(
            "SELECT original_relpath FROM media_items WHERE original_relpath IS NOT NULL").fetchall()
        return {r[0] for r in rows}

    def items_by_id(self, media_ids: Sequence[str]) -> list[dict[str, Any]]:
        """The file addresses of some items, read before they are erased."""
        ids = sorted(set(media_ids))
        out: list[dict[str, Any]] = []
        for i in range(0, len(ids), _CHUNK):
            chunk = ids[i:i + _CHUNK]
            rows = self._read().execute(
                "SELECT stored_sha256, original_relpath FROM media_items "
                f"WHERE media_id IN ({_marks(len(chunk))})", chunk).fetchall()
            out.extend({"stored_sha256": r[0], "original_relpath": r[1]} for r in rows)
        return out
