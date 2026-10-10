# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V4 | https://qualixar.com | https://varunpratap.com

"""Rows for the second look at pictures the 4.1.25 upgrade held back (see ``media/revet.py``)."""

from __future__ import annotations

import json
from typing import Any


class RevetMixin:
    """Mixed into MediaStore; needs its ``_write()`` and ``_read()`` helpers."""

    def revet_batch(self, after: str, limit: int) -> list[dict[str, Any]]:
        """Held-back, not yet looked at pictures and pages, in id order, after ``after``."""
        rows = self._read().execute(
            "SELECT media_id, profile_id, kind, anchor_memory_id, document_id, page_no FROM media_items"
            " WHERE remote_ok = 0 AND remote_checked = 0 AND state = 'active' AND media_id > ?"
            " ORDER BY media_id LIMIT ?", (after, int(limit))).fetchall()
        return [dict(r) for r in rows]

    def page_memory_ids(self, document_id: str, page_no: int) -> list[str]:
        """The memories that hold one page's text, in reading order."""
        row = self._read().execute(
            "SELECT memory_ids_json FROM doc_pages WHERE document_id = ? AND page_no = ?",
            (document_id, int(page_no))).fetchone()
        try:
            found = json.loads(row[0]) if row else []
        except ValueError:
            found = []
        return [str(m) for m in found] if isinstance(found, list) else []

    def finish_revet(self, media_id: str, *, clean: bool) -> None:
        """Record the verdict. A picture is offered to web apps only when ``clean``; this never holds one back."""
        with self._write() as conn:
            conn.execute(
                "UPDATE media_items SET remote_checked = 1, remote_ok = CASE WHEN ? THEN 1 ELSE remote_ok END"
                " WHERE media_id = ? AND remote_ok = 0", (1 if clean else 0, media_id))
