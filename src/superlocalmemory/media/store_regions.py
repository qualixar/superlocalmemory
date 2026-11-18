# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Rows for region provenance: where on a picture or page each piece of read text was (WS-8)."""

from __future__ import annotations

import uuid
from typing import Any, Sequence

from superlocalmemory.media.store_jobs import utc_stamp


class RegionsMixin:
    """Mixed into MediaStore; needs its ``_write()`` and ``_read()`` helpers."""

    def put_regions(self, media_id: str, profile_id: str, page_no: int, rows: Sequence[dict[str, Any]],
                    *, engine: str = "") -> int:
        """Replace one item's regions with ``rows`` (each ``bbox`` four fractions, ``snippet``, ``label``)."""
        now = utc_stamp()
        with self._write() as conn:
            conn.execute("DELETE FROM media_regions WHERE media_id = ? AND page_no = ?", (media_id, int(page_no)))
            for row in rows:
                x0, y0, x1, y1 = row["bbox"]
                conn.execute(
                    "INSERT INTO media_regions(region_id, media_id, profile_id, page_no, x0, y0, x1, y1, engine,"
                    " label, snippet, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (uuid.uuid4().hex, media_id, profile_id, int(page_no), x0, y0, x1, y1, engine,
                     str(row.get("label") or ""), str(row.get("snippet") or ""), now))
        return len(rows)
