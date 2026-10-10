# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V4 | https://qualixar.com | https://varunpratap.com

"""A second look at pictures the 4.1.25 upgrade held back from web apps.

Up to 4.1.24 a picture counted as clean when nothing had been *counted* in it, so the upgrade
holds every such picture back (``schema._revet_remote_once``). The text each picture was read to
hold is still stored in its memory (``ingest._segments``: the person's words, then the text
read from the picture after ``TEXT_MARKER``; a document page is the same text in parts headed
``[Page N]``). This reads that stored text with the save path's own full-text scan
(``memory_core.scan_sensitive``) and offers the picture to web apps only when the scan is clean.

A picture stays held back when anything makes "clean" a guess: no stored text, a memory that cannot
be found, an earlier redaction mark (something was removed before the text was stored, so the
original held a credential or personal data), a scan that cannot read the whole text, or a text
long enough that saving may have cut it (only the first 8,000 characters of a picture's text, or
1,000,000 of a page's, are kept). Nothing is ever held back that was offered, and nothing is
deleted. The verdict is recorded on each row (``remote_checked``), so the work resumes after a
restart and never reads a picture twice.
"""

from __future__ import annotations

import logging
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from superlocalmemory.media import media_db_exists, open_media_store
from superlocalmemory.media.gc import _memory_conn
from superlocalmemory.media.ingest import MAX_OCR_CHARS
from superlocalmemory.media.labels import TEXT_MARKER
from superlocalmemory.memory_core import scan_sensitive

logger = logging.getLogger(__name__)

BATCH = 100
#: Saving keeps at most this much page text (documents/pipeline.py MAX_PAGE_TEXT).
MAX_PAGE_CHARS = 1_000_000
#: A stored text this close to the cap may be the cut-off beginning of a longer one.
_NEAR_PICTURE_CAP = MAX_OCR_CHARS - 1_000
_NEAR_PAGE_CAP = MAX_PAGE_CHARS - 10_000
#: What saving leaves in place of something it removed (core/security_primitives.py, core/pii.py).
_REDACTION_MARK = re.compile(r"\[REDACTED:|\[PII:[A-Z_]+\]")
_IN_CLAUSE = 400


@dataclass
class RevetReport:
    cleared: int = 0     # now offered to web apps
    held: int = 0        # looked at, stays local-only
    waiting: int = 0     # memory not saved yet; looked at again next time
    more: bool = False   # stopped at ``max_items`` with more to look at


def _is_clean(text: str, cut_floor: int) -> bool:
    if not text.strip() or len(text) >= cut_floor:
        return False
    if _REDACTION_MARK.search(text):
        return False
    try:
        return scan_sensitive(text).clean
    except Exception:  # noqa: BLE001 - a text that could not be read in full is never clean
        return False


def _contents(conn: sqlite3.Connection, profile_id: str, memory_ids: list[str]) -> dict[str, str]:
    found: dict[str, str] = {}
    for i in range(0, len(memory_ids), _IN_CLAUSE):
        chunk = memory_ids[i:i + _IN_CLAUSE]
        rows = conn.execute(
            f"SELECT memory_id, content FROM memories WHERE profile_id = ? AND memory_id IN ({','.join('?' * len(chunk))})",
            [profile_id, *chunk]).fetchall()
        found.update({r[0]: str(r[1] or "") for r in rows})
    return found


def _picture_verdict(conn: sqlite3.Connection, row: dict[str, Any]) -> bool | None:
    if not row["anchor_memory_id"]:
        return None  # its memory was still queued when it was saved
    content = _contents(conn, row["profile_id"], [row["anchor_memory_id"]]).get(row["anchor_memory_id"])
    if content is None:
        return False
    at = content.find(TEXT_MARKER)
    if at < 0:
        return False  # no text was read from it, or it is not kept
    return _is_clean(content[at + len(TEXT_MARKER):], _NEAR_PICTURE_CAP)


def _page_verdict(store: Any, conn: sqlite3.Connection, row: dict[str, Any]) -> bool:
    ids = store.page_memory_ids(row["document_id"], row["page_no"])
    if not ids:
        return False
    found = _contents(conn, row["profile_id"], ids)
    header = f"[Page {row['page_no']}]\n"
    parts = [found.get(i) for i in ids]
    if any(p is None or not p.startswith(header) for p in parts):
        return False
    return _is_clean("\n\n".join(p[len(header):] for p in parts if p is not None), _NEAR_PAGE_CAP)


def _verdict(store: Any, conn: sqlite3.Connection, row: dict[str, Any]) -> bool | None:
    if row["kind"] == "page":
        return _page_verdict(store, conn, row)
    return _picture_verdict(conn, row)


def revet_remote_flags(*, data_root: str | Path | None = None, max_items: int | None = None,
                       batch: int = BATCH) -> RevetReport:
    """Look at held-back pictures and pages once each; at most ``max_items`` if given.

    Safe to call again at any time: a row already looked at is skipped, so an interrupted run resumes.
    """
    report = RevetReport()
    if data_root is None:
        from superlocalmemory.infra.data_root import canonical_data_root

        data_root = canonical_data_root()
    root = Path(data_root)
    if not media_db_exists(root):
        return report
    store = open_media_store(data_root=root)
    if store is None:
        return report
    conn = None
    try:
        conn = _memory_conn(root)
        if conn is None:
            return report  # nothing to read the text from; the pictures stay held back
        _look_at_all(store, conn, report, max_items, batch)
    finally:
        if conn is not None:
            conn.close()
        store.close()
    return report


def _look_at_all(store: Any, conn: sqlite3.Connection, report: RevetReport, max_items: int | None,
                 batch: int) -> None:
    after, left = "", max_items
    while left is None or left > 0:
        rows = store.revet_batch(after, batch if left is None else min(batch, left))
        if not rows:
            return
        for row in rows:
            after = row["media_id"]
            verdict = _verdict(store, conn, row)
            if verdict is None:
                report.waiting += 1
                continue
            store.finish_revet(row["media_id"], clean=verdict)
            report.cleared += int(verdict)
            report.held += int(not verdict)
        if left is not None:
            left -= len(rows)
    report.more = bool(store.revet_batch(after, 1))


__all__ = ["RevetReport", "revet_remote_flags"]
