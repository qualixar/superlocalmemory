# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V4 | https://qualixar.com | https://varunpratap.com

"""What a remote caller may see of pictures, documents and connected folders.

A caller on this computer sees everything its profile and scope allow; nothing
here touches that path. A remote caller (a remote key) gets a
``VisibilityContext`` for the request:

* never a fact that came from a connected folder (folders are not shown to
  remote apps in this release);
* without the media permission, no picture or document page either;
* with it, only the pictures and pages whose text held no secret or personal
  data when it was read (``remote_ok`` in media.db), one lookup per request.
  A picture or page with no such record is not shown.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Any

from superlocalmemory.retrieval import visibility
from superlocalmemory.retrieval.visibility import VisibilityContext

logger = logging.getLogger(__name__)

#: Name of the request parameter that tells the daemon how the caller came in.
VIEW_PARAM = "caller_view"
REMOTE = "remote"
REMOTE_MEDIA = "remote_media"
_VIEWS = (REMOTE, REMOTE_MEDIA)


def parse_view(raw: Any) -> str:
    """``raw`` as a view name, or ``""`` (a local caller)."""
    value = str(raw or "").strip().lower()
    return value if value in _VIEWS else ""


def _vetted(db: Any, profile_id: str) -> frozenset[str]:
    """Tokens (see ``visibility.media_token``) of this profile's pictures and pages that may be shown."""
    path = getattr(db, "db_path", None)
    if not isinstance(path, (str, Path)):
        return frozenset()
    media_db = Path(path).parent / "media.db"
    if not media_db.is_file():
        return frozenset()
    try:
        conn = sqlite3.connect(f"file:{media_db}?mode=ro", uri=True, timeout=5)
        try:
            ok: set[str] = set()
            for media_id, document_id, page_no, remote_ok, kind in conn.execute(
                    "SELECT media_id, document_id, page_no, remote_ok, kind FROM media_items"
                    " WHERE profile_id = ?", (profile_id,)):
                if remote_ok and kind == "image":
                    ok.add(f"m:{media_id}")
                elif remote_ok and document_id is not None:
                    ok.add(f"p:{document_id}:{page_no}")
            held_back = {r[0] for r in conn.execute(
                "SELECT DISTINCT i.document_id FROM media_items i JOIN doc_pages p"
                " ON p.document_id = i.document_id AND p.page_no = i.page_no"
                " WHERE i.profile_id = ? AND i.remote_ok = 0 AND p.text_origin != 'none'",
                (profile_id,))}
            ok |= {f"d:{r[0]}" for r in conn.execute(
                "SELECT document_id FROM documents WHERE profile_id = ?", (profile_id,))
                if r[0] not in held_back}
            return frozenset(ok)
        finally:
            conn.close()
    except Exception as exc:  # noqa: BLE001 - fail closed: nothing is vetted
        logger.warning("media visibility lookup failed, hiding media (%s)", type(exc).__name__)
        return frozenset()


def context_for(view: str, db: Any, profile_id: str) -> VisibilityContext | None:
    """The context for ``view`` in ``profile_id``, or ``None`` for a local caller."""
    if view == REMOTE:
        return VisibilityContext(hide_media=True, hide_sources=True)
    if view == REMOTE_MEDIA:
        return VisibilityContext(hide_sources=True, vetted_media=_vetted(db, profile_id))
    return None


def hidden_among(view: str, db: Any, profile_id: str, fact_ids: Any) -> set[str]:
    """Which of ``fact_ids`` the ``view`` hides (none for a local caller; all if the lookup fails)."""
    ctx = context_for(view, db, profile_id)
    ids = [i for i in fact_ids if i]
    if ctx is None or not ids:
        return set()
    with visibility.use(ctx):
        return visibility.hidden_among(db, profile_id, ids)


__all__ = ["REMOTE", "REMOTE_MEDIA", "VIEW_PARAM", "context_for", "hidden_among", "parse_view"]
