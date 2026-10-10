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
    """``raw`` as a view name; ``""`` (a local caller) only when nothing was sent."""
    value = str(raw or "").strip().lower()
    if not value:
        return ""
    # A value this version does not know is still a remote caller: the strictest view.
    return value if value in _VIEWS else REMOTE


class VettedMedia:
    """Which pictures and pages of one profile may be shown, asked for only the ones in play.

    ``token in vetted`` (see ``visibility.media_token``) looks up that picture, page or document
    and nothing else; :meth:`prime` does it for a batch in one connection. A lookup that fails
    answers no, so nothing is shown.
    """

    def __init__(self, media_db: Path | None, profile_id: str) -> None:
        self._media_db = media_db
        self._profile_id = profile_id
        self._known: dict[str, bool] = {}

    def prime(self, tokens: Any) -> None:
        todo = [t for t in dict.fromkeys(tokens) if t and t not in self._known]
        if not todo:
            return
        if self._media_db is None:
            self._known.update(dict.fromkeys(todo, False))
            return
        try:
            conn = sqlite3.connect(f"file:{self._media_db}?mode=ro", uri=True, timeout=5)
            try:
                for token in todo:
                    self._known[token] = self._ask(conn, token)
            finally:
                conn.close()
        except Exception as exc:  # noqa: BLE001 - fail closed: nothing is vetted
            logger.warning("media visibility lookup failed, hiding media (%s)", type(exc).__name__)
            self._known.update(dict.fromkeys(todo, False))

    def _ask(self, conn: sqlite3.Connection, token: str) -> bool:
        kind, _, rest = token.partition(":")
        pid = self._profile_id
        if kind == "m":
            return conn.execute(
                "SELECT 1 FROM media_items WHERE media_id = ? AND profile_id = ? AND remote_ok"
                " AND kind = 'image'", (rest, pid)).fetchone() is not None
        if kind == "p":
            doc, _, page = rest.rpartition(":")
            return conn.execute(
                "SELECT 1 FROM media_items WHERE document_id = ? AND page_no = ? AND profile_id = ?"
                " AND remote_ok", (doc, page, pid)).fetchone() is not None
        if kind == "d":
            if conn.execute("SELECT 1 FROM documents WHERE document_id = ? AND profile_id = ?",
                            (rest, pid)).fetchone() is None:
                return False
            return conn.execute(
                "SELECT 1 FROM media_items i JOIN doc_pages p"
                " ON p.document_id = i.document_id AND p.page_no = i.page_no"
                " WHERE i.document_id = ? AND i.profile_id = ? AND i.remote_ok = 0"
                " AND p.text_origin != 'none' LIMIT 1", (rest, pid)).fetchone() is None
        return False

    def __contains__(self, token: object) -> bool:
        if not isinstance(token, str):
            return False
        self.prime([token])
        return self._known.get(token, False)


def _vetted(db: Any, profile_id: str) -> VettedMedia:
    """The pictures and pages of ``profile_id`` that may be shown, looked up as they come up."""
    path = getattr(db, "db_path", None)
    media_db = Path(path).parent / "media.db" if isinstance(path, (str, Path)) else None
    return VettedMedia(media_db if media_db is not None and media_db.is_file() else None, profile_id)


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


__all__ = ["REMOTE", "REMOTE_MEDIA", "VIEW_PARAM", "VettedMedia", "context_for", "hidden_among", "parse_view"]
