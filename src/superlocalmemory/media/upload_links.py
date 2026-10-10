# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V4 | https://qualixar.com | https://varunpratap.com

"""One-time upload links: how a web app hands this computer a picture or a PDF.

A web app cannot type a file into a tool call. Instead it asks for a link; the
person opens it in any browser, picks the file, and the gateway streams it here
in small chunks over the connection that is already open. This module is the
laptop's half: it mints the token, keeps only its hash, accepts the chunks in
order into a private scratch file, and decides when a link is spent.

The token is the only capability. It is 32 random bytes, bound to one
connection, one key and one profile, valid for ten minutes (twenty once the
upload has started) and usable for one saved file. Everything the gateway says
about size or type is ignored: limits and the file's first bytes are checked
here. Nothing in this module logs content, tokens or paths.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import secrets
import sqlite3
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from superlocalmemory.media import files

LINK_TTL_S = 600
STARTED_TTL_S = 1200
MAX_OPEN_PER_CONNECTION = 3
MAX_UPLOADS_PER_DAY = 20
MAX_ATTEMPTS = 3
MAX_CHUNK_BYTES = 700_000
MAX_NOTE_CHARS = 2000
IMAGE_MAX_BYTES = 25 * 1024 * 1024
DAY_S = 86_400
RETENTION_S = DAY_S + 3_600
STRAY_FILE_S = 3_600
KINDS = ("image", "document")
DB_NAME = "uploads.db"
#: Where a link opens: the gateway's public MCP host. The page is ``<base>/u/<connection>/<token>``.
UPLOAD_BASE_URL = "https://mcp.superlocalmemory.com"

_TOKEN = re.compile(r"[A-Za-z0-9_-]{43}")
_DOMAIN = b"superlocalmemory-upload-link-v1\0"
_DDL = """CREATE TABLE IF NOT EXISTS upload_links (
  upload_id TEXT PRIMARY KEY, token_hash TEXT NOT NULL UNIQUE,
  connection_id TEXT NOT NULL, key_id TEXT NOT NULL, profile_id TEXT NOT NULL,
  kind TEXT NOT NULL CHECK (kind IN ('image','document')), note TEXT NOT NULL,
  max_bytes INTEGER NOT NULL,
  state TEXT NOT NULL CHECK (state IN ('open','receiving','finishing','done','failed')),
  total INTEGER NOT NULL DEFAULT 0, received INTEGER NOT NULL DEFAULT 0,
  next_index INTEGER NOT NULL DEFAULT 0, attempts INTEGER NOT NULL DEFAULT 0,
  created_at INTEGER NOT NULL, expires_at INTEGER NOT NULL, started_at INTEGER,
  result_json TEXT NOT NULL DEFAULT '')"""

_MESSAGES = {
    "invalid_kind": "An upload link is for an image or a document.",
    "note_too_long": "The note for an upload link is limited to 2000 characters.",
    "too_many_open": "There are already three upload links waiting. Use one, or wait ten minutes.",
    "invalid_link": "This upload link is not valid.",
    "expired": "This upload link has expired. Ask the app for a new one.",
    "used": "This upload link has already been used. Ask the app for a new one.",
    "empty": "That file is empty.",
    "chunk_too_large": "That piece of the file is too large.",
    "too_large": "That file is too large for this link.",
    "wrong_type": "That is not the kind of file this link is for.",
    "bad_order": "The file arrived out of order. Start the upload again.",
    "size_changed": "The file changed size while it was being sent. Start the upload again.",
    "too_much_data": "More data arrived than the file's size. Start the upload again.",
    "too_many_attempts": "This link has been tried too many times. Ask the app for a new one.",
    "daily_limit": "The limit of 20 uploads a day was reached. Try again tomorrow.",
    "not_started": "No file has been sent on this link yet.",
    "incomplete": "The whole file did not arrive. Start the upload again.",
    "not_allowed": "This computer no longer lets this app add files. Ask the owner to allow it again.",
    "invalid_request": "That upload request was not understood.",
}


class UploadError(Exception):
    """A refusal with a stable ``code`` and a plain ``message`` safe to show a person."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code
        self.message = _MESSAGES.get(code, "The upload could not be completed.")


@dataclass(frozen=True)
class MintedLink:
    token: str
    upload_id: str
    expires_at: int
    max_bytes: int


@dataclass(frozen=True)
class LinkInfo:
    kind: str
    max_bytes: int
    expires_at: int


@dataclass(frozen=True)
class UploadRow:
    upload_id: str
    connection_id: str
    key_id: str
    profile_id: str
    kind: str
    note: str
    max_bytes: int
    state: str
    total: int
    received: int
    next_index: int
    attempts: int
    created_at: int
    expires_at: int
    started_at: int | None
    result_json: str


@dataclass(frozen=True)
class FinishPlan:
    """``run``: save it now. ``working``: a save is already running. ``result``: it already ended."""

    action: str
    row: UploadRow
    result: dict[str, Any] | None = None


def max_bytes_for(kind: str) -> int:
    if kind == "image":
        return IMAGE_MAX_BYTES
    from superlocalmemory.documents.submit import _max_bytes

    return _max_bytes()


def looks_like(kind: str, head: bytes) -> bool:
    """Whether the first bytes are a PNG, JPEG or WebP picture, or a PDF, as ``kind`` says."""
    if kind == "document":
        return head.startswith(b"%PDF-")
    return (head.startswith(b"\x89PNG\r\n\x1a\n") or head.startswith(b"\xff\xd8\xff")
            or (head[:4] == b"RIFF" and head[8:12] == b"WEBP"))


def _hash(token: str) -> str:
    return hashlib.sha256(_DOMAIN + token.encode("ascii")).hexdigest()


def _deadline(row: UploadRow) -> int:
    started = row.started_at
    return row.expires_at if started is None else max(row.expires_at, started + STARTED_TTL_S)


class UploadLinks:
    def __init__(self, data_root: str | Path, *, clock: Callable[[], float] = time.time) -> None:
        self._root = Path(data_root)
        self._clock = clock
        self._lock = threading.RLock()
        self._ready = False

    # -- storage ------------------------------------------------------------

    @property
    def temp_dir(self) -> Path:
        return files.tmp_dir(self._root)

    def temp_path(self, upload_id: str) -> Path:
        if not re.fullmatch(r"[0-9a-f]{32}", upload_id):
            raise ValueError("invalid upload id")
        return self.temp_dir / f"upload-{upload_id}.part"

    def _now(self) -> int:
        return int(self._clock())

    @contextlib.contextmanager
    def _tx(self) -> Iterator[sqlite3.Connection]:
        path = self._root / "media" / DB_NAME
        with self._lock:
            if not self._ready:
                path.parent.mkdir(parents=True, exist_ok=True)
                if not path.exists():
                    os.close(os.open(path, os.O_RDWR | os.O_CREAT, 0o600))
                self._ready = True
            conn = sqlite3.connect(path, timeout=5, isolation_level=None)
            conn.row_factory = sqlite3.Row
            try:
                conn.execute(_DDL)
                conn.execute("BEGIN IMMEDIATE")
                try:
                    yield conn
                    conn.execute("COMMIT")
                except BaseException:
                    conn.execute("ROLLBACK")
                    raise
            finally:
                conn.close()

    @staticmethod
    def _row(found: sqlite3.Row | None) -> UploadRow | None:
        return None if found is None else UploadRow(**{k: found[k] for k in UploadRow.__dataclass_fields__})

    def _by_token(self, conn: sqlite3.Connection, token: str, connection_id: str) -> UploadRow:
        row = None
        if isinstance(token, str) and _TOKEN.fullmatch(token):
            row = self._row(conn.execute("SELECT * FROM upload_links WHERE token_hash = ?",
                                         (_hash(token),)).fetchone())
        if row is None or row.connection_id != connection_id:
            raise UploadError("invalid_link")  # unknown and foreign look the same
        return row

    def _live(self, row: UploadRow, states: tuple[str, ...] = ("open", "receiving")) -> None:
        if row.state not in states:
            raise UploadError("used")
        if self._now() > _deadline(row):
            raise UploadError("expired")

    # -- minting ------------------------------------------------------------

    def mint(self, connection_id: str, key_id: str, profile_id: str, kind: str, note: str) -> MintedLink:
        if kind not in KINDS:
            raise UploadError("invalid_kind")
        if len(note or "") > MAX_NOTE_CHARS:
            raise UploadError("note_too_long")
        now = self._now()
        with self._tx() as conn:
            open_now = conn.execute(
                "SELECT COUNT(*) FROM upload_links WHERE connection_id = ? "
                "AND state IN ('open','receiving') AND MAX(expires_at, COALESCE(started_at + ?, 0)) >= ?",
                (connection_id, STARTED_TTL_S, now)).fetchone()[0]
            if open_now >= MAX_OPEN_PER_CONNECTION:
                raise UploadError("too_many_open")
            token, upload_id = secrets.token_urlsafe(32), secrets.token_hex(16)
            limit = max_bytes_for(kind)
            conn.execute(
                "INSERT INTO upload_links (upload_id, token_hash, connection_id, key_id, profile_id, kind, note, "
                "max_bytes, state, created_at, expires_at) VALUES (?,?,?,?,?,?,?,?,'open',?,?)",
                (upload_id, _hash(token), connection_id, key_id, profile_id, kind, note or "", limit,
                 now, now + LINK_TTL_S))
        return MintedLink(token, upload_id, now + LINK_TTL_S, limit)

    # -- reading ------------------------------------------------------------

    def find(self, token: str, connection_id: str) -> UploadRow:
        with self._tx() as conn:
            return self._by_token(conn, token, connection_id)

    def get(self, upload_id: str) -> UploadRow | None:
        with self._tx() as conn:
            return self._row(conn.execute("SELECT * FROM upload_links WHERE upload_id = ?",
                                          (upload_id,)).fetchone())

    def info(self, token: str, connection_id: str) -> LinkInfo:
        row = self.find(token, connection_id)
        self._live(row)
        return LinkInfo(row.kind, row.max_bytes, _deadline(row))

    # -- receiving ----------------------------------------------------------

    def accept_chunk(self, token: str, connection_id: str, index: int, total: int, data: bytes) -> int:
        """Append one chunk; returns the bytes held so far. Chunk 0 (re)starts the upload."""
        if not data:
            raise UploadError("empty")
        if len(data) > MAX_CHUNK_BYTES:
            raise UploadError("chunk_too_large")
        with self._tx() as conn:
            row = self._by_token(conn, token, connection_id)
            self._live(row)
            if not 1 <= total:
                raise UploadError("empty")
            if total > row.max_bytes:
                raise UploadError("too_large")
            if index == 0:
                return self._start(conn, row, total, data)
            return self._append(conn, row, index, total, data)

    def _start(self, conn: sqlite3.Connection, row: UploadRow, total: int, data: bytes) -> int:
        if row.attempts >= MAX_ATTEMPTS:
            raise UploadError("too_many_attempts")
        now = self._now()
        if row.state == "open":
            started = conn.execute(
                "SELECT COUNT(*) FROM upload_links WHERE connection_id = ? AND started_at >= ?",
                (row.connection_id, now - DAY_S)).fetchone()[0]
            if started >= MAX_UPLOADS_PER_DAY:
                raise UploadError("daily_limit")
        if not looks_like(row.kind, data[:16]):
            raise UploadError("wrong_type")
        if len(data) > total:
            raise UploadError("too_much_data")
        path = self.temp_path(row.upload_id)
        path.unlink(missing_ok=True)
        self._write(path, data, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
        conn.execute(
            "UPDATE upload_links SET state='receiving', total=?, received=?, next_index=1, attempts=attempts+1, "
            "started_at=COALESCE(started_at, ?) WHERE upload_id=?", (total, len(data), now, row.upload_id))
        return len(data)

    def _append(self, conn: sqlite3.Connection, row: UploadRow, index: int, total: int, data: bytes) -> int:
        if row.state != "receiving" or index != row.next_index:
            raise UploadError("bad_order")
        if total != row.total:
            raise UploadError("size_changed")
        if row.received + len(data) > row.total:
            raise UploadError("too_much_data")
        path = self.temp_path(row.upload_id)
        try:
            if path.stat().st_size != row.received:
                raise UploadError("bad_order")
        except OSError:
            raise UploadError("bad_order") from None
        self._write(path, data, os.O_WRONLY | os.O_APPEND)
        received = row.received + len(data)
        conn.execute("UPDATE upload_links SET received=?, next_index=next_index+1 WHERE upload_id=?",
                     (received, row.upload_id))
        return received

    @staticmethod
    def _write(path: Path, data: bytes, flags: int) -> None:
        fd = os.open(path, flags | getattr(os, "O_NOFOLLOW", 0), 0o600)
        with os.fdopen(fd, "ab" if flags & os.O_APPEND else "wb") as out:
            out.write(data)

    # -- finishing ----------------------------------------------------------

    def begin_finish(self, token: str, connection_id: str) -> FinishPlan:
        with self._tx() as conn:
            row = self._by_token(conn, token, connection_id)
            if row.state in ("done", "failed"):
                return FinishPlan("result", row, json.loads(row.result_json or "{}"))
            if row.state == "finishing":
                return FinishPlan("working", row)
            if row.state == "open":
                raise UploadError("not_started")
            self._live(row)
            if row.received != row.total or row.total < 1:
                raise UploadError("incomplete")
            conn.execute("UPDATE upload_links SET state='finishing' WHERE upload_id=?", (row.upload_id,))
            return FinishPlan("run", row)

    def _end(self, upload_id: str, state: str, result: dict[str, Any]) -> None:
        with self._tx() as conn:
            conn.execute("UPDATE upload_links SET state=?, result_json=? WHERE upload_id=?",
                         (state, json.dumps(result), upload_id))
        self.temp_path(upload_id).unlink(missing_ok=True)

    def finish_done(self, upload_id: str, result: dict[str, Any]) -> None:
        self._end(upload_id, "done", result)

    def finish_failed(self, upload_id: str, result: dict[str, Any]) -> None:
        self._end(upload_id, "failed", result)

    def finish_retry(self, upload_id: str) -> None:
        """A save that may work later (the picture tools are starting): keep the bytes, reopen the link."""
        with self._tx() as conn:
            conn.execute("UPDATE upload_links SET state='receiving' WHERE upload_id=? AND state='finishing'",
                         (upload_id,))

    # -- housekeeping -------------------------------------------------------

    def cleanup(self) -> int:
        """Remove scratch files of spent or abandoned uploads and old rows; returns files removed."""
        now = self._now()
        removed = 0
        with self._tx() as conn:
            rows = [self._row(r) for r in conn.execute("SELECT * FROM upload_links").fetchall()]
            for row in rows:
                gone = row.state in ("done", "failed") or (
                    row.state in ("open", "receiving") and now > _deadline(row))
                if gone and row.state in ("open", "receiving"):
                    conn.execute("UPDATE upload_links SET state='failed', result_json=? WHERE upload_id=?",
                                 (json.dumps({"ok": False, "code": "expired",
                                              "message": _MESSAGES["expired"]}), row.upload_id))
                if gone:
                    removed += self._unlink(self.temp_path(row.upload_id))
            conn.execute("DELETE FROM upload_links WHERE created_at < ?", (now - RETENTION_S,))
        for entry in self.temp_dir.glob("upload-*.part"):
            with contextlib.suppress(OSError):
                if entry.lstat().st_mtime < now - STRAY_FILE_S:
                    removed += self._unlink(entry)
        return removed

    @staticmethod
    def _unlink(path: Path) -> int:
        try:
            path.unlink()
            return 1
        except OSError:
            return 0


_SHARED: dict[str, UploadLinks] = {}
_SHARED_LOCK = threading.Lock()


def default_links() -> UploadLinks:
    """The one store for this computer's data folder (one lock for every caller in this process)."""
    from superlocalmemory.infra.data_root import canonical_data_root

    root = str(canonical_data_root())
    with _SHARED_LOCK:
        if root not in _SHARED:
            _SHARED[root] = UploadLinks(root)
        return _SHARED[root]
