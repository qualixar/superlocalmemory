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
connection, one key, one profile and the authorization (the web app) that asked
for it, valid for ten minutes (twenty once the
upload has started) and usable for one saved file. Everything the gateway says
about size or type is ignored: limits and the file's first bytes are checked
here. Nothing in this module logs content, tokens or paths.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
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
HOUSEKEEPING_S = 3_600
#: A second upload may take over a link only after this long without a byte from the first.
IDLE_TAKEOVER_S = 60
#: The daemon waits at most this long for a save (``daemon_finisher``); a longer "finishing" never ended.
FINISHER_TIMEOUT_S = 300
#: Where a link opens: the gateway's public MCP host. The page is ``<base>/u/<connection>/<token>``.
UPLOAD_BASE_URL = "https://mcp.superlocalmemory.com"

logger = logging.getLogger(__name__)
_TOKEN = re.compile(r"[A-Za-z0-9_-]{43}")
_NONCE = re.compile(r"[A-Za-z0-9_-]{22}")
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
  result_json TEXT NOT NULL DEFAULT '', nonce TEXT, touched_at INTEGER,
  authorization_id TEXT NOT NULL DEFAULT '')"""
#: Added after the first release of the table: databases made before it get the column on first use.
_ADD_AUTHORIZATION = "ALTER TABLE upload_links ADD COLUMN authorization_id TEXT NOT NULL DEFAULT ''"

#: ``PRAGMA user_version`` once the links without an app have been ended (see ``UploadLinks._migrate``).
_MIGRATION_DONE = 1

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
    "rate_limited": "Too many invalid upload links were tried. Wait ten minutes and try again.",
    "in_progress": "Another upload is already using this link.",
    "interrupted": "The save was interrupted. Ask the app for a new link.",
    "warming": "The picture tools on your computer are starting. Send the file again in a minute.",
    "revoked": "This upload link no longer works. Ask the app for a new one.",
    "outdated": "This upload link expired with the update. Ask the app for a new one.",
    "not_allowed": "This computer no longer lets this app add files. Ask the owner to allow it again.",
    "invalid_request": "That upload request was not understood.",
    "disk_full": files.DISK_FULL,
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
    nonce: str | None
    touched_at: int | None
    #: The web app (authorization) that asked for the link; empty for a link made before apps were recorded.
    authorization_id: str = ""


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
    """Whether the first bytes are a PNG, JPEG, GIF or WebP picture, or a PDF, as ``kind`` says."""
    if kind == "document":
        return head.startswith(b"%PDF-")
    return (head.startswith(b"\x89PNG\r\n\x1a\n") or head.startswith(b"\xff\xd8\xff")
            or head[:6] in (b"GIF87a", b"GIF89a") or (head[:4] == b"RIFF" and head[8:12] == b"WEBP"))


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
        self._columns_ok = False
        self._migrated = False
        self._last_clean: int | None = None

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
                self._ensure_columns(conn)
                conn.execute("BEGIN IMMEDIATE")
                try:
                    self._migrate(conn)
                    yield conn
                    conn.execute("COMMIT")
                except BaseException:
                    conn.execute("ROLLBACK")
                    raise
            finally:
                conn.close()

    def _ensure_columns(self, conn: sqlite3.Connection) -> None:
        if self._columns_ok:
            return
        names = {row[1] for row in conn.execute("PRAGMA table_info(upload_links)")}
        if "authorization_id" not in names:
            with contextlib.suppress(sqlite3.OperationalError):  # another process added it first
                conn.execute(_ADD_AUTHORIZATION)
        self._columns_ok = True

    def _migrate(self, conn: sqlite3.Connection) -> None:
        """Once per database: end the unfinished links no app asked for.

        Links made before ``authorization_id`` existed carry an empty one and used to be honoured
        for "any consenting app" until they ran out. They are ended here, with a plain message,
        and the database is marked (``user_version``) so this never runs again.
        """
        if self._migrated:
            return
        if conn.execute("PRAGMA user_version").fetchone()[0] < _MIGRATION_DONE:
            result = json.dumps({"ok": False, "code": "outdated", "message": _MESSAGES["outdated"]})
            stale = conn.execute(
                "SELECT upload_id FROM upload_links WHERE authorization_id = '' "
                "AND state IN ('open','receiving','finishing')").fetchall()
            for found in stale:
                conn.execute("UPDATE upload_links SET state='failed', result_json=? WHERE upload_id=?",
                             (result, found[0]))
                self._unlink(self.temp_path(found[0]))
            conn.execute(f"PRAGMA user_version = {_MIGRATION_DONE}")
        self._migrated = True

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

    def mint(self, connection_id: str, key_id: str, profile_id: str, kind: str, note: str,
             authorization_id: str = "") -> MintedLink:
        if kind not in KINDS:
            raise UploadError("invalid_kind")
        if len(note or "") > MAX_NOTE_CHARS:
            raise UploadError("note_too_long")
        self._housekeep()
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
                "max_bytes, state, created_at, expires_at, authorization_id) VALUES (?,?,?,?,?,?,?,?,'open',?,?,?)",
                (upload_id, _hash(token), connection_id, key_id, profile_id, kind, note or "", limit,
                 now, now + LINK_TTL_S, authorization_id or ""))
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

    def accept_chunk(self, token: str, connection_id: str, index: int, total: int, data: bytes,
                     nonce: str) -> int:
        """Append one chunk; returns the bytes held so far.

        ``nonce`` is made by the gateway for each upload (one POST). Chunk 0 binds it; every later
        chunk and the finish must carry the same one. Chunk 0 with another nonce restarts the link
        only when the first upload has been quiet for a minute, so a second holder of the link
        cannot swap the file under an upload that is still moving.
        """
        if not _NONCE.fullmatch(nonce or ""):
            raise UploadError("invalid_request")
        if not data:
            raise UploadError("empty")
        if len(data) > MAX_CHUNK_BYTES:
            raise UploadError("chunk_too_large")
        if index == 0:
            self._housekeep()
        with self._tx() as conn:
            row = self._by_token(conn, token, connection_id)
            self._live(row)
            if not 1 <= total:
                raise UploadError("empty")
            if total > row.max_bytes:
                raise UploadError("too_large")
            if index == 0:
                return self._start(conn, row, total, data, nonce)
            return self._append(conn, row, index, total, data, nonce)

    def _start(self, conn: sqlite3.Connection, row: UploadRow, total: int, data: bytes,
               nonce: str) -> int:
        now = self._now()
        if row.state == "receiving" and row.nonce is not None:
            if row.nonce == nonce:
                raise UploadError("bad_order")
            if now - (row.touched_at or row.started_at or now) <= IDLE_TAKEOVER_S:
                raise UploadError("in_progress")
        if row.attempts >= MAX_ATTEMPTS:
            raise UploadError("too_many_attempts")
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
        try:
            self._write(path, data, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
        except OSError as exc:
            path.unlink(missing_ok=True)  # no half-written scratch file, and no attempt is spent
            if files.is_disk_full(exc):
                raise UploadError("disk_full") from None
            raise
        conn.execute(
            "UPDATE upload_links SET state='receiving', total=?, received=?, next_index=1, attempts=attempts+1, "
            "started_at=COALESCE(started_at, ?), nonce=?, touched_at=?, result_json='' WHERE upload_id=?",
            (total, len(data), now, nonce, now, row.upload_id))
        return len(data)

    def _append(self, conn: sqlite3.Connection, row: UploadRow, index: int, total: int, data: bytes,
                nonce: str) -> int:
        if row.state != "receiving":
            raise UploadError("bad_order")
        if row.nonce != nonce:
            raise UploadError("in_progress")
        if index != row.next_index:
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
        try:
            self._write(path, data, os.O_WRONLY | os.O_APPEND)
        except OSError as exc:
            if files.is_disk_full(exc):
                self._trim(path, row.received)  # drop a half-written chunk; what arrived stays
                raise UploadError("disk_full") from None
            raise
        received = row.received + len(data)
        conn.execute("UPDATE upload_links SET received=?, next_index=next_index+1, touched_at=? WHERE upload_id=?",
                     (received, self._now(), row.upload_id))
        return received

    @staticmethod
    def _trim(path: Path, size: int) -> None:
        """Cut the scratch file back to ``size`` bytes (best effort)."""
        with contextlib.suppress(OSError):
            os.truncate(path, size)

    @staticmethod
    def _write(path: Path, data: bytes, flags: int) -> None:
        fd = os.open(path, flags | getattr(os, "O_NOFOLLOW", 0), 0o600)
        with os.fdopen(fd, "ab" if flags & os.O_APPEND else "wb") as out:
            out.write(data)

    # -- finishing ----------------------------------------------------------

    def begin_finish(self, token: str, connection_id: str, nonce: str) -> FinishPlan:
        if not _NONCE.fullmatch(nonce or ""):
            raise UploadError("invalid_request")
        now = self._now()
        with self._tx() as conn:
            row = self._by_token(conn, token, connection_id)
            if row.state in ("done", "failed"):
                return FinishPlan("result", row, json.loads(row.result_json or "{}"))
            if row.state == "finishing":
                if now - (row.touched_at or now) > FINISHER_TIMEOUT_S:
                    return self._give_up(conn, row)
                return FinishPlan("working", row)
            if row.state == "open":
                raise UploadError("not_started")
            self._live(row)
            warming = self._warming_result(row)
            if warming is not None:  # the save had to wait for the picture tools; say so, do not call it a clash
                return FinishPlan("result", row, warming)
            if row.nonce != nonce:
                raise UploadError("in_progress")
            if row.received != row.total or row.total < 1:
                raise UploadError("incomplete")
            conn.execute("UPDATE upload_links SET state='finishing', touched_at=? WHERE upload_id=?",
                         (now, row.upload_id))
            return FinishPlan("run", row)

    def _give_up(self, conn: sqlite3.Connection, row: UploadRow) -> FinishPlan:
        """A save that began long enough ago that it cannot still be running."""
        result = {"ok": False, "code": "interrupted", "message": _MESSAGES["interrupted"]}
        conn.execute("UPDATE upload_links SET state='failed', result_json=? WHERE upload_id=?",
                     (json.dumps(result), row.upload_id))
        self._unlink(self.temp_path(row.upload_id))
        return FinishPlan("result", row, result)

    def _end(self, upload_id: str, state: str, result: dict[str, Any]) -> None:
        with self._tx() as conn:
            conn.execute("UPDATE upload_links SET state=?, result_json=? WHERE upload_id=?",
                         (state, json.dumps(result), upload_id))
        self.temp_path(upload_id).unlink(missing_ok=True)

    def finish_done(self, upload_id: str, result: dict[str, Any]) -> None:
        self._end(upload_id, "done", result)

    def finish_failed(self, upload_id: str, result: dict[str, Any]) -> None:
        self._end(upload_id, "failed", result)

    @staticmethod
    def _warming_result(row: UploadRow) -> dict[str, Any] | None:
        """The stored "tools are starting" answer of a reopened link, until a new upload clears it."""
        if row.state != "receiving" or row.nonce is not None or not row.result_json:
            return None
        try:
            result = json.loads(row.result_json)
        except ValueError:
            return None
        return result if isinstance(result, dict) and result.get("code") == "warming" else None

    def finish_retry(self, upload_id: str, result: dict[str, Any] | None = None) -> None:
        """A save that may work later (the picture tools are starting): keep the bytes, reopen the link.

        The nonce is cleared so a fresh upload restarts at once, and the warming answer is kept so the
        gateway's next ``finish`` (still carrying the old nonce) gets it rather than a clash.
        """
        warming = {"ok": False, "code": "warming", **{k: v for k, v in (result or {}).items()
                                                      if k in ("message",) and v}}
        warming.setdefault("message", _MESSAGES["warming"])
        with self._tx() as conn:
            conn.execute("UPDATE upload_links SET state='receiving', nonce=NULL, touched_at=?, result_json=? "
                         "WHERE upload_id=? AND state='finishing'",
                         (self._now(), json.dumps(warming), upload_id))

    def claim_warm_retry(self, upload_id: str) -> UploadRow | None:
        """Take a link whose save is waiting for the picture tools, to try that save again.

        Only a link still waiting on warming (all bytes held, no new upload begun, not expired) can be
        taken; it goes back to ``finishing``, so a gateway ``finish`` meanwhile is told "working".
        Returns ``None`` when there is nothing left to retry, which ends the caller's loop.
        """
        now = self._now()
        with self._tx() as conn:
            row = self._row(conn.execute("SELECT * FROM upload_links WHERE upload_id = ?",
                                         (upload_id,)).fetchone())
            if (row is None or self._warming_result(row) is None or now > _deadline(row)
                    or row.total < 1 or row.received != row.total):
                return None
            taken = conn.execute("UPDATE upload_links SET state='finishing', touched_at=? "
                                 "WHERE upload_id=? AND state='receiving' AND nonce IS NULL", (now, upload_id))
            return row if taken.rowcount == 1 else None

    def fail_open_links(self, connection_id: str, authorization_id: str | None = None) -> int:
        """End every unfinished link of a connection (its consent or grant key was revoked or replaced).

        With ``authorization_id`` only the links that app asked for end, plus links made before apps
        were recorded (their owner is unknown); the other apps' links stay open.
        """
        if authorization_id is None:
            return self._fail_where(connection_id, "", ())
        return self._fail_where(connection_id, "AND (authorization_id = ? OR authorization_id = '')",
                                (authorization_id,))

    def fail_unlisted_authorizations(self, connection_id: str, listed: set[str] | frozenset[str]) -> int:
        """End the unfinished links of apps the gateway no longer lists for this connection.

        ``listed`` must be the whole list: the caller never passes a partial one.
        """
        if not (self._root / "media" / DB_NAME).exists():
            return 0
        with self._tx() as conn:
            owners = {row[0] for row in conn.execute(
                "SELECT DISTINCT authorization_id FROM upload_links WHERE connection_id = ? "
                "AND state IN ('open','receiving','finishing') AND authorization_id != ''",
                (connection_id,)).fetchall()}
        return sum(self.fail_open_links(connection_id, gone) for gone in sorted(owners - set(listed)))

    def _fail_where(self, connection_id: str, extra: str, params: tuple) -> int:
        if not (self._root / "media" / DB_NAME).exists():
            return 0
        result = json.dumps({"ok": False, "code": "revoked", "message": _MESSAGES["revoked"]})
        with self._tx() as conn:
            rows = conn.execute(
                "SELECT upload_id FROM upload_links WHERE connection_id = ? "
                f"AND state IN ('open','receiving','finishing') {extra}", (connection_id, *params)).fetchall()
            for found in rows:
                conn.execute("UPDATE upload_links SET state='failed', result_json=? WHERE upload_id=?",
                             (result, found[0]))
                self._unlink(self.temp_path(found[0]))
        return len(rows)

    # -- housekeeping -------------------------------------------------------

    def _housekeep(self) -> None:
        """Clean up at the first use after start, then at most once an hour. Never fails the caller."""
        now = self._now()
        if self._last_clean is not None and now - self._last_clean < HOUSEKEEPING_S:
            return
        self._last_clean = now
        try:
            self.cleanup()
        except Exception as exc:  # noqa: BLE001 - housekeeping must not block an upload
            logger.warning("upload housekeeping skipped (%s)", type(exc).__name__)

    def cleanup(self) -> int:
        """Remove scratch files of spent or abandoned uploads and old rows; returns files removed.

        Creates nothing: a computer that never made an upload link has nothing to clean.
        """
        now = self._now()
        removed = self._expire_rows(now) if (self._root / "media" / DB_NAME).exists() else 0
        scratch = self._root / "media" / "tmp"
        for entry in (scratch.glob("upload-*.part") if scratch.is_dir() else ()):
            with contextlib.suppress(OSError):
                if entry.lstat().st_mtime < now - STRAY_FILE_S:
                    removed += self._unlink(entry)
        return removed

    def _expire_rows(self, now: int) -> int:
        removed = 0
        with self._tx() as conn:
            for found in conn.execute("SELECT * FROM upload_links").fetchall():
                row = self._row(found)
                if row.state == "finishing" and now - (row.touched_at or now) > FINISHER_TIMEOUT_S:
                    self._give_up(conn, row)
                    continue
                running = row.state in ("open", "receiving")
                if running and now <= _deadline(row):
                    continue
                if running:
                    conn.execute("UPDATE upload_links SET state='failed', result_json=? WHERE upload_id=?",
                                 (json.dumps({"ok": False, "code": "expired",
                                              "message": _MESSAGES["expired"]}), row.upload_id))
                if row.state != "finishing":
                    removed += self._unlink(self.temp_path(row.upload_id))
            conn.execute("DELETE FROM upload_links WHERE created_at < ?", (now - RETENTION_S,))
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
