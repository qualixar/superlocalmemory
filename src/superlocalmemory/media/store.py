# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""The media store: items, vectors per embedding space, jobs and profile moves.

One writer connection (a lock serializes writers) and one read connection per
thread, the same shape as the other SQLite stores. No imaging or ML library is
imported here; vectors are plain float32 blobs handled by sqlite-vec.
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
import struct
import threading
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Sequence

from superlocalmemory.media.schema import (
    MEDIA_SCHEMA_VERSION, apply_schema, stored_version,
)
from superlocalmemory.media.store_jobs import JobsMixin, utc_stamp

logger = logging.getLogger(__name__)

DEFAULT_DIM = 768
MAX_K = 200
_SPACE_ID = re.compile(r"^[0-9a-f]{32}$")
_ITEM_FIELDS = (
    "profile_id", "kind", "sha256", "phash", "mime", "bytes", "width", "height",
    "original_relpath", "exif_json", "captured_at", "anchor_memory_id", "document_id",
    "page_no", "source_id", "origin", "state", "thumb_webp",
)
_REQUIRED = ("profile_id", "kind", "sha256", "mime", "bytes", "origin")


class MediaStoreReadOnly(RuntimeError):
    """The file was written by a newer version; this one only reads it."""


def vec_table(space_id: str) -> str:
    if not _SPACE_ID.fullmatch(space_id):
        raise ValueError("invalid space id")
    return f"media_vec_{space_id}"


def _load_vec(conn: sqlite3.Connection) -> None:
    import sqlite_vec

    conn.enable_load_extension(True)
    try:
        sqlite_vec.load(conn)
    finally:
        conn.enable_load_extension(False)


#: The only camera/capture fields kept. Anything else (maker notes, embedded
#: blobs, location sub-records) is dropped before it reaches the database.
EXIF_ALLOWED = frozenset({
    "DateTime", "DateTimeOriginal", "DateTimeDigitized", "Make", "Model", "Orientation", "LensModel",
    "Software", "ExposureTime", "FNumber", "ISOSpeedRatings", "FocalLength", "Flash", "WhiteBalance",
    "ColorSpace", "XResolution", "YResolution", "ResolutionUnit", "PixelXDimension", "PixelYDimension",
    "ImageWidth", "ImageLength",
})
_GPS_TAG_ID = "34853"


def _is_location_key(key: Any) -> bool:
    text = str(key).strip().upper()
    return text.startswith("GPS") or text == _GPS_TAG_ID


def _reject_location(value: Any) -> None:
    if isinstance(value, dict):
        for key, inner in value.items():
            if _is_location_key(key):
                raise ValueError("location data is never stored")
            _reject_location(inner)
    elif isinstance(value, list):
        for inner in value:
            _reject_location(inner)


def _exif_text(raw: Any) -> str:
    data = json.loads(raw) if isinstance(raw, (str, bytes)) else raw
    _reject_location(data)
    if not isinstance(data, dict):
        raise ValueError("exif must be an object")
    kept = {str(k): v for k, v in data.items()
            if str(k) in EXIF_ALLOWED and isinstance(v, (str, int, float, bool))}
    return json.dumps(kept, sort_keys=True)


class MediaStore(JobsMixin):
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._wlock = threading.RLock()
        self._local = threading.local()
        self._readers: list[sqlite3.Connection] = []
        self._rlock = threading.Lock()
        self._closed = False
        self._w = self._connect()
        self.read_only = False
        self._init_schema()

    # -- connections -------------------------------------------------------
    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.path), timeout=5, check_same_thread=False,
                               isolation_level=None)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA busy_timeout=5000")
            conn.execute("PRAGMA foreign_keys=ON")
            _load_vec(conn)
        except BaseException:
            conn.close()
            raise
        return conn

    def _init_schema(self) -> None:
        version = stored_version(self._w)
        if version > MEDIA_SCHEMA_VERSION:
            logger.warning("media.db is version %s, newer than this build knows (%s); "
                           "opening it read-only", version, MEDIA_SCHEMA_VERSION)
            self.read_only = True
            self._w.execute("PRAGMA query_only=ON")
            return
        self._w.execute("PRAGMA journal_mode=WAL")
        from superlocalmemory import __version__

        with self._write() as conn:
            apply_schema(conn, created_by=__version__)

    @contextmanager
    def _write(self) -> Iterator[sqlite3.Connection]:
        if self.read_only:
            raise MediaStoreReadOnly("media.db was written by a newer version")
        with self._wlock:
            self._w.execute("BEGIN IMMEDIATE")
            try:
                yield self._w
            except BaseException:
                self._w.execute("ROLLBACK")
                raise
            self._w.execute("COMMIT")

    def _read(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = self._connect()
            conn.execute("PRAGMA query_only=ON")
            self._local.conn = conn
            with self._rlock:
                self._readers.append(conn)
        return conn

    def close(self) -> None:
        with self._rlock:
            readers, self._readers = self._readers, []
        for conn in [*readers, self._w]:
            try:
                conn.close()
            except sqlite3.Error:
                pass
        self._closed = True

    # -- spaces ------------------------------------------------------------
    def active_space(self) -> dict[str, Any] | None:
        row = self._read().execute("SELECT * FROM media_spaces WHERE state = 'active'").fetchone()
        return dict(row) if row else None

    def ensure_active_space(self, model_id: str, model_revision: str, dim: int = DEFAULT_DIM) -> str:
        """The active space for this model, created (and the old one retired) when it differs."""
        with self._write() as conn:
            row = conn.execute("SELECT * FROM media_spaces WHERE state = 'active'").fetchone()
            if row and (row["model_id"], row["model_revision"], row["dim"]) == (model_id, model_revision, dim):
                return row["space_id"]
            conn.execute("UPDATE media_spaces SET state = 'previous' WHERE state = 'active'")
            space_id = uuid.uuid4().hex
            conn.execute(
                "INSERT INTO media_spaces(space_id, model_id, model_revision, dim, state, created_at)"
                " VALUES (?, ?, ?, ?, 'active', ?)", (space_id, model_id, model_revision, int(dim), utc_stamp()))
            conn.execute(
                f"CREATE VIRTUAL TABLE IF NOT EXISTS {vec_table(space_id)} USING vec0("
                f"profile_id TEXT PARTITION KEY, embedding float[{int(dim)}] distance_metric=cosine)")
            return space_id

    # -- items -------------------------------------------------------------
    def insert_item(self, **fields: Any) -> str:
        unknown = set(fields) - set(_ITEM_FIELDS)
        missing = [k for k in _REQUIRED if k not in fields]
        if unknown or missing:
            raise ValueError(f"unknown fields {sorted(unknown)}, missing fields {missing}")
        fields["exif_json"] = _exif_text(fields.get("exif_json", {}))
        media_id = uuid.uuid4().hex
        cols = ["media_id", "created_at", *fields]
        with self._write() as conn:
            conn.execute(
                f"INSERT INTO media_items({','.join(cols)}) VALUES ({','.join('?' * len(cols))})",
                [media_id, utc_stamp(), *fields.values()])
        return media_id

    def get_item(self, media_id: str) -> dict[str, Any] | None:
        row = self._read().execute("SELECT * FROM media_items WHERE media_id = ?", (media_id,)).fetchone()
        return dict(row) if row else None

    def find_by_sha(self, profile_id: str, sha256: str) -> dict[str, Any] | None:
        row = self._read().execute(
            "SELECT * FROM media_items WHERE profile_id = ? AND sha256 = ? AND state = 'active'"
            " ORDER BY created_at LIMIT 1", (profile_id, sha256)).fetchone()
        return dict(row) if row else None

    def list_items(self, profile_id: str, *, kind: str | None = None, state: str = "active",
                   limit: int = 50, offset: int = 0) -> list[dict[str, Any]]:
        sql, args = "SELECT * FROM media_items WHERE profile_id = ? AND state = ?", [profile_id, state]
        if kind:
            sql += " AND kind = ?"
            args.append(kind)
        rows = self._read().execute(sql + " ORDER BY created_at, media_id LIMIT ? OFFSET ?",
                                    [*args, int(limit), int(offset)]).fetchall()
        return [dict(r) for r in rows]

    def set_state(self, media_id: str, state: str) -> None:
        stamp = utc_stamp() if state == "tombstoned" else None
        with self._write() as conn:
            conn.execute("UPDATE media_items SET state = ?, tombstoned_at = ? WHERE media_id = ?",
                         (state, stamp, media_id))

    def count_and_bytes(self, profile_id: str) -> tuple[int, int]:
        row = self._read().execute(
            "SELECT COUNT(*), COALESCE(SUM(bytes), 0) FROM media_items WHERE profile_id = ? AND state = 'active'",
            (profile_id,)).fetchone()
        return int(row[0]), int(row[1])

    # -- vectors -----------------------------------------------------------
    def put_vector(self, media_id: str, space_id: str, profile_id: str, vector: Sequence[float]) -> None:
        table = vec_table(space_id)
        with self._write() as conn:
            space = conn.execute("SELECT dim FROM media_spaces WHERE space_id = ?", (space_id,)).fetchone()
            if space is None:
                raise ValueError("unknown embedding space")
            if len(vector) != space["dim"]:
                raise ValueError(f"expected {space['dim']} numbers, got {len(vector)}")
            blob = struct.pack(f"<{len(vector)}f", *vector)
            cur = conn.execute(f"INSERT INTO {table}(profile_id, embedding) VALUES (?, ?)", (profile_id, blob))
            conn.execute("INSERT INTO media_vector_rows(space_id, vec_rowid, media_id, profile_id)"
                         " VALUES (?, ?, ?, ?)", (space_id, cur.lastrowid, media_id, profile_id))

    def knn(self, vector: Sequence[float], profile_id: str, k: int,
            space_id: str | None = None) -> list[tuple[str, float]]:
        """Nearest items of one profile, closest first: [(media_id, cosine distance)]."""
        conn = self._read()
        if space_id is None:
            space = self.active_space()
            if space is None:
                return []
            space_id = space["space_id"]
        blob = struct.pack(f"<{len(vector)}f", *vector)
        rows = conn.execute(
            f"SELECT rowid, distance FROM {vec_table(space_id)} WHERE embedding MATCH ? AND k = ?"
            " AND profile_id = ?", (blob, max(1, min(int(k), MAX_K)), profile_id)).fetchall()
        out: list[tuple[str, float]] = []
        for rowid, distance in rows:
            m = conn.execute("SELECT media_id FROM media_vector_rows WHERE space_id = ? AND vec_rowid = ?",
                             (space_id, rowid)).fetchone()
            if m:
                out.append((m[0], float(distance)))
        return sorted(out, key=lambda t: t[1])

    def _drop_vector_rows(self, conn: sqlite3.Connection, where: str, args: Sequence[Any]) -> list[tuple]:
        rows = conn.execute(
            f"SELECT space_id, vec_rowid, media_id, profile_id FROM media_vector_rows WHERE {where}",
            args).fetchall()
        for space_id, rowid, _, _ in rows:
            conn.execute(f"DELETE FROM {vec_table(space_id)} WHERE rowid = ?", (rowid,))
        conn.execute(f"DELETE FROM media_vector_rows WHERE {where}", args)
        return [tuple(r) for r in rows]

    def delete_vectors(self, media_id: str) -> None:
        with self._write() as conn:
            self._drop_vector_rows(conn, "media_id = ?", (media_id,))

    # -- profile-wide operations --------------------------------------------
    def delete_profile_rows(self, profile_id: str) -> int:
        """Erase everything of one profile in one transaction; returns rows removed."""
        total = 0
        with self._write() as conn:
            total += len(self._drop_vector_rows(conn, "profile_id = ?", (profile_id,)))
            docs = "SELECT document_id FROM documents WHERE profile_id = ?"
            srcs = "SELECT source_id FROM sources WHERE profile_id = ?"
            for sql in (f"DELETE FROM doc_pages WHERE document_id IN ({docs})",
                        f"DELETE FROM source_files WHERE source_id IN ({srcs})",
                        f"DELETE FROM source_links WHERE source_id IN ({srcs})"):
                total += conn.execute(sql, (profile_id,)).rowcount
            for table in ("media_items", "documents", "jobs", "sources"):
                total += conn.execute(f"DELETE FROM {table} WHERE profile_id = ?", (profile_id,)).rowcount
        return total

    def move_profile_rows(self, from_profile: str, to_profile: str) -> int:
        """Re-home every row of one profile (profile deletion keeps the content)."""
        if not from_profile or from_profile == to_profile:
            return 0
        total = 0
        with self._write() as conn:
            total += self._retire_clashing_sources(conn, from_profile, to_profile)
            total += self._move_vectors(conn, from_profile, to_profile)
            for table in ("media_items", "documents", "jobs", "sources"):
                total += conn.execute(f"UPDATE {table} SET profile_id = ? WHERE profile_id = ?",
                                      (to_profile, from_profile)).rowcount
        return total

    @staticmethod
    def _retire_clashing_sources(conn: sqlite3.Connection, from_profile: str, to_profile: str) -> int:
        """A folder both profiles watch stays with the target; the mover's copy is marked removed."""
        return conn.execute(
            "UPDATE sources SET state = 'removed' WHERE profile_id = ? AND state != 'removed' AND root_path IN "
            "(SELECT root_path FROM sources WHERE profile_id = ? AND state != 'removed')",
            (from_profile, to_profile)).rowcount

    def _move_vectors(self, conn: sqlite3.Connection, from_profile: str, to_profile: str) -> int:
        moved = 0
        for space_id, rowid, _, _ in self._vector_rows(conn, from_profile):
            table = vec_table(space_id)
            emb = conn.execute(f"SELECT embedding FROM {table} WHERE rowid = ?", (rowid,)).fetchone()
            if emb is None:  # a map row whose vector is gone: drop the mapping, move nothing
                conn.execute("DELETE FROM media_vector_rows WHERE space_id = ? AND vec_rowid = ?",
                             (space_id, rowid))
                continue
            conn.execute(f"DELETE FROM {table} WHERE rowid = ?", (rowid,))
            cur = conn.execute(f"INSERT INTO {table}(profile_id, embedding) VALUES (?, ?)", (to_profile, emb[0]))
            conn.execute("UPDATE media_vector_rows SET vec_rowid = ?, profile_id = ? "
                         "WHERE space_id = ? AND vec_rowid = ?", (cur.lastrowid, to_profile, space_id, rowid))
            moved += 1
        return moved

    @staticmethod
    def _vector_rows(conn: sqlite3.Connection, profile_id: str) -> list[sqlite3.Row]:
        return conn.execute("SELECT space_id, vec_rowid, media_id, profile_id FROM media_vector_rows "
                            "WHERE profile_id = ?", (profile_id,)).fetchall()
