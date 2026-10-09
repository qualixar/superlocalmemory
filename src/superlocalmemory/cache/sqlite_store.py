# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com
"""The on-disk derivation cache: one SQLite file, content-addressed, never a source of truth."""

from __future__ import annotations

import logging
import os
import sqlite3
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from superlocalmemory.cache.keys import CacheKey
from superlocalmemory.cache.port import check_payload

logger = logging.getLogger(__name__)

DEFAULT_MAX_MB = 2048
TRIM_TO = 0.9
_TOUCH_EVERY_S = 60.0
_DDL = (
    "CREATE TABLE IF NOT EXISTS derivations ("
    "content_sha256 TEXT NOT NULL, deriver_id TEXT NOT NULL, deriver_version TEXT NOT NULL, "
    "model_id TEXT NOT NULL DEFAULT '', params_hash TEXT NOT NULL DEFAULT '', "
    "payload_kind TEXT NOT NULL CHECK (payload_kind IN ('text','vector_f32','json','png')), "
    "payload BLOB NOT NULL, bytes INTEGER NOT NULL, created_at TEXT NOT NULL, "
    "last_used_at TEXT NOT NULL, "
    "PRIMARY KEY (content_sha256, deriver_id, deriver_version, model_id, params_hash))",
    "CREATE INDEX IF NOT EXISTS ix_deriv_lru ON derivations(last_used_at)",
    "CREATE TABLE IF NOT EXISTS cache_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)",
    "INSERT OR IGNORE INTO cache_meta (key, value) VALUES ('schema_version', '1')",
)
_WHERE = "content_sha256=? AND deriver_id=? AND deriver_version=? AND model_id=? AND params_hash=?"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def max_bytes() -> int:
    raw = os.environ.get("SLM_DERIVE_CACHE_MAX_MB", "")
    try:
        mb = float(raw) if raw else DEFAULT_MAX_MB
    except ValueError:
        mb = DEFAULT_MAX_MB
    return int(max(mb, 0.001) * 1024 * 1024)


class SqliteDeriveCache:
    """Raw-bytes cache in one SQLite file, created on the first write."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._write_lock = threading.Lock()
        self._local = threading.local()
        self._touched: dict[tuple, float] = {}
        self._total: int | None = None

    # -- connections -----------------------------------------------------------

    def _open(self, create: bool) -> sqlite3.Connection | None:
        conn = getattr(self._local, "conn", None)
        if conn is not None and self.path.exists():
            return conn
        self._drop_local()
        if not create and not self.path.exists():
            return None
        try:
            return self._connect(create)
        except sqlite3.DatabaseError as exc:
            self._quarantine(exc)
            return self._connect(create) if create else None

    def _connect(self, create: bool) -> sqlite3.Connection:
        new = not self.path.exists()
        if new:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.path), timeout=10.0, check_same_thread=False)
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            for statement in _DDL:
                conn.execute(statement)
            conn.commit()
        except sqlite3.DatabaseError:
            conn.close()
            raise
        if new and sys.platform != "win32":
            try:
                os.chmod(self.path, 0o600)
            except OSError:
                pass
        self._local.conn = conn
        return conn

    def _drop_local(self) -> None:
        conn = getattr(self._local, "conn", None)
        self._local.conn = None
        if conn is not None:
            try:
                conn.close()
            except sqlite3.Error:
                pass

    def _quarantine(self, exc: Exception) -> None:
        self._drop_local()
        target = self.path.with_name(f"{self.path.name}.corrupt-{int(time.time())}")
        logger.warning("derive cache %s is unreadable (%s); setting it aside", self.path, exc)
        for suffix in ("", "-wal", "-shm"):
            part = self.path.with_name(self.path.name + suffix)
            try:
                if suffix == "":
                    os.replace(part, target)
                else:
                    part.unlink()
            except OSError:
                pass
        self._total = None

    # -- the port --------------------------------------------------------------

    def get(self, key: CacheKey) -> bytes | None:
        try:
            conn = self._open(False)
            if conn is None:
                return None
            row = conn.execute(f"SELECT payload FROM derivations WHERE {_WHERE}",
                               key.as_tuple()).fetchone()
        except sqlite3.DatabaseError as exc:
            self._quarantine(exc)
            return None
        if row is None:
            return None
        self._touch(conn, key)
        return bytes(row[0])

    def _touch(self, conn: sqlite3.Connection, key: CacheKey) -> None:
        mono = time.monotonic()
        tup = key.as_tuple()
        if mono - self._touched.get(tup, -_TOUCH_EVERY_S) < _TOUCH_EVERY_S:
            return
        try:
            with self._write_lock:
                conn.execute(f"UPDATE derivations SET last_used_at=? WHERE {_WHERE}",
                             (_now(), *tup))
                conn.commit()
            if len(self._touched) > 50_000:
                self._touched.clear()
            self._touched[tup] = mono
        except sqlite3.Error:
            pass  # best effort

    def put(self, key: CacheKey, payload: bytes, *, kind: str) -> None:
        check_payload(payload, kind)
        data = bytes(payload)
        now = _now()
        with self._write_lock:
            try:
                conn = self._open(True)
                self._store(conn, key, data, kind, now)
            except sqlite3.DatabaseError as exc:
                self._quarantine(exc)
                self._store(self._open(True), key, data, kind, now)
            self._touched[key.as_tuple()] = time.monotonic()

    def _store(self, conn: sqlite3.Connection, key: CacheKey, data: bytes, kind: str,
               now: str) -> None:
        conn.execute(
            "INSERT OR REPLACE INTO derivations (content_sha256, deriver_id, deriver_version, "
            "model_id, params_hash, payload_kind, payload, bytes, created_at, last_used_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?)", (*key.as_tuple(), kind, data, len(data), now, now))
        conn.commit()
        self._total = None
        if self._size(conn) > max_bytes():
            self._trim(conn)

    def _size(self, conn: sqlite3.Connection) -> int:
        if self._total is None:
            self._total = int(conn.execute(
                "SELECT COALESCE(SUM(bytes), 0) FROM derivations").fetchone()[0])
        return self._total

    def _trim(self, conn: sqlite3.Connection) -> None:
        goal = int(max_bytes() * TRIM_TO)
        total = self._size(conn)
        rows = conn.execute("SELECT rowid, bytes FROM derivations ORDER BY last_used_at, rowid")
        doomed = []
        for rowid, size in rows.fetchall():
            if total <= goal:
                break
            doomed.append((rowid,))
            total -= size
        conn.executemany("DELETE FROM derivations WHERE rowid=?", doomed)
        conn.commit()
        self._total = None

    def invalidate(self, *, deriver_id: str | None = None, model_id: str | None = None) -> int:
        if deriver_id is None and model_id is None:
            raise ValueError("invalidate needs deriver_id or model_id; use clear() for all")
        clauses, args = [], []
        for column, value in (("deriver_id", deriver_id), ("model_id", model_id)):
            if value is not None:
                clauses.append(f"{column}=?")
                args.append(value)
        with self._write_lock:
            try:
                conn = self._open(False)
                if conn is None:
                    return 0
                count = conn.execute("DELETE FROM derivations WHERE " + " AND ".join(clauses),
                                     args).rowcount
                conn.commit()
            except sqlite3.DatabaseError as exc:
                self._quarantine(exc)
                return 0
            self._total = None
        return max(count, 0)

    def invalidate_content(self, content_sha256: str) -> int:
        """Drop every entry derived from one file (all derivers). Never creates the cache file."""
        with self._write_lock:
            try:
                conn = self._open(False)
                if conn is None:
                    return 0
                count = conn.execute("DELETE FROM derivations WHERE content_sha256=?",
                                     (content_sha256,)).rowcount
                conn.commit()
            except sqlite3.DatabaseError as exc:
                self._quarantine(exc)
                return 0
            self._total = None
            self._touched = {k: v for k, v in self._touched.items() if k[0] != content_sha256}
        return max(count, 0)

    def clear(self) -> None:
        with self._write_lock:
            self._drop_local()
            for suffix in ("", "-wal", "-shm"):
                try:
                    self.path.with_name(self.path.name + suffix).unlink()
                except FileNotFoundError:
                    pass
            self._total = None
            self._touched.clear()

    def stats(self) -> dict:
        out = {"backend": "sqlite", "entries": 0, "bytes": 0, "path": str(self.path),
               "exists": self.path.exists()}
        try:
            conn = self._open(False)
            if conn is not None:
                n, size = conn.execute(
                    "SELECT COUNT(*), COALESCE(SUM(bytes), 0) FROM derivations").fetchone()
                out.update(entries=int(n), bytes=int(size))
        except sqlite3.DatabaseError as exc:
            self._quarantine(exc)
            out["exists"] = self.path.exists()
        return out
