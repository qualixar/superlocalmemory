# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory v3.4.22 — LLD-01 §4.1

"""Context cache — separate SQLite WAL DB, read-path <10 ms.

LLD reference: `.backup/active-brain/lld/LLD-01-context-cache-and-hot-path-hooks.md`
Section 4.1.

Two concerns in one module:
  - Writer (``ContextCache``) — used by the daemon only. Owns pragmas,
    schema bootstrap, install-binding row, LRU sweep.
  - Reader (``read_entry_fast``) — used by the UserPromptSubmit hook.
    Read-only SQLite URI, no pragmas, NEVER raises.

Hot-path contract (``read_entry_fast``):
  - stdlib-only imports.
  - Never raises: any exception → returns ``None`` (fail-open miss).
  - Verifies install-binding HMAC to reject a DB at an env-var-hijacked
    path.
  - Applies TTL in SQL, not Python, to avoid fetching stale rows.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path

from superlocalmemory.core.security_primitives import (
    PathTraversalError,
    ensure_install_token,
    redact_secrets,
    safe_resolve,
)


CACHE_DB_DEFAULT: Path = Path.home() / ".superlocalmemory" / "active_brain_cache.db"
INSTALL_TOKEN_DEFAULT: Path = Path.home() / ".superlocalmemory" / ".install_token"

TTL_SECONDS: int = 120
CLEANUP_HORIZON_SECONDS: int = 600
MAX_BYTES: int = 50 * 1024 * 1024
MAX_CONTENT_CHARS: int = 4000
SCHEMA_VERSION: str = "3.4.22"

_HMAC_MATERIAL: bytes = b"active_brain_cache"
_HMAC_HEX_LEN: int = 32


@dataclass(frozen=True, slots=True)
class CacheEntry:
    """Single cache row. ``content`` is expected to be pre-redacted."""

    session_id: str
    topic_sig: str
    content: str
    fact_ids: list[str] = field(default_factory=list)
    provenance: str = "tool_observation"
    computed_at: int = 0
    byte_size: int = 0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _expected_binding_hmac(token: str) -> str:
    """Compute the HMAC-SHA256 first-32-hex value for the install token."""
    return hmac.new(
        token.encode("utf-8"), _HMAC_MATERIAL, hashlib.sha256,
    ).hexdigest()[:_HMAC_HEX_LEN]


def _read_install_token(home: Path) -> str | None:
    """Read the install token without creating it. Returns None if missing."""
    token_path = home / ".install_token"
    if not token_path.exists():
        return None
    try:
        token = token_path.read_text(encoding="utf-8").strip()
    except OSError:  # pragma: no cover — disk-IO failure under contention
        return None
    return token or None


# ---------------------------------------------------------------------------
# Writer (daemon-side)
# ---------------------------------------------------------------------------


class ContextCache:
    """Writer-side cache. One instance per daemon process.

    Opens the DB with WAL + NORMAL sync + bounded mmap, bootstraps the
    schema, and writes an install-bound HMAC into ``slm_meta`` so the
    reader can reject a foreign DB pointed at via ``SLM_CACHE_DB``.
    """

    def __init__(
        self,
        db_path: Path | None = None,
        home_dir: Path | None = None,
    ) -> None:
        self._home = home_dir or (Path.home() / ".superlocalmemory")
        self._home.mkdir(parents=True, exist_ok=True)

        raw = db_path or (self._home / "active_brain_cache.db")
        self._db_path = safe_resolve(self._home, Path(raw).name) \
            if Path(raw).parent == self._home else \
            safe_resolve(self._home, raw)

        self._write_conn = self._open_writer()
        self._bootstrap_schema_and_meta()

    # -- Open / bootstrap ---------------------------------------------------

    def _open_writer(self) -> sqlite3.Connection:
        # Pre-create the file with 0600 if it's missing, so sqlite inherits
        # the restrictive mode instead of umask default.
        if not self._db_path.exists():
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            try:
                fd = os.open(str(self._db_path), flags, 0o600)
                os.close(fd)
            except FileExistsError:  # pragma: no cover — race on concurrent start
                pass
        # On POSIX, enforce mode in case an earlier process created it.
        if os.name != "nt":
            try:
                os.chmod(self._db_path, 0o600)
            except OSError:  # pragma: no cover — remote FS without chmod
                pass

        conn = sqlite3.connect(
            str(self._db_path), isolation_level=None, timeout=5.0,
        )
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA temp_store=MEMORY")
        conn.execute("PRAGMA cache_size=-32768")
        conn.execute("PRAGMA busy_timeout=500")
        # 64 MB writer mmap — reader opens with defaults to keep budget tight.
        try:
            conn.execute("PRAGMA mmap_size=67108864")
        except sqlite3.Error:  # pragma: no cover — some builds disable mmap
            pass
        return conn

    def _bootstrap_schema_and_meta(self) -> None:
        self._write_conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS context_entries (
                session_id  TEXT NOT NULL,
                topic_sig   TEXT NOT NULL,
                content     TEXT NOT NULL,
                fact_ids    TEXT NOT NULL,
                provenance  TEXT NOT NULL DEFAULT 'tool_observation',
                computed_at INTEGER NOT NULL,
                byte_size   INTEGER NOT NULL,
                PRIMARY KEY (session_id, topic_sig)
            ) WITHOUT ROWID;

            CREATE INDEX IF NOT EXISTS idx_ctx_session_time
                ON context_entries(session_id, computed_at);
            CREATE INDEX IF NOT EXISTS idx_ctx_time
                ON context_entries(computed_at);

            CREATE TABLE IF NOT EXISTS slm_meta (
                key        TEXT PRIMARY KEY,
                value      TEXT NOT NULL,
                created_at INTEGER NOT NULL
            );
            """
        )
        token = ensure_install_token()
        now = int(time.time())
        self._write_conn.execute(
            "INSERT OR IGNORE INTO slm_meta (key, value, created_at) "
            "VALUES (?, ?, ?)",
            ("install_token_hmac", _expected_binding_hmac(token), now),
        )
        self._write_conn.execute(
            "INSERT OR IGNORE INTO slm_meta (key, value, created_at) "
            "VALUES (?, ?, ?)",
            ("schema_version", SCHEMA_VERSION, now),
        )

    # -- Write path ---------------------------------------------------------

    def upsert(self, entry: CacheEntry) -> None:
        """Insert-or-replace a cache row.

        Content is redacted (belt-and-suspenders — caller should have done
        this already per LLD-07 §6.3) and truncated. Byte size is computed
        here, not trusted from caller, so LRU accounting stays accurate.
        Does NOT run LRU sweep inline (PERF-01-07) — that's a background
        task on the daemon.
        """
        content = redact_secrets(entry.content)[:MAX_CONTENT_CHARS]
        fact_ids_json = json.dumps(list(entry.fact_ids))
        byte_size = (
            len(content.encode("utf-8"))
            + len(fact_ids_json)
            + len(entry.session_id)
            + len(entry.topic_sig)
        )
        computed_at = entry.computed_at or int(time.time())
        self._write_conn.execute(
            """
            INSERT OR REPLACE INTO context_entries
                (session_id, topic_sig, content, fact_ids,
                 provenance, computed_at, byte_size)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (entry.session_id, entry.topic_sig, content, fact_ids_json,
             entry.provenance, computed_at, byte_size),
        )

    # -- Cleanup ------------------------------------------------------------

    def cleanup_session(
        self, session_id: str, *, older_than: int = CLEANUP_HORIZON_SECONDS,
    ) -> int:
        """Delete rows for ``session_id`` older than ``older_than`` seconds."""
        cutoff = int(time.time()) - older_than
        cur = self._write_conn.execute(
            "DELETE FROM context_entries "
            "WHERE session_id=? AND computed_at < ?",
            (session_id, cutoff),
        )
        return cur.rowcount

    def cleanup_global_lru(self) -> int:
        """Background sweep — runs every 60s in the daemon.

        Two passes:
          1. Time-based — delete rows older than ``CLEANUP_HORIZON_SECONDS``.
          2. Byte-based — if total size still exceeds ``MAX_BYTES``, delete
             oldest rows (by ``computed_at``) until total is <= 90% of cap.
        Returns total deletions.
        """
        cutoff = int(time.time()) - CLEANUP_HORIZON_SECONDS
        cur = self._write_conn.execute(
            "DELETE FROM context_entries WHERE computed_at < ?", (cutoff,),
        )
        deleted = cur.rowcount

        total = self._write_conn.execute(
            "SELECT COALESCE(SUM(byte_size), 0) FROM context_entries",
        ).fetchone()[0]
        if total <= MAX_BYTES:
            return deleted

        target = int(MAX_BYTES * 0.9)
        while total > target:
            rows = self._write_conn.execute(
                "SELECT session_id, topic_sig, byte_size "
                "FROM context_entries "
                "ORDER BY computed_at ASC LIMIT 100",
            ).fetchall()
            if not rows:  # pragma: no cover — reached only if table empties mid-sweep
                break
            for sess, sig, size in rows:
                self._write_conn.execute(
                    "DELETE FROM context_entries "
                    "WHERE session_id=? AND topic_sig=?",
                    (sess, sig),
                )
                deleted += 1
                total -= size
                if total <= target:
                    break
        return deleted

    def close(self) -> None:
        try:
            self._write_conn.close()
        except sqlite3.Error:  # pragma: no cover — defensive
            pass


# ---------------------------------------------------------------------------
# Reader (hook-side hot path — NEVER raises)
# ---------------------------------------------------------------------------


def read_entry_fast(
    session_id: str,
    topic_sig: str,
    *,
    db_path: Path | None = None,
    home_dir: Path | None = None,
) -> CacheEntry | None:
    """Hot-path reader used by the UserPromptSubmit hook.

    Contract (LLD-01 §4.1):
      - NEVER raises. Any exception → returns ``None``.
      - Returns ``None`` on miss, stale (TTL exceeded), missing DB, path
        traversal attempt, or failed install-binding check.
      - Opens the SQLite file read-only via ``?mode=ro`` URI.
      - stdlib-only — no heavy imports, no daemon HTTP call.
    """
    try:
        home = home_dir or (Path.home() / ".superlocalmemory")
        if not home.exists():
            return None

        requested = db_path or Path(
            os.environ.get("SLM_CACHE_DB") or (home / "active_brain_cache.db"),
        )
        try:
            resolved = safe_resolve(home, Path(requested).resolve())
        except (PathTraversalError, OSError, ValueError):
            return None

        if not resolved.exists():
            return None

        token = _read_install_token(home)
        if token is None:
            return None

        # Read-only URI connection. Short timeout so busy WAL doesn't stall
        # the hot path.
        conn = sqlite3.connect(
            f"file:{resolved}?mode=ro", uri=True, timeout=0.5,
        )
        try:
            # Verify install binding first — cheap SELECT.
            row = conn.execute(
                "SELECT value FROM slm_meta WHERE key='install_token_hmac'",
            ).fetchone()
            if row is None:
                return None
            expected = _expected_binding_hmac(token)
            if not hmac.compare_digest(row[0], expected):
                return None

            now = int(time.time())
            row = conn.execute(
                """
                SELECT content, fact_ids, provenance, computed_at, byte_size
                FROM context_entries
                WHERE session_id=? AND topic_sig=?
                  AND computed_at > ?
                """,
                (session_id, topic_sig, now - TTL_SECONDS),
            ).fetchone()
        finally:
            try:
                conn.close()
            except sqlite3.Error:  # pragma: no cover — defensive
                pass

        if row is None:
            return None

        try:
            fact_ids = json.loads(row[1])
            if not isinstance(fact_ids, list):
                fact_ids = []
        except (ValueError, TypeError):
            fact_ids = []

        return CacheEntry(
            session_id=session_id,
            topic_sig=topic_sig,
            content=row[0],
            fact_ids=fact_ids,
            provenance=row[2],
            computed_at=int(row[3]),
            byte_size=int(row[4]),
        )
    except Exception:  # pragma: no cover — last-resort fail-open
        return None


__all__ = (
    "CACHE_DB_DEFAULT",
    "CLEANUP_HORIZON_SECONDS",
    "CacheEntry",
    "ContextCache",
    "MAX_BYTES",
    "MAX_CONTENT_CHARS",
    "SCHEMA_VERSION",
    "TTL_SECONDS",
    "read_entry_fast",
)
