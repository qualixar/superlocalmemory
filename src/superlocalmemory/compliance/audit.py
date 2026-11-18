# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Tamper-evident hash-chain audit log for compliance.

Every operation is logged with a SHA-256 hash that includes the previous
entry's hash, creating a chain. Changing any recorded field of an entry breaks
the chain and is detectable via verify_integrity() / verify_report().

**Hash version 2 (4.1.25, audit OPS-01).** Version 1 hashed six fields run
together with no separators and left the metadata out entirely, so moving
characters between the actor and the workspace, or rewriting a record's
target, outcome or fact ids, kept every hash valid. Version 2 hashes one
canonical JSON object holding every stored field (including the metadata text),
so a field boundary cannot move and nothing shown is outside the hash. Entries
written before the upgrade stay version 1 and are reported as such: their order
and their six fields are checked, their metadata is not, and they are never
presented as fully verified. A version-1 entry after a version-2 one is a
downgrade and fails verification.

**Hash version 3 (4.2.30, WS-5, D-16).** While the daemon holds the signing key
from the OS keychain (``trust.signing`` on), each new link is an HMAC-SHA256 under
that key instead of a plain hash (``compliance/audit_keyed.py``). An anchor entry
is written before the first keyed link and whenever the key changes. Plain links
written before, and by processes that do not hold the key, stay verifiable by
their plain hash.

What this proves, and what it does not: a plain link is SHA-256, not a keyed
MAC, so someone who can rewrite this file can also recompute it; a keyed link
cannot be recomputed without the keychain's key, so a copy of the file made
without the user's login session cannot be rewritten and still verify (it does
not stop a process running as the user). The external anchor (count + head hash
in a sidecar file) catches truncation and a replaced tail; it does not stop an
attacker who controls both files.

The audit chain uses its OWN sqlite3 connection (not shared DB manager)
for independence — audit must survive even if the main DB is corrupted.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Union

from superlocalmemory.compliance.audit_keyed import (
    ANCHOR_OPERATION,
    KEYED_VERSION,
    Tally,
    anchor_metadata,
    anchor_reason,
    check_row,
    keyed_hash,
)

logger = logging.getLogger(__name__)

_GENESIS_HASH = "genesis"
# This database is audit-owned. Its header records that anchoring has ever
# occurred, independently of the marker table and even of the retained rows.
_ANCHOR_SCHEMA_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS audit_chain (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    operation TEXT NOT NULL,
    agent_id TEXT DEFAULT '',
    profile_id TEXT DEFAULT '',
    content_hash TEXT DEFAULT '',
    prev_hash TEXT DEFAULT '',
    event_hash TEXT NOT NULL,
    metadata TEXT DEFAULT '{}',
    hash_version INTEGER NOT NULL DEFAULT 2,
    key_id TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_audit_chain_profile ON audit_chain(profile_id, id);
CREATE TABLE IF NOT EXISTS audit_anchor_state (
    singleton INTEGER PRIMARY KEY CHECK(singleton = 1),
    created_at TEXT NOT NULL
);
"""

#: The hash version new entries are written with.
HASH_VERSION = 2
_V2_DOMAIN = "superlocalmemory-audit-v2"


def _compute_hash(
    prev_hash: str,
    operation: str,
    agent_id: str,
    profile_id: str,
    content_hash: str,
    timestamp: str,
) -> str:
    """Version-1 hash, kept to verify entries written before 4.1.25.

    Concatenates six fields with no separators and omits the metadata, so it
    does not bind field boundaries or the record's details (OPS-01).
    """
    payload = (
        f"{prev_hash}{operation}{agent_id}"
        f"{profile_id}{content_hash}{timestamp}"
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _compute_hash_v2(
    prev_hash: str,
    operation: str,
    agent_id: str,
    profile_id: str,
    content_hash: str,
    timestamp: str,
    metadata: str,
) -> str:
    """Version-2 hash: every stored field, framed as canonical JSON."""
    record = {
        "domain": _V2_DOMAIN, "v": 2, "prev": prev_hash, "ts": timestamp,
        "op": operation, "agent": agent_id, "profile": profile_id,
        "content": content_hash, "meta": metadata,
    }
    canonical = json.dumps(record, sort_keys=True, separators=(",", ":"),
                           ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


#: Versions whose entries cover every stored field (plain canonical JSON, or keyed).
FULL_VERSIONS = (HASH_VERSION, KEYED_VERSION)


def _entry_hash(row: dict) -> str:
    fields = (row["prev_hash"], row["operation"], row["agent_id"] or "",
              row["profile_id"] or "", row["content_hash"] or "", row["timestamp"])
    if int(row.get("hash_version") or 1) >= 2:
        return _compute_hash_v2(*fields, row.get("metadata") or "")
    return _compute_hash(*fields)


class AuditChain:
    """Tamper-evident hash-chain audit log for compliance.

    Every operation is logged with a hash that includes the previous hash,
    creating a chain. Tampering with any entry breaks the chain.

    Uses its own SQLite connection for independence from main DB.
    """

    def __init__(self, db_path: Optional[Union[str, Path]] = None, *,
                 read_only: bool = False) -> None:
        """``read_only`` opens an existing file with ``mode=ro`` and changes nothing (not the
        schema, not the anchor, not permissions): for ``slm doctor``, which must not alter
        what it checks. Only verification and queries work on such a chain."""
        self._read_only = bool(read_only and db_path)
        self._db_path = str(db_path) if db_path else ":memory:"
        self._is_memory = self._db_path == ":memory:"
        # External truncation anchor: a sidecar file holding the current chain
        # length and head hash. The in-chain hash links detect mutation of a
        # retained prefix, but not truncation of a suffix (or the whole chain);
        # comparing against this externally-persisted (count, head) closes that
        # gap. Not used for in-memory chains.
        self._anchor_path = None if self._is_memory else (self._db_path + ".anchor")
        self._lock = threading.Lock()
        # For in-memory DBs, keep a persistent connection (each connect()
        # to ":memory:" creates a separate empty database).
        self._persistent_conn: Optional[sqlite3.Connection] = None
        if self._is_memory:
            self._persistent_conn = self._make_conn()
        if not self._read_only:
            self._init_db()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_conn_from_path(path: str) -> sqlite3.Connection:
        """Create a configured SQLite connection."""
        conn = sqlite3.connect(path)
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.row_factory = sqlite3.Row
            # C4: audit chain holds a tamper-evident record — keep it owner-only.
            try:
                from superlocalmemory.core.security_primitives import harden_db_perms
                harden_db_perms(path)
            except Exception:
                pass
        except BaseException:
            # Setup failed: close it now, or the file stays open (Windows
            # then cannot delete or replace it) for as long as the error lives.
            conn.close()
            raise
        return conn

    def _make_conn(self) -> sqlite3.Connection:
        """Create a new configured connection to the audit database."""
        if self._read_only:
            conn = sqlite3.connect(f"file:{self._db_path}?mode=ro", uri=True)
            conn.row_factory = sqlite3.Row
            return conn
        return self._make_conn_from_path(self._db_path)

    def _get_conn(self) -> sqlite3.Connection:
        """Get a connection to the audit database.

        For in-memory databases, returns the persistent connection.
        For file databases, creates a new connection each time.
        """
        if self._is_memory and self._persistent_conn is not None:
            return self._persistent_conn
        return self._make_conn()

    def _release_conn(self, conn: sqlite3.Connection) -> None:
        """Release a connection. Only closes file-based connections."""
        if not self._is_memory:
            conn.close()

    def _init_db(self) -> None:
        """Initialize the audit_chain table; mark an older chain's rows version 1.

        ``ALTER TABLE ... ADD COLUMN`` with a constant default rewrites no
        rows, so this is instant on a chain of any length.
        """
        conn = self._get_conn()
        try:
            conn.executescript(_SCHEMA)
            columns = {row[1] for row in conn.execute("PRAGMA table_info(audit_chain)")}
            if "hash_version" not in columns:
                conn.execute(
                    "ALTER TABLE audit_chain ADD COLUMN hash_version "
                    "INTEGER NOT NULL DEFAULT 1"
                )
            if "key_id" not in columns:
                conn.execute(
                    "ALTER TABLE audit_chain ADD COLUMN key_id TEXT NOT NULL DEFAULT ''"
                )
            conn.commit()
            self._upgrade_anchor(conn)
        finally:
            self._release_conn(conn)

    def _upgrade_anchor(self, conn: sqlite3.Connection) -> None:
        """Initialize a pre-anchor store once, recording the upgrade in its chain."""
        if self._is_memory:
            return
        conn.execute("BEGIN IMMEDIATE")
        try:
            recorded = conn.execute("SELECT 1 FROM audit_anchor_state").fetchone()
            anchored_schema = conn.execute("PRAGMA user_version").fetchone()[0]
            if recorded or anchored_schema >= _ANCHOR_SCHEMA_VERSION:
                self._record_anchor_created(conn)
                conn.commit()
                return
            anchor, problem = self._anchor_state()
            if anchor is not None:
                self._record_anchor_created(conn)
            elif problem == "missing":
                rows = [dict(row) for row in conn.execute("SELECT * FROM audit_chain ORDER BY id")]
                if rows and self._link_failure(rows) is None:
                    head = self._append_event(conn, "anchor_initialized_on_upgrade", "system", "",
                                              "", {"previous_entries": len(rows)})
                    self._record_anchor_created(conn)
                    self._write_anchor(
                        int(conn.execute("SELECT COUNT(*) FROM audit_chain").fetchone()[0]), head)
            conn.commit()
        except BaseException:
            conn.rollback()
            raise

    @staticmethod
    def _record_anchor_created(conn: sqlite3.Connection) -> None:
        # Dropping audit_anchor_state cannot turn an anchored store into a
        # legacy one. Restore its table marker on reopen without re-anchoring.
        if conn.execute("PRAGMA user_version").fetchone()[0] < _ANCHOR_SCHEMA_VERSION:
            conn.execute(f"PRAGMA user_version = {_ANCHOR_SCHEMA_VERSION}")
        conn.execute("INSERT OR IGNORE INTO audit_anchor_state VALUES (1, ?)",
                     (datetime.now(timezone.utc).isoformat(),))

    @staticmethod
    def _signer() -> Any:
        """The signing key this process holds (the daemon, key loaded at its start), else None.

        Memory only: never asks the keychain, so logging an event never waits on it.
        """
        try:
            from superlocalmemory.trust import signing_key

            return signing_key.current()
        except Exception:  # noqa: BLE001 - an unkeyed link is still a valid link
            return None

    def _append_event(self, conn, operation, agent_id, profile_id, content_hash, metadata):
        signer = self._signer()
        if signer is not None:
            self._ensure_anchor(conn, signer)
        return self._insert_link(conn, operation, agent_id, profile_id, content_hash,
                                 metadata, signer)

    def _ensure_anchor(self, conn, signer) -> None:
        """Write the anchor entry the next keyed link needs: the first one, or after a key change."""
        row = conn.execute("SELECT key_id FROM audit_chain WHERE hash_version = ? "
                           "ORDER BY id DESC LIMIT 1", (KEYED_VERSION,)).fetchone()
        last = row[0] if row else None
        empty = conn.execute("SELECT 1 FROM audit_chain LIMIT 1").fetchone() is None
        reason = anchor_reason(last, signer, empty)
        if reason:
            self._insert_link(conn, ANCHOR_OPERATION, "system", "", "",
                              anchor_metadata(reason, signer, last), signer)

    def _insert_link(self, conn, operation, agent_id, profile_id, content_hash, metadata, signer):
        ts = datetime.now(timezone.utc).isoformat()
        metadata_str = json.dumps(metadata or {}, sort_keys=True, default=str)
        prev_hash = self._get_last_hash(conn)
        if signer is not None:
            version, key_id = KEYED_VERSION, signer.key_id
            event_hash = keyed_hash(signer, prev_hash, operation, agent_id or "",
                                    profile_id or "", content_hash or "", ts, metadata_str)
        else:
            version, key_id = HASH_VERSION, ""
            event_hash = _compute_hash_v2(prev_hash, operation, agent_id or "", profile_id or "",
                                          content_hash or "", ts, metadata_str)
        conn.execute(
            "INSERT INTO audit_chain (timestamp, operation, agent_id, profile_id, "
            "content_hash, prev_hash, event_hash, metadata, hash_version, key_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (ts, operation, agent_id or "", profile_id or "", content_hash or "",
             prev_hash, event_hash, metadata_str, version, key_id))
        return event_hash

    def _get_last_hash(self, conn: sqlite3.Connection) -> str:
        """Get the hash of the most recent entry, or genesis."""
        row = conn.execute(
            "SELECT event_hash FROM audit_chain "
            "ORDER BY id DESC LIMIT 1"
        ).fetchone()
        return row["event_hash"] if row else _GENESIS_HASH

    def _write_anchor(self, count: int, head: str) -> None:
        """Persist the external (count, head) truncation anchor."""
        if not self._anchor_path:
            return
        try:
            payload = json.dumps({"count": int(count), "head": head}, sort_keys=True)
            with open(self._anchor_path, "w", encoding="utf-8") as fh:
                fh.write(payload)
            try:
                from superlocalmemory.core.security_primitives import harden_db_perms
                harden_db_perms(self._anchor_path)
            except Exception:  # noqa: BLE001
                pass
        except Exception as exc:  # noqa: BLE001
            logger.warning("audit anchor write failed: %s", exc)

    def _read_anchor(self) -> Optional[dict[str, Any]]:
        """Read the external truncation anchor, or None if absent/unreadable."""
        return self._anchor_state()[0]

    def _anchor_state(self) -> tuple[Optional[dict[str, Any]], str]:
        """``(anchor, problem)``: problem is ``""``, ``"missing"`` or ``"unreadable"``."""
        if not self._anchor_path:
            return None, ""
        try:
            with open(self._anchor_path, encoding="utf-8") as fh:
                data = json.loads(fh.read())
            if isinstance(data, dict) and "count" in data and "head" in data:
                return data, ""
        except FileNotFoundError:
            return None, "missing"
        except Exception as exc:  # noqa: BLE001
            logger.warning("audit anchor read failed: %s", exc)
        return None, "unreadable"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def log(
        self,
        operation: str,
        agent_id: str = "",
        profile_id: str = "",
        content_hash: str = "",
        metadata: Optional[dict[str, Any]] = None,
    ) -> str:
        """Log an audit event. Returns the event hash.

        Args:
            operation: Type of operation (store, recall, delete, etc.).
            agent_id: ID of the agent performing the operation.
            profile_id: ID of the profile being accessed.
            content_hash: Hash of the content involved (optional).
            metadata: Additional metadata dict (optional).

        Returns:
            The SHA-256 event hash for this entry.
        """
        with self._lock:
            conn = self._get_conn()
            try:
                if not self._is_memory:
                    conn.execute("BEGIN IMMEDIATE")
                recorded = conn.execute("SELECT 1 FROM audit_anchor_state").fetchone()
                _, problem = self._anchor_state()
                event_hash = self._append_event(conn, operation, agent_id, profile_id,
                                                content_hash, metadata)
                count = conn.execute("SELECT COUNT(*) FROM audit_chain").fetchone()[0]
                if not self._is_memory and not (recorded and problem):
                    self._record_anchor_created(conn)
                    # Keep the database writer lock through the anchor update so
                    # concurrent processes cannot publish an older head last.
                    self._write_anchor(int(count), event_hash)
                conn.commit()
                return event_hash
            except BaseException:
                conn.rollback()
                raise
            finally:
                self._release_conn(conn)

    def query(
        self,
        profile_id: Optional[Union[str, tuple[str, ...]]] = None,
        agent_id: Optional[str] = None,
        operation: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """Query audit events with filters, one page at a time.

        Args:
            profile_id: Filter by profile ID.
            agent_id: Filter by agent ID.
            operation: Filter by operation type.
            start_date: ISO date string for range start.
            end_date: ISO date string for range end.
            limit: Maximum number of events to return.
            offset: Events to skip (newest first), for paging.

        Returns:
            List of event dicts ordered by id descending.
        """
        where, params = self._filters(profile_id, agent_id, operation, start_date, end_date)
        sql = (
            f"SELECT id, timestamp, operation, agent_id, profile_id, "
            f"content_hash, prev_hash, event_hash, metadata, hash_version, key_id "
            f"FROM audit_chain {where} "
            f"ORDER BY id DESC LIMIT ? OFFSET ?"
        )
        params.extend([max(0, int(limit)), max(0, int(offset))])

        with self._lock:
            conn = self._get_conn()
            try:
                rows = conn.execute(sql, params).fetchall()
                return [dict(r) for r in rows]
            finally:
                self._release_conn(conn)

    @staticmethod
    def _filters(
        profile_id: Optional[Union[str, tuple[str, ...]]], agent_id: Optional[str],
        operation: Optional[str],
        start_date: Optional[str], end_date: Optional[str],
    ) -> tuple[str, list[Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if isinstance(profile_id, (list, tuple, frozenset, set)):
            # Several workspaces (the active one plus "" = installation scope).
            spaces = sorted(profile_id)
            clauses.append(f"profile_id IN ({','.join('?' for _ in spaces)})")
            params.extend(spaces)
            profile_id = None
        for column, op, value in (
            ("profile_id", "=", profile_id), ("agent_id", "=", agent_id),
            ("operation", "=", operation), ("timestamp", ">=", start_date),
            ("timestamp", "<=", end_date),
        ):
            if value is not None:
                clauses.append(f"{column} {op} ?")
                params.append(value)
        return (f"WHERE {' AND '.join(clauses)}" if clauses else ""), params

    def count(
        self,
        profile_id: Optional[Union[str, tuple[str, ...]]] = None,
        agent_id: Optional[str] = None,
        operation: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> int:
        """How many events match, counted by the database and not by loading them.

        The Compliance view used to ``len(query(limit=100000))`` for its header:
        a hundred thousand rows loaded and discarded on every page view.
        """
        where, params = self._filters(profile_id, agent_id, operation, start_date, end_date)
        with self._lock:
            conn = self._get_conn()
            try:
                return int(conn.execute(
                    f"SELECT COUNT(*) FROM audit_chain {where}", params).fetchone()[0])
            finally:
                self._release_conn(conn)

    def verify_integrity(self) -> bool:
        """True when the whole chain verifies (see :meth:`verify_report`)."""
        return bool(self.verify_report()["ok"])

    def verify_report(self, signer: Any = None) -> dict[str, Any]:
        """Walk the chain and say exactly what was verified.

        Returns ``ok``, ``entries``, ``fully_covered`` (version-2 and keyed
        entries: every field, metadata included), ``legacy`` (version-1
        entries: order and six fields only, metadata NOT covered), ``keyed``
        (keyed links whose HMAC was checked), ``keyed_unverified`` (keyed under
        a key this process does not hold; order still checked),
        ``plain_after_keyed`` (plain links after the first keyed one),
        ``anchors``, ``failed_at`` (entry id or None) and ``reason``.
        ``signer`` is the key to check keyed links with; default: the one this
        process holds.
        """
        return self.verify_loaded(self.load_rows(), signer)

    def load_rows(self) -> list[dict]:
        """Every entry, oldest first (what a full walk of the chain reads)."""
        conn = self._get_conn()
        try:
            # ``*``: a chain written before 4.2.30 has no key_id column until it is next opened.
            return [dict(r) for r in conn.execute(
                "SELECT * FROM audit_chain ORDER BY id").fetchall()]
        finally:
            self._release_conn(conn)

    def verify_loaded(self, rows: list[dict], signer: Any = None) -> dict[str, Any]:
        """The :meth:`verify_report` answer for entries already read."""
        signer = signer if signer is not None else self._signer()
        legacy = sum(1 for r in rows if r.get("hash_version", 1) == 1)
        covered = sum(1 for r in rows if r.get("hash_version") in FULL_VERSIONS)
        tally = Tally()
        report: dict[str, Any] = {
            "ok": True, "entries": len(rows), "fully_covered": covered,
            "legacy": legacy, "failed_at": None, "reason": "",
        }
        failure = (self._anchor_failure(rows)
                   or self._link_walk(rows, _GENESIS_HASH, False, signer, tally)[0])
        report.update(tally.as_report())
        if failure is not None:
            report.update(ok=False, failed_at=failure[0], reason=failure[1])
            logger.warning("Audit chain verification failed at %s: %s", *failure)
        return report

    def _anchor_failure(self, rows: list[dict]) -> Optional[tuple[Optional[int], str]]:
        """Truncation or a replaced tail, judged against the external anchor.

        A chain with entries and no readable anchor is reported, never passed:
        without it, deleting the tail together with the anchor would verify as
        intact (PR 248 review, finding 4). An in-memory chain has no anchor.
        """
        return self._anchor_failure_at(
            len(rows),
            lambda position: (rows[position]["id"], rows[position]["event_hash"]),
            bool(rows),
        )

    def _anchor_failure_at(self, total: int, entry_at, has_rows: bool,
                           ) -> Optional[tuple[Optional[int], str]]:
        """The anchor judgement, given the entry count and a way to read entry N."""
        anchor, problem = self._anchor_state()
        if anchor is None:
            conn = self._get_conn()
            try:
                try:
                    recorded = conn.execute("SELECT 1 FROM audit_anchor_state").fetchone()
                except sqlite3.OperationalError:  # a chain older than the anchor table (read-only)
                    recorded = None
            finally:
                self._release_conn(conn)
            if problem and (has_rows or recorded):
                return None, f"no anchor ({problem}): truncation cannot be checked"
            return None
        count = int(anchor.get("count", 0))
        if total < count:
            return None, f"truncated: the anchor records {count} entries, found {total}"
        if count > 0:
            entry_id, entry_hash = entry_at(count - 1)
            if entry_hash != anchor.get("head"):
                return entry_id, "the entry the anchor records as the head was replaced"
        return None

    @staticmethod
    def _link_failure(rows: list[dict]) -> Optional[tuple[Optional[int], str]]:
        return AuditChain._link_walk(rows, _GENESIS_HASH, False)[0]

    @staticmethod
    def _link_walk(rows: list[dict], expected_prev: str, seen_v2: bool,
                   signer: Any = None, tally: Optional[Tally] = None,
                   ) -> tuple[Optional[tuple[Optional[int], str]], str, bool]:
        """Check ``rows`` from a known start; returns ``(failure, last hash, seen v2)``.

        Keyed links are checked by HMAC when ``signer`` holds their key; counts go to ``tally``.
        """
        tally = tally if tally is not None else Tally()
        for row in rows:
            version = row.get("hash_version", 1)
            if version not in (1, HASH_VERSION, KEYED_VERSION):
                return (row["id"], "unsupported hash version"), expected_prev, seen_v2
            if version < 2 and seen_v2:
                return ((row["id"], "a version-1 entry follows a version-2 entry (downgrade)"),
                        expected_prev, seen_v2)
            seen_v2 = seen_v2 or version >= 2
            if row["prev_hash"] != expected_prev:
                return (row["id"], "the link to the previous entry is broken"), expected_prev, seen_v2
            if version == KEYED_VERSION:
                problem = check_row(row, signer, tally)
                if problem:
                    return (row["id"], problem), expected_prev, seen_v2
            else:
                if tally.seen_keyed and version == HASH_VERSION:
                    tally.plain_after_keyed += 1
                if _entry_hash(row) != row["event_hash"]:
                    return ((row["id"], "the entry's fields do not match its hash"),
                            expected_prev, seen_v2)
            expected_prev = row["event_hash"]
        return None, expected_prev, seen_v2

    def get_stats(self) -> dict[str, int]:
        """Get audit statistics (event counts by operation type).

        Returns:
            Dict mapping operation names to their counts, plus
            a 'total' key with the chain length.
        """
        conn = self._get_conn()
        try:
            rows = conn.execute(
                "SELECT operation, COUNT(*) AS cnt "
                "FROM audit_chain GROUP BY operation"
            ).fetchall()
            stats: dict[str, int] = {}
            total = 0
            for row in rows:
                count = row["cnt"]
                stats[row["operation"]] = count
                total += count
            stats["total"] = total
            return stats
        finally:
            self._release_conn(conn)
