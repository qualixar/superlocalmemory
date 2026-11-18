# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""DB-backed recall queue — schema, enqueue, claim, complete, poll.

Part of Qualixar | Author: Varun Pratap Bhardwaj
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Iterable, Optional

from superlocalmemory.core.safe_fs import _safe_open_db


class QueueTimeoutError(Exception):
    def __init__(self, request_id: str, elapsed_s: float) -> None:
        self.request_id = request_id
        self.elapsed_s = elapsed_s
        super().__init__(f"poll timed out after {elapsed_s:.2f}s")


class QueueCancelledError(Exception):
    def __init__(self, request_id: str) -> None:
        self.request_id = request_id
        super().__init__(f"request {request_id} was cancelled")


class DeadLetterError(Exception):
    def __init__(self, request_id: str, reason: str) -> None:
        self.request_id = request_id
        self.reason = reason
        super().__init__(f"request {request_id}: {reason}")


_SCHEMA = """
CREATE TABLE IF NOT EXISTS recall_requests (
    request_id        TEXT    PRIMARY KEY,
    query_hash        TEXT    NOT NULL,
    job_type          TEXT    NOT NULL DEFAULT 'recall',
    idempotency_key   TEXT,
    session_id        TEXT    NOT NULL DEFAULT '',
    agent_id          TEXT    NOT NULL DEFAULT '',
    namespace         TEXT    NOT NULL DEFAULT '',
    tenant_id         TEXT    NOT NULL DEFAULT '',
    query             TEXT    NOT NULL DEFAULT '',
    limit_n           INTEGER NOT NULL DEFAULT 10,
    mode              TEXT    NOT NULL DEFAULT 'B',
    priority          TEXT    NOT NULL DEFAULT 'high',
    weight            INTEGER NOT NULL DEFAULT 70,
    claim_expires_at  REAL,
    received          INTEGER NOT NULL DEFAULT 0,
    completed         INTEGER NOT NULL DEFAULT 0,
    cancelled         INTEGER NOT NULL DEFAULT 0,
    dead_letter       INTEGER NOT NULL DEFAULT 0,
    result_json       TEXT,
    error_reason      TEXT,
    subscriber_count  INTEGER NOT NULL DEFAULT 1,
    last_subscribe_at REAL,
    created_at        REAL    NOT NULL,
    worker_pid        INTEGER,
    worker_create_time INTEGER,
    worker_progress   TEXT,
    stall_timeout_s   REAL,
    cost_usd          REAL    NOT NULL DEFAULT 0.0,
    deadline_at       REAL,
    CHECK (completed IN (0, 1)),
    CHECK (cancelled IN (0, 1)),
    CHECK (dead_letter IN (0, 1)),
    CHECK (completed + cancelled + dead_letter <= 1),
    CHECK (NOT (completed = 1 AND result_json IS NULL)),
    CHECK (subscriber_count >= 0)
);

CREATE INDEX IF NOT EXISTS idx_recall_visible
    ON recall_requests(priority, created_at, request_id)
    WHERE completed = 0 AND cancelled = 0 AND dead_letter = 0;

CREATE INDEX IF NOT EXISTS idx_recall_dedup
    ON recall_requests(query_hash)
    WHERE completed = 0 AND cancelled = 0 AND dead_letter = 0;

CREATE UNIQUE INDEX IF NOT EXISTS idx_recall_idem_key
    ON recall_requests(job_type, idempotency_key)
    WHERE idempotency_key IS NOT NULL
          AND completed = 0 AND cancelled = 0 AND dead_letter = 0;
"""


def _now() -> float:
    return time.time()


def _make_request_id() -> str:
    return "r-" + uuid.uuid4().hex[:12]


def _query_hash(
    *, session_id: str, agent_id: str, query: str, limit_n: int,
    mode: str, tenant_id: str, namespace: str = "",
) -> str:
    blob = "||".join((
        tenant_id, namespace, session_id, agent_id, mode, str(limit_n), query,
    )).encode("utf-8")
    return hashlib.blake2b(blob, digest_size=16).hexdigest()


class RecallQueue:
    def __init__(self, db_path: Path | str) -> None:
        self._db_path = Path(db_path)
        self._lock = threading.RLock()
        self._conn = _safe_open_db(self._db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.execute("PRAGMA mmap_size=67108864")
        for stmt in _SCHEMA.strip().split(";"):
            s = stmt.strip()
            if s:
                self._conn.execute(s)
        self._add_missing_columns()
        self._closed = False

    def _add_missing_columns(self) -> None:
        """A queue file written before 4.2.30 has no ``deadline_at`` (A-14): add it."""
        have = {r["name"] for r in self._conn.execute("PRAGMA table_info(recall_requests)")}
        if "deadline_at" not in have:
            self._conn.execute("ALTER TABLE recall_requests ADD COLUMN deadline_at REAL")

    # ----- schema-level helpers (tests / ops) -----
    def _raw_execute(self, sql: str, params: Iterable[Any] = ()) -> None:
        with self._lock:
            self._conn.execute(sql, tuple(params))

    def _get_row(self, request_id: str) -> Optional[sqlite3.Row]:
        with self._lock:
            cur = self._conn.execute(
                "SELECT * FROM recall_requests WHERE request_id = ?",
                (request_id,),
            )
            return cur.fetchone()

    def _force_cancelled(self, request_id: str) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE recall_requests SET cancelled=1 WHERE request_id=?",
                (request_id,),
            )

    # ----- enqueue -----
    def enqueue(
        self,
        *,
        query: str,
        limit_n: int,
        mode: str,
        agent_id: str,
        session_id: str,
        tenant_id: str = "",
        namespace: str = "",
        priority: str = "high",
        stall_timeout_s: float = 25.0,
        deadline_s: float | None = None,
    ) -> str:
        """Queue one recall. ``deadline_s`` is how long the client will wait: stored as an
        absolute time, so an answer that finishes after it can be dropped, not recorded as
        delivered (A-14). A request already queued by the same query gains the later deadline."""
        qhash = _query_hash(
            session_id=session_id, agent_id=agent_id, query=query,
            limit_n=limit_n, mode=mode, tenant_id=tenant_id,
            namespace=namespace,
        )
        deadline_at = None if deadline_s is None else _now() + float(deadline_s)
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                cur = self._conn.execute(
                    "SELECT request_id FROM recall_requests "
                    "WHERE query_hash = ? AND completed = 0 "
                    "AND cancelled = 0 AND dead_letter = 0 "
                    "LIMIT 1",
                    (qhash,),
                )
                existing = cur.fetchone()
                if existing is not None:
                    rid = existing["request_id"]
                    self._conn.execute(
                        "UPDATE recall_requests "
                        "SET subscriber_count = subscriber_count + 1, "
                        "    last_subscribe_at = ?, "
                        "    deadline_at = CASE WHEN ? IS NULL THEN deadline_at "
                        "                       WHEN deadline_at IS NULL THEN ? "
                        "                       ELSE MAX(deadline_at, ?) END "
                        "WHERE request_id = ?",
                        (_now(), deadline_at, deadline_at, deadline_at, rid),
                    )
                    self._conn.execute("COMMIT")
                    return rid
                rid = _make_request_id()
                self._conn.execute(
                    "INSERT INTO recall_requests "
                    "(request_id, query_hash, job_type, session_id, agent_id, "
                    " namespace, tenant_id, query, limit_n, mode, priority, "
                    " weight, created_at, subscriber_count, last_subscribe_at, "
                    " stall_timeout_s, deadline_at) "
                    "VALUES (?, ?, 'recall', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?)",
                    (
                        rid, qhash, session_id, agent_id, namespace, tenant_id,
                        query, limit_n, mode, priority,
                        70 if priority == "high" else 30,
                        _now(), _now(), stall_timeout_s, deadline_at,
                    ),
                )
                self._conn.execute("COMMIT")
                return rid
            except Exception:
                self._conn.execute("ROLLBACK")
                raise

    def enqueue_job(
        self,
        *,
        job_type: str,
        idempotency_key: str,
        agent_id: str,
        session_id: str,
        priority: str = "low",
        stall_timeout_s: float = 40.0,
        query: str = "",
    ) -> str:
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                cur = self._conn.execute(
                    "SELECT request_id FROM recall_requests "
                    "WHERE job_type = ? AND idempotency_key = ? "
                    "AND completed = 0 AND cancelled = 0 AND dead_letter = 0 "
                    "LIMIT 1",
                    (job_type, idempotency_key),
                )
                existing = cur.fetchone()
                if existing is not None:
                    self._conn.execute("COMMIT")
                    return existing["request_id"]
                rid = _make_request_id()
                self._conn.execute(
                    "INSERT INTO recall_requests "
                    "(request_id, query_hash, job_type, idempotency_key, "
                    " session_id, agent_id, query, priority, weight, "
                    " created_at, subscriber_count, last_subscribe_at, "
                    " stall_timeout_s) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?)",
                    (
                        rid, idempotency_key, job_type, idempotency_key,
                        session_id, agent_id, query, priority,
                        70 if priority == "high" else 30,
                        _now(), _now(), stall_timeout_s,
                    ),
                )
                self._conn.execute("COMMIT")
                return rid
            except Exception:
                self._conn.execute("ROLLBACK")
                raise

    # ----- claim / complete / fence -----
    def claim_pending(
        self,
        *,
        priority: str = "high",
        stall_timeout_s: float = 25.0,
    ) -> Optional[dict[str, Any]]:
        now = _now()
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                cur = self._conn.execute(
                    "SELECT request_id, received FROM recall_requests "
                    "WHERE priority = ? "
                    "AND completed = 0 AND cancelled = 0 AND dead_letter = 0 "
                    "AND (claim_expires_at IS NULL OR claim_expires_at < ?) "
                    "ORDER BY created_at, request_id LIMIT 1",
                    (priority, now),
                )
                row = cur.fetchone()
                if row is None:
                    self._conn.execute("COMMIT")
                    return None
                rid = row["request_id"]
                received = row["received"] + 1
                expires = now + stall_timeout_s
                self._conn.execute(
                    "UPDATE recall_requests "
                    "SET received = ?, claim_expires_at = ? "
                    "WHERE request_id = ?",
                    (received, expires, rid),
                )
                self._conn.execute("COMMIT")
                out = self._get_row(rid)
                return dict(out) if out is not None else None
            except Exception:
                self._conn.execute("ROLLBACK")
                raise

    def complete(
        self, request_id: str, *, received: int, result_json: str,
    ) -> int:
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                cur = self._conn.execute(
                    "UPDATE recall_requests "
                    "SET completed = 1, result_json = ? "
                    "WHERE request_id = ? AND received = ? "
                    "AND completed = 0 AND cancelled = 0 AND dead_letter = 0",
                    (result_json, request_id, received),
                )
                self._conn.execute("COMMIT")
            except Exception:
                self._conn.execute("ROLLBACK")
                raise
            if cur.rowcount == 0:
                import logging as _log
                _log.getLogger(__name__).warning(
                    "fenced out: request_id=%s received=%d (stale or already terminal)",
                    request_id, received,
                )
            return cur.rowcount

    def cancel(self, request_id: str) -> None:
        """The client is gone: drop a request that has not completed (``cancelled = 1``)."""
        with self._lock:
            self._conn.execute(
                "UPDATE recall_requests SET cancelled = 1 "
                "WHERE request_id = ? AND completed = 0 AND dead_letter = 0",
                (request_id,),
            )

    def mark_dead_letter(self, request_id: str, *, reason: str) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE recall_requests "
                "SET dead_letter = 1, error_reason = ? "
                "WHERE request_id = ? AND completed = 0 AND cancelled = 0",
                (reason, request_id),
            )

    def unsubscribe(self, request_id: str) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE recall_requests "
                "SET subscriber_count = MAX(0, subscriber_count - 1) "
                "WHERE request_id = ?",
                (request_id,),
            )

    # ----- poll with DLQ fast-fail -----
    _POLL_SCHEDULE = (0.05, 0.1, 0.2, 0.3, 0.5)

    def poll_result(self, request_id: str, *, timeout_s: float) -> dict[str, Any]:
        deadline = time.monotonic() + timeout_s
        idx = 0
        while True:
            row = self._get_row(request_id)
            if row is not None:
                if row["dead_letter"]:
                    raise DeadLetterError(
                        request_id,
                        reason=row["error_reason"] or "max_receives_exceeded",
                    )
                if row["cancelled"] and not row["completed"]:
                    raise QueueCancelledError(request_id)
                if row["completed"]:
                    return json.loads(row["result_json"])
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise QueueTimeoutError(request_id, timeout_s)
            sleep_for = min(
                self._POLL_SCHEDULE[min(idx, len(self._POLL_SCHEDULE) - 1)],
                remaining,
            )
            time.sleep(sleep_for)
            idx += 1

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._conn.close()
        except Exception:
            pass
