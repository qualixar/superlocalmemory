# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V4 | https://qualixar.com | https://varunpratap.com

"""Durable remote outbox for SLM mesh (3b-1).

Persists unsent remote mesh messages in the shared SQLite DB so they
survive peer downtime and process restarts. A background drain in
RemoteSyncClient._sync_loop re-attempts delivery with exponential
back-off + jitter.

Design choices documented in class docstring.
"""

from __future__ import annotations

import json
import logging
import math
import random
import sqlite3
import time
from typing import Any

logger = logging.getLogger("superlocalmemory.mesh.outbox_remote")

# ---------------------------------------------------------------------------
# Constants — aligned with broker.py values
# ---------------------------------------------------------------------------
#: 48h TTL matches broker MESSAGE_TTL_HOURS so remote dead-letter horizon
#: is consistent with the local (broker-side) message lifetime.
_TTL_SECONDS: int = 48 * 3600

#: Per-peer cap matches broker MAX_QUEUED_PER_TARGET = 50.
_CAP_PER_PEER: int = 50

#: Maximum rows returned by due() per cycle. Caps how long the sync thread
#: blocks in a single drain pass (each row can take up to 10s on timeout).
_BATCH_LIMIT: int = 20

#: Hard dead-letter threshold. After this many attempts the row is deleted.
_MAX_RETRIES: int = 12

#: Backoff formula: min(BASE * 2^retry, CAP) * jitter_factor.
_BACKOFF_BASE: float = 5.0
_BACKOFF_CAP: float = 300.0  # 5 minutes

#: ±25% jitter fraction applied to the raw backoff to prevent thundering-herd
#: when many stalled sends all become due simultaneously after a peer restart.
_JITTER_FRACTION: float = 0.25

# ---------------------------------------------------------------------------
# DDL
# ---------------------------------------------------------------------------
_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS mesh_outbox_remote (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    peer_url       TEXT    NOT NULL,
    to_peer        TEXT    NOT NULL,
    payload        TEXT    NOT NULL,
    headers        TEXT,
    retry_count    INTEGER NOT NULL DEFAULT 0,
    next_retry_at  REAL    NOT NULL,
    created_at     REAL    NOT NULL,
    expires_at     REAL    NOT NULL
);
"""

_CREATE_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_outbox_remote_next_retry
    ON mesh_outbox_remote (next_retry_at);
"""

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _backoff(retry_count: int) -> float:
    """Exponential backoff with ±25% jitter.

    Formula: min(BASE * 2^retry_count, CAP) × (1 + U[-0.25, +0.25])
    Floor at 1.0s so a jitter-deflated value stays positive.

    The jitter prevents the thundering-herd problem: when a peer restarts
    after downtime, all enqueued rows that became due at roughly the same
    time would otherwise pile onto the peer simultaneously.
    """
    raw = min(_BACKOFF_BASE * math.pow(2, retry_count), _BACKOFF_CAP)
    jitter = raw * _JITTER_FRACTION * (2.0 * random.random() - 1.0)
    return max(1.0, raw + jitter)


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------


class RemoteOutbox:
    """Durable store for unsent remote mesh messages (3b-1).

    Persists messages into the shared SQLite DB (same file as the mesh
    broker) so sends that fail due to peer downtime survive restarts.

    All public methods are **fail-soft**: SQLite errors are logged but
    never propagated to callers so the online send path is never blocked
    by outbox failures. The sole exception is ``__init__``: if the table
    cannot be created ``_active`` is set to False and every subsequent
    method becomes a no-op — callers check ``_active`` before use.

    Design choices
    --------------
    *Drop oldest on cap*: when a peer_url accumulates 50 rows, the oldest
    row is evicted before the new one is inserted. This keeps a
    permanently-down peer from filling the DB and ensures newer (more
    relevant) messages are preserved. Clients must expect at-least-once
    delivery semantics regardless.

    *No header replay*: headers stored here are informational/audit-only.
    The drain loop in RemoteSyncClient re-signs each message fresh with a
    new nonce + timestamp to avoid stale-timestamp rejections at the peer.

    *Bounded drain*: due() returns at most _BATCH_LIMIT rows so each
    sync-loop cycle has a predictable worst-case duration.
    """

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._active: bool = False
        self._init_table()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, check_same_thread=False, timeout=5.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _init_table(self) -> None:
        """Create the outbox table and index idempotently.

        Uses CREATE TABLE/INDEX IF NOT EXISTS so calling this on an existing
        DB with the table already present is safe (backward-compatible).
        Sets _active=False on any error so every subsequent method no-ops.
        """
        try:
            conn = self._connect()
            try:
                conn.execute(_CREATE_TABLE_SQL)
                conn.execute(_CREATE_INDEX_SQL)
                conn.commit()
            finally:
                conn.close()
            self._active = True
            logger.debug("RemoteOutbox: table ready at %s", self._db_path)
        except sqlite3.Error as exc:
            logger.error(
                "RemoteOutbox: failed to initialise table at %s — outbox "
                "disabled (online send path unaffected): %s",
                self._db_path,
                exc,
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def enqueue(
        self,
        peer_url: str,
        to_peer: str,
        payload: dict[str, Any],
        headers: dict[str, str] | None,
        now: float,
    ) -> None:
        """Persist a message for later delivery.

        *Cap enforcement*: when peer_url already has _CAP_PER_PEER (50) rows,
        the oldest is evicted (FIFO) before insertion so a permanently-down
        peer never exhausts the DB.

        *TTL*: 48h from ``now`` — consistent with broker MESSAGE_TTL_HOURS.

        *next_retry_at = now* so the drain loop attempts delivery on the
        very next sync cycle (~30s after enqueueing).

        Args:
            peer_url: Remote peer base URL (cap scope key).
            to_peer: Target peer ID on remote machine.
            payload: Message dict — serialised as JSON.
            headers: HTTP headers at time of original send — serialised as
                JSON for audit purposes. The drain re-signs fresh headers
                rather than replaying these to avoid stale-timestamp errors.
            now: Epoch seconds (caller-supplied for testability).
        """
        if not self._active:
            return
        try:
            payload_str = json.dumps(payload)
            headers_str = json.dumps(headers) if headers is not None else None
            expires_at = now + _TTL_SECONDS

            conn = self._connect()
            try:
                count_row = conn.execute(
                    "SELECT COUNT(*) FROM mesh_outbox_remote WHERE peer_url=?",
                    (peer_url,),
                ).fetchone()
                count = count_row[0] if count_row else 0

                if count >= _CAP_PER_PEER:
                    # Evict the single oldest row to make room.
                    conn.execute(
                        """
                        DELETE FROM mesh_outbox_remote
                        WHERE id = (
                            SELECT id FROM mesh_outbox_remote
                            WHERE peer_url = ?
                            ORDER BY created_at ASC
                            LIMIT 1
                        )
                        """,
                        (peer_url,),
                    )
                    logger.debug(
                        "RemoteOutbox: per-peer cap reached for %s — evicted oldest row",
                        peer_url,
                    )

                conn.execute(
                    """
                    INSERT INTO mesh_outbox_remote
                        (peer_url, to_peer, payload, headers,
                         retry_count, next_retry_at, created_at, expires_at)
                    VALUES (?, ?, ?, ?, 0, ?, ?, ?)
                    """,
                    (peer_url, to_peer, payload_str, headers_str, now, now, expires_at),
                )
                conn.commit()
                logger.debug(
                    "RemoteOutbox: enqueued message for %s → %s",
                    peer_url,
                    to_peer,
                )
            finally:
                conn.close()
        except sqlite3.Error as exc:
            logger.error("RemoteOutbox.enqueue: DB error: %s", exc)

    def due(self, now: float) -> list[sqlite3.Row]:
        """Return up to _BATCH_LIMIT rows ready for re-delivery (oldest first).

        A row is due when ``next_retry_at <= now`` and ``expires_at > now``.
        Bounded by _BATCH_LIMIT to cap sync-thread blocking time.
        """
        if not self._active:
            return []
        try:
            conn = self._connect()
            try:
                rows = conn.execute(
                    """
                    SELECT id, peer_url, to_peer, payload, headers,
                           retry_count, next_retry_at, created_at, expires_at
                    FROM mesh_outbox_remote
                    WHERE next_retry_at <= ? AND expires_at > ?
                    ORDER BY next_retry_at ASC
                    LIMIT ?
                    """,
                    (now, now, _BATCH_LIMIT),
                ).fetchall()
                return list(rows)
            finally:
                conn.close()
        except sqlite3.Error as exc:
            logger.error("RemoteOutbox.due: DB error: %s", exc)
            return []

    def mark_retry(self, row_id: int, now: float) -> None:
        """Increment retry_count and schedule the next attempt with backoff.

        Deletes the row (dead-letters it) if:
        - ``retry_count + 1 > _MAX_RETRIES`` — too many failures, or
        - ``now >= expires_at`` — TTL elapsed since enqueue.

        Backoff uses ±25% jitter to prevent thundering-herd.
        """
        if not self._active:
            return
        try:
            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT retry_count, expires_at FROM mesh_outbox_remote WHERE id=?",
                    (row_id,),
                ).fetchone()

                if row is None:
                    return  # Already deleted by a concurrent drain cycle

                new_count = row["retry_count"] + 1
                expires_at = row["expires_at"]

                if new_count > _MAX_RETRIES or now >= expires_at:
                    conn.execute(
                        "DELETE FROM mesh_outbox_remote WHERE id=?",
                        (row_id,),
                    )
                    conn.commit()
                    logger.debug(
                        "RemoteOutbox: row %d dead-lettered "
                        "(retries=%d, ttl_expired=%s)",
                        row_id,
                        new_count,
                        now >= expires_at,
                    )
                    return

                next_retry = now + _backoff(new_count)
                conn.execute(
                    """
                    UPDATE mesh_outbox_remote
                       SET retry_count = ?, next_retry_at = ?
                     WHERE id = ?
                    """,
                    (new_count, next_retry, row_id),
                )
                conn.commit()
                logger.debug(
                    "RemoteOutbox: row %d rescheduled (attempt=%d, next=+%.0fs)",
                    row_id,
                    new_count,
                    next_retry - now,
                )
            finally:
                conn.close()
        except sqlite3.Error as exc:
            logger.error("RemoteOutbox.mark_retry(%d): DB error: %s", row_id, exc)

    def delete(self, row_id: int) -> None:
        """Remove a successfully-delivered row."""
        if not self._active:
            return
        try:
            conn = self._connect()
            try:
                conn.execute(
                    "DELETE FROM mesh_outbox_remote WHERE id=?",
                    (row_id,),
                )
                conn.commit()
            finally:
                conn.close()
        except sqlite3.Error as exc:
            logger.error("RemoteOutbox.delete(%d): DB error: %s", row_id, exc)

    def prune_expired(self, now: float) -> None:
        """Delete all rows whose TTL has elapsed."""
        if not self._active:
            return
        try:
            conn = self._connect()
            try:
                deleted = conn.execute(
                    "DELETE FROM mesh_outbox_remote WHERE expires_at <= ?",
                    (now,),
                ).rowcount
                conn.commit()
                if deleted:
                    logger.debug("RemoteOutbox: pruned %d expired rows", deleted)
            finally:
                conn.close()
        except sqlite3.Error as exc:
            logger.error("RemoteOutbox.prune_expired: DB error: %s", exc)

    def row_count(self, peer_url: str | None = None) -> int:
        """Return total rows in the outbox, optionally scoped to a peer_url.

        Utility method for testing and monitoring.
        """
        if not self._active:
            return 0
        try:
            conn = self._connect()
            try:
                if peer_url is not None:
                    row = conn.execute(
                        "SELECT COUNT(*) FROM mesh_outbox_remote WHERE peer_url=?",
                        (peer_url,),
                    ).fetchone()
                else:
                    row = conn.execute(
                        "SELECT COUNT(*) FROM mesh_outbox_remote",
                    ).fetchone()
                return row[0] if row else 0
            finally:
                conn.close()
        except sqlite3.Error as exc:
            logger.error("RemoteOutbox.row_count: DB error: %s", exc)
            return 0
