# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""SLM Mesh Broker — core orchestration for P2P agent communication.

Manages peer lifecycle, scheduled cleanup, and event logging.
All operations use the shared memory.db via SQLite tables with mesh_ prefix.

Part of Qualixar | Author: Varun Pratap Bhardwaj
"""

from __future__ import annotations

import logging
import sqlite3
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, TypeVar

logger = logging.getLogger("superlocalmemory.mesh")
import os as _os

# Remote sync support (optional, try/except to avoid import issues)
try:
    from .remote_sync import RemoteSyncClient
except ImportError:
    RemoteSyncClient = None  # type: ignore

LOCAL_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})



MAX_MESSAGE_SIZE = 4096  # 4KB cap — mesh messages are notifications, not data dumps
MESSAGE_TTL_HOURS = 48   # Offline messages expire after 48h
LOCK_TTL_HOURS = 8       # M-02: file locks auto-expire so a crashed session can't deadlock a path
_NEVER_EXPIRES = "9999-12-31T23:59:59Z"  # legacy sentinel default; treated as stale/free
MAX_QUEUED_PER_TARGET = 50  # Max unread messages per broadcast/project target

_T = TypeVar("_T")
# Retry budget for a transient writer collision. 250ms was too thin under
# multi-agent load; 2000ms matches the RBAC store. 4 attempts keeps the total
# worst-case budget bounded (~2s x backoff) rather than unbounded stalls.
_WRITE_RETRY_ATTEMPTS = 4
_WRITE_RETRY_BASE_SECONDS = 0.025
_WRITE_BUSY_TIMEOUT_MS = 2000


class MeshBroker:
    """Lightweight mesh broker for SLM's unified daemon.

    Provides peer management, messaging, state, locks, and events.
    v3.4.6: broadcast, project-based routing, offline message queue.
    All methods are synchronous (called from FastAPI via run_in_executor
    or directly for quick operations).
    """

    def __init__(self, db_path: str | Path):
        self._db_path = str(db_path)
        self._started_at = time.monotonic()
        self._cleanup_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._host = _os.environ.get("SLM_MESH_HOST", "127.0.0.1")
        self._shared_secret = _os.environ.get("SLM_MESH_SHARED_SECRET", "") or None
        self._is_remote = self._host not in LOCAL_HOSTS
        self._ws_port = int(_os.environ.get("SLM_MESH_WS_PORT", "7900"))
        self._discovery_enabled = self._is_remote and _os.environ.get("SLM_MESH_DISCOVERY", "on") != "off"
        self._remote_peers: dict[str, dict] = {}
        self._remote_peers_lock = threading.RLock()
        self._peer_url: str | None = _os.environ.get("SLM_MESH_PEER_URL", "") or None
        self._sync_client: Any = None
        if self._is_remote and not self._shared_secret:
            raise RuntimeError(
                "SLM_MESH_SHARED_SECRET is required when SLM_MESH_HOST is not localhost"
            )


    # -- Remote / Multi-Machine support (v3.4.47) --

    def get_remote_peers(self, profile_id: str = "default") -> list[dict]:
        """Return peers from discovered remote brokers for this tenant.

        Remote peers carry their home ``profile_id`` in the info dict (populated
        by the /mesh/peers sync). A peer with no profile_id is a legacy remote
        and treated as 'default'. Cross-tenant peers are never returned.
        """
        with self._remote_peers_lock:
            return [
                info for info in self._remote_peers.values()
                if info.get("profile_id", "default") == profile_id
            ]

    def add_remote_peer(self, peer_id: str, info: dict) -> None:
        """Register a peer from a remote broker."""
        with self._remote_peers_lock:
            self._remote_peers[peer_id] = info

    def remove_remote_peer(self, peer_id: str) -> None:
        """Remove a remote peer."""
        with self._remote_peers_lock:
            self._remote_peers.pop(peer_id, None)

    def list_all_peers(self, profile_id: str = "default") -> list[dict]:
        """Return local + remote peers for this tenant, merged."""
        local = self.list_peers(profile_id)
        remote = self.get_remote_peers(profile_id)
        return local + remote

    def start_cleanup(self) -> None:
        """Start background cleanup thread for stale peers/messages."""
        self._cleanup_thread = threading.Thread(
            target=self._cleanup_loop, daemon=True, name="mesh-cleanup",
        )
        self._cleanup_thread.start()

        # Start remote sync client if peer URL configured or remote mode
        if RemoteSyncClient and (
            self._peer_url or (self._is_remote and self._host not in LOCAL_HOSTS)
        ):
            self._sync_client = RemoteSyncClient(self)
            self._sync_client.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._sync_client:
            self._sync_client.stop()

    # -- Connection helper --

    def _conn(self) -> sqlite3.Connection:
        # WAL is configured during database initialization and persists with the
        # database. Reissuing journal_mode=WAL for every short-lived mesh
        # connection is itself a schema-level write that can contend with the
        # daemon. Mesh writes below use bounded whole-transaction retries.
        conn = sqlite3.connect(
            self._db_path,
            timeout=_WRITE_BUSY_TIMEOUT_MS / 1000,
        )
        conn.execute(f"PRAGMA busy_timeout={_WRITE_BUSY_TIMEOUT_MS}")
        conn.row_factory = sqlite3.Row
        return conn

    @staticmethod
    def _is_transient_lock(exc: sqlite3.OperationalError) -> bool:
        message = str(exc).lower()
        return "database is locked" in message or "database is busy" in message

    def _write_with_retry(
        self,
        operation: Callable[[sqlite3.Connection], _T],
    ) -> _T:
        """Run one idempotent mesh mutation with a bounded SQLite retry budget.

        SQLite WAL lets reads continue while a write is in progress, but it
        still permits a single writer. Retrying the entire short transaction on
        a fresh connection avoids leaking a transient writer collision to an
        agent heartbeat or mesh command.
        """
        last_error: sqlite3.OperationalError | None = None
        for attempt in range(_WRITE_RETRY_ATTEMPTS):
            conn = self._conn()
            try:
                return operation(conn)
            except sqlite3.OperationalError as exc:
                if not self._is_transient_lock(exc):
                    raise
                last_error = exc
                try:
                    conn.rollback()
                except sqlite3.Error:
                    pass
            finally:
                conn.close()

            if attempt < _WRITE_RETRY_ATTEMPTS - 1:
                time.sleep(_WRITE_RETRY_BASE_SECONDS * (2 ** attempt))

        assert last_error is not None
        raise last_error

    # -- Peers --

    def register_peer(self, session_id: str, summary: str = "",
                      host: str = "", port: int = 0,
                      project_path: str = "", agent_type: str = "unknown",
                      profile_id: str = "default") -> dict:
        def _register(conn: sqlite3.Connection) -> dict:
            now = datetime.now(timezone.utc).isoformat()
            effective_host = host or self._host
            # Idempotent within the tenant: update if same session_id exists
            # under this profile. A session_id in another tenant is a different
            # peer and must not be adopted here.
            existing = conn.execute(
                "SELECT peer_id FROM mesh_peers WHERE session_id = ? AND profile_id = ?",
                (session_id, profile_id),
            ).fetchone()
            if existing:
                peer_id = existing["peer_id"]
                conn.execute(
                    "UPDATE mesh_peers SET summary=?, host=?, port=?, last_heartbeat=?, "
                    "status='active', project_path=?, agent_type=? "
                    "WHERE peer_id=? AND profile_id=?",
                    (summary, effective_host, port, now, project_path, agent_type,
                     peer_id, profile_id),
                )
            else:
                peer_id = str(uuid.uuid4())[:12]
                conn.execute(
                    "INSERT INTO mesh_peers (peer_id, session_id, summary, status, host, port, "
                    "registered_at, last_heartbeat, project_path, agent_type, profile_id) "
                    "VALUES (?, ?, ?, 'active', ?, ?, ?, ?, ?, ?, ?)",
                    (peer_id, session_id, summary, effective_host, port, now, now,
                     project_path, agent_type, profile_id),
                )
            self._log_event(conn, "peer_registered", peer_id, {
                "session_id": session_id, "project_path": project_path,
            }, profile_id=profile_id)
            conn.commit()

            # v3.4.6: Deliver pending broadcast/project messages on registration
            pending = self._get_pending_for_peer(conn, peer_id, project_path, profile_id)
            return {"peer_id": peer_id, "ok": True, "pending_messages": len(pending)}

        return self._write_with_retry(_register)

    def deregister_peer(self, peer_id: str, profile_id: str = "default") -> dict:
        def _deregister(conn: sqlite3.Connection) -> dict:
            row = conn.execute(
                "SELECT 1 FROM mesh_peers WHERE peer_id=? AND profile_id=?",
                (peer_id, profile_id),
            ).fetchone()
            if not row:
                return {"ok": False, "error": "peer not found"}
            conn.execute(
                "DELETE FROM mesh_peers WHERE peer_id=? AND profile_id=?",
                (peer_id, profile_id),
            )
            self._log_event(conn, "peer_deregistered", peer_id, profile_id=profile_id)
            conn.commit()
            return {"ok": True}

        return self._write_with_retry(_deregister)

    def heartbeat(self, peer_id: str, profile_id: str = "default") -> dict:
        def _heartbeat(conn: sqlite3.Connection) -> dict:
            now = datetime.now(timezone.utc).isoformat()
            cursor = conn.execute(
                "UPDATE mesh_peers SET last_heartbeat=?, status='active' "
                "WHERE peer_id=? AND profile_id=?",
                (now, peer_id, profile_id),
            )
            if cursor.rowcount == 0:
                return {"ok": False, "error": "peer not found"}
            conn.commit()
            return {"ok": True}

        return self._write_with_retry(_heartbeat)

    def update_summary(self, peer_id: str, summary: str,
                       profile_id: str = "default") -> dict:
        def _update_summary(conn: sqlite3.Connection) -> dict:
            cursor = conn.execute(
                "UPDATE mesh_peers SET summary=? WHERE peer_id=? AND profile_id=?",
                (summary, peer_id, profile_id),
            )
            if cursor.rowcount == 0:
                return {"ok": False, "error": "peer not found"}
            conn.commit()
            return {"ok": True}

        return self._write_with_retry(_update_summary)

    def list_peers(self, profile_id: str = "default") -> list[dict]:
        conn = self._conn()
        try:
            rows = conn.execute(
                "SELECT peer_id, session_id, summary, status, host, port, "
                "registered_at, last_heartbeat, project_path, agent_type, profile_id "
                "FROM mesh_peers WHERE profile_id=? ORDER BY last_heartbeat DESC",
                (profile_id,),
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    # -- Messages --

    def send_message(self, from_peer: str, to_peer: str, content: str,
                     msg_type: str = "text", project_path: str = "",
                     profile_id: str = "default") -> dict:
        # Guard: 4KB message size cap
        if len(content) > MAX_MESSAGE_SIZE:
            return {"ok": False, "error": f"message too large ({len(content)} bytes, max {MAX_MESSAGE_SIZE}). "
                    "Mesh messages are notifications — reference a file path instead."}

        # Remote delivery is an external side effect, so do it outside the
        # retry envelope. Local writes below are retried as a whole short
        # transaction when another daemon-owned operation has SQLite's writer.
        # Hold the lock only for the membership check, not for the HTTP call.
        # A remote peer only counts if it belongs to this tenant (profile).
        with self._remote_peers_lock:
            remote_info = self._remote_peers.get(to_peer)
            is_remote = (
                remote_info is not None
                and remote_info.get("profile_id", "default") == profile_id
            )
        if is_remote and self._sync_client:
            return self._sync_client.send_to_remote(to_peer, {
                "from_peer": from_peer,
                "to": to_peer,
                "content": content,
                "type": msg_type,
                "profile_id": profile_id,
            })

        def _send(conn: sqlite3.Connection) -> dict:
            # Derive locals fresh on EVERY call. A nonlocal mutation of to_peer
            # persisted across _write_with_retry attempts: a 'project:' address
            # rewritten to 'project' on the first try then misrouted to the
            # direct-peer branch on retry ("recipient peer not found").
            _to_peer = to_peer
            _project_path = project_path
            now = datetime.now(timezone.utc).isoformat()
            expires_at = self._compute_expires(now)

            # Determine target type
            if _to_peer == "broadcast":
                target_type = "broadcast"
            elif _to_peer.startswith("project:"):
                target_type = "project"
                _project_path = _to_peer[len("project:"):]
                _to_peer = "project"
            else:
                target_type = "peer"
                # Verify recipient exists WITHIN this tenant for direct messages.
                # A peer_id in another profile is not a valid recipient here.
                if not conn.execute(
                    "SELECT 1 FROM mesh_peers WHERE peer_id=? AND profile_id=?",
                    (_to_peer, profile_id),
                ).fetchone():
                    return {"ok": False, "error": "recipient peer not found"}

            # Enforce per-target queue cap (per tenant)
            if target_type in ("broadcast", "project"):
                count = conn.execute(
                    "SELECT COUNT(*) FROM mesh_messages "
                    "WHERE profile_id=? AND target_type=? AND project_path=? AND read=0",
                    (profile_id, target_type, _project_path),
                ).fetchone()[0]
                if count >= MAX_QUEUED_PER_TARGET:
                    # Delete oldest to make room
                    conn.execute(
                        "DELETE FROM mesh_messages WHERE id IN ("
                        "  SELECT id FROM mesh_messages "
                        "  WHERE profile_id=? AND target_type=? AND project_path=? AND read=0 "
                        "  ORDER BY created_at ASC LIMIT ?)",
                        (profile_id, target_type, _project_path,
                         count - MAX_QUEUED_PER_TARGET + 1),
                    )

            cursor = conn.execute(
                "INSERT INTO mesh_messages (from_peer, to_peer, msg_type, content, read, "
                "created_at, expires_at, target_type, project_path, profile_id) "
                "VALUES (?, ?, ?, ?, 0, ?, ?, ?, ?, ?)",
                (from_peer, _to_peer, msg_type, content, now, expires_at,
                 target_type, _project_path, profile_id),
            )
            self._log_event(conn, "message_sent", from_peer, {
                "to": _to_peer, "target_type": target_type, "project": _project_path,
            }, profile_id=profile_id)
            conn.commit()
            return {"ok": True, "id": cursor.lastrowid, "target_type": target_type,
                    "expires_at": expires_at}

        return self._write_with_retry(_send)

    def get_inbox(self, peer_id: str, project_path: str = "",
                  profile_id: str = "default") -> list[dict]:
        """Get all messages for this peer: direct + broadcast + project.

        Scoped to the peer's tenant (profile_id): a peer never sees another
        tenant's direct, broadcast, or project traffic.
        """
        conn = self._conn()
        try:
            now = datetime.now(timezone.utc).isoformat()
            # Direct messages to this peer
            # v3.6.12 (mesh-3): only UNREAD direct messages — was returning read
            # ones too, so every poll re-listed already-read messages until the
            # 24h cleanup (broadcast/project already filter unread via mesh_reads).
            direct = conn.execute(
                "SELECT id, from_peer, to_peer, msg_type, content, read, created_at, "
                "target_type, project_path FROM mesh_messages "
                "WHERE profile_id=? AND to_peer=? AND target_type='peer' "
                "AND COALESCE(read, 0) = 0 "
                "AND (expires_at IS NULL OR expires_at > ?) "
                "ORDER BY created_at DESC LIMIT 100",
                (profile_id, peer_id, now),
            ).fetchall()

            # Broadcast messages not from this peer and not yet read by this peer
            broadcast = conn.execute(
                "SELECT m.id, m.from_peer, m.to_peer, m.msg_type, m.content, "
                "CASE WHEN r.peer_id IS NOT NULL THEN 1 ELSE 0 END AS read, "
                "m.created_at, m.target_type, m.project_path "
                "FROM mesh_messages m "
                "LEFT JOIN mesh_reads r ON m.id = r.message_id AND r.peer_id = ? "
                "WHERE m.profile_id=? AND m.target_type='broadcast' AND m.from_peer != ? "
                "AND r.peer_id IS NULL "
                "AND (m.expires_at IS NULL OR m.expires_at > ?) "
                "ORDER BY m.created_at DESC LIMIT 50",
                (peer_id, profile_id, peer_id, now),
            ).fetchall()

            # Project messages for my project, not from me, not yet read
            project_msgs = []
            if project_path:
                project_msgs = conn.execute(
                    "SELECT m.id, m.from_peer, m.to_peer, m.msg_type, m.content, "
                    "CASE WHEN r.peer_id IS NOT NULL THEN 1 ELSE 0 END AS read, "
                    "m.created_at, m.target_type, m.project_path "
                    "FROM mesh_messages m "
                    "LEFT JOIN mesh_reads r ON m.id = r.message_id AND r.peer_id = ? "
                    "WHERE m.profile_id=? AND m.target_type='project' "
                    "AND m.project_path=? AND m.from_peer != ? "
                    "AND r.peer_id IS NULL "
                    "AND (m.expires_at IS NULL OR m.expires_at > ?) "
                    "ORDER BY m.created_at DESC LIMIT 50",
                    (peer_id, profile_id, project_path, peer_id, now),
                ).fetchall()

            all_msgs = [dict(r) for r in direct] + [dict(r) for r in broadcast] + [dict(r) for r in project_msgs]
            # Sort by created_at descending
            all_msgs.sort(key=lambda m: m.get("created_at", ""), reverse=True)
            return all_msgs[:100]
        finally:
            conn.close()

    def mark_read(self, peer_id: str, message_ids: list[int],
                  profile_id: str = "default") -> dict:
        def _mark_read(conn: sqlite3.Connection) -> dict:
            if not message_ids:
                return {"ok": True, "marked": 0}
            now = datetime.now(timezone.utc).isoformat()
            ph = ",".join("?" * len(message_ids))
            # One batched read of target types (tenant-scoped), then batched
            # writes — was 2N round-trips per N messages.
            rows = conn.execute(
                f"SELECT id, target_type FROM mesh_messages "
                f"WHERE id IN ({ph}) AND profile_id=?",
                (*message_ids, profile_id),
            ).fetchall()
            direct_ids = [r["id"] for r in rows if r["target_type"] == "peer"]
            shared_ids = [r["id"] for r in rows if r["target_type"] != "peer"]
            if direct_ids:
                dph = ",".join("?" * len(direct_ids))
                conn.execute(
                    f"UPDATE mesh_messages SET read=1 "
                    f"WHERE id IN ({dph}) AND to_peer=? AND profile_id=?",
                    (*direct_ids, peer_id, profile_id),
                )
            if shared_ids:
                conn.executemany(
                    "INSERT OR IGNORE INTO mesh_reads (message_id, peer_id, read_at) "
                    "VALUES (?, ?, ?)",
                    [(mid, peer_id, now) for mid in shared_ids],
                )
            conn.commit()
            return {"ok": True, "marked": len(message_ids)}

        return self._write_with_retry(_mark_read)

    # -- State --

    def get_state(self, profile_id: str = "default") -> dict:
        conn = self._conn()
        try:
            rows = conn.execute(
                "SELECT key, value, set_by, updated_at FROM mesh_state WHERE profile_id=?",
                (profile_id,),
            ).fetchall()
            return {r["key"]: {"value": r["value"], "set_by": r["set_by"], "updated_at": r["updated_at"]} for r in rows}
        finally:
            conn.close()

    def set_state(self, key: str, value: str, set_by: str,
                  profile_id: str = "default") -> dict:
        def _set_state(conn: sqlite3.Connection) -> dict:
            now = datetime.now(timezone.utc).isoformat()
            conn.execute(
                "INSERT INTO mesh_state (profile_id, key, value, set_by, updated_at) "
                "VALUES (?, ?, ?, ?, ?) "
                "ON CONFLICT(profile_id, key) DO UPDATE SET value=excluded.value, "
                "set_by=excluded.set_by, updated_at=excluded.updated_at",
                (profile_id, key, value, set_by, now),
            )
            conn.commit()
            return {"ok": True}

        return self._write_with_retry(_set_state)

    def get_state_key(self, key: str, profile_id: str = "default") -> dict | None:
        conn = self._conn()
        try:
            row = conn.execute(
                "SELECT key, value, set_by, updated_at FROM mesh_state "
                "WHERE profile_id=? AND key=?",
                (profile_id, key),
            ).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    # -- Locks --

    def lock_action(self, file_path: str, locked_by: str, action: str,
                    profile_id: str = "default") -> dict:
        if action == "query":
            conn = self._conn()
            try:
                row = conn.execute(
                    "SELECT locked_by, locked_at FROM mesh_locks "
                    "WHERE profile_id=? AND file_path=?",
                    (profile_id, file_path),
                ).fetchone()
                if row:
                    return {"locked": True, "by": row["locked_by"], "since": row["locked_at"]}
                return {"locked": False}
            finally:
                conn.close()

        if action not in {"acquire", "release"}:
            return {"ok": False, "error": f"unknown action: {action}"}

        def _lock_action(conn: sqlite3.Connection) -> dict:
            now = datetime.now(timezone.utc).isoformat()

            if action == "acquire":
                existing = conn.execute(
                    "SELECT locked_by, locked_at, expires_at FROM mesh_locks "
                    "WHERE profile_id=? AND file_path=?",
                    (profile_id, file_path),
                ).fetchone()
                # M-02: another peer's lock only blocks while it is still live.
                # An expired lock — or a legacy never-expires sentinel left by a
                # crashed session — is treated as free, so a stale lock can never
                # deadlock a path forever.
                if existing and existing["locked_by"] != locked_by:
                    exp = existing["expires_at"]
                    still_live = bool(exp) and exp != _NEVER_EXPIRES and exp > now
                    if still_live:
                        return {"locked": True, "by": existing["locked_by"],
                                "since": existing["locked_at"]}
                lock_expires = (
                    datetime.fromisoformat(now) + timedelta(hours=LOCK_TTL_HOURS)
                ).isoformat()
                conn.execute(
                    "INSERT INTO mesh_locks (profile_id, file_path, locked_by, locked_at, expires_at) "
                    "VALUES (?, ?, ?, ?, ?) "
                    "ON CONFLICT(profile_id, file_path) DO UPDATE SET locked_by=excluded.locked_by, "
                    "locked_at=excluded.locked_at, expires_at=excluded.expires_at",
                    (profile_id, file_path, locked_by, now, lock_expires),
                )
                conn.commit()
                return {"ok": True, "action": "acquired", "expires_at": lock_expires}

            elif action == "release":
                # v3.6.12 (mesh-2): report whether we actually released. The
                # DELETE is correctly owner-scoped, but it previously returned
                # released=ok:true even when a NON-owner released nothing.
                cur = conn.execute(
                    "DELETE FROM mesh_locks WHERE profile_id=? AND file_path=? AND locked_by=?",
                    (profile_id, file_path, locked_by),
                )
                conn.commit()
                if cur.rowcount and cur.rowcount > 0:
                    return {"ok": True, "action": "released"}
                return {"ok": False, "action": "not_released",
                        "error": "no lock held by this peer for that file"}

            raise AssertionError("validated action was not handled")

        return self._write_with_retry(_lock_action)

    # -- Helpers (v3.4.6) --

    @staticmethod
    def _compute_expires(now_iso: str) -> str:
        """Compute expiry timestamp MESSAGE_TTL_HOURS from now."""
        from datetime import timedelta
        now = datetime.fromisoformat(now_iso)
        return (now + timedelta(hours=MESSAGE_TTL_HOURS)).isoformat()

    def _get_pending_for_peer(self, conn: sqlite3.Connection,
                              peer_id: str, project_path: str,
                              profile_id: str = "default") -> list[dict]:
        """Get unread broadcast/project messages for a newly registered peer
        within this tenant."""
        now = datetime.now(timezone.utc).isoformat()
        rows = conn.execute(
            "SELECT m.id, m.from_peer, m.content, m.target_type, m.project_path, m.created_at "
            "FROM mesh_messages m "
            "LEFT JOIN mesh_reads r ON m.id = r.message_id AND r.peer_id = ? "
            "WHERE m.profile_id = ? AND r.peer_id IS NULL AND m.from_peer != ? "
            "AND (m.expires_at IS NULL OR m.expires_at > ?) "
            "AND (m.target_type = 'broadcast' "
            "     OR (m.target_type = 'project' AND m.project_path = ?)) "
            "ORDER BY m.created_at DESC LIMIT 50",
            (peer_id, profile_id, peer_id, now, project_path),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_pending(self, peer_id: str, project_path: str = "",
                    profile_id: str = "default") -> list[dict]:
        """Public API to get pending broadcast/project messages."""
        conn = self._conn()
        try:
            return self._get_pending_for_peer(conn, peer_id, project_path, profile_id)
        finally:
            conn.close()

    # -- Events --

    def get_events(self, limit: int = 100, profile_id: str = "default") -> list[dict]:
        conn = self._conn()
        try:
            rows = conn.execute(
                "SELECT id, event_type, payload, emitted_by, created_at "
                "FROM mesh_events WHERE profile_id=? ORDER BY id DESC LIMIT ?",
                (profile_id, limit),
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def _log_event(self, conn: sqlite3.Connection, event_type: str,
                   emitted_by: str, payload: dict | None = None,
                   profile_id: str = "default") -> None:
        import json as _json
        now = datetime.now(timezone.utc).isoformat()
        conn.execute(
            "INSERT INTO mesh_events (event_type, payload, emitted_by, created_at, profile_id) "
            "VALUES (?, ?, ?, ?, ?)",
            (event_type, _json.dumps(payload or {}), emitted_by, now, profile_id),
        )

    # -- Status --

    def get_status(self, profile_id: str = "default") -> dict:
        conn = self._conn()
        try:
            peer_count = conn.execute(
                "SELECT COUNT(*) FROM mesh_peers WHERE profile_id=? AND status='active'",
                (profile_id,),
            ).fetchone()[0]
            return {
                "broker_up": True,
                "peer_count": peer_count,
                "uptime_s": round(time.monotonic() - self._started_at),
            }
        finally:
            conn.close()

    # -- Cleanup --

    def _cleanup_loop(self) -> None:
        """Background cleanup: mark stale peers, delete old messages."""
        while not self._stop_event.is_set():
            self._stop_event.wait(300)  # Every 5 min
            if self._stop_event.is_set():
                break
            try:
                self._run_cleanup()
            except Exception as exc:
                logger.debug("Mesh cleanup error: %s", exc)

    def _run_cleanup(self) -> None:
        def _cleanup(conn: sqlite3.Connection) -> None:
            # Precompute ISO cutoffs in Python and compare the stored ISO strings
            # directly. ISO-8601 UTC timestamps sort lexicographically, so a bare
            # `col < ?` is both correct AND sargable — the datetime() wrapper
            # previously forced a full table scan on every 5-minute cleanup.
            now = datetime.now(timezone.utc)
            now_iso = now.isoformat()
            five_min = (now - timedelta(minutes=5)).isoformat()
            thirty_min = (now - timedelta(minutes=30)).isoformat()
            day_ago = (now - timedelta(hours=24)).isoformat()
            week_ago = (now - timedelta(days=7)).isoformat()
            # Mark stale peers (no heartbeat for 5 min)
            conn.execute(
                "UPDATE mesh_peers SET status='stale' "
                "WHERE status='active' AND last_heartbeat < ?",
                (five_min,),
            )
            # Delete dead peers (stale > 30 min)
            conn.execute(
                "UPDATE mesh_peers SET status='dead' "
                "WHERE status='stale' AND last_heartbeat < ?",
                (thirty_min,),
            )
            conn.execute("DELETE FROM mesh_peers WHERE status='dead'")
            # Delete read direct messages > 24hr old
            conn.execute(
                "DELETE FROM mesh_messages WHERE target_type='peer' AND read=1 "
                "AND created_at < ?",
                (day_ago,),
            )
            # v3.4.6: Delete EXPIRED messages (48h TTL for broadcast/project)
            conn.execute(
                "DELETE FROM mesh_messages WHERE expires_at IS NOT NULL "
                "AND expires_at < ?",
                (now_iso,),
            )
            # v3.4.6: Clean up orphaned mesh_reads entries
            conn.execute(
                "DELETE FROM mesh_reads WHERE message_id NOT IN "
                "(SELECT id FROM mesh_messages)",
            )
            # Delete expired locks (the 9999-… sentinel sorts after any real now)
            conn.execute(
                "DELETE FROM mesh_locks WHERE expires_at < ?",
                (now_iso,),
            )
            # v3.4.6: Delete old events (keep last 7 days)
            conn.execute(
                "DELETE FROM mesh_events WHERE created_at < ?",
                (week_ago,),
            )
            conn.commit()

        self._write_with_retry(_cleanup)
