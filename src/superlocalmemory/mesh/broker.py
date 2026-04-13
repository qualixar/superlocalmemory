# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under the Elastic License 2.0 - see LICENSE file
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
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger("superlocalmemory.mesh")


MAX_MESSAGE_SIZE = 4096  # 4KB cap — mesh messages are notifications, not data dumps
MESSAGE_TTL_HOURS = 48   # Offline messages expire after 48h
MAX_QUEUED_PER_TARGET = 50  # Max unread messages per broadcast/project target


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

    def start_cleanup(self) -> None:
        """Start background cleanup thread for stale peers/messages."""
        self._cleanup_thread = threading.Thread(
            target=self._cleanup_loop, daemon=True, name="mesh-cleanup",
        )
        self._cleanup_thread.start()

    def stop(self) -> None:
        self._stop_event.set()

    # -- Connection helper --

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        conn.row_factory = sqlite3.Row
        return conn

    # -- Peers --

    def register_peer(self, session_id: str, summary: str = "",
                      host: str = "127.0.0.1", port: int = 0,
                      project_path: str = "", agent_type: str = "unknown") -> dict:
        conn = self._conn()
        try:
            now = datetime.now(timezone.utc).isoformat()
            # Idempotent: update if same session_id exists
            existing = conn.execute(
                "SELECT peer_id FROM mesh_peers WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            if existing:
                peer_id = existing["peer_id"]
                conn.execute(
                    "UPDATE mesh_peers SET summary=?, host=?, port=?, last_heartbeat=?, "
                    "status='active', project_path=?, agent_type=? WHERE peer_id=?",
                    (summary, host, port, now, project_path, agent_type, peer_id),
                )
            else:
                peer_id = str(uuid.uuid4())[:12]
                conn.execute(
                    "INSERT INTO mesh_peers (peer_id, session_id, summary, status, host, port, "
                    "registered_at, last_heartbeat, project_path, agent_type) "
                    "VALUES (?, ?, ?, 'active', ?, ?, ?, ?, ?, ?)",
                    (peer_id, session_id, summary, host, port, now, now, project_path, agent_type),
                )
            self._log_event(conn, "peer_registered", peer_id, {
                "session_id": session_id, "project_path": project_path,
            })
            conn.commit()

            # v3.4.6: Deliver pending broadcast/project messages on registration
            pending = self._get_pending_for_peer(conn, peer_id, project_path)
            return {"peer_id": peer_id, "ok": True, "pending_messages": len(pending)}
        finally:
            conn.close()

    def deregister_peer(self, peer_id: str) -> dict:
        conn = self._conn()
        try:
            row = conn.execute("SELECT 1 FROM mesh_peers WHERE peer_id=?", (peer_id,)).fetchone()
            if not row:
                return {"ok": False, "error": "peer not found"}
            conn.execute("DELETE FROM mesh_peers WHERE peer_id=?", (peer_id,))
            self._log_event(conn, "peer_deregistered", peer_id)
            conn.commit()
            return {"ok": True}
        finally:
            conn.close()

    def heartbeat(self, peer_id: str) -> dict:
        conn = self._conn()
        try:
            now = datetime.now(timezone.utc).isoformat()
            cursor = conn.execute(
                "UPDATE mesh_peers SET last_heartbeat=?, status='active' WHERE peer_id=?",
                (now, peer_id),
            )
            if cursor.rowcount == 0:
                return {"ok": False, "error": "peer not found"}
            conn.commit()
            return {"ok": True}
        finally:
            conn.close()

    def update_summary(self, peer_id: str, summary: str) -> dict:
        conn = self._conn()
        try:
            cursor = conn.execute(
                "UPDATE mesh_peers SET summary=? WHERE peer_id=?",
                (summary, peer_id),
            )
            if cursor.rowcount == 0:
                return {"ok": False, "error": "peer not found"}
            conn.commit()
            return {"ok": True}
        finally:
            conn.close()

    def list_peers(self) -> list[dict]:
        conn = self._conn()
        try:
            rows = conn.execute(
                "SELECT peer_id, session_id, summary, status, host, port, "
                "registered_at, last_heartbeat, project_path, agent_type "
                "FROM mesh_peers ORDER BY last_heartbeat DESC",
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    # -- Messages --

    def send_message(self, from_peer: str, to_peer: str, content: str,
                     msg_type: str = "text", project_path: str = "") -> dict:
        # Guard: 4KB message size cap
        if len(content) > MAX_MESSAGE_SIZE:
            return {"ok": False, "error": f"message too large ({len(content)} bytes, max {MAX_MESSAGE_SIZE}). "
                    "Mesh messages are notifications — reference a file path instead."}

        conn = self._conn()
        try:
            now = datetime.now(timezone.utc).isoformat()
            expires_at = self._compute_expires(now)

            # Determine target type
            if to_peer == "broadcast":
                target_type = "broadcast"
            elif to_peer.startswith("project:"):
                target_type = "project"
                project_path = to_peer[len("project:"):]
                to_peer = "project"
            else:
                target_type = "peer"
                # Verify recipient exists for direct messages
                if not conn.execute("SELECT 1 FROM mesh_peers WHERE peer_id=?", (to_peer,)).fetchone():
                    return {"ok": False, "error": "recipient peer not found"}

            # Enforce per-target queue cap
            if target_type in ("broadcast", "project"):
                count = conn.execute(
                    "SELECT COUNT(*) FROM mesh_messages WHERE target_type=? AND project_path=? AND read=0",
                    (target_type, project_path),
                ).fetchone()[0]
                if count >= MAX_QUEUED_PER_TARGET:
                    # Delete oldest to make room
                    conn.execute(
                        "DELETE FROM mesh_messages WHERE id IN ("
                        "  SELECT id FROM mesh_messages WHERE target_type=? AND project_path=? AND read=0 "
                        "  ORDER BY created_at ASC LIMIT ?)",
                        (target_type, project_path, count - MAX_QUEUED_PER_TARGET + 1),
                    )

            cursor = conn.execute(
                "INSERT INTO mesh_messages (from_peer, to_peer, msg_type, content, read, "
                "created_at, expires_at, target_type, project_path) "
                "VALUES (?, ?, ?, ?, 0, ?, ?, ?, ?)",
                (from_peer, to_peer, msg_type, content, now, expires_at, target_type, project_path),
            )
            self._log_event(conn, "message_sent", from_peer, {
                "to": to_peer, "target_type": target_type, "project": project_path,
            })
            conn.commit()
            return {"ok": True, "id": cursor.lastrowid, "target_type": target_type,
                    "expires_at": expires_at}
        finally:
            conn.close()

    def get_inbox(self, peer_id: str, project_path: str = "") -> list[dict]:
        """Get all messages for this peer: direct + broadcast + project."""
        conn = self._conn()
        try:
            now = datetime.now(timezone.utc).isoformat()
            # Direct messages to this peer
            direct = conn.execute(
                "SELECT id, from_peer, to_peer, msg_type, content, read, created_at, "
                "target_type, project_path FROM mesh_messages "
                "WHERE to_peer=? AND target_type='peer' "
                "AND (expires_at IS NULL OR expires_at > ?) "
                "ORDER BY created_at DESC LIMIT 100",
                (peer_id, now),
            ).fetchall()

            # Broadcast messages not from this peer and not yet read by this peer
            broadcast = conn.execute(
                "SELECT m.id, m.from_peer, m.to_peer, m.msg_type, m.content, "
                "CASE WHEN r.peer_id IS NOT NULL THEN 1 ELSE 0 END AS read, "
                "m.created_at, m.target_type, m.project_path "
                "FROM mesh_messages m "
                "LEFT JOIN mesh_reads r ON m.id = r.message_id AND r.peer_id = ? "
                "WHERE m.target_type='broadcast' AND m.from_peer != ? "
                "AND (m.expires_at IS NULL OR m.expires_at > ?) "
                "ORDER BY m.created_at DESC LIMIT 50",
                (peer_id, peer_id, now),
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
                    "WHERE m.target_type='project' AND m.project_path=? AND m.from_peer != ? "
                    "AND (m.expires_at IS NULL OR m.expires_at > ?) "
                    "ORDER BY m.created_at DESC LIMIT 50",
                    (peer_id, project_path, peer_id, now),
                ).fetchall()

            all_msgs = [dict(r) for r in direct] + [dict(r) for r in broadcast] + [dict(r) for r in project_msgs]
            # Sort by created_at descending
            all_msgs.sort(key=lambda m: m.get("created_at", ""), reverse=True)
            return all_msgs[:100]
        finally:
            conn.close()

    def mark_read(self, peer_id: str, message_ids: list[int]) -> dict:
        conn = self._conn()
        try:
            now = datetime.now(timezone.utc).isoformat()
            for msg_id in message_ids:
                # Check if this is a direct message or broadcast/project
                row = conn.execute(
                    "SELECT target_type FROM mesh_messages WHERE id=?", (msg_id,),
                ).fetchone()
                if not row:
                    continue
                if row["target_type"] == "peer":
                    # Direct: update read flag on the message itself
                    conn.execute(
                        "UPDATE mesh_messages SET read=1 WHERE id=? AND to_peer=?",
                        (msg_id, peer_id),
                    )
                else:
                    # Broadcast/project: insert into mesh_reads
                    conn.execute(
                        "INSERT OR IGNORE INTO mesh_reads (message_id, peer_id, read_at) "
                        "VALUES (?, ?, ?)",
                        (msg_id, peer_id, now),
                    )
            conn.commit()
            return {"ok": True, "marked": len(message_ids)}
        finally:
            conn.close()

    # -- State --

    def get_state(self) -> dict:
        conn = self._conn()
        try:
            rows = conn.execute("SELECT key, value, set_by, updated_at FROM mesh_state").fetchall()
            return {r["key"]: {"value": r["value"], "set_by": r["set_by"], "updated_at": r["updated_at"]} for r in rows}
        finally:
            conn.close()

    def set_state(self, key: str, value: str, set_by: str) -> dict:
        conn = self._conn()
        try:
            now = datetime.now(timezone.utc).isoformat()
            conn.execute(
                "INSERT INTO mesh_state (key, value, set_by, updated_at) VALUES (?, ?, ?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value, set_by=excluded.set_by, updated_at=excluded.updated_at",
                (key, value, set_by, now),
            )
            conn.commit()
            return {"ok": True}
        finally:
            conn.close()

    def get_state_key(self, key: str) -> dict | None:
        conn = self._conn()
        try:
            row = conn.execute(
                "SELECT key, value, set_by, updated_at FROM mesh_state WHERE key=?", (key,),
            ).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    # -- Locks --

    def lock_action(self, file_path: str, locked_by: str, action: str) -> dict:
        conn = self._conn()
        try:
            now = datetime.now(timezone.utc).isoformat()

            if action == "acquire":
                existing = conn.execute(
                    "SELECT locked_by, locked_at FROM mesh_locks WHERE file_path=?",
                    (file_path,),
                ).fetchone()
                if existing and existing["locked_by"] != locked_by:
                    return {"locked": True, "by": existing["locked_by"], "since": existing["locked_at"]}
                conn.execute(
                    "INSERT INTO mesh_locks (file_path, locked_by, locked_at) VALUES (?, ?, ?) "
                    "ON CONFLICT(file_path) DO UPDATE SET locked_by=excluded.locked_by, locked_at=excluded.locked_at",
                    (file_path, locked_by, now),
                )
                conn.commit()
                return {"ok": True, "action": "acquired"}

            elif action == "release":
                conn.execute("DELETE FROM mesh_locks WHERE file_path=? AND locked_by=?",
                             (file_path, locked_by))
                conn.commit()
                return {"ok": True, "action": "released"}

            elif action == "query":
                row = conn.execute(
                    "SELECT locked_by, locked_at FROM mesh_locks WHERE file_path=?",
                    (file_path,),
                ).fetchone()
                if row:
                    return {"locked": True, "by": row["locked_by"], "since": row["locked_at"]}
                return {"locked": False}

            return {"ok": False, "error": f"unknown action: {action}"}
        finally:
            conn.close()

    # -- Helpers (v3.4.6) --

    @staticmethod
    def _compute_expires(now_iso: str) -> str:
        """Compute expiry timestamp MESSAGE_TTL_HOURS from now."""
        from datetime import timedelta
        now = datetime.fromisoformat(now_iso)
        return (now + timedelta(hours=MESSAGE_TTL_HOURS)).isoformat()

    def _get_pending_for_peer(self, conn: sqlite3.Connection,
                              peer_id: str, project_path: str) -> list[dict]:
        """Get unread broadcast/project messages for a newly registered peer."""
        now = datetime.now(timezone.utc).isoformat()
        rows = conn.execute(
            "SELECT m.id, m.from_peer, m.content, m.target_type, m.project_path, m.created_at "
            "FROM mesh_messages m "
            "LEFT JOIN mesh_reads r ON m.id = r.message_id AND r.peer_id = ? "
            "WHERE r.peer_id IS NULL AND m.from_peer != ? "
            "AND (m.expires_at IS NULL OR m.expires_at > ?) "
            "AND (m.target_type = 'broadcast' "
            "     OR (m.target_type = 'project' AND m.project_path = ?)) "
            "ORDER BY m.created_at DESC LIMIT 50",
            (peer_id, peer_id, now, project_path),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_pending(self, peer_id: str, project_path: str = "") -> list[dict]:
        """Public API to get pending broadcast/project messages."""
        conn = self._conn()
        try:
            return self._get_pending_for_peer(conn, peer_id, project_path)
        finally:
            conn.close()

    # -- Events --

    def get_events(self, limit: int = 100) -> list[dict]:
        conn = self._conn()
        try:
            rows = conn.execute(
                "SELECT id, event_type, payload, emitted_by, created_at "
                "FROM mesh_events ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def _log_event(self, conn: sqlite3.Connection, event_type: str,
                   emitted_by: str, payload: dict | None = None) -> None:
        import json as _json
        now = datetime.now(timezone.utc).isoformat()
        conn.execute(
            "INSERT INTO mesh_events (event_type, payload, emitted_by, created_at) VALUES (?, ?, ?, ?)",
            (event_type, _json.dumps(payload or {}), emitted_by, now),
        )

    # -- Status --

    def get_status(self) -> dict:
        conn = self._conn()
        try:
            peer_count = conn.execute("SELECT COUNT(*) FROM mesh_peers WHERE status='active'").fetchone()[0]
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
        conn = self._conn()
        try:
            now = datetime.now(timezone.utc)
            now_iso = now.isoformat()
            # Mark stale peers (no heartbeat for 5 min)
            conn.execute(
                "UPDATE mesh_peers SET status='stale' "
                "WHERE status='active' AND datetime(last_heartbeat) < datetime(?, '-5 minutes')",
                (now_iso,),
            )
            # Delete dead peers (stale > 30 min)
            conn.execute(
                "UPDATE mesh_peers SET status='dead' "
                "WHERE status='stale' AND datetime(last_heartbeat) < datetime(?, '-30 minutes')",
                (now_iso,),
            )
            conn.execute("DELETE FROM mesh_peers WHERE status='dead'")
            # Delete read direct messages > 24hr old
            conn.execute(
                "DELETE FROM mesh_messages WHERE target_type='peer' AND read=1 "
                "AND datetime(created_at) < datetime(?, '-24 hours')",
                (now_iso,),
            )
            # v3.4.6: Delete EXPIRED messages (48h TTL for broadcast/project)
            conn.execute(
                "DELETE FROM mesh_messages WHERE expires_at IS NOT NULL "
                "AND datetime(expires_at) < datetime(?)",
                (now_iso,),
            )
            # v3.4.6: Clean up orphaned mesh_reads entries
            conn.execute(
                "DELETE FROM mesh_reads WHERE message_id NOT IN "
                "(SELECT id FROM mesh_messages)",
            )
            # Delete expired locks
            conn.execute(
                "DELETE FROM mesh_locks WHERE datetime(expires_at) < datetime(?)",
                (now_iso,),
            )
            # v3.4.6: Delete old events (keep last 7 days)
            conn.execute(
                "DELETE FROM mesh_events WHERE datetime(created_at) < datetime(?, '-7 days')",
                (now_iso,),
            )
            conn.commit()
        finally:
            conn.close()
