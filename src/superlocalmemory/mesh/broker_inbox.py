# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Mesh inbox reads: the inbox query, the bounded wait, and envelope lookups.

Every function takes an open connection (or a fetch callable) so the broker
keeps ownership of connections, retries and tenancy.
"""

from __future__ import annotations

import math
import sqlite3
import threading
import time
from collections.abc import Callable
from datetime import datetime, timezone

from .envelope import envelope_for

WAIT_MIN_S = 1
WAIT_MAX_S = 20
MAX_CONCURRENT_WAITS = 8
POLL_FALLBACK_S = 0.5  # another process may insert without notifying us


def query_inbox(conn: sqlite3.Connection, peer_id: str, project_path: str,
                profile_id: str, *, direct_only: bool = False) -> list[dict]:
    """Unread direct + broadcast + project messages for one peer in one tenant.

    With ``direct_only`` only the messages addressed to the peer itself.
    """
    now = datetime.now(timezone.utc).isoformat()
    # v3.6.12 (mesh-3): only UNREAD direct messages; broadcast/project already
    # filter unread via mesh_reads.
    direct = conn.execute(
        "SELECT id, from_peer, to_peer, msg_type, content, read, created_at, "
        "target_type, project_path FROM mesh_messages "
        "WHERE profile_id=? AND to_peer=? AND target_type='peer' "
        "AND COALESCE(read, 0) = 0 "
        "AND (expires_at IS NULL OR expires_at > ?) "
        "ORDER BY created_at DESC LIMIT 100",
        (profile_id, peer_id, now),
    ).fetchall()
    if direct_only:
        return [dict(r) for r in direct]
    shared_select = (
        "SELECT m.id, m.from_peer, m.to_peer, m.msg_type, m.content, "
        "CASE WHEN r.peer_id IS NOT NULL THEN 1 ELSE 0 END AS read, "
        "m.created_at, m.target_type, m.project_path "
        "FROM mesh_messages m "
        "LEFT JOIN mesh_reads r ON m.id = r.message_id AND r.peer_id = ? "
    )
    broadcast = conn.execute(
        shared_select
        + "WHERE m.profile_id=? AND m.target_type='broadcast' AND m.from_peer != ? "
        "AND r.peer_id IS NULL "
        "AND (m.expires_at IS NULL OR m.expires_at > ?) "
        "ORDER BY m.created_at DESC LIMIT 50",
        (peer_id, profile_id, peer_id, now),
    ).fetchall()
    project_msgs = []
    if project_path:
        project_msgs = conn.execute(
            shared_select
            + "WHERE m.profile_id=? AND m.target_type='project' "
            "AND m.project_path=? AND m.from_peer != ? "
            "AND r.peer_id IS NULL "
            "AND (m.expires_at IS NULL OR m.expires_at > ?) "
            "ORDER BY m.created_at DESC LIMIT 50",
            (peer_id, profile_id, project_path, peer_id, now),
        ).fetchall()
    all_msgs = [dict(r) for r in (*direct, *broadcast, *project_msgs)]
    all_msgs.sort(key=lambda m: m.get("created_at", ""), reverse=True)
    return all_msgs[:100]


def has_unread_direct(conn: sqlite3.Connection, peer_id: str, profile_id: str) -> bool:
    """Whether any unexpired direct message waits for the peer (a plain read)."""
    return conn.execute(
        "SELECT 1 FROM mesh_messages WHERE profile_id=? AND to_peer=? "
        "AND target_type='peer' AND COALESCE(read, 0)=0 "
        "AND (expires_at IS NULL OR expires_at > ?) LIMIT 1",
        (profile_id, peer_id, datetime.now(timezone.utc).isoformat()),
    ).fetchone() is not None


MAX_QUEUED_PER_TARGET = 50  # Max unread messages per broadcast/project target
MAX_UNREAD_DIRECT = 50      # Max unread direct messages a web app may queue for one peer


def _web_target_refusal(conn: sqlite3.Connection, to_peer: str,
                        profile_id: str) -> dict | None:
    """Why a web app may not send here, or None.

    A web app addresses one peer by id (never everyone, never a project), and
    it never evicts: a full inbox refuses the new message instead.
    """
    if to_peer == "broadcast" or to_peer.startswith("project:"):
        return {"ok": False, "error": "web apps can only message one peer by id"}
    if not conn.in_transaction:
        # Take the write lock before counting, so two senders cannot both
        # pass the last free slot.
        conn.execute("BEGIN IMMEDIATE")
    unread = conn.execute(
        "SELECT COUNT(*) FROM mesh_messages WHERE profile_id=? AND to_peer=? "
        "AND target_type='peer' AND COALESCE(read, 0)=0 "
        "AND (expires_at IS NULL OR expires_at > ?)",
        (profile_id, to_peer, datetime.now(timezone.utc).isoformat()),
    ).fetchone()[0]
    if unread >= MAX_UNREAD_DIRECT:
        return {"ok": False, "error": "recipient inbox is full"}
    return None


def resolve_target(conn: sqlite3.Connection, to_peer: str, project_path: str,
                   profile_id: str, *, web_sender: bool = False) -> dict:
    """Work out where a send goes and make room in a shared queue.

    Returns ``{"target_type", "to_peer", "project_path"}`` or an error result.
    Derived fresh on every call: a retry must see the caller's original address.
    """
    if web_sender:
        refused = _web_target_refusal(conn, to_peer, profile_id)
        if refused is not None:
            return refused
    if to_peer == "broadcast":
        target_type = "broadcast"
    elif to_peer.startswith("project:"):
        target_type = "project"
        project_path = to_peer[len("project:"):]
        to_peer = "project"
    else:
        # A direct recipient must exist WITHIN this tenant.
        if not conn.execute(
            "SELECT 1 FROM mesh_peers WHERE peer_id=? AND profile_id=?",
            (to_peer, profile_id),
        ).fetchone():
            return {"ok": False, "error": "recipient peer not found"}
        return {"target_type": "peer", "to_peer": to_peer, "project_path": project_path}
    count = conn.execute(
        "SELECT COUNT(*) FROM mesh_messages "
        "WHERE profile_id=? AND target_type=? AND project_path=? AND read=0",
        (profile_id, target_type, project_path),
    ).fetchone()[0]
    if count >= MAX_QUEUED_PER_TARGET:
        conn.execute(
            "DELETE FROM mesh_messages WHERE id IN ("
            "  SELECT id FROM mesh_messages "
            "  WHERE profile_id=? AND target_type=? AND project_path=? AND read=0 "
            "  ORDER BY created_at ASC LIMIT ?)",
            (profile_id, target_type, project_path, count - MAX_QUEUED_PER_TARGET + 1),
        )
    return {"target_type": target_type, "to_peer": to_peer, "project_path": project_path}


def attach_envelopes(conn: sqlite3.Connection, msgs: list[dict],
                     *, remote_view: bool) -> list[dict]:
    """Add an ``envelope`` to each message; in a remote view also datamark ``content``."""
    if not msgs:
        return msgs
    ids = [m["id"] for m in msgs]
    marks = ",".join("?" * len(ids))
    rows = conn.execute(
        "SELECT m.id AS message_id, m.expires_at, e.from_kind, e.from_app, "
        "e.hop, e.refs_json, e.reply_to FROM mesh_messages m "
        "LEFT JOIN mesh_message_envelopes e ON e.message_id = m.id "
        f"WHERE m.id IN ({marks})",
        ids,
    ).fetchall()
    by_id = {r["message_id"]: dict(r) for r in rows}
    for msg in msgs:
        env = envelope_for(msg, by_id.get(msg["id"]), remote_view=remote_view)
        msg["envelope"] = env
        if env["trust"] == "untrusted-peer":
            msg["content"] = env["content"]
    return msgs


def query_messages(conn: sqlite3.Connection, profile_id: str, limit: int,
                   peer: str = "") -> list[dict]:
    """Envelopes of stored messages, newest first (the owner's own data)."""
    sql = (
        "SELECT m.id, m.from_peer, m.to_peer, m.content, m.created_at, m.expires_at, "
        "e.from_kind, e.from_app, e.hop, e.refs_json, e.reply_to "
        "FROM mesh_messages m LEFT JOIN mesh_message_envelopes e ON e.message_id = m.id "
        "WHERE m.profile_id=? "
    )
    args: list = [profile_id]
    if peer:
        sql += "AND (m.from_peer=? OR m.to_peer=?) "
        args += [peer, peer]
    rows = conn.execute(sql + "ORDER BY m.id DESC LIMIT ?", (*args, limit)).fetchall()
    return [envelope_for(dict(r), dict(r), remote_view=False) for r in rows]


def mark_messages_read(conn: sqlite3.Connection, peer_id: str,
                       message_ids: list[int], profile_id: str) -> dict:
    """Mark direct messages read, and record broadcast/project reads per peer."""
    if not message_ids:
        return {"ok": True, "marked": 0}
    now = datetime.now(timezone.utc).isoformat()
    ph = ",".join("?" * len(message_ids))
    # One batched read of target types (tenant-scoped), then batched writes.
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


class InboxWaiter:
    """Wakes bounded waits when a message is committed, with a polling fallback."""

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._generation = 0
        self._active = 0

    def notify(self) -> None:
        with self._cond:
            self._generation += 1
            self._cond.notify_all()

    def _acquire_slot(self) -> None:
        with self._cond:
            if self._active >= MAX_CONCURRENT_WAITS:
                raise RuntimeError("too many waits")
            self._active += 1

    def _release_slot(self) -> None:
        with self._cond:
            self._active -= 1

    def wait(self, fetch: Callable[[], list[dict]],
             timeout_s: float) -> tuple[list[dict], bool]:
        """Return ``(messages, timed_out)``; ``timeout_s`` is clamped to the bounds."""
        try:
            wanted = float(timeout_s)
        except (TypeError, ValueError):
            wanted = WAIT_MIN_S
        if not math.isfinite(wanted):
            wanted = WAIT_MIN_S
        timeout = min(max(wanted, WAIT_MIN_S), WAIT_MAX_S)
        self._acquire_slot()
        try:
            deadline = time.monotonic() + timeout
            while True:
                seen = self._generation
                msgs = fetch()
                if msgs:
                    return msgs, False
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return [], True
                with self._cond:
                    if self._generation == seen:
                        self._cond.wait(min(POLL_FALLBACK_S, remaining))
        finally:
            self._release_slot()
