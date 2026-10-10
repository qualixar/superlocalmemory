# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Peer profiles and message envelope rows for the mesh.

Connection-level helpers: the broker opens the connection, runs these inside
its own transaction and commits. Nothing here commits.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import unicodedata
from collections import deque
from collections.abc import Sequence
from datetime import datetime, timezone

from .envelope import check_hop, next_hop, validate_refs

SEND_LIMIT = 20
SEND_WINDOW_S = 60
MAX_DISPLAY_NAME = 64


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class SendRateLimiter:
    """Sliding-window send limit per (profile, sender), in memory."""

    def __init__(self, limit: int = SEND_LIMIT, window_s: float = SEND_WINDOW_S) -> None:
        self._limit = limit
        self._window = window_s
        self._lock = threading.Lock()
        self._sent: dict[tuple[str, str], deque[float]] = {}

    def _trim(self, key: tuple[str, str], now: float) -> deque[float] | None:
        q = self._sent.get(key)
        if q is None:
            return None
        while q and now - q[0] >= self._window:
            q.popleft()
        if not q:
            del self._sent[key]
            return None
        return q

    def try_acquire(self, profile_id: str, sender: str) -> int | None:
        """Take one slot, or return the seconds until the sender may send again.

        The check and the record happen under one lock, so concurrent sends
        cannot both pass the last slot.
        """
        key, now = (profile_id, sender), time.monotonic()
        with self._lock:
            q = self._trim(key, now)
            if q is not None and len(q) >= self._limit:
                return max(1, int(self._window - (now - q[0])) + 1)
            self._sent.setdefault(key, deque()).append(now)
            return None

    def refund(self, profile_id: str, sender: str) -> None:
        """Give back the newest slot (the send did not happen)."""
        key = (profile_id, sender)
        with self._lock:
            q = self._sent.get(key)
            if q:
                q.pop()
                if not q:
                    del self._sent[key]


def upsert_profile(conn: sqlite3.Connection, peer_id: str, *, kind: str,
                   app_name: str = "", display_name: str = "",
                   authorization_ref: str | None = None) -> None:
    conn.execute(
        "INSERT INTO mesh_peer_profiles (peer_id, kind, app_name, display_name, "
        "authorization_ref, muted, updated_at) VALUES (?, ?, ?, ?, ?, 0, ?) "
        "ON CONFLICT(peer_id) DO UPDATE SET kind=excluded.kind, "
        "app_name=excluded.app_name, display_name=excluded.display_name, "
        "authorization_ref=excluded.authorization_ref, updated_at=excluded.updated_at",
        (peer_id, kind, app_name, display_name, authorization_ref, _now()),
    )


def peer_exists(conn: sqlite3.Connection, peer_id: str, profile_id: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM mesh_peers WHERE peer_id=? AND profile_id=?",
        (peer_id, profile_id),
    ).fetchone() is not None


def _ensure_profile(conn: sqlite3.Connection, peer_id: str) -> None:
    conn.execute(
        "INSERT OR IGNORE INTO mesh_peer_profiles (peer_id, kind, updated_at) "
        "VALUES (?, 'local', ?)",
        (peer_id, _now()),
    )


def clean_name(name: object) -> str:
    """A display name from outside: Unicode format characters (direction
    overrides, zero-width marks) and control characters dropped, 64 characters at most."""
    if not isinstance(name, str):
        return ""
    kept = "".join(c for c in name if unicodedata.category(c) not in ("Cf", "Cc"))
    return kept.strip()[:MAX_DISPLAY_NAME]


def set_muted(conn: sqlite3.Connection, peer_id: str, muted: bool,
              profile_id: str) -> dict:
    if not peer_exists(conn, peer_id, profile_id):
        return {"ok": False, "error": "peer not found"}
    _ensure_profile(conn, peer_id)
    conn.execute(
        "UPDATE mesh_peer_profiles SET muted=?, updated_at=? WHERE peer_id=?",
        (1 if muted else 0, _now(), peer_id),
    )
    return {"ok": True, "muted": bool(muted)}


def rename(conn: sqlite3.Connection, peer_id: str, display_name: str,
           profile_id: str) -> dict:
    name = display_name if isinstance(display_name, str) else ""
    if not 1 <= len(name) <= MAX_DISPLAY_NAME or not name.isprintable() or not name.strip():
        return {"ok": False, "error": "display_name must be 1-64 printable characters"}
    if not peer_exists(conn, peer_id, profile_id):
        return {"ok": False, "error": "peer not found"}
    _ensure_profile(conn, peer_id)
    conn.execute(
        "UPDATE mesh_peer_profiles SET display_name=?, updated_at=? WHERE peer_id=?",
        (name, _now(), peer_id),
    )
    return {"ok": True, "display_name": name}


def retire(conn: sqlite3.Connection, peer_id: str, profile_id: str) -> dict:
    """Retire a peer: drop its queued unread messages (both ways) and its live row."""
    if not peer_exists(conn, peer_id, profile_id):
        return {"ok": False, "error": "peer not found"}
    _ensure_profile(conn, peer_id)
    now = _now()
    session = conn.execute(
        "SELECT session_id FROM mesh_peers WHERE peer_id=? AND profile_id=?",
        (peer_id, profile_id),
    ).fetchone()
    # The retirement is kept against the connection's own reference, so the
    # same reference cannot simply register again as a new peer.
    conn.execute(
        "UPDATE mesh_peer_profiles SET retired_at=?, updated_at=?, "
        "authorization_ref=COALESCE(authorization_ref, ?) WHERE peer_id=?",
        (now, now, session["session_id"] if session else None, peer_id),
    )
    ids = [r[0] for r in conn.execute(
        "SELECT id FROM mesh_messages WHERE profile_id=? AND COALESCE(read, 0)=0 "
        "AND (from_peer=? OR (to_peer=? AND target_type='peer'))",
        (profile_id, peer_id, peer_id),
    )]
    for start in range(0, len(ids), 500):
        chunk = ids[start:start + 500]
        marks = ",".join("?" * len(chunk))
        conn.execute(f"DELETE FROM mesh_message_envelopes WHERE message_id IN ({marks})", chunk)
        conn.execute(f"DELETE FROM mesh_reads WHERE message_id IN ({marks})", chunk)
        conn.execute(f"DELETE FROM mesh_messages WHERE id IN ({marks})", chunk)
    conn.execute("DELETE FROM mesh_peers WHERE peer_id=? AND profile_id=?",
                 (peer_id, profile_id))
    return {"ok": True, "dropped": len(ids)}


def is_retired_ref(conn: sqlite3.Connection, ref: str) -> bool:
    """Has the owner retired the peer that registered under this reference?"""
    return conn.execute(
        "SELECT 1 FROM mesh_peer_profiles WHERE authorization_ref=? "
        "AND retired_at IS NOT NULL LIMIT 1", (ref,),
    ).fetchone() is not None


def sender_block(conn: sqlite3.Connection, peer_id: str) -> str | None:
    """Why this sender may not send, or None."""
    row = conn.execute(
        "SELECT muted, retired_at FROM mesh_peer_profiles WHERE peer_id=?", (peer_id,),
    ).fetchone()
    if row is None:
        return None
    if row["retired_at"]:
        return "peer is retired"
    return "peer is muted by the owner" if row["muted"] else None


def resolve_hop(conn: sqlite3.Connection, reply_to: int | None,
                profile_id: str) -> int:
    """Hop for a new message; raises ValueError("hop limit") past the limit."""
    parent = None
    if reply_to is not None:
        parent = conn.execute(
            "SELECT e.hop, e.from_kind FROM mesh_message_envelopes e "
            "JOIN mesh_messages m ON m.id = e.message_id "
            "WHERE e.message_id=? AND m.profile_id=?",
            (reply_to, profile_id),
        ).fetchone()
    hop = next_hop(parent["hop"] if parent else None,
                   parent["from_kind"] if parent else None)
    check_hop(hop)
    return hop


def gate_send(conn: sqlite3.Connection, *, kind: str, from_peer: str,
              refs: Sequence[str], reply_to: int | None,
              profile_id: str) -> tuple[dict | None, dict | None]:
    """Gate a send and compute its envelope fields: ``(error, fields)``.

    Mute and retire apply to every sender, local or web. ``fields`` is None
    when the message needs no envelope row (a plain local send).
    """
    blocked = sender_block(conn, from_peer)
    if blocked:
        return {"ok": False, "error": blocked}, None
    if kind != "web" and not refs and reply_to is None:
        return None, None
    try:
        clean_refs = validate_refs(refs)
        hop = resolve_hop(conn, reply_to, profile_id) if kind == "web" else 0
    except ValueError as exc:
        return {"ok": False, "error": str(exc)}, None
    return None, {"refs": clean_refs, "hop": hop, "reply_to": reply_to}


def insert_envelope(conn: sqlite3.Connection, message_id: int, *, kind: str,
                    app: str, fields: dict) -> None:
    conn.execute(
        "INSERT INTO mesh_message_envelopes "
        "(message_id, from_kind, from_app, hop, refs_json, reply_to) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (message_id, kind, app, fields["hop"], json.dumps(fields["refs"]),
         fields["reply_to"]),
    )
