# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Broker methods for connected web apps, which the daemon serves in process.

A web app is a peer whose id is its stable reference (never a random id), who
is not a process that heartbeats, and who only ever reaches the broker through
the daemon's own code. It can message one peer by id and reads only mail
addressed to it. A mixin so ``MeshBroker`` stays small; it relies on the
broker's ``_conn``, ``_write_with_retry``, ``_log_event``, ``_waiter``,
``_remote_peers`` and ``send_message``.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Collection, Sequence
from datetime import datetime, timezone

from . import broker_inbox, broker_profiles
from .envelope import Origin

FALLBACK_NAME = "Web app "
DIRECTORY_LIMIT = 100
SUMMARY_LIMIT = 200


def fallback_name(peer_ref: str) -> str:
    """The name shown for a web app nobody has named yet."""
    return FALLBACK_NAME + peer_ref[2:8]


def _renamed(old: str, new: str, peer_ref: str) -> str:
    """The display name to keep: the owner's rename and a real name both beat the fallback."""
    return new if old in ("", fallback_name(peer_ref)) else old


def register_web_peer(conn: sqlite3.Connection, peer_ref: str, *, app: str,
                      display_name: str, connection_id: str, profile_id: str,
                      host: str) -> tuple[dict, bool]:
    """Create or refresh the peer row of a web app: ``(result, created)``."""
    prof = conn.execute(
        "SELECT retired_at, display_name FROM mesh_peer_profiles WHERE peer_id=?",
        (peer_ref,),
    ).fetchone()
    if prof is not None and prof["retired_at"]:
        return {"ok": False, "error": "peer is retired"}, False
    now = datetime.now(timezone.utc).isoformat()
    row = conn.execute("SELECT profile_id FROM mesh_peers WHERE peer_id=?",
                       (peer_ref,)).fetchone()
    if row is not None and row["profile_id"] != profile_id:
        return {"ok": False, "error": "peer belongs to a different profile"}, False
    if row is None:
        conn.execute(
            "INSERT INTO mesh_peers (peer_id, session_id, summary, status, host, port, "
            "registered_at, last_heartbeat, project_path, agent_type, profile_id) "
            "VALUES (?, ?, ?, 'active', ?, 0, ?, ?, '', ?, ?)",
            (peer_ref, peer_ref, display_name, host, now, now, app, profile_id),
        )
    else:
        conn.execute(
            "UPDATE mesh_peers SET last_heartbeat=?, status='active' WHERE peer_id=?",
            (now, peer_ref),
        )
    name = _renamed(prof["display_name"], display_name, peer_ref) if prof else display_name
    broker_profiles.upsert_profile(conn, peer_ref, kind="web", app_name=app,
                                   display_name=name, authorization_ref=peer_ref)
    conn.execute("UPDATE mesh_peer_profiles SET connection_ref=? WHERE peer_id=?",
                 (connection_id, peer_ref))
    return {"ok": True, "peer_id": peer_ref}, row is None


class WebPeersMixin:
    """Registration, directory, atomic reads and revoke sync for web apps."""

    def ensure_web_peer(self, peer_ref: str, *, app: str, display_name: str,
                        connection_id: str, profile_id: str = "default") -> dict:
        """Register the web app, or refresh it: the same reference is the same peer."""
        def _ensure(conn: sqlite3.Connection) -> dict:
            result, created = register_web_peer(
                conn, peer_ref, app=app, display_name=display_name,
                connection_id=connection_id, profile_id=profile_id, host=self._host)
            if created:
                self._log_event(conn, "peer_registered", peer_ref,
                                {"kind": "web"}, profile_id=profile_id)
            conn.commit()
            return result

        return self._write_with_retry(_ensure)

    def is_web_peer(self, peer_id: str) -> bool:
        conn = self._conn()
        try:
            return conn.execute(
                "SELECT 1 FROM mesh_peer_profiles WHERE peer_id=? AND kind='web'",
                (peer_id,),
            ).fetchone() is not None
        finally:
            conn.close()

    def list_peer_directory(self, profile_id: str = "default") -> list[dict]:
        """The peers a web app may address: no host, port, path or session detail."""
        conn = self._conn()
        try:
            found = conn.execute(
                "SELECT m.peer_id, m.summary, m.status, m.agent_type, "
                "COALESCE(p.kind, 'local') AS kind, COALESCE(p.display_name, '') AS display_name, "
                "COALESCE(p.app_name, '') AS app_name FROM mesh_peers m "
                "LEFT JOIN mesh_peer_profiles p ON p.peer_id = m.peer_id "
                "WHERE m.profile_id=? AND p.retired_at IS NULL "
                "ORDER BY m.last_heartbeat DESC LIMIT ?",
                (profile_id, DIRECTORY_LIMIT),
            ).fetchall()
        finally:
            conn.close()
        return [{
            "peer_id": r["peer_id"], "name": r["display_name"] or r["agent_type"] or "",
            "kind": r["kind"], "app": r["app_name"], "agent_type": r["agent_type"] or "",
            "summary": (r["summary"] or "")[:SUMMARY_LIMIT], "status": r["status"],
        } for r in found]

    def web_send(self, peer_ref: str, app: str, to: str, content: str, *,
                 refs: Sequence[str] = (), reply_to: int | None = None,
                 profile_id: str = "default") -> dict:
        """A send made for a web app: it addresses one peer of this computer."""
        with self._remote_peers_lock:
            if to in self._remote_peers:
                return {"ok": False, "error": "recipient peer not found"}
        return self.send_message(
            peer_ref, to, content, "text", "", profile_id,
            origin=Origin("web", app), refs=refs, reply_to=reply_to,
        )

    def claim_web_inbox(self, peer_id: str, profile_id: str = "default") -> list[dict]:
        """Take the unread direct mail of a web app: select and mark read in one
        transaction, so two readers never receive the same message."""
        def _claim(conn: sqlite3.Connection) -> list[dict]:
            conn.execute("BEGIN IMMEDIATE")
            msgs = broker_inbox.query_inbox(conn, peer_id, "", profile_id, direct_only=True)
            if msgs:
                marks = ",".join("?" * len(msgs))
                conn.execute(
                    f"UPDATE mesh_messages SET read=1 WHERE id IN ({marks}) "
                    "AND to_peer=? AND profile_id=? AND COALESCE(read, 0)=0",
                    (*[m["id"] for m in msgs], peer_id, profile_id),
                )
            broker_inbox.attach_envelopes(conn, msgs, remote_view=True)
            conn.commit()
            return msgs

        return self._write_with_retry(_claim)

    def wait_web_inbox(self, peer_id: str, *, timeout_s: float,
                       profile_id: str = "default") -> tuple[list[dict], bool]:
        """Block up to a clamped timeout for direct mail; ``(messages, timed_out)``.

        Raises ``RuntimeError("too many waits")`` past the concurrent-wait cap.
        """
        return self._waiter.wait(
            lambda: self.claim_web_inbox(peer_id, profile_id), timeout_s)

    def retire_missing_web_peers(self, connection_id: str, listed: Collection[str], *,
                                 registered_before: str | None = None) -> list[str]:
        """Retire the web peers of one connection that are not in ``listed``.

        ``listed`` holds the peer references of the apps the gateway still lists
        for the connection. Queued messages of a retired peer are dropped. With
        ``registered_before`` (an ISO time) only peers registered earlier are
        considered: an app first used after the list was read is not in it yet.
        """
        def _retire(conn: sqlite3.Connection) -> list[str]:
            found = conn.execute(
                "SELECT p.peer_id AS peer_id, m.profile_id AS profile_id "
                "FROM mesh_peer_profiles p JOIN mesh_peers m ON m.peer_id = p.peer_id "
                "WHERE p.kind='web' AND p.connection_ref=? AND p.retired_at IS NULL "
                "AND (? IS NULL OR m.registered_at < ?)",
                (connection_id, registered_before, registered_before),
            ).fetchall()
            gone = [(r["peer_id"], r["profile_id"]) for r in found
                    if r["peer_id"] not in listed]
            for peer_id, profile_id in gone:
                broker_profiles.retire(conn, peer_id, profile_id)
                self._log_event(conn, "peer_retired", peer_id, profile_id=profile_id)
            conn.commit()
            return [peer_id for peer_id, _ in gone]

        return self._write_with_retry(_retire)
