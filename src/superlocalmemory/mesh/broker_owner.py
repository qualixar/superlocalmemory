# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Broker methods for the bounded inbox wait and the owner's peer controls.

A mixin so ``MeshBroker`` stays small; it relies on the broker's own
``_conn``, ``_write_with_retry``, ``_log_event``, ``get_inbox`` and ``_waiter``.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable

from . import broker_inbox, broker_profiles


class OwnerControlsMixin:
    """Wait, owner message list, and peer profile controls."""

    def wait_inbox(self, peer_id: str, *, timeout_s: float, project_path: str = "",
                   profile_id: str = "default",
                   remote_view: bool = False) -> tuple[list[dict], bool]:
        """Block up to a clamped timeout for unread mail; ``(messages, timed_out)``.

        Raises ``RuntimeError("too many waits")`` past the concurrent-wait cap.
        """
        return self._waiter.wait(
            lambda: self.get_inbox(peer_id, project_path, profile_id,
                                   remote_view=remote_view),
            timeout_s,
        )

    def list_messages(self, limit: int = 50, peer: str = "",
                      profile_id: str = "default") -> list[dict]:
        """Envelopes of stored messages, newest first (for the owner's dashboard)."""
        conn = self._conn()
        try:
            return broker_inbox.query_messages(conn, profile_id, limit, peer)
        finally:
            conn.close()

    # -- Peer profiles (owner controls) --

    def upsert_peer_profile(self, peer_id: str, *, kind: str, app_name: str = "",
                            display_name: str = "",
                            authorization_ref: str | None = None) -> dict:
        def _upsert(conn: sqlite3.Connection) -> dict:
            broker_profiles.upsert_profile(
                conn, peer_id, kind=kind, app_name=app_name,
                display_name=display_name, authorization_ref=authorization_ref,
            )
            conn.commit()
            return {"ok": True}

        return self._write_with_retry(_upsert)

    def set_muted(self, peer_id: str, muted: bool,
                  profile_id: str = "default") -> dict:
        return self._profile_change(
            "peer_muted", peer_id, profile_id,
            lambda c: broker_profiles.set_muted(c, peer_id, muted, profile_id),
        )

    def rename_peer(self, peer_id: str, display_name: str,
                    profile_id: str = "default") -> dict:
        return self._profile_change(
            "peer_renamed", peer_id, profile_id,
            lambda c: broker_profiles.rename(c, peer_id, display_name, profile_id),
        )

    def retire_peer(self, peer_id: str, profile_id: str = "default") -> dict:
        """Retire a peer and drop its queued unread messages (a revoke)."""
        return self._profile_change(
            "peer_retired", peer_id, profile_id,
            lambda c: broker_profiles.retire(c, peer_id, profile_id),
        )

    def _profile_change(self, event: str, peer_id: str, profile_id: str,
                        change: Callable[[sqlite3.Connection], dict]) -> dict:
        """Apply one owner change and log it (ids and counts only, never text)."""
        def _apply(conn: sqlite3.Connection) -> dict:
            result = change(conn)
            if result.get("ok"):
                payload = {k: v for k, v in result.items()
                           if k in ("muted", "dropped") and not isinstance(v, str)}
                self._log_event(conn, event, peer_id, payload, profile_id=profile_id)
                conn.commit()
            return result

        return self._write_with_retry(_apply)

    def peer_id_for_session(self, session_id: str,
                            profile_id: str = "default") -> str | None:
        conn = self._conn()
        try:
            row = conn.execute(
                "SELECT peer_id FROM mesh_peers WHERE session_id=? AND profile_id=?",
                (session_id, profile_id),
            ).fetchone()
            return row["peer_id"] if row else None
        finally:
            conn.close()
