# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""The mesh's periodic cleanup: stale peers, read and expired messages, old data.

Runs inside one transaction the broker opens and retries. Nothing here commits
except :func:`run_cleanup` at its end.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone

#: A connected web app is not a process that heartbeats, so the liveness rules
#: below never touch it. It leaves only when the owner revokes the app.
_NOT_WEB = "peer_id NOT IN (SELECT peer_id FROM mesh_peer_profiles WHERE kind='web')"


def _reap_peers(conn: sqlite3.Connection, now: datetime) -> None:
    # ISO-8601 UTC timestamps sort lexicographically, so a bare `col < ?` is
    # both correct and sargable.
    five_min = (now - timedelta(minutes=5)).isoformat()
    thirty_min = (now - timedelta(minutes=30)).isoformat()
    conn.execute(
        "UPDATE mesh_peers SET status='stale' "
        f"WHERE status='active' AND last_heartbeat < ? AND {_NOT_WEB}",
        (five_min,),
    )
    conn.execute(
        "UPDATE mesh_peers SET status='dead' "
        f"WHERE status='stale' AND last_heartbeat < ? AND {_NOT_WEB}",
        (thirty_min,),
    )
    conn.execute("DELETE FROM mesh_peers WHERE status='dead'")


def _reap_messages(conn: sqlite3.Connection, now: datetime) -> None:
    now_iso = now.isoformat()
    day_ago = (now - timedelta(hours=24)).isoformat()
    week_ago = (now - timedelta(days=7)).isoformat()
    conn.execute(
        "DELETE FROM mesh_messages WHERE target_type='peer' AND read=1 "
        "AND created_at < ?",
        (day_ago,),
    )
    conn.execute(
        "DELETE FROM mesh_messages WHERE expires_at IS NOT NULL "
        "AND expires_at < ?",
        (now_iso,),
    )
    conn.execute(
        "DELETE FROM mesh_reads WHERE message_id NOT IN "
        "(SELECT id FROM mesh_messages)",
    )
    # Envelope rows go with their message, whichever rule removed it.
    conn.execute(
        "DELETE FROM mesh_message_envelopes WHERE message_id NOT IN "
        "(SELECT id FROM mesh_messages)",
    )
    conn.execute(
        "DELETE FROM mesh_web_deliveries WHERE message_id NOT IN "
        "(SELECT id FROM mesh_messages)",
    )
    # The 9999-... sentinel of a legacy lock sorts after any real now.
    conn.execute("DELETE FROM mesh_locks WHERE expires_at < ?", (now_iso,))
    conn.execute("DELETE FROM mesh_events WHERE created_at < ?", (week_ago,))


def run_cleanup(conn: sqlite3.Connection) -> None:
    """Mark stale peers, delete dead ones and expired data, then commit."""
    now = datetime.now(timezone.utc)
    _reap_peers(conn, now)
    _reap_messages(conn, now)
    conn.commit()
