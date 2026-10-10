# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Delivery leases for the mail of a connected web app.

A web app reaches the broker through a relay that can drop the reply after the
broker has handed the mail over. So handing mail over does not mark it read: it
takes a *lease* on each message (a row in ``mesh_web_deliveries``, created with the other broker tables in
``broker_security``). While the
lease runs, the message is in flight and no other call receives it. The app
acknowledges a message by listing its id in ``ack`` on its next ``mesh_inbox`` or
``mesh_wait`` call; the message is then marked read for good. A message that is
not acknowledged is handed over again once the lease has run out, flagged
``"repeat": true``, and after ``MAX_DELIVERIES`` hand-overs in total it is
treated as delivered and marked read, so a message can never loop forever.

Every function takes an open connection inside the caller's write transaction.
Local agents never touch this table: their inbox is unchanged.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable

LEASE_S = 120.0
MAX_DELIVERIES = 3
MAX_ACK = 100

def clean_ack(ack: object) -> list[int]:
    """The message ids in ``ack``: whole numbers only, at most ``MAX_ACK``."""
    if not isinstance(ack, (list, tuple)):
        return []
    ids = [i for i in ack if isinstance(i, int) and not isinstance(i, bool) and i > 0]
    return ids[:MAX_ACK]


def _marks(ids: Iterable[object]) -> str:
    return ",".join("?" * len(list(ids)))


def acknowledge(conn: sqlite3.Connection, peer_id: str, profile_id: str,
                ids: list[int]) -> None:
    """Mark the app's own acknowledged mail read; ids of other peers are ignored."""
    if not ids:
        return
    marks = _marks(ids)
    mine = [r[0] for r in conn.execute(
        f"SELECT id FROM mesh_messages WHERE id IN ({marks}) AND to_peer=? "
        "AND profile_id=? AND target_type='peer'", (*ids, peer_id, profile_id))]
    if not mine:
        return
    marks = _marks(mine)
    conn.execute(f"UPDATE mesh_messages SET read=1 WHERE id IN ({marks})", mine)
    conn.execute(f"DELETE FROM mesh_web_deliveries WHERE message_id IN ({marks})", mine)


def _leases(conn: sqlite3.Connection, ids: list[int]) -> dict[int, tuple[float, int]]:
    if not ids:
        return {}
    rows = conn.execute(
        "SELECT message_id, delivered_at, deliveries FROM mesh_web_deliveries "
        f"WHERE message_id IN ({_marks(ids)})", ids).fetchall()
    return {r[0]: (r[1], r[2]) for r in rows}


def deliverable(conn: sqlite3.Connection, msgs: list[dict], now: float) -> list[dict]:
    """The messages to hand over now, with repeats flagged.

    Skips what is in flight, and retires (marks read) what has used up its
    hand-overs. Returns new dicts; the input is not changed.
    """
    leases = _leases(conn, [m["id"] for m in msgs])
    out: list[dict] = []
    spent: list[int] = []
    for msg in msgs:
        lease = leases.get(msg["id"])
        if lease is None:
            out.append(dict(msg))
        elif lease[0] + LEASE_S > now:
            continue
        elif lease[1] >= MAX_DELIVERIES:
            spent.append(msg["id"])
        else:
            out.append({**msg, "repeat": True, "delivery": lease[1] + 1})
    if spent:
        conn.execute(f"UPDATE mesh_messages SET read=1 WHERE id IN ({_marks(spent)})", spent)
        conn.execute(
            f"DELETE FROM mesh_web_deliveries WHERE message_id IN ({_marks(spent)})", spent)
    return out


def record(conn: sqlite3.Connection, msgs: list[dict], now: float) -> None:
    """Take (or renew) the lease on every message handed over."""
    conn.executemany(
        "INSERT INTO mesh_web_deliveries (message_id, delivered_at, deliveries) "
        "VALUES (?, ?, 1) ON CONFLICT(message_id) DO UPDATE SET "
        "delivered_at=excluded.delivered_at, deliveries=deliveries+1",
        [(m["id"], now) for m in msgs])


def has_deliverable(conn: sqlite3.Connection, peer_id: str, profile_id: str,
                    now: float, expires_iso: str) -> bool:
    """Whether an unread direct message is free to hand over (a plain read)."""
    return conn.execute(
        "SELECT 1 FROM mesh_messages m LEFT JOIN mesh_web_deliveries d "
        "ON d.message_id = m.id WHERE m.profile_id=? AND m.to_peer=? "
        "AND m.target_type='peer' AND COALESCE(m.read, 0)=0 "
        "AND (m.expires_at IS NULL OR m.expires_at > ?) "
        "AND (d.message_id IS NULL OR (d.delivered_at + ? <= ? AND d.deliveries < ?)) "
        "LIMIT 1",
        (profile_id, peer_id, expires_iso, LEASE_S, now, MAX_DELIVERIES),
    ).fetchone() is not None
