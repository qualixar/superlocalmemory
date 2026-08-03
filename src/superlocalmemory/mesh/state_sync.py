# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V4 | https://qualixar.com | https://varunpratap.com

"""LWW remote state convergence for the SLM mesh (3c-1).

Convergence guarantee
---------------------
Leaderless, pull-based, deterministic Last-Writer-Wins.  Every node that
exchanges deltas converges on the same ``(value, revision, node_id)`` for
every key.

This is NOT linearizable consensus.  Concurrent writes can interleave; the
winner is chosen deterministically by a TOTAL ORDER on ``(revision: int,
node_id: str)``.  Same revision → higher ``node_id`` wins (lexicographic).
Because the order is total, every node independently computes the same
winner given the same set of writes.

Backward compatibility
----------------------
``set_state`` in the broker never sets ``origin_node``; rows it writes have
``origin_node = ''`` (the column default added here).  At merge and
serialization time ``''`` is interpreted as the LOCAL node_id, so existing
rows participate correctly in the LWW protocol without any change to
``broker.py``.

Author: Varun Pratap Bhardwaj
"""

from __future__ import annotations

import logging
import sqlite3
from typing import Any

from superlocalmemory.mesh.node_identity import get_node_id

logger = logging.getLogger("superlocalmemory.mesh.state_sync")

# ---------------------------------------------------------------------------
# Schema migration (additive, idempotent)
# ---------------------------------------------------------------------------

_ALTER_ORIGIN_NODE = (
    "ALTER TABLE mesh_state ADD COLUMN origin_node TEXT NOT NULL DEFAULT ''"
)


# ---------------------------------------------------------------------------
# StateSyncer
# ---------------------------------------------------------------------------


class StateSyncer:
    """Pull-based LWW convergence helper for the ``mesh_state`` table.

    Designed to be instantiated on-demand (route handler, sync loop); all DB
    connections are short-lived.  Every operation is fail-soft: errors are
    logged and the caller never receives an unhandled exception from here.
    """

    def __init__(self, broker: Any) -> None:
        self._broker = broker
        self._db_path: str = str(broker._db_path)
        # Resolved once; stable for the lifetime of this instance.
        self._node_id: str = get_node_id(self._db_path)
        self._ensure_origin_node_column()

    # ------------------------------------------------------------------
    # Schema bootstrap
    # ------------------------------------------------------------------

    def _ensure_origin_node_column(self) -> None:
        """Add ``origin_node`` to ``mesh_state``; no-op if already present."""
        try:
            conn = sqlite3.connect(self._db_path, timeout=5.0)
            try:
                conn.execute(_ALTER_ORIGIN_NODE)
                conn.commit()
            except sqlite3.OperationalError as exc:
                if "duplicate column" in str(exc).lower():
                    pass  # Expected on second startup / upgrade
                else:
                    logger.error(
                        "StateSyncer._ensure_origin_node_column: unexpected "
                        "OperationalError at %s: %s",
                        self._db_path, exc,
                    )
            finally:
                conn.close()
        except sqlite3.Error as exc:
            logger.error(
                "StateSyncer._ensure_origin_node_column: DB error at %s: %s",
                self._db_path, exc,
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _effective_node(self, stored_origin: str) -> str:
        """Resolve ``origin_node=''`` (BC rows) to this node's id at call time."""
        return stored_origin if stored_origin else self._node_id

    def _open_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, timeout=5.0)
        conn.row_factory = sqlite3.Row
        return conn

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def local_delta(
        self,
        profile_id: str = "default",
        since_revision: int = 0,
    ) -> list[dict]:
        """Return rows with ``revision > since_revision`` for a profile.

        The ``node_id`` field in each entry reflects the *effective* node:
        the stored ``origin_node`` when non-empty, or the local node_id for
        BC rows written by the broker (``origin_node = ''``).

        Returns ``[]`` on any DB error (fail-soft).
        """
        try:
            conn = self._open_conn()
            try:
                rows = conn.execute(
                    "SELECT key, value, set_by, updated_at,"
                    " COALESCE(revision, 0) AS revision,"
                    " COALESCE(origin_node, '') AS origin_node"
                    " FROM mesh_state"
                    " WHERE profile_id = ? AND COALESCE(revision, 0) > ?",
                    (profile_id, int(since_revision)),
                ).fetchall()
                return [
                    {
                        "key": row["key"],
                        "value": row["value"],
                        "set_by": row["set_by"],
                        "updated_at": row["updated_at"],
                        "revision": int(row["revision"]),
                        "node_id": self._effective_node(row["origin_node"]),
                    }
                    for row in rows
                ]
            finally:
                conn.close()
        except sqlite3.Error as exc:
            logger.error(
                "StateSyncer.local_delta: DB error at %s: %s",
                self._db_path, exc,
            )
            return []

    def merge_remote(
        self,
        profile_id: str,
        remote_entries: list[dict],
    ) -> dict:
        """Merge remote delta entries using deterministic LWW.

        For each entry ``{key, value, set_by, updated_at, revision, node_id}``:

        * If ``(remote_rev, remote_node) > (local_rev, local_node)`` (total
          order): UPSERT the local row.  The winning ``revision`` is preserved
          as-is — never incremented — so re-merging the same delta is a no-op
          (idempotent convergence).
        * Otherwise: do nothing.

        Returns ``{"applied": N, "skipped": M}``.  Errors per-entry are
        logged and counted as skipped; they never propagate to the caller.
        """
        applied = 0
        skipped = 0
        for entry in remote_entries:
            # Compute a safe log key BEFORE the try so a non-dict entry can
            # never make the error handler itself raise (audit P0/P2 — the
            # handler previously called entry.get() on possibly-non-dict).
            entry_key = entry.get("key") if isinstance(entry, dict) else repr(entry)
            try:
                if self._merge_one(profile_id, entry):
                    applied += 1
                else:
                    skipped += 1
            except sqlite3.Error as exc:
                logger.error(
                    "StateSyncer.merge_remote: DB error on key=%r: %s",
                    entry_key, exc,
                )
                skipped += 1
            except (KeyError, TypeError, ValueError, AttributeError) as exc:
                logger.error(
                    "StateSyncer.merge_remote: malformed entry key=%r: %s",
                    entry_key, exc,
                )
                skipped += 1
        return {"applied": applied, "skipped": skipped}

    def _merge_one(self, profile_id: str, entry: dict) -> bool:
        """Apply one remote entry if it wins the LWW comparison.

        Returns ``True`` if the local row was updated; ``False`` if local
        won or the comparison was a tie (same revision, same node_id).

        CRITICAL: ``revision`` is never incremented here.  Preserving the
        winning revision is what makes convergence idempotent — a second
        merge of the same delta computes the identical comparison result and
        takes no action.
        """
        # Audit P0/P2: a non-dict entry must never crash the merge.
        if not isinstance(entry, dict):
            return False
        key = str(entry["key"])
        # Guard: cast revision to int to prevent string lexicographic miscompare
        # (e.g. "10" < "9" as strings but 10 > 9 as ints).
        remote_rev: int = int(entry["revision"])
        remote_node: str = str(entry.get("node_id", "")).strip()
        # Audit P1: an empty node_id would be stored as origin_node='' and then
        # re-exported under THIS node's id, rewriting provenance and corrupting
        # the tie-break. Refuse an entry without a real origin.
        if not remote_node:
            return False
        # Audit P2: never persist a NULL value as the literal string "None".
        if entry.get("value") is None:
            return False

        conn = self._open_conn()
        # Audit P1 (TOCTOU): the read-compare-write MUST be atomic against a
        # concurrent broker.set_state on the same key. Without a write lock, a
        # stale local read lets a lower remote revision overwrite a fresher
        # local write (losing local-highest-revision). BEGIN IMMEDIATE takes the
        # write lock for the whole critical section.
        conn.isolation_level = None  # manual transaction control
        try:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT COALESCE(revision, 0) AS revision,"
                " COALESCE(origin_node, '') AS origin_node"
                " FROM mesh_state"
                " WHERE profile_id = ? AND key = ?",
                (profile_id, key),
            ).fetchone()

            if row is None:
                local_rev: int = 0
                local_node: str = ""  # No row — treat as the absolute minimum
            else:
                local_rev = int(row["revision"])
                local_node = self._effective_node(row["origin_node"])

            # Total order: compare revision (int) first; break ties by node_id (str lex).
            # A strict > means ties-to-local (same rev, same node) → do nothing (idempotent).
            remote_wins: bool = (remote_rev, remote_node) > (local_rev, local_node)
            if not remote_wins:
                conn.execute("ROLLBACK")
                return False

            # UPSERT: set origin_node = remote_node so subsequent merges of the
            # same delta compute the same winner without re-resolving local node_id.
            conn.execute(
                "INSERT INTO mesh_state"
                " (profile_id, key, value, set_by, updated_at, revision, origin_node)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)"
                " ON CONFLICT(profile_id, key) DO UPDATE SET"
                " value     = excluded.value,"
                " set_by    = excluded.set_by,"
                " updated_at = excluded.updated_at,"
                " revision  = excluded.revision,"
                " origin_node = excluded.origin_node",
                (
                    profile_id,
                    key,
                    str(entry["value"]),
                    str(entry["set_by"]),
                    str(entry["updated_at"]),
                    remote_rev,
                    remote_node,
                ),
            )
            conn.execute("COMMIT")
            return True
        finally:
            conn.close()
