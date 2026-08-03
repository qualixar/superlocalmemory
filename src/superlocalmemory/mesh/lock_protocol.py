# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V4 | https://qualixar.com | https://varunpratap.com

"""SLM Mesh — leaderless cross-node lock coordination (3c-2).

SAFETY MODEL (never overclaim):
  ``resolve()`` converges the ADVISORY lock VIEW across nodes using a
  deterministic ``(fencing_token, node_id)`` total order.  Higher wins.
  When the remote node wins, the local ``mesh_locks`` row is deleted so
  the local node's token becomes stale.

  The FENCING TOKEN (``broker.validate_lock_fence``) is the single-writer
  SAFETY guarantee: even during a brief split where both nodes held the
  lock, any write attempt with the yielded node's old token is REJECTED
  by the storage layer — because either:
    (a) the row is gone → "no lock held for this resource", or
    (b) the winning node has since acquired with a higher token →
        "fencing token X is stale; current token is Y".

  **The advisory lock alone does NOT guarantee mutual exclusion; only the
  fence does.**  This is NOT linearizable consensus (which requires a
  quorum).

  All new behavior is ADDITIVE and only active when ``resolve()`` is called
  by the sync layer.  Node-local ``lock_action`` behavior is UNCHANGED.

Part of Qualixar | Author: Varun Pratap Bhardwaj
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from superlocalmemory.mesh.broker import _NEVER_EXPIRES
from superlocalmemory.mesh.node_identity import get_node_id

if TYPE_CHECKING:
    from superlocalmemory.mesh.broker import MeshBroker

logger = logging.getLogger("superlocalmemory.mesh.lock_protocol")


class LockCoordinator:
    """Leaderless cross-node advisory lock coordinator.

    Wraps a ``MeshBroker`` instance and adds the cross-machine protocol
    layer: export local live locks (``local_lock_delta``) and converge
    the lock view against a remote peer's claims (``resolve``).

    All database operations are fail-soft: errors are logged and an empty
    / no-op result is returned so callers (including the daemon sync loop)
    are never interrupted.
    """

    def __init__(self, broker: "MeshBroker") -> None:
        self._broker = broker
        self._db_path: str = broker._db_path
        self._node_id: str = get_node_id(self._db_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def local_lock_delta(
        self,
        profile_id: str = "default",
        now_iso: str | None = None,
    ) -> list[dict]:
        """Return live local locks for a profile, annotated with this node's id.

        A lock is *live* iff its ``expires_at`` is set, not the legacy
        ``_NEVER_EXPIRES`` sentinel, and not yet elapsed.

        Args:
            profile_id: Tenant scope (default ``"default"``).
            now_iso:    UTC ISO timestamp; injectable for test determinism.

        Returns:
            List of ``{"file_path","locked_by","locked_at","expires_at",
            "fencing_token","node_id"}`` dicts.  ``[]`` on DB error (fail-soft).
        """
        now: str = now_iso or datetime.now(timezone.utc).isoformat()
        conn: sqlite3.Connection | None = None
        try:
            conn = sqlite3.connect(self._db_path, timeout=5.0)
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT file_path, locked_by, locked_at, expires_at,"
                " COALESCE(fencing_token, 0) AS fencing_token"
                " FROM mesh_locks WHERE profile_id=?",
                (profile_id,),
            ).fetchall()
        except sqlite3.Error as exc:
            logger.error("local_lock_delta: DB error for profile %r: %s", profile_id, exc)
            return []
        finally:
            if conn is not None:
                conn.close()

        result: list[dict] = []
        for row in rows:
            if not self._row_is_live(row["expires_at"], now):
                continue
            result.append({
                "file_path": row["file_path"],
                "locked_by": row["locked_by"],
                "locked_at": row["locked_at"],
                "expires_at": row["expires_at"],
                "fencing_token": int(row["fencing_token"]),
                "node_id": self._node_id,
            })
        return result

    def resolve(
        self,
        profile_id: str,
        remote_locks: list[dict],
    ) -> dict:
        """Converge the local lock view against a remote peer's live claims.

        For each *live* remote lock, compare it to the local ``mesh_locks``
        row using ``(fencing_token DESC, node_id DESC)``.  Remote wins →
        delete local row (token goes stale; fence will reject it).  Only
        ``file_path`` values in ``remote_locks`` are examined (others untouched).
        Idempotent: a second call after a yield is a no-op.

        Args:
            profile_id:   Tenant scope.
            remote_locks: Dicts from the remote peer's ``local_lock_delta``
                          (must include ``file_path``, ``fencing_token``,
                          ``node_id``, ``expires_at``).

        Returns:
            ``{"yielded": [paths], "kept": N}``.
            ``{"yielded": [], "kept": 0}`` on any error (fail-soft).
        """
        try:
            return self._resolve_inner(profile_id, remote_locks)
        except Exception as exc:  # broad catch — must never crash daemon
            logger.error(
                "resolve: unexpected error for profile %r: %s",
                profile_id, exc,
                exc_info=True,
            )
            return {"yielded": [], "kept": 0}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_inner(
        self,
        profile_id: str,
        remote_locks: list[dict],
    ) -> dict:
        """Core resolve logic — separated so the outer method can catch all errors."""
        now: str = datetime.now(timezone.utc).isoformat()

        # Build a map of file_path → strongest LIVE remote claim.
        # Audit P2: a non-dict entry must never crash resolve (per-entry skip).
        # Audit P2: if a caller ever concatenates multiple peers' deltas, keep
        # the winner under the SAME total order (token, node_id) rather than
        # letting a later, weaker claim overwrite a stronger one.
        live_remote: dict[str, dict] = {}
        for rlock in remote_locks:
            if not isinstance(rlock, dict):
                continue
            fp = rlock.get("file_path", "")
            if not fp or not self._dict_is_live(rlock, now):
                continue
            existing = live_remote.get(fp)
            if existing is None or self._lock_key(rlock) > self._lock_key(existing):
                live_remote[fp] = rlock

        if not live_remote:
            return {"yielded": [], "kept": 0}

        # Fetch local rows for the contested file_paths only — including
        # expires_at (audit P1: an EXPIRED local row must not win over a live
        # remote claim just because it carries a higher token).
        file_paths: tuple[str, ...] = tuple(live_remote.keys())
        placeholders = ",".join("?" * len(file_paths))
        conn: sqlite3.Connection = sqlite3.connect(self._db_path, timeout=5.0)
        try:
            conn.row_factory = sqlite3.Row
            try:
                local_rows = conn.execute(
                    f"SELECT file_path, expires_at,"
                    f" COALESCE(fencing_token, 0) AS fencing_token"
                    f" FROM mesh_locks WHERE profile_id=? AND file_path IN ({placeholders})",
                    (profile_id, *file_paths),
                ).fetchall()
            except sqlite3.Error as exc:
                logger.error(
                    "resolve: failed to query local locks for profile %r: %s",
                    profile_id, exc,
                )
                return {"yielded": [], "kept": 0}

            local_by_path: dict[str, dict] = {
                row["file_path"]: {
                    "fencing_token": int(row["fencing_token"]),
                    "expires_at": row["expires_at"],
                }
                for row in local_rows
            }

            yielded: list[str] = []
            kept: int = 0

            for fp, rlock in live_remote.items():
                local = local_by_path.get(fp)
                if local is None:
                    continue  # No local row — nothing to yield or keep.

                snapshot_token: int = local["fencing_token"]
                local_live: bool = self._row_is_live(local["expires_at"], now)

                # Yield when: (a) the local row is EXPIRED garbage while a live
                # remote claim exists (audit P1 — clean it up so its stale token
                # can't pass the fence), or (b) the live remote strictly wins the
                # total order.
                if local_live and not self._remote_wins(rlock, local):
                    kept += 1
                    continue

                # Token-CONDITIONAL delete (audit P1/TOCTOU): only delete the
                # row we compared against. If the local node reacquired to a new
                # (higher) token between snapshot and now, the predicate misses
                # and we do NOT wipe the fresher live lock.
                try:
                    cur = conn.execute(
                        "DELETE FROM mesh_locks WHERE profile_id=? AND file_path=?"
                        " AND COALESCE(fencing_token, 0)=?",
                        (profile_id, fp, snapshot_token),
                    )
                    conn.commit()
                    if cur.rowcount and cur.rowcount > 0:
                        logger.info(
                            "resolve: yielded lock %r to remote node %r "
                            "(remote (tok=%s,node=%s) vs local (tok=%s), local_live=%s)",
                            fp, rlock.get("node_id", "?"), self._safe_token(rlock),
                            rlock.get("node_id", "?"), snapshot_token, local_live,
                        )
                        yielded.append(fp)
                    # rowcount==0 → lost the race (reacquired); leave it, don't count.
                except sqlite3.Error as exc:
                    logger.error(
                        "resolve: failed to yield lock %r for profile %r: %s",
                        fp, profile_id, exc,
                    )

            return {"yielded": yielded, "kept": kept}
        finally:
            conn.close()

    @staticmethod
    def _row_is_live(expires_at: str | None, now_iso: str) -> bool:
        """Return True iff a DB row's expires_at represents a live lock."""
        if not expires_at:
            return False
        if expires_at == _NEVER_EXPIRES:
            return False
        return expires_at > now_iso

    @staticmethod
    def _dict_is_live(lock: dict, now_iso: str) -> bool:
        """Return True iff a remote lock dict represents a live claim."""
        exp = lock.get("expires_at", "") or ""
        if not exp:
            return False
        if exp == _NEVER_EXPIRES:
            return False
        return exp > now_iso

    @staticmethod
    def _safe_token(lock: dict) -> int:
        """Cast fencing_token to int defensively (remote dicts may carry strings)."""
        try:
            return int(lock.get("fencing_token", 0))
        except (TypeError, ValueError):
            return 0

    def _lock_key(self, lock: dict) -> tuple[int, str]:
        """Total-order key ``(fencing_token, node_id)`` for comparing claims."""
        return (self._safe_token(lock), str(lock.get("node_id", "")))

    def _remote_wins(self, remote: dict, local: dict) -> bool:
        """Total order: ``(fencing_token DESC, node_id DESC)``.

        Compare as ``int`` for the token (critical — JSON may deliver strings,
        and string comparison ``"10" > "9"`` is ``False``, which would invert
        the order).  Tie-break via lexicographic ``node_id`` comparison; both
        sides use consistent lowercase hex from ``get_node_id``, so the order
        is stable across nodes.

        Returns True iff the remote lock should be considered the effective
        holder over the local lock.
        """
        r_tok: int = self._safe_token(remote)
        l_tok: int = local.get("fencing_token", 0)

        if r_tok != l_tok:
            return r_tok > l_tok

        # Token tie — use node_id as deterministic tie-breaker.
        r_nid: str = str(remote.get("node_id", ""))
        l_nid: str = self._node_id
        return r_nid > l_nid
