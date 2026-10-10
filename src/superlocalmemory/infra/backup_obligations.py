# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V4 | https://qualixar.com

"""Backup erasure obligation ledger — GDPR Art.17 backup residue tracking.

After Art.17 erasure the live stores are clean, but rotating backup snapshots
still hold the erased personal data.  This module tracks ``outstanding
obligations`` — one per (profile_id, snapshot_path) pair — so that:

  1. The erasure receipt accurately declares completeness as FALSE while any
     snapshot still contains the data (C1 gap closure).

  2. A restore that would resurrect erased data is intercepted and the profile
     data re-erased from the restored store before any caller can read it
     (restore-replay invariant).

  3. Obligations age out automatically when a snapshot exceeds the configured
     retention window, keeping the ledger finite.

IMPORTANT: backup_obligations.db is intentionally NOT in MANAGED_DATABASES
and is therefore never backed up itself.  This prevents a restore from loading
stale obligation state.
"""

from __future__ import annotations

import logging
import sqlite3
import time
import uuid
from pathlib import Path

logger = logging.getLogger("superlocalmemory.infra.backup_obligations")

_OBLIGATION_DB_NAME = "backup_obligations.db"

_DDL = """
CREATE TABLE IF NOT EXISTS backup_obligations (
    obligation_id    TEXT PRIMARY KEY,
    profile_id       TEXT NOT NULL,
    erasure_id       TEXT NOT NULL,
    snapshot_path    TEXT NOT NULL,
    snapshot_epoch   INTEGER NOT NULL,
    retention_days   INTEGER NOT NULL DEFAULT 90,
    recorded_at      REAL NOT NULL,
    status           TEXT NOT NULL DEFAULT 'pending',
    discharged_at    REAL,
    discharge_reason TEXT
);
CREATE INDEX IF NOT EXISTS idx_bko_profile
    ON backup_obligations(profile_id, status);
CREATE INDEX IF NOT EXISTS idx_bko_snapshot
    ON backup_obligations(snapshot_path, status);
CREATE INDEX IF NOT EXISTS idx_bko_erasure
    ON backup_obligations(erasure_id);
"""


class BackupObligationStore:
    """Persistent ledger of outstanding erasure obligations against snapshots.

    An obligation is created for each backup snapshot that was found to contain
    data for a profile that has since been Art.17-erased from the live stores.
    The obligation is discharged when:
      * The snapshot ages past the configured retention window (auto-discharge).
      * The snapshot is restored and the profile data is re-erased during
        restore-replay (explicit discharge).

    The store lives at ``<data_root>/backup_obligations.db``.  It is a plain
    SQLite file that is NOT in MANAGED_DATABASES and therefore never appears
    inside a backup set.
    """

    def __init__(self, data_root: Path) -> None:
        self._db_path = Path(data_root) / _OBLIGATION_DB_NAME
        self._ensure_schema()

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path))
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            conn.execute("PRAGMA synchronous=NORMAL")
        except BaseException:
            # Setup failed: close it now, or the file stays open (Windows
            # then cannot delete or replace it) for as long as the error lives.
            conn.close()
            raise
        return conn

    def _ensure_schema(self) -> None:
        try:
            with self._connect() as conn:
                for stmt in _DDL.strip().split(";"):
                    stmt = stmt.strip()
                    if stmt:
                        conn.execute(stmt)
                conn.commit()
        except Exception as exc:  # noqa: BLE001
            logger.warning("BackupObligationStore: schema init failed: %s", exc)

    # ------------------------------------------------------------------
    # Write path
    # ------------------------------------------------------------------

    def record(
        self,
        profile_id: str,
        erasure_id: str,
        snapshot_path: str,
        snapshot_epoch: int,
        retention_days: int = 90,
    ) -> str:
        """Record one outstanding obligation.  Returns the obligation_id."""
        oid = uuid.uuid4().hex
        try:
            with self._connect() as conn:
                conn.execute(
                    "INSERT OR IGNORE INTO backup_obligations "
                    "(obligation_id, profile_id, erasure_id, snapshot_path, "
                    " snapshot_epoch, retention_days, recorded_at, status) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, 'pending')",
                    (
                        oid, profile_id, erasure_id, str(snapshot_path),
                        int(snapshot_epoch), int(retention_days), time.time(),
                    ),
                )
                conn.commit()
        except Exception as exc:  # noqa: BLE001
            logger.warning("BackupObligationStore.record failed: %s", exc)
        return oid

    def discharge(self, obligation_id: str, reason: str) -> None:
        """Mark a single obligation as discharged."""
        try:
            with self._connect() as conn:
                conn.execute(
                    "UPDATE backup_obligations "
                    "SET status='discharged', discharged_at=?, discharge_reason=? "
                    "WHERE obligation_id=?",
                    (time.time(), reason, obligation_id),
                )
                conn.commit()
        except Exception as exc:  # noqa: BLE001
            logger.warning("BackupObligationStore.discharge failed: %s", exc)

    def discharge_for_snapshot(self, snapshot_path: str, reason: str) -> int:
        """Discharge ALL pending obligations for a snapshot path.  Returns count."""
        try:
            with self._connect() as conn:
                cur = conn.execute(
                    "UPDATE backup_obligations "
                    "SET status='discharged', discharged_at=?, discharge_reason=? "
                    "WHERE snapshot_path=? AND status='pending'",
                    (time.time(), reason, str(snapshot_path)),
                )
                conn.commit()
                return cur.rowcount
        except Exception as exc:  # noqa: BLE001
            logger.warning("BackupObligationStore.discharge_for_snapshot: %s", exc)
            return 0

    # ------------------------------------------------------------------
    # Read path
    # ------------------------------------------------------------------

    def _discharge_aged_out(self) -> int:
        """Auto-discharge obligations whose snapshot has aged past retention."""
        try:
            with self._connect() as conn:
                now = time.time()
                cur = conn.execute(
                    "UPDATE backup_obligations "
                    "SET status='discharged', discharged_at=?, "
                    "    discharge_reason='snapshot_aged_out' "
                    "WHERE status='pending' "
                    "  AND (snapshot_epoch + retention_days * 86400) < ?",
                    (now, now),
                )
                conn.commit()
                return cur.rowcount
        except Exception as exc:  # noqa: BLE001
            logger.warning("BackupObligationStore._discharge_aged_out: %s", exc)
            return 0

    def count_pending(self, profile_id: str) -> int:
        """Count pending obligations for *profile_id* (auto-discharges aged-out first).

        Fail-closed: if the store is unreadable, returns 1 to block completeness.
        """
        self._discharge_aged_out()
        try:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT COUNT(*) FROM backup_obligations "
                    "WHERE profile_id=? AND status='pending'",
                    (profile_id,),
                ).fetchone()
                return int(row[0]) if row else 0
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "BackupObligationStore.count_pending failed: %s — "
                "returning 1 to block completeness claim",
                exc,
            )
            return 1  # fail-closed

    def list_pending_for_snapshot(self, snapshot_path: str) -> list[dict]:
        """Return all pending obligations for *snapshot_path*."""
        try:
            with self._connect() as conn:
                rows = conn.execute(
                    "SELECT * FROM backup_obligations "
                    "WHERE snapshot_path=? AND status='pending'",
                    (str(snapshot_path),),
                ).fetchall()
                return [dict(r) for r in rows]
        except Exception as exc:  # noqa: BLE001
            logger.warning("BackupObligationStore.list_pending_for_snapshot: %s", exc)
            return []

    def list_pending_for_profile(self, profile_id: str) -> list[dict]:
        """Return all pending obligations for *profile_id* (auto-discharges first)."""
        self._discharge_aged_out()
        try:
            with self._connect() as conn:
                rows = conn.execute(
                    "SELECT * FROM backup_obligations "
                    "WHERE profile_id=? AND status='pending'",
                    (profile_id,),
                ).fetchall()
                return [dict(r) for r in rows]
        except Exception as exc:  # noqa: BLE001
            logger.warning("BackupObligationStore.list_pending_for_profile: %s", exc)
            return []


# ---------------------------------------------------------------------------
# Snapshot helpers (used by gdpr.py and backup.py)
# ---------------------------------------------------------------------------


def scan_backup_snapshots_for_profile(
    backup_dir: Path,
    profile_id: str,
) -> list[tuple[str, int]]:
    """Scan *backup_dir* for snapshots containing *profile_id*'s data.

    Returns a list of (snapshot_path_str, epoch) tuples.  Handles both legacy
    per-file backups (``*.db`` files) and new-style BackupCoordinator sets
    (``backup_XXXX/`` directories containing a ``manifest.json``).
    """
    backup_dir = Path(backup_dir)
    results: list[tuple[str, int]] = []

    if not backup_dir.exists():
        return results

    # New-style backup sets: backup_XXXX/ directories with manifest.json.
    # Record obligation at the set-directory level (one obligation per set).
    for candidate in sorted(backup_dir.iterdir()):
        if not candidate.is_dir() or not (candidate / "manifest.json").exists():
            continue
        hit = False
        for db_file in candidate.glob("*.db"):
            if _snapshot_db_contains_profile(db_file, profile_id):
                hit = True
                break
        if hit:
            epoch = int(candidate.stat().st_mtime)
            results.append((str(candidate), epoch))

    # Legacy per-file backups: ``memory-YYYYMMDD-HHMMSS.db``, etc.
    for db_file in sorted(backup_dir.glob("*.db")):
        if db_file.name.startswith("."):
            continue
        if _snapshot_db_contains_profile(db_file, profile_id):
            epoch = int(db_file.stat().st_mtime)
            results.append((str(db_file), epoch))

    return results


def _snapshot_db_contains_profile(db_path: Path, profile_id: str) -> bool:
    """Return True if the SQLite file at *db_path* contains rows for *profile_id*.

    Opens read-only (immutable) to avoid side-effects.  Returns False on any
    error so an unreadable snapshot does not block erasure; caller logs the
    warning separately.
    """
    try:
        uri = f"file:{db_path}?mode=ro"
        conn = sqlite3.connect(uri, uri=True, timeout=5)
        try:
            tables = {
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
            for table in ("profiles", "atomic_facts", "memories", "graph_nodes", "media_items"):
                if table not in tables:
                    continue
                cols = {
                    row[1]
                    for row in conn.execute(f"PRAGMA table_info({table})").fetchall()
                }
                if "profile_id" not in cols:
                    # graph_nodes has no profile_id — flag unconditionally as personal
                    if table == "graph_nodes":
                        row = conn.execute(
                            "SELECT 1 FROM graph_nodes LIMIT 1"
                        ).fetchone()
                        if row is not None:
                            return True
                    continue
                row = conn.execute(
                    f"SELECT 1 FROM {table} WHERE profile_id=? LIMIT 1",
                    (profile_id,),
                ).fetchone()
                if row is not None:
                    return True
        finally:
            conn.close()
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "_snapshot_db_contains_profile: error reading %s: %s — treating as clean",
            db_path.name, exc,
        )
    return False


def _is_media_snapshot(db_path: Path) -> bool:
    """True for a copy of media.db (images and documents)."""
    if not Path(db_path).name.startswith("media"):
        return False
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=5)
        try:
            return conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' "
                                "AND name='media_items'").fetchone() is not None
        finally:
            conn.close()
    except sqlite3.Error:
        return False


def erase_profile_from_snapshot(db_path: Path, profile_id: str) -> dict[str, int]:
    """Delete all *profile_id* rows from a snapshot SQLite file.

    Used during restore-replay.  Raises on any fatal error so the caller can
    refuse to mark the restore complete when personal data cannot be purged.
    """
    deleted: dict[str, int] = {}
    if _is_media_snapshot(db_path):  # vectors and originals need the media store's own scrub
        from superlocalmemory.media.erasure import scrub_snapshot

        return scrub_snapshot(db_path, profile_id)
    # isolation_level=None → autocommit so VACUUM can run outside any transaction.
    conn = sqlite3.connect(str(db_path))
    conn.isolation_level = None
    try:
        conn.execute("PRAGMA foreign_keys=OFF")
        tables = [
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
            if not row[0].startswith("sqlite_")
        ]
        conn.execute("BEGIN")
        for table in tables:
            cols = {
                row[1]
                for row in conn.execute(f"PRAGMA table_info({table})").fetchall()
            }
            if "profile_id" not in cols:
                continue
            cur = conn.execute(
                f"DELETE FROM {table} WHERE profile_id=?", (profile_id,)
            )
            if cur.rowcount:
                deleted[table] = cur.rowcount
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("COMMIT")
        # VACUUM must run in autocommit (no active transaction).
        conn.execute("VACUUM")
    except Exception:
        try:
            conn.execute("ROLLBACK")
        except Exception:  # noqa: BLE001
            pass
        conn.close()
        raise
    else:
        conn.close()
    return deleted


def erase_code_graph_from_snapshot(db_path: Path) -> dict[str, int]:
    """Wipe all tables from a code_graph.db snapshot (no profile_id column).

    The code graph is installation-level personal data (repo paths, symbol
    names).  On erasure or restore-replay the entire graph is cleared.
    """
    deleted: dict[str, int] = {}
    if not db_path.exists():
        return deleted
    conn = sqlite3.connect(str(db_path))
    conn.isolation_level = None  # autocommit so VACUUM runs outside any transaction
    try:
        conn.execute("PRAGMA foreign_keys=OFF")
        tables = [
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name NOT LIKE 'sqlite_%'"
            ).fetchall()
        ]
        conn.execute("BEGIN")
        for table in tables:
            # Skip FTS virtual tables — deleting the base table handles them
            if table.endswith(("_fts", "_fts_data", "_fts_idx", "_fts_content",
                                "_fts_docsize", "_fts_config")):
                continue
            cur = conn.execute(f"DELETE FROM {table}")
            if cur.rowcount:
                deleted[table] = cur.rowcount
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("COMMIT")
        # VACUUM must run in autocommit (no active transaction).
        conn.execute("VACUUM")
    except Exception:
        try:
            conn.execute("ROLLBACK")
        except Exception:  # noqa: BLE001
            pass
        conn.close()
        raise
    else:
        conn.close()
    return deleted
