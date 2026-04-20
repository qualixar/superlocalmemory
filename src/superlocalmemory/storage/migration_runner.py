# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory v3.4.21 — LLD-07 §4

"""Forward-only additive migrations for SLM v3.4.21.

LLD reference: ``.backup/active-brain/lld/LLD-07-schema-migrations-and-security-primitives.md``
Section 4 (Migration Runner).

Contract:
  - ``apply_all(learning_db, memory_db, *, dry_run=False) -> dict`` —
    runs every v3.4.21 migration, idempotent and transactional. Returns
    ``{"applied": [names], "skipped": [names], "failed": [names],
       "details": {name: str}}``.
  - ``status(learning_db, memory_db) -> dict[str, str]`` — returns the
    status of each migration as recorded in the target DB's ``migration_log``
    (``"complete"``, ``"failed"``, ``"in_progress"``, or ``"missing"``).

Hard rules enforced (LLD-07 §7):
  - MIG-HR-01: idempotent — re-applying is a no-op.
  - MIG-HR-02: atomic — each migration wrapped in BEGIN IMMEDIATE / COMMIT
    via the DDL itself (or by the single-statement guarantee).
  - MIG1: ``ddl_sha256`` prevents silent DDL drift.
  - MIG3: a failing migration does NOT prevent the runner from attempting
    the rest, and does NOT raise to the caller — result comes through the
    returned stats dict.
"""

from __future__ import annotations

import hashlib
import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from superlocalmemory.storage.migrations import (
    M001_add_signal_features_columns as _M001,
    M002_model_state_history as _M002,
    M003_migration_log as _M003,
    M004_cross_platform_sync_log as _M004,
    M005_bandit_tables as _M005,
    M006_action_outcomes_reward as _M006,
    M007_pending_outcomes as _M007,
    M009_model_lineage as _M009,
    M010_evolution_config as _M010,
    M011_archive_and_merge as _M011,
    M012_shadow_observations as _M012,
    M013_bi_temporal_columns as _M013,
)

# Map migration name → module (used for the optional ``verify(conn)`` hook
# that lets the runner detect "already applied" state when an idempotent
# retry would otherwise trigger duplicate-column / duplicate-table errors).
_MODULES = {
    _M001.NAME: _M001,
    _M002.NAME: _M002,
    _M003.NAME: _M003,
    _M004.NAME: _M004,
    _M005.NAME: _M005,
    _M006.NAME: _M006,
    _M007.NAME: _M007,
    _M009.NAME: _M009,
    _M010.NAME: _M010,
    _M011.NAME: _M011,
    _M012.NAME: _M012,
    _M013.NAME: _M013,
}

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Migration:
    """Single migration definition."""

    name: str
    db_target: str  # 'learning' or 'memory'
    ddl: str
    dependencies: tuple[str, ...] = field(default_factory=tuple)


# Order matters: M003 creates the log table. The runner handles M003's own
# bootstrap (it can't record itself before it exists).
MIGRATIONS: list[Migration] = [
    Migration(name=_M003.NAME, db_target="learning", ddl=_M003.DDL),
    Migration(name=_M001.NAME, db_target="learning", ddl=_M001.DDL,
              dependencies=(_M003.NAME,)),
    Migration(name=_M002.NAME, db_target="learning", ddl=_M002.DDL,
              dependencies=(_M003.NAME,)),
    Migration(name=_M005.NAME, db_target="learning", ddl=_M005.DDL,
              dependencies=(_M003.NAME,)),
    # M009 extends learning_model_state (created by M002).
    Migration(name=_M009.NAME, db_target="learning", ddl=_M009.DDL,
              dependencies=(_M002.NAME,)),
    # M010 creates evolution_config + evolution_llm_cost_log (learning.db).
    Migration(name=_M010.NAME, db_target="learning", ddl=_M010.DDL,
              dependencies=(_M003.NAME,)),
    # M012 creates shadow_observations (learning.db) — paired NDCG@10
    # observations for ShadowTest persistence across daemon restart.
    Migration(name=_M012.NAME, db_target="learning", ddl=_M012.DDL,
              dependencies=(_M003.NAME,)),
    Migration(name=_M004.NAME, db_target="memory", ddl=_M004.DDL),
    # M007 creates pending_outcomes (memory.db, LLD-00 §1.2).
    Migration(name=_M007.NAME, db_target="memory", ddl=_M007.DDL),
    # M006 + M011 are deliberately NOT here — see DEFERRED_MIGRATIONS below.
]


# Deferred migrations run AFTER ``MemoryEngine.initialize()`` has called
# ``storage.schema.create_all_tables`` to bootstrap runtime tables such as
# ``action_outcomes``. Running them during ``apply_all`` (which fires BEFORE
# engine init on daemon startup) would blow up with "no such table".
#
# ``learning.database.fetch_training_examples`` already checks
# ``_migration_applied("M006_action_outcomes_reward")`` and falls back to the
# position proxy when the column is absent, so a failed deferred apply never
# crashes the trainer — it just keeps the old label path.
DEFERRED_MIGRATIONS: list[Migration] = [
    Migration(name=_M006.NAME, db_target="memory", ddl=_M006.DDL),
    # M011 extends atomic_facts + creates memory_archive / memory_merge_log.
    # atomic_facts is bootstrapped at engine init, so M011 defers alongside M006.
    Migration(name=_M011.NAME, db_target="memory", ddl=_M011.DDL),
    # M013 adds bi-temporal columns (valid_from / valid_until) to
    # atomic_facts. Deferred for the same engine-init-bootstrap reason
    # as M011.
    Migration(name=_M013.NAME, db_target="memory", ddl=_M013.DDL),
]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ddl_hash(ddl: str) -> str:
    return hashlib.sha256(ddl.encode("utf-8")).hexdigest()


def _connect(db_path: Path) -> sqlite3.Connection:
    # isolation_level=None → we manage transactions explicitly via DDL.
    conn = sqlite3.connect(db_path, isolation_level=None)
    conn.execute("PRAGMA foreign_keys = OFF;")
    return conn


def _migration_log_exists(conn: sqlite3.Connection) -> bool:
    row = conn.execute(
        "SELECT name FROM sqlite_master "
        "WHERE type='table' AND name='migration_log'"
    ).fetchone()
    return row is not None


def _ensure_migration_log(conn: sqlite3.Connection) -> None:
    """Bootstrap the migration_log table on a DB if absent.

    Uses the M003 DDL verbatim so the runner treats migration_log identically
    on both learning.db and memory.db.
    """
    conn.executescript(_M003.DDL)


def _get_log_row(conn: sqlite3.Connection, name: str) -> tuple | None:
    return conn.execute(
        "SELECT name, applied_at, ddl_sha256, rows_affected, status "
        "FROM migration_log WHERE name = ?",
        (name,),
    ).fetchone()


def _upsert_log(
    conn: sqlite3.Connection,
    name: str,
    ddl_hash: str,
    status: str,
    rows_affected: int = 0,
) -> None:
    conn.execute(
        "INSERT INTO migration_log "
        "(name, applied_at, ddl_sha256, rows_affected, status) "
        "VALUES (?, ?, ?, ?, ?) "
        "ON CONFLICT(name) DO UPDATE SET "
        "    applied_at = excluded.applied_at, "
        "    ddl_sha256 = excluded.ddl_sha256, "
        "    rows_affected = excluded.rows_affected, "
        "    status = excluded.status",
        (name, _now_iso(), ddl_hash, rows_affected, status),
    )


def _delete_log(conn: sqlite3.Connection, name: str) -> None:
    conn.execute("DELETE FROM migration_log WHERE name = ?", (name,))


def _apply_single(
    conn: sqlite3.Connection,
    migration: Migration,
    *,
    dry_run: bool,
) -> tuple[str, str]:
    """Apply one migration against ``conn``.

    Returns (outcome, detail) where outcome is one of:
      - "applied"
      - "skipped"
      - "failed"
    """
    ddl_hash = _ddl_hash(migration.ddl)

    # Bootstrap: if migration_log doesn't exist yet, this MUST be M003.
    if not _migration_log_exists(conn):
        if migration.name != _M003.NAME:
            # Other migrations can't check state → treat as unrecoverable here.
            return ("failed",
                    f"migration_log missing when attempting {migration.name}")
        if dry_run:
            return ("skipped", "dry-run: would create migration_log")
        try:
            _ensure_migration_log(conn)
            _upsert_log(conn, migration.name, ddl_hash, "complete")
            return ("applied", "bootstrapped migration_log")
        except sqlite3.Error as exc:  # pragma: no cover — defensive
            logger.warning("M003 bootstrap failed: %s", exc)
            return ("failed", f"bootstrap error: {exc}")

    # M003 specifically — if log already exists, ensure M003's own row is there
    # (records the fact that the table was bootstrapped previously).
    existing = _get_log_row(conn, migration.name)

    if existing is not None:
        _, _, logged_hash, _, status = existing
        if status == "complete":
            if logged_hash != ddl_hash:
                detail = (
                    f"DDL drift detected for {migration.name}: "
                    f"logged={logged_hash[:8]}... current={ddl_hash[:8]}..."
                )
                logger.warning(detail)
                return ("failed", detail)
            return ("skipped", "already complete")
        # status is 'failed' or 'in_progress' → retry from scratch.
        if dry_run:
            return ("skipped", f"dry-run: would retry (status={status})")
        try:
            _delete_log(conn, migration.name)
        except sqlite3.Error as exc:  # pragma: no cover — log table exists
            return ("failed", f"cannot clear prior log: {exc}")

    if dry_run:
        return ("skipped", "dry-run: would apply")

    # Mark in_progress, execute, update status. If DDL fails we roll our log
    # entry to 'failed' so next attempt will retry cleanly.
    try:
        _upsert_log(conn, migration.name, ddl_hash, "in_progress")
    except sqlite3.Error as exc:  # pragma: no cover
        return ("failed", f"cannot record in_progress: {exc}")

    try:
        conn.executescript(migration.ddl)
    except sqlite3.Error as exc:
        # Best-effort rollback.
        try:
            conn.execute("ROLLBACK")
        except sqlite3.Error:  # pragma: no cover — best-effort
            pass
        # Before marking failed, check if the migration's end-state is
        # already in place (e.g. crash-recovery retry against a DB where the
        # columns were added in a previous partial apply). If so, this is
        # effectively a successful idempotent re-run.
        mod = _MODULES.get(migration.name)
        verify_fn = getattr(mod, "verify", None) if mod is not None else None
        if verify_fn is not None:
            try:
                if verify_fn(conn):
                    try:
                        _upsert_log(conn, migration.name, ddl_hash, "complete")
                    except sqlite3.Error:  # pragma: no cover
                        pass
                    return ("applied",
                            "already applied (verified via schema inspection)")
            except sqlite3.Error:  # pragma: no cover
                pass

        logger.warning("Migration %s failed: %s", migration.name, exc)
        try:
            _upsert_log(conn, migration.name, ddl_hash, "failed")
        except sqlite3.Error:  # pragma: no cover
            pass
        return ("failed", f"{type(exc).__name__}: {exc}")

    # S9-W1 H-DATA-01: optional post-DDL Python hook. Runs inside the same
    # connection (same DB file) after the DDL commits. Used by M002 to
    # backfill ``bytes_sha256`` on rows copied forward by the new-table
    # rename. If the hook raises, the migration is marked failed; the DDL
    # is NOT rolled back (already committed) but the runner reports the
    # problem so operators can intervene. Non-existent hooks are a no-op.
    mod = _MODULES.get(migration.name)
    post_hook = getattr(mod, "post_ddl_hook", None) if mod is not None else None
    if post_hook is not None:
        try:
            post_hook(conn)
        except Exception as exc:  # noqa: BLE001 — report + mark failed
            logger.warning(
                "Migration %s DDL applied but post_ddl_hook failed: %s",
                migration.name, exc,
            )
            try:
                _upsert_log(conn, migration.name, ddl_hash, "failed")
            except sqlite3.Error:  # pragma: no cover
                pass
            return ("failed", f"post_ddl_hook: {type(exc).__name__}: {exc}")

    try:
        _upsert_log(conn, migration.name, ddl_hash, "complete")
    except sqlite3.Error as exc:  # pragma: no cover
        return ("failed", f"cannot record complete: {exc}")
    return ("applied", "ok")


def _db_for(target: str, learning_db: Path, memory_db: Path) -> Path:
    if target == "learning":
        return learning_db
    if target == "memory":
        return memory_db
    raise ValueError(f"unknown db_target: {target}")  # pragma: no cover


def _bootstrap_both_migration_logs(
    learning_db: Path, memory_db: Path, *, dry_run: bool,
) -> tuple[list[str], dict[str, str]]:
    """S9-W1 C3: bootstrap ``migration_log`` on BOTH DBs up-front.

    Prior versions deferred memory-side bootstrap until the first memory
    migration ran in ``apply_all``, and ``apply_deferred`` did its own
    independent bootstrap. That created a split-brain failure mode: if
    ``apply_all`` crashed before any memory migration ran (e.g. disk-full
    on learning-side M005), the memory DB never got its log table, and
    ``apply_deferred`` would later create one without any record of the
    sync-set attempt. Memory DB is sacred — 18k+ atomic_facts.

    By bootstrapping both DBs up-front here, we make the invariant
    "migration_log exists on both DBs before any migration runs" hold
    unconditionally. Returns (failed_names, details) for any DB where
    bootstrap fails.
    """
    failed: list[str] = []
    details: dict[str, str] = {}
    if dry_run:
        return failed, details
    for label, db_path in (("learning_db", learning_db),
                           ("memory_db", memory_db)):
        try:
            conn = _connect(db_path)
        except sqlite3.Error as exc:  # pragma: no cover — defensive
            failed.append(label)
            details[label] = f"cannot open db for log bootstrap: {exc}"
            continue
        try:
            if not _migration_log_exists(conn):
                _ensure_migration_log(conn)
        except sqlite3.Error as exc:  # pragma: no cover — defensive
            failed.append(label)
            details[label] = f"migration_log bootstrap failed: {exc}"
        finally:
            try:
                conn.close()
            except sqlite3.Error:  # pragma: no cover
                pass
    return failed, details


def apply_all(
    learning_db: Path,
    memory_db: Path,
    *,
    dry_run: bool = False,
) -> dict:
    """Apply all v3.4.21 migrations; return stats.

    Idempotent: already-applied migrations are skipped. Non-fatal: any
    migration that fails is recorded in ``failed`` and the runner moves on.
    """
    applied: list[str] = []
    skipped: list[str] = []
    failed: list[str] = []
    details: dict[str, str] = {}

    # S9-W1 C3: unify the migration_log bootstrap across both DBs up-front.
    bs_failed, bs_details = _bootstrap_both_migration_logs(
        learning_db, memory_db, dry_run=dry_run,
    )
    failed.extend(bs_failed)
    details.update(bs_details)

    for migration in MIGRATIONS:
        db_path = _db_for(migration.db_target, learning_db, memory_db)
        try:
            conn = _connect(db_path)
        except sqlite3.Error as exc:  # pragma: no cover — defensive
            failed.append(migration.name)
            details[migration.name] = f"cannot open db: {exc}"
            continue

        try:
            outcome, detail = _apply_single(conn, migration, dry_run=dry_run)
            details[migration.name] = detail
            if outcome == "applied":
                applied.append(migration.name)
            elif outcome == "skipped":
                skipped.append(migration.name)
            else:
                failed.append(migration.name)
        finally:
            try:
                conn.close()
            except sqlite3.Error:  # pragma: no cover
                pass

    return {
        "applied": applied,
        "skipped": skipped,
        "failed": failed,
        "details": details,
    }


def apply_deferred(
    learning_db: Path,
    memory_db: Path,
    *,
    dry_run: bool = False,
) -> dict:
    """Apply deferred migrations; return the same stats shape as apply_all.

    Deferred migrations target runtime-bootstrapped tables (e.g.
    ``action_outcomes``) that don't exist until ``MemoryEngine.initialize()``
    has run ``storage.schema.create_all_tables``. The daemon lifespan calls
    this immediately after engine init.

    Same idempotency + non-fatal guarantees as ``apply_all``. If the target
    table is still missing, the underlying DDL raises ``no such table`` and
    the migration is recorded as ``failed`` — safe, the trainer already
    falls back to the position proxy when M006 hasn't completed.
    """
    applied: list[str] = []
    skipped: list[str] = []
    failed: list[str] = []
    details: dict[str, str] = {}

    for migration in DEFERRED_MIGRATIONS:
        db_path = _db_for(migration.db_target, learning_db, memory_db)
        try:
            conn = _connect(db_path)
        except sqlite3.Error as exc:  # pragma: no cover — defensive
            failed.append(migration.name)
            details[migration.name] = f"cannot open db: {exc}"
            continue

        try:
            # S9-W1 C3: apply_deferred must NOT independently bootstrap
            # migration_log. apply_all is the single source of truth for
            # log-table creation (bootstraps BOTH DBs up-front). A missing
            # log here means apply_all never ran or crashed catastrophically
            # before touching this DB — fail loudly so the operator can
            # run apply_all first instead of letting the deferred path
            # silently create a table that records nothing of the sync set.
            if not _migration_log_exists(conn):
                failed.append(migration.name)
                details[migration.name] = (
                    "migration_log missing on target DB — apply_all must "
                    "run first (or failed before reaching this DB); "
                    "refusing to create split-brain log"
                )
                continue

            outcome, detail = _apply_single(conn, migration, dry_run=dry_run)
            details[migration.name] = detail
            if outcome == "applied":
                applied.append(migration.name)
            elif outcome == "skipped":
                skipped.append(migration.name)
            else:
                failed.append(migration.name)
        finally:
            try:
                conn.close()
            except sqlite3.Error:  # pragma: no cover
                pass

    return {
        "applied": applied,
        "skipped": skipped,
        "failed": failed,
        "details": details,
    }


def status(learning_db: Path, memory_db: Path) -> dict[str, str]:
    """Return the per-migration status as recorded in the target DB.

    Values: ``"complete"``, ``"failed"``, ``"in_progress"``, or ``"missing"``.
    Includes both ``MIGRATIONS`` and ``DEFERRED_MIGRATIONS``.
    """
    out: dict[str, str] = {}
    # Read-only — if the DB doesn't have migration_log, every migration is
    # reported as "missing".
    cached: dict[str, dict[str, str]] = {}
    for migration in (*MIGRATIONS, *DEFERRED_MIGRATIONS):
        db_path = _db_for(migration.db_target, learning_db, memory_db)
        db_key = str(db_path)
        if db_key not in cached:
            cached[db_key] = _read_log(db_path)
        out[migration.name] = cached[db_key].get(migration.name, "missing")
    return out


def _read_log(db_path: Path) -> dict[str, str]:
    try:
        conn = sqlite3.connect(db_path)
    except sqlite3.Error:  # pragma: no cover
        return {}
    try:
        if not _migration_log_exists(conn):
            return {}
        rows = conn.execute(
            "SELECT name, status FROM migration_log"
        ).fetchall()
        return {name: status for (name, status) in rows}
    except sqlite3.Error:  # pragma: no cover
        return {}
    finally:
        conn.close()


__all__ = (
    "Migration",
    "MIGRATIONS",
    "DEFERRED_MIGRATIONS",
    "apply_all",
    "apply_deferred",
    "status",
)
