# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory v3.4.22 — LLD-07 §4

"""Internal apply-engine for the forward-only migration runner.

This module holds the private machinery that ``migration_runner`` builds its
public ``apply_all`` / ``apply_deferred`` / ``status`` API on top of:

  - ``Migration`` — the single-migration record type.
  - ``_MODULES`` / ``_KNOWN_EQUIVALENT_DDL_HASHES`` — the name→module registry
    and the allowlist of historically-benign DDL fingerprints.
  - The ``sqlite3`` connection / ``migration_log`` primitives.
  - ``_apply_single`` — the transactional apply-one-migration engine shared by
    both the eager (``apply_all``) and deferred (``apply_deferred``) passes.

Nothing here is part of the public surface; ``migration_runner`` re-imports the
symbols it needs. It carries no dependency on ``migration_runner`` itself, so
importing it never risks a cycle. The catalogue (the ordered ``MIGRATIONS`` /
``DEFERRED_MIGRATIONS`` lists) and the public orchestration functions remain in
``migration_runner``.
"""

from __future__ import annotations

import hashlib
import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from superlocalmemory.storage.migrations import (
    M001_add_signal_features_columns as _M001,
)
from superlocalmemory.storage.migrations import (
    M002_model_state_history as _M002,
)
from superlocalmemory.storage.migrations import (
    M003_migration_log as _M003,
)
from superlocalmemory.storage.migrations import (
    M004_cross_platform_sync_log as _M004,
)
from superlocalmemory.storage.migrations import (
    M005_bandit_tables as _M005,
)
from superlocalmemory.storage.migrations import (
    M006_action_outcomes_reward as _M006,
)
from superlocalmemory.storage.migrations import (
    M007_pending_outcomes as _M007,
)
from superlocalmemory.storage.migrations import (
    M009_model_lineage as _M009,
)
from superlocalmemory.storage.migrations import (
    M010_evolution_config as _M010,
)
from superlocalmemory.storage.migrations import (
    M011_archive_and_merge as _M011,
)
from superlocalmemory.storage.migrations import (
    M012_shadow_observations as _M012,
)
from superlocalmemory.storage.migrations import (
    M013_bi_temporal_columns as _M013,
)
from superlocalmemory.storage.migrations import (
    M014_v345_scale_ready as _M014,
)
from superlocalmemory.storage.migrations import (
    M015_add_pinned_column as _M015,
)
from superlocalmemory.storage.migrations import (
    M016_add_scope_support as _M016,
)
from superlocalmemory.storage.migrations import (
    M017_ccq_scope_column as _M017,
)
from superlocalmemory.storage.migrations import (
    M018_ingestion_operations as _M018,
)
from superlocalmemory.storage.migrations import (
    M019_derivation_lineage as _M019,
)
from superlocalmemory.storage.migrations import (
    M020_model_state_integrity as _M020,
)
from superlocalmemory.storage.migrations import (
    M021_ingestion_log_profile as _M021,
)
from superlocalmemory.storage.migrations import (
    M022_entity_aliases_profile as _M022,
)
from superlocalmemory.storage.migrations import (
    M023_mesh_profile_isolation as _M023,
)
from superlocalmemory.storage.migrations import (
    M024_rbac_users_roles as _M024,
)
from superlocalmemory.storage.migrations import (
    M025_perf_indexes as _M025,
)
from superlocalmemory.storage.migrations import (
    M026_rbac_memberships_fk as _M026,
)
from superlocalmemory.storage.migrations import (
    M027_transferable_patterns_profile as _M027,
)
from superlocalmemory.storage.migrations import (
    M028_fact_entity_associations as _M028,
)
from superlocalmemory.storage.migrations import (
    M029_behavioral_history_indexes as _M029,
)
from superlocalmemory.storage.migrations import (
    M030_entity_explorer_indexes as _M030,
)
from superlocalmemory.storage.migrations import (
    M031_dead_letter_operations as _M031,
)
from superlocalmemory.storage.migrations import (
    M032_write_coordinator_admission as _M032,
)
from superlocalmemory.storage.migrations import (
    M033_projection_transactions as _M033,
)
from superlocalmemory.storage.migrations import (
    M034_obligation_integrity as _M034,
)
from superlocalmemory.storage.migrations import (
    M035_erasure_receipts as _M035,
)
from superlocalmemory.storage.migrations import (
    M036_vector_row_map as _M036,
)

# Emit under the runner's logger name so operational log filters that key on
# "superlocalmemory.storage.migration_runner" keep matching after this split.
logger = logging.getLogger("superlocalmemory.storage.migration_runner")

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
    _M014.NAME: _M014,
    _M015.NAME: _M015,
    _M016.NAME: _M016,
    _M017.NAME: _M017,
    _M018.NAME: _M018,
    _M019.NAME: _M019,
    _M020.NAME: _M020,
    _M021.NAME: _M021,
    _M022.NAME: _M022,
    _M023.NAME: _M023,
    _M024.NAME: _M024,
    _M025.NAME: _M025,
    _M026.NAME: _M026,
    _M027.NAME: _M027,
    _M028.NAME: _M028,
    _M029.NAME: _M029,
    _M030.NAME: _M030,
    _M031.NAME: _M031,
    _M032.NAME: _M032,
    _M033.NAME: _M033,
    _M034.NAME: _M034,
    _M035.NAME: _M035,
    _M036.NAME: _M036,
}

# Exact historical DDL fingerprints whose resulting schema is intentionally
# accepted by the current migration. Unknown hashes are never reconciled.
_KNOWN_EQUIVALENT_DDL_HASHES: dict[str, frozenset[str]] = {
    _M002.NAME: frozenset({
        # v3.4.21 hardened copy-forward variant.
        "347eeb2ec8aac89f7cbf373da49ac9446be9ed150e6105c382c656cd22426d4b",
        # v3.4.22 model_version-default variant shipped through 3.6.x.
        "d28666fa1dfa66e6514efd288e6748363513da2255a4cee95d80f233e6728ae7",
    }),
    _M032.NAME: frozenset({
        # Provisional 3.8.6 development ledger: global idempotency_key and
        # operation_id uniqueness. Its standalone table is safely rebuilt by
        # M032.repair() into the profile-scoped receipt contract.
        "e45df41becba3d0c3342eca5ec3bd83aa899eef76943c819d2da73b4ca1625a7",
    }),
}


@dataclass(frozen=True, slots=True)
class Migration:
    """Single migration definition."""

    name: str
    db_target: str  # 'learning' or 'memory'
    ddl: str
    dependencies: tuple[str, ...] = field(default_factory=tuple)


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
                # v3.7.6 (#70): a complete migration whose logged DDL hash no
                # longer matches the current text is only a real failure if the
                # schema it guarantees is actually absent. Historically-benign
                # DDL edits (e.g. M002's V3.4.21 <-> S9-W1 variants that build the
                # identical end-state) would otherwise brick readiness forever on
                # upgrade. Consult the migration's own verify(); if the schema is
                # in place, reconcile the log to the current hash and treat as
                # already-applied instead of failing the daemon into permanent
                # not_ready. Absent/failing verify keeps the hard failure.
                allowed_hashes = _KNOWN_EQUIVALENT_DDL_HASHES.get(
                    migration.name, frozenset(),
                )
                mod = _MODULES.get(migration.name)
                verify_fn = (
                    getattr(mod, "verify", None) if mod is not None else None
                )
                if logged_hash in allowed_hashes and verify_fn is not None:
                    try:
                        if verify_fn(conn):
                            if not dry_run:
                                try:
                                    _upsert_log(
                                        conn, migration.name, ddl_hash, "complete"
                                    )
                                except sqlite3.Error:  # pragma: no cover
                                    pass
                            return (
                                "skipped",
                                "allowlisted historical DDL reconciled after "
                                "full schema verification",
                            )
                    except sqlite3.Error:  # pragma: no cover
                        pass
                    if dry_run:
                        return (
                            "skipped",
                            "dry-run: would repair allowlisted historical schema",
                        )
                    repair_fn = getattr(mod, "repair", None) if mod is not None else None
                    if callable(repair_fn):
                        try:
                            repair_fn(conn)
                            if not bool(verify_fn(conn)):
                                return (
                                    "failed",
                                    f"safe repair did not restore {migration.name}",
                                )
                            _upsert_log(conn, migration.name, ddl_hash, "complete")
                            return (
                                "applied",
                                "allowlisted historical schema repaired safely",
                            )
                        except sqlite3.Error as exc:
                            return (
                                "failed",
                                f"safe repair failed for {migration.name}: {exc}",
                            )
                detail = (
                    f"DDL drift detected for {migration.name}: "
                    f"logged={logged_hash[:8]}... current={ddl_hash[:8]}..."
                )
                logger.warning(detail)
                return ("failed", detail)
            # A matching migration-log row is not proof that the promised
            # schema still exists. Existing installs can retain a migration log
            # while a partial restore drops an additive table or index.
            #
            # Never replay a historical migration merely because verify()
            # fails. Some migrations rebuild tables and transform data; replay
            # would be destructive (M002 is the canonical example). Only a
            # module-supplied repair(conn) hook is allowed to reconcile a
            # completed migration's end-state.
            mod = _MODULES.get(migration.name)
            verify_fn = (
                getattr(mod, "verify", None) if mod is not None else None
            )
            if verify_fn is None:
                return ("skipped", "already complete")
            try:
                schema_complete = bool(verify_fn(conn))
            except sqlite3.Error as exc:
                return (
                    "failed",
                    f"schema verification failed for {migration.name}: {exc}",
                )
            if schema_complete:
                return ("skipped", "already complete (schema verified)")
            if dry_run:
                return (
                    "skipped",
                    "dry-run: would repair missing migration end-state",
                )
            repair_fn = (
                getattr(mod, "repair", None) if mod is not None else None
            )
            if not callable(repair_fn):
                detail = (
                    f"schema incomplete for completed migration "
                    f"{migration.name}; automatic replay is disabled"
                )
                logger.warning(detail)
                return ("failed", detail)
            try:
                repair_fn(conn)
            except sqlite3.Error as exc:
                return (
                    "failed",
                    f"safe repair failed for {migration.name}: {exc}",
                )
            try:
                if not bool(verify_fn(conn)):
                    return (
                        "failed",
                        f"safe repair did not restore {migration.name}",
                    )
            except sqlite3.Error as exc:
                return (
                    "failed",
                    f"post-repair verification failed for "
                    f"{migration.name}: {exc}",
                )
            return ("applied", "missing end-state repaired safely")
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
        # A migration module may ship a custom apply(conn) for conditional logic
        # that static DDL can't express (e.g. SQLite has no ADD COLUMN IF NOT
        # EXISTS, and ALTER on a missing/already-altered table can't be guarded
        # in one executescript). If present, it runs instead of the DDL string;
        # otherwise the DDL is applied as before. Pure-DDL migrations are
        # unaffected.
        _mod = _MODULES.get(migration.name)
        _apply_fn = getattr(_mod, "apply", None) if _mod is not None else None
        if callable(_apply_fn):
            _apply_fn(conn)
        else:
            # Atomicity is opt-in per migration: a migration that must be all-or-
            # nothing wraps its own BEGIN/COMMIT (or ships a custom apply()); a
            # bare DDL script is deliberately best-effort so a non-essential
            # trailing statement (e.g. a perf index over a table that may not
            # exist yet) can fail without discarding the essential leading DDL.
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
