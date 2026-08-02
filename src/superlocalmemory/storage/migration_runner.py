# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory v3.4.22 — LLD-07 §4

"""Forward-only additive migrations for SLM v3.4.22.

LLD reference: ``.backup/active-brain/lld/LLD-07-schema-migrations-and-security-primitives.md``
Section 4 (Migration Runner).

Contract:
  - ``apply_all(learning_db, memory_db, *, dry_run=False) -> dict`` —
    runs every v3.4.22 migration, idempotent and transactional. Returns
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

The private apply-engine (``Migration``, the ``migration_log`` primitives,
``_apply_single``, and the name→module registry) lives in
``superlocalmemory.storage._migration_internals``; this module owns the ordered
catalogue and the public orchestration functions.
"""

from __future__ import annotations

import logging
import sqlite3
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
from superlocalmemory.storage._schema_version import (
    SUPPORTED_SCHEMA_VERSION,
    SchemaVersionError,
    check_version_or_raise as _check_version_or_raise,
    ensure_schema_version_table as _ensure_schema_version_table,
    write_schema_version as _write_schema_version,
)
from superlocalmemory.storage._migration_internals import (
    Migration,
    _MODULES,  # noqa: F401 — re-exported for test/introspection compatibility
    _apply_single,
    _connect,
    _db_for,
    _ensure_migration_log,
    _migration_log_exists,
    _read_log,
)

logger = logging.getLogger(__name__)


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
    # M020 owns post-release integrity repair. M002 remains byte-for-byte
    # compatible with databases that recorded its historical DDL hash.
    Migration(name=_M020.NAME, db_target="learning", ddl=_M020.DDL,
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
    # M018 is additive and independent of runtime-bootstrapped tables.
    Migration(name=_M018.NAME, db_target="memory", ddl=_M018.DDL),
    # M024 creates RBAC tables (users / memberships / sessions). Independent
    # brand-new tables, so it runs pre-engine-init.
    Migration(name=_M024.NAME, db_target="memory", ddl=_M024.DDL),
    Migration(name=_M019.NAME, db_target="memory", ddl=_M019.DDL,
              dependencies=(_M018.NAME,)),
    # M031 creates dead_letter_operations — standalone table, no FK to engine-
    # bootstrapped tables, so it can run during apply_all (before engine init).
    Migration(name=_M031.NAME, db_target="memory", ddl=_M031.DDL),
    # M032 is standalone and must precede daemon readiness: typed writes use
    # this append-only receipt ledger for durable idempotency.
    Migration(name=_M032.NAME, db_target="memory", ddl=_M032.DDL),
    Migration(name=_M033.NAME, db_target="memory", ddl=_M033.DDL),
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
    # M028 captures an atomic_facts rowid high-water mark before readiness.
    # atomic_facts/canonical_entities are bootstrapped by MemoryEngine.
    Migration(name=_M028.NAME, db_target="memory", ddl=_M028.DDL,
              dependencies=(_M018.NAME,)),
    Migration(name=_M006.NAME, db_target="memory", ddl=_M006.DDL),
    # M011 extends atomic_facts + creates memory_archive / memory_merge_log.
    # atomic_facts is bootstrapped at engine init, so M011 defers alongside M006.
    Migration(name=_M011.NAME, db_target="memory", ddl=_M011.DDL),
    # M013 adds bi-temporal columns (valid_from / valid_until) to
    # atomic_facts. Deferred for the same engine-init-bootstrap reason
    # as M011.
    Migration(name=_M013.NAME, db_target="memory", ddl=_M013.DDL),
    Migration(name=_M014.NAME, db_target="memory", ddl=_M014.DDL),
    # M015 adds pinned column to atomic_facts (v3.4.65 core-memory pins).
    Migration(name=_M015.NAME, db_target="memory", ddl=_M015.DDL),
    # M016 adds scope and shared_with columns to 5 core tables for
    # multi-scope memory support (personal/global/shared).
    Migration(name=_M016.NAME, db_target="memory", ddl=_M016.DDL),
    # M017 adds scope to the engine-bootstrapped CCQ consolidation table.
    Migration(name=_M017.NAME, db_target="memory", ddl=_M017.DDL),
    # M021 rebuilds ingestion_log with a profile-scoped dedup constraint.
    # Deferred: ingestion_log is created at engine init (apply_v343_schema).
    Migration(name=_M021.NAME, db_target="memory", ddl=_M021.DDL),
    # M022 adds profile_id to entity_aliases, backfilled from the parent entity.
    # Deferred: entity_aliases is created at engine init (create_all_tables).
    Migration(name=_M022.NAME, db_target="memory", ddl=_M022.DDL),
    # M023 profile-scopes every mesh coordination table (peers/messages/state/
    # locks/events). Deferred: mesh tables are created at engine init
    # (apply_v343_schema), same as M021.
    Migration(name=_M023.NAME, db_target="memory", ddl=_M023.DDL),
    # M025 adds hot-path perf indexes (atomic_facts dedup, mesh cleanup/list).
    Migration(name=_M025.NAME, db_target="memory", ddl=_M025.DDL),
    # M026 rebuilds rbac_memberships with a profiles FK (ON DELETE CASCADE) so
    # deleting a profile cascade-purges its role grants (SEC-H-01 defense-in-
    # depth). Deferred: the FK target `profiles` is created at engine init.
    Migration(name=_M026.NAME, db_target="memory", ddl=_M026.DDL),
    # M027 rebuilds transferable_patterns with profile_id + UNIQUE(profile_id,
    # pattern_type, key) to prevent cross-profile preference contamination (H-01,
    # cycle-3 audit). Deferred: CrossProjectAggregator creates the table on first
    # consolidation run, not during engine init or apply_all. apply() is a no-op
    # when the table is absent (first install after the schema change).
    Migration(name=_M027.NAME, db_target="learning", ddl=_M027.DDL),
    # M029 indexes behavioral tables bootstrapped during engine initialization.
    Migration(name=_M029.NAME, db_target="memory", ddl=_M029.DDL),
    # M030 bounds Entity Explorer pagination and profile-summary ranking.
    Migration(name=_M030.NAME, db_target="memory", ddl=_M030.DDL),
]


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


def _bootstrap_learning_schema(learning_db: Path, *, dry_run: bool) -> str | None:
    """Create the base learning tables before forward migrations extend them.

    ``apply_all`` is called by the daemon before ``MemoryEngine`` exists.  A
    blank first-install therefore has no ``learning_signals`` or
    ``learning_model_state`` tables for M001/M002/M009 to alter.  The runner
    owns this prerequisite so every caller has the same first-boot contract.
    """
    if dry_run:
        return None
    try:
        from superlocalmemory.learning.database import LearningDatabase

        LearningDatabase(learning_db)
    except Exception as exc:  # noqa: BLE001 - retain runner's non-fatal API
        return f"learning schema bootstrap failed: {type(exc).__name__}: {exc}"
    return None


def apply_all(
    learning_db: Path,
    memory_db: Path,
    *,
    dry_run: bool = False,
) -> dict:
    """Apply all v3.4.22 migrations; return stats.

    Idempotent: already-applied migrations are skipped. Non-fatal: any
    migration that fails is recorded in ``failed`` and the runner moves on.

    Raises SchemaVersionError before touching any data when the learning DB
    reports a schema_version that exceeds SUPPORTED_SCHEMA_VERSION.  This
    prevents silent data corruption when downgrading to an older build.
    """
    # Non-mutating version check: must run before any write. Both managed
    # databases are validated so a downgrade is detectable regardless of which
    # store carries the newer stamp.
    _check_version_or_raise(learning_db)
    _check_version_or_raise(memory_db)

    applied: list[str] = []
    skipped: list[str] = []
    failed: list[str] = []
    details: dict[str, str] = {}

    schema_error = _bootstrap_learning_schema(learning_db, dry_run=dry_run)
    if schema_error is not None:
        failed.append("learning_schema_bootstrap")
        details["learning_schema_bootstrap"] = schema_error
        return {
            "applied": applied,
            "skipped": skipped,
            "failed": failed,
            "details": details,
        }

    # S9-W1 C3: unify the migration_log bootstrap across both DBs up-front.
    bs_failed, bs_details = _bootstrap_both_migration_logs(
        learning_db, memory_db, dry_run=dry_run,
    )
    failed.extend(bs_failed)
    details.update(bs_details)

    blocked: set[str] = set()
    for migration in MIGRATIONS:
        # A migration whose declared dependency did not complete must not run
        # against a base schema that is missing that dependency's changes.
        unmet = [d for d in migration.dependencies if d in failed or d in blocked]
        if unmet:
            skipped.append(migration.name)
            blocked.add(migration.name)
            details[migration.name] = "dependency not satisfied: " + ", ".join(unmet)
            continue

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

    # Monotonic schema_version stamp: advance only on a zero-failure run so
    # the stored version accurately reflects what was fully applied.
    if not failed and not dry_run:
        for _stamp_db in (learning_db, memory_db):
            try:
                _stamp_conn = _connect(_stamp_db)
                try:
                    _ensure_schema_version_table(_stamp_conn)
                    _write_schema_version(_stamp_conn, SUPPORTED_SCHEMA_VERSION)
                finally:
                    try:
                        _stamp_conn.close()
                    except sqlite3.Error:  # pragma: no cover
                        pass
            except sqlite3.Error as exc:  # pragma: no cover — best-effort stamp
                logger.warning(
                    "schema_version stamp failed for %s: %s", _stamp_db, exc
                )

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

    Raises SchemaVersionError before touching any data when either managed
    database reports a schema_version newer than SUPPORTED_SCHEMA_VERSION. This
    path runs after engine init, so it must fail closed on a downgrade exactly
    as ``apply_all`` does rather than write DDL an older build cannot interpret.
    """
    # Non-mutating downgrade guard: must run before any write, mirroring
    # apply_all. Both managed databases are validated so a newer stamp on
    # either store halts the deferred pass.
    _check_version_or_raise(learning_db)
    _check_version_or_raise(memory_db)

    applied: list[str] = []
    skipped: list[str] = []
    failed: list[str] = []
    details: dict[str, str] = {}

    blocked: set[str] = set()
    for migration in DEFERRED_MIGRATIONS:
        unmet = [d for d in migration.dependencies if d in failed or d in blocked]
        if unmet:
            skipped.append(migration.name)
            blocked.add(migration.name)
            details[migration.name] = "dependency not satisfied: " + ", ".join(unmet)
            continue

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


__all__ = (
    "Migration",
    "MIGRATIONS",
    "DEFERRED_MIGRATIONS",
    "SUPPORTED_SCHEMA_VERSION",
    "SchemaVersionError",
    "apply_all",
    "apply_deferred",
    "status",
)
