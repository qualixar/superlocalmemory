# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory v3.4.22 — LLD-07 §4

"""Forward-only additive migrations for SLM v3.4.22.


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
import os
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
from superlocalmemory.storage.migrations import (
    M034_obligation_integrity as _M034,
)
from superlocalmemory.storage.migrations import (
    M035_erasure_receipts as _M035,
)
from superlocalmemory.storage.migrations import (
    M036_vector_row_map as _M036,
)
from superlocalmemory.storage.migrations import (
    M037_manifest_hmac_version as _M037,
)
from superlocalmemory.storage.migrations import (
    M038_learning_feedback_channel as _M038,
)
from superlocalmemory.storage.migrations import (
    M039_scene_fact_members as _M039,
)
from superlocalmemory.storage.migrations import (
    M040_agent_experience_receipts as _M040,
)
from superlocalmemory.storage.migrations import (
    M041_external_evidence_receipts as _M041,
)
from superlocalmemory.storage.migrations import (
    M042_correction_case_ledger as _M042,
)
from superlocalmemory.storage.migrations import (
    M044_play_carries_its_own_evidence as _M044,
)
from superlocalmemory.storage.migrations import (
    M045_fact_outcome_score as _M045,
    M046_prospective_memory_has_its_own_name as _M046,
    M047_fisher_vectors_are_stored_like_every_other_vector as _M047,
    M048_upcoming_holds_only_what_is_upcoming as _M048,
    M049_a_schema_version_marker_is_one_row as _M049,
    M050_execution_learning_v2 as _M050,
)
from superlocalmemory.storage.migrations import (
    M043_quarantine_display_summaries as _M043,
)
from superlocalmemory.storage._schema_version import (
    SUPPORTED_SCHEMA_VERSION,
    SchemaVersionError,
    check_version_or_raise as _check_version_or_raise,
    ensure_schema_version_table as _ensure_schema_version_table,
    read_schema_version as _read_schema_version,
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
from superlocalmemory.storage.backup import (
    _gc_old_backups,
    _pre_migration_backup,
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
    Migration(name=_M034.NAME, db_target="memory", ddl=_M034.DDL,
              dependencies=(_M033.NAME,)),
    Migration(name=_M035.NAME, db_target="memory", ddl=_M035.DDL,
              dependencies=(_M033.NAME,)),
    Migration(name=_M036.NAME, db_target="memory", ddl=_M036.DDL,
              dependencies=(_M033.NAME,)),
    Migration(name=_M037.NAME, db_target="memory", ddl=_M037.DDL,
              dependencies=(_M033.NAME, _M035.NAME)),
    # Main-line M033 is renumbered in V4 because V4 already owns M033-M037.
    # It repairs the legacy learning_feedback schema before any reader mines
    # channel patterns.
    Migration(name=_M038.NAME, db_target="learning", ddl=_M038.DDL,
              dependencies=(_M003.NAME,)),
    # Receipt writes are a learning-plane concern and must never share the
    # memory.db recall lock domain.  The tables are self-contained: profile
    # lifecycle performs explicit cross-store erasure rather than an FK.
    Migration(name=_M040.NAME, db_target="learning", ddl=_M040.DDL,
              dependencies=(_M003.NAME,)),
    Migration(name=_M041.NAME, db_target="learning", ddl=_M041.DDL,
              dependencies=(_M040.NAME,)),
    Migration(name=_M050.NAME, db_target="learning", ddl=_M050.DDL,
              dependencies=(_M041.NAME,)),
    # Review-gated correction metadata is self-contained in memory.db. It
    # contains identifiers only and does not alter temporal fact state.
    Migration(name=_M042.NAME, db_target="memory", ddl=_M042.DDL,
              dependencies=(_M032.NAME,)),
    # M044 lets a bandit play record which memories it showed, so the reward
    # proxy can settle it from evidence instead of always falling through to
    # the 120-second neutral default. Additive column on M005's bandit_plays,
    # and eager on purpose: nothing bootstraps that table at engine init, so
    # there is no reason to defer it.
    Migration(name=_M044.NAME, db_target="learning", ddl=_M044.DDL,
              dependencies=(_M005.NAME,)),
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
    # Main-line M034 is renumbered in V4. It must remain deferred because its
    # backfill joins engine-bootstrapped memory_scenes and atomic_facts.
    Migration(name=_M039.NAME, db_target="memory", ddl=_M039.DDL),
    # M043 withholds model-written summaries from the retrieval corpus and
    # un-hides the memories they displaced. Deferred because it reads and
    # writes atomic_facts + fact_retention, both bootstrapped at engine init —
    # the same reason M011/M013/M015/M016 are deferred. apply_deferred takes a
    # verified snapshot before the first migration it actually applies, so the
    # store is recoverable.
    Migration(name=_M043.NAME, db_target="memory", ddl=_M043.DDL,
              dependencies=(_M011.NAME,)),
    # M045 holds the per-fact outcome score. Deferred because its backfill
    # reads action_outcomes, which engine init bootstraps — the same reason
    # M006 and M011 are deferred. Depends on M006 for the reward column it
    # averages.
    Migration(name=_M045.NAME, db_target="memory", ddl=_M045.DDL,
              dependencies=(_M006.NAME,)),
    # M046 renames the fact type used for planned future events, which means
    # rebuilding atomic_facts to widen a CHECK constraint SQLite cannot alter.
    # Deferred for the same reason as M043: atomic_facts is bootstrapped at
    # engine init, and apply_deferred takes a verified snapshot before the first
    # migration it applies, so a table rebuild has something to fall back to.
    # Depends on M043 so the two never contend for the same table in one pass.
    Migration(name=_M046.NAME, db_target="memory", ddl=_M046.DDL,
              dependencies=(_M043.NAME,)),
    # M047 rewrites the two Fisher vectors on each fact as float32 rather than
    # as decimal text. Deferred because it walks every fact in atomic_facts,
    # which engine init bootstraps. It changes no schema and both forms stay
    # readable, so it is resumable and an interrupted store still works.
    # Depends on M046 so a table rebuild and a full-table update never run in
    # the same pass over the same table.
    Migration(name=_M047.NAME, db_target="memory", ddl=_M047.DDL,
              dependencies=(_M046.NAME,)),
    # M048 finishes what M046 started: M046 renamed the type used for planned
    # events without re-reading a single one of them, so the same wrongly-filed
    # rows now carry a more confident name. Depends on M046 for the rename.
    Migration(name=_M048.NAME, db_target="memory", ddl=_M048.DDL,
              dependencies=(_M046.NAME,)),
    # M049 gives schema_version the unique constraint its six writers all
    # assumed it had. Every one uses INSERT OR IGNORE, which ignores nothing
    # without a constraint, so each appended a duplicate per run: seven distinct
    # versions held as 3,496 rows on one store and 234,348 on another. No
    # dependency -- it touches a bookkeeping table no other migration reads.
    Migration(name=_M049.NAME, db_target="memory", ddl=_M049.DDL),
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


def _foreign_live_daemon(memory_db: Path) -> "int | None":
    """Return the pid of another live daemon holding this data dir, or None.

    Migrations are not fenced against concurrent writers. The realistic hazard
    is an OLD daemon still running after an upgrade while a NEW one starts: its
    WAL appends continue while DDL is applied, which can make a migration fail
    non-deterministically. The snapshot itself stays consistent — the SQLite
    backup API copies committed pages only — and a racing migration is recorded
    as ``failed`` and is non-fatal, so this does not corrupt data.

    This detects the condition and reports it. It deliberately does NOT refuse:
    ``apply_all`` runs inside the daemon's own startup, so refusing whenever "a
    daemon is running" would refuse on itself, and blocking on a lock here would
    risk wedging startup — a worse outcome than a retryable failed step.
    """
    try:
        pid_file = memory_db.parent / "daemon.pid"
        if not pid_file.is_file():
            return None
        pid = int(pid_file.read_text().strip() or 0)
        if pid <= 0 or pid == os.getpid():
            return None
        os.kill(pid, 0)          # signal 0 tests liveness without touching it
        return pid
    except (OSError, ValueError):
        return None


def _nothing_left_to_apply(learning_db: Path, memory_db: Path) -> bool:
    """True when every migration is already recorded in its target database.

    Used to decide whether a snapshot is worth taking. A snapshot is only
    valuable when something is about to change; taking one on a start where
    nothing changes copies the ALREADY-MIGRATED store and then prunes a
    generation — so after two such starts the last copy of the original is gone,
    and the safety net has quietly deleted the thing it exists to protect.

    Errs toward False, which means "take the snapshot" — the safe direction.
    """
    try:
        for migration in MIGRATIONS:
            db_path = _db_for(migration.db_target, learning_db, memory_db)
            if not db_path.exists():
                return False
            conn = _connect(db_path)
            try:
                if not _deferred_already_applied(conn, migration.name):
                    return False
            finally:
                conn.close()
    except Exception:  # noqa: BLE001 — any doubt means take the snapshot
        return False
    return True


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

    # Take a consistent snapshot of both databases before any migration runs.
    # The backup uses the SQLite backup API so in-flight WAL writers are
    # never captured mid-transaction. InsufficientDiskSpaceError propagates
    # to the caller — migration is intentionally aborted when disk is too
    # tight to keep a recoverable copy.
    # A snapshot is only worth taking when something is about to change. This
    # runs on every engine construction, not just upgrades, so snapshotting
    # unconditionally meant an ordinary start copied the already-migrated store
    # and pruned a generation — two extra starts and the original was gone.
    _pending = not _nothing_left_to_apply(learning_db, memory_db)
    if not dry_run and not _pending:
        details["_backup"] = "skipped: every migration already applied"

    if not dry_run and _pending:
        _other = _foreign_live_daemon(memory_db)
        if _other is not None:
            logger.warning(
                "Another SuperLocalMemory daemon (pid %s) is still running and "
                "writing to this data directory. Migrations are not fenced "
                "against concurrent writers, so a step may fail and need a "
                "retry. Your data is not at risk: the snapshot copies committed "
                "pages only, and a failed step is recorded, never forced. Stop "
                "the other daemon and restart if a step fails.",
                _other,
            )
            details["_concurrent_daemon_pid"] = str(_other)

        backup_dir = _pre_migration_backup(
            learning_db, memory_db,
            backups_root=memory_db.parent / "pre-migration-snapshots",
        )
        # _pre_migration_backup returns the snapshots root itself, so this is
        # the directory to prune. Passing .parent pointed the collector at the
        # data directory, where it matched nothing and pruned nothing — leaving
        # every snapshot on disk for ever.
        _gc_old_backups(backup_dir)
        details["_backup"] = str(backup_dir)

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

    return {
        "applied": applied,
        "skipped": skipped,
        "failed": failed,
        "details": details,
    }


def _deferred_already_applied(conn: sqlite3.Connection, name: str) -> bool:
    """True when ``name`` is recorded as ``complete`` in this database's migration_log.

    Used only to decide whether a snapshot is needed. On any error it returns
    False, which errs toward taking a snapshot — the safe direction.

    A row whose status is ``failed`` or ``in_progress`` is NOT considered applied:
    the runner will retry those entries, and the store deserves a fresh snapshot
    before any retry runs DDL against it.  Counting any row (regardless of status)
    caused ``_nothing_left_to_apply`` to return True after a failed migration,
    so the retry ran against the already-partial store with no new safety copy.
    """
    try:
        row = conn.execute(
            "SELECT 1 FROM migration_log WHERE name = ? AND status = 'complete' LIMIT 1",
            (name,),
        ).fetchone()
        return row is not None
    except sqlite3.Error:
        return False


def _breaking_floor(learning_db: Path, memory_db: Path) -> int:
    """Highest floor declared by a migration that is recorded complete.

    A migration declares ``BREAKING_VERSION`` when a store it has touched must
    not be opened by an older build. Only completed ones count: a migration that
    failed has not changed anything an older build would trip over.
    """
    from superlocalmemory.storage._migration_internals import _MODULES

    logs = {"learning": _read_log(learning_db), "memory": _read_log(memory_db)}
    floor = 0
    for migration in (*MIGRATIONS, *DEFERRED_MIGRATIONS):
        module = _MODULES.get(migration.name)
        declared = getattr(module, "BREAKING_VERSION", 0) if module else 0
        if not declared:
            continue
        if logs.get(migration.db_target, {}).get(migration.name) == "complete":
            floor = max(floor, int(declared))
    return floor


def _stamp_breaking_floor(
    learning_db: Path, memory_db: Path, details: dict[str, str],
) -> None:
    """Raise the recorded version to the highest completed breaking floor.

    Monotonic: never lowers a stored version, so it cannot undo the completion
    certificate on an already-current store. Never fatal — a store that cannot
    be stamped is reported, because failing the whole run here would block an
    upgrade over a guard that only matters to older builds.
    """
    floor = _breaking_floor(learning_db, memory_db)
    if floor <= 0:
        return
    for db_path in (learning_db, memory_db):
        try:
            current = _read_schema_version(db_path)
            if current >= floor:
                continue
            conn = _connect(db_path)
            try:
                _ensure_schema_version_table(conn)
                _write_schema_version(conn, floor)
            finally:
                try:
                    conn.close()
                except sqlite3.Error:  # pragma: no cover
                    pass
        except sqlite3.Error as exc:  # pragma: no cover — reported, not fatal
            details["schema_version_floor"] = (
                f"cannot raise the floor on {db_path}: {exc}"
            )


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

    # apply_all snapshots before it touches anything; this pass did not, yet it
    # applies real DDL to both managed databases — including the column the
    # daemon needs to start. An interrupted deferred pass therefore had no
    # recoverable copy at all. The snapshot is taken LAZILY, immediately before
    # the first migration that will actually be applied, so a pass with nothing
    # to do costs no disk and does not capture post-init state unnecessarily.
    _snapshot_state: dict[str, object] = {"taken": dry_run}

    def _ensure_snapshot() -> None:
        if _snapshot_state["taken"]:
            return
        _snapshot_state["taken"] = True
        backup_dir = _pre_migration_backup(
            learning_db, memory_db,
            backups_root=memory_db.parent / "pre-migration-snapshots",
        )
        _gc_old_backups(backup_dir)
        details["_deferred_backup"] = str(backup_dir)

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

            if not dry_run and not _deferred_already_applied(conn, migration.name):
                _ensure_snapshot()

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

    # A migration that makes the store unusable by an older build declares a
    # floor, and that floor is written as soon as the migration is recorded
    # complete — BEFORE and independent of the completion certificate below.
    #
    # The certificate is all-or-nothing across both databases by design. That is
    # right for "is this store fully migrated" and wrong for "may an older build
    # write to it": an unrelated failure on the other database would otherwise
    # leave a rebuilt table guarded by the old ceiling, and the first planned
    # event an older build stored would be rejected by the new constraint and
    # lost. Raising the floor turns that into a refusal to start, which is what
    # the ceiling is for.
    if not dry_run:
        _stamp_breaking_floor(learning_db, memory_db, details)

    # The version ceiling is a completion certificate, not an intent marker.
    # M039 is deferred until engine-owned tables exist, so apply_all must not
    # stamp version 39. Stamp both stores only after every eager and deferred
    # migration is recorded complete on its declared target.
    if not failed and not dry_run:
        logs = {
            "learning": _read_log(learning_db),
            "memory": _read_log(memory_db),
        }
        incomplete = [
            migration.name
            for migration in (*MIGRATIONS, *DEFERRED_MIGRATIONS)
            if logs[migration.db_target].get(migration.name) != "complete"
        ]
        if incomplete:
            failed.append("schema_version_stamp")
            details["schema_version_stamp"] = (
                "not stamped; incomplete migrations: " + ", ".join(incomplete)
            )
        else:
            for _stamp_db in (learning_db, memory_db):
                try:
                    _stamp_conn = _connect(_stamp_db)
                    try:
                        _ensure_schema_version_table(_stamp_conn)
                        _write_schema_version(
                            _stamp_conn, SUPPORTED_SCHEMA_VERSION,
                        )
                    finally:
                        try:
                            _stamp_conn.close()
                        except sqlite3.Error:  # pragma: no cover
                            pass
                except sqlite3.Error as exc:  # pragma: no cover
                    failed.append("schema_version_stamp")
                    details["schema_version_stamp"] = (
                        f"cannot stamp {_stamp_db}: {exc}"
                    )
                    break

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
