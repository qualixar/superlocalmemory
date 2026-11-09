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

from superlocalmemory.storage import _boot_snapshot as _boot_copy
from superlocalmemory.storage._migration_catalogue import (  # noqa: F401 -- re-exported
    _M001,
    _M002,
    _M003,
    _M004,
    _M005,
    _M006,
    _M007,
    _M009,
    _M010,
    _M011,
    _M012,
    _M013,
    _M014,
    _M015,
    _M016,
    _M017,
    _M018,
    _M019,
    _M020,
    _M021,
    _M022,
    _M023,
    _M024,
    _M025,
    _M026,
    _M027,
    _M028,
    _M029,
    _M030,
    _M031,
    _M032,
    _M033,
    _M034,
    _M035,
    _M036,
    _M037,
    _M038,
    _M039,
    _M040,
    _M041,
    _M042,
    _M043,
    _M044,
    _M045,
    _M046,
    _M047,
    _M048,
    _M049,
    _M050,
    _M051,
    _M052,
    DEFERRED_MIGRATIONS,
    MIGRATIONS,
)
from superlocalmemory.storage._migration_internals import (
    _MODULES,  # noqa: F401 — re-exported for test/introspection compatibility
    Migration,
    _apply_single,
    _connect,
    _db_for,
    _ensure_migration_log,
    _migration_log_exists,
    _read_log,
)
from superlocalmemory.storage._schema_version import (
    SUPPORTED_SCHEMA_VERSION,
    SchemaVersionError,
)
from superlocalmemory.storage._schema_version import (
    check_version_or_raise as _check_version_or_raise,
)
from superlocalmemory.storage._schema_version import (
    ensure_schema_version_table as _ensure_schema_version_table,
)
from superlocalmemory.storage._schema_version import (
    read_schema_version as _read_schema_version,
)
from superlocalmemory.storage._schema_version import (
    write_schema_version as _write_schema_version,
)
from superlocalmemory.storage.backup import (
    _gc_old_backups,
    _pre_migration_backup,
)

logger = logging.getLogger(__name__)


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
        pid = int(pid_file.read_text(encoding="utf-8").strip() or 0)
        if pid <= 0 or pid == os.getpid():
            return None
    except (OSError, ValueError):
        return None
    from superlocalmemory.core.platform_utils import is_pid_alive

    return pid if is_pid_alive(pid) else None


def _downgrade_hold_active(data_root: Path) -> bool:
    """True while the user has prepared this store for an older version.

    The predicate (marker present and no other build has run since) is owned by
    ``storage.upgrade_restore``. Absent module means no hold. A predicate that
    raises also means no hold: stamping keeps an older build refused, which is
    the safe direction.
    """
    # TODO(WP-6): upgrade_restore.downgrade_hold_active lands with WP-6.
    try:
        from superlocalmemory.storage.upgrade_restore import downgrade_hold_active
    except ImportError:
        return False
    try:
        return bool(downgrade_hold_active(data_root))
    except Exception as exc:  # noqa: BLE001 - fail toward stamping
        logger.warning("downgrade hold check failed; stamping normally: %s", exc)
        return False


_ALL_DATABASES = frozenset({"learning", "memory"})


def _pending_targets(migrations, learning_db: Path, memory_db: Path) -> frozenset[str]:
    """The databases (``"learning"``, ``"memory"``) a pending migration will change.

    A migration counts as pending until it is recorded ``complete`` in its own
    target database. Used to decide which databases a snapshot must hold: an
    update that changes learning.db alone used to copy memory.db as well --
    about 8 s per GB of a store it does not touch, on the first start of the
    new version. Errs toward both databases on any doubt — the safe direction.
    """
    targets: set[str] = set()
    try:
        for migration in migrations:
            target = migration.db_target
            if target in targets:
                continue
            db_path = _db_for(target, learning_db, memory_db)
            if not db_path.exists():
                targets.add(target)
                continue
            conn = _connect(db_path)
            try:
                if not _deferred_already_applied(conn, migration.name):
                    targets.add(target)
            finally:
                conn.close()
    except Exception:  # noqa: BLE001 — any doubt means copy everything
        return _ALL_DATABASES
    if targets - _ALL_DATABASES:  # a target this runner cannot name: copy everything
        return _ALL_DATABASES
    return frozenset(targets)


def _nothing_left_to_apply(learning_db: Path, memory_db: Path) -> bool:
    """True when every migration is already recorded in its target database.

    Used to decide whether a snapshot is worth taking. A snapshot is only
    valuable when something is about to change; taking one on a start where
    nothing changes copies the ALREADY-MIGRATED store and then prunes a
    generation — so after two such starts the last copy of the original is gone,
    and the safety net has quietly deleted the thing it exists to protect.

    Errs toward False, which means "take the snapshot" — the safe direction.
    """
    return not _pending_targets(MIGRATIONS, learning_db, memory_db)


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
    # Only the databases this start will change are copied: those of a pending
    # migration here, and of a pending deferred one, so that the deferred pass
    # of this same start can stand on this one copy (_boot_snapshot).
    _eager_targets = _pending_targets(MIGRATIONS, learning_db, memory_db)
    _pending = bool(_eager_targets)
    _targets = (_eager_targets | _pending_targets(DEFERRED_MIGRATIONS, learning_db, memory_db)
                if _pending else frozenset())
    if not dry_run and not _pending:
        details["_backup"] = "skipped: every migration already applied"

    if not dry_run and _pending:
        # daemon.pid alone cannot see an old daemon here: the starting daemon
        # has already written its own pid into it. The writer lease names the
        # process actually holding the store.
        _other = _foreign_live_daemon(memory_db)
        if _other is None:
            _holder = _boot_copy.lease_holder(memory_db)
            if _holder is not None:
                _other = "unknown" if _holder == _boot_copy.UNKNOWN_HOLDER else _holder
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

        # A copy from an earlier pass must not outlive a new attempt to take one.
        _boot_copy.forget()
        _snapshots_root = memory_db.parent / "pre-migration-snapshots"
        _names_before = _boot_copy.copy_names(_snapshots_root)
        backup_dir = _pre_migration_backup(
            learning_db, memory_db, backups_root=_snapshots_root, databases=_targets,
        )
        # _pre_migration_backup returns the snapshots root itself, so this is
        # the directory to prune. Passing .parent pointed the collector at the
        # data directory, where it matched nothing and pruned nothing — leaving
        # every snapshot on disk for ever.
        _gc_old_backups(
            backup_dir,
            protect=_boot_copy.copy_names(_snapshots_root) - _names_before,
        )
        details["_backup"] = str(backup_dir)
        # The deferred pass of this same start may stand on this copy rather
        # than take a second one -- unless another daemon can write meanwhile.
        if _other is None and not _boot_copy.another_writer_holds(memory_db):
            _boot_copy.remember(learning_db, memory_db, backup_dir, _names_before,
                                databases=_targets)

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
    # The copy the eager pass of this start took, claimed once whether used or
    # not; _boot_snapshot documents why it is a faithful "before" for this pass.
    _start_copy = None if dry_run else _boot_copy.claim(learning_db, memory_db)
    # Only the databases a pending deferred migration changes are copied.
    _targets = (frozenset() if dry_run
                else _pending_targets(DEFERRED_MIGRATIONS, learning_db, memory_db))

    def _ensure_snapshot() -> None:
        if _snapshot_state["taken"]:
            return
        _snapshot_state["taken"] = True
        if (_start_copy is not None and _boot_copy.still_on_disk(_start_copy)
                and _targets <= _start_copy.databases
                and _foreign_live_daemon(memory_db) is None
                and not _boot_copy.another_writer_holds(memory_db)):
            logger.info(
                "Deferred migrations are covered by the safety copy this start "
                "already took in %s; not copying the store a second time.",
                _start_copy.directory,
            )
            details["_deferred_backup"] = str(_start_copy.directory)
            return
        _root = memory_db.parent / "pre-migration-snapshots"
        _before = _boot_copy.copy_names(_root)
        backup_dir = _pre_migration_backup(learning_db, memory_db, backups_root=_root,
                                           databases=_targets or _ALL_DATABASES)
        _gc_old_backups(
            backup_dir, protect=_boot_copy.copy_names(_root) - _before,
        )
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
        elif _downgrade_hold_active(memory_db.parent):
            details["schema_version_stamp"] = (
                "held: this store is prepared for an older version"
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
