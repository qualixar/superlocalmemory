# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory v3.4.22 — LLD-06 §7.2

"""CLI handler for ``slm db migrate``.


Thin wrapper over the canonical runner in
``superlocalmemory.storage.migration_runner``. This module owns only
the user-facing surface (stdout formatting + exit codes). All DDL +
runner logic lives in LLD-07 territory — per H15, no migration schema
is defined or duplicated here.
"""
from __future__ import annotations

from argparse import Namespace
from pathlib import Path

from superlocalmemory.infra.data_root import DynamicStatePath

# Canonical paths — match LLD-01 / LLD-07 layout. Callers can override
# via the ``learning_db_path`` / ``memory_db_path`` attributes on args
# (tests rely on this to point at fixture DBs).
DEFAULT_HOME = DynamicStatePath()


def _resolve_paths(args: Namespace) -> tuple[Path, Path]:
    learning = getattr(args, "learning_db_path", None)
    memory = getattr(args, "memory_db_path", None)
    if learning is None:
        learning = DEFAULT_HOME / "learning.db"
    if memory is None:
        memory = DEFAULT_HOME / "memory.db"
    return Path(learning), Path(memory)


def _end_state_disagreements(learning_db, memory_db) -> dict[str, str]:
    """Migrations recorded complete whose own verification no longer passes.

    ``--status`` reads ``migration_log``, which records what was *done*. The
    daemon re-runs each completed migration's ``verify()`` on every start, which
    checks whether what was done still *holds*. When those two answers differ
    the log says ``complete`` and the health endpoint says the migration failed,
    and until now nothing on any surface showed the two were even asking
    different questions -- so the only available reading was that one of them
    was lying. Reported as #125.

    Read-only and fail-quiet: this is a diagnostic printed beside a status line,
    and it must never be the reason a status command exits non-zero.
    """
    import sqlite3

    from superlocalmemory.storage._migration_internals import _MODULES
    from superlocalmemory.storage.migration_runner import MIGRATIONS

    try:
        from superlocalmemory.storage.migration_runner import DEFERRED_MIGRATIONS
    except ImportError:  # pragma: no cover — older layouts
        DEFERRED_MIGRATIONS = ()

    out: dict[str, str] = {}
    for migration in list(MIGRATIONS) + list(DEFERRED_MIGRATIONS):
        verify_fn = getattr(_MODULES.get(migration.name), "verify", None)
        if not callable(verify_fn):
            continue
        db_path = memory_db if migration.db_target != "learning" else learning_db
        try:
            conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        except sqlite3.Error:
            continue
        try:
            row = conn.execute(
                "SELECT status FROM migration_log WHERE name = ? LIMIT 1",
                (migration.name,),
            ).fetchone()
            if not row or row[0] != "complete":
                continue
            if not bool(verify_fn(conn)):
                out[migration.name] = "  <- recorded complete, end-state no longer holds"
        except Exception:  # noqa: BLE001 — a diagnostic must not break status
            pass
        finally:
            try:
                conn.close()
            except sqlite3.Error:  # pragma: no cover
                pass
    return out


def cmd_db_migrate(args: Namespace) -> int:
    """Apply pending migrations or report status.

    Behaviour:
      - ``--status`` prints the per-migration status recorded in each
        DB's ``migration_log``. Exit 0 unless reading fails.
      - ``--dry-run`` runs the runner in dry-run mode (no writes).
      - Default: runs ``apply_all``.

    Exit codes (also returned for tests that capture return value):
      - 0 on success (no failed migrations).
      - 1 if any migration is reported as ``failed``.
    """
    from superlocalmemory.storage.migration_runner import apply_all, status

    learning_db, memory_db = _resolve_paths(args)

    if getattr(args, "status", False):
        report = status(learning_db, memory_db)
        if not report:
            print("(no migrations registered)")
        else:
            disagreements = _end_state_disagreements(learning_db, memory_db)
            for name, state in report.items():
                note = disagreements.get(name, "")
                print(f"  {name}: {state}{note}")
            if disagreements:
                print()
                print(
                    "  A migration marked complete is re-checked on every start "
                    "by its own\n"
                    "  verification. The ones flagged above are recorded as done "
                    "and their\n"
                    "  end-state no longer holds, which is what the daemon "
                    "reports as a\n"
                    "  migration failure while this log still reads complete. "
                    "Run `slm db\n"
                    "  migrate` to let each one try to repair itself, and see "
                    "`migration_failure_reasons`\n"
                    "  in `slm health --json` for what specifically did not hold."
                )
        return 0

    dry_run = bool(getattr(args, "dry_run", False))
    result = apply_all(learning_db, memory_db, dry_run=dry_run)
    applied = result.get("applied", [])
    skipped = result.get("skipped", [])
    failed = result.get("failed", [])
    print(
        f"Applied={len(applied)} Skipped={len(skipped)} Failed={len(failed)}"
    )
    if failed:
        details = result.get("details", {})
        for name in failed:
            print(f"  FAILED {name}: {details.get(name, '(no detail)')}")
        return 1
    return 0


__all__ = ("cmd_db_migrate", "DEFAULT_HOME")
