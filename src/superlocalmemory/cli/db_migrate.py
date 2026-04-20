# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory v3.4.22 — LLD-06 §7.2

"""CLI handler for ``slm db migrate``.

LLD reference: ``.backup/active-brain/lld/LLD-06-windows-binary-and-legacy-migration.md``
Section 7.2 (slm db migrate CLI).

Thin wrapper over the canonical runner in
``superlocalmemory.storage.migration_runner``. This module owns only
the user-facing surface (stdout formatting + exit codes). All DDL +
runner logic lives in LLD-07 territory — per H15, no migration schema
is defined or duplicated here.
"""
from __future__ import annotations

from argparse import Namespace
from pathlib import Path


# Canonical paths — match LLD-01 / LLD-07 layout. Callers can override
# via the ``learning_db_path`` / ``memory_db_path`` attributes on args
# (tests rely on this to point at fixture DBs).
DEFAULT_HOME = Path.home() / ".superlocalmemory"


def _resolve_paths(args: Namespace) -> tuple[Path, Path]:
    learning = getattr(args, "learning_db_path", None)
    memory = getattr(args, "memory_db_path", None)
    if learning is None:
        learning = DEFAULT_HOME / "learning.db"
    if memory is None:
        memory = DEFAULT_HOME / "memory.db"
    return Path(learning), Path(memory)


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
            for name, state in report.items():
                print(f"  {name}: {state}")
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
