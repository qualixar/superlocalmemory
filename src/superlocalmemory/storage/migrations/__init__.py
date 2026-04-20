# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory v3.4.22 — LLD-07

"""v3.4.22 migration definitions.

One module per migration. Each exposes:
  - ``NAME``: canonical migration name (used in ``migration_log``)
  - ``DB_TARGET``: either ``"learning"`` or ``"memory"``
  - ``DDL``: the full DDL string including ``BEGIN IMMEDIATE`` / ``COMMIT``
    (where the migration warrants a transaction; see each module).

The ordered list here is the authoritative order for ``migration_runner``.
M003 must run first (bootstraps the log table); the others follow in the
order LLD-07 §4 specifies.
"""

from __future__ import annotations

from . import (
    M001_add_signal_features_columns,
    M002_model_state_history,
    M003_migration_log,
    M004_cross_platform_sync_log,
    M005_bandit_tables,
)

# ---------------------------------------------------------------------------
# Backward-compat shim: the legacy flat module
# ``superlocalmemory/storage/migrations.py`` shipped in v3.4.20 carried
# ``CURRENT_SCHEMA_VERSION`` / ``get_schema_version`` / ``set_schema_version``
# / ``is_v1_database`` / ``needs_migration`` / ``backup_database``. Creating
# the ``migrations/`` package in Wave 1 shadowed that flat module, so any
# caller that did ``from superlocalmemory.storage.migrations import X``
# broke. We re-load the legacy module under a distinct name and re-export
# the symbols here so pre-3.4.22 callers (and the pre-existing tests at
# ``tests/test_storage/test_migrations.py``) keep working unchanged.
# ---------------------------------------------------------------------------


def _load_legacy_module():  # pragma: no cover — trivial file loader
    import importlib.util
    from pathlib import Path as _Path

    legacy_path = _Path(__file__).resolve().parents[1] / "migrations.py"
    if not legacy_path.is_file():
        return None
    spec = importlib.util.spec_from_file_location(
        "superlocalmemory.storage._legacy_migrations_flat",
        str(legacy_path),
    )
    if spec is None or spec.loader is None:
        return None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_legacy = _load_legacy_module()
if _legacy is not None:
    CURRENT_SCHEMA_VERSION = getattr(_legacy, "CURRENT_SCHEMA_VERSION", 1)
    get_schema_version = getattr(_legacy, "get_schema_version", None)
    set_schema_version = getattr(_legacy, "set_schema_version", None)
    is_v1_database = getattr(_legacy, "is_v1_database", None)
    needs_migration = getattr(_legacy, "needs_migration", None)
    backup_database = getattr(_legacy, "backup_database", None)

__all__ = (
    "M001_add_signal_features_columns",
    "M002_model_state_history",
    "M003_migration_log",
    "M004_cross_platform_sync_log",
    "M005_bandit_tables",
    # Legacy re-exports (backward compat):
    "CURRENT_SCHEMA_VERSION",
    "get_schema_version",
    "set_schema_version",
    "is_v1_database",
    "needs_migration",
    "backup_database",
)
