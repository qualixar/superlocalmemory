# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""SLM 3.2.3 -> 3.3 migration: idempotent schema upgrade.

Detects a 3.2.3 database (missing 3.3 tables) and creates
new tables via CREATE TABLE IF NOT EXISTS. Safe to call on
any DB version — existing tables are never altered.

New tables in 3.3 (all from Phases A-G):
  - fact_retention           (Phase A: Ebbinghaus forgetting)
  - polar_embeddings         (Phase B: PolarQuant quantization)
  - embedding_quantization_metadata  (Phase B: EAP bit-width tracking)
  - ccq_consolidated_blocks  (Phase E: Cognitive Consolidation)
  - ccq_audit_log            (Phase E: CCQ audit trail)
  - soft_prompt_templates    (Phase F: Learning Brain)

All six are already defined in schema_v32.py V32_DDL and created
by create_all_tables(). This migration module detects their absence
and creates them for databases that were created before SLM 3.3.

Part of Qualixar | Author: Varun Pratap Bhardwaj
License: AGPL-3.0-or-later
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass, field
from typing import Final

logger = logging.getLogger(__name__)


# Tables introduced or required by SLM 3.3 phases
_V33_TABLES: Final[tuple[str, ...]] = (
    "fact_retention",
    "polar_embeddings",
    "embedding_quantization_metadata",
    "ccq_consolidated_blocks",
    "ccq_audit_log",
    "soft_prompt_templates",
)


@dataclass(frozen=True)
class MigrationReport:
    """Report of migration actions taken."""

    tables_checked: int = 0
    tables_created: tuple[str, ...] = ()
    tables_existed: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()

    @property
    def is_clean(self) -> bool:
        return len(self.errors) == 0


def _get_existing_tables(conn: sqlite3.Connection) -> frozenset[str]:
    """Return set of all table names in the database."""
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()
    return frozenset(row[0] for row in rows)


def detect_v323_database(conn: sqlite3.Connection) -> bool:
    """Detect if this is a pre-3.3 database missing new tables.

    Returns True if at least one V33 table is missing.
    """
    existing = _get_existing_tables(conn)
    return any(table not in existing for table in _V33_TABLES)


def migrate_v323_to_v33(conn: sqlite3.Connection) -> MigrationReport:
    """Idempotent migration from SLM 3.2.3 to 3.3.

    Creates missing 3.3 tables via the canonical DDL from schema_v32.py.
    Uses CREATE TABLE IF NOT EXISTS — safe to run on any DB version.

    Args:
        conn: Open SQLite connection. Caller manages commit.

    Returns:
        MigrationReport with details of actions taken.
    """
    from superlocalmemory.storage.schema_v32 import V32_DDL

    existing = _get_existing_tables(conn)
    created: list[str] = []
    existed: list[str] = []
    errors: list[str] = []

    for table in _V33_TABLES:
        if table in existing:
            existed.append(table)
            logger.debug("Table %s already exists — skipping", table)
            continue

        # Find the DDL statement that creates this table
        ddl_found = False
        for ddl in V32_DDL:
            if f"CREATE TABLE IF NOT EXISTS {table}" in ddl:
                try:
                    conn.executescript(ddl)
                    created.append(table)
                    ddl_found = True
                    logger.info("Created table: %s", table)
                except sqlite3.Error as exc:
                    msg = f"Failed to create {table}: {exc}"
                    errors.append(msg)
                    logger.error(msg)
                    ddl_found = True
                break

        if not ddl_found:
            msg = f"No DDL found for table: {table}"
            errors.append(msg)
            logger.warning(msg)

    report = MigrationReport(
        tables_checked=len(_V33_TABLES),
        tables_created=tuple(created),
        tables_existed=tuple(existed),
        errors=tuple(errors),
    )

    logger.info(
        "Migration report: checked=%d, created=%d, existed=%d, errors=%d",
        report.tables_checked,
        len(report.tables_created),
        len(report.tables_existed),
        len(report.errors),
    )

    return report
