# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Keeping the bytes of a damaged JSON value before a writer replaces it.

A JSON text column that holds bytes which are not UTF-8 reads as empty
(``json_columns.py``). When a writer then replaces that column with a value it
rebuilt from the source of truth, the original bytes would be gone.
:func:`keep_bytes` records them first in the change ledger
(``enrichment_changes``, one row per column, under the run ``damaged-values``),
the same place the enrichment pass records what it replaces. The bytes are kept
base64-encoded by ``enrichment_ledger.dumps``; ``enrichment_ledger.loads``
returns them. Entity profiles and scenes keep theirs in the version row
(``compiler/versions.py``) instead.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from datetime import UTC, datetime
from typing import Any

from superlocalmemory.storage.json_columns import is_damaged

logger = logging.getLogger(__name__)

RUN_ID = "damaged-values"
_SOURCE = "damaged_value"
_FOREIGN = {"atomic_facts": "fact_id", "memories": "memory_id"}


def damaged_ids(db: Any, table: str, id_column: str, profile_id: str, column: str) -> set[str]:
    """Ids of the rows of ``profile_id`` whose ``column`` holds bytes that are not UTF-8."""
    try:
        rows = db.execute(
            f'SELECT "{id_column}" AS id, "{column}" AS v FROM "{table}" '  # noqa: S608
            f'WHERE profile_id = ? AND typeof("{column}") = \'blob\'', (profile_id,))
        return {str(dict(r)["id"]) for r in rows if is_damaged(dict(r)["v"])}
    except Exception as exc:  # noqa: BLE001 - a store without the column has nothing damaged
        logger.debug("damaged-value scan of %s.%s skipped: %s", table, column, exc)
        return set()


def keep_bytes(db: Any, table: str, id_column: str, target_id: str, profile_id: str,
               columns: Iterable[str]) -> set[str]:
    """Record the damaged bytes in ``columns`` of one row; return the columns that could NOT be kept.

    Columns that are not damaged are not looked at. An empty result means the
    writer may replace every damaged column. A column in the result still holds
    its bytes and must be left as it is.
    """
    unkept: set[str] = set()
    wanted = [c for c in columns]
    if not wanted:
        return unkept
    try:
        rows = db.execute(
            "SELECT " + ", ".join(f'"{c}" AS "{c}"' for c in wanted)  # noqa: S608
            + f' FROM "{table}" WHERE "{id_column}" = ? AND profile_id = ?',
            (target_id, profile_id))
    except Exception as exc:  # noqa: BLE001
        logger.debug("damaged-value read of %s skipped: %s", table, exc)
        return unkept
    if not rows:
        return unkept
    row = dict(rows[0])
    damaged = [c for c in wanted if is_damaged(row.get(c))]
    if not damaged:
        return unkept
    from superlocalmemory.core.description_metadata import VERSION
    from superlocalmemory.storage import enrichment_ledger as ledger

    stamp = datetime.now(UTC).isoformat()
    try:
        db.execute(
            "INSERT OR IGNORE INTO enrichment_runs(run_id,enrichment_version,action,status,"
            "upper_rowid,total,created_at,updated_at) VALUES (?,?,?,?,?,?,?,?)",
            (RUN_ID, VERSION, "damaged_values", "completed", 0, 0, stamp, stamp))
        for column in damaged:
            before = ledger.dumps(bytes(row[column]))
            seen = db.execute(
                "SELECT 1 FROM enrichment_changes WHERE run_id=? AND target_table=? "
                "AND target_id=? AND field=? AND before_json=? LIMIT 1",
                (RUN_ID, table, target_id, column, before))
            if seen:
                continue
            foreign = _FOREIGN.get(table)
            db.execute(
                "INSERT INTO enrichment_changes(run_id,profile_id,fact_id,memory_id,"
                "target_table,target_id,field,before_json,after_json,source,enrichment_version) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (RUN_ID, profile_id, target_id if foreign == "fact_id" else None,
                 target_id if foreign == "memory_id" else None, table, target_id, column,
                 before, ledger.dumps(None), _SOURCE, VERSION))
    except Exception as exc:  # noqa: BLE001 - never lose the bytes to keep going
        logger.warning("could not keep the damaged %s.%s of %s (%s); it is left as it is",
                       table, ",".join(damaged), target_id, exc)
        return set(damaged)
    return unkept


__all__ = ["RUN_ID", "damaged_ids", "keep_bytes"]
