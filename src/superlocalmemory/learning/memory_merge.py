# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory v3.4.22 — LLD-12 §2.3 + §1.4

"""Reversible merge-log writer + unmerge for consolidation.

LLD-12 §1 hard invariant: consolidation NEVER issues
``DELETE FROM atomic_facts``. Duplicates merge by:

1. INSERT into ``memory_merge_log`` (canonical + merged fact_ids, scores,
   timestamp, reversible flag).
2. UPDATE ``atomic_facts`` SET archive_status='merged',
   merged_into=<canonical>. Row stays; only status flips.

``unmerge(merge_id)`` reverses the operation by flipping archive_status
back to 'live' and clearing merged_into. The log row is marked
``reversible=0`` once reversed so a second unmerge is a no-op.

All operations run inside a single SQLite transaction with
``busy_timeout=2000``. Partial failures roll back cleanly, leaving the
DB in its pre-merge state.
"""

from __future__ import annotations

import logging
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def apply_merges(
    memory_db_path: str | Path,
    candidates: list[tuple[str, str, float, float]],
    *,
    profile_id: str,
) -> int:
    """Apply merge candidates transactionally. Returns number applied.

    Each candidate is ``(canonical_fact_id, merged_fact_id, cosine, jaccard)``.

    Never deletes from ``atomic_facts``. Always writes a row to
    ``memory_merge_log`` per applied merge.

    Idempotent: if ``merged_fact_id`` is already ``archive_status='merged'``
    from a prior run, the candidate is skipped (count not incremented).
    """
    if not candidates:
        return 0

    conn = sqlite3.connect(str(memory_db_path), timeout=10.0)
    conn.execute("PRAGMA busy_timeout=2000")
    applied = 0
    # S-L02: track the candidate list in flight so a rollback diagnostic
    # can blame the exact set of (canonical, merged) pairs instead of a
    # blanket "rollback" message. Operators on the dashboard previously
    # saw zero fidelity about which candidates were in the transaction
    # at commit-time.
    in_flight: list[tuple[str, str]] = []
    try:
        conn.execute("BEGIN IMMEDIATE")
        for canonical_id, merged_id, cos, jac in candidates:
            # Skip if already merged in a prior cycle.
            row = conn.execute(
                "SELECT archive_status FROM atomic_facts WHERE fact_id=?",
                (merged_id,),
            ).fetchone()
            if row is None:
                continue
            if row[0] == "merged":
                continue

            conn.execute(
                "INSERT INTO memory_merge_log "
                "(merge_id, profile_id, canonical_fact_id, merged_fact_id, "
                " cosine_sim, entity_jaccard, merged_at, reversible) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, 1)",
                (
                    str(uuid.uuid4()),
                    profile_id,
                    canonical_id,
                    merged_id,
                    float(cos),
                    float(jac),
                    _iso_now(),
                ),
            )
            conn.execute(
                "UPDATE atomic_facts "
                "SET archive_status='merged', "
                "    archive_reason='cosine_dup', "
                "    merged_into=? "
                "WHERE fact_id=?",
                (canonical_id, merged_id),
            )
            applied += 1
            in_flight.append((canonical_id, merged_id))
        conn.commit()
    except sqlite3.Error as exc:
        conn.rollback()
        logger.warning(
            "apply_merges rollback: profile=%s pre-rollback_applied=%d "
            "in_flight=%s error=%s",
            profile_id, applied, in_flight, exc,
        )
        applied = 0
    finally:
        conn.close()
    return applied


def unmerge(memory_db_path: str | Path, merge_id: str) -> bool:
    """Reverse a merge by merge_id. Returns True on success.

    Flips the merged fact's archive_status back to 'live', clears
    merged_into, and marks the log row ``reversible=0``.
    """
    conn = sqlite3.connect(str(memory_db_path), timeout=10.0)
    conn.execute("PRAGMA busy_timeout=2000")
    try:
        row = conn.execute(
            "SELECT merged_fact_id, reversible FROM memory_merge_log "
            "WHERE merge_id=?",
            (merge_id,),
        ).fetchone()
        if row is None:
            return False
        merged_fid, reversible = row
        if not reversible:
            return False

        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            "UPDATE atomic_facts "
            "SET archive_status='live', archive_reason=NULL, merged_into=NULL "
            "WHERE fact_id=?",
            (merged_fid,),
        )
        conn.execute(
            "UPDATE memory_merge_log SET reversible=0 WHERE merge_id=?",
            (merge_id,),
        )
        conn.commit()
        return True
    except sqlite3.Error as exc:
        conn.rollback()
        logger.warning("unmerge rollback: %s", exc)
        return False
    finally:
        conn.close()


__all__ = ("apply_merges", "unmerge")
