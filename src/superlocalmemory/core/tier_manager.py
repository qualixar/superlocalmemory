# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""SuperLocalMemory V3.4.11 "Scale-Ready" — Tier Manager.

Manages the lifecycle tiers of atomic facts:
  - active (hot): Recent, frequently accessed. Full retrieval priority.
  - warm: Consolidated or aging. Reduced retrieval weight (0.7x).
  - cold: Old, rarely accessed. Low retrieval weight (0.3x).
  - archived: Superseded or consolidated. Excluded from default retrieval,
              but searchable via deep recall.

CRITICAL RULE: Facts are NEVER deleted. Only moved between tiers.
The forgetting curve affects RETRIEVAL RANKING, not data existence.

Demotion logic: A fact is demoted based on time since last access
(via fact_retention.last_accessed_at), NOT total age from created_at.
If no access record exists, created_at is used as fallback.

Runs on the maintenance scheduler alongside Ebbinghaus/Langevin.

Part of Qualixar | Author: Varun Pratap Bhardwaj
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, UTC
from typing import TYPE_CHECKING

from superlocalmemory.core.lifecycle_state import (
    reconcile_profile_lifecycle,
    set_fact_lifecycle_zone,
)

if TYPE_CHECKING:
    from superlocalmemory.storage.database import DatabaseManager

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tier thresholds (configurable via SLMConfig in future)
# ---------------------------------------------------------------------------

WARM_AFTER_DAYS = 30       # active → warm after 30 days without access
COLD_AFTER_DAYS = 180      # warm → cold after 180 days without access
ARCHIVE_AFTER_DAYS = 365   # cold → archived after 365 days without access

ACCESS_BOOST_THRESHOLD = 5    # 5+ accesses: boost demotion timer
ACCESS_BOOST_MULTIPLIER = 2.0

IMPORTANCE_RESIST_THRESHOLD = 0.8  # importance >= 0.8: boost demotion timer
IMPORTANCE_RESIST_MULTIPLIER = 3.0

# Cap: when both boosts apply, use max (not multiplicative) to prevent
# 6x suppression (which would delay archival for 6+ years).
MAX_COMBINED_MULTIPLIER = 3.0

_BATCH_SIZE = 1000  # Process facts in batches to prevent OOM at scale


def evaluate_tiers(
    db: DatabaseManager,
    profile_id: str = "default",
    dry_run: bool = False,
) -> dict[str, int]:
    """Evaluate and update lifecycle tiers for all facts in a profile.

    Rules:
      1. Pinned facts ALWAYS stay 'active' regardless of age/access.
      2. Recently accessed facts resist demotion (access_count boost).
      3. High-importance facts resist demotion (importance boost).
      4. Boosts cap at MAX_COMBINED_MULTIPLIER (3x), not multiplicative.
      5. NEVER delete facts. NEVER.
    """
    stats = {
        "demoted_to_warm": 0,
        "demoted_to_cold": 0,
        "demoted_to_archive": 0,
        "pinned_protected": 0,
        "total_evaluated": 0,
    }

    # Repair any rows written by pre-V3.7 split lifecycle paths before making
    # new transition decisions.
    reconcile_profile_lifecycle(db, profile_id)

    now = datetime.now(UTC)
    pinned_ids = _get_pinned_fact_ids(db, profile_id)

    stats["demoted_to_warm"] = _demote_tier(
        db, profile_id, "active", "warm",
        WARM_AFTER_DAYS, pinned_ids, now, dry_run,
    )
    stats["demoted_to_cold"] = _demote_tier(
        db, profile_id, "warm", "cold",
        COLD_AFTER_DAYS, pinned_ids, now, dry_run,
    )
    stats["demoted_to_archive"] = _demote_tier(
        db, profile_id, "cold", "archived",
        ARCHIVE_AFTER_DAYS, pinned_ids, now, dry_run,
    )

    stats["pinned_protected"] = len(pinned_ids)

    # Count only non-archived facts (archived are not inspected by _demote_tier)
    rows = db.execute(
        "SELECT COUNT(*) as c FROM atomic_facts "
        "WHERE profile_id = ? AND lifecycle != 'archived'",
        (profile_id,),
    )
    stats["total_evaluated"] = rows[0]["c"] if rows else 0

    total_demoted = (
        stats["demoted_to_warm"]
        + stats["demoted_to_cold"]
        + stats["demoted_to_archive"]
    )
    if total_demoted > 0:
        logger.info(
            "Tier evaluation: %d demoted (warm=%d, cold=%d, archive=%d), %d pinned",
            total_demoted, stats["demoted_to_warm"],
            stats["demoted_to_cold"], stats["demoted_to_archive"],
            stats["pinned_protected"],
        )

    return stats


def promote_on_access_batch(db: DatabaseManager, fact_ids: list[str]) -> int:
    """Batch-promote facts back to 'active' when accessed during recall.

    Single UPDATE for all fact IDs — avoids N sequential writes on hot path.
    """
    if not fact_ids:
        return 0
    return set_fact_lifecycle_zone(
        db,
        fact_ids,
        "active",
        from_atomic=("warm", "cold"),
    )


def promote_on_access(db: DatabaseManager, fact_id: str) -> None:
    """Promote a single fact back to 'active' when accessed during recall.

    Kept for backward compatibility. Prefer promote_on_access_batch.
    """
    set_fact_lifecycle_zone(
        db,
        [fact_id],
        "active",
        from_atomic=("warm", "cold"),
    )


def pin_fact(
    db: DatabaseManager,
    fact_id: str,
    profile_id: str,
    reason: str = "",
) -> bool:
    """Pin a fact to stay in active tier forever.

    Both the pin record and lifecycle update are scoped to profile_id.
    """
    now = datetime.now(UTC).isoformat()
    try:
        db.execute(
            "INSERT OR REPLACE INTO pinned_facts "
            "(fact_id, profile_id, pinned_at, reason) VALUES (?, ?, ?, ?)",
            (fact_id, profile_id, now, reason),
        )
        set_fact_lifecycle_zone(
            db, [fact_id], "active", profile_id=profile_id,
        )
        return True
    except Exception as exc:
        logger.warning("Failed to pin fact %s: %s", fact_id, exc, exc_info=True)
        return False


def unpin_fact(db: DatabaseManager, fact_id: str) -> bool:
    """Unpin a fact, allowing normal tier demotion to resume."""
    try:
        db.execute("DELETE FROM pinned_facts WHERE fact_id = ?", (fact_id,))
        return True
    except Exception as exc:
        logger.warning("Failed to unpin fact %s: %s", fact_id, exc, exc_info=True)
        return False


def get_tier_stats(db: DatabaseManager, profile_id: str = "default") -> dict:
    """Get tier distribution stats for the dashboard."""
    rows = db.execute(
        "SELECT lifecycle, COUNT(*) as cnt FROM atomic_facts "
        "WHERE profile_id = ? GROUP BY lifecycle",
        (profile_id,),
    )
    dist = {r["lifecycle"]: r["cnt"] for r in rows}

    pinned_rows = db.execute(
        "SELECT COUNT(*) as c FROM pinned_facts WHERE profile_id = ?",
        (profile_id,),
    )
    pinned = pinned_rows[0]["c"] if pinned_rows else 0

    total = sum(dist.values())
    return {
        "active": dist.get("active", 0),
        "warm": dist.get("warm", 0),
        "cold": dist.get("cold", 0),
        "archived": dist.get("archived", 0),
        "total": total,
        "pinned": pinned,
        "active_pct": round(dist.get("active", 0) / max(total, 1) * 100, 1),
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_pinned_fact_ids(db: DatabaseManager, profile_id: str) -> frozenset[str]:
    """Load all pinned fact IDs for a profile."""
    try:
        rows = db.execute(
            "SELECT fact_id FROM pinned_facts WHERE profile_id = ?",
            (profile_id,),
        )
        return frozenset(r["fact_id"] for r in rows)
    except Exception as exc:
        logger.warning(
            "Failed to load pinned facts for profile %s: %s",
            profile_id, exc, exc_info=True,
        )
        return frozenset()


def _demote_tier(
    db: DatabaseManager,
    profile_id: str,
    from_tier: str,
    to_tier: str,
    base_days: int,
    pinned_ids: frozenset[str],
    now: datetime,
    dry_run: bool,
) -> int:
    """Demote facts from one tier to the next based on idle time.

    Uses last_accessed_at from fact_retention as the reference date
    (time since last access). Falls back to created_at if no access record.
    Processes in batches of _BATCH_SIZE to prevent OOM at scale.
    """
    demoted_ids: list[str] = []
    offset = 0

    while True:
        rows = db.execute(
            "SELECT af.fact_id, af.access_count, af.importance, "
            "       af.created_at, fr.last_accessed_at "
            "FROM atomic_facts af "
            "LEFT JOIN fact_retention fr ON af.fact_id = fr.fact_id "
            "WHERE af.profile_id = ? AND af.lifecycle = ? "
            "LIMIT ? OFFSET ?",
            (profile_id, from_tier, _BATCH_SIZE, offset),
        )

        if not rows:
            break

        for row in rows:
            fid = row["fact_id"]

            if fid in pinned_ids:
                continue

            effective_days = float(base_days)

            access_count = row["access_count"] or 0
            access_mult = (
                ACCESS_BOOST_MULTIPLIER
                if access_count >= ACCESS_BOOST_THRESHOLD
                else 1.0
            )

            importance = row["importance"] or 0.5
            importance_mult = (
                IMPORTANCE_RESIST_MULTIPLIER
                if importance >= IMPORTANCE_RESIST_THRESHOLD
                else 1.0
            )

            # Cap: use max of the two boosts, not multiplicative
            effective_days *= min(max(access_mult, importance_mult), MAX_COMBINED_MULTIPLIER)

            # Reference: last access time, fallback to created_at
            ref_str = row["last_accessed_at"] or row["created_at"] or ""
            if not ref_str:
                continue

            try:
                ref_date = datetime.fromisoformat(ref_str.replace("Z", "+00:00"))
                if ref_date.tzinfo is None:
                    ref_date = ref_date.replace(tzinfo=UTC)
                idle_time = now - ref_date
                if idle_time < timedelta(days=effective_days):
                    continue
            except (ValueError, TypeError):
                continue

            demoted_ids.append(fid)

        if len(rows) < _BATCH_SIZE:
            break
        offset += _BATCH_SIZE

    if demoted_ids and not dry_run:
        # Batch UPDATE in chunks of 500
        for i in range(0, len(demoted_ids), 500):
            batch = demoted_ids[i:i + 500]
            set_fact_lifecycle_zone(
                db,
                batch,
                to_tier,
                profile_id=profile_id,
                from_atomic=(from_tier,),
            )

    return len(demoted_ids)


# ---------------------------------------------------------------------------
# Hot-path access recording (v3.4.5 — Sprint 1)
# ---------------------------------------------------------------------------

# In-memory counter for batch-flush access recording.
# Thread-safe. Flushed to SQLite every _ACCESS_FLUSH_THRESHOLD records
# or _ACCESS_FLUSH_SECONDS, whichever comes first.
import threading as _threading
from datetime import datetime as _datetime, timedelta as _timedelta

_pending_accesses: dict[str, int] = {}
_access_lock = _threading.Lock()
_last_access_flush = _datetime.now()

_ACCESS_FLUSH_THRESHOLD = 100
_ACCESS_FLUSH_SECONDS = 60


def record_access_batch(db: DatabaseManager, fact_ids: list[str]) -> None:
    """Hot-path access recording for recall. Must be < 1ms average.

    Uses in-memory counter, batch-flushes to SQLite every
    _ACCESS_FLUSH_THRESHOLD accesses or _ACCESS_FLUSH_SECONDS.
    Promotes cold/archived facts on access.

    Thread-safe — called from multiple MCP connections.
    """
    global _last_access_flush
    if not fact_ids:
        return

    with _access_lock:
        for fid in fact_ids:
            _pending_accesses[fid] = _pending_accesses.get(fid, 0) + 1

        now = _datetime.now()
        should_flush = (
            len(_pending_accesses) >= _ACCESS_FLUSH_THRESHOLD
            or (now - _last_access_flush).total_seconds() > _ACCESS_FLUSH_SECONDS
        )
        if should_flush:
            _flush_access_batch(db)
            _last_access_flush = now

    # Promote cold/archived facts on access (F-13: promote to warm)
    try:
        promote_on_access_batch(db, fact_ids)
    except Exception:
        pass


def _flush_access_batch(db: DatabaseManager) -> None:
    """Batch UPDATE access_count_30d and access_count in SQLite."""
    if not _pending_accesses:
        return
    try:
        for fid, count in list(_pending_accesses.items()):
            db.execute(
                "UPDATE atomic_facts SET access_count_30d = access_count_30d + ?, "
                "access_count = access_count + ? WHERE fact_id = ?",
                (count, count, fid),
            )
        _pending_accesses.clear()
    except Exception as exc:
        logger.warning("Failed to flush access counts: %s", exc)


# ---------------------------------------------------------------------------
# 30-day access window reset (F-14)
# ---------------------------------------------------------------------------

def reset_access_count_30d(db: DatabaseManager, profile_id: str = "default") -> int:
    """Reset access_count_30d using actual 30-day window from fact_access_log.

    Called during nightly rebalance. Returns number of facts updated.
    """
    try:
        db.execute(
            "UPDATE atomic_facts SET access_count_30d = COALESCE(("
            "  SELECT COUNT(*) FROM fact_access_log "
            "  WHERE fact_id = atomic_facts.fact_id "
            "  AND accessed_at >= datetime('now', '-30 days')"
            "), 0) WHERE profile_id = ?",
            (profile_id,),
        )
        # Can't get rowcount reliably after subquery UPDATE
        return -1
    except Exception as exc:
        logger.warning("Failed to reset access_count_30d: %s", exc)
        return 0


# ---------------------------------------------------------------------------
# Backend sync stubs (v3.4.5 — Sprint 2+)
# ---------------------------------------------------------------------------

_cozo_backend: object | None = None
_lancedb_backend: object | None = None


def set_backends(cozo: object | None = None, lancedb: object | None = None) -> None:
    """Register CozoDB/LanceDB backends for tier sync. Called by BackendOrchestrator."""
    global _cozo_backend, _lancedb_backend
    _cozo_backend = cozo
    _lancedb_backend = lancedb


def _sync_tiers_to_backends(
    added: list[str], removed: list[str], db: DatabaseManager,
) -> None:
    """Sync tier changes to CozoDB/LanceDB. Non-fatal on failure."""
    if _cozo_backend and hasattr(_cozo_backend, "sync_tier_changes"):
        try:
            _cozo_backend.sync_tier_changes(added=added, removed=removed)
        except Exception as exc:
            logger.warning("CozoDB tier sync failed: %s", exc)

    if _lancedb_backend and hasattr(_lancedb_backend, "bulk_update_tiers_from_sqlite"):
        try:
            _lancedb_backend.bulk_update_tiers_from_sqlite(db.conn)
        except Exception as exc:
            logger.warning("LanceDB tier sync failed: %s", exc)
