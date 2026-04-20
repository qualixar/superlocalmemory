# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory v3.4.22 — F4.A Stage-8 H-18/H-06 fix

"""Reward-gated Ebbinghaus archive.

Flags Ebbinghaus-cold facts as archived *only* when they show no
positive reward in the last 60 days AND are not marked important
(LLD-12 §4). Writes a payload-preserving row to ``memory_archive`` and
updates ``atomic_facts.archive_status='archived'``.

**Never issues DELETE FROM atomic_facts** (SOUL directive, LLD-12 §1 —
memory is sacred across v3.4.22 with 18 000 live users).

JSON1 join helper (``iter_outcomes_for_fact``) replaces the former
``fact_ids_json LIKE '%"<fid>"%'`` pattern so that overlapping fact_id
prefixes no longer collide (H-06).

Contract refs:
  - LLD-12 §4 — reward-gated archive criteria.
  - Stage 8 H-18 — split from monolithic hnsw_dedup.py.
  - Stage 8 H-06 — JSON1 replaces fragile LIKE.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

from superlocalmemory.learning.fact_outcome_joins import (
    has_recent_positive_reward,
)

logger = logging.getLogger(__name__)


REWARD_WINDOW_DAYS: int = 60
ARCHIVE_REWARD_THRESHOLD: float = 0.3

# SEC-L3 — soft cap on ``memory_archive.payload_json`` length. The DDL
# in M011 is ``TEXT NOT NULL`` with no column cap (SQLite has no native
# column-length limit), so a runaway fact could write multi-MB blobs
# and blow past the MASTER-PLAN §2 I4 disk budget. Enforced on the
# write path; oversize payloads are truncated with a breadcrumb in the
# ``reason`` column so an operator can still retrieve the original
# fact via ``atomic_facts`` (archive does not DELETE).
PAYLOAD_JSON_MAX_BYTES: int = 262_144


__all__ = (
    "run_reward_gated_archive",
    "REWARD_WINDOW_DAYS",
    "ARCHIVE_REWARD_THRESHOLD",
    "PAYLOAD_JSON_MAX_BYTES",
)


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_reward_gated_archive(
    memory_db_path: str | Path,
    profile_id: str,
    *,
    candidate_fact_ids: list[str],
) -> list[str]:
    """Archive candidate facts that have no positive reward in 60 days and
    are not flagged important. Returns the list of fact_ids archived.

    LLD-12 §1 hard invariant: this function NEVER issues
    ``DELETE FROM atomic_facts``. It UPDATEs archive_status + writes a
    payload snapshot to ``memory_archive``.
    """
    if not candidate_fact_ids:
        return []

    archived: list[str] = []
    conn = sqlite3.connect(str(memory_db_path), timeout=10.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=2000")

    try:
        placeholders = ",".join("?" for _ in candidate_fact_ids)
        rows = conn.execute(
            f"SELECT fact_id, content, canonical_entities_json, importance, "
            f"       confidence, embedding, created_at "
            f"FROM atomic_facts "
            f"WHERE profile_id = ? AND fact_id IN ({placeholders}) "
            f"  AND (archive_status IS NULL OR archive_status = 'live')",
            (profile_id, *candidate_fact_ids),
        ).fetchall()

        # H-12/H-P-02: compute the archivable set BEFORE acquiring the
        # writer lock. Previously ``BEGIN IMMEDIATE`` wrapped the full
        # per-candidate reward-lookup loop — holding RESERVED for the
        # entire scan starved concurrent ``record_recall`` writers out
        # with SQLITE_BUSY after their 50 ms busy_timeout. Splitting the
        # read phase keeps the writer lock held only for the bulk
        # INSERT/UPDATE pass.
        to_archive: list[dict] = []
        for row in rows:
            fid = row["fact_id"]
            # 1. Important flag skip (LLD-12 §4 criterion 3).
            if float(row["importance"] or 0.0) >= 1.0:
                continue
            # 2. Recent positive reward skip (criterion 2).
            #    H-06 fix — JSON1 equality join via helper.
            if has_recent_positive_reward(
                conn, profile_id, fid,
                min_reward=ARCHIVE_REWARD_THRESHOLD,
                window_days=REWARD_WINDOW_DAYS,
            ):
                continue
            to_archive.append({
                "fid": fid,
                "content": row["content"],
                "canonical_entities_json": row["canonical_entities_json"],
                "importance": row["importance"],
                "confidence": row["confidence"],
                "embedding": row["embedding"],
                "created_at": row["created_at"],
            })

        if not to_archive:
            return []

        conn.execute("BEGIN IMMEDIATE")
        # S9-SKEP-07: re-verify reward under RESERVED lock. Between the
        # read-only reward scan above and this BEGIN IMMEDIATE another
        # writer may have inserted a positive reward row ("user liked
        # the memory we are about to archive"). With the writer lock
        # held we now know no further inserts are landing; one last
        # check per entry catches everything that raced in.
        verified: list[dict] = []
        for entry in to_archive:
            if has_recent_positive_reward(
                conn, profile_id, entry["fid"],
                min_reward=ARCHIVE_REWARD_THRESHOLD,
                window_days=REWARD_WINDOW_DAYS,
            ):
                continue
            verified.append(entry)
        to_archive = verified
        if not to_archive:
            conn.execute("COMMIT")
            return []
        for entry in to_archive:
            fid = entry["fid"]
            payload = {
                "fact_id": fid,
                "content": entry["content"],
                "canonical_entities_json": entry["canonical_entities_json"],
                "importance": entry["importance"],
                "confidence": entry["confidence"],
                "embedding": entry["embedding"],
                "created_at": entry["created_at"],
            }
            # SEC-L3 — cap payload_json at 256 KB. Oversize blobs are
            # replaced with a minimal stub + ``truncated`` reason so the
            # archive row stays within the I4 disk budget while still
            # pointing back to the original ``fact_id`` in atomic_facts.
            payload_str = json.dumps(payload)
            reason = "reward_gated_ebbinghaus"
            if len(payload_str.encode("utf-8")) > PAYLOAD_JSON_MAX_BYTES:
                payload_str = json.dumps({
                    "fact_id": fid,
                    "truncated": True,
                    "original_bytes": len(payload_str.encode("utf-8")),
                })
                reason = "reward_gated_ebbinghaus_truncated"
                logger.warning(
                    "memory_archive payload >%d bytes for fact_id=%s; "
                    "truncated to stub", PAYLOAD_JSON_MAX_BYTES, fid,
                )
            conn.execute(
                "INSERT INTO memory_archive "
                "(archive_id, fact_id, profile_id, payload_json, "
                " archived_at, reason) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    str(uuid.uuid4()),
                    fid,
                    profile_id,
                    payload_str,
                    _iso_now(),
                    reason,
                ),
            )
            conn.execute(
                "UPDATE atomic_facts "
                "SET archive_status='archived', "
                "    archive_reason='reward_gated_ebbinghaus' "
                "WHERE fact_id=? "
                "  AND (archive_status IS NULL OR archive_status='live')",
                (fid,),
            )
            archived.append(fid)

        conn.commit()
    except sqlite3.Error as exc:
        conn.rollback()
        logger.warning("run_reward_gated_archive rollback: %s", exc)
    finally:
        conn.close()

    return archived
