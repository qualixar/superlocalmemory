# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Bring back memories the hours-clock ELC step archived (M066).

WHICH MEMORIES
Archived (``atomic_facts.lifecycle = 'archived'`` with retention zone
``archive``, which is exactly what the ELC step wrote) and with nothing on record
that makes the archive a decision. Each of these keeps a memory archived:

* the user archived or forgot it, or it was merged (``archive_status``,
  ``archive_reason``, ``merged_into``), or it sits in ``memory_archive``;
* an erasure is in flight or done for it (``projection_tombstones``,
  ``erasure_receipts``);
* a newer fact replaced it (``consolidation_log`` supersede/update, or an
  expired ``fact_temporal_validity`` row);
* a compliance retention rule that archives covers its age;
* the decay pass itself leaves it alone: pinned, rewarded, a Core Memory
  source, or a consolidation source with no use since (``metadata_query``).

WHICH ZONE
The decay pass's own computation, nothing new: ``metadata_query`` inputs,
``attach_canonical_trust``, ``EbbinghausCurve.batch_compute_retention`` (the
store-scaled clock, GitHub #136) and the user's inactivity policy if one is set
(``apply_configured_policy``). A memory it calls active, warm or cold goes back
to that zone in both the authority (``fact_retention``) and the mirror
(``atomic_facts``); one it calls archive or forgotten stays where it is.

Two defects archived without a record, and both are repaired here: the ELC step
on the hours clock (#266), and the maintenance Langevin pass, which until 4.2.8
decided tiers on the default forgetting settings instead of the person's own
(on the author's store, 630 memories archived that the person's settings keep).
A memory the fixed Langevin pass would still archive stays archived, so the
next pass never undoes a restore.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from superlocalmemory.storage.data_repair_undo import UndoWriter

logger = logging.getLogger(__name__)

RESTORABLE_ZONES = frozenset({"active", "warm", "cold"})
_SUPERSEDING_ACTIONS = ("supersede", "update")
_ARCHIVING_RULES = ("archive", "tombstone")


@dataclass(frozen=True)
class RepairResult:
    checked: int
    restored: int
    by_zone: dict[str, int] = field(default_factory=dict)


class _Db:
    """The read surface the decay helpers expect of ``DatabaseManager``."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def execute(self, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
        cursor = self._conn.cursor()
        cursor.row_factory = sqlite3.Row
        return cursor.execute(sql, params).fetchall()

    def get_config(self, key: str) -> str | None:
        if not _has(self._conn, "config"):
            return None
        rows = self.execute("SELECT value FROM config WHERE key = ?", (key,))
        return str(rows[0]["value"]) if rows else None


def _has(conn: sqlite3.Connection, table: str) -> bool:
    return conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
                        (table,)).fetchone() is not None


def _cols(conn: sqlite3.Connection, table: str) -> set[str]:
    return {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}


def _decision_clauses(conn: sqlite3.Connection) -> list[str]:
    """SQL that keeps an archive which was somebody's decision."""
    fact_cols = _cols(conn, "atomic_facts")
    clauses = []
    if "archive_status" in fact_cols:
        clauses.append("COALESCE(f.archive_status, 'live') = 'live'")
    if "archive_reason" in fact_cols:
        clauses.append("COALESCE(f.archive_reason, '') = ''")
    if "merged_into" in fact_cols:
        clauses.append("f.merged_into IS NULL")
    absent = {
        "memory_archive": "x.fact_id = f.fact_id",
        "projection_tombstones": "x.fact_id = f.fact_id",
        "erasure_receipts": "x.subject_type = 'fact' AND x.subject_id = f.fact_id",
        "consolidation_log": ("x.existing_fact_id = f.fact_id AND x.action_type IN "
                              f"{_SUPERSEDING_ACTIONS!r}"),
        "fact_temporal_validity": "x.fact_id = f.fact_id AND x.system_expired_at IS NOT NULL",
    }
    clauses += [f"NOT EXISTS (SELECT 1 FROM {table} x WHERE {where})"
                for table, where in absent.items() if _has(conn, table)]
    return clauses


def _candidates(conn: sqlite3.Connection) -> list[tuple[str, str, str]]:
    """``(fact_id, profile_id, created_at)`` of archives with no decision on record."""
    if not _has(conn, "fact_retention"):
        return []
    where = " AND ".join(["f.lifecycle = 'archived'", "r.lifecycle_zone = 'archive'",
                          *_decision_clauses(conn)])
    return [tuple(r) for r in conn.execute(
        "SELECT f.fact_id, f.profile_id, f.created_at FROM atomic_facts f "  # noqa: S608
        f"JOIN fact_retention r ON r.fact_id = f.fact_id WHERE {where} ORDER BY f.fact_id")]


def _rule_age_days(conn: sqlite3.Connection, profile_id: str) -> float | None:
    """Youngest age a compliance rule archives at, for this profile."""
    if not _has(conn, "retention_rules"):
        return None
    marks = ",".join("?" * len(_ARCHIVING_RULES))
    row = conn.execute(f"SELECT MIN(days) FROM retention_rules WHERE profile_id = ? "  # noqa: S608
                       f"AND action IN ({marks})", (profile_id, *_ARCHIVING_RULES)).fetchone()
    return None if row is None or row[0] is None else float(row[0])


def _age_days(created_at: object) -> float:
    try:
        created = datetime.fromisoformat(str(created_at).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return float("inf")  # unknown age: treat as covered by any rule
    created = created if created.tzinfo else created.replace(tzinfo=UTC)
    return (datetime.now(UTC) - created).total_seconds() / 86400.0


@dataclass(frozen=True)
class _Settings:
    forgetting: Any
    langevin_on: bool


def _settings(conn: sqlite3.Connection) -> _Settings:
    """The user's settings when this store is the configured one, else defaults."""
    from superlocalmemory.core.config import ForgettingConfig, MathConfig, SLMConfig

    row = conn.execute("PRAGMA database_list").fetchone()
    forgetting, math_cfg = ForgettingConfig(), MathConfig()
    try:
        config = SLMConfig.load()
        configured = config.db_path
        if (row and row[2] and configured is not None
                and Path(configured).resolve() == Path(row[2]).resolve()):
            forgetting, math_cfg = config.forgetting, config.math
    except Exception as exc:  # noqa: BLE001 - a broken config must not block a repair
        logger.info("M066: user config unreadable (%s); using default settings", exc)
    return _Settings(forgetting, bool(math_cfg.langevin_persist_positions))


def _langevin_archives(conn: sqlite3.Connection, fact_ids: list[str],
                       settings: _Settings) -> set[str]:
    """Facts the maintenance Langevin pass would archive again, on the person's
    own settings.

    Since 4.2.8 that pass decides the tier with the decay pass's function and
    the person's ``forgetting`` settings (``langevin_tier.langevin_lifecycle``);
    before, it used the defaults and archived memories those settings keep --
    those are exactly what this repair brings back. A memory the fixed pass
    would still archive (its inputs are age, use and importance) is left
    where it is, so a restore is never undone by the next pass.
    """
    if not settings.langevin_on or not fact_ids:
        return set()
    from superlocalmemory.core.langevin_tier import _age_days as langevin_age
    from superlocalmemory.core.langevin_tier import langevin_lifecycle
    from superlocalmemory.math.ebbinghaus import EbbinghausCurve

    curve = EbbinghausCurve(settings.forgetting)
    archived = set()
    for fact_id in fact_ids:
        access, importance, created = conn.execute(
            "SELECT access_count, importance, created_at FROM atomic_facts WHERE fact_id = ?",
            (fact_id,)).fetchone()
        if langevin_lifecycle(int(access or 0), langevin_age(created),
                              float(importance or 0.0), curve) == "archived":
            archived.add(fact_id)
    return archived


def _zones(conn: sqlite3.Connection, profile_id: str, fact_ids: list[str],
           curve: Any) -> dict[str, str]:
    """The decay pass's zone for each fact it would consider (protected ones drop out)."""
    from superlocalmemory.learning.forgetting_metadata import (
        attach_canonical_trust,
        metadata_query,
    )
    from superlocalmemory.learning.forgetting_scheduler import (
        _CONSOLIDATED_SOURCE_SQL,
        _INTERNAL_ACCESS_SQL,
    )
    from superlocalmemory.learning.lifecycle_policy import apply_configured_policy

    db = _Db(conn)
    consolidated = _CONSOLIDATED_SOURCE_SQL if _has(conn, "ccq_consolidated_blocks") else ""
    # The decay pass's own query, narrowed to the candidates: its WHERE ends in
    # AND-clauses, so one more keeps every protection it applies and spares the
    # per-fact consolidation and reward checks for the rest of the store.
    sql = (metadata_query(db, _INTERNAL_ACCESS_SQL, consolidated)
           + " AND f.fact_id IN (SELECT value FROM json_each(?))")
    params = (profile_id,) * (7 if consolidated else 5) + (json.dumps(fact_ids),)
    rows = [dict(r) for r in db.execute(sql, params)]
    results = apply_configured_policy(
        db, curve.batch_compute_retention(attach_canonical_trust(db, rows, profile_id)),
        profile_id)
    return {r["fact_id"]: str(r["zone"]) for r in results}


def _restore(conn: sqlite3.Connection, writer: UndoWriter, fact: tuple[str, str],
             zone: str) -> bool:
    """Zone back in the authority and the mirror, each with its undo record.

    Both writes are guarded by the archived state the plan saw, so a memory
    that changed since (a recall, a user action) is skipped, never overwritten.
    """
    from superlocalmemory.core.lifecycle_state import atomic_lifecycle_for
    from superlocalmemory.storage.fabricated_dates_repair import _Rows
    from superlocalmemory.storage.projection_outbox import enqueue

    fact_id, profile_id = fact
    still = conn.execute(
        "SELECT 1 FROM atomic_facts f JOIN fact_retention r USING (fact_id) WHERE "
        "f.fact_id = ? AND f.lifecycle = 'archived' AND r.lifecycle_zone = 'archive'",
        (fact_id,)).fetchone()
    if still is None:
        return False
    mirror = atomic_lifecycle_for(zone)
    writer.record("fact_retention", fact_id, fact_id, profile_id,
                  {"lifecycle_zone": "archive"}, {"lifecycle_zone": zone})
    conn.execute("UPDATE fact_retention SET lifecycle_zone = ?, last_computed_at = "
                 "datetime('now') WHERE fact_id = ?", (zone, fact_id))
    writer.record("atomic_facts", fact_id, fact_id, profile_id,
                  {"lifecycle": "archived"}, {"lifecycle": mirror})
    conn.execute("UPDATE atomic_facts SET lifecycle = ? WHERE fact_id = ?", (mirror, fact_id))
    enqueue(_Rows(conn), fact_id, profile_id)  # lifecycle is projected; no-op without a queue
    return True


def _plan(conn: sqlite3.Connection, settings: _Settings,
          ) -> tuple[int, list[tuple[str, str, str]]]:
    """``(checked, [(fact_id, profile_id, zone)])``. Reads only."""
    from superlocalmemory.math.ebbinghaus import EbbinghausCurve

    curve = EbbinghausCurve(settings.forgetting)
    candidates = _candidates(conn)
    by_profile: dict[str, list[tuple[str, str]]] = {}
    for fact_id, profile_id, created_at in candidates:
        by_profile.setdefault(profile_id, []).append((fact_id, created_at))
    restores = []
    for profile_id, facts in by_profile.items():
        rule_age = _rule_age_days(conn, profile_id)
        eligible = [fid for fid, created in facts
                    if rule_age is None or _age_days(created) <= rule_age]
        zones = _zones(conn, profile_id, eligible, curve) if eligible else {}
        live = [fid for fid in eligible if zones.get(fid) in RESTORABLE_ZONES]
        langevin = _langevin_archives(conn, live, settings)
        restores += [(fid, profile_id, zones[fid]) for fid in live if fid not in langevin]
    return len(candidates), restores


def _count(zones: list[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for zone in zones:
        out[zone] = out.get(zone, 0) + 1
    return out


def apply(conn: sqlite3.Connection, repair: str) -> RepairResult:
    """Plan without the write lock (the decay inputs take a full scan), then
    restore inside one short ``BEGIN IMMEDIATE`` transaction."""
    checked, restores = _plan(conn, _settings(conn))
    if not restores:
        return RepairResult(checked, 0, {})
    done: list[str] = []
    conn.execute("BEGIN IMMEDIATE")
    try:
        writer = UndoWriter(conn, repair)
        for fact_id, profile_id, zone in restores:
            if _restore(conn, writer, (fact_id, profile_id), zone):
                done.append(zone)
        conn.execute("COMMIT")
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    return RepairResult(checked, len(done), _count(done))


def simulate(conn: sqlite3.Connection) -> RepairResult:
    """What :func:`apply` would restore, without writing anything."""
    checked, restores = _plan(conn, _settings(conn))
    return RepairResult(checked, len(restores), _count([z for _f, _p, z in restores]))


__all__ = ["RESTORABLE_ZONES", "RepairResult", "apply", "simulate"]
