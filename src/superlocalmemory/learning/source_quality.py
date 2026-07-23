#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Part of Qualixar | Author: Varun Pratap Bhardwaj (qualixar.com | varunpratap.com)
"""
SourceQualityScorer -- Beta-binomial source quality scoring for V3 learning.

Each memory source (agent, URL, manual, etc.) gets a quality score based on
how often its memories are confirmed vs contradicted or ignored.

Scoring (Beta-Binomial with Laplace smoothing):
    quality = (alpha + positives) / (alpha + beta + total)

    With alpha=1, beta=1 (uniform prior):
        - New source, 0 evidence  -> 1/2 = 0.50
        - 8 positive out of 10   -> 9/12 = 0.75
        - 1 positive out of 10   -> 2/12 = 0.17

Storage:
    Uses direct sqlite3 with a self-contained ``source_quality`` table.
    NOT coupled to V3 DatabaseManager.
"""

from __future__ import annotations

import json
import logging
import math
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger("superlocalmemory.learning.source_quality")

# Beta-Binomial prior (Laplace / uniform)
_ALPHA = 1.0
_BETA = 1.0

# Default quality for unknown sources = alpha / (alpha + beta)
DEFAULT_QUALITY = _ALPHA / (_ALPHA + _BETA)  # 0.5

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS source_quality (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    profile_id   TEXT    NOT NULL,
    source_id    TEXT    NOT NULL,
    alpha        REAL    NOT NULL DEFAULT 1.0,
    beta         REAL    NOT NULL DEFAULT 1.0,
    updated_at   TEXT    NOT NULL
)
"""

_CREATE_UNIQUE = """
CREATE UNIQUE INDEX IF NOT EXISTS idx_sq_profile_source
    ON source_quality (profile_id, source_id)
"""

_CREATE_OBSERVATIONS = """
CREATE TABLE IF NOT EXISTS source_quality_observations (
    profile_id   TEXT NOT NULL,
    outcome_id   TEXT NOT NULL,
    source_id    TEXT NOT NULL,
    reward       REAL NOT NULL,
    observed_at  TEXT NOT NULL,
    PRIMARY KEY (profile_id, outcome_id, source_id)
)
"""

_CREATE_REPAIR_STATE = """
CREATE TABLE IF NOT EXISTS source_quality_repair_state (
    profile_id   TEXT PRIMARY KEY,
    last_rowid   INTEGER NOT NULL DEFAULT 0,
    last_settled_at TEXT NOT NULL DEFAULT '',
    last_outcome_id TEXT NOT NULL DEFAULT '',
    updated_at   TEXT NOT NULL
)
"""

_MAX_FACTS_PER_OUTCOME = 100
_MAX_SOURCES_PER_OUTCOME = 100
_PROVENANCE_QUERY_CHUNK = 500


class SourceQualityRepairUnavailable(RuntimeError):
    """A repair read failed transiently and must not be interpreted as EOF."""


def _utcnow_iso() -> str:
    """Return current UTC time as ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


class SourceQualityScorer:
    """
    Beta-binomial source quality scoring.

    Maintains per-(profile, source) alpha/beta parameters.  Positive
    outcomes increment alpha; negative outcomes increment beta.
    Quality = alpha / (alpha + beta).

    Args:
        db_path: Path to the sqlite3 database file.
    """

    def __init__(self, db_path: Path) -> None:
        self._db_path = Path(db_path)
        self._lock = threading.Lock()
        self._ensure_schema()

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _ensure_schema(self) -> None:
        conn = self._connect()
        try:
            # Separate scorer instances can be constructed concurrently during
            # first startup (background history repair + outcome settlement).
            # Serialize the read/ALTER sequence at SQLite's transaction
            # boundary so two processes cannot both observe a legacy column as
            # missing and race into ``duplicate column name``.
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(_CREATE_TABLE)
            conn.execute(_CREATE_UNIQUE)
            conn.execute(_CREATE_OBSERVATIONS)
            conn.execute(_CREATE_REPAIR_STATE)
            repair_columns = {
                str(row["name"])
                for row in conn.execute(
                    "PRAGMA table_info(source_quality_repair_state)"
                ).fetchall()
            }
            if "last_settled_at" not in repair_columns:
                conn.execute(
                    "ALTER TABLE source_quality_repair_state "
                    "ADD COLUMN last_settled_at TEXT NOT NULL DEFAULT ''"
                )
            if "last_outcome_id" not in repair_columns:
                conn.execute(
                    "ALTER TABLE source_quality_repair_state "
                    "ADD COLUMN last_outcome_id TEXT NOT NULL DEFAULT ''"
                )
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path), timeout=10)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        conn.row_factory = sqlite3.Row
        return conn

    # ------------------------------------------------------------------
    # Public API: record outcome
    # ------------------------------------------------------------------

    def record_outcome(
        self,
        profile_id: str,
        source_id: str,
        outcome: str,
    ) -> None:
        """
        Record an observation for a source.

        Args:
            profile_id: Profile context.
            source_id:  Identifier of the source (agent name, URL, etc.).
            outcome:    ``"positive"`` or ``"negative"``.

        Raises:
            ValueError: If outcome is not ``"positive"`` or ``"negative"``.
        """
        if outcome not in ("positive", "negative"):
            raise ValueError(
                f"outcome must be 'positive' or 'negative', got {outcome!r}"
            )
        if not profile_id or not source_id:
            return

        now = _utcnow_iso()

        with self._lock:
            conn = self._connect()
            try:
                # Ensure row exists (INSERT OR IGNORE with defaults)
                conn.execute(
                    "INSERT OR IGNORE INTO source_quality "
                    "(profile_id, source_id, alpha, beta, updated_at) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (profile_id, source_id, _ALPHA, _BETA, now),
                )

                # Update the appropriate parameter
                if outcome == "positive":
                    conn.execute(
                        "UPDATE source_quality "
                        "SET alpha = alpha + 1.0, updated_at = ? "
                        "WHERE profile_id = ? AND source_id = ?",
                        (now, profile_id, source_id),
                    )
                else:
                    conn.execute(
                        "UPDATE source_quality "
                        "SET beta = beta + 1.0, updated_at = ? "
                        "WHERE profile_id = ? AND source_id = ?",
                        (now, profile_id, source_id),
                    )

                conn.commit()
            finally:
                conn.close()

    def record_reward(
        self,
        profile_id: str,
        outcome_id: str,
        source_ids: list[str],
        reward: float,
    ) -> int:
        """Apply one fractional Beta observation per unique source.

        The observation ledger makes retries idempotent. A reward of 0.8
        contributes ``+0.8`` to alpha and ``+0.2`` to beta rather than
        inventing a binary success label.
        """
        return self.record_rewards([
            (profile_id, outcome_id, source_ids, reward),
        ])

    def record_rewards(
        self,
        observations: list[tuple[str, str, list[str], float]],
    ) -> int:
        """Batch bounded reward observations in one learning-DB transaction."""
        inserted = 0
        now = _utcnow_iso()
        with self._lock:
            conn = self._connect()
            try:
                conn.execute("BEGIN IMMEDIATE")
                for profile_id, outcome_id, sources, raw_reward in observations:
                    if not profile_id or not outcome_id:
                        continue
                    numeric_reward = float(raw_reward)
                    if not math.isfinite(numeric_reward):
                        continue
                    reward = max(0.0, min(1.0, numeric_reward))
                    for source_id in sorted(set(sources))[
                        :_MAX_SOURCES_PER_OUTCOME
                    ]:
                        if not source_id:
                            continue
                        cursor = conn.execute(
                            "INSERT OR IGNORE INTO source_quality_observations "
                            "(profile_id, outcome_id, source_id, reward, observed_at) "
                            "VALUES (?, ?, ?, ?, ?)",
                            (profile_id, outcome_id, source_id, reward, now),
                        )
                        if cursor.rowcount != 1:
                            continue
                        conn.execute(
                            "INSERT OR IGNORE INTO source_quality "
                            "(profile_id, source_id, alpha, beta, updated_at) "
                            "VALUES (?, ?, ?, ?, ?)",
                            (profile_id, source_id, _ALPHA, _BETA, now),
                        )
                        conn.execute(
                            "UPDATE source_quality SET "
                            "alpha = alpha + ?, beta = beta + ?, updated_at = ? "
                            "WHERE profile_id = ? AND source_id = ?",
                            (
                                reward, 1.0 - reward, now,
                                profile_id, source_id,
                            ),
                        )
                        inserted += 1
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()
        return inserted

    def get_repair_cursor(self, profile_id: str) -> int:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT last_rowid FROM source_quality_repair_state "
                "WHERE profile_id = ?",
                (profile_id,),
            ).fetchone()
            return int(row["last_rowid"] or 0) if row else 0
        finally:
            conn.close()

    def set_repair_cursor(self, profile_id: str, rowid: int) -> None:
        now = _utcnow_iso()
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    "INSERT INTO source_quality_repair_state "
                    "(profile_id, last_rowid, updated_at) VALUES (?, ?, ?) "
                    "ON CONFLICT(profile_id) DO UPDATE SET "
                    "last_rowid = excluded.last_rowid, "
                    "updated_at = excluded.updated_at",
                    (profile_id, int(rowid), now),
                )
                conn.commit()
            finally:
                conn.close()

    def get_repair_position(self, profile_id: str) -> tuple[str, str]:
        """Return the durable settlement-order cursor for historical repair."""
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT last_settled_at, last_outcome_id "
                "FROM source_quality_repair_state WHERE profile_id = ?",
                (profile_id,),
            ).fetchone()
            if row is None:
                return ("", "")
            return (
                str(row["last_settled_at"] or ""),
                str(row["last_outcome_id"] or ""),
            )
        finally:
            conn.close()

    def set_repair_position(
        self,
        profile_id: str,
        settled_at: str,
        outcome_id: str,
    ) -> None:
        """Advance repair by settlement order, not immutable insertion rowid."""
        now = _utcnow_iso()
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    "INSERT INTO source_quality_repair_state "
                    "(profile_id,last_rowid,last_settled_at,last_outcome_id,"
                    "updated_at) VALUES (?,0,?,?,?) "
                    "ON CONFLICT(profile_id) DO UPDATE SET "
                    "last_settled_at=excluded.last_settled_at,"
                    "last_outcome_id=excluded.last_outcome_id,"
                    "updated_at=excluded.updated_at",
                    (profile_id, str(settled_at), str(outcome_id), now),
                )
                conn.commit()
            finally:
                conn.close()

    # ------------------------------------------------------------------
    # Public API: read quality
    # ------------------------------------------------------------------

    def get_quality(self, profile_id: str, source_id: str) -> float:
        """
        Get the quality score for a specific source.

        Returns the Beta-binomial posterior mean:
            quality = alpha / (alpha + beta)

        If the source has never been observed, returns the prior
        mean (0.5).

        Args:
            profile_id: Profile context.
            source_id:  Source identifier.

        Returns:
            Quality score in [0.0, 1.0].
        """
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT alpha, beta FROM source_quality "
                "WHERE profile_id = ? AND source_id = ?",
                (profile_id, source_id),
            ).fetchone()

            if row is None:
                return DEFAULT_QUALITY

            alpha = float(row["alpha"])
            beta = float(row["beta"])
            denom = alpha + beta
            if denom <= 0:
                return DEFAULT_QUALITY
            return alpha / denom
        finally:
            conn.close()

    def get_all_qualities(self, profile_id: str) -> Dict[str, float]:
        """
        Get quality scores for all sources observed under a profile.

        Args:
            profile_id: Profile context.

        Returns:
            Dict mapping source_id -> quality score (0.0 to 1.0).
        """
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT source_id, alpha, beta FROM source_quality "
                "WHERE profile_id = ?",
                (profile_id,),
            ).fetchall()

            result: Dict[str, float] = {}
            for r in rows:
                alpha = float(r["alpha"])
                beta = float(r["beta"])
                denom = alpha + beta
                score = alpha / denom if denom > 0 else DEFAULT_QUALITY
                result[r["source_id"]] = score
            return result
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Public API: diagnostics
    # ------------------------------------------------------------------

    def get_detailed(
        self, profile_id: str, source_id: str,
    ) -> Dict[str, Any]:
        """
        Get detailed quality information for a single source.

        Returns:
            Dict with alpha, beta, quality, updated_at.
            Returns defaults if the source has not been observed.
        """
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT alpha, beta, updated_at FROM source_quality "
                "WHERE profile_id = ? AND source_id = ?",
                (profile_id, source_id),
            ).fetchone()

            if row is None:
                return {
                    "alpha": _ALPHA,
                    "beta": _BETA,
                    "quality": DEFAULT_QUALITY,
                    "updated_at": None,
                }

            alpha = float(row["alpha"])
            beta = float(row["beta"])
            denom = alpha + beta
            return {
                "alpha": alpha,
                "beta": beta,
                "quality": alpha / denom if denom > 0 else DEFAULT_QUALITY,
                "updated_at": row["updated_at"],
            }
        finally:
            conn.close()

    def get_all_detailed(self, profile_id: str) -> Dict[str, Dict[str, Any]]:
        """
        Get detailed quality data for all sources under a profile.

        Returns:
            Dict mapping source_id -> detail dict.
        """
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT source_id, alpha, beta, updated_at "
                "FROM source_quality WHERE profile_id = ? "
                "ORDER BY (alpha / (alpha + beta)) DESC",
                (profile_id,),
            ).fetchall()

            result: Dict[str, Dict[str, Any]] = {}
            for r in rows:
                alpha = float(r["alpha"])
                beta = float(r["beta"])
                denom = alpha + beta
                result[r["source_id"]] = {
                    "alpha": alpha,
                    "beta": beta,
                    "quality": alpha / denom if denom > 0 else DEFAULT_QUALITY,
                    "updated_at": r["updated_at"],
                }
            return result
        finally:
            conn.close()


def _source_key(row: sqlite3.Row) -> str:
    source_type = str(row["source_type"] or "").strip()[:100]
    actor = str(row["created_by"] or "").strip()
    operation_or_legacy_source = str(row["source_id"] or "").strip()
    # Canonical ingestion stores its unique operation UUID in source_id for
    # lineage/idempotency and the stable trusted client in created_by. Quality
    # must aggregate by the stable actor; operation IDs would create one
    # single-observation "source" per remember call. Legacy provenance often
    # has no actor, so retain its established source_id fallback.
    identifier = (
        actor
        if actor and actor.lower() != "unknown"
        else operation_or_legacy_source
    )[:100]
    if source_type and identifier:
        return f"{source_type}:{identifier}"
    if identifier:
        return f"source:{identifier}"
    return source_type


def _load_source_map(
    memory_db_path: Path,
    profile_id: str,
    fact_ids: list[str],
    *,
    strict: bool = False,
) -> dict[str, set[str]]:
    """Read only the provenance needed by this bounded reward batch."""
    unique_facts = list(dict.fromkeys(str(fid) for fid in fact_ids if fid))
    if not unique_facts or not Path(memory_db_path).exists():
        return {}
    result: dict[str, set[str]] = {}
    try:
        conn = sqlite3.connect(
            f"file:{Path(memory_db_path)}?mode=ro", uri=True, timeout=1.0,
        )
        conn.row_factory = sqlite3.Row
        try:
            columns = {
                str(row["name"])
                for row in conn.execute("PRAGMA table_info(provenance)")
            }
            required = {
                "profile_id", "fact_id", "source_type",
                "source_id", "created_by",
            }
            if not required.issubset(columns):
                return {}
            for start in range(0, len(unique_facts), _PROVENANCE_QUERY_CHUNK):
                chunk = unique_facts[start:start + _PROVENANCE_QUERY_CHUNK]
                placeholders = ",".join("?" for _ in chunk)
                rows = conn.execute(
                    "SELECT DISTINCT fact_id, source_type, source_id, created_by "
                    "FROM provenance WHERE profile_id = ? "
                    f"AND fact_id IN ({placeholders})",
                    (profile_id, *chunk),
                ).fetchall()
                for row in rows:
                    key = _source_key(row)
                    if key:
                        result.setdefault(str(row["fact_id"]), set()).add(key)
        finally:
            conn.close()
    except sqlite3.Error as exc:
        logger.debug("source provenance unavailable: %s", exc)
        if strict:
            raise SourceQualityRepairUnavailable(
                "source provenance temporarily unavailable",
            ) from exc
        return {}
    return result


def update_source_quality_for_reward(
    *,
    memory_db_path: Path,
    learning_db_path: Path,
    profile_id: str,
    outcome_id: str,
    fact_ids: list[str],
    reward: float,
) -> int:
    """Fail-soft online bridge from a finalized reward to real provenance."""
    return update_source_quality_for_reward_batch(
        memory_db_path=memory_db_path,
        learning_db_path=learning_db_path,
        rewards=[(profile_id, outcome_id, fact_ids, reward)],
    )


def update_source_quality_for_reward_batch(
    *,
    memory_db_path: Path,
    learning_db_path: Path,
    rewards: list[tuple[str, str, list[str], float]],
) -> int:
    """Fail-soft bounded bridge for worker-finalized reward batches."""
    try:
        bounded_rewards = rewards[:1000]
        normalized = []
        for profile_id, outcome_id, fact_ids, reward in bounded_rewards:
            bounded_facts = list(dict.fromkeys(fact_ids))[
                :_MAX_FACTS_PER_OUTCOME
            ]
            normalized.append((
                profile_id, outcome_id, bounded_facts, float(reward),
            ))
        if not normalized:
            return 0
        # Each reward batch is profile-homogeneous in current callers. Split
        # defensively so provenance can never cross a profile boundary.
        by_profile: dict[str, list[tuple[str, list[str], float]]] = {}
        for profile_id, outcome_id, fact_ids, reward in normalized:
            by_profile.setdefault(profile_id, []).append(
                (outcome_id, fact_ids, reward),
            )
        scorer = SourceQualityScorer(Path(learning_db_path))
        observations = []
        for profile_id, profile_rewards in by_profile.items():
            profile_facts = [
                fact_id
                for _, fact_ids, _ in profile_rewards
                for fact_id in fact_ids
            ]
            source_map = _load_source_map(
                Path(memory_db_path), profile_id, profile_facts,
            )
            for outcome_id, fact_ids, reward in profile_rewards:
                source_ids = sorted({
                    source
                    for fact_id in fact_ids
                    for source in source_map.get(fact_id, set())
                })[:_MAX_SOURCES_PER_OUTCOME]
                observations.append((
                    profile_id, outcome_id, source_ids, reward,
                ))
        return scorer.record_rewards(observations)
    except (OSError, sqlite3.Error, TypeError, ValueError) as exc:
        logger.debug("source-quality reward update skipped: %s", exc)
        return 0


def enumerate_source_quality_repair_profiles(
    memory_db_path: Path,
) -> list[str]:
    """Return profiles with settled numeric outcomes eligible for repair."""
    path = Path(memory_db_path)
    if not path.exists():
        return []
    try:
        conn = sqlite3.connect(
            f"file:{path}?mode=ro", uri=True, timeout=1.0,
        )
        conn.row_factory = sqlite3.Row
        try:
            columns = {
                str(row["name"])
                for row in conn.execute("PRAGMA table_info(action_outcomes)")
            }
            required = {"profile_id", "reward", "settled"}
            if not required.issubset(columns):
                return []
            rows = conn.execute(
                "SELECT DISTINCT profile_id FROM action_outcomes "
                "WHERE settled = 1 AND reward IS NOT NULL "
                "AND typeof(reward) IN ('integer', 'real') "
                "AND profile_id IS NOT NULL AND profile_id != '' "
                "ORDER BY profile_id ASC",
            ).fetchall()
            return [str(row["profile_id"]) for row in rows]
        finally:
            conn.close()
    except sqlite3.Error as exc:
        logger.debug("source-quality profile enumeration unavailable: %s", exc)
        raise SourceQualityRepairUnavailable(
            "repair profile enumeration temporarily unavailable",
        ) from exc


def _parse_repair_rows(
    rows: list[sqlite3.Row],
) -> tuple[list[tuple[sqlite3.Row, list[str]]], list[str]]:
    parsed: list[tuple[sqlite3.Row, list[str]]] = []
    all_facts: list[str] = []
    for row in rows:
        try:
            value = json.loads(str(row["fact_ids_json"] or "[]"))
            fact_ids = (
                [str(item) for item in value if item][:_MAX_FACTS_PER_OUTCOME]
                if isinstance(value, list) else []
            )
        except (TypeError, ValueError, json.JSONDecodeError):
            fact_ids = []
        parsed.append((row, fact_ids))
        all_facts.extend(fact_ids)
    return parsed, all_facts


def _repair_observations(
    profile_id: str,
    parsed: list[tuple[sqlite3.Row, list[str]]],
    source_map: dict[str, set[str]],
) -> list[tuple[str, str, list[str], float]]:
    observations = []
    for row, fact_ids in parsed:
        sources = sorted({
            source for fact_id in fact_ids
            for source in source_map.get(fact_id, set())
        })[:_MAX_SOURCES_PER_OUTCOME]
        observations.append((
            profile_id, str(row["outcome_id"]), sources, float(row["reward"]),
        ))
    return observations


def repair_historical_source_quality(
    memory_db_path: Path,
    learning_db_path: Path,
    profile_id: str,
    *,
    batch_size: int = 250,
    max_batches: int = 4,
) -> dict[str, int | bool]:
    """Explicit, resumable historical repair; never invoked by API handlers.

    Work is capped to 1,000 outcomes per call by default. The cursor advances
    only after the idempotent observation ledger commits, so interruption can
    replay safely without double-counting.
    """
    safe_batch = max(1, min(1000, int(batch_size)))
    safe_batches = max(1, min(100, int(max_batches)))
    scorer = SourceQualityScorer(Path(learning_db_path))
    scanned = 0
    observations = 0
    complete = False
    for _ in range(safe_batches):
        settled_cursor, outcome_cursor = scorer.get_repair_position(profile_id)
        rows = _load_reward_repair_batch(
            Path(memory_db_path),
            profile_id,
            settled_cursor,
            outcome_cursor,
            safe_batch,
        )
        if not rows:
            complete = True
            break
        parsed, all_facts = _parse_repair_rows(rows)
        source_map = _load_source_map(
            Path(memory_db_path), profile_id, all_facts, strict=True,
        )
        batch_observations = _repair_observations(
            profile_id, parsed, source_map,
        )
        observations += scorer.record_rewards(batch_observations)
        scanned += len(rows)
        scorer.set_repair_position(
            profile_id,
            str(rows[-1]["settled_key"] or ""),
            str(rows[-1]["outcome_id"]),
        )
    return {
        "scanned": scanned,
        "observations": observations,
        "complete": complete,
    }


def _load_reward_repair_batch(
    memory_db_path: Path,
    profile_id: str,
    after_settled_at: str,
    after_outcome_id: str,
    limit: int,
) -> list[sqlite3.Row]:
    if not memory_db_path.exists():
        return []
    try:
        conn = sqlite3.connect(
            f"file:{memory_db_path}?mode=ro", uri=True, timeout=1.0,
        )
        conn.row_factory = sqlite3.Row
        try:
            columns = {
                str(row["name"])
                for row in conn.execute("PRAGMA table_info(action_outcomes)")
            }
            required = {
                "outcome_id", "profile_id", "fact_ids_json",
                "reward", "settled",
            }
            if not required.issubset(columns):
                return []
            return conn.execute(
                "SELECT outcome_id, fact_ids_json, reward, "
                "COALESCE(settled_at, '') AS settled_key "
                "FROM action_outcomes WHERE profile_id = ? "
                "AND settled = 1 AND reward IS NOT NULL "
                "AND typeof(reward) IN ('integer', 'real') "
                "AND (COALESCE(settled_at, '') > ? OR "
                "(COALESCE(settled_at, '') = ? AND outcome_id > ?)) "
                "ORDER BY COALESCE(settled_at, '') ASC, outcome_id ASC LIMIT ?",
                (
                    profile_id,
                    str(after_settled_at),
                    str(after_settled_at),
                    str(after_outcome_id),
                    int(limit),
                ),
            ).fetchall()
        finally:
            conn.close()
    except sqlite3.Error as exc:
        logger.debug("source-quality repair unavailable: %s", exc)
        raise SourceQualityRepairUnavailable(
            "repair batch temporarily unavailable",
        ) from exc
