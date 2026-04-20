# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory v3.4.22 — LLD-03 §3 + §5.3

"""Contextual Thompson-sampling bandit over discrete channel-weight arms.

LLD reference: ``.backup/active-brain/lld/LLD-03-contextual-bandit-and-ensemble.md``
Sections 3 (algorithm), 5.3 (file spec), 8 (hard rules).

Schema: ``bandit_arms`` + ``bandit_plays`` live in ``learning.db``, created by
LLD-07 M005. This module NEVER defines DDL — it only READs / WRITEs.

Hard rules:
  - B1: ``secrets.SystemRandom`` used for Beta sampling (NOT ``random``).
  - B2: α, β clamped at ``SLM_BANDIT_ALPHA_CAP`` (default 1000).
  - B4: stratum cardinality == 48 (4 query_types × 3 entity bins × 4 buckets).
  - B5: cache invalidated on every successful ``update``.
  - B6: raw query text NEVER written to bandit tables.
  - B7: ``choose`` p99 ≤ 10 ms.
  - B8: ``retention_sweep`` only deletes settled rows older than cutoff.

All SQL is parameterised — grep guard in CI ensures no f-string SQL here.
"""

from __future__ import annotations

import logging
import os
import secrets
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from superlocalmemory.learning.arm_catalog import ARM_CATALOG
from superlocalmemory.learning.bandit_cache import (
    _BanditCache,
    get_shared_cache,
)

logger = logging.getLogger(__name__)

_FALLBACK_ARM_ID = "fallback_default"

_DEFAULT_ALPHA_CAP = float(os.environ.get("SLM_BANDIT_ALPHA_CAP", "1000.0"))

# Query-type bins (must match features.py one-hot exactly).
_QUERY_TYPES: tuple[str, ...] = (
    "single_hop",
    "multi_hop",
    "temporal",
    "open_domain",
)
_ENTITY_BINS: tuple[str, ...] = ("0", "1-2", "3+")
_TIME_BUCKETS: tuple[str, ...] = ("morning", "afternoon", "evening", "night")

_UNKNOWN_QTYPE = "open_domain"  # safe default if caller hands us a new label


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BanditChoice:
    """Result of one ``choose`` call. Immutable (D8 channel-weight bundle)."""

    stratum: str
    arm_id: str
    weights: dict[str, float] = field(default_factory=dict)
    play_id: int | None = None


# ---------------------------------------------------------------------------
# Stratum computation
# ---------------------------------------------------------------------------


def _bin_entities(count: int) -> str:
    if count <= 0:
        return "0"
    if count <= 2:
        return "1-2"
    return "3+"


def _time_bucket_from_hour(hour: int) -> str:
    # wall-clock local: 05:00-11:59 morning, 12:00-16:59 afternoon,
    # 17:00-20:59 evening, 21:00-04:59 night.
    if 5 <= hour <= 11:
        return "morning"
    if 12 <= hour <= 16:
        return "afternoon"
    if 17 <= hour <= 20:
        return "evening"
    return "night"


def current_time_bucket(now: datetime | None = None) -> str:
    """Return the wall-clock time bucket for ``now`` (local timezone)."""
    dt = now if now is not None else datetime.now().astimezone()
    return _time_bucket_from_hour(dt.hour)


def compute_stratum(context: dict[str, Any]) -> str:
    """Compute the 3-tuple stratum from a context dict.

    Context keys consumed:
      - ``query_type`` — one of ``_QUERY_TYPES``; unknown → ``open_domain``.
      - ``entity_count_bin`` — if present, used verbatim; else derived from
        ``entity_count`` (int).
      - ``time_bucket`` — if present, used verbatim; else derived from clock.

    B4: enumerating all Cartesian products yields exactly 48 strata.
    """
    qtype = context.get("query_type")
    if qtype not in _QUERY_TYPES:
        qtype = _UNKNOWN_QTYPE

    ebin = context.get("entity_count_bin")
    if ebin not in _ENTITY_BINS:
        try:
            ecount = int(context.get("entity_count", 0))
        except (TypeError, ValueError):
            ecount = 0
        ebin = _bin_entities(ecount)

    tbucket = context.get("time_bucket")
    if tbucket not in _TIME_BUCKETS:
        tbucket = current_time_bucket()

    return f"{qtype}|{ebin}|{tbucket}"


# ---------------------------------------------------------------------------
# Threadlocal sqlite connection (mirrors LLD-02 §4.2 recipe)
# ---------------------------------------------------------------------------


class _ConnHolder(threading.local):
    conn: sqlite3.Connection | None = None
    path: str | None = None


_holder = _ConnHolder()


def _conn_for(db_path: Path) -> sqlite3.Connection:
    """Return a WAL-configured threadlocal connection to ``db_path``.

    Reused across calls on the same thread; reopened on path change.
    """
    path_str = str(db_path)
    existing = _holder.conn
    if existing is not None and _holder.path == path_str:
        return existing
    if existing is not None:
        try:
            existing.close()
        except sqlite3.Error:  # pragma: no cover
            pass
    conn = sqlite3.connect(path_str, timeout=10.0, isolation_level=None)
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA busy_timeout=5000")
    except sqlite3.Error:  # pragma: no cover — best-effort
        pass
    conn.row_factory = sqlite3.Row
    _holder.conn = conn
    _holder.path = path_str
    return conn


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# ContextualBandit
# ---------------------------------------------------------------------------


class ContextualBandit:
    """Thompson-sampling bandit over the 40-arm catalog.

    One instance per ``(profile_id, db_path)`` is fine — stateless apart from
    the shared posterior cache. ``choose`` is hot-path; ``update`` runs on
    the reward settler worker (async).
    """

    def __init__(
        self,
        db_path: Path | str,
        profile_id: str,
        *,
        catalog: dict[str, dict[str, float]] | None = None,
        cache: _BanditCache | None = None,
        alpha_cap: float = _DEFAULT_ALPHA_CAP,
    ) -> None:
        self._db_path = Path(db_path)
        self._profile = str(profile_id)
        self._catalog = catalog or ARM_CATALOG
        self._cache = cache or get_shared_cache()
        self._alpha_cap = float(alpha_cap)
        # Fresh SystemRandom per instance; cheap, seeded from os.urandom.
        self._rng = secrets.SystemRandom()

    # ------------------------------------------------------------------
    # choose
    # ------------------------------------------------------------------

    def choose(
        self,
        context: dict[str, Any],
        query_id: str,
    ) -> BanditChoice:
        """Sample one arm under the context stratum; record play row.

        Never raises. On DB error, returns a fallback_default choice with
        ``play_id=None`` and logs at WARN level (no PII).
        """
        stratum = compute_stratum(context)
        try:
            posteriors = self._cache.get(
                self._profile, stratum, self._load_stratum_posteriors,
            )
        except sqlite3.Error as exc:
            logger.warning(
                "bandit.choose: posterior load failed stratum=%s: %s",
                stratum, exc,
            )
            posteriors = {}

        arm_id = self._sample_best(posteriors)
        play_id = self._insert_play(query_id, stratum, arm_id)
        return BanditChoice(
            stratum=stratum,
            arm_id=arm_id,
            weights=dict(self._catalog[arm_id]),
            play_id=play_id,
        )

    def _sample_best(
        self,
        posteriors: dict[str, tuple[float, float]],
    ) -> str:
        """Draw one Beta sample per arm, return argmax."""
        rng = self._rng  # B1: secrets.SystemRandom
        best_arm = _FALLBACK_ARM_ID
        best_sample = float("-inf")
        for arm_id in self._catalog:
            a, b = posteriors.get(arm_id, (1.0, 1.0))
            # Defensive: reject non-positive (shouldn't happen; we clamp on
            # write, but guard against external DB tampering).
            if a <= 0 or b <= 0:  # pragma: no cover — defensive
                a = max(a, 1.0)
                b = max(b, 1.0)
            try:
                sample = rng.betavariate(a, b)
            except ValueError:  # pragma: no cover — defensive
                continue
            if sample > best_sample:
                best_sample = sample
                best_arm = arm_id
        return best_arm

    # ------------------------------------------------------------------
    # update
    # ------------------------------------------------------------------

    def update(
        self,
        play_id: int,
        reward: float,
        kind: str = "proxy_position",
    ) -> bool:
        """Apply the reward to the (profile, stratum, arm) posterior.

        Returns True on success. Never raises — DB failures logged at WARN.
        Cache invalidated on success (B5).
        """
        try:
            reward_f = float(reward)
        except (TypeError, ValueError):
            logger.warning("bandit.update: non-numeric reward, ignoring")
            return False
        # Clamp reward ∈ [0, 1].
        if reward_f < 0.0:
            reward_f = 0.0
        elif reward_f > 1.0:
            reward_f = 1.0

        try:
            conn = _conn_for(self._db_path)
        except sqlite3.Error as exc:  # pragma: no cover — defensive
            logger.warning("bandit.update: cannot open db: %s", exc)
            return False

        try:
            row = conn.execute(
                "SELECT profile_id, stratum, arm_id, settled_at "
                "FROM bandit_plays WHERE play_id = ?",
                (int(play_id),),
            ).fetchone()
        except sqlite3.Error as exc:
            logger.warning("bandit.update: lookup failed: %s", exc)
            return False

        if row is None:
            logger.warning("bandit.update: play_id %s not found", play_id)
            return False
        if row["settled_at"] is not None:
            # Idempotent no-op: already settled.
            return False

        profile_id = row["profile_id"]
        stratum = row["stratum"]
        arm_id = row["arm_id"]
        now = _now_iso()
        cap = self._alpha_cap

        try:
            # Ensure an arm row exists with prior (1,1). INSERT OR IGNORE is
            # cheap — PK composite guarantees uniqueness.
            conn.execute(
                "INSERT OR IGNORE INTO bandit_arms "
                "(profile_id, stratum, arm_id, alpha, beta, plays, "
                " last_played_at) VALUES (?, ?, ?, 1.0, 1.0, 0, ?)",
                (profile_id, stratum, arm_id, now),
            )
            # B2: MIN(cap, value) clamp.
            conn.execute(
                "UPDATE bandit_arms "
                "SET alpha = MIN(?, alpha + ?), "
                "    beta  = MIN(?, beta  + ?), "
                "    plays = plays + 1, "
                "    last_played_at = ? "
                "WHERE profile_id = ? AND stratum = ? AND arm_id = ?",
                (cap, reward_f, cap, 1.0 - reward_f, now,
                 profile_id, stratum, arm_id),
            )
            conn.execute(
                "UPDATE bandit_plays "
                "SET reward = ?, settled_at = ?, settlement_type = ? "
                "WHERE play_id = ?",
                (reward_f, now, str(kind), int(play_id)),
            )
        except sqlite3.Error as exc:
            logger.warning("bandit.update: write failed: %s", exc)
            return False

        # B5: invalidate the (profile, stratum) cache entry.
        try:
            self._cache.invalidate(profile_id, stratum)
        except Exception:  # pragma: no cover — defensive
            pass
        return True

    # ------------------------------------------------------------------
    # snapshot (for dashboard — LLD-04 consumer)
    # ------------------------------------------------------------------

    def snapshot(self, top_n: int = 5) -> dict[str, list[dict[str, Any]]]:
        """Return ``{stratum → top-N arms by plays}``.

        Lightweight read. Never raises — DB failures return ``{}``.
        """
        try:
            conn = _conn_for(self._db_path)
            rows = conn.execute(
                "SELECT stratum, arm_id, alpha, beta, plays "
                "FROM bandit_arms WHERE profile_id = ? "
                "ORDER BY stratum ASC, plays DESC",
                (self._profile,),
            ).fetchall()
        except sqlite3.Error as exc:
            logger.debug("bandit.snapshot: %s", exc)
            return {}

        out: dict[str, list[dict[str, Any]]] = {}
        for r in rows:
            bucket = out.setdefault(r["stratum"], [])
            if len(bucket) < int(top_n):
                bucket.append({
                    "arm_id": r["arm_id"],
                    "alpha": float(r["alpha"]),
                    "beta": float(r["beta"]),
                    "plays": int(r["plays"]),
                })
        return out

    # ------------------------------------------------------------------
    # Loader for cache (executed outside the cache lock)
    # ------------------------------------------------------------------

    def _load_stratum_posteriors(
        self,
        profile_id: str,
        stratum: str,
    ) -> dict[str, tuple[float, float]]:
        """Read ``{arm_id: (α, β)}`` for the given (profile, stratum)."""
        try:
            conn = _conn_for(self._db_path)
            rows = conn.execute(
                "SELECT arm_id, alpha, beta FROM bandit_arms "
                "WHERE profile_id = ? AND stratum = ?",
                (profile_id, stratum),
            ).fetchall()
        except sqlite3.Error as exc:
            logger.debug(
                "bandit._load_stratum_posteriors: %s", exc,
            )
            return {}
        return {
            r["arm_id"]: (float(r["alpha"]), float(r["beta"]))
            for r in rows
        }

    def _insert_play(
        self,
        query_id: str,
        stratum: str,
        arm_id: str,
    ) -> int | None:
        """Insert a bandit_plays row; return lastrowid or None on failure.

        B6: raw query text is NEVER stored — only ``query_id`` (opaque UUID).
        """
        try:
            conn = _conn_for(self._db_path)
            cur = conn.execute(
                "INSERT INTO bandit_plays "
                "(profile_id, query_id, stratum, arm_id, played_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (self._profile, str(query_id), stratum, arm_id, _now_iso()),
            )
            return int(cur.lastrowid) if cur.lastrowid is not None else None
        except sqlite3.Error as exc:
            logger.debug("bandit._insert_play: %s", exc)
            return None


# ---------------------------------------------------------------------------
# Retention sweep (LLD-03 §3.6 — B8)
# ---------------------------------------------------------------------------


def retention_sweep(
    db_path: Path | str,
    retention_days: int = 7,
    *,
    now: datetime | None = None,
    chunk_size: int = 1000,
) -> int:
    """Delete settled bandit_plays older than ``now - retention_days``.

    Only rows with ``settled_at IS NOT NULL AND settled_at < cutoff`` are
    removed. Unsettled rows are NEVER touched (B8).

    Returns total number of rows deleted.
    """
    if retention_days < 0:
        raise ValueError("retention_days must be >= 0")
    current = now if now is not None else datetime.now(timezone.utc)
    cutoff_iso = (current - timedelta(days=int(retention_days))).isoformat(
        timespec="seconds",
    )
    deleted_total = 0
    path = Path(db_path)

    # Fresh connection — sweeps may be called from arbitrary threads / the
    # scheduler, not necessarily the hot-path thread.
    conn = sqlite3.connect(str(path), timeout=10.0, isolation_level=None)
    try:
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=5000")
        except sqlite3.Error:  # pragma: no cover
            pass

        while True:
            try:
                cur = conn.execute(
                    "DELETE FROM bandit_plays "
                    "WHERE rowid IN ("
                    "  SELECT rowid FROM bandit_plays "
                    "  WHERE settled_at IS NOT NULL AND settled_at < ? "
                    "  LIMIT ?"
                    ")",
                    (cutoff_iso, int(chunk_size)),
                )
            except sqlite3.Error as exc:
                logger.warning("retention_sweep: delete failed: %s", exc)
                break
            affected = cur.rowcount or 0
            if affected <= 0:
                break
            deleted_total += affected
            # Guard against infinite loop on drivers that return -1.
            if affected < int(chunk_size) and cur.rowcount != -1:
                break
    finally:
        try:
            conn.close()
        except sqlite3.Error:  # pragma: no cover
            pass

    logger.info(
        "bandit_plays_retention_sweep: deleted=%d, cutoff=%s",
        deleted_total, cutoff_iso,
    )
    return deleted_total


__all__ = (
    "BanditChoice",
    "ContextualBandit",
    "compute_stratum",
    "current_time_bucket",
    "retention_sweep",
)
