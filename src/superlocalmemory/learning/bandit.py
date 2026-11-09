# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory v3.4.22 — LLD-03 §3 + §5.3

"""Contextual Thompson-sampling bandit over discrete channel-weight arms.


Schema: ``bandit_arms`` + ``bandit_plays`` live in ``learning.db``, created by
LLD-07 M005. This module NEVER defines DDL — it only READs / WRITEs.

Hard rules:
  - B1: the Beta draw is keyed by an install secret (``bandit_draw``):
    unpredictable without it, and one question over one posterior is one arm
    (4.1.20 — a fresh system-random draw reordered identical recalls).
  - B2: α, β clamped at ``SLM_BANDIT_ALPHA_CAP`` (default 1000).
  - B4: stratum cardinality == 48 (4 query_types × 3 entity bins × 4 buckets).
  - B5: cache invalidated on every successful ``update``.
  - B6: raw query text NEVER written to bandit tables.
  - B7: ``choose`` p99 ≤ 10 ms.
  - B8: ``retention_sweep`` only deletes settled rows older than cutoff.

All SQL is parameterised — grep guard in CI ensures no f-string SQL here.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from superlocalmemory.learning.arm_catalog import ARM_CATALOG
from superlocalmemory.learning.bandit_cache import (
    _BanditCache,
    get_shared_cache,
)
from superlocalmemory.learning.bandit_draw import draw_rng

logger = logging.getLogger(__name__)

#: How many shown fact_ids a play records. The settler reads the top 3; the
#: extra two are headroom for a reranker that reorders after the play is
#: written, and a bound so this cannot become an unread blob on a hot path.
_SHOWN_FACT_LIMIT = 5

_FALLBACK_ARM_ID = "fallback_default"

#: Marks a play whose shown order was replaced after the arm chose it (the
#: opt-in hosted reorder). Written to ``settlement_type`` while the play is
#: open; ``update`` then settles the play but leaves the arm's posterior alone,
#: because the outcome describes an order the arm did not produce.
ORDER_REPLACED = "order_replaced"

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
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA busy_timeout=5000")
        except sqlite3.Error:  # pragma: no cover — best-effort
            pass
        conn.row_factory = sqlite3.Row
        _holder.conn = conn
        _holder.path = path_str
    except BaseException:
        # Setup failed: close it now, or the file stays open (Windows
        # then cannot delete or replace it) for as long as the error lives.
        conn.close()
        raise
    return conn


def close_threadlocal_conn() -> None:
    """Close the threadlocal bandit connection on the calling thread.

    v3.4.33: background callers (asyncio.to_thread pool threads) MUST call
    this after finishing bandit work.  Without it, each pool thread keeps a
    leaked connection to learning.db for the process lifetime — observed as
    12+ open file descriptors and ~100 MB wasted page-cache RAM.
    """
    if _holder.conn is not None:
        try:
            _holder.conn.close()
        except sqlite3.Error:  # pragma: no cover
            pass
        _holder.conn = None
        _holder.path = None


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _clamp_reward(reward: Any) -> float | None:
    """Reward clamped to [0, 1]; None when it is not a number."""
    try:
        value = float(reward)
    except (TypeError, ValueError):
        return None
    return min(1.0, max(0.0, value))


def _positive_weight(weight: Any) -> float:
    """A usable IPS weight: non-positive or non-numeric becomes 1.0."""
    try:
        value = float(weight)
    except (TypeError, ValueError):
        return 1.0
    return value if value > 0.0 else 1.0


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

    # ------------------------------------------------------------------
    # choose
    # ------------------------------------------------------------------

    def choose(
        self,
        context: dict[str, Any],
        query_id: str,
        *,
        draw_key: str | None = None,
    ) -> BanditChoice:
        """Sample one arm under the context stratum; record play row.

        ``draw_key`` is the question. The draw is keyed by it and the stratum's
        posterior, so asking the same thing twice picks the same arm until a
        settled reward changes what the bandit knows. Without one, the opaque
        ``query_id`` keys the draw. The question is never stored (B6).

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

        key = draw_key if draw_key is not None else query_id
        arm_id = self._sample_best(posteriors, stratum, key)
        play_id = self._insert_play(query_id, stratum, arm_id)
        return BanditChoice(
            stratum=stratum,
            arm_id=arm_id,
            weights=dict(self._catalog[arm_id]),
            play_id=play_id,
        )

    def choose_readonly(
        self,
        context: dict[str, Any],
        *,
        draw_key: str | None = None,
    ) -> BanditChoice:
        """Sample an arm from a read-only snapshot without recording a play.

        ``draw_key`` keys the draw exactly as in :meth:`choose`.

        Recall uses this method so the established bandit weighting and
        ensemble quality path remain available without turning a query into a
        ``bandit_plays`` write.  It deliberately bypasses the writer-oriented
        thread-local connection factory: that factory configures WAL mode and
        is not a physical read-only guarantee.
        """
        stratum = compute_stratum(context)
        try:
            posteriors = self._load_stratum_posteriors_readonly(stratum)
        except sqlite3.Error as exc:
            logger.warning(
                "bandit.choose_readonly: posterior load failed stratum=%s: %s",
                stratum,
                exc,
            )
            posteriors = {}
        arm_id = self._sample_best(posteriors, stratum, draw_key or "")
        return BanditChoice(
            stratum=stratum,
            arm_id=arm_id,
            weights=dict(self._catalog[arm_id]),
            play_id=None,
        )

    def _sample_best(
        self,
        posteriors: dict[str, tuple[float, float]],
        stratum: str,
        draw_key: str,
    ) -> str:
        """Draw one Beta sample per arm, return argmax.

        B1: the generator is keyed (``bandit_draw.draw_rng``), so the same
        question over the same posterior always yields the same arm.
        """
        rng = draw_rng(self._profile, stratum, draw_key, posteriors)
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
        weight: float = 1.0,
        *,
        outcome_id: str | None = None,
    ) -> bool:
        """Apply the reward to the (profile, stratum, arm) posterior.

        ``weight``: inverse-propensity scale (``propensity.py``); 1.0 = plain.
        ``outcome_id``: the reported outcome spent on this play, written in the
        SAME transaction as the posterior change under a unique index (M055),
        so one outcome settles one play across passes, restarts and concurrent
        settlers; a second claim is refused and nothing moves.
        Returns True on success. Never raises. Cache invalidated (B5).
        """
        reward_f = _clamp_reward(reward)
        if reward_f is None:
            logger.warning("bandit.update: non-numeric reward, ignoring")
            return False
        weight_f = _positive_weight(weight)
        try:
            conn = _conn_for(self._db_path)
        except sqlite3.Error as exc:  # pragma: no cover — defensive
            logger.warning("bandit.update: cannot open db: %s", exc)
            return False
        try:
            touched = self._transact(conn, int(play_id), reward_f, kind,
                                     weight_f, outcome_id)
        except sqlite3.IntegrityError:
            logger.info("bandit.update: outcome %s already spent", outcome_id)
            return False
        except sqlite3.OperationalError as exc:
            if outcome_id and "outcome_id" in str(exc):
                # A store M055 has not reached: settle without the durable
                # spend record rather than not at all, and say so.
                logger.warning("bandit.update: no outcome ledger (M055 "
                               "pending); spend is not durable")
                return self.update(play_id, reward_f, kind, weight_f)
            logger.warning("bandit.update: write failed: %s", exc)
            return False
        except sqlite3.Error as exc:
            logger.warning("bandit.update: write failed: %s", exc)
            return False
        if touched is None:
            return False
        if touched:
            try:
                self._cache.invalidate(*touched)
            except Exception:  # pragma: no cover — defensive
                pass
        return True

    def _transact(self, conn: sqlite3.Connection, play_id: int,
                  reward_f: float, kind: str, weight_f: float,
                  outcome_id: str | None):
        """``_apply_update`` inside BEGIN IMMEDIATE; rolls back on any error."""
        conn.execute("BEGIN IMMEDIATE")
        try:
            touched = self._apply_update(conn, play_id, reward_f, kind,
                                         weight_f, outcome_id)
        except BaseException:
            conn.execute("ROLLBACK")
            raise
        conn.execute("COMMIT")
        return touched

    def _apply_update(
        self, conn: sqlite3.Connection, play_id: int, reward_f: float,
        kind: str, weight_f: float, outcome_id: str | None,
    ) -> tuple[str, str] | tuple[()] | None:
        """The writes of one settlement, inside the caller's transaction.

        Returns ``None`` when nothing was settled, ``()`` when the play closed
        without touching an arm, else the (profile, stratum) to invalidate.
        """
        row = conn.execute(
            "SELECT profile_id, stratum, arm_id, settled_at, settlement_type "
            "FROM bandit_plays WHERE play_id = ?", (play_id,),
        ).fetchone()
        if row is None:
            logger.warning("bandit.update: play_id %s not found", play_id)
            return None
        if row["settled_at"] is not None:
            return None  # idempotent no-op: already settled
        now = _now_iso()
        if row["settlement_type"] == ORDER_REPLACED:
            kind = f"{kind}:{ORDER_REPLACED}"
        else:
            self._bump_arm(conn, row, reward_f, weight_f, now)
        if outcome_id:
            conn.execute(
                "UPDATE bandit_plays SET reward = ?, settled_at = ?, "
                "settlement_type = ?, outcome_id = ? WHERE play_id = ?",
                (reward_f, now, str(kind), str(outcome_id), play_id),
            )
        else:
            conn.execute(
                "UPDATE bandit_plays SET reward = ?, settled_at = ?, "
                "settlement_type = ? WHERE play_id = ?",
                (reward_f, now, str(kind), play_id),
            )
        if row["settlement_type"] == ORDER_REPLACED:
            return ()
        return (row["profile_id"], row["stratum"])

    def _bump_arm(self, conn: sqlite3.Connection, row: sqlite3.Row,
                  reward_f: float, weight_f: float, now: str) -> None:
        """Fold one weighted reward into the play's arm (B2: capped)."""
        cap = self._alpha_cap
        conn.execute(
            "INSERT OR IGNORE INTO bandit_arms "
            "(profile_id, stratum, arm_id, alpha, beta, plays, "
            " last_played_at) VALUES (?, ?, ?, 1.0, 1.0, 0, ?)",
            (row["profile_id"], row["stratum"], row["arm_id"], now),
        )
        conn.execute(
            "UPDATE bandit_arms SET alpha = MIN(?, alpha + ?), "
            "beta = MIN(?, beta + ?), plays = plays + 1, last_played_at = ? "
            "WHERE profile_id = ? AND stratum = ? AND arm_id = ?",
            (cap, weight_f * reward_f, cap, weight_f * (1.0 - reward_f),
             now, row["profile_id"], row["stratum"], row["arm_id"]),
        )

    def mark_order_replaced(self, play_id: int) -> bool:
        """Say that what this play showed first was not the arm's order.

        Only an open play can be marked; a settled one is history. Best-effort,
        like ``record_shown``: a failure means the arm is credited as before.
        """
        if not play_id:
            return False
        try:
            conn = _conn_for(self._db_path)
            cur = conn.execute(
                "UPDATE bandit_plays SET settlement_type = ? "
                "WHERE play_id = ? AND settled_at IS NULL",
                (ORDER_REPLACED, int(play_id)),
            )
            return bool(cur.rowcount)
        except sqlite3.Error as exc:
            logger.debug("bandit.mark_order_replaced: %s", exc)
            return False

    def record_shown(self, play_id: int, fact_ids: Sequence[str]) -> bool:
        """Record which memories this play surfaced, for later settlement.

        A play is settled from evidence — did anything downstream reference a
        memory this query returned? The settler used to answer "which memories"
        by reading ``learning_signals`` for the same ``query_id``, and those
        rows are written by an enqueue that is deliberately off (it costs twenty
        rows per query and inflated the ranking-phase counter 2,675x). With no
        evidence to look for, every settlement fell through to the 120-second
        neutral default of 0.5, which is why 165 arms all read alpha == beta.

        So the play carries its own evidence. Best-effort by design: the reward
        is a nicety and a recall is not. A failure here means this one play
        settles as ``default``, exactly as before.

        Bounded to the first few ids because the settler only ever reads the
        top of the list, and an unbounded JSON blob on the recall path is a
        write cost with no reader.
        """
        if not play_id:
            return False
        ids = [str(f) for f in list(fact_ids)[:_SHOWN_FACT_LIMIT] if f]
        if not ids:
            return False
        try:
            conn = _conn_for(self._db_path)
            conn.execute(
                "UPDATE bandit_plays SET shown_fact_ids = ? WHERE play_id = ?",
                (json.dumps(ids), int(play_id)),
            )
            return True
        except sqlite3.Error as exc:
            # Includes "no such column" on a store where M044 has not run.
            logger.debug("bandit.record_shown: %s", exc)
            return False

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

    def _load_stratum_posteriors_readonly(
        self,
        stratum: str,
    ) -> dict[str, tuple[float, float]]:
        """Read posteriors through SQLite's URI read-only boundary."""
        uri = f"{self._db_path.resolve().as_uri()}?mode=ro"
        conn = sqlite3.connect(uri, uri=True, timeout=0.25)
        try:
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA query_only=ON")
            conn.execute("PRAGMA busy_timeout=250")
            rows = conn.execute(
                "SELECT arm_id, alpha, beta FROM bandit_arms "
                "WHERE profile_id = ? AND stratum = ?",
                (self._profile, stratum),
            ).fetchall()
        finally:
            conn.close()
        return {
            row["arm_id"]: (float(row["alpha"]), float(row["beta"]))
            for row in rows
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
    "ORDER_REPLACED",
    "BanditChoice",
    "ContextualBandit",
    "close_threadlocal_conn",
    "compute_stratum",
    "current_time_bucket",
    "retention_sweep",
)
