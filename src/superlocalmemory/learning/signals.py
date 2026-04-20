# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Zero-Cost Learning Signals — mathematical learning without LLM tokens.

Four signal types that improve retrieval quality over time:

1. Entropy Gap     — Surprising content gets deeper indexing.
2. Co-Retrieval    — Memories retrieved together strengthen graph edges.
3. Channel Credit  — Track which retrieval channel works for which query type.
4. Confidence Lifecycle — Boost on access, decay over time.

All signals are computed locally with zero LLM cost.
Inspired by: Nemori (entropy), A-Mem (link evolution), RMM (citation feedback).

Part of Qualixar | Author: Varun Pratap Bhardwaj
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import queue
import sqlite3
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ===========================================================================
# LLD-02 §4.1 — SignalBatch + enqueue + record_signal_batch
# ===========================================================================
#
# These module-level helpers are the v3.4.22 signal pipeline. The class
# ``LearningSignals`` below stays in place for v3.4.20 compatibility (D5);
# new writers go through ``enqueue`` / ``record_signal_batch``.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SignalCandidate:
    """One candidate returned by the retrieval pipeline for signal recording.

    Immutable (frozen=True). Carries the minimum needed to write both a
    ``learning_signals`` row and a ``learning_features`` row in a single TX.
    """

    fact_id: str
    channel_scores: dict[str, float] = field(default_factory=dict)
    cross_encoder_score: float | None = None
    # Full result dict — used by FeatureExtractor.extract(); kept lazily so we
    # only serialise features at drain time, not enqueue time.
    result_dict: dict[str, Any] = field(default_factory=dict)

    def to_result_dict(self) -> dict[str, Any]:
        """Return a result dict suitable for ``FeatureExtractor.extract()``.

        Includes channel_scores and cross_encoder_score. Callers can override
        by placing richer fields in ``result_dict`` at construction time.
        """
        merged: dict[str, Any] = {"fact_id": self.fact_id}
        if self.channel_scores:
            merged["channel_scores"] = dict(self.channel_scores)
        if self.cross_encoder_score is not None:
            merged["cross_encoder_score"] = self.cross_encoder_score
        # Caller-provided fields override defaults.
        merged.update(self.result_dict)
        return merged


@dataclass(frozen=True)
class SignalBatch:
    """One recall's worth of signal rows. Enqueued onto the worker."""

    profile_id: str
    query_id: str
    query_text: str
    candidates: tuple[SignalCandidate, ...] = field(default_factory=tuple)
    query_context: dict[str, Any] = field(default_factory=dict)


# Module-level bounded queue — one per process. Sized per LLD-02 §9
# ``SLM_SIGNAL_QUEUE_MAX`` (default 5000). Readers are the signal_worker.
_QUEUE_MAX: int = 5000
_Q: "queue.Queue[SignalBatch]" = queue.Queue(maxsize=_QUEUE_MAX)

# Observability counters — module-level so tests can reset/inspect.
_counters: dict[str, int] = {
    "signal_dropped_total": 0,
    "signal_enqueued_total": 0,
    "enqueue_failed_total": 0,
    "signal_drop_on_flush_total": 0,
}
_counters_lock = threading.Lock()

# Throttle drop-warning logging to once per 60 seconds (LLD-02 §4.2).
_last_drop_log_ts: list[float] = [0.0]


def _bump(counter: str, n: int = 1) -> None:
    with _counters_lock:
        _counters[counter] = _counters.get(counter, 0) + n


# S8-ARC-03 (v3.4.22): public producer/consumer contract. ``signal_worker``
# used to reach through ``signals._Q`` and ``signals._bump`` by name,
# which made the private-by-convention boundary the actual test seam
# too. These wrappers are the sanctioned surface; ``_Q`` / ``_bump`` stay
# internal, and test-only helpers live on the ``_testing`` submodule.
def get_queue() -> "queue.Queue[SignalBatch]":
    """Return the module-level producer queue (shared across threads)."""
    import sys as _sys
    # Tests may monkeypatch _Q by attribute — resolve dynamically.
    return getattr(_sys.modules[__name__], "_Q", None) or _Q


def bump_counter(counter: str, n: int = 1) -> None:
    """Public counter increment (identical semantics to internal ``_bump``)."""
    _bump(counter, n)


def get_counters() -> dict[str, int]:
    """Return a snapshot of signal pipeline counters."""
    with _counters_lock:
        return dict(_counters)


def reset_counters() -> None:
    """Reset counters to zero — TEST-ONLY helper."""
    with _counters_lock:
        for k in _counters:
            _counters[k] = 0
    _last_drop_log_ts[0] = 0.0


def _drain_queue_for_tests() -> None:
    """Drain the module queue — TEST-ONLY."""
    while True:
        try:
            _Q.get_nowait()
        except queue.Empty:
            return


def queue_size() -> int:
    """Return current queue depth — used by worker + tests."""
    import sys as _sys
    q = getattr(_sys.modules[__name__], "_Q", None) or _Q
    return q.qsize()


def _hash_query(query_text: str) -> str:
    """Compute ``query_text_hash`` per LLD-02 §4.1.

    Lowercased, stripped, SHA-256 truncated to 32 hex chars. Stored in the
    ``learning_signals.query_text_hash`` column. The raw ``query`` column
    MUST stay empty (S2 privacy rule).
    """
    normalised = (query_text or "").lower().strip().encode("utf-8")
    return hashlib.sha256(normalised).hexdigest()[:32]


def enqueue(batch: SignalBatch) -> None:
    """Non-blocking enqueue of a SignalBatch.

    Hot-path-safe: never raises, never blocks longer than a ``put_nowait``.
    Drops with a counter bump if the queue is full (SW2).
    Wraps ``queue.put_nowait`` exceptions (RP1 — never propagate).
    """
    import sys as _sys
    import time as _time

    if batch is None or not isinstance(batch, SignalBatch):
        _bump("enqueue_failed_total")
        return

    # Resolve the queue through the module to honour monkeypatches in tests
    # and future runtime reconfig. This is cheap — one dict lookup.
    q = getattr(_sys.modules[__name__], "_Q", None) or _Q

    try:
        q.put_nowait(batch)
    except queue.Full:
        _bump("signal_dropped_total")
        now = _time.monotonic()
        if now - _last_drop_log_ts[0] >= 60.0:
            _last_drop_log_ts[0] = now
            logger.warning(
                "signal queue full; dropped batch (total dropped=%d)",
                get_counters()["signal_dropped_total"],
            )
        return
    except Exception as exc:  # pragma: no cover — defensive; never propagate.
        _bump("enqueue_failed_total")
        logger.debug("enqueue failed: %s", exc)
        return

    _bump("signal_enqueued_total")


def enqueue_shown_flip(query_id: str, fact_id: str, shown: bool) -> None:
    """Record whether a candidate was shown to the user.

    LLD-02 §4.9 — replaces the old fake-positive ``recall_hit`` emission.
    Updates ``learning_signals.signal_type`` to ``'shown'`` (or
    ``'not_shown'``) for an existing candidate row. Non-blocking;
    defers the actual UPDATE to the signal_worker via a sentinel batch.
    """
    # Use a zero-candidate batch carrying the flip in ``query_context``.
    batch = SignalBatch(
        profile_id="",
        query_id=query_id,
        query_text="",
        candidates=(),
        query_context={
            "_shown_flip": {"fact_id": fact_id, "shown": bool(shown)},
        },
    )
    enqueue(batch)


def _apply_shown_flip(conn: sqlite3.Connection, batch: SignalBatch) -> None:
    """Apply a shown-flip sentinel batch (see enqueue_shown_flip).

    Updates the signal_type of matching ``(query_id, fact_id)`` rows.
    Never invents reward data (S2 / M1 honesty rule).
    """
    flip = batch.query_context.get("_shown_flip") or {}
    fact_id = flip.get("fact_id")
    shown = bool(flip.get("shown", False))
    if not fact_id or not batch.query_id:
        return
    new_type = "shown" if shown else "not_shown"
    conn.execute(
        "UPDATE learning_signals SET signal_type = ? "
        "WHERE query_id = ? AND fact_id = ?",
        (new_type, batch.query_id, fact_id),
    )


def record_signal_batch(
    conn: sqlite3.Connection, batch: SignalBatch,
) -> list[int]:
    """Synchronous write path used by the signal_worker drain.

    Atomic (S1): signals+features INSERTs inside a single ``with conn:`` TX.
    Privacy (S2): stores only ``query_text_hash``; ``query`` column is empty.
    Handles the empty-candidate case (S3): returns ``[]`` with no side effect.

    Args:
        conn: sqlite3.Connection already configured (WAL, busy_timeout).
              The caller owns the lifecycle.
        batch: A ``SignalBatch``; if it carries a ``_shown_flip`` sentinel
               the UPDATE path runs instead of the INSERT path.

    Returns:
        List of inserted ``learning_signals.id`` values in insert order.
        Empty list if no candidates were present.
    """
    # Shown-flip path — LLD-02 §4.9.
    if batch.query_context and "_shown_flip" in batch.query_context:
        with conn:  # implicit BEGIN/COMMIT
            _apply_shown_flip(conn, batch)
        return []

    if not batch.candidates:
        return []

    # Import lazily — avoids a circular import at module load time.
    from superlocalmemory.learning.features import FeatureExtractor

    query_hash = _hash_query(batch.query_text)
    now_iso = datetime.now(timezone.utc).isoformat(timespec="seconds")
    signal_ids: list[int] = []

    with conn:  # BEGIN ... COMMIT on success, ROLLBACK on exception (S1).
        for i, cand in enumerate(batch.candidates):
            cur = conn.execute(
                "INSERT INTO learning_signals "
                "(profile_id, query, fact_id, signal_type, value, created_at, "
                " query_id, query_text_hash, position, channel_scores, "
                " cross_encoder) "
                "VALUES (?, '', ?, ?, 1.0, ?, ?, ?, ?, ?, ?)",
                (
                    batch.profile_id,
                    cand.fact_id,
                    "candidate",
                    now_iso,
                    batch.query_id,
                    query_hash,
                    i,
                    json.dumps(cand.channel_scores, separators=(",", ":")),
                    cand.cross_encoder_score,
                ),
            )
            sid = cur.lastrowid
            if sid is None:  # pragma: no cover — should not occur.
                raise sqlite3.OperationalError("no lastrowid from signal insert")

            # PERF-v2-02: if ensemble_rerank already built features for this
            # candidate (during the hot path), reuse them instead of calling
            # FeatureExtractor.extract a second time. The reranker stashes a
            # {fact_id: features_json_str} dict under a reserved key on
            # ``query_context``. Cache miss falls through to extract.
            fv_cache = batch.query_context.get(
                "_precomputed_features_json", None,
            ) if isinstance(batch.query_context, dict) else None
            features_json_str: str
            if isinstance(fv_cache, dict) and cand.fact_id in fv_cache:
                raw = fv_cache[cand.fact_id]
                features_json_str = raw if isinstance(raw, str) \
                    else json.dumps(raw, separators=(",", ":"))
            else:
                fv = FeatureExtractor.extract(
                    cand.to_result_dict(), batch.query_context,
                ).features
                features_json_str = json.dumps(fv, separators=(",", ":"))
            # label column is NOT NULL REAL → use 0.0 sentinel (unlabeled).
            # Real label comes from labeler.label_for_row at training time.
            conn.execute(
                "INSERT INTO learning_features "
                "(profile_id, query_id, fact_id, features_json, label, "
                " created_at, signal_id, is_synthetic) "
                "VALUES (?, ?, ?, ?, 0.0, ?, ?, 0)",
                (
                    batch.profile_id,
                    batch.query_id,
                    cand.fact_id,
                    features_json_str,
                    now_iso,
                    sid,
                ),
            )
            signal_ids.append(sid)

    return signal_ids


class LearningSignals:
    """Compute and apply zero-cost learning signals.

    Uses the main memory.db via direct sqlite3 (no engine dependency).
    Thread-safe via lock.
    """

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = str(db_path)
        self._lock = threading.Lock()
        self._ensure_tables()

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _ensure_tables(self) -> None:
        """Create learning signal tables if they don't exist."""
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS channel_credits ("
                    "id INTEGER PRIMARY KEY AUTOINCREMENT, "
                    "profile_id TEXT NOT NULL, "
                    "query_type TEXT NOT NULL, "
                    "channel TEXT NOT NULL, "
                    "hits INTEGER DEFAULT 0, "
                    "total INTEGER DEFAULT 0, "
                    "updated_at TEXT NOT NULL)"
                )
                conn.execute(
                    "CREATE UNIQUE INDEX IF NOT EXISTS idx_channel_credit_unique "
                    "ON channel_credits(profile_id, query_type, channel)"
                )
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS co_retrieval_edges ("
                    "id INTEGER PRIMARY KEY AUTOINCREMENT, "
                    "profile_id TEXT NOT NULL, "
                    "fact_id_a TEXT NOT NULL, "
                    "fact_id_b TEXT NOT NULL, "
                    "co_count INTEGER DEFAULT 1, "
                    "updated_at TEXT NOT NULL)"
                )
                conn.execute(
                    "CREATE UNIQUE INDEX IF NOT EXISTS idx_co_retrieval_unique "
                    "ON co_retrieval_edges(profile_id, fact_id_a, fact_id_b)"
                )
                conn.commit()
            finally:
                conn.close()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, timeout=10)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        conn.row_factory = sqlite3.Row
        return conn

    # ------------------------------------------------------------------
    # Signal 1: Entropy Gap (store-time)
    # ------------------------------------------------------------------

    @staticmethod
    def compute_entropy_gap(
        new_embedding: list[float],
        cluster_embeddings: list[list[float]],
    ) -> float:
        """Compute how surprising new content is relative to existing cluster.

        High gap = surprising content = should get deeper indexing.
        Low gap = redundant content = standard indexing.

        Returns a value in [0.0, 1.0]. >0.7 is 'surprising'.
        """
        if not cluster_embeddings or not new_embedding:
            return 0.5  # neutral when no comparison available

        similarities = []
        for existing in cluster_embeddings:
            sim = _cosine_sim(new_embedding, existing)
            similarities.append(sim)

        avg_sim = sum(similarities) / len(similarities)
        gap = max(0.0, min(1.0, 1.0 - avg_sim))
        return gap

    # ------------------------------------------------------------------
    # Signal 2: Co-Retrieval (recall-time)
    # ------------------------------------------------------------------

    def record_co_retrieval(
        self, profile_id: str, fact_ids: list[str],
    ) -> int:
        """Record that these facts were co-retrieved.

        All pairs of facts in the result set get their co-retrieval
        count incremented. This strengthens implicit graph edges.
        """
        if len(fact_ids) < 2:
            return 0

        now = datetime.now(timezone.utc).isoformat()
        pairs = []
        for i in range(len(fact_ids)):
            for j in range(i + 1, min(len(fact_ids), i + 5)):
                a, b = sorted([fact_ids[i], fact_ids[j]])
                pairs.append((profile_id, a, b, now))

        if not pairs:
            return 0

        with self._lock:
            conn = self._connect()
            try:
                for pid, a, b, ts in pairs:
                    conn.execute(
                        "INSERT INTO co_retrieval_edges "
                        "(profile_id, fact_id_a, fact_id_b, co_count, updated_at) "
                        "VALUES (?, ?, ?, 1, ?) "
                        "ON CONFLICT(profile_id, fact_id_a, fact_id_b) "
                        "DO UPDATE SET co_count = co_count + 1, updated_at = ?",
                        (pid, a, b, ts, ts),
                    )
                conn.commit()
                return len(pairs)
            finally:
                conn.close()

    def get_co_retrieval_boost(
        self, profile_id: str, fact_id: str, top_k: int = 5,
    ) -> list[dict]:
        """Get top co-retrieved facts for boosting."""
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    "SELECT fact_id_a, fact_id_b, co_count FROM co_retrieval_edges "
                    "WHERE profile_id = ? AND (fact_id_a = ? OR fact_id_b = ?) "
                    "ORDER BY co_count DESC LIMIT ?",
                    (profile_id, fact_id, fact_id, top_k),
                ).fetchall()
                results = []
                for r in rows:
                    d = dict(r)
                    other = d["fact_id_b"] if d["fact_id_a"] == fact_id else d["fact_id_a"]
                    results.append({"fact_id": other, "co_count": d["co_count"]})
                return results
            finally:
                conn.close()

    # ------------------------------------------------------------------
    # Signal 3: Channel Credit (recall-time)
    # ------------------------------------------------------------------

    def credit_channel(
        self, profile_id: str, query_type: str, channel: str, hit: bool,
    ) -> None:
        """Credit a retrieval channel for a hit or miss."""
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            conn = self._connect()
            try:
                hit_val = 1 if hit else 0
                conn.execute(
                    "INSERT INTO channel_credits "
                    "(profile_id, query_type, channel, hits, total, updated_at) "
                    "VALUES (?, ?, ?, ?, 1, ?) "
                    "ON CONFLICT(profile_id, query_type, channel) "
                    "DO UPDATE SET hits = hits + ?, total = total + 1, updated_at = ?",
                    (profile_id, query_type, channel, hit_val, now, hit_val, now),
                )
                conn.commit()
            finally:
                conn.close()

    def get_channel_weights(
        self, profile_id: str, query_type: str,
    ) -> dict[str, float]:
        """Get learned channel weights for a query type.

        Returns weight multipliers based on historical hit rates.
        """
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    "SELECT channel, hits, total FROM channel_credits "
                    "WHERE profile_id = ? AND query_type = ? AND total >= 5",
                    (profile_id, query_type),
                ).fetchall()
                if not rows:
                    return {}
                weights = {}
                for r in rows:
                    d = dict(r)
                    rate = d["hits"] / max(d["total"], 1)
                    weights[d["channel"]] = 0.7 + (rate * 0.8)
                return weights
            finally:
                conn.close()

    # ------------------------------------------------------------------
    # Signal 4: Confidence Lifecycle (store + recall time)
    # ------------------------------------------------------------------

    @staticmethod
    def boost_confidence(db_path: str, fact_id: str, amount: float = 0.02) -> None:
        """Boost a fact's confidence on access. Capped at 1.0."""
        try:
            conn = sqlite3.connect(db_path, timeout=10)
            conn.execute("PRAGMA busy_timeout=5000")
            conn.execute(
                "UPDATE atomic_facts SET confidence = MIN(1.0, confidence + ?) "
                "WHERE fact_id = ?",
                (amount, fact_id),
            )
            conn.execute(
                "UPDATE atomic_facts SET access_count = access_count + 1 "
                "WHERE fact_id = ?",
                (fact_id,),
            )
            conn.commit()
            conn.close()
        except Exception:
            pass

    @staticmethod
    def decay_confidence(db_path: str, profile_id: str, rate: float = 0.001) -> int:
        """Decay confidence on unused facts. Floor: 0.1."""
        try:
            conn = sqlite3.connect(db_path, timeout=10)
            conn.execute("PRAGMA busy_timeout=5000")
            cursor = conn.execute(
                "UPDATE atomic_facts SET confidence = MAX(0.1, confidence - ?) "
                "WHERE profile_id = ? AND access_count = 0 "
                "AND created_at < datetime('now', '-7 days')",
                (rate, profile_id),
            )
            conn.commit()
            affected = cursor.rowcount
            conn.close()
            return affected
        except Exception:
            return 0

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_signal_stats(self, profile_id: str) -> dict:
        """Get learning signal statistics for dashboard."""
        with self._lock:
            conn = self._connect()
            try:
                co_rows = conn.execute(
                    "SELECT COUNT(*) AS c, COALESCE(SUM(co_count), 0) AS total "
                    "FROM co_retrieval_edges WHERE profile_id = ?",
                    (profile_id,),
                ).fetchone()
                co = dict(co_rows) if co_rows else {"c": 0, "total": 0}

                ch_rows = conn.execute(
                    "SELECT channel, hits, total FROM channel_credits "
                    "WHERE profile_id = ? ORDER BY total DESC",
                    (profile_id,),
                ).fetchall()
                channels = {
                    dict(r)["channel"]: {
                        "hits": dict(r)["hits"],
                        "total": dict(r)["total"],
                        "rate": round(dict(r)["hits"] / max(dict(r)["total"], 1), 3),
                    }
                    for r in ch_rows
                }

                return {
                    "co_retrieval_edges": co["c"],
                    "co_retrieval_events": co["total"],
                    "channel_performance": channels,
                }
            finally:
                conn.close()


def _cosine_sim(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors."""
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0
    return dot / (norm_a * norm_b)
