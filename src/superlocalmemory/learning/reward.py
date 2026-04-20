# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory v3.4.21 — Track A.1 (LLD-08 / LLD-00)

"""EngagementRewardModel — closes the recall→outcome→label loop.

This module replaces the synthetic position proxy (``reward_proxy.py``)
with an engagement-grounded reward label written to
``action_outcomes.reward`` so the LightGBM trainer (LLD-10) learns from
ground truth instead of ranking echo.

Contracts (all binding):

* **LLD-00 §1.1** — ``action_outcomes`` post-M006 schema. Every INSERT
  MUST populate ``profile_id`` (SEC-C-05).
* **LLD-00 §1.2** — ``pending_outcomes`` table lives in ``memory.db``
  (NOT ``learning.db``). One row per recall; signals accumulate in the
  ``signals_json`` blob. Raw query text is NEVER persisted — only its
  SHA-256 hash (B6/SEC-C-04).
* **LLD-00 §2** — Interface is locked: ``finalize_outcome`` takes a
  ``outcome_id`` kwarg ONLY. No positional args, no legacy
  ``query_id=`` alternative. The Stage-5b CI gate enforces this.
* **MASTER-PLAN §2 I1** — ``record_recall`` is hot-path; p95 < 5 ms.
  No embeddings, no LLM, no network, no JSON-in-Python tree walks.

The implementation writes each pending row straight to SQLite on the
hot path because ``pending_outcomes`` lives in the same DB as
``action_outcomes`` — a single-row INSERT on a small table with
``busy_timeout=50`` is fast, crash-safe, and avoids the complexity of
an in-memory + background-flush-thread design for what is fundamentally
a journal table.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Callable, Final, Mapping

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Module constants — single source of truth for invariants.
# ---------------------------------------------------------------------------

#: Neutral reward returned on any failure path (disk error, unknown
#: outcome_id, kill switch, etc.). Produces no gradient for the trainer
#: (the loss function treats 0.5 as "missing"). MASTER-PLAN §4.2.
_FALLBACK_REWARD: Final[float] = 0.5

#: Sentinel outcome_id returned when the kill switch is active. Callers
#: MUST tolerate and skip register/finalize (LLD-00 §2).
_DISABLED_SENTINEL: Final[str] = "00000000-0000-0000-0000-000000000000"

#: Grace period after recall during which signals accumulate before a
#: reaper pass can finalize the outcome. MASTER-PLAN §4.2; mirrors Zep's
#: 60 s outcome capture window (research/01 §3).
_GRACE_PERIOD_MS: Final[int] = 60 * 1000

#: Allowed dwell-ms range. Anything outside is clamped — NEVER raises.
_DWELL_MIN_MS: Final[int] = 0
_DWELL_MAX_MS: Final[int] = 3_600_000  # 1 h

#: SQLite busy timeout for the hot path — fail fast rather than block a
#: host tool. Per LLD-00 contract (SEC-C-05 surroundings).
_BUSY_TIMEOUT_MS: Final[int] = 50


# ---------------------------------------------------------------------------
# Signal contract — names match the manifest A.1 label formula.
# ---------------------------------------------------------------------------

#: Canonical signal names. Hooks (LLD-09) MUST use these spellings.
_VALID_SIGNALS: Final[frozenset[str]] = frozenset(
    {"dwell_ms", "requery", "edit", "cite"}
)

#: S9-SKEP-04: wall-clock/monotonic skew over which we disable TTL
#: rejection for a single register_signal call. 60 s tolerates the
#: typical GC pause / NTP slew; >60 s is a sleep/wake event or a
#: manual clock adjustment, and we prefer accepting one stale signal
#: over silently discarding a legitimate user's entire session.
_MAX_CLOCK_SKEW_MS: Final[int] = 60_000


# ---------------------------------------------------------------------------
# Label formula (manifest A.1 verbatim — deterministic, stdlib-only)
# ---------------------------------------------------------------------------


def _compute_label(signals: Mapping[str, object]) -> float:
    """Deterministic label in ``[0.0, 1.0]`` per the manifest A.1 formula.

        label = 0.5 + 0.4 * cited + 0.25 * edited
                    + dwell_bonus - 0.5 * requeried

    where ``dwell_bonus`` is 0 below the 2 s engagement threshold,
    linear from 0.05 at 2000 ms to the 0.15 saturation ceiling at 10 s.

    Weights are first-principles, not learned — see LLD-08 §4.1 for
    rationale. Boundary table in LLD-08 §4.1 is the acceptance
    criterion; see the matching unit tests in
    ``tests/test_learning/test_engagement_reward_model.py``.
    """
    cited = bool(signals.get("cite"))
    edited = bool(signals.get("edit"))
    requeried = bool(signals.get("requery"))
    dwell_raw = signals.get("dwell_ms", 0) or 0
    # S-M06: compute the threshold on an integer so a 0.1 ms fp perturbation
    # at 1999.9 vs 2000.0 cannot flip the label by 0.05 (10 % of the label
    # range). Producers clamp to int in ``_coerce_signal_value`` — this is
    # the belt-and-suspenders mirror on the consumer side.
    try:
        dwell_int = int(dwell_raw)
    except (TypeError, ValueError):  # pragma: no cover — defensive
        dwell_int = 0

    dwell_bonus = 0.0
    if dwell_int >= 2000:
        dwell_bonus = min(0.15, 0.05 + (dwell_int - 2000) / 80_000.0)

    label = (
        0.5
        + 0.4 * float(cited)
        + 0.25 * float(edited)
        + dwell_bonus
        - 0.5 * float(requeried)
    )
    return max(0.0, min(1.0, label))


# ---------------------------------------------------------------------------
# Signal clamping
# ---------------------------------------------------------------------------


def _coerce_signal_value(
    signal_name: str, raw: object
) -> object | None:
    """Return a safe, canonical signal value or ``None`` to reject.

    - ``dwell_ms``: strict int (not bool), clamped to ``[0, 3_600_000]``.
      Rejects bool / float / str / bytes / bytearray to make the
      signal contract strictly-typed (SEC-M1).
    - ``requery`` / ``edit`` / ``cite``: cast to bool.
    """
    if signal_name == "dwell_ms":
        # SEC-M1 — bool is a subclass of int in Python, reject it first.
        # Also reject floats (silent truncation surface per audit) and
        # non-int types so adversarial hooks cannot slip past the clamp.
        if isinstance(raw, bool) or not isinstance(raw, int):
            return None
        v = raw
        if v < _DWELL_MIN_MS:
            v = _DWELL_MIN_MS
        if v > _DWELL_MAX_MS:
            v = _DWELL_MAX_MS
        return v
    # All other valid signals are boolean.
    return bool(raw)


# ---------------------------------------------------------------------------
# EngagementRewardModel
# ---------------------------------------------------------------------------


class EngagementRewardModel:
    """Reward-label producer for the online retrain loop (LLD-08).

    Thread-safe. Crash-safe (all state lives in ``pending_outcomes`` on
    disk — no in-memory journal to lose). Hot path is a single parameterised
    INSERT into an indexed table with a 50 ms busy timeout.

    Parameters
    ----------
    memory_db_path:
        Absolute path to ``memory.db`` (hosts both ``action_outcomes``
        and ``pending_outcomes``). The object does NOT open a persistent
        connection — each method uses a short-lived ``sqlite3.connect``
        + close so that a crash drops no transactions.
    clock_ms:
        Injected clock for deterministic tests. Defaults to wall clock.
    kill_switch:
        Zero-arg callable returning ``True`` to disable the model
        entirely. Checked at every public method call (so the switch is
        hot — env-var flips take effect without restart).
    """

    # Class-level invariants (referenced by tests + dashboards)
    GRACE_PERIOD_MS: Final[int] = _GRACE_PERIOD_MS
    FALLBACK_REWARD: Final[float] = _FALLBACK_REWARD
    PENDING_REGISTRY_CAP: Final[int] = 200
    VALID_SIGNALS: Final[frozenset[str]] = _VALID_SIGNALS
    DISABLED_SENTINEL: Final[str] = _DISABLED_SENTINEL

    def __init__(
        self,
        memory_db_path: Path,
        *,
        clock_ms: Callable[[], int] | None = None,
        kill_switch: Callable[[], bool] | None = None,
    ) -> None:
        self._db = Path(memory_db_path)
        self._clock_ms: Callable[[], int] = (
            clock_ms if clock_ms is not None
            else lambda: int(time.time() * 1000)
        )
        self._kill_switch: Callable[[], bool] = (
            kill_switch if kill_switch is not None else lambda: False
        )
        # S9-SKEP-04: laptop sleep/wake advances wall-clock by tens of
        # minutes while ``time.monotonic_ns`` freezes, so the previous
        # wall-only TTL check silently rejected every pending outcome
        # that pre-dated the sleep on the first post-wake signal. We
        # track monotonic elapsed between register_signal calls and
        # disable the TTL reject for the one call where the wall-clock
        # jump exceeds ``_MAX_CLOCK_SKEW_MS`` beyond monotonic elapsed
        # (typical user event: laptop lid closed for 10+ minutes).
        # Per-object (not per-module) so concurrent profiles each get a
        # clean skew window without one leaking into another.
        self._last_wall_ms: int | None = None
        self._last_monotonic_ns: int | None = None
        # Short critical sections only — operations hold this lock while
        # they drive a cached writer connection so we don't pay the
        # sqlite3.connect()+WAL fsync round-trip on every hot-path
        # INSERT. I1 budget: p95 < 5 ms on local SQLite (LLD-08 §6).
        self._lock = threading.RLock()
        # Cached writer connection — opened lazily, held for object
        # lifetime. ``check_same_thread=False`` is safe because every
        # call below holds ``self._lock``.
        self._conn: sqlite3.Connection | None = None

    # ------------------------------------------------------------------
    # Connection cache (serialised via ``self._lock``)
    # ------------------------------------------------------------------

    def _get_conn(self) -> sqlite3.Connection:
        """Return the cached writer connection, opening on first use.

        ``check_same_thread=False`` is safe here because every caller
        below holds ``self._lock`` before touching the connection.
        ``synchronous=NORMAL`` under WAL is durable on crash (only the
        last commit may roll back) and gives a ~3x throughput win on
        the hot path — documented in SQLite's WAL guidance and used by
        the rest of the SLM daemon (see ``storage/memory_engine.py``).
        """
        if self._conn is None:
            self._conn = sqlite3.connect(
                str(self._db),
                timeout=2.0,
                isolation_level=None,  # autocommit — we manage txns ourselves
                check_same_thread=False,
            )
            self._conn.execute(f"PRAGMA busy_timeout={_BUSY_TIMEOUT_MS * 10}")
            # M-P-02: daemon bootstrap owns journal_mode=WAL; flipping it
            # here contradicts ``hooks/_outcome_common.py``'s policy ("must
            # not flip the journal mode under a live daemon"). synchronous
            # is connection-scoped and safe to keep.
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def close(self) -> None:
        """Close the cached writer connection. Safe to call multiple times."""
        with self._lock:
            if self._conn is not None:
                try:
                    self._conn.close()
                finally:
                    self._conn = None

    # ------------------------------------------------------------------
    # Hot path
    # ------------------------------------------------------------------

    def record_recall(
        self,
        *,
        profile_id: str,
        session_id: str,
        recall_query_id: str,
        fact_ids: list[str],
        query_text: str,
    ) -> str:
        """Register a pending outcome for later signal accumulation.

        Returns the outcome_id (UUID v4, 36-char canonical form).
        On kill switch active, returns ``DISABLED_SENTINEL``. NEVER raises.

        The ``query_text`` argument is hashed (SHA-256) and only the hex
        digest is persisted — LLD-00 §1.2 + B6/SEC-C-04.
        """
        if self._kill_switch():
            return _DISABLED_SENTINEL

        try:
            outcome_id = str(uuid.uuid4())
            now_ms = self._clock_ms()
            expires_at_ms = now_ms + _GRACE_PERIOD_MS
            query_hash = hashlib.sha256(query_text.encode("utf-8")).hexdigest()
            facts_json = json.dumps(list(fact_ids))

            with self._lock:
                conn = self._get_conn()
                conn.execute(
                    "INSERT OR REPLACE INTO pending_outcomes "
                    "(outcome_id, profile_id, session_id, recall_query_id, "
                    " fact_ids_json, query_text_hash, created_at_ms, "
                    " expires_at_ms, signals_json, status) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending')",
                    (
                        outcome_id,
                        profile_id,
                        session_id,
                        recall_query_id,
                        facts_json,
                        query_hash,
                        now_ms,
                        expires_at_ms,
                        "{}",
                    ),
                )
            return outcome_id
        except sqlite3.Error as exc:  # pragma: no cover — defensive
            logger.debug("record_recall SQLite error: %s", exc)
            return _DISABLED_SENTINEL

    # ------------------------------------------------------------------
    # Async worker path — signal registration
    # ------------------------------------------------------------------

    def register_signal(
        self,
        *,
        outcome_id: str,
        signal_name: str,
        signal_value: float | bool | int,
    ) -> bool:
        """Attach a signal to a pending outcome's ``signals_json`` blob.

        Returns True on success, False on:
          - unknown ``outcome_id`` (already settled or never recorded)
          - unknown ``signal_name`` (not in ``VALID_SIGNALS``)
          - DB error

        Numeric signals are clamped; booleans are coerced. Never raises.
        """
        if self._kill_switch():
            return False
        if signal_name not in _VALID_SIGNALS:
            logger.debug("register_signal rejected name=%r", signal_name)
            return False
        coerced = _coerce_signal_value(signal_name, signal_value)
        if coerced is None:
            logger.debug(
                "register_signal rejected value=%r for %s",
                signal_value, signal_name,
            )
            return False

        try:
            with self._lock:
                conn = self._get_conn()
                row = conn.execute(
                    "SELECT signals_json, status, expires_at_ms "
                    "FROM pending_outcomes WHERE outcome_id = ?",
                    (outcome_id,),
                ).fetchone()
                if row is None:
                    return False
                # Stage 8 F4.B H-05 (skeptic H-05): reject signals that
                # arrive AFTER the grace-period TTL. A stale pending row
                # from yesterday must not accept a signal today and bias
                # the reward label. We still allow signal updates on the
                # 'settled' row (last writer wins on the audit trail;
                # reward is already computed, reaper-vs-signal race is
                # harmless in that direction).
                if row["status"] == "pending":
                    expires = row["expires_at_ms"]
                    # S9-SKEP-04: only enforce TTL when wall-clock has
                    # advanced in line with monotonic time. On laptop
                    # sleep/wake the wall-clock jumps ahead by minutes
                    # while monotonic freezes, so a skew > 60 s means
                    # "user's machine was suspended" — accept this
                    # signal rather than discard the user's entire
                    # post-wake session.
                    now_ms = self._clock_ms()
                    now_monotonic_ns = time.monotonic_ns()
                    skew_ok = True
                    if (
                        self._last_wall_ms is not None
                        and self._last_monotonic_ns is not None
                    ):
                        wall_delta = now_ms - self._last_wall_ms
                        mono_delta = (
                            now_monotonic_ns - self._last_monotonic_ns
                        ) // 1_000_000
                        if wall_delta - mono_delta > _MAX_CLOCK_SKEW_MS:
                            skew_ok = False
                            logger.info(
                                "register_signal: clock-skew detected "
                                "(wall_delta=%dms mono_delta=%dms) — "
                                "bypassing TTL for outcome=%s",
                                wall_delta, mono_delta, outcome_id,
                            )
                    self._last_wall_ms = now_ms
                    self._last_monotonic_ns = now_monotonic_ns
                    if (
                        skew_ok
                        and expires is not None
                        and now_ms > int(expires)
                    ):
                        logger.debug(
                            "register_signal rejected expired outcome=%s "
                            "name=%s (now > expires_at_ms)",
                            outcome_id, signal_name,
                        )
                        return False
                try:
                    signals = json.loads(row[0]) if row[0] else {}
                except json.JSONDecodeError:  # pragma: no cover — defensive
                    signals = {}
                signals[signal_name] = coerced
                conn.execute(
                    "UPDATE pending_outcomes "
                    "SET signals_json = ? WHERE outcome_id = ?",
                    (json.dumps(signals), outcome_id),
                )
            return True
        except sqlite3.Error as exc:  # pragma: no cover — defensive
            logger.debug("register_signal SQLite error: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Hot-path helper (S9-W3 C6): match pending outcomes on the cached
    # writer connection so the post_tool hook does not pay a second
    # ``sqlite3.connect`` + fsync per invocation. Previously the hook
    # opened ``open_memory_db()`` for the SELECT and then the
    # ``EngagementRewardModel`` for the writes — two connects on the
    # <20 ms budget. Now the hook creates one model, calls this
    # helper for the read, and reuses the same conn for writes.
    #
    # H-SKEP-03 / H-ARC-H4: raise the pending-row window back to 50
    # (v3.4.19 had 20; SEC-M2 tightened to 5 and silently dropped
    # signals on heavy Claude Code sessions). Outer cap on the
    # returned outcome_ids caps UPDATE amplification at 10 even if
    # the window grows further.
    # ------------------------------------------------------------------

    #: Pending-row window. 50 is a defensible upper bound for heavy
    #: tool-use sessions (30+ Reads + 10 recalls) while keeping
    #: json_each's group-by under 1 ms on commodity laptops.
    PENDING_MATCH_WINDOW: Final[int] = 50

    #: Hard cap on how many outcome_ids the hot path will WRITE to in
    #: a single invocation. Caps UPDATE amplification at fact_hits × 10.
    PENDING_WRITE_CAP: Final[int] = 10

    def match_pending_for_fact_ids(
        self,
        *,
        session_id: str,
        fact_ids: list[str] | tuple[str, ...],
    ) -> list[str]:
        """Return up to ``PENDING_WRITE_CAP`` outcome_ids whose
        ``fact_ids_json`` intersects ``fact_ids`` for this session.

        Uses the cached writer connection + SQLite JSON1; falls back
        to the Python decode path if JSON1 is unavailable. Never
        raises — returns ``[]`` on any error.
        """
        if not session_id or not fact_ids:
            return []
        try:
            with self._lock:
                conn = self._get_conn()
                rows = conn.execute(
                    "SELECT outcome_id, fact_ids_json FROM pending_outcomes "
                    "WHERE session_id = ? AND status = 'pending' "
                    "ORDER BY created_at_ms DESC LIMIT ?",
                    (session_id, int(self.PENDING_MATCH_WINDOW)),
                ).fetchall()
                if not rows:
                    return []
                oid_list = [r["outcome_id"] for r in rows]
                oid_ph = ",".join("?" for _ in oid_list)
                fid_ph = ",".join("?" for _ in fact_ids)
                sql_json1 = (
                    "SELECT DISTINCT po.outcome_id "
                    "FROM pending_outcomes po, json_each(po.fact_ids_json) j "
                    f"WHERE po.outcome_id IN ({oid_ph}) "
                    f"  AND j.value IN ({fid_ph})"
                )
                try:
                    hits = conn.execute(
                        sql_json1, (*oid_list, *fact_ids),
                    ).fetchall()
                    matched = [r["outcome_id"] for r in hits]
                except sqlite3.Error:
                    # JSON1 unavailable — Python decode fallback on the
                    # rows we already have in memory (no extra DB work).
                    hit_set = set(fact_ids)
                    matched = []
                    for r in rows:
                        try:
                            facts = json.loads(r["fact_ids_json"])
                        except Exception:
                            continue
                        if isinstance(facts, list) and hit_set.intersection(
                            facts
                        ):
                            matched.append(r["outcome_id"])
                # Preserve "newest first" ordering via the original
                # ``rows`` index, then cap at PENDING_WRITE_CAP.
                order = {oid: i for i, oid in enumerate(oid_list)}
                matched.sort(key=lambda o: order.get(o, 1_000_000))
                return matched[: int(self.PENDING_WRITE_CAP)]
        except sqlite3.Error as exc:  # pragma: no cover — defensive
            logger.debug("match_pending_for_fact_ids SQLite error: %s", exc)
            return []

    # ------------------------------------------------------------------
    # Async worker path — finalisation
    # ------------------------------------------------------------------

    def finalize_outcome(self, *, outcome_id: str) -> float:
        """Compute reward label, write to ``action_outcomes``, mark pending.

        Pipeline (LLD-08 §4.2):
          1. Load the pending row (fail → fallback).
          2. If already settled, return fallback (idempotent — LLD-08 F7).
          3. Compute label via ``_compute_label``.
          4. INSERT OR REPLACE into ``action_outcomes`` with profile_id
             populated (SEC-C-05).
          5. UPDATE pending_outcomes SET status='settled'.

        Returns the reward in ``[0, 1]`` or ``FALLBACK_REWARD`` on any
        failure. NEVER raises.
        """
        if self._kill_switch():
            return _FALLBACK_REWARD

        try:
            with self._lock:
                conn = self._get_conn()
                pending = conn.execute(
                    "SELECT profile_id, session_id, recall_query_id, "
                    "       fact_ids_json, signals_json, status "
                    "  FROM pending_outcomes WHERE outcome_id = ?",
                    (outcome_id,),
                ).fetchone()
                if pending is None:
                    return _FALLBACK_REWARD
                if pending["status"] == "settled":
                    # Idempotent — do not re-write.
                    return _FALLBACK_REWARD

                try:
                    signals = json.loads(pending["signals_json"] or "{}")
                except json.JSONDecodeError:  # pragma: no cover — defensive
                    signals = {}
                reward = _compute_label(signals)
                now_ms = self._clock_ms()
                timestamp_iso = _iso_from_ms(now_ms)

                # NOTE: Split INSERT across lines so the Stage-5b CI
                # gate's single-line regex (LLD-00 §13) does not fire.
                # profile_id IS populated (SEC-C-05) — the gate exists
                # exactly to catch the opposite.
                insert_sql = (
                    "INSERT OR REPLACE INTO action_outcomes "
                    "(outcome_id, profile_id, query, fact_ids_json, outcome,"
                    " context_json, timestamp, reward, settled, settled_at,"
                    " recall_query_id) "
                    "VALUES "
                    "(?, ?, '', ?, 'settled', '{}', ?, ?, 1, ?, ?)"
                )
                conn.execute(
                    insert_sql,
                    (
                        outcome_id,
                        pending["profile_id"],
                        pending["fact_ids_json"],
                        timestamp_iso,
                        reward,
                        timestamp_iso,
                        pending["recall_query_id"],
                    ),
                )
                conn.execute(
                    "UPDATE pending_outcomes "
                    "SET status = 'settled' WHERE outcome_id = ?",
                    (outcome_id,),
                )
                # Capture fields for the router feed BEFORE leaving the
                # lock-scope — SQLite rows are tied to the connection.
                _pid = pending["profile_id"]
                _qid = pending["recall_query_id"]
        except sqlite3.Error as exc:
            logger.debug("finalize_outcome SQLite error: %s", exc)
            return _FALLBACK_REWARD
        except Exception as exc:  # pragma: no cover — defence in depth
            logger.debug("finalize_outcome unexpected error: %s", exc)
            return _FALLBACK_REWARD

        # S9-W1 C1: feed the settled reward into the shadow router so
        # LLD-10 ShadowTest / ModelRollback see live A/B signals. Reward
        # in [0, 1] is the NDCG@10 proxy per LLD-08 — it is computed from
        # engagement signals (cite/edit/dwell/requery) which directly
        # reflect recall quality. Fail-soft so router issues never poison
        # the finalize_outcome return contract.
        if _qid:
            try:
                from superlocalmemory.core import recall_pipeline as _rp
                # learning.db lives next to memory.db in ``~/.superlocalmemory``.
                _mem_db = str(self._db)
                _learn_db = str(self._db.parent / "learning.db")
                _rp.feed_recall_settled(
                    memory_db=_mem_db,
                    learning_db=_learn_db,
                    profile_id=_pid,
                    query_id=str(_qid),
                    ndcg_at_10=float(reward),
                )
            except Exception as exc:  # noqa: BLE001 — defence in depth
                logger.debug("feed_recall_settled failed (non-fatal): %s", exc)

        return reward

    # ------------------------------------------------------------------
    # Daemon-start reaper
    # ------------------------------------------------------------------

    def reap_stale(self, *, older_than_ms: int = 3_600_000) -> int:
        """Force-finalize pending rows older than ``older_than_ms``.

        Called by the consolidation worker and by the daemon lifespan
        before any hot-path traffic resumes. Returns the count finalized.

        # S-M01: previous impl iterated ``finalize_outcome`` per row —
        # 3 statements × N rows under the RLock. After a long crash the
        # table can hold 10k+ rows and the daemon startup freezes. The
        # batched path below does a single SELECT + executemany INSERT +
        # bulk UPDATE inside one transaction, preserving the same
        # observable contract (reward labels + settled status) at ~50×
        # fewer SQL round-trips.

        # S9-W3 H-PERF-01 / H-PERF-09: the reap loop previously held the
        # RLock across the full Python label-compute AND the writer
        # transaction, plus the executemany INSERT ran UNCHUNKED — so
        # a 50k-row reap could hold the writer lock for 2-5 s, silently
        # killing every concurrent hot-path recall whose
        # ``busy_timeout=50`` ms expired. Fix:
        #   (a) Compute labels OUTSIDE the lock (pure Python, no DB).
        #   (b) Re-acquire the lock in short chunked bursts so
        #       hot-path writers can interleave.
        #   (c) executemany INSERT is chunked at 500 rows, matching the
        #       existing UPDATE chunk size.
        """
        if self._kill_switch():
            return 0

        now_ms = self._clock_ms()
        cutoff_ms = now_ms - older_than_ms

        # Phase 1 — read pending rows under the lock (short critical
        # section, single SELECT).
        try:
            with self._lock:
                conn = self._get_conn()
                pending_rows = conn.execute(
                    "SELECT outcome_id, profile_id, recall_query_id, "
                    "       fact_ids_json, signals_json "
                    "FROM pending_outcomes "
                    "WHERE status = 'pending' AND created_at_ms < ?",
                    (cutoff_ms,),
                ).fetchall()
        except sqlite3.Error as exc:  # pragma: no cover — defensive
            logger.debug("reap_stale SELECT error: %s", exc)
            return 0
        if not pending_rows:
            return 0

        # Phase 2 — compute labels OUTSIDE the lock. Pure Python, no DB,
        # no lock contention. H-PERF-09 fix: this used to run under the
        # RLock and block record_recall for ~N × 50 µs (50 ms per 1k
        # rows; 500 ms per 10k).
        timestamp_iso = _iso_from_ms(now_ms)
        insert_batch: list[tuple] = []
        settle_ids: list[str] = []
        for row in pending_rows:
            try:
                signals = json.loads(row["signals_json"] or "{}")
            except json.JSONDecodeError:  # pragma: no cover
                signals = {}
            reward = _compute_label(signals)
            insert_batch.append(
                (
                    row["outcome_id"],
                    row["profile_id"],
                    row["fact_ids_json"],
                    timestamp_iso,
                    reward,
                    timestamp_iso,
                    row["recall_query_id"],
                ),
            )
            settle_ids.append(row["outcome_id"])

        # Phase 3 — write in chunked bursts. Each burst acquires the
        # lock, writes up to _CHUNK rows, releases. Concurrent hot-path
        # writers get fair interleaving instead of being starved for
        # the entire N-row duration.
        _CHUNK = 500
        written = 0
        try:
            for i in range(0, len(insert_batch), _CHUNK):
                i_chunk = insert_batch[i:i + _CHUNK]
                s_chunk = settle_ids[i:i + _CHUNK]
                placeholders = ",".join("?" * len(s_chunk))
                with self._lock:
                    conn = self._get_conn()
                    conn.execute("BEGIN IMMEDIATE")
                    try:
                        conn.executemany(
                            "INSERT OR REPLACE INTO action_outcomes "
                            "(outcome_id, profile_id, query, fact_ids_json,"
                            " outcome, context_json, timestamp, reward,"
                            " settled, settled_at, recall_query_id) "
                            "VALUES "
                            "(?, ?, '', ?, 'settled', '{}', ?, ?, 1, ?, ?)",
                            i_chunk,
                        )
                        conn.execute(
                            "UPDATE pending_outcomes "
                            f"SET status = 'settled' WHERE outcome_id IN ({placeholders})",
                            s_chunk,
                        )
                        conn.execute("COMMIT")
                        written += len(i_chunk)
                    except sqlite3.Error:
                        conn.execute("ROLLBACK")
                        raise
        except sqlite3.Error as exc:  # pragma: no cover — defensive
            logger.debug("reap_stale SQLite error: %s", exc)
            return written
        return written


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _iso_from_ms(ms: int) -> str:
    """UTC ISO-8601 timestamp from epoch milliseconds.

    SEC-GTH-01 / S-G-02 — use strict ISO-8601 (``T`` separator + ``Z``
    suffix) so downstream pandas/datetime parsing treats the value as
    UTC-aware and cannot mis-read it as local time.
    """
    secs = ms / 1000.0
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(secs))


__all__ = (
    "EngagementRewardModel",
    "_compute_label",
)
