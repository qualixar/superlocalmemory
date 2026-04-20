# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory v3.4.22 — LLD-11 §Budget

"""Evolution budget enforcement.

Hard caps per MASTER-PLAN §4.4 + LLD-11:
  - Wall-time per cycle: 30 minutes (1800 seconds)
  - LLM calls per cycle: 10
  - Cycles per day per profile: 3
  - Single-flight via per-profile lock file resolved with
    ``safe_resolve_identifier`` (LLD-00 §4)

All four constraints are non-negotiable — crossing any of them raises
``BudgetExhausted`` so the caller can abort the cycle safely without
poisoning ``action_outcomes`` or the recall pipeline.

Author: Varun Pratap Bhardwaj / Qualixar
"""

from __future__ import annotations

import fcntl
import logging
import os
import sqlite3
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from superlocalmemory.core.security_primitives import safe_resolve_identifier

logger = logging.getLogger(__name__)


# M-P-07: path-component hints for filesystems where ``fcntl.flock`` is
# known to degrade to a no-op. We do NOT abort — some users accept the
# risk — but we emit a one-shot warning so a double-cycle is at least
# attributable to a known root cause. If/when we ship a threading-only
# fallback, this list becomes the trigger for switching modes.
_SYNC_ROOT_HINTS: tuple[str, ...] = (
    "iCloud Drive",
    "Library/Mobile Documents",       # macOS iCloud backing store
    "OneDrive",
    "Dropbox",
    "Google Drive",
    "pCloudDrive",
)
_WARNED_SYNC_PATHS: set[str] = set()


def _detect_sync_root(path: Path) -> str | None:
    """Return the matching sync-root hint if ``path`` lives under one."""
    try:
        parts = tuple(path.resolve().parts)
    except Exception:  # pragma: no cover — path resolution failures
        parts = path.parts
    joined = "/".join(parts)
    for hint in _SYNC_ROOT_HINTS:
        if hint in joined:
            return hint
    return None


# ---------------------------------------------------------------------------
# Contract constants (non-negotiable per MASTER-PLAN §4.4)
# ---------------------------------------------------------------------------

MAX_WALL_TIME_SEC: int = 1800        # 30 minutes per cycle
MAX_LLM_CALLS_PER_CYCLE: int = 10    # 10 LLM calls per cycle
MAX_CYCLES_PER_DAY: int = 3          # 3 cycles per profile per UTC day


class BudgetExhausted(RuntimeError):
    """Raised when a budget dimension is crossed.

    The ``dimension`` attribute records which cap was hit so callers can
    log structured metrics without string-parsing the message.
    """

    def __init__(self, dimension: str, detail: str = "") -> None:
        self.dimension = dimension
        suffix = f": {detail}" if detail else ""
        super().__init__(f"budget exhausted [{dimension}]{suffix}")


class EvolutionBudget:
    """Per-profile, per-cycle budget gate.

    Usage::

        budget = EvolutionBudget(profile_id="default",
                                 learning_db=Path("~/.slm/learning.db"),
                                 lock_dir=Path("~/.superlocalmemory"))
        with budget.cycle():
            budget.check_time()
            budget.charge_llm_call()
            ...

    The ``cycle()`` context manager acquires a single-flight lock (via
    ``safe_resolve_identifier`` on the profile name) and enforces the
    3-cycles-per-day cap. Inside the cycle, callers must call
    ``check_time()`` before each expensive step and ``charge_llm_call()``
    before each LLM dispatch.
    """

    def __init__(
        self,
        *,
        profile_id: str,
        learning_db: Path | str,
        lock_dir: Path | str,
    ) -> None:
        self._profile_id = profile_id
        self._learning_db = Path(learning_db)
        self._lock_dir = Path(lock_dir)
        self._lock_dir.mkdir(parents=True, exist_ok=True)
        # Resolve lock path through the LLD-00 §4 safe helper so a malicious
        # profile_id cannot escape the lock directory. The helper regex
        # rejects '.' so we hand it the identifier portion only, then
        # append the fixed ``.lock`` suffix to the validated path.
        safe_stem = safe_resolve_identifier(
            self._lock_dir, f"evolution-{profile_id}",
        )
        self._lock_path = safe_stem.with_suffix(".lock")
        # M-P-07: warn once per lock-path when we detect the lock sits on
        # a known sync-backed filesystem. ``fcntl.flock`` silently
        # degrades on iCloud/OneDrive/Dropbox — two concurrent cycles
        # would each acquire and burn LLM budget. Documentation-only for
        # now; a future release may refuse to run until the lock_dir is
        # moved off the sync root.
        try:
            root_hint = _detect_sync_root(self._lock_path)
        except Exception:  # pragma: no cover — defensive
            root_hint = None
        if root_hint is not None:
            key = str(self._lock_path)
            if key not in _WARNED_SYNC_PATHS:
                _WARNED_SYNC_PATHS.add(key)
                logger.warning(
                    "evolution lock at %s lives under %r — fcntl.flock "
                    "may silently no-op on sync-backed filesystems. "
                    "Concurrent cycles could double-bill LLM cost. Move "
                    "the lock_dir off the sync root to make single-flight "
                    "enforceable.",
                    key, root_hint,
                )

        self._cycle_start_mono: float | None = None
        self._llm_calls_this_cycle: int = 0
        self._lock_fd: int | None = None

    # ------------------------------------------------------------------
    # Per-day cycle accounting (sqlite-backed via evolution_config.last_cycle_at
    # + cycles_this_week; today-count is derived from last_cycle_at day)
    # ------------------------------------------------------------------

    def _count_cycles_today(self) -> int:
        """Count cycles recorded for ``profile_id`` on the current UTC day.

        Reads from ``evolution_llm_cost_log`` — ``cycle_id`` distinct values
        scoped to today. Evolution cycles stamp a ``cycle_id`` on every
        cost-log row they emit, so distinct cycle_ids == distinct cycles.

        A cycle with zero LLM calls still needs to count; the budget stamps
        a zero-row sentinel via ``_record_cycle_start`` on acquire.
        """
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        conn = sqlite3.connect(self._learning_db)
        try:
            row = conn.execute(
                "SELECT COUNT(DISTINCT cycle_id) "
                "FROM evolution_llm_cost_log "
                "WHERE profile_id=? AND substr(ts,1,10)=? "
                "  AND cycle_id IS NOT NULL",
                (self._profile_id, today),
            ).fetchone()
            return int(row[0]) if row and row[0] is not None else 0
        finally:
            conn.close()

    def _record_cycle_start(self, cycle_id: str) -> None:
        """Write a sentinel row so this cycle is counted toward the daily cap.

        Tokens/cost are zero — the real LLM cost rows land as
        ``charge_llm_call`` is invoked by the dispatcher.

        H-16 (Stage 8): profile_id must be a non-empty string. Enforced
        here so a mis-constructed EvolutionBudget fails at the first
        write instead of silently attributing cost to an empty bucket.
        """
        if not isinstance(self._profile_id, str) or not self._profile_id.strip():
            raise ValueError(
                "EvolutionBudget.profile_id must be a non-empty string "
                f"(got {self._profile_id!r})"
            )
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        conn = sqlite3.connect(self._learning_db)
        try:
            conn.execute(
                "INSERT INTO evolution_llm_cost_log "
                "(profile_id, ts, model, tokens_in, tokens_out, cost_usd, cycle_id) "
                "VALUES (?,?,?,?,?,?,?)",
                (self._profile_id, now, "cycle-start", 0, 0, 0.0, cycle_id),
            )
            conn.commit()
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Public runtime checks
    # ------------------------------------------------------------------

    def check_time(self) -> None:
        """Raise ``BudgetExhausted`` if wall-time cap exceeded."""
        if self._cycle_start_mono is None:
            raise RuntimeError("check_time() called outside cycle()")
        elapsed = time.monotonic() - self._cycle_start_mono
        if elapsed > MAX_WALL_TIME_SEC:
            raise BudgetExhausted(
                "wall_time",
                f"elapsed {elapsed:.1f}s > {MAX_WALL_TIME_SEC}s",
            )

    def charge_llm_call(self) -> None:
        """Charge one LLM call toward the per-cycle cap.

        Raises ``BudgetExhausted`` AFTER the cap is already exhausted.
        Call this BEFORE dispatching the LLM so the cap is protective.
        """
        if self._cycle_start_mono is None:
            raise RuntimeError("charge_llm_call() called outside cycle()")
        if self._llm_calls_this_cycle >= MAX_LLM_CALLS_PER_CYCLE:
            raise BudgetExhausted(
                "llm_calls",
                f"charged {self._llm_calls_this_cycle} "
                f"(cap {MAX_LLM_CALLS_PER_CYCLE})",
            )
        self._llm_calls_this_cycle += 1

    # ------------------------------------------------------------------
    # Context manager — single-flight + daily cap + wall-time init
    # ------------------------------------------------------------------

    @contextmanager
    def cycle(self, cycle_id: str | None = None) -> Iterator["EvolutionBudget"]:
        """Acquire single-flight lock + enforce daily cap + start wall timer."""
        # Daily cap check BEFORE taking the lock — a blocked cycle must
        # not hold the file lock while other cycles wait.
        today_count = self._count_cycles_today()
        if today_count >= MAX_CYCLES_PER_DAY:
            raise BudgetExhausted(
                "cycles_per_day",
                f"profile={self._profile_id} today={today_count} "
                f"cap={MAX_CYCLES_PER_DAY}",
            )

        # Single-flight lock (non-blocking flock). A second concurrent
        # acquire raises BlockingIOError — surface as BudgetExhausted.
        fd = os.open(str(self._lock_path), os.O_CREAT | os.O_RDWR, 0o600)
        try:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as e:
                os.close(fd)
                raise BudgetExhausted(
                    "single_flight",
                    f"another cycle holds {self._lock_path}",
                ) from e
            self._lock_fd = fd
        except BudgetExhausted:
            raise
        except Exception:
            try:
                os.close(fd)
            except OSError:
                pass
            raise

        # Write pid marker (best-effort, never fails the acquire).
        try:
            os.write(fd, f"{os.getpid()}:{self._profile_id}\n".encode())
        except OSError:
            pass

        cid = cycle_id or (
            datetime.now(timezone.utc).strftime("cyc-%Y%m%d-%H%M%S-")
            + uuid.uuid4().hex[:8]
        )
        try:
            self._record_cycle_start(cid)
        except sqlite3.Error as e:
            # Release lock before propagating — a failed cycle-record should
            # not leave the lock dangling.
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)
                self._lock_fd = None
            raise RuntimeError(
                f"failed to record cycle start: {e}",
            ) from e

        self._cycle_start_mono = time.monotonic()
        self._llm_calls_this_cycle = 0

        try:
            yield self
        finally:
            self._cycle_start_mono = None
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                try:
                    os.close(fd)
                except OSError:
                    pass
                self._lock_fd = None
