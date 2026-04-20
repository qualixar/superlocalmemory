# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory v3.4.21 — Stage 8 SB-1 / Track A.3 (LLD-10)

"""Daemon-resident recall-time A/B router for LLD-10 shadow + rollback.

This module is the single seam that wires ``ShadowTest`` (pre-promotion)
and ``ModelRollback`` (post-promotion) into the live recall path. Before
this module existed (pre-v3.4.21 Stage 8 SB-1) both classes were
defined + unit-tested but had zero production callers in ``src/``.

Design constraints:

  * Process-local state only — one ``ShadowRouter`` instance per
    ``(memory_db, learning_db, profile_id)`` tuple, held by the
    daemon's in-process singleton cache.
  * Deterministic A/B routing — ``route_query(qid)`` uses
    ``sha256(install_token + qid)`` so an attacker who controls
    ``qid`` still cannot bias the split without reading the install
    token on disk (closes skeptic H-02 + H-03).
  * Promotion and rollback are DELEGATED to the canonical helpers in
    ``learning/consolidation_worker`` and ``learning/model_rollback``.
    The router never writes to ``learning_model_state`` directly.
  * Fail-soft: any exception during recall-time ingestion is logged
    and swallowed — we must not break the user's recall path.

References:
  - LLD-00 §8    — two-phase shadow + auto-rollback.
  - LLD-10 §4.1  — deterministic A/B routing.
  - LLD-10 §5    — atomic BEGIN IMMEDIATE promotion + rollback.
  - Stage 8 SB-1 — architect S8-ARC-C1, skeptic C-01/C-02/H-07.
"""

from __future__ import annotations

import hashlib
import logging
import threading
from typing import Final

from superlocalmemory.core.security_primitives import ensure_install_token
from superlocalmemory.learning.model_rollback import ModelRollback
from superlocalmemory.learning.shadow_test import ShadowTest

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Arm vocabulary — externalised so hooks can grep-check.
# ---------------------------------------------------------------------------

ARM_BASELINE: Final[str] = "baseline"
ARM_CANDIDATE: Final[str] = "candidate"


class ShadowRouter:
    """Per-profile A/B router + post-promotion watcher.

    Not thread-safe for write state (`_shadow`, `_rollback`) — the daemon
    calls ``on_recall_settled`` from a single recall-settled worker queue.
    Reads (``route_query``) are safe under concurrency.
    """

    def __init__(
        self,
        *,
        memory_db: str,
        learning_db: str,
        profile_id: str,
    ) -> None:
        self._memory_db = str(memory_db)
        self._learning_db = str(learning_db)
        self._profile_id = str(profile_id)
        # Shadow pre-promotion accumulator (created lazily when a
        # candidate is persisted).
        self._shadow: ShadowTest | None = None
        self._candidate_id: int | None = None
        # S9-SKEP-11: pin the routing token at router creation.
        # ``install_token`` rotation is a documented user action (SEC-M07
        # ``slm escape-hatch rotate-token``) but must not flip the arm
        # assignment of every in-flight query_id — that would silently
        # mix baseline and candidate observations for the same qid and
        # break the paired-t assumption. We snapshot the token once at
        # init; subsequent rotations are picked up only when a new
        # router is created (i.e. at daemon restart / candidate rotate).
        try:
            self._routing_token = ensure_install_token()
        except Exception:  # pragma: no cover — defensive
            self._routing_token = ""
        # Rollback post-promotion watcher (created when arm_post_promotion
        # is called after a promote).
        self._rollback: ModelRollback | None = None
        # S9-W4 H-ARC-01: re-attach an existing candidate across daemon
        # restart. The prior implementation lost the candidate_id on
        # process exit — a 6-hour shadow test that had persisted a
        # candidate an hour in would restart the observation window
        # from zero on the next boot, extending the test indefinitely.
        # Observations themselves do NOT survive restart (they live
        # only in ``ShadowTest._active/_candidate`` lists and would
        # require a new schema to persist durably); re-attaching the
        # candidate_id is the minimum that keeps the A/B loop from
        # silently stalling.
        self._reattach_existing_candidate()

    def _reattach_existing_candidate(self) -> None:
        """H-ARC-01: look up any ``is_candidate=1`` row for this profile
        and restore ``self._candidate_id`` + a fresh ``ShadowTest``.

        Fail-soft — on any DB error we leave the router empty and the
        normal lazy-init path takes over as candidates land.
        """
        try:
            import sqlite3 as _sq
            conn = _sq.connect(self._learning_db, timeout=2.0)
            try:
                row = conn.execute(
                    "SELECT id FROM learning_model_state "
                    "WHERE profile_id = ? AND is_candidate = 1 "
                    "LIMIT 1",
                    (self._profile_id,),
                ).fetchone()
            finally:
                conn.close()
        except Exception:
            return  # learning.db or table missing — normal cold start.
        if row and row[0] is not None:
            cid = int(row[0])
            self._candidate_id = cid
            # Pass learning_db so ShadowTest reloads persisted paired
            # observations from ``shadow_observations`` (M012).
            self._shadow = ShadowTest(
                profile_id=self._profile_id,
                candidate_model_id=str(cid),
                learning_db=self._learning_db,
            )

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    def route_query(self, query_id: str) -> str:
        """Return the arm for ``query_id``: ``'baseline'`` or ``'candidate'``.

        Deterministic per ``(install_token, query_id)`` — survives
        daemon restart. Install-token dependence closes the
        attacker-picks-query-id bias vector (skeptic H-02).

        S9-SKEP-11: uses ``self._routing_token`` (pinned at router
        init) instead of a fresh ``ensure_install_token()`` per call,
        so mid-test token rotation does not silently flip the arm of
        every subsequent qid and mix the paired-t arms.
        """
        digest = hashlib.sha256(
            (self._routing_token + str(query_id)).encode("utf-8"),
        ).hexdigest()[:8]
        return ARM_CANDIDATE if int(digest, 16) % 2 == 1 else ARM_BASELINE

    # ------------------------------------------------------------------
    # Pre-promotion accumulator
    # ------------------------------------------------------------------

    def attach_candidate(self, candidate_id: int) -> None:
        """Called after ``_persist_candidate`` writes a fresh candidate row.
        Creates a new ``ShadowTest`` to collect paired recall results.
        """
        self._candidate_id = int(candidate_id)
        # Pass learning_db so new paired observations persist to
        # ``shadow_observations`` (M012) for restart durability.
        self._shadow = ShadowTest(
            profile_id=self._profile_id,
            candidate_model_id=str(candidate_id),
            learning_db=self._learning_db,
        )

    # ------------------------------------------------------------------
    # Post-promotion watcher
    # ------------------------------------------------------------------

    def arm_post_promotion_watch(self, *, baseline_ndcg: float) -> None:
        """Install a fresh ``ModelRollback`` observer for the 200-recall
        post-promotion watch window."""
        self._rollback = ModelRollback(
            learning_db_path=self._learning_db,
            profile_id=self._profile_id,
            baseline_ndcg=float(baseline_ndcg),
        )

    # ------------------------------------------------------------------
    # Recall-settled ingestion
    # ------------------------------------------------------------------

    def on_recall_settled(
        self,
        *,
        query_id: str,
        arm: str,
        ndcg_at_10: float,
    ) -> None:
        """Feed one settled recall into whichever phase is active.

        Precedence:
          1. If a ShadowTest is active → record paired observation; on
             ``decide() == 'promote'`` fire the atomic promotion.
          2. If a ModelRollback watch is active → record observation;
             on ``should_rollback() is True`` fire execute_rollback.

        Both phases may run for the same profile briefly if a promote
        fires mid-batch; that is intentional — the first watch window
        starts immediately on promotion.
        """
        try:
            if self._shadow is not None and self._candidate_id is not None:
                # ShadowTest expects arm='active'|'candidate'. Our router
                # uses 'baseline' as the externally-visible arm name.
                st_arm = "active" if arm == ARM_BASELINE else "candidate"
                self._shadow.record_recall_pair(
                    query_id=str(query_id), arm=st_arm,
                    ndcg_at_10=float(ndcg_at_10),
                )
                decision, _stats = self._shadow.decide()
                if decision == "promote":
                    self._fire_promotion()
                elif decision == "reject":
                    # Release the candidate reservation — a future
                    # _run_shadow_cycle may insert a new candidate.
                    self._shadow = None
                    self._candidate_id = None

            if self._rollback is not None:
                self._rollback.record_post_promotion(
                    query_id=str(query_id), ndcg_at_10=float(ndcg_at_10),
                )
                if self._rollback.should_rollback():
                    self._fire_rollback()
        except Exception as exc:  # pragma: no cover — defensive
            logger.debug("shadow_router on_recall_settled error: %s", exc)

    # ------------------------------------------------------------------
    # Internal: promotion + rollback triggers
    # ------------------------------------------------------------------

    def _fire_promotion(self) -> None:
        """Call the canonical ``_promote_candidate`` helper and reset
        the shadow state so a new A/B cycle can start next retrain."""
        if self._candidate_id is None:
            return
        try:
            from superlocalmemory.learning.consolidation_worker import (
                _promote_candidate,
            )
            _promote_candidate(
                self._learning_db,
                profile_id=self._profile_id,
                candidate_id=int(self._candidate_id),
            )
        except Exception as exc:  # pragma: no cover — defensive
            logger.warning("shadow_router promotion failed: %s", exc)
        finally:
            self._shadow = None
            self._candidate_id = None

    def _fire_rollback(self) -> None:
        """Call ``ModelRollback.execute_rollback`` and clear the watcher."""
        try:
            if self._rollback is not None:
                self._rollback.execute_rollback(
                    reason="watch_window_regression",
                )
        except Exception as exc:  # pragma: no cover — defensive
            logger.warning("shadow_router rollback failed: %s", exc)
        finally:
            self._rollback = None


# ---------------------------------------------------------------------------
# Process-local singleton cache — one ShadowRouter per (learning_db, profile).
# Thread-safe; keyed by a stable tuple so repeated factory calls reuse state.
# ---------------------------------------------------------------------------


_CACHE: dict[tuple[str, str, str], ShadowRouter] = {}
_CACHE_LOCK = threading.Lock()


def get_shadow_router(
    *,
    memory_db: str,
    learning_db: str,
    profile_id: str,
) -> ShadowRouter:
    """Factory — returns the process-local ``ShadowRouter`` for
    ``(memory_db, learning_db, profile_id)``, creating one on first call.
    """
    key = (str(memory_db), str(learning_db), str(profile_id))
    with _CACHE_LOCK:
        router = _CACHE.get(key)
        if router is None:
            router = ShadowRouter(
                memory_db=memory_db,
                learning_db=learning_db,
                profile_id=profile_id,
            )
            _CACHE[key] = router
        return router


def reset_for_testing() -> None:
    """Clear the singleton cache — tests only."""
    with _CACHE_LOCK:
        _CACHE.clear()


__all__ = (
    "ARM_BASELINE",
    "ARM_CANDIDATE",
    "ShadowRouter",
    "get_shadow_router",
    "reset_for_testing",
)
