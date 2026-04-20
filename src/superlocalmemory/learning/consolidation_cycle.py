# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory v3.4.21 — F4.A Stage-8 H-01 fix

"""ConsolidationWorker — background memory maintenance lifecycle.

Runs periodically (every 6 hours or on-demand) to:
  1. Decay confidence on unused facts (floor 0.1).
  2. Deduplicate near-identical facts via HNSW (LLD-12).
  3. Mine behavioural patterns (``pattern_miner.generate_patterns``).
  4. Recompute graph intelligence.
  5. Auto-retrain the adaptive ranker — online (LLD-10) or legacy
     cold-start gated by Stage-8 H-07.
  6. Compile entity truth blocks (v3.4.3).

The class itself is kept lean: every heavy helper lives in a dedicated
module (``pattern_miner``, ``ranker_retrain_online``,
``ranker_retrain_legacy``).

Contract refs:
  - LLD-10 §2 + §3 — online retrain orchestration.
  - LLD-12 §2 — HNSW dedup path.
  - Stage 8 H-01 + H-07 — file split + legacy gating.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

__all__ = ("ConsolidationWorker",)


class ConsolidationWorker:
    """Background memory maintenance worker.

    Call :py:meth:`run` periodically or via the dashboard button. All
    operations are safe — they improve quality without losing data.
    """

    def __init__(
        self, memory_db: str | Path, learning_db: str | Path,
    ) -> None:
        self._memory_db = str(memory_db)
        self._learning_db = str(learning_db)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, profile_id: str, dry_run: bool = False) -> dict:
        """Run full consolidation cycle. Returns stats."""
        stats = {
            "decayed": 0,
            "deduped": 0,
            "retrained": False,
            "signal_count": 0,
            "ranker_phase": 1,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        # 1. Confidence decay on unused facts.
        try:
            from superlocalmemory.learning.signals import LearningSignals
            decayed = LearningSignals.decay_confidence(
                self._memory_db, profile_id, rate=0.001,
            )
            stats["decayed"] = decayed
            if not dry_run:
                logger.info(
                    "Confidence decay: %d facts affected", decayed,
                )
        except Exception as exc:
            logger.debug("Decay failed: %s", exc)

        # 2. Deduplication (HNSW + fallback).
        try:
            deduped = self._deduplicate(profile_id, dry_run)
            stats["deduped"] = deduped
        except Exception as exc:
            logger.debug("Dedup failed: %s", exc)

        # 3. Behavioural patterns.
        try:
            patterns = self._generate_patterns(profile_id, dry_run)
            stats["patterns_generated"] = patterns
        except Exception as exc:
            logger.debug("Pattern generation failed: %s", exc)

        # 4. Recompute graph intelligence (v3.4.2).
        try:
            from superlocalmemory.core.graph_analyzer import GraphAnalyzer
            conn_ga = sqlite3.connect(self._memory_db, timeout=10)
            conn_ga.execute("PRAGMA busy_timeout=5000")
            conn_ga.row_factory = sqlite3.Row

            class _DBProxy:
                """Minimal DB proxy for GraphAnalyzer compatibility."""

                def __init__(self, connection: sqlite3.Connection) -> None:
                    self._conn = connection

                def execute(self, sql: str, params: tuple = ()) -> list:
                    cursor = self._conn.execute(sql, params)
                    if sql.strip().upper().startswith(
                        ("INSERT", "UPDATE", "DELETE", "ALTER", "CREATE"),
                    ):
                        self._conn.commit()
                        return []
                    return cursor.fetchall()

            ga = GraphAnalyzer(_DBProxy(conn_ga))
            if not dry_run:
                ga_result = ga.compute_and_store(profile_id)
                stats["graph_nodes"] = ga_result.get("node_count", 0)
                stats["graph_communities"] = ga_result.get(
                    "community_count", 0,
                )
                logger.info(
                    "Graph analysis: %d nodes, %d communities",
                    stats["graph_nodes"], stats["graph_communities"],
                )
            conn_ga.close()
        except Exception as exc:
            logger.debug("Graph analysis failed: %s", exc)

        # 5. Ranker retrain — online (LLD-10) or legacy cold-start.
        #
        # Gating (Stage-8 H-07): once a profile has an active model the
        # online path wins unconditionally. The legacy cold-start path
        # only fires when there is NO active model (``_should_retrain``
        # returns False because no active row exists) AND the raw
        # signal_count crosses 200. Partial unique indexes M009 keep
        # both paths from racing.
        try:
            from superlocalmemory.learning.feedback import FeedbackCollector
            collector = FeedbackCollector(Path(self._learning_db))
            signal_count = collector.get_feedback_count(profile_id)
            stats["signal_count"] = signal_count
            stats["ranker_phase"] = (
                1 if signal_count < 50 else (2 if signal_count < 200 else 3)
            )

            if not dry_run:
                # Late import — the shim hosts ``_run_shadow_cycle`` so
                # that monkey-patches on ``consolidation_worker`` reach
                # into the orchestrator's helper lookups.
                from superlocalmemory.learning import consolidation_worker \
                    as _shim
                if self._should_retrain(profile_id):
                    stats["online_retrain"] = _shim._run_shadow_cycle(
                        memory_db_path=self._memory_db,
                        learning_db_path=self._learning_db,
                        profile_id=profile_id,
                    )
                elif signal_count >= 200:
                    # Cold-start only: no active model yet.
                    retrained = self._retrain_ranker(
                        profile_id, signal_count,
                    )
                    stats["retrained"] = retrained
        except Exception as exc:
            logger.debug("Retrain check failed: %s", exc)

        # 6. Entity compilation (v3.4.3).
        if not dry_run:
            try:
                from superlocalmemory.learning.entity_compiler import (
                    EntityCompiler,
                )
                from superlocalmemory.core.config import SLMConfig
                config = SLMConfig.load()
                compiler = EntityCompiler(self._memory_db, config)
                ec_result = compiler.compile_all(profile_id)
                stats["entities_compiled"] = ec_result.get("compiled", 0)
                if ec_result["compiled"] > 0:
                    logger.info(
                        "Entity compilation: %d entities compiled",
                        ec_result["compiled"],
                    )
            except Exception as exc:
                logger.debug("Entity compilation failed: %s", exc)

        return stats

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _deduplicate(self, profile_id: str, dry_run: bool) -> int:
        """Find and mark near-duplicate facts.

        v3.4.21 (LLD-12): prefer HNSW ANN + entity-overlap dedup with a
        reversible merge log. On any error (missing schema columns,
        hnswlib unavailable, RAM budget exceeded) fall back to the
        legacy prefix dedup so existing deployments keep working.

        Never DELETEs from atomic_facts — merges flip archive_status
        and write memory_merge_log rows.
        """
        # v3.4.21 preferred path: HNSW + memory_merge (LLD-12).
        try:
            from superlocalmemory.learning.hnsw_dedup import (
                HnswDeduplicator,
            )
            from superlocalmemory.learning.memory_merge import apply_merges

            dedup = HnswDeduplicator(memory_db_path=self._memory_db)
            candidates = dedup.find_merge_candidates(profile_id)
            if not candidates:
                return 0
            if dry_run:
                return len(candidates)
            applied = apply_merges(
                self._memory_db, candidates, profile_id=profile_id,
            )
            return applied
        except sqlite3.OperationalError as exc:
            # Schema predates M011 — fall through to legacy path.
            logger.debug("hnsw dedup schema missing, fallback: %s", exc)
        except Exception as exc:
            logger.debug("hnsw dedup unexpected error, fallback: %s", exc)

        # Legacy fallback (pre-v3.4.21 behaviour).
        # S9-defer H-P-09: hard cap on the fallback scan so a profile
        # with 5M+ atomic_facts cannot OOM the consolidation worker
        # when hnswlib is unavailable. The fallback is a prefix-match
        # heuristic anyway; scanning more than 100k rows via this
        # code path is noise, not signal. The cap can be raised via
        # env var for benchmarking.
        _LEGACY_DEDUP_SCAN_CAP = int(
            os.environ.get("SLM_LEGACY_DEDUP_SCAN_CAP", "100000")
        )
        try:
            conn = sqlite3.connect(self._memory_db, timeout=10)
            conn.execute("PRAGMA busy_timeout=5000")
            conn.row_factory = sqlite3.Row

            rows = conn.execute(
                "SELECT fact_id, content FROM atomic_facts "
                "WHERE profile_id = ? ORDER BY created_at LIMIT ?",
                (profile_id, _LEGACY_DEDUP_SCAN_CAP),
            ).fetchall()

            seen_prefixes: dict[str, str] = {}
            duplicates = []

            for r in rows:
                d = dict(r)
                prefix = d["content"][:100].strip().lower()
                if prefix in seen_prefixes:
                    duplicates.append(d["fact_id"])
                else:
                    seen_prefixes[prefix] = d["fact_id"]

            if duplicates and not dry_run:
                for fid in duplicates:
                    conn.execute(
                        "UPDATE atomic_facts "
                        "SET confidence = MAX(0.1, confidence * 0.5) "
                        "WHERE fact_id = ?",
                        (fid,),
                    )
                conn.commit()

            conn.close()
            return len(duplicates)
        except Exception:
            return 0

    def _generate_patterns(
        self, profile_id: str, dry_run: bool = False,
    ) -> int:
        """Back-compat shim delegating to ``pattern_miner.generate_patterns``.

        Preserved so the MCP ``run_maintenance`` tool and any external
        caller that bound this method directly keeps working.
        """
        from superlocalmemory.learning.pattern_miner import generate_patterns
        return generate_patterns(
            self._memory_db, self._learning_db, profile_id, dry_run,
        )

    def _retrain_ranker(self, profile_id: str, signal_count: int) -> bool:
        """Legacy cold-start retrain. DEPRECATED.

        Delegates to the deprecated legacy impl; a one-shot
        ``DeprecationWarning`` fires on first invocation per process.
        """
        try:
            from superlocalmemory.learning import consolidation_worker \
                as _shim
            return _shim._retrain_ranker_impl(self._learning_db, profile_id)
        except Exception as exc:
            logger.debug("Retrain failed: %s", exc)
            return False

    # ------------------------------------------------------------------
    # LLD-10 — online retrain trigger
    # ------------------------------------------------------------------

    def _should_retrain(self, profile_id: str) -> bool:
        """Return True if the outcome-count or 24h trigger has fired.

        Reads ``learning_model_state.metadata_json`` on the active row.
        Honours ``metadata_json.retrain_disabled_until`` (post-rollback
        cooldown). No DB writes — pure SELECT + JSON parse.
        """
        from superlocalmemory.learning.ranker_retrain_online import (
            RETRAIN_NEW_OUTCOMES_THRESHOLD,
            RETRAIN_HOURS_THRESHOLD,
        )
        try:
            conn = sqlite3.connect(self._learning_db, timeout=5)
            try:
                conn.row_factory = sqlite3.Row
                row = conn.execute(
                    "SELECT metadata_json FROM learning_model_state "
                    "WHERE profile_id = ? AND is_active = 1 LIMIT 1",
                    (profile_id,),
                ).fetchone()
            finally:
                conn.close()
        except sqlite3.Error as exc:
            logger.debug("_should_retrain sqlite error: %s", exc)
            return False

        if row is None:
            # No active model yet — let the legacy cold-start path
            # (signal_count >= 200) drive first training.
            return False

        try:
            meta = json.loads(row["metadata_json"] or "{}")
        except (TypeError, ValueError):
            meta = {}

        now = datetime.now(timezone.utc)

        # Cooldown: honour retrain_disabled_until (post-rollback).
        disabled_until = meta.get("retrain_disabled_until")
        if disabled_until:
            try:
                dt = datetime.fromisoformat(disabled_until)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                if dt > now:
                    return False
            except (TypeError, ValueError):
                pass  # malformed → ignore the cooldown

        # Trigger A — outcome-count delta.
        try:
            new_outcomes = int(
                meta.get("new_outcomes_since_last_retrain", 0) or 0,
            )
        except (TypeError, ValueError):
            new_outcomes = 0
        if new_outcomes >= RETRAIN_NEW_OUTCOMES_THRESHOLD:
            return True

        # Trigger B — hours since last retrain.
        last = meta.get("last_retrain_at")
        if last:
            try:
                last_dt = datetime.fromisoformat(last)
                if last_dt.tzinfo is None:
                    last_dt = last_dt.replace(tzinfo=timezone.utc)
                hours = (now - last_dt).total_seconds() / 3600.0
                if hours >= RETRAIN_HOURS_THRESHOLD:
                    return True
            except (TypeError, ValueError):
                pass

        return False
