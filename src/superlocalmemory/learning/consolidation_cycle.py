# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory v3.4.22 — F4.A Stage-8 H-01 fix

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

import logging
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from superlocalmemory.storage.write_lock import get_write_lock

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

        # 1. Confidence decay on unused facts. A dry run COUNTS the facts it
        # would touch and writes nothing (the UPDATE used to run
        # before the dry_run check, so a preview changed stored confidence).
        try:
            from superlocalmemory.learning.signals import LearningSignals
            decayed = LearningSignals.decay_confidence(
                self._memory_db, profile_id, rate=0.001, dry_run=dry_run,
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

            # Wrap _DBProxy.execute writes in the process-level write lock.
            # Reads pass through without the lock (WAL-safe).
            _ga_write_lock = get_write_lock(self._memory_db)

            class _DBProxy:
                """Minimal DB proxy for GraphAnalyzer compatibility."""

                def __init__(self, connection: sqlite3.Connection) -> None:
                    self._conn = connection

                def execute(self, sql: str, params: tuple = ()) -> list:
                    if sql.strip().upper().startswith(
                        ("INSERT", "UPDATE", "DELETE", "ALTER", "CREATE"),
                    ):
                        with _ga_write_lock:
                            cursor = self._conn.execute(sql, params)
                            self._conn.commit()
                        return []
                    return self._conn.execute(sql, params).fetchall()

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
        # Since 4.1.25 the online shadow cycle owns cold start as well: a first
        # model is a CANDIDATE tested live against the heuristic. The legacy
        # path (``_retrain_ranker``) installed a model as ACTIVE with no live
        # test whenever 200 feedback signals existed — that is how a model
        # trained on 972 all-zero labels went live. It is no longer run
        # automatically; ``_retrain_ranker`` stays callable explicitly.
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
                from superlocalmemory.learning import consolidation_worker as _shim
                if self._should_retrain(profile_id):
                    stats["online_retrain"] = _shim._run_shadow_cycle(
                        memory_db_path=self._memory_db,
                        learning_db_path=self._learning_db,
                        profile_id=profile_id,
                    )
                    stats["retrained"] = bool(
                        stats["online_retrain"].get("candidate_persisted"))
        except Exception as exc:
            logger.debug("Retrain check failed: %s", exc)

        # 6. Entity compilation (v3.4.3).
        if not dry_run:
            try:
                from superlocalmemory.core.config import SLMConfig
                from superlocalmemory.learning.entity_compiler import (
                    EntityCompiler,
                )
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

        v3.4.22 (LLD-12): prefer HNSW ANN + entity-overlap dedup with a
        reversible merge log. On any error (missing schema columns,
        hnswlib unavailable, RAM budget exceeded) fall back to the
        legacy prefix dedup so existing deployments keep working.

        Never DELETEs from atomic_facts — merges flip archive_status
        and write memory_merge_log rows.
        """
        # v3.4.22 preferred path: HNSW + memory_merge (LLD-12).
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

        # Legacy fallback (pre-v3.4.22 behaviour).
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
            # READ phase — no write lock needed (WAL allows concurrent reads).
            conn_r = sqlite3.connect(self._memory_db, timeout=10)
            conn_r.execute("PRAGMA busy_timeout=5000")
            conn_r.row_factory = sqlite3.Row

            rows = conn_r.execute(
                "SELECT fact_id, content FROM atomic_facts "
                "WHERE profile_id = ? ORDER BY created_at LIMIT ?",
                (profile_id, _LEGACY_DEDUP_SCAN_CAP),
            ).fetchall()
            conn_r.close()

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
                # WRITE phase — acquire the process-level write lock before
                # opening the sqlite3 connection, so this UPDATE is serialised
                # with all other in-process writers (DatabaseManager,
                # VectorStore, adapter sync) and cannot cause SQLITE_BUSY.
                with get_write_lock(self._memory_db):
                    conn_w = sqlite3.connect(self._memory_db, timeout=10)
                    try:
                        for fid in duplicates:
                            conn_w.execute(
                                "UPDATE atomic_facts "
                                "SET confidence = MAX(0.1, confidence * 0.5) "
                                "WHERE fact_id = ?",
                                (fid,),
                            )
                        conn_w.commit()
                    finally:
                        conn_w.close()

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
            from superlocalmemory.learning import consolidation_worker as _shim
            return _shim._retrain_ranker_impl(self._learning_db, profile_id)
        except Exception as exc:
            logger.debug("Retrain failed: %s", exc)
            return False

    # ------------------------------------------------------------------
    # LLD-10 — online retrain trigger
    # ------------------------------------------------------------------

    def _should_retrain(self, profile_id: str) -> bool:
        """Whether the online retrain is due — see ``learning.retrain_trigger``.

        Covers cold start too (no active model -> a candidate tested live
        against the heuristic), refuses while a candidate is under test, and
        counts new labelled outcomes from the data instead of a counter
        nothing incremented.
        """
        from superlocalmemory.learning.ranker_retrain_online import (
            RETRAIN_HOURS_THRESHOLD,
            RETRAIN_NEW_OUTCOMES_THRESHOLD,
        )
        from superlocalmemory.learning.retrain_trigger import should_retrain

        return should_retrain(
            self._learning_db, self._memory_db, profile_id,
            new_outcomes_threshold=RETRAIN_NEW_OUTCOMES_THRESHOLD,
            hours_threshold=RETRAIN_HOURS_THRESHOLD,
        )
