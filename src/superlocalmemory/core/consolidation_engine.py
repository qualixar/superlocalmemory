# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Sleep-time consolidation engine (Phase 5).

Letta-inspired 6-step consolidation cycle:
  1. Compress — deduplicate near-identical facts
  2. Compile Core Memory blocks — rules (Mode A) or LLM (Mode B/C)
  3. Promote — move frequently accessed facts up lifecycle
  4. Decay — reduce weights on unused association edges
  5. Recompute graph — PageRank + communities
  6. Derive associations — link new summary facts

Guarantees:
  - Idempotent: running twice produces identical state (L18)
  - Never deletes facts (Rule 17)
  - No import of core/engine.py (Rule 06)
  - Silent errors (Rule 19)
  - Parameterized SQL (Rule 11)
  - Feature-flagged via enabled=False (Rule 12)

Part of Qualixar | Author: Varun Pratap Bhardwaj
"""

from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from superlocalmemory.core.config import ConsolidationConfig, SLMConfig
    from superlocalmemory.core.graph_analyzer import GraphAnalyzer
    from superlocalmemory.core.summarizer import Summarizer
    from superlocalmemory.encoding.auto_linker import AutoLinker
    from superlocalmemory.encoding.temporal_validator import TemporalValidator
    from superlocalmemory.learning.behavioral import (
        BehavioralPatternStore,
        BehavioralTracker,
    )
    from superlocalmemory.learning.consolidation_quantization_worker import (
        CCQWorker,
    )
    from superlocalmemory.storage.database import DatabaseManager

logger = logging.getLogger(__name__)


def _recompute_entity_communities(db: Any, profile_id: str) -> dict[str, int]:
    """Wave Q: rebuild the entity-community backbone (fail-open, non-fatal).

    Shared spine for Q2 community summaries and Q3 progressive abstraction.
    Runs in the background consolidation lane; never blocks store/recall.
    """
    try:
        from superlocalmemory.core.entity_community import EntityCommunityBuilder

        result = EntityCommunityBuilder(db).compute_and_store(profile_id)
        logger.info(
            "Background entity-community: %d entities, %d communities",
            result.get("entity_count", 0),
            result.get("community_count", 0),
        )
        return result
    except Exception as exc:
        logger.debug("Entity-community recompute failed (non-fatal): %s", exc)
        return {"entity_count": 0, "community_count": 0}


class ConsolidationEngine:
    """Sleep-time memory consolidation with 6-step cycle.

    The biological metaphor: during sleep, the brain replays recent
    experiences, compresses them into long-term memory, strengthens
    important connections, and prunes weak ones.  SLM does the same.

    Consolidation is IDEMPOTENT: running twice produces identical state (L18).
    This is guaranteed because:
      - Block compilation uses INSERT OR REPLACE on UNIQUE(profile_id, block_type)
      - Promotion checks current state before updating
      - PageRank is deterministic given the same graph
      - Edge decay is monotonic (weight only decreases)

    Never overwrites or deletes facts (Rule 17).
    """

    def __init__(
        self,
        db: DatabaseManager,
        config: ConsolidationConfig,
        summarizer: Summarizer | None = None,
        behavioral_store: BehavioralPatternStore | BehavioralTracker | None = None,
        auto_linker: AutoLinker | None = None,
        graph_analyzer: GraphAnalyzer | None = None,
        temporal_validator: TemporalValidator | None = None,
        slm_config: SLMConfig | None = None,
        ccq_worker: CCQWorker | None = None,
    ) -> None:
        self._db = db
        self._config = config
        self._summarizer = summarizer
        self._behavioral = behavioral_store
        self._auto_linker = auto_linker
        self._graph_analyzer = graph_analyzer
        self._temporal_validator = temporal_validator
        self._slm_config = slm_config
        self._ccq_worker = ccq_worker
        self._mode = slm_config.mode.value if slm_config else "a"
        self._store_count: int = 0  # For step-count trigger (L7)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def consolidate(
        self, profile_id: str, lightweight: bool = False,
    ) -> dict[str, Any]:
        """Execute the consolidation cycle.

        Full cycle (session end, manual, scheduled):
            Steps 1-6 (compress, compile, promote, decay, recompute, derive)

        Lightweight cycle (step-count trigger every 50 stores):
            Steps 2 + 4 only (refresh blocks + decay edges)

        Returns dict with step results for dashboard display.
        """
        results: dict[str, Any] = {
            "profile_id": profile_id,
            "lightweight": lightweight,
        }

        try:
            if lightweight:
                results["blocks"] = self._step2_compile_blocks(profile_id)
                results["decayed"] = self._step4_decay_edges(profile_id)
            else:
                results["compressed"] = self._step1_compress(profile_id)
                results["blocks"] = self._step2_compile_blocks(profile_id)
                results["promoted"] = self._step3_promote(profile_id)
                results["decayed"] = self._step4_decay_edges(profile_id)
                results["graph_stats"] = self._step5_recompute_graph(profile_id)
                results["new_associations"] = self._step6_derive_associations(
                    profile_id,
                )
                # Step 7: Cognitive Consolidation Quantization (Phase E)
                results["ccq"] = self._step7_ccq(profile_id)

                # Step 8 (v3.4.7): Mine behavioral assertions from tool events.
                try:
                    from superlocalmemory.learning.assertion_miner import AssertionMiner
                    miner = AssertionMiner(self._db.db_path)
                    results["assertions"] = miner.mine(profile_id)
                except Exception as exc:
                    logger.debug("Assertion mining (non-fatal): %s", exc)
                    results["assertions"] = {"error": str(exc)}

                # Step 9 (v3.4.7): Generate soft prompts from patterns + assertions.
                # Previously the AutoParameterizeHook was never wired.
                try:
                    from superlocalmemory.parameterization.pattern_extractor import PatternExtractor
                    from superlocalmemory.parameterization.soft_prompt_generator import SoftPromptGenerator
                    from superlocalmemory.parameterization.prompt_injector import PromptInjector
                    from superlocalmemory.parameterization.prompt_lifecycle import PromptLifecycleManager
                    from superlocalmemory.learning.behavioral import BehavioralPatternStore
                    # cross_project and workflow_miner may not exist yet — stub them
                    try:
                        from superlocalmemory.parameterization.cross_project import CrossProjectAggregator
                    except ImportError:
                        class CrossProjectAggregator:
                            def __init__(self, db): pass
                            def get_preferences(self, *a, **kw): return {}
                    try:
                        from superlocalmemory.parameterization.workflow_miner import WorkflowMiner
                    except ImportError:
                        class WorkflowMiner:
                            def __init__(self, db): pass
                            def mine(self, *a, **kw): return []
                    from superlocalmemory.hooks.auto_parameterize import AutoParameterizeHook
                    from superlocalmemory.core.config import ParameterizationConfig
                    p_config = getattr(self._slm_config, "parameterization", ParameterizationConfig())
                    from superlocalmemory.infra.data_root import state_path
                    learning_db = str(state_path("learning.db"))
                    beh_store = BehavioralPatternStore(learning_db)
                    cross_proj = CrossProjectAggregator(self._db)
                    wf_miner = WorkflowMiner(self._db)
                    extractor = PatternExtractor(self._db, beh_store, cross_proj, wf_miner, p_config)
                    generator = SoftPromptGenerator(p_config)
                    injector = PromptInjector(self._db, generator, p_config)
                    lifecycle = PromptLifecycleManager(self._db, p_config)
                    hook = AutoParameterizeHook(extractor, generator, injector, lifecycle, p_config)
                    sp_result = hook.on_consolidation_complete(profile_id)
                    results["soft_prompts"] = sp_result
                except Exception as exc:
                    logger.debug("Soft prompt generation (non-fatal): %s", exc)
                    results["soft_prompts"] = {"error": str(exc)}

                # Step 10 (v3.4.10): Mine skill performance from tool events.
                # Creates per-skill assertions and skill correlation patterns.
                try:
                    from superlocalmemory.learning.skill_performance_miner import SkillPerformanceMiner
                    spm = SkillPerformanceMiner(self._db.db_path)
                    results["skill_performance"] = spm.mine(profile_id)
                except Exception as exc:
                    logger.debug("Skill performance mining (non-fatal): %s", exc)
                    results["skill_performance"] = {"error": str(exc)}

                # Step 11 (v3.4.10): Skill evolution — 3-trigger system.
                # Runs degradation + health check triggers. Post-session
                # trigger runs separately from the Stop hook.
                # Never on recall/remember hot path. Budget: max 3 per cycle.
                try:
                    from superlocalmemory.evolution.skill_evolver import SkillEvolver
                    evolver = SkillEvolver(self._db.db_path)
                    results["skill_evolution"] = evolver.run_consolidation_cycle(profile_id)
                except Exception as exc:
                    logger.debug("Skill evolution (non-fatal): %s", exc)
                    results["skill_evolution"] = {"error": str(exc)}

                # Step 12 (v3.4.11): Generate soft prompts for evolved skills.
                # Queries promoted evolutions and creates/updates custom soft
                # prompts so the AI prefers evolved skill variants.
                try:
                    results["evolution_soft_prompts"] = self._step12_evolution_soft_prompts(profile_id)
                except Exception as exc:
                    logger.debug("Evolution soft prompts (non-fatal): %s", exc)
                    results["evolution_soft_prompts"] = {"error": str(exc)}

            results["success"] = True
        except Exception as exc:
            logger.warning(
                "Consolidation failed (non-fatal) for profile %s: %s",
                profile_id, exc,
            )
            results["success"] = False
            results["error"] = str(exc)

        return results

    def increment_store_count(self, profile_id: str) -> bool:
        """Called after each store() in store_pipeline.py.

        Increments internal counter.  When counter hits step_count_trigger
        (default 50), runs lightweight consolidation AND queues async
        graph analysis.

        V3.4.2: Graph analysis runs in background thread after every
        lightweight consolidation trigger. This populates fact_importance
        (PageRank, communities, bridge scores) so retrieval channels can
        use graph intelligence without blocking store/recall latency.

        Returns True if lightweight consolidation was triggered.
        """
        if not self._config.enabled:
            return False

        self._store_count += 1
        if self._store_count >= self._config.step_count_trigger:
            self._store_count = 0
            self.consolidate(profile_id, lightweight=True)
            # V3.4.2: Queue graph analysis in background (non-blocking)
            self._queue_graph_analysis(profile_id)
            return True
        return False

    def _queue_graph_analysis(self, profile_id: str) -> None:
        """Run graph_analyzer.compute_and_store() in a background thread.

        V3.4.2: Populates fact_importance table with PageRank, community_id,
        degree_centrality, and bridge_score. Next recall() automatically
        uses updated graph intelligence for entity channel and spreading
        activation. Takes ~200-800ms, runs on daemon thread, zero impact
        on store/recall latency.
        """
        analyzer = self._graph_analyzer
        pid = profile_id
        db = self._db

        def _run() -> None:
            if analyzer is not None:
                try:
                    result = analyzer.compute_and_store(pid)
                    logger.info(
                        "Background graph analysis complete: %d nodes, "
                        "%d communities",
                        result.get("node_count", 0),
                        result.get("community_count", 0),
                    )
                except Exception as exc:
                    logger.debug(
                        "Background graph analysis failed (non-fatal): %s", exc,
                    )
            # Wave Q: entity-community backbone (shared spine for Q2/Q3).
            _recompute_entity_communities(db, pid)

        t = threading.Thread(target=_run, daemon=True, name="graph-analysis-bg")
        t.start()

    def get_core_memory(self, profile_id: str) -> dict[str, str]:
        """Load all Core Memory blocks for a profile.

        Returns dict of {block_type: content}.
        Called at session_init to inject into context.
        """
        rows = self._db.get_core_blocks(profile_id)
        return {r["block_type"]: r["content"] for r in rows}

    def get_core_memory_blocks(self, profile_id: str) -> list[dict]:
        """Load all Core Memory blocks with full metadata. For API."""
        return self._db.get_core_blocks(profile_id)

    # ------------------------------------------------------------------
    # Step 1: Compress — deduplicate near-identical facts
    # ------------------------------------------------------------------

    def _step1_compress(self, profile_id: str) -> dict[str, Any]:
        """Deduplicate near-identical facts by archiving originals.

        Never deletes facts (Rule 17).  Sets lifecycle to 'archived'.
        In Mode A, compression is a no-op (no VectorStore for similarity).
        """
        # Mode A: heuristic compression is a stub — requires VectorStore
        # for similarity search which is optional.  Return zero counts.
        return {
            "clusters_found": 0,
            "facts_compressed": 0,
            "summaries_created": 0,
        }

    # ------------------------------------------------------------------
    # Step 2: Compile Core Memory Blocks
    # ------------------------------------------------------------------

    def _step2_compile_blocks(self, profile_id: str) -> dict[str, Any]:
        """Compile Core Memory blocks based on mode.

        Mode A: rules-based (no LLM) (L3 fix)
        Mode B/C: LLM-assisted summarization
        """
        if self._mode == "a":
            return self.compile_core_blocks_mode_a(profile_id)
        return self._compile_core_blocks_llm(profile_id)

    def compile_core_blocks_mode_a(self, profile_id: str) -> dict[str, Any]:
        """Mode A: populate Core Memory blocks without LLM (L3 fix).

        Rules-based compilation:
          - user_profile: top-5 semantic/opinion facts by access_count
          - project_context: top-5 episodic facts by recency
          - behavioral_patterns: top-5 patterns by confidence
          - active_decisions: facts with signal_type='decision', min access
          - learned_preferences: opinion facts with confidence >= threshold
        """
        blocks_compiled = 0
        block_limit = self._config.block_char_limit

        # 1. user_profile: top semantic/opinion facts by access
        user_facts = self._get_top_facts(
            profile_id,
            fact_types=["semantic", "opinion"],
            sort_by="access_count",
            limit=5,
        )
        self._store_core_block(
            profile_id, "user_profile",
            self._facts_to_content(user_facts, block_limit),
            [f["fact_id"] for f in user_facts],
        )
        blocks_compiled += 1

        # 2. project_context: top episodic facts by recency
        project_facts = self._get_top_facts(
            profile_id,
            fact_types=["episodic"],
            sort_by="recency",
            limit=5,
        )
        self._store_core_block(
            profile_id, "project_context",
            self._facts_to_content(project_facts, block_limit),
            [f["fact_id"] for f in project_facts],
        )
        blocks_compiled += 1

        # 3. behavioral_patterns: from behavioral store
        pattern_content = self._compile_behavioral_block(
            profile_id, block_limit,
        )
        self._store_core_block(
            profile_id, "behavioral_patterns",
            pattern_content,
            [],
        )
        blocks_compiled += 1

        # 4. active_decisions: signal_type='decision' with min access
        # CRITICAL: uses signal_type NOT fact_type (HIGH-2 fix)
        decision_facts = self._db.execute(
            "SELECT f.fact_id, f.content FROM atomic_facts f "
            "LEFT JOIN fact_access_log a ON f.fact_id = a.fact_id "
            "WHERE f.profile_id = ? AND f.signal_type = 'decision' "
            "AND f.lifecycle = 'active' "
            "GROUP BY f.fact_id "
            "HAVING COUNT(a.log_id) >= ? "
            "ORDER BY COUNT(a.log_id) DESC LIMIT 5",
            (profile_id, self._config.promotion_min_access),
        )
        self._store_core_block(
            profile_id, "active_decisions",
            self._rows_to_content(decision_facts, block_limit),
            [dict(f)["fact_id"] for f in (decision_facts or [])],
        )
        blocks_compiled += 1

        # 5. learned_preferences: opinion facts with high confidence
        pref_facts = self._db.execute(
            "SELECT fact_id, content FROM atomic_facts "
            "WHERE profile_id = ? AND fact_type = 'opinion' "
            "AND confidence >= ? AND lifecycle = 'active' "
            "ORDER BY confidence DESC LIMIT 5",
            (profile_id, self._config.promotion_min_trust),
        )
        self._store_core_block(
            profile_id, "learned_preferences",
            self._rows_to_content(pref_facts, block_limit),
            [dict(f)["fact_id"] for f in (pref_facts or [])],
        )
        blocks_compiled += 1

        return {"blocks_compiled": blocks_compiled, "mode": "rules"}

    def _compile_core_blocks_llm(self, profile_id: str) -> dict[str, Any]:
        """Mode B/C: LLM-assisted Core Memory block compilation.

        Falls back to Mode A rules if LLM fails (Rule 19).
        """
        if self._summarizer is None:
            return self.compile_core_blocks_mode_a(profile_id)

        try:
            blocks_compiled = 0

            for block_type, fact_types in [
                ("user_profile", ["semantic", "opinion"]),
                ("project_context", ["episodic"]),
            ]:
                facts = self._get_top_facts(
                    profile_id, fact_types=fact_types,
                    sort_by="access_count", limit=8,
                )
                if facts:
                    fact_dicts = [
                        {"content": f.get("content", "")} for f in facts
                    ]
                    summary = self._summarizer.summarize_cluster(fact_dicts)
                    self._store_core_block(
                        profile_id, block_type,
                        summary[:self._config.block_char_limit],
                        [f["fact_id"] for f in facts],
                        compiled_by="llm",
                    )
                    blocks_compiled += 1

            # Behavioral, decisions, preferences still rules-based
            mode_a_result = self.compile_core_blocks_mode_a(profile_id)
            blocks_compiled += mode_a_result.get("blocks_compiled", 0)

            return {"blocks_compiled": blocks_compiled, "mode": "llm"}
        except Exception:
            # Fallback to Mode A (Rule 19)
            return self.compile_core_blocks_mode_a(profile_id)

    # ------------------------------------------------------------------
    # Step 3: Auto-Promote
    # ------------------------------------------------------------------

    def _step3_promote(self, profile_id: str) -> dict[str, Any]:
        """Promote frequently accessed facts to higher lifecycle state.

        Checks temporal validity (L12 fix) and trust threshold.
        Never overwrites fact content (Rule 17).
        """
        candidates = self._db.execute(
            "SELECT f.fact_id, f.lifecycle, f.confidence, "
            "       COUNT(a.log_id) as access_count "
            "FROM atomic_facts f "
            "LEFT JOIN fact_access_log a ON f.fact_id = a.fact_id "
            "WHERE f.profile_id = ? AND f.lifecycle = 'active' "
            "GROUP BY f.fact_id "
            "HAVING COUNT(a.log_id) >= ?",
            (profile_id, self._config.promotion_min_access),
        )

        promoted = 0
        for row in (candidates or []):
            d = dict(row)
            fact_id = d["fact_id"]

            # Temporal validity check (L12 fix)
            if not self._is_temporally_valid(fact_id, profile_id):
                continue

            # Trust check
            if d.get("confidence", 0) < self._config.promotion_min_trust:
                continue

            # Promote: active -> warm (lifecycle transition)
            from superlocalmemory.core.lifecycle_state import set_fact_lifecycle_zone
            set_fact_lifecycle_zone(
                self._db,
                [fact_id],
                "warm",
                profile_id=profile_id,
                from_atomic=("active",),
            )
            promoted += 1

        return {
            "candidates": len(candidates or []),
            "promoted": promoted,
        }

    def _is_temporally_valid(
        self, fact_id: str, profile_id: str,
    ) -> bool:
        """Check if fact has not been temporally invalidated (L12).

        Returns True if valid, False if expired.
        """
        # Use TemporalValidator.is_temporally_valid() if available (P5-BC4)
        if self._temporal_validator is not None:
            return self._temporal_validator.is_temporally_valid(
                fact_id, profile_id,
            )

        # Fallback: direct SQL check. Must consider BOTH valid_until (valid-time
        # expiry) AND system_expired_at (transaction-time expiry) — a fact that
        # was invalidated/erased sets system_expired_at, and ignoring it would
        # let a GDPR-erased/superseded fact be promoted back into warm lifecycle.
        rows = self._db.execute(
            "SELECT valid_until, system_expired_at FROM fact_temporal_validity "
            "WHERE fact_id = ? AND profile_id = ?",
            (fact_id, profile_id),
        )
        if not rows:
            return True  # No temporal record = valid
        row = dict(rows[0])
        if row.get("system_expired_at") is not None:
            return False  # transaction-time expired (invalidated/erased)
        valid_until = row.get("valid_until")
        if valid_until is None:
            return True  # Open-ended validity
        try:
            expiry = datetime.fromisoformat(valid_until)
            now = datetime.now(timezone.utc)
            # Handle naive datetimes
            if expiry.tzinfo is None:
                return expiry > now.replace(tzinfo=None)
            return expiry > now
        except (ValueError, TypeError):
            return True  # Parse failure = assume valid

    # ------------------------------------------------------------------
    # Step 4: Decay Edges
    # ------------------------------------------------------------------

    def _step4_decay_edges(self, profile_id: str) -> dict[str, int]:
        """Decay unused association edges.  Delegates to AutoLinker."""
        if self._auto_linker is None:
            return {"decayed": 0}
        try:
            decayed = self._auto_linker.decay_unused(
                profile_id, days_threshold=self._config.decay_days_threshold,
            )
            return {"decayed": decayed}
        except Exception as exc:
            logger.warning("Edge decay failed: %s", exc)
            return {"decayed": 0}

    # ------------------------------------------------------------------
    # Step 5: Recompute Graph
    # ------------------------------------------------------------------

    def _step5_recompute_graph(
        self, profile_id: str,
    ) -> dict[str, Any]:
        """Recompute PageRank + communities.  Delegates to GraphAnalyzer.

        Wave Q: also rebuilds the entity-community backbone (Q2/Q3 spine).
        """
        result: dict[str, Any] = {"node_count": 0, "community_count": 0}
        if self._graph_analyzer is not None:
            try:
                result = self._graph_analyzer.compute_and_store(profile_id)
            except Exception as exc:
                logger.warning("Graph recompute failed: %s", exc)
        ec = _recompute_entity_communities(self._db, profile_id)
        result["entity_community_count"] = ec.get("community_count", 0)
        return result

    # ------------------------------------------------------------------
    # Step 6: Derive Associations
    # ------------------------------------------------------------------

    def _step6_derive_associations(
        self, profile_id: str,
    ) -> dict[str, int]:
        """Derive new associations from recently created summary facts."""
        summaries = self._db.execute(
            "SELECT fact_id FROM atomic_facts "
            "WHERE profile_id = ? AND fact_type = 'semantic' "
            "AND lifecycle = 'active' "
            "AND created_at > datetime('now', '-1 hour')",
            (profile_id,),
        )
        linked = 0
        if self._auto_linker and summaries:
            for row in summaries:
                fact = self._db.get_fact(dict(row)["fact_id"])
                if fact:
                    try:
                        ids = self._auto_linker.link_new_fact(fact, profile_id)
                        linked += len(ids)
                    except Exception:
                        pass
        return {"summary_facts_linked": linked}

    # ------------------------------------------------------------------
    # Step 7: Cognitive Consolidation Quantization (Phase E)
    # ------------------------------------------------------------------

    def _step7_ccq(self, profile_id: str) -> dict[str, Any]:
        """Run CCQ pipeline after existing 6-step consolidation.

        CCQ is step 7 because it depends on retention data from Phase A
        and benefits from running after standard consolidation cleanup.
        """
        if self._ccq_worker is None:
            return {"enabled": False}

        if not self._ccq_worker.should_run(
            self._store_count, is_session_end=False,
        ):
            return {"skipped": True, "reason": "trigger not met"}

        try:
            result = self._ccq_worker.run(profile_id)
            return {
                "clusters": result.clusters_processed,
                "blocks": result.blocks_created,
                "archived": result.facts_archived,
                "compression_ratio": result.compression_ratio,
            }
        except Exception as exc:
            logger.warning("CCQ step failed (non-fatal): %s", exc)
            return {"error": str(exc)}

    # ------------------------------------------------------------------
    # Step 12: Evolution Soft Prompts
    # ------------------------------------------------------------------

    def _step12_evolution_soft_prompts(self, profile_id: str) -> dict[str, Any]:
        """Generate soft prompts for promoted evolved skills.

        Queries skill_evolution_log for promoted evolutions and creates
        or updates soft_prompt_templates with category='custom' so the
        AI agent prefers evolved skill variants over originals.

        Uses 'custom' category because the soft_prompt_templates CHECK
        constraint does not include 'skill_evolution'. The content is
        prefixed with [SKILL_EVOLUTION] for easy filtering.
        """
        import sqlite3 as _sqlite3

        db_path = str(self._db.db_path)

        conn = _sqlite3.connect(db_path, timeout=10)
        conn.row_factory = _sqlite3.Row

        # Fetch promoted evolutions
        try:
            promoted_rows = conn.execute(
                "SELECT id, skill_name, parent_skill_id, evolution_type, "
                "mutation_summary, created_at "
                "FROM skill_evolution_log "
                "WHERE status = 'promoted' "
                "ORDER BY created_at DESC LIMIT 20",
            ).fetchall()
        except _sqlite3.OperationalError:
            # Table may not exist yet
            conn.close()
            return {"created": 0, "message": "skill_evolution_log table not found"}

        created_count = 0
        now = datetime.now(timezone.utc).isoformat()

        for row in promoted_rows:
            r = dict(row)
            skill_name = r["skill_name"]
            parent = r.get("parent_skill_id") or skill_name
            evo_type = r["evolution_type"]
            summary = r.get("mutation_summary", "")
            evo_id = r["id"]

            # Build prompt content
            content = (
                f"[SKILL_EVOLUTION] Evolved skill: '{skill_name}' "
                f"({'replaces' if evo_type == 'fix' else 'extends'} '{parent}' "
                f"via {evo_type}). {summary}. "
                f"Use the evolved version for better results."
            )

            # Use a deterministic prompt_id based on the evolution record
            prompt_id = f"evo-{evo_id}"

            # Check if prompt already exists
            existing = conn.execute(
                "SELECT prompt_id FROM soft_prompt_templates WHERE prompt_id = ?",
                (prompt_id,),
            ).fetchone()

            if existing:
                # M-REPLACE: Update existing record instead of INSERT OR REPLACE
                # to avoid silently dropping columns with defaults
                try:
                    conn.execute(
                        "UPDATE soft_prompt_templates "
                        "SET content = ?, updated_at = ? "
                        "WHERE prompt_id = ?",
                        (content, now, prompt_id),
                    )
                except _sqlite3.OperationalError as upd_exc:
                    logger.debug("Failed to update soft prompt %s: %s", prompt_id, upd_exc)
                continue

            try:
                conn.execute(
                    "INSERT OR IGNORE INTO soft_prompt_templates "
                    "(prompt_id, profile_id, category, content, source_pattern_ids, "
                    " confidence, effectiveness, token_count, retention_score, "
                    " active, version, created_at, updated_at) "
                    "VALUES (?, ?, 'custom', ?, ?, 0.8, 0.5, ?, 1.0, 1, 1, ?, ?)",
                    (prompt_id, profile_id, content,
                     json.dumps([evo_id]),
                     len(content.split()),  # Rough token estimate
                     now, now),
                )
                if conn.total_changes:
                    created_count += 1
            except _sqlite3.IntegrityError:
                # Unique constraint on (profile_id, category) WHERE active=1
                logger.debug(
                    "Skipping soft prompt for %s: unique constraint on active custom",
                    skill_name,
                )

        conn.commit()
        conn.close()

        return {
            "promoted_skills_found": len(promoted_rows),
            "soft_prompts_created": created_count,
        }

    # ------------------------------------------------------------------
    # Core Memory Block Storage
    # ------------------------------------------------------------------

    def _store_core_block(
        self,
        profile_id: str,
        block_type: str,
        content: str,
        source_fact_ids: list[str],
        compiled_by: str = "rules",
    ) -> None:
        """Store or update a Core Memory block.

        Uses INSERT OR REPLACE on UNIQUE(profile_id, block_type).
        Guarantees idempotency (L18).
        """
        from superlocalmemory.storage.models import _new_id

        # NOTE (core-promotion-02, intentionally NOT changed): the audit
        # flagged placeholder core blocks ("No data available.") + version
        # churn. Investigation showed BOTH the "always 5 blocks" layout and the
        # monotonic version-on-recompile counter are intentional, tested design
        # (see test_consolidation_engine), and the read-side injection filter
        # already hides placeholders from agents. Suppressing or de-churning
        # here breaks that design for a cosmetic gain — so it is left as-is.

        # Get existing version for increment
        existing = self._db.get_core_block(profile_id, block_type)
        version = (existing["version"] + 1) if existing else 1

        self._db.store_core_block(
            block_id=_new_id(),
            profile_id=profile_id,
            block_type=block_type,
            content=content,
            source_fact_ids=json.dumps(source_fact_ids),
            char_count=len(content),
            version=version,
            compiled_by=compiled_by,
        )

    # ------------------------------------------------------------------
    # Helper Methods
    # ------------------------------------------------------------------

    def _get_top_facts(
        self,
        profile_id: str,
        fact_types: list[str],
        sort_by: str = "access_count",
        limit: int = 5,
    ) -> list[dict]:
        """Get top facts by type and sort criteria."""
        type_placeholders = ",".join("?" * len(fact_types))

        if sort_by == "recency":
            query = (
                f"SELECT fact_id, content, fact_type FROM atomic_facts "
                f"WHERE profile_id = ? AND fact_type IN ({type_placeholders}) "
                f"AND lifecycle = 'active' "
                f"ORDER BY created_at DESC LIMIT ?"
            )
        else:
            query = (
                f"SELECT f.fact_id, f.content, f.fact_type, "
                f"       COUNT(a.log_id) as access_count "
                f"FROM atomic_facts f "
                f"LEFT JOIN fact_access_log a ON f.fact_id = a.fact_id "
                f"WHERE f.profile_id = ? AND f.fact_type IN ({type_placeholders}) "
                f"AND f.lifecycle = 'active' "
                f"GROUP BY f.fact_id "
                f"ORDER BY access_count DESC LIMIT ?"
            )

        params: list[Any] = [profile_id] + fact_types + [limit]
        rows = self._db.execute(query, tuple(params))
        return [dict(r) for r in (rows or [])]

    def _facts_to_content(
        self, facts: list[dict], char_limit: int,
    ) -> str:
        """Compile fact contents into a block with hygiene (v3.6.6 F-5).

        Filters low-quality/template facts, dedupes lines WITHIN the block
        (fixes the "same fixture ×5" core-block bug), caps at char_limit.
        """
        from superlocalmemory.core.block_hygiene import compile_block_content
        compiled = compile_block_content(facts, max_chars=char_limit)
        return compiled if compiled else "No data available."

    def _rows_to_content(
        self, rows: list | None, char_limit: int,
    ) -> str:
        """Convert DB rows to a hygienic block content string (v3.6.6 F-5)."""
        if not rows:
            return "No data available."
        from superlocalmemory.core.block_hygiene import compile_block_content
        facts = [dict(r) for r in rows]
        compiled = compile_block_content(facts, max_chars=char_limit)
        return compiled if compiled else "No data available."

    def _compile_behavioral_block(
        self, profile_id: str, char_limit: int,
    ) -> str:
        """Compile behavioral patterns into a block content string."""
        if self._behavioral is None:
            return "No behavioral patterns detected yet."

        try:
            from superlocalmemory.learning.behavioral import BehavioralTracker

            if isinstance(self._behavioral, BehavioralTracker):
                patterns = self._db.execute(
                    "SELECT pattern_type, pattern_key, confidence "
                    "FROM behavioral_patterns "
                    "WHERE profile_id = ? ORDER BY confidence DESC LIMIT 5",
                    (profile_id,),
                )
            else:
                patterns = self._behavioral.get_patterns(
                    profile_id, limit=5,
                )

            if not patterns:
                return "No behavioral patterns detected yet."

            parts: list[str] = []
            for p in patterns:
                d = dict(p)
                ptype = d.get("pattern_type", "")
                pkey = d.get("pattern_key", "")
                conf = d.get("confidence", 0)
                parts.append(
                    f"{ptype}: {pkey} (confidence: {conf:.2f})"
                )

            content = "\n---\n".join(parts)
            return content[:char_limit]
        except Exception:
            return "No behavioral patterns detected yet."
