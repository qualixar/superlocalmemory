# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under the Elastic License 2.0 - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""SuperLocalMemory V3.3 — New MCP Tools (6 tools).

forget               — Ebbinghaus forgetting decay cycle.
quantize             — EAP embedding quantization cycle.
consolidate_cognitive — CCQ cognitive consolidation pipeline.
get_soft_prompts     — Retrieve active soft prompts.
reap_processes       — Find and kill orphaned SLM processes.
get_retention_stats  — Memory retention zone distribution.

Part of Qualixar | Author: Varun Pratap Bhardwaj
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)

MEMORY_DIR = Path.home() / ".superlocalmemory"
DB_PATH = MEMORY_DIR / "memory.db"


def _emit_event(event_type: str, payload: dict | None = None,
                source_agent: str = "mcp_client") -> None:  # V3.3.12: see also mcp/shared.py
    """Emit an event to the EventBus (best-effort, never raises)."""
    try:
        from superlocalmemory.infra.event_bus import EventBus
        bus = EventBus.get_instance(str(DB_PATH))
        bus.emit(event_type, payload=payload, source_agent=source_agent,
                 source_protocol="mcp")
    except Exception:
        pass


def register_v33_tools(server, get_engine: Callable) -> None:
    """Register 6 V3.3 MCP tools on *server*."""

    # ------------------------------------------------------------------
    # 1. forget — Ebbinghaus forgetting decay cycle
    # ------------------------------------------------------------------
    @server.tool()
    async def forget(
        profile_id: str = "",
        dry_run: bool = True,
    ) -> dict:
        """Run Ebbinghaus forgetting decay cycle.

        Computes retention scores for all facts and transitions
        memories between zones (active -> warm -> cold -> archive ->
        forgotten) based on access patterns and importance.

        Run with dry_run=True first to preview changes.

        Args:
            profile_id: Profile to process (default: active profile).
            dry_run: If True, compute stats but don't apply transitions.
        """
        try:
            engine = get_engine()
            pid = profile_id or engine.profile_id

            from superlocalmemory.math.ebbinghaus import EbbinghausCurve
            from superlocalmemory.learning.forgetting_scheduler import (
                ForgettingScheduler,
            )

            ebbinghaus = EbbinghausCurve(engine._config.forgetting)
            scheduler = ForgettingScheduler(
                engine._db, ebbinghaus, engine._config.forgetting,
            )

            if dry_run:
                # Dry run: compute retention stats without applying changes
                facts = engine._db.get_all_facts(pid)
                zones = {"active": 0, "warm": 0, "cold": 0, "archive": 0, "forgotten": 0}
                for f in facts:
                    strength = ebbinghaus.memory_strength(f.access_count or 0, f.importance or 0.5, 0, 0.0)
                    r = ebbinghaus.retention(hours_since_access = 0.0, strength=strength)
                    zone = ebbinghaus.lifecycle_zone(r)
                    zones[zone] = zones.get(zone, 0) + 1
                result = {"total": len(facts), "transitions": 0, "dry_run_zones": zones}
            else:
                result = scheduler.run_decay_cycle(pid, force=True)

            _emit_event("forgetting.cycle_complete", {
                "profile_id": pid,
                "dry_run": dry_run,
                "total": result.get("total", 0),
                "transitions": result.get("transitions", 0),
            })

            return {"success": True, "dry_run": dry_run, **result}

        except Exception as exc:
            logger.exception("forget tool failed")
            return {"success": False, "error": str(exc)}

    # ------------------------------------------------------------------
    # 2. quantize — EAP embedding quantization cycle
    # ------------------------------------------------------------------
    @server.tool()
    async def quantize(
        profile_id: str = "",
        dry_run: bool = True,
    ) -> dict:
        """Run EAP quantization cycle.

        Maps Ebbinghaus retention scores to embedding precision
        levels (32/8/4/2/0 bits). Downgrades low-retention embeddings
        to save storage; upgrades when retention improves.

        Run with dry_run=True first to preview changes.

        Args:
            profile_id: Profile to process (default: active profile).
            dry_run: If True, compute stats but don't apply changes.
        """
        try:
            engine = get_engine()
            pid = profile_id or engine.profile_id

            from superlocalmemory.math.ebbinghaus import EbbinghausCurve
            from superlocalmemory.math.polar_quant import PolarQuantEncoder
            from superlocalmemory.math.qjl import QJLEncoder
            from superlocalmemory.dynamics.eap_scheduler import EAPScheduler
            from superlocalmemory.storage.quantized_store import (
                QuantizedEmbeddingStore,
            )

            ebbinghaus = EbbinghausCurve(engine._config.forgetting)
            polar = PolarQuantEncoder(engine._config.quantization.polar)
            qjl = QJLEncoder(engine._config.quantization.qjl)
            qstore = QuantizedEmbeddingStore(
                engine._db, polar, qjl, engine._config.quantization,
            )
            scheduler = EAPScheduler(
                engine._db, ebbinghaus, qstore, engine._config.quantization,
            )

            if dry_run:
                # Dry run: report current quantization state without changes
                facts = engine._db.get_all_facts(pid)
                result = {"total": len(facts), "would_quantize": 0, "dry_run": True}
            else:
                result = scheduler.run_eap_cycle(pid)

            _emit_event("eap.cycle_complete", {
                "profile_id": pid,
                "dry_run": dry_run,
                "total": result.get("total", 0),
                "downgrades": result.get("downgrades", 0),
                "upgrades": result.get("upgrades", 0),
            })

            return {"success": True, "dry_run": dry_run, **result}

        except Exception as exc:
            logger.exception("quantize tool failed")
            return {"success": False, "error": str(exc)}

    # ------------------------------------------------------------------
    # 3. consolidate_cognitive — CCQ cognitive consolidation
    # ------------------------------------------------------------------
    @server.tool()
    async def consolidate_cognitive(
        profile_id: str = "",
    ) -> dict:
        """Run CCQ cognitive consolidation pipeline.

        Extracts patterns from cold/archive memories by clustering
        related facts, generating gist summaries, and compressing
        source embeddings. Like sleep-time memory consolidation.

        Args:
            profile_id: Profile to process (default: active profile).
        """
        try:
            engine = get_engine()
            pid = profile_id or engine.profile_id

            from superlocalmemory.encoding.cognitive_consolidator import (
                CognitiveConsolidator,
            )

            consolidator = CognitiveConsolidator(db=engine._db)
            result = consolidator.run_pipeline(pid)

            _emit_event("ccq.consolidation_complete", {
                "profile_id": pid,
                "clusters_processed": result.clusters_processed,
                "blocks_created": result.blocks_created,
            })

            return {
                "success": True,
                "clusters_processed": result.clusters_processed,
                "blocks_created": result.blocks_created,
                "facts_archived": result.facts_archived,
                "compression_ratio": round(result.compression_ratio, 3),
            }

        except Exception as exc:
            logger.exception("consolidate_cognitive tool failed")
            return {"success": False, "error": str(exc)}

    # ------------------------------------------------------------------
    # 4. get_soft_prompts — Retrieve active soft prompts
    # ------------------------------------------------------------------
    @server.tool()
    async def get_soft_prompts(
        profile_id: str = "",
    ) -> dict:
        """Get active soft prompts (auto-learned user patterns).

        Returns soft prompt templates generated from behavioral
        patterns. These are injected into conversation context to
        personalize AI responses.

        Args:
            profile_id: Profile to query (default: active profile).
        """
        try:
            engine = get_engine()
            pid = profile_id or engine.profile_id

            rows = engine._db.execute(
                "SELECT prompt_id, category, content, confidence, "
                "  effectiveness, token_count, version, created_at "
                "FROM soft_prompt_templates "
                "WHERE profile_id = ? AND active = 1 "
                "ORDER BY confidence DESC",
                (pid,),
            )

            prompts = []
            for row in rows:
                r = dict(row)
                prompts.append({
                    "prompt_id": r["prompt_id"],
                    "category": r["category"],
                    "content": r["content"],
                    "confidence": round(float(r["confidence"]), 3),
                    "effectiveness": round(float(r["effectiveness"]), 3),
                    "token_count": int(r["token_count"]),
                    "version": int(r["version"]),
                    "created_at": r["created_at"],
                })

            return {
                "success": True,
                "prompts": prompts,
                "count": len(prompts),
                "profile": pid,
            }

        except Exception as exc:
            logger.exception("get_soft_prompts tool failed")
            return {"success": False, "error": str(exc)}

    # ------------------------------------------------------------------
    # 5. reap_processes — Find and kill orphaned SLM processes
    # ------------------------------------------------------------------
    @server.tool()
    async def reap_processes(
        dry_run: bool = True,
    ) -> dict:
        """Find and kill orphaned SLM processes.

        Detects SLM embedding workers and other subprocesses whose
        parent has died (orphans). Safely terminates them with SIGTERM.

        Run with dry_run=True first to preview what would be killed.

        Args:
            dry_run: If True, report orphans but don't kill them.
        """
        try:
            from superlocalmemory.infra.process_reaper import (
                cleanup_all_orphans,
                ReaperConfig,
            )

            config = ReaperConfig()
            result = cleanup_all_orphans(config, dry_run=dry_run)

            return {
                "success": True,
                "dry_run": dry_run,
                "total_found": result.get("total_found", 0),
                "orphans_found": result.get("orphans_found", 0),
                "killed": result.get("killed", 0),
                "skipped": result.get("skipped", 0),
            }

        except Exception as exc:
            logger.exception("reap_processes tool failed")
            return {"success": False, "error": str(exc)}

    # ------------------------------------------------------------------
    # 6. get_retention_stats — Memory retention zone distribution
    # ------------------------------------------------------------------
    @server.tool()
    async def get_retention_stats(
        profile_id: str = "",
    ) -> dict:
        """Get memory retention statistics (zone distribution, decay rates).

        Queries the fact_retention table for zone counts and average
        retention scores per zone. Shows how memories are distributed
        across the Ebbinghaus decay lifecycle.

        Args:
            profile_id: Profile to query (default: active profile).
        """
        try:
            engine = get_engine()
            pid = profile_id or engine.profile_id

            # Zone distribution counts
            rows = engine._db.execute(
                "SELECT lifecycle_zone, COUNT(*) as cnt, "
                "  AVG(retention_score) as avg_score "
                "FROM fact_retention "
                "WHERE profile_id = ? "
                "GROUP BY lifecycle_zone",
                (pid,),
            )

            zones: dict[str, dict] = {}
            total = 0
            for row in rows:
                r = dict(row)
                zone = r["lifecycle_zone"]
                count = int(r["cnt"])
                avg = round(float(r["avg_score"]), 3) if r["avg_score"] else 0.0
                zones[zone] = {"count": count, "avg_retention": avg}
                total += count

            return {
                "success": True,
                "total": total,
                "active": zones.get("active", {}).get("count", 0),
                "warm": zones.get("warm", {}).get("count", 0),
                "cold": zones.get("cold", {}).get("count", 0),
                "archive": zones.get("archive", {}).get("count", 0),
                "forgotten": zones.get("forgotten", {}).get("count", 0),
                "zones": zones,
                "profile": pid,
            }

        except Exception as exc:
            logger.exception("get_retention_stats tool failed")
            return {"success": False, "error": str(exc)}

    # ------------------------------------------------------------------
    # 7. run_maintenance — V3.3.12: Combined periodic maintenance cycle
    # ------------------------------------------------------------------
    @server.tool()
    async def run_maintenance(profile_id: str = "") -> dict:
        """Run all periodic maintenance tasks in a single call.

        Combines Langevin dynamics stepping, Ebbinghaus forgetting decay,
        and behavioral pattern mining into one convenient maintenance cycle.
        Clients should call this periodically (e.g., at session end).

        Args:
            profile_id: Profile to maintain (default: active profile).
        """
        try:
            engine = get_engine()
            pid = profile_id or engine.profile_id
            results = {}

            # 1. Langevin dynamics step (lifecycle evolution)
            try:
                from superlocalmemory.core.maintenance import run_maintenance as _run_maint
                maint_result = _run_maint(engine._db, engine._config, pid)
                results["langevin"] = {"updated": maint_result.get("updated", 0)}
            except Exception as exc:
                results["langevin"] = {"error": str(exc)}

            # 2. Ebbinghaus forgetting decay
            try:
                from superlocalmemory.math.ebbinghaus import EbbinghausCurve
                from superlocalmemory.learning.forgetting_scheduler import ForgettingScheduler
                ebbinghaus = EbbinghausCurve(engine._config.forgetting)
                scheduler = ForgettingScheduler(engine._db, ebbinghaus, engine._config.forgetting)
                decay_result = scheduler.run_decay_cycle(pid, force=False)
                results["forgetting"] = decay_result
            except Exception as exc:
                results["forgetting"] = {"error": str(exc)}

            # 3. Behavioral pattern mining
            try:
                from superlocalmemory.learning.consolidation_worker import ConsolidationWorker
                cw = ConsolidationWorker(MEMORY_DIR / "memory.db", MEMORY_DIR / "learning.db")
                count = cw._generate_patterns(pid, False)
                results["behavioral"] = {"patterns_mined": count}
            except Exception as exc:
                results["behavioral"] = {"error": str(exc)}

            return {"success": True, "profile": pid, **results}

        except Exception as exc:
            logger.exception("run_maintenance failed")
            return {"success": False, "error": str(exc)}
