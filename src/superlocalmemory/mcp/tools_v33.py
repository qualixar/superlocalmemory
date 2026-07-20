# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
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
from typing import Callable

from mcp.types import ToolAnnotations

logger = logging.getLogger(__name__)

from superlocalmemory.infra.data_root import state_path
from superlocalmemory.mcp.shared import authorize_mcp_mutation


def _try_daemon_post(path: str, body: dict, timeout_s: float = 60.0) -> dict | None:
    """POST to the daemon, return parsed dict or None if daemon unavailable.

    Keeps heavy imports (CognitiveConsolidator, ForgettingScheduler,
    ConsolidationWorker) out of the MCP process when a daemon is up.
    Returns None on any failure so the caller can fall back to local
    execution.
    """
    try:
        from superlocalmemory.cli.daemon import _get_port, is_daemon_running
        if not is_daemon_running():
            return None
        import json as _json
        import urllib.request as _urq
        port = _get_port()
        req = _urq.Request(
            f"http://127.0.0.1:{port}{path}",
            data=_json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with _urq.urlopen(req, timeout=timeout_s) as resp:
            raw = resp.read().decode() or "{}"
        return _json.loads(raw)
    except Exception as exc:
        logger.warning("daemon POST %s failed: %s", path, exc)
        return None


def _emit_event(event_type: str, payload: dict | None = None,
                source_agent: str = "mcp_client") -> None:  # V3.3.12: see also mcp/shared.py
    """Emit an event to the EventBus (best-effort, never raises)."""
    try:
        from superlocalmemory.infra.event_bus import EventBus
        bus = EventBus.get_instance(str(state_path("memory.db")))
        bus.emit(event_type, payload=payload, source_agent=source_agent,
                 source_protocol="mcp")
    except Exception:
        pass


def register_v33_tools(server, get_engine: Callable) -> None:
    """Register 6 V3.3 MCP tools on *server*."""

    # ------------------------------------------------------------------
    # 1. forget — Ebbinghaus forgetting decay cycle
    # ------------------------------------------------------------------
    @server.tool(annotations=ToolAnnotations(destructiveHint=True))
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
                rows = engine._db.execute(
                    "SELECT lifecycle_zone, COUNT(*) as cnt "
                    "FROM fact_retention WHERE profile_id = ? "
                    "GROUP BY lifecycle_zone",
                    (pid,),
                )
                zones = {"active": 0, "warm": 0, "cold": 0, "archive": 0, "forgotten": 0}
                total = 0
                for row in rows:
                    r = dict(row)
                    zones[r["lifecycle_zone"]] = int(r["cnt"])
                    total += int(r["cnt"])
                result = {"total": total, "transitions": 0, "dry_run_zones": zones}
            else:
                authorization = authorize_mcp_mutation(
                    engine,
                    "delete",
                    mutation_source="mcp-forgetting-cycle",
                    profile_id=pid,
                )
                result = scheduler.run_decay_cycle(pid, force=True)
                authorization.complete()

            if not dry_run:
                _emit_event("forgetting.cycle_complete", {
                    "profile_id": pid,
                    "dry_run": False,
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
                authorization = authorize_mcp_mutation(
                    engine,
                    "update",
                    mutation_source="mcp-quantization-cycle",
                    profile_id=pid,
                )
                result = scheduler.run_eap_cycle(pid)
                authorization.complete()

            if not dry_run:
                _emit_event("eap.cycle_complete", {
                    "profile_id": pid,
                    "dry_run": False,
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

            # v3.4.26: prefer the daemon's /consolidate/cognitive endpoint
            # so the heavy CognitiveConsolidator import stays out of the
            # MCP process. Fall back to local import only if no daemon.
            import asyncio
            # blocking urllib (60s timeout) — keep it off the MCP event loop.
            daemon_result = await asyncio.to_thread(
                _try_daemon_post, "/consolidate/cognitive", {"profile_id": pid},
            )
            if daemon_result is not None:
                _emit_event("ccq.consolidation_complete", {
                    "profile_id": pid,
                    "clusters_processed": daemon_result.get("clusters_processed", 0),
                    "blocks_created": daemon_result.get("blocks_created", 0),
                })
                return {
                    "success": True,
                    "clusters_processed": daemon_result.get("clusters_processed", 0),
                    "blocks_created": daemon_result.get("blocks_created", 0),
                    "profile_id": pid,
                    "via": "daemon",
                }

            from superlocalmemory.encoding.cognitive_consolidator import (
                CognitiveConsolidator,
            )

            authorization = authorize_mcp_mutation(
                engine,
                "update",
                mutation_source="mcp-cognitive-consolidation",
                profile_id=pid,
            )
            consolidator = CognitiveConsolidator(db=engine._db)
            result = consolidator.run_pipeline(pid)
            authorization.complete()

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
    @server.tool(annotations=ToolAnnotations(readOnlyHint=True))
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
            engine = get_engine()
            authorization = None
            if not dry_run:
                authorization = authorize_mcp_mutation(
                    engine,
                    "delete",
                    mutation_source="mcp-process-reaper",
                )
            from superlocalmemory.infra.process_reaper import (
                cleanup_all_orphans,
                ReaperConfig,
            )

            config = ReaperConfig()
            result = cleanup_all_orphans(config, dry_run=dry_run)
            if authorization is not None:
                authorization.complete()

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

            # v3.4.26: prefer the daemon so ForgettingScheduler /
            # ConsolidationWorker / EbbinghausCurve don't load inside
            # the MCP process.
            import asyncio
            # blocking urllib — keep it off the MCP event loop.
            daemon_result = await asyncio.to_thread(
                _try_daemon_post, "/maintenance/run", {"profile_id": pid},
            )
            if daemon_result is not None:
                daemon_result.setdefault("success", True)
                daemon_result["via"] = "daemon"
                return daemon_result

            authorization = authorize_mcp_mutation(
                engine,
                "update",
                mutation_source="mcp-maintenance-cycle",
                profile_id=pid,
            )
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
                cw = ConsolidationWorker(engine._db.db_path, engine._db.db_path.parent / "learning.db",)
                count = cw._generate_patterns(pid, False)
                results["behavioral"] = {"patterns_mined": count}
            except Exception as exc:
                results["behavioral"] = {"error": str(exc)}

            authorization.complete()
            return {"success": True, "profile": pid, **results}

        except Exception as exc:
            logger.exception("run_maintenance failed")
            return {"success": False, "error": str(exc)}
