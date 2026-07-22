# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""SLM v3.4.11 "Skill Evolution" — Evolution MCP Tools.

Three evolution tools:
  - evolve_skill: Manually trigger evolution for a specific skill
  - skill_health: Get health metrics for a skill or all skills
  - skill_lineage: Get evolution lineage for a skill

Part of Qualixar | Author: Varun Pratap Bhardwaj
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from mcp.types import ToolAnnotations
from superlocalmemory.infra.data_root import state_path

logger = logging.getLogger(__name__)

def register_evolution_tools(server, get_engine: Callable) -> None:
    """Register evolution MCP tools for skill evolution intelligence."""

    @server.tool()
    async def evolve_skill(
        skill_name: str,
        evolution_type: str = "fix",
        reason: str = "",
    ) -> dict:
        """Manually trigger evolution for a specific skill.

        Runs the full evolution pipeline: screen -> confirm -> mutate ->
        blind verify -> persist. Requires evolution to be enabled in config.

        Args:
            skill_name: Name of the skill to evolve (e.g. "brainstorming")
            evolution_type: One of 'fix', 'derived', 'captured'
            reason: Optional reason / evidence for the evolution
        """
        try:
            # Check if evolution is enabled in config
            config_path = state_path("config.json")
            evo_cfg = {}
            if config_path.exists():
                with open(config_path) as f:
                    cfg = json.load(f)
                evo_cfg = cfg.get("evolution", {})

            if not evo_cfg.get("enabled", False):
                return {
                    "success": False,
                    "error": "Evolution is disabled. Enable via: slm config set evolution.enabled true",
                }

            from superlocalmemory.evolution.skill_evolver import SkillEvolver
            from superlocalmemory.evolution.types import (
                EvolutionCandidate,
                EvolutionType,
                TriggerType,
            )

            # Map string to enum
            type_map = {"fix": EvolutionType.FIX, "derived": EvolutionType.DERIVED, "captured": EvolutionType.CAPTURED}
            evo_type = type_map.get(evolution_type, EvolutionType.FIX)

            # Build a minimal config for the evolver
            class _EvoCfg:
                enabled = True
                backend = evo_cfg.get("backend", "auto")
                max_evolutions_per_cycle = evo_cfg.get("max_evolutions_per_cycle", 3)

            class _Cfg:
                evolution = _EvoCfg()

            db_path = str(state_path("memory.db"))
            evolver = SkillEvolver(db_path, _Cfg())

            # Build candidate from manual trigger
            evidence = (reason,) if reason else ("Manual evolution trigger via MCP",)
            candidate = EvolutionCandidate(
                skill_name=skill_name,
                evolution_type=evo_type,
                trigger=TriggerType.HEALTH_CHECK,
                evidence=evidence,
                effective_score=0.0,
                invocation_count=0,
            )

            # Process through the pipeline
            engine = get_engine()
            profile_id = engine.profile_id if engine else "default"

            evolver._store.reset_cycle(profile_id)
            # audit-10 fix: go through evolve_candidate so the manual/MCP path
            # runs under a budget cycle and honours the LLM-call / wall-time /
            # per-day caps (previously _process_candidate ran uncapped here).
            outcome = evolver.evolve_candidate(candidate, profile_id)

            # Fetch the latest record for this skill to return details
            recent = evolver._store.get_skill_history(skill_name, profile_id, limit=1)
            record_info = {}
            if recent:
                r = recent[0]
                record_info = {
                    "id": r.id,
                    "status": r.status.value,
                    "mutation_summary": r.mutation_summary,
                    "blind_verified": r.blind_verified,
                    "rejection_reason": r.rejection_reason,
                }

            return {
                "success": outcome == "evolved",
                "outcome": outcome,
                "skill_name": skill_name,
                "evolution_type": evolution_type,
                **record_info,
            }
        except Exception as exc:
            logger.debug("evolve_skill failed: %s", exc)
            return {"success": False, "error": str(exc)}

    @server.tool(annotations=ToolAnnotations(readOnlyHint=True))
    async def skill_health(
        skill_name: str = "",
        include_history: bool = False,
    ) -> dict:
        """Get health metrics for a skill or all skills.

        Queries behavioral assertions (skill_performance category) and
        tool_events to compute per-skill invocation counts, effective
        rates, and status.

        Args:
            skill_name: Specific skill name (empty = all skills)
            include_history: Include recent tool event history per skill
        """
        try:
            engine = get_engine()
            profile_id = engine.profile_id if engine else "default"
            db_path = str(state_path("memory.db"))

            conn = sqlite3.connect(db_path, timeout=10)
            conn.row_factory = sqlite3.Row

            # Gather per-skill invocation stats from tool_events
            # Skills are logged as tool_name='Skill' with actual skill name in input_summary
            if skill_name:
                # M-LIKE: Escape LIKE wildcards in user-provided skill_name
                safe_name = skill_name.replace('\\', '\\\\').replace('%', r'\%').replace('_', r'\_')
                event_query = (
                    "SELECT input_summary, event_type, created_at, duration_ms "
                    "FROM tool_events "
                    "WHERE profile_id = ? AND tool_name = 'Skill' "
                    "AND input_summary LIKE ? ESCAPE '\\' "
                    "ORDER BY created_at DESC"
                )
                event_rows = conn.execute(event_query, (profile_id, f"%{safe_name}%")).fetchall()
                # Aggregate
                invocations = len(event_rows)
                errors = sum(1 for r in event_rows if dict(r).get("event_type") == "error")
                last_invoked = dict(event_rows[0]).get("created_at", "") if event_rows else ""
                effective_rate = ((invocations - errors) / invocations) if invocations > 0 else 0.0
                skill_entries = [{
                    "name": skill_name,
                    "invocations": invocations,
                    "errors": errors,
                    "effective_rate": round(effective_rate, 4),
                    "last_invoked": last_invoked,
                    "status": "healthy" if effective_rate >= 0.7 else ("degraded" if effective_rate >= 0.4 else "critical"),
                }]
                if include_history:
                    skill_entries[0]["recent_events"] = [
                        dict(r) for r in event_rows[:10]
                    ]
            else:
                # Get all Skill tool events and extract skill names from input_summary
                event_query = (
                    "SELECT input_summary, event_type, created_at "
                    "FROM tool_events "
                    "WHERE profile_id = ? AND tool_name = 'Skill' "
                    "ORDER BY created_at DESC LIMIT 500"
                )
                event_rows = conn.execute(event_query, (profile_id,)).fetchall()

                # Parse skill names from input_summary and aggregate
                from collections import defaultdict
                skill_stats: dict = defaultdict(lambda: {"invocations": 0, "errors": 0, "last_invoked": ""})
                for row in event_rows:
                    r = dict(row)
                    summary = r.get("input_summary", "")
                    # Extract skill name from JSON or plain text
                    sname = ""
                    try:
                        parsed = json.loads(summary)
                        sname = parsed.get("skill", "") or parsed.get("name", "")
                    except (json.JSONDecodeError, TypeError):
                        if ":" in summary:
                            sname = summary.split('"')[1] if '"' in summary else summary.strip()
                    if not sname:
                        continue
                    stats = skill_stats[sname]
                    stats["invocations"] += 1
                    if r.get("event_type") == "error":
                        stats["errors"] += 1
                    if not stats["last_invoked"]:
                        stats["last_invoked"] = r.get("created_at", "")

                skill_entries = []
                for sname, stats in sorted(skill_stats.items(), key=lambda x: x[1]["invocations"], reverse=True)[:50]:
                    inv = stats["invocations"]
                    errs = stats["errors"]
                    eff = ((inv - errs) / inv) if inv > 0 else 0.0
                    skill_entries.append({
                        "name": sname,
                        "invocations": inv,
                        "errors": errs,
                        "effective_rate": round(eff, 4),
                        "last_invoked": stats["last_invoked"],
                        "status": "healthy" if eff >= 0.7 else ("degraded" if eff >= 0.4 else "critical"),
                    })

            # Gather skill_performance assertions
            assertion_query = (
                "SELECT trigger_condition, action, confidence "
                "FROM behavioral_assertions "
                "WHERE profile_id = ? AND category = 'skill_performance'"
            )
            assertion_params = [profile_id]
            if skill_name:
                safe_assert_name = skill_name.replace('\\', '\\\\').replace('%', r'\%').replace('_', r'\_')
                assertion_query += " AND trigger_condition LIKE ? ESCAPE '\\'"
                assertion_params.append(f"%{safe_assert_name}%")
            assertion_rows = conn.execute(assertion_query, tuple(assertion_params)).fetchall()

            skills = skill_entries

            # Add assertion insights
            assertion_insights = [
                {"trigger": dict(a)["trigger_condition"], "action": dict(a)["action"], "confidence": dict(a)["confidence"]}
                for a in assertion_rows
            ]

            conn.close()

            return {
                "skills": skills,
                "skill_count": len(skills),
                "assertion_insights": assertion_insights,
                "profile_id": profile_id,
            }
        except Exception as exc:
            logger.debug("skill_health failed: %s", exc)
            return {"skills": [], "skill_count": 0, "error": str(exc)}

    @server.tool(annotations=ToolAnnotations(readOnlyHint=True))
    async def skill_lineage(
        skill_name: str = "",
    ) -> dict:
        """Get evolution lineage for a skill.

        Queries the skill_evolution_log table and builds a version tree
        showing how skills evolved from their parents.

        Args:
            skill_name: Specific skill name (empty = all skills)
        """
        try:
            db_path = str(state_path("memory.db"))
            conn = sqlite3.connect(db_path, timeout=10)
            conn.row_factory = sqlite3.Row

            if skill_name:
                rows = conn.execute(
                    "SELECT id, skill_name, parent_skill_id, evolution_type, "
                    "trigger_type, generation, status, mutation_summary, "
                    "blind_verified, created_at, completed_at "
                    "FROM skill_evolution_log "
                    "WHERE skill_name = ? OR parent_skill_id = ? "
                    "ORDER BY created_at ASC",
                    (skill_name, skill_name),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT id, skill_name, parent_skill_id, evolution_type, "
                    "trigger_type, generation, status, mutation_summary, "
                    "blind_verified, created_at, completed_at "
                    "FROM skill_evolution_log "
                    "ORDER BY created_at DESC LIMIT 100",
                ).fetchall()

            conn.close()

            lineage = [
                {
                    "id": dict(r)["id"],
                    "skill_name": dict(r)["skill_name"],
                    "parent_skill_id": dict(r).get("parent_skill_id", ""),
                    "evolution_type": dict(r)["evolution_type"],
                    "trigger": dict(r)["trigger_type"],
                    "generation": dict(r).get("generation", 0),
                    "status": dict(r)["status"],
                    "mutation_summary": dict(r).get("mutation_summary", ""),
                    "blind_verified": bool(dict(r).get("blind_verified", 0)),
                    "created_at": dict(r).get("created_at", ""),
                    "completed_at": dict(r).get("completed_at", ""),
                }
                for r in rows
            ]

            # Build tree structure: group by root skill
            tree: dict = {}
            for entry in lineage:
                root = entry.get("parent_skill_id") or entry["skill_name"]
                if root not in tree:
                    tree[root] = {"root": root, "evolutions": []}
                tree[root]["evolutions"].append({
                    "id": entry["id"],
                    "skill_name": entry["skill_name"],
                    "evolution_type": entry["evolution_type"],
                    "status": entry["status"],
                    "generation": entry["generation"],
                    "created_at": entry["created_at"],
                })

            return {
                "lineage": lineage,
                "lineage_count": len(lineage),
                "tree": tree,
            }
        except Exception as exc:
            logger.debug("skill_lineage failed: %s", exc)
            return {"lineage": [], "lineage_count": 0, "tree": {}, "error": str(exc)}
