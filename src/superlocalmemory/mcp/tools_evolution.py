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
from typing import Callable


from superlocalmemory.mcp.tool_hints import READ_ONLY, hints
from superlocalmemory.core.admission import admits
from superlocalmemory.core.operation_request import OperationKind
from superlocalmemory.infra.data_root import state_path
from superlocalmemory.storage.read_connection import ReadConnectionFactory

logger = logging.getLogger(__name__)


def _skill_profile(engine, profile_id: str) -> tuple[str, dict | None]:
    """The named profile (it must exist), else the active one ("default"
    when there is no engine, as before)."""
    from superlocalmemory.mcp.request_profile import requested_profile, tool_profile

    if engine is None or not requested_profile(profile_id):
        return (engine.profile_id if engine else "default"), None
    return tool_profile(engine, profile_id)


def register_evolution_tools(server, get_engine: Callable) -> None:
    """Register evolution MCP tools for skill evolution intelligence."""

    @server.tool(title="Evolve Skill", annotations=hints(destructive=True, open_world=True))
    @admits(OperationKind.EVOLVE_SKILL)
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
                with open(config_path, encoding="utf-8") as f:
                    cfg = json.load(f)
                evo_cfg = cfg.get("evolution", {})

            if not evo_cfg.get("enabled", False):
                return {
                    "success": False,
                    "error": (
                        "Evolution is disabled. Enable via: slm config set evolution.enabled true"
                    ),
                }

            from superlocalmemory.evolution.skill_evolver import SkillEvolver
            from superlocalmemory.evolution.types import (
                EvolutionCandidate,
                EvolutionType,
                TriggerType,
            )

            # Map string to enum
            type_map = {
                "fix": EvolutionType.FIX,
                "derived": EvolutionType.DERIVED,
                "captured": EvolutionType.CAPTURED,
            }
            evo_type = type_map.get(evolution_type, EvolutionType.FIX)

            # Build a minimal config for the evolver
            class _EvoCfg:
                enabled = True
                backend = evo_cfg.get("backend", "auto")
                max_evolutions_per_cycle = evo_cfg.get("max_evolutions_per_cycle", 3)

            class _Cfg:
                evolution = _EvoCfg()

            db_path = str(state_path("memory.db"))
            from superlocalmemory.evolution.workspace_artifacts import workspace_activator

            engine = get_engine()
            profile_id = engine.profile_id if engine else "default"
            activator = workspace_activator(state_path("memory.db").parent, profile_id)
            evolver = SkillEvolver(db_path, _Cfg(), profile_id=profile_id, activator=activator)

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

            # Process through the pipeline using the same workspace as its artifacts.

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

    @server.tool(title="Get Skill Health", annotations=READ_ONLY)
    async def skill_health(
        skill_name: str = "",
        include_history: bool = False,
        profile_id: str = "",
    ) -> dict:
        """Get health metrics for a skill or all skills.

        Queries behavioral assertions (skill_performance category) and
        tool_events to compute per-skill invocation counts, effective
        rates, and status.

        Args:
            skill_name: Specific skill name (empty = all skills)
            include_history: Include recent tool event history per skill
            profile_id: The profile to read (empty = the active one)
        """
        try:
            engine = get_engine()
            profile_id, refused = _skill_profile(engine, profile_id)
            if refused:
                return {"skills": [], "skill_count": 0, **refused}
            db_path = state_path("memory.db")
            from superlocalmemory.learning.skill_health import skill_health as read_health

            return read_health(db_path, profile_id, skill_name, include_history)
        except Exception as exc:
            logger.debug("skill_health failed: %s", exc)
            return {"skills": [], "skill_count": 0, "error": str(exc)}

    @server.tool(title="Get Skill Lineage", annotations=READ_ONLY)
    async def skill_lineage(
        skill_name: str = "",
        profile_id: str = "",
    ) -> dict:
        """Get evolution lineage for a skill.

        Queries the skill_evolution_log table and builds a version tree
        showing how skills evolved from their parents.

        Args:
            skill_name: Specific skill name (empty = all skills)
            profile_id: The profile to read (empty = the active one)
        """
        try:
            engine = get_engine()
            profile_id, refused = _skill_profile(engine, profile_id)
            if refused:
                return {"lineage": [], "lineage_count": 0, "tree": {}, **refused}
            db_path = state_path("memory.db")
            conn = ReadConnectionFactory(db_path).open()

            if skill_name:
                rows = conn.execute(
                    "SELECT id, skill_name, parent_skill_id, evolution_type, "
                    "trigger_type, generation, status, mutation_summary, "
                    "blind_verified, created_at, completed_at "
                    "FROM skill_evolution_log "
                    "WHERE profile_id = ? AND (skill_name = ? OR parent_skill_id = ?) "
                    "ORDER BY created_at ASC",
                    (profile_id, skill_name, skill_name),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT id, skill_name, parent_skill_id, evolution_type, "
                    "trigger_type, generation, status, mutation_summary, "
                    "blind_verified, created_at, completed_at "
                    "FROM skill_evolution_log "
                    "WHERE profile_id = ? "
                    "ORDER BY created_at DESC LIMIT 100",
                    (profile_id,),
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
                tree[root]["evolutions"].append(
                    {
                        "id": entry["id"],
                        "skill_name": entry["skill_name"],
                        "evolution_type": entry["evolution_type"],
                        "status": entry["status"],
                        "generation": entry["generation"],
                        "created_at": entry["created_at"],
                    }
                )

            return {
                "lineage": lineage,
                "lineage_count": len(lineage),
                "tree": tree,
            }
        except Exception as exc:
            # 4.2.3: a store where evolution never ran has no log table yet;
            # that is an empty lineage, not a failure.
            if "no such table" in str(exc):
                return {"success": True, "lineage": [], "lineage_count": 0, "tree": {},
                        "note": "No skill has evolved in this profile yet."}
            logger.debug("skill_lineage failed: %s", exc)
            return {"success": False, "lineage": [], "lineage_count": 0, "tree": {},
                    "error": str(exc)}
