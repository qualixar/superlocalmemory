# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""MCP ``expand``: drill into one recall result (layered recall, depth 2).

Registered by ``tools_core.register_core_tools`` so it sits in the same
profiles as ``recall`` and ``fetch``; kept in its own module for size.
"""

from __future__ import annotations

import logging
from typing import Callable


from superlocalmemory.mcp.tool_hints import READ_ONLY
from superlocalmemory.core.admission import admits
from superlocalmemory.core.operation_request import OperationKind
from superlocalmemory.mcp.remote_visibility import visible_facts

logger = logging.getLogger(__name__)


def register_expand_tool(server, get_engine: Callable) -> None:
    @server.tool(title="Expand Memory", annotations=READ_ONLY)
    @admits(OperationKind.RECALL)
    async def expand(memory_id: str, depth: int = 2, max_tokens: int = 0,
                     profile_id: str = "") -> dict:
        """Drill into one recall result: the full fact plus, at depth 2, its
        source memory, same-session neighbours, facts about the same entities
        and the correction that replaced it. ``memory_id`` takes the result's
        ``fact_id`` or ``memory_id``. Same profile, quarantine and visibility
        checks as ``fetch``; ``max_tokens`` is never exceeded.
        """
        from superlocalmemory.core import recall_layers
        try:
            _depth = recall_layers.parse_depth(depth, default=2)
            _budget = recall_layers.parse_max_tokens(max_tokens)
        except (TypeError, ValueError) as exc:
            return {"success": False, "code": "INVALID_LAYERS", "error": str(exc)}
        try:
            from superlocalmemory.mcp import tools_core

            engine = get_engine()
            pid, refused = await tools_core._call_profile(get_engine, profile_id)
            if refused:
                return refused
            ident = str(memory_id or "").strip()
            facts = engine._db.get_facts_by_ids([ident], pid) if ident else []
            if not facts and ident:
                rows = engine._db.execute(
                    "SELECT fact_id FROM atomic_facts WHERE memory_id = ? AND profile_id = ? "
                    "ORDER BY created_at LIMIT 20", (ident, pid))
                facts = engine._db.get_facts_by_ids([dict(r)["fact_id"] for r in rows], pid)
            facts = visible_facts(engine._db, pid, facts)
            if not facts:
                return {"success": False, "error": f"no memory {ident!r} in profile {pid!r}",
                        "results": [], "count": 0}
            items = [{"fact_id": f.fact_id, "memory_id": f.memory_id, "content": f.content,
                      "fact_type": f.fact_type.value, "entities": f.canonical_entities,
                      "created_at": f.created_at} for f in facts]
            items, layers = recall_layers.shape(
                items, _depth, _budget,
                context_of=lambda item: recall_layers.context_for(
                    engine._db, pid, str(item.get("fact_id", "")),
                    visible=lambda found: visible_facts(engine._db, pid, found)))
            return {"success": True, "results": items, "count": len(items), "layers": layers}
        except Exception as exc:
            logger.exception("expand failed")
            return {"success": False, "error": type(exc).__name__}


__all__ = ["register_expand_tool"]
