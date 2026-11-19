# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""The actions of the MCP ``core_memory`` tool (pin, unpin, promote, withdraw, list).

Split out of ``mcp/tools_active.py`` (final audit F-15, 19 Nov 2026), which had crossed 800 lines,
and ``core_memory`` itself had grown to 102. Names used from ``tools_active`` (``_get_agent_id``,
``_owns_fact``, ``authorize_mcp_mutation``) are looked up there at call time, so a test or caller
that patches them on ``tools_active`` still reaches this code.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

logger = logging.getLogger("superlocalmemory.mcp.tools_active")


def _ta():
    from superlocalmemory.mcp import tools_active

    return tools_active


def _pin_needs_approval(db, fact_id: str) -> dict | None:
    """The instruction firewall at the pin door (WS-5, D-49, D-53): a memory whose taint is below
    ``agent_inferred`` (web, tool, peer, import, model, unknown) is not pinned by itself. The
    request is recorded and listed under "Waiting for your OK"; one explicit
    ``core_memory(action="promote")`` (any caller) releases it. ``None``: pin as usual."""
    from superlocalmemory.trust import firewall

    verdict = firewall.check_promotion(db, fact_id, firewall.PIN)
    if verdict.allowed:
        return None
    try:
        with db.raw_connection() as conn:
            firewall.request(conn, fact_id, firewall.PIN, by=_ta()._get_agent_id(), channel="mcp")
    except Exception as exc:  # noqa: BLE001 - the refusal matters more than the note
        logger.debug("pin request not recorded (%s)", type(exc).__name__)
    return {"success": False, "needs_approval": True, "fact_id": fact_id,
            "taint": verdict.taint, "error": verdict.reason}


def _promote_or_withdraw(engine, db, action: str, fact_id: str, profile_id: str) -> dict:
    """One explicit promotion (or its withdrawal) by this agent, recorded on channel ``mcp``."""
    from superlocalmemory.trust import firewall

    ta = _ta()
    authorization = ta.authorize_mcp_mutation(
        engine, "update", mutation_source=f"mcp-core-memory-{action}",
        profile_id=profile_id, fact_id=fact_id)
    try:
        with db.raw_connection() as conn:
            done = (firewall.approve if action == "promote" else firewall.withdraw)(
                conn, fact_id, by=f"agent:{ta._get_agent_id()}", channel="mcp")
    except LookupError as exc:
        return {"success": False, "error": str(exc)}
    authorization.complete()
    return {"success": True, "action": action, **done}


def _need_owned(db, pid: str, fact_id: str, action: str) -> dict | None:
    """The refusal for a missing or foreign fact id, or ``None`` when the caller may act on it."""
    if not fact_id:
        return {"success": False, "error": f"fact_id required for {action}"}
    if not _ta()._owns_fact(db, pid, fact_id):
        return {"success": False, "error": f"Memory {fact_id} not found"}
    return None


def _set_pin(engine, db, pid: str, fact_id: str, on: bool) -> dict:
    action = "pin" if on else "unpin"
    authorization = _ta().authorize_mcp_mutation(
        engine, "update", mutation_source=f"mcp-core-memory-{action}",
        profile_id=pid, fact_id=fact_id)
    db.set_pinned(fact_id, on)
    authorization.complete()
    return {"success": True, "action": action, "fact_id": fact_id}


def pin(engine, db, pid: str, fact_id: str, *, action: str, promote: bool) -> dict:
    refused = _need_owned(db, pid, fact_id, action)
    if refused is not None:
        return refused
    held = _pin_needs_approval(db, fact_id)
    if held is None:
        return _set_pin(engine, db, pid, fact_id, True)
    if not promote:
        return held
    promoted = _promote_or_withdraw(engine, db, "promote", fact_id, pid)
    if not promoted.get("success"):
        return promoted
    db.set_pinned(fact_id, True)
    return {"success": True, "action": "pin", "fact_id": fact_id, "promoted": True}


def unpin(engine, db, pid: str, fact_id: str, *, action: str, promote: bool) -> dict:
    refused = _need_owned(db, pid, fact_id, action)
    return refused if refused is not None else _set_pin(engine, db, pid, fact_id, False)


def promote_or_withdraw(engine, db, pid: str, fact_id: str, *, action: str, promote: bool) -> dict:
    refused = _need_owned(db, pid, fact_id, action)
    return refused if refused is not None else _promote_or_withdraw(engine, db, action, fact_id, pid)


def list_pins(engine, db, pid: str, fact_id: str, *, action: str, promote: bool) -> dict:
    from superlocalmemory.mcp.remote_visibility import visible_facts

    pinned = visible_facts(db, pid, db.get_pinned(pid))
    cfg_inj = getattr(getattr(engine, "config", None), "injection", None)
    max_tok = getattr(cfg_inj, "per_memory_max_tokens", 600) if cfg_inj else 600
    return {
        "success": True,
        "pinned": [{"fact_id": f.fact_id, "content": f.content[: max_tok * 4],
                    "importance": getattr(f, "importance", 0.0)} for f in pinned],
        "count": len(pinned),
    }


ACTIONS: dict[str, Callable[..., dict[str, Any]]] = {
    "pin": pin, "unpin": unpin, "promote": promote_or_withdraw,
    "withdraw": promote_or_withdraw, "list": list_pins,
}
