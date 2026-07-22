# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com
"""SuperLocalMemory V3 - Agent Registry + Trust Routes
 - AGPL-3.0-or-later

Routes: /api/agents, /api/agents/stats, /api/trust/stats, /api/trust/signals/{agent_id}
Uses V3 TrustScorer and core.registry.
"""
import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, Request

from superlocalmemory.infra.data_root import state_path

from .helpers import DB_PATH

logger = logging.getLogger("superlocalmemory.routes.agents")
router = APIRouter()


def _internal_error(detail: str = "Internal server error") -> HTTPException:
    """SEC-H-02: log full traceback server-side; return a generic message to the client."""
    logger.exception("agents route error")
    return HTTPException(status_code=500, detail=detail)

# Feature flag: V3 trust scorer
TRUST_AVAILABLE = False
try:
    from superlocalmemory.trust.scorer import TrustScorer
    TRUST_AVAILABLE = True
except ImportError:
    pass

REGISTRY_AVAILABLE = False
try:
    from superlocalmemory.core.registry import AgentRegistry
    REGISTRY_AVAILABLE = True
except ImportError:
    pass


def _registry_path():
    return state_path("agents.json")


@router.get("/api/agents")
async def get_agents(
    request: Request,
    protocol: Optional[str] = None,
    limit: int = Query(50, ge=1, le=200),
):
    """List registered agents with optional protocol filter."""
    if not REGISTRY_AVAILABLE:
        return {"agents": [], "count": 0, "message": "Agent registry not available"}
    try:
        registry = AgentRegistry(persist_path=_registry_path())
        agents = registry.list_agents()
        return {
            "agents": agents,
            "count": len(agents),
            "stats": {"total_agents": len(agents)},
        }
    except Exception:
        raise _internal_error("Agent registry error")


@router.get("/api/agents/stats")
async def get_agent_stats(request: Request):
    """Get agent registry statistics."""
    if not REGISTRY_AVAILABLE:
        return {"total_agents": 0, "message": "Agent registry not available"}
    try:
        registry = AgentRegistry(persist_path=_registry_path())
        agents = registry.list_agents()
        return {"total_agents": len(agents)}
    except Exception:
        raise _internal_error("Agent stats error")


@router.get("/api/agents/memory-activity")
async def get_agent_memory_activity(
    request: Request,
    limit: int = Query(20, ge=1, le=100),
):
    """Per-agent memory attribution for the multi-agent memory view.

    Reports how many memories each writing agent contributed, when each was
    last active, which ingestion sources they used, and the most recent
    entries — grouped by ``ingestion_operations.trusted_actor_id`` (the agent
    that wrote the memory). Profile-scoped. Uses a direct DB read because the
    dashboard runs without the engine subprocess. Never raises to the client;
    returns empty structures if the operations table is absent.
    """
    import sqlite3

    from .helpers import get_active_profile

    pid = get_active_profile()
    agents: list[dict] = []
    recent: list[dict] = []
    total = 0

    if DB_PATH.exists():
        conn = sqlite3.connect(str(DB_PATH))
        conn.row_factory = sqlite3.Row
        try:
            try:
                rows = conn.execute(
                    "SELECT CASE WHEN trusted_actor_id='' THEN 'unknown' "
                    "ELSE trusted_actor_id END AS agent_id, "
                    "COUNT(*) AS cnt, MAX(created_at) AS last_active, "
                    "GROUP_CONCAT(DISTINCT source_type) AS sources "
                    "FROM ingestion_operations WHERE profile_id=? "
                    "GROUP BY agent_id ORDER BY cnt DESC, agent_id ASC",
                    (pid,),
                ).fetchall()
                for r in rows:
                    agents.append({
                        "agent_id": r["agent_id"],
                        "count": r["cnt"],
                        "last_active": r["last_active"],
                        "source_types": (
                            [s for s in (r["sources"] or "").split(",") if s]
                        ),
                    })
                    total += r["cnt"]
            except sqlite3.OperationalError:
                pass

            try:
                rows = conn.execute(
                    "SELECT CASE WHEN trusted_actor_id='' THEN 'unknown' "
                    "ELSE trusted_actor_id END AS agent_id, "
                    "substr(raw_content, 1, 160) AS snippet, "
                    "created_at, source_type, session_id "
                    "FROM ingestion_operations WHERE profile_id=? "
                    "ORDER BY created_at DESC, rowid DESC LIMIT ?",
                    (pid, int(limit)),
                ).fetchall()
                recent = [{
                    "agent_id": r["agent_id"],
                    "content": r["snippet"],
                    "created_at": r["created_at"],
                    "source_type": r["source_type"],
                    "session_id": r["session_id"],
                } for r in rows]
            except sqlite3.OperationalError:
                pass
        finally:
            conn.close()

    return {
        "ok": True,
        "profile_id": pid,
        "total_memories": total,
        "agent_count": len(agents),
        "agents": agents,
        "recent": recent,
    }


@router.get("/api/trust/stats")
async def get_trust_stats(request: Request):
    """Get trust scoring statistics.

    Queries trust_scores and trust_signals tables directly (no engine needed).
    Falls back to engine._trust_scorer if available.
    """
    try:
        # Try engine-based scorer first
        try:
            engine = getattr(request.app.state, "engine", None)
            if engine and getattr(engine, "_trust_scorer", None):
                return engine._trust_scorer.get_trust_stats()
        except (AttributeError, Exception):
            pass  # Fall through to direct DB query

        # Direct DB query (dashboard runs without engine subprocess)
        import sqlite3
        from .helpers import get_active_profile
        pid = get_active_profile()

        total_signals = 0
        avg_trust_score = 0.667
        by_signal_type = {}

        if DB_PATH.exists():
            conn = sqlite3.connect(str(DB_PATH))
            conn.row_factory = sqlite3.Row
            try:
                # Count trust signals
                row = conn.execute(
                    "SELECT COUNT(*) AS cnt FROM trust_signals "
                    "WHERE profile_id = ?", (pid,),
                ).fetchone()
                total_signals = row["cnt"] if row else 0
            except sqlite3.OperationalError:
                pass

            try:
                # Average trust score
                row = conn.execute(
                    "SELECT AVG(trust_score) AS avg_ts FROM trust_scores "
                    "WHERE profile_id = ?", (pid,),
                ).fetchone()
                if row and row["avg_ts"] is not None:
                    avg_trust_score = round(float(row["avg_ts"]), 3)
            except sqlite3.OperationalError:
                pass

            try:
                # Signal breakdown by type
                rows = conn.execute(
                    "SELECT signal_type, COUNT(*) AS cnt "
                    "FROM trust_signals WHERE profile_id = ? "
                    "GROUP BY signal_type", (pid,),
                ).fetchall()
                by_signal_type = {r["signal_type"]: r["cnt"] for r in rows}
            except sqlite3.OperationalError:
                pass

            conn.close()

        # Enforcement status: SLM uses "Silent Collection" by default
        enforcement = "Silent Collection"

        return {
            "total_signals": total_signals,
            "avg_trust_score": avg_trust_score,
            "enforcement": enforcement,
            "by_signal_type": by_signal_type,
        }
    except Exception:
        raise _internal_error("Trust stats error")


@router.get("/api/trust/signals/{agent_id}")
async def get_agent_trust_signals(
    request: Request, agent_id: str,
    limit: int = Query(50, ge=1, le=200),
):
    """Get trust signal history for a specific agent."""
    if not TRUST_AVAILABLE:
        return {"signals": [], "count": 0}
    try:
        engine = getattr(request.app.state, "engine", None)
        if engine and engine._trust_scorer:
            scorer = engine._trust_scorer
            signals = scorer.get_signals(agent_id, limit=limit)
            score = scorer.get_trust_score(agent_id)
            return {
                "agent_id": agent_id, "trust_score": score,
                "signals": signals, "count": len(signals),
            }
        return {"agent_id": agent_id, "signals": [], "count": 0}
    except Exception:
        raise _internal_error("Trust signals error")
