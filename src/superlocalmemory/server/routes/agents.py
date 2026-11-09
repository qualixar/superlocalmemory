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

from .helpers import DB_PATH, get_read_connection

logger = logging.getLogger("superlocalmemory.routes.agents")
router = APIRouter()


def _internal_error(detail: str = "Internal server error") -> HTTPException:
    """SEC-H-02: log full traceback server-side; return a generic message to the client."""
    logger.exception("agents route error")
    return HTTPException(status_code=500, detail=detail)

# Feature flag: V3 trust scorer
TRUST_AVAILABLE = False
try:
    from superlocalmemory.trust.scorer import TrustScorer  # noqa: F401 (availability probe)
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
def get_agents(
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
def get_agent_stats(request: Request):
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
def get_agent_memory_activity(
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
        conn = get_read_connection(DB_PATH)
        try:
            try:
                # Group by the agent's OWN identity, not by the capability that
                # authorised the write.
                #
                # This grouped by ``trusted_actor_id``, which on a real store is
                # a capability digest — 43 distinct agents rendered as 43 rows of
                # ``daemon-capability:923b7d6e616f46d3...`` and not one readable
                # name. For a pane whose entire purpose is telling agents apart,
                # that is the same as showing nothing.
                #
                # The real name was already being stored the whole time: writers
                # pass ``agent_id`` and it lands in ``raw_metadata_json``. On this
                # store that yields claude-desktop, claude, gemini, codex, grok
                # and mcp_client. Capability digests remain available per row for
                # audit — they answer "what was allowed to write this", which is a
                # different and also useful question, just not this pane's.
                # raw_metadata_json is absent on stores predating it. Try the
                # identity query and fall back to capability grouping if the
                # column is missing — without this, the OperationalError is
                # caught below and an older install shows ZERO agents while
                # having plenty. Deliberately not PRAGMA table_info: this
                # handler is on the dashboard read path, which is gated against
                # anything that parses as DDL.
                _tail = (
                    "COUNT(*) AS cnt, MAX(created_at) AS last_active, "
                    "GROUP_CONCAT(DISTINCT source_type) AS sources, "
                    "COUNT(DISTINCT NULLIF(trusted_actor_id, '')) AS capabilities, "
                    # 4.1.25 L-15/G-M2: how many of these the connection vouched for.
                    "SUM(CASE WHEN json_extract(raw_metadata_json, '$.agent_id_source')"
                    " = 'connection' THEN 1 ELSE 0 END) AS verified "
                    "FROM ingestion_operations WHERE profile_id=? "
                    "GROUP BY agent_id ORDER BY cnt DESC, agent_id ASC "
                    "LIMIT 500"
                )
                try:
                    rows = conn.execute(
                        "SELECT COALESCE("
                        "  NULLIF(json_extract(raw_metadata_json, '$.agent_id'), ''),"
                        "  NULLIF(trusted_actor_id, ''),"
                        "  'unknown'"
                        ") AS agent_id, " + _tail,
                        (pid,),
                    ).fetchall()
                except sqlite3.OperationalError:
                    rows = conn.execute(
                        "SELECT COALESCE(NULLIF(trusted_actor_id, ''), 'unknown')"
                        " AS agent_id, " + _tail.replace(
                            "SUM(CASE WHEN json_extract(raw_metadata_json, '$.agent_id_source')"
                            " = 'connection' THEN 1 ELSE 0 END)", "0"),
                        (pid,),
                    ).fetchall()
                for r in rows:
                    name = r["agent_id"]
                    agents.append({
                        "agent_id": name,
                        "count": r["cnt"],
                        "last_active": r["last_active"],
                        "source_types": (
                            [s for s in (r["sources"] or "").split(",") if s]
                        ),
                        # How many distinct capabilities this agent wrote under.
                        "capability_count": r["capabilities"],
                        # True when we fell back to a digest — the UI can then say
                        # "this writer did not identify itself" instead of
                        # presenting a hash as though it were a name.
                        "identified": not str(name).startswith(
                            ("daemon-capability:", "local-capability:")
                        ) and name != "unknown",
                        # Saved through a connection that named this agent, as
                        # against a label the caller only claimed.
                        "verified": int(r["verified"] or 0),
                        "claimed": int(r["cnt"]) - int(r["verified"] or 0),
                    })
                    total += r["cnt"]
            except sqlite3.OperationalError:
                pass

            try:
                recent = _recent_rows(conn, pid, limit)
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


def _recent_rows(conn, pid: str, limit: int) -> list[dict]:
    """Recent saves, named the same way as the cards above them (G-M2).

    The table used to name every row by its capability digest while the cards
    named the agent, so one screen disagreed with itself. Both now use the
    agent name, say whether the connection vouched for it, and keep a short
    fingerprint of the authenticated writer.
    """
    columns = {row[1] for row in conn.execute("PRAGMA table_info(ingestion_operations)")}
    if "raw_metadata_json" in columns:
        agent_name = "NULLIF(json_extract(raw_metadata_json, '$.agent_id'), ''), "
        agent_source = "json_extract(raw_metadata_json, '$.agent_id_source')"
    else:
        # Legacy ingestion tables have no connection-attribution metadata.
        agent_name = ""
        agent_source = "NULL"
    rows = conn.execute(
        "SELECT COALESCE(" + agent_name + "NULLIF(trusted_actor_id, ''), 'unknown') AS agent_id, "
        + agent_source + " AS agent_id_source, "
        "trusted_actor_id, substr(raw_content, 1, 160) AS snippet, "
        "created_at, source_type, session_id "
        "FROM ingestion_operations WHERE profile_id=? "
        "ORDER BY created_at DESC, rowid DESC LIMIT ?",
        (pid, int(limit)),
    ).fetchall()
    return [{
        "agent_id": r["agent_id"],
        "verified": r["agent_id_source"] == "connection",
        "writer": _fingerprint(r["trusted_actor_id"]),
        "content": r["snippet"],
        "created_at": r["created_at"],
        "source_type": r["source_type"],
        "session_id": r["session_id"],
    } for r in rows]


def _fingerprint(actor: str | None) -> str:
    """``kind:abcdef`` for a capability actor; the readable part otherwise."""
    if not actor:
        return ""
    parts = str(actor).split(":")
    for part in parts[1:]:
        if len(part) >= 16 and all(c in "0123456789abcdef" for c in part.lower()):
            return f"{parts[0]}:{part[:6]}"
    return str(actor)[:40]


@router.get("/api/trust/stats")
def get_trust_stats(request: Request):
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
            conn = get_read_connection(DB_PATH)
            try:
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
            finally:
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
def get_agent_trust_signals(
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
