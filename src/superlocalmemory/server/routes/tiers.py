# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com
"""SuperLocalMemory V3.4.11 "Scale-Ready" - Tier Management Routes

Routes: /api/tiers/stats, /api/tiers/evaluate, /api/tiers/pin, /api/tiers/unpin

Uses lightweight sqlite3 directly (not MemoryEngine) for fast dashboard queries.
All connections use WAL mode + busy_timeout for concurrency safety.
"""

import logging
import re
import sqlite3
from contextlib import contextmanager
from datetime import datetime, UTC

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from superlocalmemory.server.route_mutations import authorize_route_mutation

from .helpers import DB_PATH, get_active_profile

logger = logging.getLogger("superlocalmemory.routes.tiers")
router = APIRouter()

_PROFILE_PATTERN = re.compile(r'^[a-zA-Z0-9_-]+$')
_MAX_REASON_LENGTH = 500


class PinRequest(BaseModel):
    fact_id: str = Field(..., min_length=1)
    reason: str = Field(default="", max_length=_MAX_REASON_LENGTH)


@contextmanager
def _db():
    """Context-managed DB connection with WAL + busy_timeout."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def _validate_profile(profile_id: str) -> str:
    """Validate profile_id against allowed pattern."""
    if not profile_id or not _PROFILE_PATTERN.match(profile_id):
        raise HTTPException(status_code=400, detail="Invalid profile_id")
    return profile_id


@router.get("/api/tiers/stats")
async def tier_stats(profile_id: str | None = None):
    """Get tier distribution stats."""
    # Default to the ACTIVE profile, never the literal "default" — otherwise
    # tier counts reflect the default profile regardless of which is active.
    profile_id = _validate_profile(profile_id or get_active_profile())

    with _db() as conn:
        try:
            c = conn.cursor()
            c.execute(
                "SELECT lifecycle, COUNT(*) as cnt FROM atomic_facts "
                "WHERE profile_id = ? GROUP BY lifecycle", (profile_id,),
            )
            dist = {row["lifecycle"]: row["cnt"] for row in c.fetchall()}

            pinned = 0
            try:
                c.execute(
                    "SELECT COUNT(*) as c FROM pinned_facts "
                    "WHERE profile_id = ?", (profile_id,),
                )
                pinned = c.fetchone()["c"]
            except sqlite3.OperationalError:
                pass  # pinned_facts table may not exist yet

            total = sum(dist.values())
            return {
                "active": dist.get("active", 0),
                "warm": dist.get("warm", 0),
                "cold": dist.get("cold", 0),
                "archived": dist.get("archived", 0),
                "total": total,
                "pinned": pinned,
                "active_pct": round(
                    dist.get("active", 0) / max(total, 1) * 100, 1,
                ),
            }
        except Exception as exc:
            logger.error("tier_stats failed: %s", exc, exc_info=True)
            raise HTTPException(
                status_code=500, detail="Internal storage error",
            ) from None


@router.post("/api/tiers/evaluate")
async def evaluate_tiers_route(request: Request, profile_id: str | None = None):
    """Manually trigger tier evaluation.

    Uses the shared engine (via lazy import) instead of re-initializing
    DatabaseManager on every request.
    """
    profile_id = _validate_profile(profile_id or get_active_profile())

    try:
        authorization = authorize_route_mutation(
            request,
            operation="update",
            source_agent_id="http-tier-evaluate",
            profile_id=profile_id,
        )
        from superlocalmemory.core.tier_manager import evaluate_tiers
        from .helpers import get_engine_lazy

        engine = get_engine_lazy(request.app.state)
        if engine is None or not hasattr(engine, '_db') or engine._db is None:
            raise HTTPException(
                status_code=503, detail="Engine not initialized",
            )
        stats = evaluate_tiers(engine._db, profile_id)
        authorization.complete()
        return {"success": True, "stats": stats}
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("evaluate_tiers failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=500, detail="Tier evaluation failed",
        ) from None


@router.post("/api/tiers/pin")
async def pin_fact_route(
    body: PinRequest,
    request: Request,
    profile_id: str | None = None,
):
    """Pin a fact to stay in active tier forever.

    Validates fact exists in the specified profile before pinning.
    """
    profile_id = _validate_profile(profile_id or get_active_profile())
    authorization = authorize_route_mutation(
        request,
        operation="update",
        source_agent_id="http-tier-pin",
        profile_id=profile_id,
        fact_id=body.fact_id,
    )

    with _db() as conn:
        try:
            # Verify fact exists in this profile
            c = conn.cursor()
            c.execute(
                "SELECT fact_id FROM atomic_facts "
                "WHERE fact_id = ? AND profile_id = ?",
                (body.fact_id, profile_id),
            )
            if c.fetchone() is None:
                raise HTTPException(
                    status_code=404,
                    detail=f"Fact {body.fact_id[:8]}... not found",
                )

            now = datetime.now(UTC).isoformat()
            conn.execute(
                "INSERT OR REPLACE INTO pinned_facts "
                "(fact_id, profile_id, pinned_at, reason) "
                "VALUES (?, ?, ?, ?)",
                (body.fact_id, profile_id, now, body.reason),
            )
            from superlocalmemory.core.lifecycle_state import set_fact_lifecycle_zone
            set_fact_lifecycle_zone(
                conn, [body.fact_id], "active", profile_id=profile_id,
            )
            conn.commit()
            authorization.complete()
            return {"success": True, "message": f"Fact {body.fact_id[:8]}... pinned"}
        except HTTPException:
            raise
        except Exception as exc:
            logger.error("pin_fact failed: %s", exc, exc_info=True)
            raise HTTPException(
                status_code=500, detail="Failed to pin fact",
            ) from None


@router.post("/api/tiers/unpin")
async def unpin_fact_route(
    body: PinRequest,
    request: Request,
    profile_id: str | None = None,
):
    """Unpin a fact, allowing normal tier demotion.

    Lifecycle stays 'active' until the next tier evaluation cycle demotes it
    based on access patterns. This is intentional — immediate demotion would
    surprise the user.
    """
    profile_id = _validate_profile(profile_id or get_active_profile())
    authorization = authorize_route_mutation(
        request,
        operation="update",
        source_agent_id="http-tier-unpin",
        profile_id=profile_id,
        fact_id=body.fact_id,
    )

    with _db() as conn:
        try:
            conn.execute(
                "DELETE FROM pinned_facts WHERE fact_id = ? AND profile_id = ?",
                (body.fact_id, profile_id),
            )
            conn.commit()
            authorization.complete()
            return {"success": True, "unpinned": True}
        except Exception as exc:
            logger.error("unpin_fact failed: %s", exc, exc_info=True)
            raise HTTPException(
                status_code=500, detail="Failed to unpin fact",
            ) from None
