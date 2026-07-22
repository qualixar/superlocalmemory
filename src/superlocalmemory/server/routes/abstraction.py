# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com

"""Progressive-abstraction read API (Wave Q3).

Exposes the abstraction hierarchy so the dashboard can browse it and drill
down to source atoms:

  GET /api/v3/abstraction/persona      — the per-profile persona roll-up
  GET /api/v3/abstraction/communities  — community summaries (Q2)
  GET /api/v3/abstraction/sources      — drill-down (node -> source atoms)

Read-only, profile-scoped (Rule 01), direct sqlite3 (Rule 06). All handlers
fail-soft: a missing DB or table returns an empty payload, never a 500.
"""

from __future__ import annotations

import logging
import sqlite3
from typing import Any

from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse

from superlocalmemory.server.routes.helpers import DB_PATH, get_active_profile

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v3/abstraction", tags=["abstraction"])


class _ReadDB:
    """Adapt a raw sqlite3 connection to the .execute(...) -> list contract
    the read-only builder methods expect (matches DatabaseManager.execute)."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def execute(self, sql: str, params: tuple = ()) -> list:
        return self._conn.execute(sql, params).fetchall()


def _conn() -> sqlite3.Connection | None:
    if not DB_PATH.exists():
        return None
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


@router.get("/persona")
def get_persona(profile: str = Query("")) -> JSONResponse:
    pid = profile or get_active_profile()
    conn = _conn()
    if conn is None:
        return JSONResponse({"profile": pid, "persona": None})
    try:
        from superlocalmemory.core.progressive_abstraction import ProgressiveAbstraction

        persona = ProgressiveAbstraction(_ReadDB(conn)).get_persona(pid)
        return JSONResponse({"profile": pid, "persona": persona})
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("persona read failed: %s", exc)
        return JSONResponse({"profile": pid, "persona": None})
    finally:
        conn.close()


@router.get("/communities")
def get_communities(profile: str = Query("")) -> JSONResponse:
    pid = profile or get_active_profile()
    conn = _conn()
    if conn is None:
        return JSONResponse({"profile": pid, "communities": []})
    try:
        from superlocalmemory.core.community_summary import CommunitySummaryBuilder

        summaries = CommunitySummaryBuilder(_ReadDB(conn)).get_summaries(pid)
        return JSONResponse({"profile": pid, "communities": summaries})
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("communities read failed: %s", exc)
        return JSONResponse({"profile": pid, "communities": []})
    finally:
        conn.close()


@router.get("/sources")
def get_sources(
    profile: str = Query(""), node: str = Query("persona"),
) -> JSONResponse:
    pid = profile or get_active_profile()
    conn = _conn()
    empty: dict[str, Any] = {
        "node_id": node, "node_type": "unknown", "communities": [], "fact_ids": [],
    }
    if conn is None:
        return JSONResponse({"profile": pid, "sources": empty})
    try:
        from superlocalmemory.core.progressive_abstraction import ProgressiveAbstraction

        node_val: Any = node
        if node != "persona":
            try:
                node_val = int(node)
            except (ValueError, TypeError):
                node_val = node
        sources = ProgressiveAbstraction(_ReadDB(conn)).get_sources(pid, node_val)
        return JSONResponse({"profile": pid, "sources": sources})
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("sources read failed: %s", exc)
        return JSONResponse({"profile": pid, "sources": empty})
    finally:
        conn.close()
