# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com
"""SuperLocalMemory V3 - Stats Routes
 - AGPL-3.0-or-later

Routes: /api/stats, /api/timeline
"""
import logging
import sqlite3
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from .helpers import get_db_connection, dict_factory, get_active_profile, DB_PATH

logger = logging.getLogger("superlocalmemory.routes.stats")
router = APIRouter()


def _query_ingestion_sources(cursor, profile_id: str) -> list[dict]:
    """Query profile-scoped ingestion provenance without double-counting facts."""
    cursor.execute(
        """
        SELECT COALESCE(NULLIF(TRIM(source_type), ''), 'unknown') AS source_type,
               COUNT(DISTINCT fact_id) AS count
        FROM provenance
        WHERE profile_id = ?
        GROUP BY COALESCE(NULLIF(TRIM(source_type), ''), 'unknown')
        ORDER BY count DESC, source_type ASC
        """,
        (profile_id,),
    )
    return [dict(row) for row in cursor.fetchall()]


def _load_ingestion_sources_with_status(
    cursor, profile_id: str,
) -> tuple[list[dict], dict]:
    """Return provenance plus a truthful availability state.

    Older databases may not have the provenance table.  In that case the
    dashboard must report an empty/unknown state rather than inventing a split.
    Lock or I/O failures are distinct from a legitimate legacy-empty state.
    """
    try:
        return _query_ingestion_sources(cursor, profile_id), {
            "available": True,
            "state": "recorded",
            "source": "memory.db:provenance",
        }
    except sqlite3.OperationalError as exc:
        if "no such table" in str(exc).lower():
            return [], {
                "available": True,
                "state": "legacy_not_recorded",
                "source": "memory.db:provenance",
            }
        logger.warning("provenance metrics temporarily unavailable: %s", exc)
        return [], {
            "available": False,
            "state": "temporarily_unavailable",
            "source": "memory.db:provenance",
        }
    except sqlite3.Error as exc:
        logger.warning("provenance metrics unavailable: %s", exc)
        return [], {
            "available": False,
            "state": "temporarily_unavailable",
            "source": "memory.db:provenance",
        }


def _load_ingestion_sources(cursor, profile_id: str) -> list[dict]:
    """Compatibility helper returning only the provenance rows."""
    return _load_ingestion_sources_with_status(cursor, profile_id)[0]


def _internal_error(detail: str = "Internal server error") -> HTTPException:
    """SEC-H-02: log full traceback server-side; return a generic message to the client."""
    logger.exception("stats route error")
    return HTTPException(status_code=500, detail=detail)


@router.get("/api/stats")
def get_stats():
    """Get comprehensive system statistics."""
    try:
        conn = get_db_connection()
        conn.row_factory = dict_factory
        cursor = conn.cursor()
        active_profile = get_active_profile()

        # Detect V3 schema
        try:
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='atomic_facts'")
            use_v3 = cursor.fetchone() is not None
        except Exception:
            use_v3 = False

        if use_v3:
            cursor.execute(
                "SELECT COUNT(*) as total FROM atomic_facts WHERE profile_id = ?",
                (active_profile,),
            )
            total_facts = cursor.fetchone()['total']

            cursor.execute(
                "SELECT COUNT(*) as total FROM memories WHERE profile_id = ?",
                (active_profile,),
            )
            total_memories = cursor.fetchone()['total']

            total_sessions = 0
            try:
                cursor.execute(
                    "SELECT COUNT(DISTINCT session_id) as total FROM atomic_facts WHERE profile_id = ?",
                    (active_profile,),
                )
                total_sessions = cursor.fetchone()['total']
            except Exception:
                pass

            total_graph_nodes = total_facts
            total_graph_edges = 0
            try:
                cursor.execute(
                    "SELECT COUNT(*) as total FROM graph_edges WHERE profile_id = ?",
                    (active_profile,),
                )
                total_graph_edges = cursor.fetchone()['total']
            except Exception:
                pass

            total_clusters = 0
            # v3.4.1: Use community_id from fact_importance (graph intelligence)
            try:
                cursor.execute(
                    "SELECT COUNT(DISTINCT community_id) as total FROM fact_importance "
                    "WHERE profile_id = ? AND community_id IS NOT NULL",
                    (active_profile,),
                )
                total_clusters = cursor.fetchone()['total']
            except Exception:
                pass
            # Fallback: V2 scenes table
            if total_clusters == 0:
                try:
                    cursor.execute(
                        "SELECT COUNT(DISTINCT scene_id) as total FROM scenes WHERE profile_id = ?",
                        (active_profile,),
                    )
                    total_clusters = cursor.fetchone()['total']
                except Exception:
                    pass
            # Fallback: V2-migrated clusters stored as cluster_id on memories
            if total_clusters == 0:
                try:
                    cursor.execute(
                        "SELECT COUNT(DISTINCT cluster_id) as total FROM memories "
                        "WHERE cluster_id IS NOT NULL AND profile = ?",
                        (active_profile,),
                    )
                    total_clusters = cursor.fetchone()['total']
                except Exception:
                    pass

            # Fact type breakdown (replaces category in V3)
            cursor.execute("""
                SELECT fact_type as category, COUNT(*) as count
                FROM atomic_facts WHERE profile_id = ?
                GROUP BY fact_type ORDER BY count DESC LIMIT 10
            """, (active_profile,))
            categories = cursor.fetchall()

            # Session breakdown (replaces project in V3)
            cursor.execute("""
                SELECT session_id as project_name, COUNT(*) as count
                FROM atomic_facts WHERE profile_id = ? AND session_id IS NOT NULL
                GROUP BY session_id ORDER BY count DESC LIMIT 10
            """, (active_profile,))
            projects = cursor.fetchall()

            cursor.execute("""
                SELECT COUNT(*) as count FROM atomic_facts
                WHERE created_at >= datetime('now', '-7 days') AND profile_id = ?
            """, (active_profile,))
            recent_memories = cursor.fetchone()['count']

            importance_dist = []

        else:
            # V2 fallback — no atomic_facts; facts == memories
            cursor.execute(
                "SELECT COUNT(*) as total FROM memories WHERE profile = ?",
                (active_profile,),
            )
            total_memories = cursor.fetchone()['total']
            total_facts = total_memories

            try:
                cursor.execute("SELECT COUNT(*) as total FROM sessions")
                total_sessions = cursor.fetchone()['total']
            except Exception:
                total_sessions = 0

            cursor.execute(
                "SELECT COUNT(DISTINCT cluster_id) as total FROM memories "
                "WHERE cluster_id IS NOT NULL AND profile = ?",
                (active_profile,),
            )
            total_clusters = cursor.fetchone()['total']

            try:
                cursor.execute(
                    "SELECT COUNT(*) as total FROM graph_nodes gn "
                    "JOIN memories m ON gn.memory_id = m.id WHERE m.profile = ?",
                    (active_profile,),
                )
                total_graph_nodes = cursor.fetchone()['total']
            except Exception:
                total_graph_nodes = 0

            try:
                cursor.execute(
                    "SELECT COUNT(*) as total FROM graph_edges ge "
                    "JOIN memories m ON ge.source_memory_id = m.id WHERE m.profile = ?",
                    (active_profile,),
                )
                total_graph_edges = cursor.fetchone()['total']
            except Exception:
                total_graph_edges = 0

            cursor.execute(
                "SELECT category, COUNT(*) as count FROM memories "
                "WHERE category IS NOT NULL AND profile = ? "
                "GROUP BY category ORDER BY count DESC LIMIT 10",
                (active_profile,),
            )
            categories = cursor.fetchall()

            cursor.execute(
                "SELECT project_name, COUNT(*) as count FROM memories "
                "WHERE project_name IS NOT NULL AND profile = ? "
                "GROUP BY project_name ORDER BY count DESC LIMIT 10",
                (active_profile,),
            )
            projects = cursor.fetchall()

            cursor.execute(
                "SELECT COUNT(*) as count FROM memories "
                "WHERE created_at >= datetime('now', '-7 days') AND profile = ?",
                (active_profile,),
            )
            recent_memories = cursor.fetchone()['count']

            cursor.execute(
                "SELECT importance, COUNT(*) as count FROM memories "
                "WHERE profile = ? GROUP BY importance ORDER BY importance DESC",
                (active_profile,),
            )
            importance_dist = cursor.fetchall()

        db_size = DB_PATH.stat().st_size if DB_PATH.exists() else 0
        ingestion_sources, ingestion_sources_status = (
            _load_ingestion_sources_with_status(cursor, active_profile)
        )

        if total_graph_nodes > 1:
            max_edges = (total_graph_nodes * (total_graph_nodes - 1)) / 2
            density = total_graph_edges / max_edges if max_edges > 0 else 0
        else:
            density = 0

        conn.close()

        facts_per_memory = (
            round(total_facts / total_memories, 1)
            if total_memories > 0 else 0.0
        )

        return {
            "overview": {
                "total_memories": total_memories,
                "total_facts": total_facts,
                "facts_per_memory": facts_per_memory,
                "total_sessions": total_sessions,
                "total_clusters": total_clusters,
                "graph_nodes": total_graph_nodes,
                "graph_edges": total_graph_edges,
                "db_size_mb": round(db_size / (1024 * 1024), 2),
                "recent_memories_7d": recent_memories,
            },
            "categories": categories,
            "projects": projects,
            "importance_distribution": importance_dist,
            "graph_stats": {
                "density": round(density, 4),
                "avg_degree": (
                    round(2 * total_graph_edges / total_graph_nodes, 2)
                    if total_graph_nodes > 0 else 0
                ),
            },
            "ingestion_sources": ingestion_sources,
            "ingestion_sources_status": ingestion_sources_status,
        }

    except Exception:
        raise _internal_error("Stats error")


@router.get("/api/timeline")
def get_timeline(
    days: int = Query(30, ge=1, le=365),
    group_by: str = Query("day", pattern="^(day|week|month)$"),
    include_categories: bool = Query(True),
):
    """Get temporal memory creation; dashboard callers may skip category scans."""
    try:
        conn = get_db_connection()
        conn.row_factory = dict_factory
        cursor = conn.cursor()
        active_profile = get_active_profile()

        if group_by == "day":
            date_group = "DATE(created_at)"
        elif group_by == "week":
            date_group = "strftime('%Y-W%W', created_at)"
        else:
            date_group = "strftime('%Y-%m', created_at)"

        # Try V3 first
        try:
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='atomic_facts'")
            use_v3 = cursor.fetchone() is not None
        except Exception:
            use_v3 = False

        table = "atomic_facts" if use_v3 else "memories"
        profile_col = "profile_id" if use_v3 else "profile"
        cat_col = "fact_type" if use_v3 else "category"

        cursor.execute(f"""
            SELECT {date_group} as period, COUNT(*) as count,
                   GROUP_CONCAT(DISTINCT {cat_col}) as categories
            FROM {table}
            WHERE created_at >= datetime('now', '-' || ? || ' days') AND {profile_col} = ?
            GROUP BY {date_group} ORDER BY period DESC
        """, (days, active_profile))
        timeline = cursor.fetchall()

        if include_categories:
            cursor.execute(f"""
                SELECT {date_group} as period, {cat_col} as category, COUNT(*) as count
                FROM {table}
                WHERE created_at >= datetime('now', '-' || ? || ' days')
                  AND {cat_col} IS NOT NULL AND {profile_col} = ?
                GROUP BY {date_group}, {cat_col} ORDER BY period DESC, count DESC
            """, (days, active_profile))
            category_trend = cursor.fetchall()

            cursor.execute(f"""
                SELECT COUNT(*) as total_memories,
                       COUNT(DISTINCT {cat_col}) as categories_used
                FROM {table}
                WHERE created_at >= datetime('now', '-' || ? || ' days') AND {profile_col} = ?
            """, (days, active_profile))
            period_stats = cursor.fetchone()
        else:
            category_trend = []
            period_stats = {
                "total_memories": sum(int(row["count"] or 0) for row in timeline),
                "categories_used": None,
            }

        conn.close()

        return {
            "timeline": timeline, "category_trend": category_trend,
            "period_stats": period_stats,
            "parameters": {"days": days, "group_by": group_by},
        }

    except Exception:
        raise _internal_error("Timeline error")
