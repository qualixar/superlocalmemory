# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com
"""SuperLocalMemory V3 - Stats Routes
 - AGPL-3.0-or-later

Routes: /api/stats, /api/timeline, /api/patterns
"""
import json
import logging
from collections import defaultdict
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, Request

from .helpers import get_db_connection, dict_factory, get_active_profile, DB_PATH, MEMORY_DIR

logger = logging.getLogger("superlocalmemory.routes.stats")
router = APIRouter()


@router.post("/api/embed")
async def embed_texts(request: Request):
    """Embed texts via the daemon's in-process embedder.

    Used by McpEmbedderProxy so the MCP LIGHT-engine can obtain embeddings
    without spawning a second ONNX worker process.
    """
    engine = getattr(request.app.state, "engine", None)
    if engine is None or engine._embedder is None:
        raise HTTPException(status_code=503, detail="Embedder unavailable")
    body = await request.json()
    texts = body.get("texts", [])
    if not texts:
        raise HTTPException(status_code=400, detail="texts required")
    vectors = engine._embedder.embed_batch(texts)
    return {"vectors": vectors, "dim": engine._embedder.dimension}


@router.get("/api/stats")
async def get_stats():
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
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Stats error: {str(e)}")


@router.get("/api/timeline")
async def get_timeline(
    days: int = Query(30, ge=1, le=365),
    group_by: str = Query("day", pattern="^(day|week|month)$"),
):
    """Get temporal view of memory creation with flexible grouping."""
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

        conn.close()

        return {
            "timeline": timeline, "category_trend": category_trend,
            "period_stats": period_stats,
            "parameters": {"days": days, "group_by": group_by},
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Timeline error: {str(e)}")


@router.get("/api/patterns")
async def get_patterns():
    """Get learned patterns."""
    try:
        conn = get_db_connection()
        conn.row_factory = dict_factory
        cursor = conn.cursor()
        active_profile = get_active_profile()

        # Check for V3 learning tables or V2 identity_patterns
        patterns = []
        table_name = None
        for candidate in ('learned_patterns', 'identity_patterns'):
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                (candidate,),
            )
            if cursor.fetchone():
                table_name = candidate
                break

        if not table_name:
            conn.close()
            # Fall through to V3.1 behavioral pattern store
            try:
                from superlocalmemory.learning.behavioral import BehavioralPatternStore
                store = BehavioralPatternStore(str(MEMORY_DIR / "learning.db"))
                raw = store.get_patterns(profile_id=active_profile)
                # v3.4.7: Map all pattern types to frontend categories
                type_map = {
                    "tech_preference": "preference",
                    "interest": "preference",
                    "entity_preferences": "preference",
                    "style": "style",
                    "fact_type_distribution": "style",
                    "knowledge_structure": "style",
                    "terminology": "terminology",
                    "temporal": "workflow",
                    "session_activity": "workflow",
                    "workflow": "workflow",
                    "co_retrieval_clusters": "workflow",
                    "channel_performance": "performance",
                }
                grouped = defaultdict(list)
                for p in raw:
                    meta = p.get("metadata", {})
                    data = meta  # metadata IS the data dict
                    frontend_key = type_map.get(p.get("pattern_type", ""), "preference")
                    # Extract human-readable value from data fields
                    readable_value = (
                        data.get("value")
                        or data.get("topic")
                        or data.get("pattern_key", "")
                        or p.get("pattern_key", "")
                    )
                    readable_key = (
                        data.get("pattern_key")
                        or data.get("key")
                        or p.get("pattern_key", "")
                    )
                    grouped[frontend_key].append({
                        "pattern_type": p.get("pattern_type", ""),
                        "key": readable_key,
                        "value": readable_value,
                        "confidence": p.get("confidence", 0),
                        "evidence_count": data.get("evidence", p.get("evidence_count", 0)),
                    })
                all_patterns = [p for ps in grouped.values() for p in ps]
                confs = [p["confidence"] for p in all_patterns if p.get("confidence")]
                return {
                    "patterns": dict(grouped),
                    "total_patterns": len(all_patterns),
                    "pattern_types": list(grouped.keys()),
                    "confidence_stats": {
                        "avg": sum(confs) / len(confs) if confs else 0,
                        "min": min(confs) if confs else 0,
                        "max": max(confs) if confs else 0,
                    },
                }
            except Exception:
                return {
                    "patterns": {}, "total_patterns": 0, "pattern_types": [],
                    "message": "Pattern learning not initialized.",
                }

        if table_name == 'identity_patterns':
            cursor.execute("""
                SELECT pattern_type, key, value, confidence, evidence_count,
                       updated_at as last_updated
                FROM identity_patterns WHERE profile = ?
                ORDER BY confidence DESC, evidence_count DESC
            """, (active_profile,))
        else:
            cursor.execute("""
                SELECT pattern_type, key, value, confidence, evidence_count,
                       last_updated
                FROM learned_patterns WHERE is_active = 1
                ORDER BY confidence DESC, evidence_count DESC
            """)

        patterns = cursor.fetchall()

        for pattern in patterns:
            if pattern.get('value'):
                try:
                    pattern['value'] = json.loads(pattern['value'])
                except Exception:
                    pass

        grouped = defaultdict(list)
        for pattern in patterns:
            grouped[pattern['pattern_type']].append(pattern)

        confidences = [p['confidence'] for p in patterns if p.get('confidence')]
        confidence_stats = {
            "avg": sum(confidences) / len(confidences) if confidences else 0,
            "min": min(confidences) if confidences else 0,
            "max": max(confidences) if confidences else 0,
        }

        conn.close()

        return {
            "patterns": dict(grouped), "total_patterns": len(patterns),
            "pattern_types": list(grouped.keys()), "confidence_stats": confidence_stats,
        }

    except Exception as e:
        return {"patterns": {}, "total_patterns": 0, "error": str(e)}
