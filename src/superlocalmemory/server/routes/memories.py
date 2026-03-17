# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under the MIT License - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com
"""SuperLocalMemory V3 - Memory Routes (MIT License).
Routes: /api/memories, /api/graph, /api/search, /api/clusters, /api/clusters/{id}
Uses V3 MemoryEngine for store/recall. Falls back to direct DB for list/graph.
"""
import json
import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, Request

from .helpers import (
    get_db_connection, dict_factory, get_active_profile, get_engine_lazy,
    SearchRequest, DB_PATH, MEMORY_DIR,
)

logger = logging.getLogger("superlocalmemory.routes.memories")
router = APIRouter()


def _get_engine(request: Request):
    """Get V3 engine from app state, initializing lazily on first call."""
    return get_engine_lazy(request.app.state)


def _preview(content: str | None) -> str:
    """Truncate content for preview display."""
    if not content:
        return ""
    return content[:100] + "..." if len(content) > 100 else content


def _has_table(cursor, name: str) -> bool:
    """Check if a table exists in the database."""
    try:
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (name,))
        return cursor.fetchone() is not None
    except Exception:
        return False


def _fetch_graph_data(
    cursor, profile: str, use_v3: bool, min_importance: int, max_nodes: int,
) -> tuple[list, list, list]:
    """Fetch graph nodes, links, clusters from V3 or V2 schema."""
    if use_v3:
        # Graph-first: fetch edges, then get connected nodes, then fill slots
        cursor.execute("""
            SELECT source_id as source, target_id as target,
                   weight, edge_type as relationship_type
            FROM graph_edges WHERE profile_id = ?
            ORDER BY weight DESC
        """, (profile,))
        all_links = cursor.fetchall()

        connected_ids = set()
        for lk in all_links:
            connected_ids.add(lk['source'])
            connected_ids.add(lk['target'])

        # Fetch connected nodes first (these have edges to display)
        connected_nodes: list = []
        if connected_ids:
            ph = ','.join('?' * len(connected_ids))
            cursor.execute(f"""
                SELECT fact_id as id, content, fact_type as category,
                       confidence as importance, session_id as project_name,
                       created_at
                FROM atomic_facts
                WHERE profile_id = ? AND fact_id IN ({ph})
            """, [profile] + list(connected_ids))
            connected_nodes = cursor.fetchall()

        # Fill remaining slots with top-confidence unconnected nodes
        remaining = max_nodes - len(connected_nodes)
        if remaining > 0:
            existing = {n['id'] for n in connected_nodes}
            cursor.execute("""
                SELECT fact_id as id, content, fact_type as category,
                       confidence as importance, session_id as project_name,
                       created_at
                FROM atomic_facts
                WHERE profile_id = ? AND confidence >= ?
                ORDER BY confidence DESC, created_at DESC
                LIMIT ?
            """, (profile, min_importance / 10.0, remaining + len(existing)))
            for n in cursor.fetchall():
                if n['id'] not in existing:
                    connected_nodes.append(n)
                    if len(connected_nodes) >= max_nodes:
                        break

        nodes = connected_nodes[:max_nodes]
        for n in nodes:
            n['entities'] = []
            n['content_preview'] = _preview(n.get('content'))

        # Filter edges to only those between displayed nodes
        node_ids = {n['id'] for n in nodes}
        links = [lk for lk in all_links
                 if lk['source'] in node_ids and lk['target'] in node_ids]
        return nodes, links, []

    # V2 fallback
    try:
        cursor.execute("""
            SELECT m.id, m.content, m.summary, m.category, m.cluster_id,
                   m.importance, m.project_name, m.created_at, m.tags, gn.entities
            FROM memories m LEFT JOIN graph_nodes gn ON m.id = gn.memory_id
            WHERE m.importance >= ? AND m.profile = ?
            ORDER BY m.importance DESC, m.updated_at DESC LIMIT ?
        """, (min_importance, profile, max_nodes))
    except Exception:
        cursor.execute("""
            SELECT id, content, summary, category, cluster_id, importance,
                   project_name, created_at, tags, NULL as entities
            FROM memories WHERE importance >= ? AND profile = ?
            ORDER BY importance DESC, updated_at DESC LIMIT ?
        """, (min_importance, profile, max_nodes))
    nodes = cursor.fetchall()
    for n in nodes:
        ent = n.get('entities')
        n['entities'] = json.loads(ent) if ent else []
        n['content_preview'] = _preview(n.get('content'))
    ids = [n['id'] for n in nodes]
    links = _fetch_edges_v2(cursor, ids)
    try:
        cursor.execute("""
            SELECT cluster_id, COUNT(*) as size, AVG(importance) as avg_importance
            FROM memories WHERE cluster_id IS NOT NULL AND profile = ?
            GROUP BY cluster_id
        """, (profile,))
        clusters = cursor.fetchall()
    except Exception:
        clusters = []
    return nodes, links, clusters


def _fetch_edges_v3(cursor, profile: str, fact_ids: list) -> list:
    if not fact_ids:
        return []
    ph = ','.join('?' * len(fact_ids))
    try:
        cursor.execute(f"""
            SELECT source_id as source, target_id as target,
                   weight, edge_type as relationship_type
            FROM graph_edges WHERE profile_id = ?
              AND source_id IN ({ph}) AND target_id IN ({ph})
            ORDER BY weight DESC
        """, [profile] + fact_ids + fact_ids)
        return cursor.fetchall()
    except Exception:
        return []


def _fetch_edges_v2(cursor, memory_ids: list) -> list:
    if not memory_ids:
        return []
    ph = ','.join('?' * len(memory_ids))
    try:
        cursor.execute(f"""
            SELECT source_memory_id as source, target_memory_id as target,
                   weight, relationship_type, shared_entities
            FROM graph_edges
            WHERE source_memory_id IN ({ph}) AND target_memory_id IN ({ph})
            ORDER BY weight DESC
        """, memory_ids + memory_ids)
        links = cursor.fetchall()
        for lk in links:
            se = lk.get('shared_entities')
            if se:
                try:
                    lk['shared_entities'] = json.loads(se)
                except Exception:
                    lk['shared_entities'] = []
        return links
    except Exception:
        return []


@router.get("/api/memories")
async def get_memories(
    request: Request,
    category: Optional[str] = None,
    project_name: Optional[str] = None,
    cluster_id: Optional[int] = None,
    min_importance: Optional[int] = None,
    tags: Optional[str] = None,
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    """List memories with optional filtering and pagination."""
    try:
        conn = get_db_connection()
        conn.row_factory = dict_factory
        cursor = conn.cursor()
        active_profile = get_active_profile()

        use_v3 = _has_table(cursor, 'atomic_facts')

        if use_v3:
            query = """
                SELECT fact_id as id, memory_id, content, fact_type as category,
                       confidence as importance, access_count,
                       created_at, created_at as updated_at,
                       session_id as project_name
                FROM atomic_facts WHERE profile_id = ?
            """
            params = [active_profile]
            count_base = "SELECT COUNT(*) as total FROM atomic_facts WHERE profile_id = ?"
        else:
            query = """
                SELECT id, content, summary, category, project_name, project_path,
                       importance, cluster_id, depth, access_count, parent_id,
                       created_at, updated_at, last_accessed, tags, memory_type
                FROM memories WHERE profile = ?
            """
            params = [active_profile]
            count_base = "SELECT COUNT(*) as total FROM memories WHERE profile = ?"

        count_params = [active_profile]

        if category:
            if use_v3:
                query += " AND fact_type = ?"
            else:
                query += " AND category = ?"
            params.append(category)
            count_base += " AND category = ?" if not use_v3 else " AND fact_type = ?"
            count_params.append(category)
        if project_name:
            if use_v3:
                query += " AND session_id = ?"
            else:
                query += " AND project_name = ?"
            params.append(project_name)
            count_base += " AND project_name = ?" if not use_v3 else " AND session_id = ?"
            count_params.append(project_name)
        if cluster_id is not None and not use_v3:
            query += " AND cluster_id = ?"
            params.append(cluster_id)
            count_base += " AND cluster_id = ?"
            count_params.append(cluster_id)
        if min_importance:
            if use_v3:
                query += " AND confidence >= ?"
                params.append(min_importance / 10.0)
            else:
                query += " AND importance >= ?"
                params.append(min_importance)
        if tags and not use_v3:
            tag_list = [t.strip() for t in tags.split(',')]
            for tag in tag_list:
                query += " AND tags LIKE ?"
                params.append(f'%{tag}%')

        query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        cursor.execute(query, params)
        memories = cursor.fetchall()

        cursor.execute(count_base, count_params)
        total = cursor.fetchone()['total']

        conn.close()

        return {
            "memories": memories, "total": total,
            "limit": limit, "offset": offset,
            "has_more": (offset + limit) < total,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")


@router.get("/api/graph")
async def get_graph(
    request: Request,
    max_nodes: int = Query(100, ge=10, le=500),
    min_importance: int = Query(1, ge=1, le=10),
):
    """Get knowledge graph data for D3.js force-directed visualization."""
    try:
        conn = get_db_connection()
        conn.row_factory = dict_factory
        cursor = conn.cursor()
        active_profile = get_active_profile()

        use_v3 = _has_table(cursor, 'atomic_facts')

        nodes, links, clusters = _fetch_graph_data(
            cursor, active_profile, use_v3, min_importance, max_nodes,
        )

        conn.close()

        return {
            "nodes": nodes, "links": links, "clusters": clusters,
            "metadata": {
                "node_count": len(nodes), "edge_count": len(links),
                "cluster_count": len(clusters) if clusters else 0,
                "filters_applied": {"max_nodes": max_nodes, "min_importance": min_importance},
            },
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Graph error: {str(e)}")


@router.post("/api/search")
async def search_memories(request: Request, body: SearchRequest):
    """Semantic search via subprocess worker pool (memory-isolated)."""
    try:
        from superlocalmemory.core.worker_pool import WorkerPool
        pool = WorkerPool.shared()
        result = pool.recall(body.query, limit=body.limit)

        if result.get("ok"):
            return {
                "query": body.query,
                "results": result.get("results", []),
                "total": result.get("result_count", 0),
                "query_type": result.get("query_type", "unknown"),
                "retrieval_time_ms": result.get("retrieval_time_ms", 0),
            }

        # Fallback: direct DB text search (no engine needed)
        conn = get_db_connection()
        conn.row_factory = dict_factory
        cursor = conn.cursor()
        active_profile = get_active_profile()
        cursor.execute("""
            SELECT fact_id as id, content, confidence as score,
                   fact_type as category, created_at
            FROM atomic_facts
            WHERE profile_id = ? AND content LIKE ?
            ORDER BY confidence DESC LIMIT ?
        """, (active_profile, f'%{body.query}%', body.limit))
        results = cursor.fetchall()
        conn.close()

        return {
            "query": body.query, "results": results, "total": len(results),
            "query_type": "text_search", "retrieval_time_ms": 0,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")


@router.get("/api/clusters")
async def get_clusters(request: Request):
    """Get cluster information with member counts and statistics."""
    try:
        conn = get_db_connection()
        conn.row_factory = dict_factory
        cursor = conn.cursor()
        profile = get_active_profile()
        unclustered = 0

        if _has_table(cursor, 'scene_facts'):
            cursor.execute("""
                SELECT s.scene_id as cluster_id, COUNT(sf.fact_id) as member_count,
                       s.summary, s.created_at as first_memory
                FROM scenes s JOIN scene_facts sf ON s.scene_id = sf.scene_id
                WHERE s.profile_id = ? GROUP BY s.scene_id ORDER BY member_count DESC
            """, (profile,))
            clusters = [dict(r, top_entities=[]) for r in cursor.fetchall()]
        else:
            try:
                cursor.execute("""
                    SELECT cluster_id, COUNT(*) as member_count,
                           AVG(importance) as avg_importance,
                           GROUP_CONCAT(DISTINCT category) as categories
                    FROM memories WHERE cluster_id IS NOT NULL AND profile = ?
                    GROUP BY cluster_id ORDER BY member_count DESC
                """, (profile,))
                clusters = [dict(r, top_entities=[]) for r in cursor.fetchall()]
            except Exception:
                clusters = []
            cursor.execute("SELECT COUNT(*) as c FROM memories WHERE cluster_id IS NULL AND profile = ?", (profile,))
            unclustered = cursor.fetchone()['c']

        conn.close()
        return {"clusters": clusters, "total_clusters": len(clusters), "unclustered_count": unclustered}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cluster error: {str(e)}")


@router.get("/api/clusters/{cluster_id}")
async def get_cluster_detail(request: Request, cluster_id: int, limit: int = Query(50, ge=1, le=200)):
    """Get detailed view of a specific cluster."""
    try:
        conn = get_db_connection()
        conn.row_factory = dict_factory
        cursor = conn.cursor()
        profile = get_active_profile()

        if _has_table(cursor, 'scene_facts'):
            cursor.execute("""
                SELECT f.fact_id as id, f.content, f.fact_type as category,
                       f.confidence as importance, f.created_at
                FROM atomic_facts f JOIN scene_facts sf ON f.fact_id = sf.fact_id
                WHERE sf.scene_id = ? AND f.profile_id = ? ORDER BY f.confidence DESC LIMIT ?
            """, (str(cluster_id), profile, limit))
        else:
            cursor.execute("""
                SELECT id, content, summary, category, project_name, importance, created_at, tags
                FROM memories WHERE cluster_id = ? AND profile = ?
                ORDER BY importance DESC, created_at DESC LIMIT ?
            """, (cluster_id, profile, limit))
        members = cursor.fetchall()
        conn.close()
        if not members:
            raise HTTPException(status_code=404, detail="Cluster not found")
        # Generate cluster summary
        summary = ""
        try:
            from superlocalmemory.core.worker_pool import WorkerPool
            pool = WorkerPool.shared()
            texts = [m.get("content", "")[:200] for m in members[:10] if m.get("content")]
            if texts:
                result = pool.summarize(texts)
                summary = result.get("summary", "") if result.get("ok") else ""
        except Exception:
            pass

        return {
            "cluster_info": {"cluster_id": cluster_id, "total_members": len(members)},
            "summary": summary,
            "members": members,
            "connections": [],
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cluster detail error: {str(e)}")


@router.get("/api/memories/{memory_id}/facts")
async def get_memory_facts(request: Request, memory_id: str):
    """Get original memory text with all its child atomic facts."""
    try:
        from superlocalmemory.core.worker_pool import WorkerPool
        pool = WorkerPool.shared()
        result = pool.get_memory_facts(memory_id)
        if result.get("ok"):
            return result
        raise HTTPException(status_code=404, detail="Memory not found")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
