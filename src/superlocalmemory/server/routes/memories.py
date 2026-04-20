# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com
"""SuperLocalMemory V3 - Memory Routes (AGPL-3.0-or-later).
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
        # Recency-first: get the most recent nodes, then find their edges.
        # LEFT JOIN fact_importance for graph metrics (v3.4.1 — additive only).
        cursor.execute("""
            SELECT af.fact_id as id, af.content, af.fact_type as category,
                   af.confidence as importance, af.session_id as project_name,
                   af.created_at,
                   fi.pagerank_score, fi.community_id, fi.degree_centrality
            FROM atomic_facts af
            LEFT JOIN fact_importance fi
                ON af.fact_id = fi.fact_id AND fi.profile_id = ?
            WHERE af.profile_id = ? AND af.confidence >= ?
            ORDER BY af.created_at DESC
            LIMIT ?
        """, (profile, profile, min_importance / 10.0, max_nodes))
        nodes = cursor.fetchall()

        node_ids = {n['id'] for n in nodes}

        # Fetch edges between these nodes
        if node_ids:
            ph = ','.join('?' * len(node_ids))
            id_list = list(node_ids)
            cursor.execute(f"""
                SELECT source_id as source, target_id as target,
                       weight, edge_type as relationship_type
                FROM graph_edges
                WHERE profile_id = ?
                  AND source_id IN ({ph}) AND target_id IN ({ph})
                ORDER BY weight DESC
            """, [profile] + id_list + id_list)
            all_links = cursor.fetchall()
        else:
            all_links = []

        links = all_links
        for n in nodes:
            n['entities'] = []
            n['content_preview'] = _preview(n.get('content'))
            # v3.4.1: Default graph metrics when fact_importance has no data
            if n.get('pagerank_score') is None:
                n['pagerank_score'] = 0.0
            if n.get('community_id') is None:
                n['community_id'] = 0
            if n.get('degree_centrality') is None:
                n['degree_centrality'] = 0.0

        # Filter edges to only those between displayed nodes
        node_ids = {n['id'] for n in nodes}
        links = [lk for lk in all_links
                 if lk['source'] in node_ids and lk['target'] in node_ids]

        # Compute clusters from memory_scenes
        clusters = []
        try:
            cursor.execute("""
                SELECT scene_id, theme, fact_ids_json
                FROM memory_scenes WHERE profile_id = ?
            """, (profile,))
            for row in cursor.fetchall():
                fact_ids = []
                try:
                    fact_ids = json.loads(row.get('fact_ids_json', '[]') or '[]')
                except (json.JSONDecodeError, TypeError):
                    pass
                # Only include clusters that overlap with displayed nodes
                overlap = [fid for fid in fact_ids if fid in node_ids]
                if overlap:
                    clusters.append({
                        'cluster_id': row['scene_id'],
                        'size': len(fact_ids),
                        'visible_size': len(overlap),
                        'theme': row.get('theme', ''),
                    })
        except Exception:
            pass

        return nodes, links, clusters

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
    filter: Optional[str] = Query(
        None,
        description="Named filter: 'high_reward' | 'being_forgotten'",
    ),
):
    """List memories with optional filtering and pagination.

    S9-DASH-07: ``filter`` enables dashboard "learning-visible" views:

    * ``high_reward``: facts cited by ``action_outcomes`` with
      ``reward >= 0.7`` in the last 30 days. Surfaces what the ranker
      is actually learning from.
    * ``being_forgotten``: facts in ``archive_status='archived'`` OR
      with ``lifecycle='cold'`` AND no positive reward in 60 days.
      Makes "memory decay" tangible to the operator.
    """
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

        # S9-DASH-07: named filters — "high_reward" and "being_forgotten".
        # Only supported on the v3 (atomic_facts) path — v2 fallback
        # ignores the flag silently.
        if filter and use_v3:
            if filter == "high_reward":
                query += (
                    " AND fact_id IN ("
                    "  SELECT DISTINCT json_each.value"
                    "  FROM action_outcomes, json_each(action_outcomes.fact_ids_json)"
                    "  WHERE action_outcomes.reward >= 0.7"
                    "    AND datetime(action_outcomes.settled_at) >= "
                    "        datetime('now', '-30 day')"
                    ")"
                )
                count_base += (
                    " AND fact_id IN ("
                    "  SELECT DISTINCT json_each.value"
                    "  FROM action_outcomes, json_each(action_outcomes.fact_ids_json)"
                    "  WHERE action_outcomes.reward >= 0.7"
                    "    AND datetime(action_outcomes.settled_at) >= "
                    "        datetime('now', '-30 day')"
                    ")"
                )
            elif filter == "being_forgotten":
                # Cold / archived + no recent positive reward.
                query += (
                    " AND ("
                    "  archive_status = 'archived' OR "
                    "  (lifecycle = 'cold' AND fact_id NOT IN ("
                    "    SELECT DISTINCT json_each.value"
                    "    FROM action_outcomes, json_each(action_outcomes.fact_ids_json)"
                    "    WHERE action_outcomes.reward >= 0.5"
                    "      AND datetime(action_outcomes.settled_at) >= "
                    "          datetime('now', '-60 day')"
                    "  ))"
                    ")"
                )
                count_base += (
                    " AND ("
                    "  archive_status = 'archived' OR "
                    "  (lifecycle = 'cold' AND fact_id NOT IN ("
                    "    SELECT DISTINCT json_each.value"
                    "    FROM action_outcomes, json_each(action_outcomes.fact_ids_json)"
                    "    WHERE action_outcomes.reward >= 0.5"
                    "      AND datetime(action_outcomes.settled_at) >= "
                    "          datetime('now', '-60 day')"
                    "  ))"
                    ")"
                )

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
    max_nodes: int = Query(100, ge=10, le=10000),
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

        # V3 schema: memory_scenes stores fact_ids_json (JSON array)
        if _has_table(cursor, 'memory_scenes'):
            cursor.execute("""
                SELECT scene_id as cluster_id, theme, fact_ids_json,
                       entity_ids_json, created_at as first_memory
                FROM memory_scenes WHERE profile_id = ?
                ORDER BY created_at DESC
            """, (profile,))
            raw_scenes = cursor.fetchall()
            clusters = []
            for scene in raw_scenes:
                fact_ids = []
                try:
                    fact_ids = json.loads(scene.get('fact_ids_json', '[]') or '[]')
                except (json.JSONDecodeError, TypeError):
                    pass
                entity_ids = []
                try:
                    entity_ids = json.loads(scene.get('entity_ids_json', '[]') or '[]')
                except (json.JSONDecodeError, TypeError):
                    pass
                clusters.append({
                    'cluster_id': scene['cluster_id'],
                    'member_count': len(fact_ids),
                    'categories': scene.get('theme', ''),
                    'summary': scene.get('theme', ''),
                    'first_memory': scene.get('first_memory', ''),
                    'top_entities': entity_ids[:5],
                })
            # Filter out empty clusters
            clusters = [c for c in clusters if c['member_count'] > 0]
            clusters.sort(key=lambda c: c['member_count'], reverse=True)

            # Count facts not in any scene
            all_scene_fact_ids = set()
            for scene in raw_scenes:
                try:
                    ids = json.loads(scene.get('fact_ids_json', '[]') or '[]')
                    all_scene_fact_ids.update(ids)
                except (json.JSONDecodeError, TypeError):
                    pass
            total_facts = cursor.execute(
                "SELECT COUNT(*) as c FROM atomic_facts WHERE profile_id = ?",
                (profile,),
            ).fetchone()['c']
            unclustered = total_facts - len(all_scene_fact_ids)
        else:
            # V2 fallback
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
            try:
                cursor.execute(
                    "SELECT COUNT(*) as c FROM memories WHERE cluster_id IS NULL AND profile = ?",
                    (profile,),
                )
                unclustered = cursor.fetchone()['c']
            except Exception:
                unclustered = 0

        conn.close()
        return {"clusters": clusters, "total_clusters": len(clusters), "unclustered_count": unclustered}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cluster error: {str(e)}")


@router.get("/api/clusters/{cluster_id}")
async def get_cluster_detail(request: Request, cluster_id: str, limit: int = Query(50, ge=1, le=200)):
    """Get detailed view of a specific cluster (scene)."""
    try:
        conn = get_db_connection()
        conn.row_factory = dict_factory
        cursor = conn.cursor()
        profile = get_active_profile()

        if _has_table(cursor, 'memory_scenes'):
            # Get fact IDs from the scene's JSON array
            cursor.execute(
                "SELECT fact_ids_json, theme FROM memory_scenes "
                "WHERE scene_id = ? AND profile_id = ?",
                (cluster_id, profile),
            )
            scene_row = cursor.fetchone()
            if scene_row:
                fact_ids = []
                try:
                    fact_ids = json.loads(scene_row.get('fact_ids_json', '[]') or '[]')
                except (json.JSONDecodeError, TypeError):
                    pass
                if fact_ids:
                    ph = ','.join('?' * min(len(fact_ids), limit))
                    cursor.execute(f"""
                        SELECT fact_id as id, content, fact_type as category,
                               confidence as importance, created_at
                        FROM atomic_facts
                        WHERE profile_id = ? AND fact_id IN ({ph})
                        ORDER BY confidence DESC
                    """, [profile] + fact_ids[:limit])
                else:
                    cursor.execute("SELECT 1 WHERE 0")  # empty result
            else:
                cursor.execute("SELECT 1 WHERE 0")  # empty result
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


@router.delete("/api/memories/{fact_id}")
async def delete_memory(request: Request, fact_id: str):
    """Delete a specific memory (atomic fact) by ID."""
    try:
        conn = get_db_connection()
        conn.row_factory = dict_factory
        cursor = conn.cursor()
        active_profile = get_active_profile()
        # Verify it exists and belongs to this profile
        cursor.execute(
            "SELECT fact_id FROM atomic_facts WHERE fact_id = ? AND profile_id = ?",
            (fact_id, active_profile),
        )
        if not cursor.fetchone():
            conn.close()
            raise HTTPException(status_code=404, detail="Memory not found")
        cursor.execute("DELETE FROM atomic_facts WHERE fact_id = ?", (fact_id,))
        conn.commit()
        conn.close()
        return {"success": True, "deleted": fact_id}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Delete error: {str(e)}")


@router.post("/api/memories/{fact_id}/forget")
async def forget_memory(request: Request, fact_id: str):
    """S9-DASH-08: soft-forget a fact — flip archive_status='archived'.

    Non-destructive: the row stays in ``atomic_facts`` for audit and
    can be un-archived later. Default recall paths filter it out.
    The fact's payload is ALSO copied into ``memory_archive`` so a
    future ``slm restore`` can bring it back.
    """
    import json as _json
    try:
        conn = get_db_connection()
        conn.row_factory = dict_factory
        cursor = conn.cursor()
        active_profile = get_active_profile()
        cursor.execute(
            "SELECT fact_id, content, importance, confidence, "
            "       canonical_entities_json, embedding, created_at "
            "FROM atomic_facts WHERE fact_id = ? AND profile_id = ?",
            (fact_id, active_profile),
        )
        row = cursor.fetchone()
        if not row:
            conn.close()
            raise HTTPException(status_code=404, detail="Memory not found")
        # Archive copy — payload_json small enough for the canonical row.
        payload = {
            "fact_id": row["fact_id"],
            "content": row["content"],
            "canonical_entities_json": row.get("canonical_entities_json"),
            "importance": row.get("importance"),
            "confidence": row.get("confidence"),
            "created_at": row.get("created_at"),
        }
        from datetime import datetime, timezone
        archived_at = datetime.now(timezone.utc).isoformat()
        import uuid as _uuid
        cursor.execute(
            "INSERT INTO memory_archive "
            "(archive_id, fact_id, profile_id, payload_json, archived_at, reason) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (str(_uuid.uuid4()), fact_id, active_profile,
             _json.dumps(payload), archived_at, "user_forget_dashboard"),
        )
        cursor.execute(
            "UPDATE atomic_facts SET archive_status = 'archived' "
            "WHERE fact_id = ?",
            (fact_id,),
        )
        conn.commit()
        conn.close()
        return {"success": True, "fact_id": fact_id, "archived_at": archived_at}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Forget error: {str(e)}")


@router.post("/api/memories/{fact_id}/merge")
async def merge_memory(request: Request, fact_id: str):
    """S9-DASH-08: merge this fact into another (keep the other).

    Body: ``{into: <kept_fact_id>}``.

    Writes a ``memory_merge_log`` row (M011) for provenance and marks
    the loser's ``merged_into`` column. The loser is archived so it
    no longer appears in default recall. The winner is untouched.
    """
    try:
        body = await request.json()
        kept = str((body or {}).get("into", "")).strip()
        if not kept:
            raise HTTPException(400, "Body field 'into' is required")
        # S9-AUDIT: cap length defensively — fact_ids are UUID-v4 36 chars.
        if len(kept) > 200:
            raise HTTPException(400, "'into' exceeds 200-char limit")
        if kept == fact_id:
            raise HTTPException(400, "Cannot merge a fact into itself")
        conn = get_db_connection()
        conn.row_factory = dict_factory
        cursor = conn.cursor()
        active_profile = get_active_profile()
        # Both must belong to the active profile.
        cursor.execute(
            "SELECT fact_id FROM atomic_facts "
            "WHERE fact_id IN (?, ?) AND profile_id = ?",
            (fact_id, kept, active_profile),
        )
        found = {r["fact_id"] for r in cursor.fetchall()}
        if fact_id not in found or kept not in found:
            conn.close()
            raise HTTPException(
                404,
                "Both fact_ids must exist in the active profile",
            )
        from datetime import datetime, timezone
        merged_at = datetime.now(timezone.utc).isoformat()
        cursor.execute(
            "INSERT INTO memory_merge_log "
            "(kept_fact_id, merged_fact_id, profile_id, reason, merged_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (kept, fact_id, active_profile,
             "user_merge_dashboard", merged_at),
        )
        cursor.execute(
            "UPDATE atomic_facts "
            "SET merged_into = ?, archive_status = 'archived' "
            "WHERE fact_id = ?",
            (kept, fact_id),
        )
        conn.commit()
        conn.close()
        return {
            "success": True,
            "merged": fact_id,
            "into": kept,
            "merged_at": merged_at,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Merge error: {str(e)}")


@router.patch("/api/memories/{fact_id}")
async def edit_memory(request: Request, fact_id: str):
    """Edit the content of a specific memory (atomic fact)."""
    try:
        body = await request.json()
        new_content = (body.get("content") or "").strip()
        if not new_content:
            raise HTTPException(status_code=400, detail="content is required")
        conn = get_db_connection()
        conn.row_factory = dict_factory
        cursor = conn.cursor()
        active_profile = get_active_profile()
        cursor.execute(
            "SELECT fact_id FROM atomic_facts WHERE fact_id = ? AND profile_id = ?",
            (fact_id, active_profile),
        )
        if not cursor.fetchone():
            conn.close()
            raise HTTPException(status_code=404, detail="Memory not found")
        cursor.execute(
            "UPDATE atomic_facts SET content = ? WHERE fact_id = ?",
            (new_content, fact_id),
        )
        conn.commit()
        conn.close()
        return {"success": True, "fact_id": fact_id, "content": new_content}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Edit error: {str(e)}")
