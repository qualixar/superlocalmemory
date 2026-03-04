#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 SuperLocalMemory (superlocalmemory.com)
"""
SuperLocalMemory V2 - FastAPI UI Server
Provides REST endpoints for memory visualization and exploration.
"""

import sqlite3
import json
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta

from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
import uvicorn

from security_middleware import SecurityHeadersMiddleware

# Import local modules
import sys
sys.path.insert(0, str(Path(__file__).parent / "src"))

from memory_store_v2 import MemoryStoreV2
from graph_engine import GraphEngine
from pattern_learner import PatternLearner

# Configuration
MEMORY_DIR = Path.home() / ".claude-memory"
DB_PATH = MEMORY_DIR / "memory.db"
UI_DIR = Path(__file__).parent / "ui"

app = FastAPI(
    title="SuperLocalMemory V2 UI",
    description="Knowledge Graph Visualization for Local Memory System",
    version="2.0.0"
)

# Security middleware (add first for proper header application)
app.add_middleware(SecurityHeadersMiddleware)

# Mount static files
UI_DIR.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(UI_DIR)), name="static")

# Rate limiting (v2.6)
try:
    from rate_limiter import write_limiter, read_limiter

    @app.middleware("http")
    async def rate_limit_middleware(request, call_next):
        client_ip = request.client.host if request.client else "unknown"

        # Determine if this is a write or read endpoint
        is_write = request.method in ("POST", "PUT", "DELETE", "PATCH")
        limiter = write_limiter if is_write else read_limiter

        allowed, remaining = limiter.is_allowed(client_ip)
        if not allowed:
            from fastapi.responses import JSONResponse
            return JSONResponse(
                status_code=429,
                content={"error": "Too many requests. Please slow down."},
                headers={"Retry-After": str(limiter.window)}
            )

        response = await call_next(request)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        return response

except ImportError:
    pass  # Rate limiter not available — continue without it

# Optional API key authentication (v2.6)
try:
    from auth_middleware import check_api_key

    @app.middleware("http")
    async def auth_middleware(request, call_next):
        is_write = request.method in ("POST", "PUT", "DELETE", "PATCH")
        headers = dict(request.headers)
        if not check_api_key(headers, is_write=is_write):
            from fastapi.responses import JSONResponse
            return JSONResponse(
                status_code=401,
                content={"error": "Invalid or missing API key. Set X-SLM-API-Key header."}
            )
        response = await call_next(request)
        return response
except ImportError:
    pass  # Auth middleware not available


# ============================================================================
# Request/Response Models
# ============================================================================

class SearchRequest(BaseModel):
    query: str
    limit: int = 10
    min_score: float = 0.3


class MemoryFilter(BaseModel):
    category: Optional[str] = None
    project_name: Optional[str] = None
    cluster_id: Optional[int] = None
    min_importance: Optional[int] = None


# ============================================================================
# Database Helper Functions
# ============================================================================

def get_db_connection():
    """Get database connection."""
    if not DB_PATH.exists():
        raise HTTPException(status_code=500, detail="Memory database not found")
    return sqlite3.connect(DB_PATH)


def dict_factory(cursor, row):
    """Convert SQLite row to dictionary."""
    fields = [column[0] for column in cursor.description]
    return {key: value for key, value in zip(fields, row)}


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve main UI page."""
    index_path = UI_DIR / "index.html"
    if not index_path.exists():
        return """
        <html>
            <head><title>SuperLocalMemory V2</title></head>
            <body style="font-family: Arial; padding: 40px;">
                <h1>SuperLocalMemory V2 UI Server Running</h1>
                <p>UI not found. Please create ui/index.html</p>
                <h2>Available Endpoints:</h2>
                <ul>
                    <li><a href="/docs">/docs - Interactive API Documentation</a></li>
                    <li><a href="/api/stats">/api/stats - System Statistics</a></li>
                    <li><a href="/api/memories">/api/memories - List Memories</a></li>
                    <li><a href="/api/graph">/api/graph - Graph Data</a></li>
                    <li><a href="/api/clusters">/api/clusters - Cluster Info</a></li>
                    <li><a href="/api/patterns">/api/patterns - Learned Patterns</a></li>
                    <li><a href="/api/timeline">/api/timeline - Timeline View</a></li>
                    <li><a href="/api/tree">/api/tree - Tree Structure</a></li>
                </ul>
            </body>
        </html>
        """
    return index_path.read_text()


@app.get("/api/memories")
async def get_memories(
    category: Optional[str] = None,
    project_name: Optional[str] = None,
    cluster_id: Optional[int] = None,
    min_importance: Optional[int] = None,
    limit: int = Query(50, le=200),
    offset: int = 0
):
    """
    List memories with optional filtering.

    Query Parameters:
    - category: Filter by category
    - project_name: Filter by project
    - cluster_id: Filter by cluster
    - min_importance: Minimum importance score
    - limit: Maximum results (default 50, max 200)
    - offset: Pagination offset
    """
    conn = get_db_connection()
    conn.row_factory = dict_factory
    cursor = conn.cursor()

    # Build dynamic query
    query = """
        SELECT
            id, content, summary, category, project_name,
            importance, cluster_id, depth, access_count,
            created_at, updated_at, last_accessed, tags
        FROM memories
        WHERE 1=1
    """
    params = []

    if category:
        query += " AND category = ?"
        params.append(category)

    if project_name:
        query += " AND project_name = ?"
        params.append(project_name)

    if cluster_id is not None:
        query += " AND cluster_id = ?"
        params.append(cluster_id)

    if min_importance:
        query += " AND importance >= ?"
        params.append(min_importance)

    query += " ORDER BY updated_at DESC LIMIT ? OFFSET ?"
    params.extend([limit, offset])

    cursor.execute(query, params)
    memories = cursor.fetchall()

    # Get total count
    count_query = "SELECT COUNT(*) as total FROM memories WHERE 1=1"
    count_params = []
    if category:
        count_query += " AND category = ?"
        count_params.append(category)
    if project_name:
        count_query += " AND project_name = ?"
        count_params.append(project_name)
    if cluster_id is not None:
        count_query += " AND cluster_id = ?"
        count_params.append(cluster_id)
    if min_importance:
        count_query += " AND importance >= ?"
        count_params.append(min_importance)

    cursor.execute(count_query, count_params)
    total = cursor.fetchone()['total']

    conn.close()

    return {
        "memories": memories,
        "total": total,
        "limit": limit,
        "offset": offset
    }


@app.get("/api/graph")
async def get_graph(max_nodes: int = Query(100, le=500)):
    """
    Get graph data for D3.js force-directed visualization.

    Returns:
    - nodes: List of memory nodes with metadata
    - links: List of edges between memories
    """
    conn = get_db_connection()
    conn.row_factory = dict_factory
    cursor = conn.cursor()

    # Get nodes (memories with graph data)
    cursor.execute("""
        SELECT
            m.id, m.content, m.summary, m.category,
            m.cluster_id, m.importance, m.project_name,
            gn.entities
        FROM memories m
        LEFT JOIN graph_nodes gn ON m.id = gn.memory_id
        WHERE m.cluster_id IS NOT NULL
        ORDER BY m.importance DESC, m.updated_at DESC
        LIMIT ?
    """, (max_nodes,))
    nodes = cursor.fetchall()

    # Parse entities JSON
    for node in nodes:
        if node['entities']:
            try:
                node['entities'] = json.loads(node['entities'])
            except:
                node['entities'] = []
        else:
            node['entities'] = []

        # Truncate content for display
        if node['content'] and len(node['content']) > 100:
            node['content_preview'] = node['content'][:100] + "..."
        else:
            node['content_preview'] = node['content']

    # Get edges
    memory_ids = [n['id'] for n in nodes]
    if memory_ids:
        placeholders = ','.join('?' * len(memory_ids))
        cursor.execute(f"""
            SELECT
                source_memory_id as source,
                target_memory_id as target,
                weight,
                relationship_type,
                shared_entities
            FROM graph_edges
            WHERE source_memory_id IN ({placeholders})
              AND target_memory_id IN ({placeholders})
            ORDER BY weight DESC
        """, memory_ids + memory_ids)
        links = cursor.fetchall()

        # Parse shared entities
        for link in links:
            if link['shared_entities']:
                try:
                    link['shared_entities'] = json.loads(link['shared_entities'])
                except:
                    link['shared_entities'] = []
    else:
        links = []

    conn.close()

    return {
        "nodes": nodes,
        "links": links,
        "metadata": {
            "node_count": len(nodes),
            "edge_count": len(links)
        }
    }


@app.get("/api/clusters")
async def get_clusters():
    """
    Get cluster information with member counts and themes.

    Returns list of clusters with:
    - cluster_id
    - member_count
    - dominant_entities (most common concepts)
    - categories represented
    - importance_avg
    """
    conn = get_db_connection()
    conn.row_factory = dict_factory
    cursor = conn.cursor()

    # Get cluster statistics
    cursor.execute("""
        SELECT
            cluster_id,
            COUNT(*) as member_count,
            AVG(importance) as avg_importance,
            GROUP_CONCAT(DISTINCT category) as categories,
            GROUP_CONCAT(DISTINCT project_name) as projects
        FROM memories
        WHERE cluster_id IS NOT NULL
        GROUP BY cluster_id
        ORDER BY member_count DESC
    """)
    clusters = cursor.fetchall()

    # Get dominant entities per cluster
    for cluster in clusters:
        cluster_id = cluster['cluster_id']

        # Aggregate entities from all members
        cursor.execute("""
            SELECT gn.entities
            FROM graph_nodes gn
            JOIN memories m ON gn.memory_id = m.id
            WHERE m.cluster_id = ?
        """, (cluster_id,))

        all_entities = []
        for row in cursor.fetchall():
            if row['entities']:
                try:
                    entities = json.loads(row['entities'])
                    all_entities.extend(entities)
                except:
                    pass

        # Count and get top 5
        from collections import Counter
        entity_counts = Counter(all_entities)
        cluster['top_entities'] = [
            {"entity": e, "count": c}
            for e, c in entity_counts.most_common(5)
        ]

    conn.close()

    return {
        "clusters": clusters,
        "total_clusters": len(clusters)
    }


@app.get("/api/patterns")
async def get_patterns():
    """
    Get learned patterns from Pattern Learner (Layer 4).

    Returns user preferences, coding style, and terminology patterns.
    """
    try:
        # Initialize pattern learner
        learner = PatternLearner(DB_PATH)

        # Get all active patterns
        conn = get_db_connection()
        conn.row_factory = dict_factory
        cursor = conn.cursor()

        cursor.execute("""
            SELECT
                pattern_type, key, value, confidence,
                evidence_count, last_updated
            FROM learned_patterns
            WHERE is_active = 1
            ORDER BY confidence DESC, evidence_count DESC
        """)
        patterns = cursor.fetchall()

        # Parse value JSON
        for pattern in patterns:
            if pattern['value']:
                try:
                    pattern['value'] = json.loads(pattern['value'])
                except:
                    pass

        # Group by type
        grouped = {}
        for pattern in patterns:
            ptype = pattern['pattern_type']
            if ptype not in grouped:
                grouped[ptype] = []
            grouped[ptype].append(pattern)

        conn.close()

        return {
            "patterns": grouped,
            "total_patterns": len(patterns),
            "pattern_types": list(grouped.keys())
        }

    except Exception as e:
        return {
            "patterns": {},
            "total_patterns": 0,
            "error": str(e)
        }


@app.get("/api/stats")
async def get_stats():
    """
    Get system statistics overview.

    Returns:
    - Total memories, sessions, clusters
    - Storage usage
    - Recent activity
    - Category breakdown
    """
    conn = get_db_connection()
    conn.row_factory = dict_factory
    cursor = conn.cursor()

    # Basic counts
    cursor.execute("SELECT COUNT(*) as total FROM memories")
    total_memories = cursor.fetchone()['total']

    cursor.execute("SELECT COUNT(*) as total FROM sessions")
    total_sessions = cursor.fetchone()['total']

    cursor.execute("SELECT COUNT(DISTINCT cluster_id) as total FROM memories WHERE cluster_id IS NOT NULL")
    total_clusters = cursor.fetchone()['total']

    cursor.execute("SELECT COUNT(*) as total FROM graph_nodes")
    total_graph_nodes = cursor.fetchone()['total']

    cursor.execute("SELECT COUNT(*) as total FROM graph_edges")
    total_graph_edges = cursor.fetchone()['total']

    # Category breakdown
    cursor.execute("""
        SELECT category, COUNT(*) as count
        FROM memories
        WHERE category IS NOT NULL
        GROUP BY category
        ORDER BY count DESC
        LIMIT 10
    """)
    categories = cursor.fetchall()

    # Project breakdown
    cursor.execute("""
        SELECT project_name, COUNT(*) as count
        FROM memories
        WHERE project_name IS NOT NULL
        GROUP BY project_name
        ORDER BY count DESC
        LIMIT 10
    """)
    projects = cursor.fetchall()

    # Recent activity (last 7 days)
    cursor.execute("""
        SELECT COUNT(*) as count
        FROM memories
        WHERE created_at >= datetime('now', '-7 days')
    """)
    recent_memories = cursor.fetchone()['count']

    # Importance distribution
    cursor.execute("""
        SELECT importance, COUNT(*) as count
        FROM memories
        GROUP BY importance
        ORDER BY importance DESC
    """)
    importance_dist = cursor.fetchall()

    # Database size
    db_size = DB_PATH.stat().st_size if DB_PATH.exists() else 0

    conn.close()

    return {
        "overview": {
            "total_memories": total_memories,
            "total_sessions": total_sessions,
            "total_clusters": total_clusters,
            "graph_nodes": total_graph_nodes,
            "graph_edges": total_graph_edges,
            "db_size_mb": round(db_size / (1024 * 1024), 2),
            "recent_memories_7d": recent_memories
        },
        "categories": categories,
        "projects": projects,
        "importance_distribution": importance_dist
    }


@app.post("/api/search")
async def search_memories(request: SearchRequest):
    """
    Semantic search using TF-IDF similarity.

    Request body:
    - query: Search query
    - limit: Max results (default 10)
    - min_score: Minimum similarity score (default 0.3)
    """
    try:
        store = MemoryStoreV2(DB_PATH)
        results = store.search(
            query=request.query,
            limit=request.limit
        )

        # Filter by min_score
        filtered = [
            r for r in results
            if r.get('score', 0) >= request.min_score
        ]

        return {
            "query": request.query,
            "results": filtered,
            "total": len(filtered)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/timeline")
async def get_timeline(days: int = Query(30, le=365)):
    """
    Get temporal view of memory creation over time.

    Parameters:
    - days: Number of days to look back (default 30, max 365)

    Returns daily/weekly aggregates with category breakdown.
    """
    conn = get_db_connection()
    conn.row_factory = dict_factory
    cursor = conn.cursor()

    # Daily aggregates
    cursor.execute("""
        SELECT
            DATE(created_at) as date,
            COUNT(*) as count,
            AVG(importance) as avg_importance,
            GROUP_CONCAT(DISTINCT category) as categories
        FROM memories
        WHERE created_at >= datetime('now', '-' || ? || ' days')
        GROUP BY DATE(created_at)
        ORDER BY date DESC
    """, (days,))
    daily = cursor.fetchall()

    # Category trend over time
    cursor.execute("""
        SELECT
            DATE(created_at) as date,
            category,
            COUNT(*) as count
        FROM memories
        WHERE created_at >= datetime('now', '-' || ? || ' days')
          AND category IS NOT NULL
        GROUP BY DATE(created_at), category
        ORDER BY date DESC
    """, (days,))
    category_trend = cursor.fetchall()

    conn.close()

    return {
        "timeline": daily,
        "category_trend": category_trend,
        "period_days": days
    }


@app.get("/api/tree")
async def get_tree():
    """
    Get hierarchical tree structure.

    Returns nested tree of projects > categories > memories.
    """
    conn = get_db_connection()
    conn.row_factory = dict_factory
    cursor = conn.cursor()

    # Get all memories with hierarchy info
    cursor.execute("""
        SELECT
            id, parent_id, tree_path, depth,
            project_name, category,
            COALESCE(summary, SUBSTR(content, 1, 100)) as label,
            importance, created_at
        FROM memories
        ORDER BY tree_path
    """)
    nodes = cursor.fetchall()

    # Build tree structure
    tree = []
    node_map = {}

    for node in nodes:
        node['children'] = []
        node_map[node['id']] = node

        if node['parent_id'] is None:
            # Root node
            tree.append(node)
        elif node['parent_id'] in node_map:
            # Add to parent
            node_map[node['parent_id']]['children'].append(node)

    conn.close()

    return {
        "tree": tree,
        "total_nodes": len(nodes)
    }


# ============================================================================
# v2.8 Routes (graceful — don't fail if engines unavailable)
# ============================================================================

try:
    from routes.lifecycle import router as lifecycle_router
    app.include_router(lifecycle_router)
except ImportError:
    pass

try:
    from routes.behavioral import router as behavioral_router
    app.include_router(behavioral_router)
except ImportError:
    pass

try:
    from routes.compliance import router as compliance_router
    app.include_router(compliance_router)
except ImportError:
    pass


# ============================================================================
# Server Startup
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("SuperLocalMemory V2 - UI Server")
    print("=" * 60)
    print(f"Database: {DB_PATH}")
    print(f"UI Directory: {UI_DIR}")
    print("=" * 60)
    print("\nStarting server on http://localhost:8000")
    print("API Documentation: http://localhost:8000/docs")
    print("\nPress Ctrl+C to stop\n")

    # SECURITY: Bind to localhost only to prevent network exposure
    # For network access, use a reverse proxy with authentication
    uvicorn.run(
        app,
        host="127.0.0.1",  # localhost only - NEVER use 0.0.0.0 without auth
        port=8000,
        log_level="info"
    )
