# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory v3.4 — CodeGraph MCP Tools

"""22 MCP tools for CodeGraph: 17 graph + 5 bridge.

Registered against `_target` (filtered like all other tools) — set
SLM_MCP_ALL_TOOLS=1 or use a profile containing code-graph names to expose
them. Each tool self-guards (returns error if graph not built).

All tools return {"success": bool, ...} envelope. Never raise.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any, Callable

from superlocalmemory.core.security_primitives import (
    PathTraversalError,
    safe_resolve,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy service singleton
# ---------------------------------------------------------------------------

_service = None
_service_config = None


def _get_service():
    """Lazy-create the CodeGraphService singleton."""
    global _service
    if _service is not None:
        return _service

    try:
        from superlocalmemory.code_graph.config import CodeGraphConfig
        from superlocalmemory.code_graph.service import CodeGraphService

        config = CodeGraphConfig(enabled=True)
        _service = CodeGraphService(config)
        return _service
    except Exception as exc:
        logger.warning("Failed to create CodeGraphService: %s", exc)
        return None


def _get_db():
    """Get the CodeGraphDatabase from the service."""
    svc = _get_service()
    if svc is None:
        return None
    try:
        return svc.db
    except Exception:
        return None


def _graph_not_built_error() -> dict[str, Any]:
    """Standard error when graph is not built."""
    return {
        "success": False,
        "error": "Code graph not built. Run build_code_graph first.",
    }


def _bridge_not_enabled_error() -> dict[str, Any]:
    """Standard error when bridge is not enabled."""
    return {
        "success": False,
        "error": "Bridge not enabled. Set code_graph.bridge.enabled = true in config.",
    }


def _error_response(msg: str) -> dict[str, Any]:
    """Standard error response."""
    return {"success": False, "error": msg}


def _check_graph_exists() -> dict[str, Any] | None:
    """Check if graph has been built. Returns error dict or None if OK."""
    db = _get_db()
    if db is None:
        return _graph_not_built_error()
    stats = db.get_stats()
    if stats.get("nodes", 0) == 0 and stats.get("files", 0) == 0:
        return _graph_not_built_error()
    return None


# ---------------------------------------------------------------------------
# Registration function
# ---------------------------------------------------------------------------


def register_code_graph_tools(server, get_engine: Callable) -> None:
    """Register 22 code graph MCP tools on *server*."""

    # ==================================================================
    # Tool 1: build_code_graph
    # ==================================================================

    @server.tool()
    async def build_code_graph(
        repo_path: str,
        languages: str = "",
        exclude_patterns: str = "",
    ) -> dict:
        """Build a complete code knowledge graph from a repository.

        Parses all supported source files, extracts functions/classes/imports,
        builds call graph, detects flows, and identifies communities.

        Args:
            repo_path: Absolute path to repository root.
            languages: Comma-separated language filter (e.g. "python,typescript"). Empty = all.
            exclude_patterns: Comma-separated glob patterns to exclude.
        """
        try:
            # SEC: contain the untrusted repo_path under $HOME (same guard as
            # update_code_graph) so repo_path="/" cannot ingest the whole
            # filesystem into the code-graph DB.
            from superlocalmemory.core.security_primitives import PathTraversalError
            try:
                repo = safe_resolve(Path.home(), repo_path)
            except (PathTraversalError, OSError, ValueError) as exc:
                return _error_response(f"Invalid repo_path: {exc}")
            if not repo.exists():
                return _error_response(
                    f"Repository path does not exist: {repo_path}"
                )

            from superlocalmemory.code_graph.config import CodeGraphConfig
            from superlocalmemory.code_graph.service import CodeGraphService
            from superlocalmemory.code_graph.parser import CodeParser
            from superlocalmemory.code_graph.graph_store import GraphStore
            from superlocalmemory.code_graph.graph_engine import GraphEngine
            from superlocalmemory.code_graph.search import HybridSearch
            from superlocalmemory.code_graph.flows import FlowDetector
            from superlocalmemory.code_graph.communities import CommunityDetector

            t0 = time.time()

            # Build config
            config_kwargs: dict[str, Any] = {
                "enabled": True,
                "repo_root": repo,
            }
            if languages:
                config_kwargs["languages"] = frozenset(
                    l.strip() for l in languages.split(",") if l.strip()
                )
            if exclude_patterns:
                config_kwargs["exclude_patterns"] = tuple(
                    p.strip() for p in exclude_patterns.split(",") if p.strip()
                )

            config = CodeGraphConfig(**config_kwargs)
            global _service
            _service = CodeGraphService(config)

            # Parse
            parser = CodeParser(config)
            nodes, edges, file_records = parser.parse_all(repo)

            if not nodes:
                return {
                    "success": True,
                    "files_parsed": 0,
                    "nodes": 0,
                    "edges": 0,
                    "flows": 0,
                    "communities": 0,
                    "duration_ms": int((time.time() - t0) * 1000),
                    "message": f"No supported source files found in {repo_path}.",
                }

            # Store in DB
            db = _service.db
            store = GraphStore(db)

            # Group by file for atomic replacement
            file_groups: dict[str, tuple[list, list, Any]] = {}
            for fr in file_records:
                file_groups[fr.file_path] = ([], [], fr)
            for n in nodes:
                fp = n.file_path
                if fp in file_groups:
                    file_groups[fp][0].append(n)
            for e in edges:
                fp = e.file_path
                if fp in file_groups:
                    file_groups[fp][1].append(e)

            for fp, (ns, es, fr) in file_groups.items():
                store.store_file_nodes_edges(fp, ns, es, fr)

            # Build in-memory graph
            engine = GraphEngine(store)
            engine.build_graph()

            # Detect flows
            flow_detector = FlowDetector(db)
            entry_points = flow_detector.detect_entry_points()
            flows = []
            for ep in entry_points[:50]:
                try:
                    flow = flow_detector.trace_flow(ep)
                    if flow is not None:
                        flows.append(flow)
                except Exception:
                    pass

            # Detect communities
            comm_detector = CommunityDetector(db)
            communities = comm_detector.detect_communities()

            duration_ms = int((time.time() - t0) * 1000)

            return {
                "success": True,
                "files_parsed": len(file_records),
                "nodes": db.get_node_count(),
                "edges": db.get_edge_count(),
                "flows": len(flows),
                "communities": len(communities),
                "duration_ms": duration_ms,
            }
        except Exception as exc:
            logger.exception("build_code_graph failed")
            return _error_response(str(exc))

    # ==================================================================
    # Tool 2: update_code_graph
    # ==================================================================

    @server.tool()
    async def update_code_graph(
        repo_path: str = "",
        changed_files: str = "",
    ) -> dict:
        """Incrementally update the code graph for changed files.

        If changed_files is empty, auto-detects changes via git diff HEAD~1.

        Args:
            repo_path: Absolute path to repository root. Empty = use last built repo.
            changed_files: Comma-separated file paths (relative to repo root).
        """
        try:
            err = _check_graph_exists()
            if err is not None:
                return err

            t0 = time.time()
            svc = _get_service()
            db = svc.db

            # SEC 3.7.9 (B1): contain untrusted repo_path under $HOME and confirm
            # it is a real git repo BEFORE any git subprocess runs with it as cwd.
            # Blocks RCE via attacker-controlled .git config/hooks (core.fsmonitor,
            # core.hooksPath) that git would otherwise execute in a hostile repo.
            if repo_path:
                try:
                    repo = safe_resolve(Path.home(), repo_path)
                except PathTraversalError as exc:
                    return _error_response(f"Invalid repo_path: {exc}")
                if not (repo / ".git").exists():
                    return _error_response(
                        f"Not a git repository (no .git dir): {repo_path}"
                    )
            else:
                repo = svc.config.repo_root

            files_list = [
                f.strip() for f in changed_files.split(",") if f.strip()
            ] if changed_files else []

            if not files_list:
                # Auto-detect via git
                try:
                    import subprocess
                    # repo is the safe_resolve-contained path computed above.
                    # Harden git: disable hooks + fsmonitor + system config so a
                    # hostile repo cannot execute code via the git invocation (B1).
                    result = subprocess.run(
                        [
                            "git", "-c", "core.hooksPath=/dev/null",
                            "-c", "core.fsmonitor=",
                            "diff", "--name-only", "HEAD~1", "HEAD",
                        ],
                        capture_output=True, text=True, timeout=30,
                        cwd=str(repo),
                        env={
                            **os.environ,
                            "GIT_CONFIG_NOSYSTEM": "1",
                            "GIT_TERMINAL_PROMPT": "0",
                        },
                    )
                    files_list = [
                        f.strip() for f in result.stdout.strip().split("\n")
                        if f.strip()
                    ]
                except Exception as exc:
                    return _error_response(f"Git not available or not a git repository: {exc}")

            if not files_list:
                return {
                    "success": True,
                    "files_updated": 0,
                    "nodes_added": 0,
                    "nodes_removed": 0,
                    "edges_added": 0,
                    "edges_removed": 0,
                    "duration_ms": 0,
                }

            # Re-parse changed files
            from superlocalmemory.code_graph.parser import CodeParser
            from superlocalmemory.code_graph.graph_store import GraphStore

            config = svc.config
            parser = CodeParser(config)
            store = GraphStore(db)
            # repo already contained via safe_resolve above (B1).

            nodes_before = db.get_node_count()
            edges_before = db.get_edge_count()

            for fp in files_list:
                # fp comes from caller-supplied changed_files; contain it under
                # the (already safe-resolved) repo so "../../.ssh/id_rsa" cannot
                # be read into the graph.
                try:
                    full = safe_resolve(repo, fp)
                except Exception:
                    continue
                if not full.exists():
                    store.remove_file(fp)
                    continue
                ext = full.suffix
                lang = config.extension_map.get(ext)
                if lang is None:
                    continue
                try:
                    source = full.read_bytes()
                    file_nodes, file_edges = parser.parse_file(
                        Path(fp), source, lang
                    )
                    import hashlib
                    from superlocalmemory.code_graph.models import FileRecord
                    fr = FileRecord(
                        file_path=fp,
                        content_hash=hashlib.sha256(source).hexdigest(),
                        mtime=full.stat().st_mtime,
                        language=lang,
                        node_count=len(file_nodes),
                        edge_count=len(file_edges),
                        last_indexed=time.time(),
                    )
                    store.store_file_nodes_edges(fp, file_nodes, file_edges, fr)
                except Exception as exc:
                    logger.warning("Failed to update %s: %s", fp, exc)

            duration_ms = int((time.time() - t0) * 1000)
            nodes_after = db.get_node_count()
            edges_after = db.get_edge_count()

            return {
                "success": True,
                "files_updated": len(files_list),
                "nodes_added": max(0, nodes_after - nodes_before),
                "nodes_removed": max(0, nodes_before - nodes_after),
                "edges_added": max(0, edges_after - edges_before),
                "edges_removed": max(0, edges_before - edges_after),
                "duration_ms": duration_ms,
            }
        except Exception as exc:
            logger.exception("update_code_graph failed")
            return _error_response(str(exc))

    # ==================================================================
    # Tool 3: get_blast_radius
    # ==================================================================

    @server.tool()
    async def get_blast_radius(
        changed_files: str,
        max_depth: int = 2,
        max_nodes: int = 500,
    ) -> dict:
        """Compute impact radius for changed files.

        Uses BFS in both directions (callers + callees) to find all impacted
        code entities. Essential for code review.

        Args:
            changed_files: Comma-separated file paths (relative to repo root).
            max_depth: Maximum BFS depth (default 2).
            max_nodes: Maximum nodes to return (default 500).
        """
        try:
            err = _check_graph_exists()
            if err is not None:
                return err

            files_list = [
                f.strip() for f in changed_files.split(",") if f.strip()
            ]
            if not files_list:
                return _error_response("No changed files provided.")

            db = _get_db()
            from superlocalmemory.code_graph.graph_store import GraphStore
            from superlocalmemory.code_graph.graph_engine import GraphEngine
            from superlocalmemory.code_graph.blast_radius import BlastRadius

            store = GraphStore(db)
            engine = GraphEngine(store)
            br = BlastRadius(engine)

            result = br.compute(
                changed_files=files_list,
                max_depth=max_depth,
                max_nodes=max_nodes,
            )

            return {
                "success": True,
                "changed_nodes": list(result.changed_nodes),
                "impacted_nodes": list(result.impacted_nodes),
                "impacted_files": list(result.impacted_files),
                "edges": [
                    {"source": s, "target": t, "kind": d.get("kind", "")}
                    for s, t, d in result.edges
                ],
                "depth_reached": result.depth_reached,
                "truncated": result.truncated,
            }
        except Exception as exc:
            logger.exception("get_blast_radius failed")
            return _error_response(str(exc))

    # ==================================================================
    # Tool 4: get_review_context
    # ==================================================================

    @server.tool()
    async def get_review_context(
        changed_files: str,
        include_source: bool = True,
    ) -> dict:
        """Get token-optimized review context for changed files.

        Args:
            changed_files: Comma-separated file paths.
            include_source: Whether to include source code snippets (default True).
        """
        try:
            err = _check_graph_exists()
            if err is not None:
                return err

            files_list = [
                f.strip() for f in changed_files.split(",") if f.strip()
            ]
            db = _get_db()
            from superlocalmemory.code_graph.changes import ChangeAnalyzer

            analyzer = ChangeAnalyzer(db)
            ctx = analyzer.get_review_context(files_list)

            return {
                "success": True,
                "summary": ctx.summary,
                "review_items": [
                    {
                        "node_id": n.node_id,
                        "name": n.name,
                        "kind": n.kind,
                        "file_path": n.file_path,
                        "risk_score": round(n.risk_score, 3),
                    }
                    for n in ctx.changed_nodes
                ],
                "test_gaps": [
                    {"node_id": n.node_id, "name": n.name, "file_path": n.file_path}
                    for n in ctx.test_gaps
                ],
                "risk_score": round(ctx.overall_risk, 3),
            }
        except Exception as exc:
            logger.exception("get_review_context failed")
            return _error_response(str(exc))

    # ==================================================================
    # Tool 5: query_graph
    # ==================================================================

    VALID_PATTERNS = frozenset({
        "callers_of", "callees_of", "imports_of", "imported_by",
        "tests_for", "inherits_from", "inherited_by", "contains",
    })

    @server.tool()
    async def query_graph(
        pattern: str,
        target: str = "",
        limit: int = 20,
    ) -> dict:
        """Query the code graph for relationships.

        Args:
            pattern: Query type: callers_of, callees_of, imports_of, imported_by,
                     tests_for, inherits_from, inherited_by, contains.
            target: Qualified name or partial name to match.
            limit: Maximum results (default 20).
        """
        try:
            if pattern not in VALID_PATTERNS:
                return _error_response(
                    f"Invalid pattern '{pattern}'. Must be one of: "
                    + ", ".join(sorted(VALID_PATTERNS))
                )

            err = _check_graph_exists()
            if err is not None:
                return err

            db = _get_db()
            from superlocalmemory.code_graph.graph_store import GraphStore
            from superlocalmemory.code_graph.graph_engine import GraphEngine
            from superlocalmemory.code_graph.models import EdgeKind

            store = GraphStore(db)
            engine = GraphEngine(store)

            # Resolve target to node_id
            node_id = _resolve_target(db, target)
            if node_id is None:
                return {
                    "success": True,
                    "pattern": pattern,
                    "target": target,
                    "results": [],
                    "message": f"No node found matching '{target}'.",
                }

            # Map pattern to engine query
            results: list[dict[str, Any]] = []

            edge_kind_map = {
                "callers_of": (True, {EdgeKind.CALLS.value}),
                "callees_of": (False, {EdgeKind.CALLS.value}),
                "imports_of": (False, {EdgeKind.IMPORTS.value}),
                "imported_by": (True, {EdgeKind.IMPORTS.value}),
                "inherits_from": (False, {EdgeKind.INHERITS.value}),
                "inherited_by": (True, {EdgeKind.INHERITS.value}),
                "contains": (False, {EdgeKind.CONTAINS.value}),
            }

            if pattern == "tests_for":
                raw = engine.get_tests_for(node_id)
                results = [
                    {
                        "qualified_name": r.get("qualified_name", ""),
                        "kind": r.get("kind", ""),
                        "file_path": r.get("file_path", ""),
                        "name": r.get("name", ""),
                    }
                    for r in raw[:limit]
                ]
            elif pattern in edge_kind_map:
                is_incoming, kinds = edge_kind_map[pattern]
                if is_incoming:
                    raw = engine.get_callers(node_id, edge_kinds=kinds)
                else:
                    raw = engine.get_callees(node_id, edge_kinds=kinds)

                results = [
                    {
                        "qualified_name": r["node"].get("qualified_name", ""),
                        "kind": r["node"].get("kind", ""),
                        "file_path": r["node"].get("file_path", ""),
                        "name": r["node"].get("name", ""),
                    }
                    for r in raw[:limit]
                ]

            return {
                "success": True,
                "pattern": pattern,
                "target": target,
                "results": results,
            }
        except Exception as exc:
            logger.exception("query_graph failed")
            return _error_response(str(exc))

    # ==================================================================
    # Tool 6: semantic_search_code
    # ==================================================================

    @server.tool()
    async def semantic_search_code(
        query: str,
        kind: str = "",
        limit: int = 20,
    ) -> dict:
        """Search code entities by semantic meaning using hybrid FTS5 + vector search.

        Args:
            query: Natural language query (e.g. "authentication handler").
            kind: Filter by node kind: "Function", "Class", "File", "Test", or "" for all.
            limit: Maximum results (default 20).
        """
        try:
            err = _check_graph_exists()
            if err is not None:
                return err

            db = _get_db()
            from superlocalmemory.code_graph.search import HybridSearch

            searcher = HybridSearch(db)
            raw = searcher.search(query, limit=limit * 2)

            # Kind filter
            if kind:
                kind_lower = kind.lower()
                raw = [r for r in raw if r.kind.lower() == kind_lower]

            results = [
                {
                    "qualified_name": r.qualified_name,
                    "kind": r.kind,
                    "file_path": r.file_path,
                    "score": round(r.score, 4),
                    "line_start": r.line_start,
                    "name": r.name,
                }
                for r in raw[:limit]
            ]

            return {"success": True, "results": results}
        except Exception as exc:
            logger.exception("semantic_search_code failed")
            return _error_response(str(exc))

    # ==================================================================
    # Tool 7: list_graph_stats
    # ==================================================================

    @server.tool()
    async def list_graph_stats() -> dict:
        """Get code graph size and health metrics."""
        try:
            svc = _get_service()
            if svc is None:
                return _graph_not_built_error()

            stats = svc.get_stats()

            # Count code_memory_links
            stale_links = 0
            total_links = 0
            db = _get_db()
            if db is not None:
                try:
                    rows = db.execute(
                        "SELECT COUNT(*) as cnt FROM code_memory_links", ()
                    )
                    total_links = rows[0]["cnt"] if rows else 0
                    rows = db.execute(
                        "SELECT COUNT(*) as cnt FROM code_memory_links WHERE is_stale = 1",
                        (),
                    )
                    stale_links = rows[0]["cnt"] if rows else 0
                except Exception:
                    pass

            return {
                "success": True,
                "repo_root": stats.get("repo_root", ""),
                "total_files": stats.get("files", 0),
                "total_nodes": stats.get("nodes", 0),
                "total_edges": stats.get("edges", 0),
                "total_code_memory_links": total_links,
                "stale_links": stale_links,
                "built": stats.get("built", False),
                "db_path": stats.get("db_path", ""),
            }
        except Exception as exc:
            logger.exception("list_graph_stats failed")
            return _error_response(str(exc))

    # ==================================================================
    # Tool 8: find_large_functions
    # ==================================================================

    @server.tool()
    async def find_large_functions(
        threshold: int = 50,
        limit: int = 20,
    ) -> dict:
        """Find functions exceeding a line count threshold.

        Args:
            threshold: Minimum lines to flag (default 50).
            limit: Maximum results (default 20).
        """
        try:
            err = _check_graph_exists()
            if err is not None:
                return err

            db = _get_db()
            rows = db.execute(
                """SELECT node_id, name, qualified_name, kind, file_path,
                          line_start, line_end
                   FROM graph_nodes
                   WHERE kind IN ('function', 'method')
                   AND (line_end - line_start) >= ?
                   ORDER BY (line_end - line_start) DESC
                   LIMIT ?""",
                (threshold, limit),
            )

            functions = [
                {
                    "qualified_name": r["qualified_name"],
                    "lines": r["line_end"] - r["line_start"],
                    "file_path": r["file_path"],
                    "name": r["name"],
                }
                for r in rows
            ]

            return {"success": True, "functions": functions}
        except Exception as exc:
            logger.exception("find_large_functions failed")
            return _error_response(str(exc))

    # ==================================================================
    # Tool 9: list_flows
    # ==================================================================

    @server.tool()
    async def list_flows(
        sort_by: str = "criticality",
        limit: int = 20,
    ) -> dict:
        """List execution flows sorted by criticality or size.

        Args:
            sort_by: "criticality" (default) or "size".
            limit: Maximum flows to return (default 20).
        """
        try:
            err = _check_graph_exists()
            if err is not None:
                return err

            db = _get_db()
            from superlocalmemory.code_graph.flows import FlowDetector

            detector = FlowDetector(db)
            entries = detector.detect_entry_points()[:50]

            flows = []
            for ep in entries:
                try:
                    flow = detector.trace_flow(ep)
                    if flow is not None:
                        flows.append(flow)
                except Exception:
                    pass

            if sort_by == "size":
                flows.sort(key=lambda f: -f.node_count)
            else:
                flows.sort(key=lambda f: -f.criticality)

            result = [
                {
                    "name": f.name,
                    "entry_point": f.entry_node_id,
                    "depth": f.depth,
                    "node_count": f.node_count,
                    "file_count": f.file_count,
                    "criticality": round(f.criticality, 3),
                }
                for f in flows[:limit]
            ]

            return {"success": True, "flows": result}
        except Exception as exc:
            logger.exception("list_flows failed")
            return _error_response(str(exc))

    # ==================================================================
    # Tool 10: get_flow
    # ==================================================================

    @server.tool()
    async def get_flow(
        flow_name: str,
    ) -> dict:
        """Get detailed information about a single execution flow.

        Args:
            flow_name: The flow name or entry point name.
        """
        try:
            err = _check_graph_exists()
            if err is not None:
                return err

            db = _get_db()
            from superlocalmemory.code_graph.flows import FlowDetector

            detector = FlowDetector(db)
            entries = detector.detect_entry_points()

            for ep in entries:
                try:
                    flow = detector.trace_flow(ep)
                    if flow is not None and (
                        flow.name == flow_name
                        or flow.entry_node_id == flow_name
                    ):
                        return {
                            "success": True,
                            "flow": {
                                "name": flow.name,
                                "entry_point": flow.entry_node_id,
                                "depth": flow.depth,
                                "node_count": flow.node_count,
                                "file_count": flow.file_count,
                                "criticality": round(flow.criticality, 3),
                                "path": list(flow.path_node_ids),
                            },
                        }
                except Exception:
                    pass

            return _error_response(f"Flow '{flow_name}' not found.")
        except Exception as exc:
            logger.exception("get_flow failed")
            return _error_response(str(exc))

    # ==================================================================
    # Tool 11: get_affected_flows
    # ==================================================================

    @server.tool()
    async def get_affected_flows(
        changed_files: str,
    ) -> dict:
        """Find execution flows impacted by file changes.

        Args:
            changed_files: Comma-separated file paths.
        """
        try:
            err = _check_graph_exists()
            if err is not None:
                return err

            files_list = [
                f.strip() for f in changed_files.split(",") if f.strip()
            ]
            if not files_list:
                return _error_response("No changed files provided.")

            db = _get_db()
            from superlocalmemory.code_graph.flows import FlowDetector

            # Get all changed node IDs
            changed_node_ids: set[str] = set()
            for fp in files_list:
                rows = db.execute(
                    "SELECT node_id FROM graph_nodes WHERE file_path = ?",
                    (fp,),
                )
                changed_node_ids.update(r["node_id"] for r in rows)

            detector = FlowDetector(db)
            entries = detector.detect_entry_points()[:50]

            affected = []
            for ep in entries:
                try:
                    flow = detector.trace_flow(ep)
                    if flow is None:
                        continue
                    overlap = changed_node_ids.intersection(flow.path_node_ids)
                    if overlap:
                        affected.append({
                            "name": flow.name,
                            "criticality": round(flow.criticality, 3),
                            "affected_nodes": list(overlap),
                        })
                except Exception:
                    pass

            return {"success": True, "affected_flows": affected}
        except Exception as exc:
            logger.exception("get_affected_flows failed")
            return _error_response(str(exc))

    # ==================================================================
    # Tool 12: list_communities
    # ==================================================================

    @server.tool()
    async def list_communities(
        sort_by: str = "cohesion",
        limit: int = 20,
    ) -> dict:
        """List detected code communities (clusters of related code).

        Args:
            sort_by: "cohesion" (default) or "size".
            limit: Maximum communities (default 20).
        """
        try:
            err = _check_graph_exists()
            if err is not None:
                return err

            db = _get_db()
            from superlocalmemory.code_graph.communities import CommunityDetector

            detector = CommunityDetector(db)
            comms = detector.detect_communities()

            if sort_by == "size":
                comms.sort(key=lambda c: -c.size)
            else:
                comms.sort(key=lambda c: -c.cohesion)

            result = [
                {
                    "community_id": c.community_id,
                    "name": c.name,
                    "size": c.size,
                    "cohesion": round(c.cohesion, 3),
                    "dominant_language": c.dominant_language,
                }
                for c in comms[:limit]
            ]

            return {"success": True, "communities": result}
        except Exception as exc:
            logger.exception("list_communities failed")
            return _error_response(str(exc))

    # ==================================================================
    # Tool 13: get_community
    # ==================================================================

    @server.tool()
    async def get_community(
        community_id: int,
    ) -> dict:
        """Get detailed information about a single code community.

        Args:
            community_id: The community ID.
        """
        try:
            err = _check_graph_exists()
            if err is not None:
                return err

            db = _get_db()
            from superlocalmemory.code_graph.communities import CommunityDetector

            detector = CommunityDetector(db)
            comms = detector.detect_communities()

            for c in comms:
                if c.community_id == community_id:
                    return {
                        "success": True,
                        "community": {
                            "community_id": c.community_id,
                            "name": c.name,
                            "size": c.size,
                            "cohesion": round(c.cohesion, 3),
                            "dominant_language": c.dominant_language,
                            "directory": c.directory,
                            "file_count": c.file_count,
                            "members": list(c.node_ids[:100]),
                        },
                    }

            return _error_response(
                f"Community {community_id} not found."
            )
        except Exception as exc:
            logger.exception("get_community failed")
            return _error_response(str(exc))

    # ==================================================================
    # Tool 14: get_architecture_overview
    # ==================================================================

    @server.tool()
    async def get_architecture_overview() -> dict:
        """Get high-level architecture map showing communities and their relationships."""
        try:
            err = _check_graph_exists()
            if err is not None:
                return err

            db = _get_db()
            from superlocalmemory.code_graph.communities import CommunityDetector

            detector = CommunityDetector(db)
            overview = detector.get_architecture_overview()

            communities = [
                {
                    "community_id": c.community_id,
                    "name": c.name,
                    "size": c.size,
                    "cohesion": round(c.cohesion, 3),
                }
                for c in overview.communities
            ]

            warnings = [
                {
                    "from": w.source_community,
                    "to": w.target_community,
                    "edge_count": w.edge_count,
                    "severity": w.severity,
                }
                for w in overview.coupling_warnings
            ]

            return {
                "success": True,
                "communities": communities,
                "cross_community_edges": warnings,
                "total_nodes": overview.total_nodes,
                "total_communities": overview.total_communities,
            }
        except Exception as exc:
            logger.exception("get_architecture_overview failed")
            return _error_response(str(exc))

    # ==================================================================
    # Tool 15: detect_changes
    # ==================================================================

    @server.tool()
    async def detect_changes(
        base: str = "HEAD~1",
    ) -> dict:
        """Detect code changes and compute risk scores.

        Args:
            base: Git ref to diff against (default "HEAD~1").
        """
        try:
            err = _check_graph_exists()
            if err is not None:
                return err

            db = _get_db()
            svc = _get_service()
            from superlocalmemory.code_graph.changes import ChangeAnalyzer

            try:
                hunks = ChangeAnalyzer.parse_git_diff(svc.config.repo_root, base)
            except Exception as exc:
                return _error_response(
                    f"Git not available or not a git repository: {exc}"
                )

            changed_files = list({h.file_path for h in hunks})
            analyzer = ChangeAnalyzer(db)
            ctx = analyzer.analyze_changes(changed_files)

            return {
                "success": True,
                "summary": ctx.summary,
                "risk_score": round(ctx.overall_risk, 3),
                "changed_functions": [
                    {
                        "name": n.name,
                        "kind": n.kind,
                        "file_path": n.file_path,
                        "risk_score": round(n.risk_score, 3),
                    }
                    for n in ctx.changed_nodes
                ],
                "test_gaps": [
                    {"name": n.name, "file_path": n.file_path}
                    for n in ctx.test_gaps
                ],
                "review_priorities": [
                    {"name": n.name, "risk_score": round(n.risk_score, 3)}
                    for n in ctx.review_priorities
                ],
            }
        except Exception as exc:
            logger.exception("detect_changes failed")
            return _error_response(str(exc))

    # ==================================================================
    # Tool 16: refactor_preview
    # ==================================================================

    @server.tool()
    async def refactor_preview(
        action: str,
        target: str,
        new_name: str = "",
    ) -> dict:
        """Preview a refactoring operation without executing it.

        Args:
            action: "rename" | "find_dead_code" | "find_duplicates"
            target: Qualified name or partial name of the target.
            new_name: New name (required for "rename" action).
        """
        try:
            err = _check_graph_exists()
            if err is not None:
                return err

            if action == "rename" and not new_name:
                return _error_response(
                    "new_name is required for rename action."
                )

            db = _get_db()

            if action == "rename":
                node_id = _resolve_target(db, target)
                if node_id is None:
                    return _error_response(f"Target '{target}' not found.")

                # Find all references
                from superlocalmemory.code_graph.graph_store import GraphStore
                from superlocalmemory.code_graph.graph_engine import GraphEngine

                store = GraphStore(db)
                engine = GraphEngine(store)
                callers = engine.get_callers(node_id)

                affected_files = set()
                node_data = engine.get_node_data(node_id)
                affected_files.add(node_data.get("file_path", ""))
                for c in callers:
                    affected_files.add(c["node"].get("file_path", ""))

                return {
                    "success": True,
                    "action": "rename",
                    "affected_files": list(affected_files),
                    "affected_references": len(callers),
                    "estimated_changes": len(callers) + 1,
                }

            elif action == "find_dead_code":
                from superlocalmemory.code_graph.graph_store import GraphStore
                from superlocalmemory.code_graph.graph_engine import GraphEngine

                store = GraphStore(db)
                engine = GraphEngine(store)
                graph = engine.graph
                index = engine.index

                dead = []
                for node_id_str, rx_idx in index.id_to_rx.items():
                    data = graph[rx_idx]
                    if data.get("kind") in ("function", "method"):
                        in_edges = list(graph.in_edges(rx_idx))
                        # No callers and not a test and not an entry point
                        if not in_edges and not data.get("is_test"):
                            dead.append({
                                "qualified_name": data.get("qualified_name", ""),
                                "file_path": data.get("file_path", ""),
                                "kind": data.get("kind", ""),
                            })

                return {
                    "success": True,
                    "action": "find_dead_code",
                    "affected_files": list({d["file_path"] for d in dead}),
                    "affected_references": dead[:50],
                    "estimated_changes": len(dead),
                }

            return _error_response(
                f"Unsupported action: {action}. Use rename or find_dead_code."
            )
        except Exception as exc:
            logger.exception("refactor_preview failed")
            return _error_response(str(exc))

    # ==================================================================
    # Tool 17: apply_refactor
    # ==================================================================

    @server.tool()
    async def apply_refactor(
        action: str,
        target: str,
        new_name: str = "",
        dry_run: bool = True,
    ) -> dict:
        """Execute a refactoring operation (stub for MVP).

        Args:
            action: "rename" (only supported action currently).
            target: Qualified name of the target to rename.
            new_name: New name.
            dry_run: If True (default), show what would change without modifying files.
        """
        try:
            # MVP stub — always dry_run
            preview = await refactor_preview(action, target, new_name)
            if not preview.get("success"):
                return preview

            preview["dry_run"] = True
            preview["message"] = (
                "Apply refactor is a stub in MVP. "
                "Use the preview to guide manual refactoring."
            )
            return preview
        except Exception as exc:
            logger.exception("apply_refactor failed")
            return _error_response(str(exc))

    # ==================================================================
    # Tool 18 (BRIDGE): code_memory_search
    # ==================================================================

    @server.tool()
    async def code_memory_search(
        code_entity: str,
        link_type: str = "",
        limit: int = 10,
    ) -> dict:
        """Search SLM memories linked to a code entity.

        BRIDGE TOOL: Combines code graph structure with SLM memory content.

        Args:
            code_entity: Function/class/file name or qualified name.
            link_type: Filter by link type. Empty = all.
            limit: Maximum results (default 10).
        """
        try:
            err = _check_graph_exists()
            if err is not None:
                return err

            db = _get_db()
            node_id = _resolve_target(db, code_entity)
            if node_id is None:
                return {
                    "success": True,
                    "code_entity": code_entity,
                    "matched_node": None,
                    "memories": [],
                }

            # Get links
            links = db.get_links_for_node(node_id)
            if link_type:
                links = [lnk for lnk in links if lnk.link_type.value == link_type]

            # Get node data
            node = db.get_node(node_id)
            node_info = {
                "node_id": node.node_id,
                "name": node.name,
                "qualified_name": node.qualified_name,
                "kind": node.kind.value,
                "file_path": node.file_path,
            } if node else None

            memories = [
                {
                    "fact_id": lnk.slm_fact_id,
                    "link_type": lnk.link_type.value,
                    "confidence": round(lnk.confidence, 3),
                    "created_at": lnk.created_at,
                    "is_stale": lnk.is_stale,
                }
                for lnk in links[:limit]
            ]

            return {
                "success": True,
                "code_entity": code_entity,
                "matched_node": node_info,
                "memories": memories,
            }
        except Exception as exc:
            logger.exception("code_memory_search failed")
            return _error_response(str(exc))

    # ==================================================================
    # Tool 19 (BRIDGE): code_entity_history
    # ==================================================================

    @server.tool()
    async def code_entity_history(
        code_entity: str,
    ) -> dict:
        """Get the complete memory timeline for a code entity.

        BRIDGE TOOL: Shows all memories about a function/class ordered by time.

        Args:
            code_entity: Function/class/file name or qualified name.
        """
        try:
            err = _check_graph_exists()
            if err is not None:
                return err

            db = _get_db()
            node_id = _resolve_target(db, code_entity)
            if node_id is None:
                return {
                    "success": True,
                    "code_entity": code_entity,
                    "node": None,
                    "timeline": [],
                    "total_memories": 0,
                    "stale_count": 0,
                }

            node = db.get_node(node_id)
            links = db.get_links_for_node(node_id)

            # Sort by created_at
            links_sorted = sorted(links, key=lambda lnk: lnk.created_at)
            stale_count = sum(1 for lnk in links if lnk.is_stale)

            node_info = {
                "node_id": node.node_id,
                "name": node.name,
                "qualified_name": node.qualified_name,
                "kind": node.kind.value,
                "file_path": node.file_path,
            } if node else None

            timeline = [
                {
                    "fact_id": lnk.slm_fact_id,
                    "link_type": lnk.link_type.value,
                    "created_at": lnk.created_at,
                    "is_stale": lnk.is_stale,
                    "confidence": round(lnk.confidence, 3),
                }
                for lnk in links_sorted
            ]

            return {
                "success": True,
                "code_entity": code_entity,
                "node": node_info,
                "timeline": timeline,
                "total_memories": len(links),
                "stale_count": stale_count,
            }
        except Exception as exc:
            logger.exception("code_entity_history failed")
            return _error_response(str(exc))

    # ==================================================================
    # Tool 20 (BRIDGE): enrich_blast_radius
    # ==================================================================

    @server.tool()
    async def enrich_blast_radius(
        changed_files: str,
        max_depth: int = 2,
    ) -> dict:
        """Compute blast radius PLUS institutional memory for each impacted node.

        BRIDGE TOOL: Returns impact analysis enriched with relevant SLM memories.

        Args:
            changed_files: Comma-separated file paths.
            max_depth: BFS depth for blast radius (default 2).
        """
        try:
            err = _check_graph_exists()
            if err is not None:
                return err

            files_list = [
                f.strip() for f in changed_files.split(",") if f.strip()
            ]
            if not files_list:
                return _error_response("No changed files provided.")

            db = _get_db()
            from superlocalmemory.code_graph.graph_store import GraphStore
            from superlocalmemory.code_graph.graph_engine import GraphEngine
            from superlocalmemory.code_graph.blast_radius import BlastRadius

            store = GraphStore(db)
            engine = GraphEngine(store)
            br = BlastRadius(engine)
            result = br.compute(changed_files=files_list, max_depth=max_depth)

            # Enrich each impacted node with memories
            total_memories = 0
            impacted = []
            for nid in result.impacted_nodes:
                links = db.get_links_for_node(nid)
                try:
                    data = engine.get_node_data(nid)
                except Exception:
                    data = {}
                memories = [
                    {
                        "fact_id": lnk.slm_fact_id,
                        "link_type": lnk.link_type.value,
                        "is_stale": lnk.is_stale,
                    }
                    for lnk in links[:5]
                ]
                total_memories += len(memories)
                impacted.append({
                    "qualified_name": data.get("qualified_name", nid),
                    "kind": data.get("kind", ""),
                    "file_path": data.get("file_path", ""),
                    "memories": memories,
                    "memory_count": len(links),
                })

            return {
                "success": True,
                "changed_nodes": list(result.changed_nodes),
                "impacted_nodes": impacted,
                "total_memories_surfaced": total_memories,
            }
        except Exception as exc:
            logger.exception("enrich_blast_radius failed")
            return _error_response(str(exc))

    # ==================================================================
    # Tool 21 (BRIDGE): code_stale_check
    # ==================================================================

    @server.tool()
    async def code_stale_check(
        scope: str = "all",
    ) -> dict:
        """Find SLM memories that reference deleted or changed code.

        BRIDGE TOOL: Identifies memories that may be outdated.

        Args:
            scope: "all" (check everything) or a file path to scope the check.
        """
        try:
            err = _check_graph_exists()
            if err is not None:
                return err

            db = _get_db()

            if scope == "all":
                rows = db.execute(
                    """SELECT cml.link_id, cml.slm_fact_id, cml.code_node_id,
                              cml.is_stale, cml.last_verified,
                              gn.qualified_name, gn.file_path
                       FROM code_memory_links cml
                       LEFT JOIN graph_nodes gn ON cml.code_node_id = gn.node_id
                       WHERE cml.is_stale = 1""",
                    (),
                )
            else:
                rows = db.execute(
                    """SELECT cml.link_id, cml.slm_fact_id, cml.code_node_id,
                              cml.is_stale, cml.last_verified,
                              gn.qualified_name, gn.file_path
                       FROM code_memory_links cml
                       LEFT JOIN graph_nodes gn ON cml.code_node_id = gn.node_id
                       WHERE cml.is_stale = 1 AND gn.file_path = ?""",
                    (scope,),
                )

            stale_memories = [
                {
                    "fact_id": r["slm_fact_id"],
                    "code_entity": r["qualified_name"] or r["code_node_id"],
                    "reason": "Code entity deleted or changed",
                    "stale_since": r["last_verified"] or "",
                }
                for r in rows
            ]

            # Total links count
            total_rows = db.execute(
                "SELECT COUNT(*) as cnt FROM code_memory_links", ()
            )
            total_links = total_rows[0]["cnt"] if total_rows else 0

            return {
                "success": True,
                "stale_memories": stale_memories,
                "total_stale": len(stale_memories),
                "total_links": total_links,
            }
        except Exception as exc:
            logger.exception("code_stale_check failed")
            return _error_response(str(exc))

    # ==================================================================
    # Tool 22 (BRIDGE): link_memory_to_code
    # ==================================================================

    VALID_LINK_TYPES = frozenset({
        "mentions", "decision_about", "bug_fix", "refactor", "design_rationale",
    })

    @server.tool()
    async def link_memory_to_code(
        fact_id: str,
        code_entity: str,
        link_type: str = "mentions",
    ) -> dict:
        """Manually link an SLM memory to a code graph node.

        BRIDGE TOOL: Creates an explicit link between a memory and a code entity.

        Args:
            fact_id: The SLM atomic fact ID.
            code_entity: Function/class/file qualified name or partial name.
            link_type: One of: mentions, decision_about, bug_fix, refactor, design_rationale.
        """
        try:
            if link_type not in VALID_LINK_TYPES:
                return _error_response(
                    f"Invalid link_type '{link_type}'. Must be one of: "
                    + ", ".join(sorted(VALID_LINK_TYPES))
                )

            err = _check_graph_exists()
            if err is not None:
                return err

            db = _get_db()
            node_id = _resolve_target(db, code_entity)
            if node_id is None:
                return _error_response(
                    f"Code entity '{code_entity}' not found in graph."
                )

            from superlocalmemory.code_graph.models import CodeMemoryLink, LinkType
            from superlocalmemory.storage.models import _new_id
            from datetime import datetime, timezone

            now_str = datetime.now(timezone.utc).isoformat()
            link = CodeMemoryLink(
                link_id=_new_id(),
                code_node_id=node_id,
                slm_fact_id=fact_id,
                link_type=LinkType(link_type),
                confidence=1.0,
                created_at=now_str,
                last_verified=now_str,
                is_stale=False,
            )
            db.upsert_link(link)

            node = db.get_node(node_id)
            node_info = {
                "node_id": node.node_id,
                "name": node.name,
                "qualified_name": node.qualified_name,
            } if node else {"node_id": node_id}

            return {
                "success": True,
                "link_id": link.link_id,
                "code_node": node_info,
                "confidence": 1.0,
            }
        except Exception as exc:
            logger.exception("link_memory_to_code failed")
            return _error_response(str(exc))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_target(db, target: str) -> str | None:
    """Resolve a target name to a node_id. Returns None if not found."""
    if not target:
        return None

    # Try exact qualified_name match
    node = db.get_node_by_qualified_name(target)
    if node is not None:
        return node.node_id

    # Try exact node_id match
    node = db.get_node(target)
    if node is not None:
        return node.node_id

    # Try LIKE match on name or qualified_name
    rows = db.execute(
        """SELECT node_id FROM graph_nodes
           WHERE name = ? OR qualified_name LIKE ?
           LIMIT 1""",
        (target, f"%{target}%"),
    )
    if rows:
        return rows[0]["node_id"]

    return None
