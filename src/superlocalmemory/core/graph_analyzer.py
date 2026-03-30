# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under the MIT License - see LICENSE file
# Part of SuperLocalMemory V3

"""Graph structural analysis -- PageRank, community detection, centrality.

Reads BOTH graph_edges and association_edges for the full graph picture.
Stores results in fact_importance table.
Called during consolidation (Phase 5), not at query time.

Part of Qualixar | Author: Varun Pratap Bhardwaj
License: MIT
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class GraphAnalyzer:
    """Compute structural importance metrics for the memory graph.

    - PageRank: global structural importance via networkx
    - Community detection: Label Propagation via networkx
    - Degree centrality: connection count normalization

    Reads BOTH graph_edges and association_edges (Rule 13).
    Stores results in fact_importance table.
    """

    def __init__(self, db: Any) -> None:
        self._db = db

    def compute_and_store(self, profile_id: str) -> dict[str, Any]:
        """Run all analyses and persist to fact_importance.

        Returns summary dict with node_count, community_count, top_5_nodes.
        """
        try:
            graph = self._build_networkx_graph(profile_id)
            if graph.number_of_nodes() == 0:
                return {
                    "node_count": 0,
                    "edge_count": 0,
                    "community_count": 0,
                    "top_5_nodes": [],
                }

            pagerank = self.compute_pagerank(graph)
            communities = self.detect_communities(graph)
            centrality = self._compute_degree_centrality(graph)

            # Persist to fact_importance
            for node_id in graph.nodes():
                pr_score = pagerank.get(node_id, 0.0)
                comm_id = communities.get(node_id)
                deg_cent = centrality.get(node_id, 0.0)
                self._db.execute(
                    "INSERT OR REPLACE INTO fact_importance "
                    "(fact_id, profile_id, pagerank_score, community_id, "
                    " degree_centrality, computed_at) "
                    "VALUES (?, ?, ?, ?, ?, datetime('now'))",
                    (node_id, profile_id, round(pr_score, 6),
                     comm_id, round(deg_cent, 4)),
                )

            top_5 = sorted(
                pagerank.items(), key=lambda x: x[1], reverse=True,
            )[:5]
            unique_communities = len(
                set(c for c in communities.values() if c is not None),
            )

            return {
                "node_count": graph.number_of_nodes(),
                "edge_count": graph.number_of_edges(),
                "community_count": unique_communities,
                "top_5_nodes": [
                    (nid, round(score, 4)) for nid, score in top_5
                ],
            }
        except Exception as exc:
            logger.debug("GraphAnalyzer.compute_and_store failed: %s", exc)
            return {
                "node_count": 0,
                "edge_count": 0,
                "community_count": 0,
                "top_5_nodes": [],
            }

    def compute_pagerank(
        self,
        graph: Any = None,
        profile_id: str = "",
        alpha: float = 0.85,
    ) -> dict[str, float]:
        """Compute PageRank using networkx.

        alpha = damping factor (0.85 is standard).
        """
        import networkx as nx

        if graph is None:
            graph = self._build_networkx_graph(profile_id)
        if graph.number_of_nodes() == 0:
            return {}
        try:
            return nx.pagerank(graph, alpha=alpha, weight="weight")
        except nx.PowerIterationFailedConvergence:
            return nx.pagerank(graph, alpha=alpha, weight=None)

    def detect_communities(
        self,
        graph: Any = None,
        profile_id: str = "",
    ) -> dict[str, int]:
        """Detect communities via Label Propagation.

        O(m) where m = edges (fast), no parameter tuning needed.
        """
        import networkx as nx
        from networkx.algorithms.community import (
            label_propagation_communities,
        )

        if graph is None:
            graph = self._build_networkx_graph(profile_id)
        if graph.number_of_nodes() == 0:
            return {}

        # Label propagation needs undirected graph
        undirected = graph.to_undirected()
        communities_gen = label_propagation_communities(undirected)
        result: dict[str, int] = {}
        for comm_id, community in enumerate(communities_gen):
            for node in community:
                result[node] = comm_id
        return result

    def _compute_degree_centrality(
        self, graph: Any,
    ) -> dict[str, float]:
        """Degree centrality: fraction of nodes each node connects to."""
        import networkx as nx

        if graph.number_of_nodes() <= 1:
            return {n: 0.0 for n in graph.nodes()}
        return nx.degree_centrality(graph)

    def _build_networkx_graph(self, profile_id: str) -> Any:
        """Build networkx DiGraph from BOTH graph_edges + association_edges."""
        import networkx as nx

        g = nx.DiGraph()

        # graph_edges
        try:
            rows = self._db.execute(
                "SELECT source_id, target_id, weight, edge_type "
                "FROM graph_edges WHERE profile_id = ?",
                (profile_id,),
            )
            for row in rows:
                d = dict(row)
                g.add_edge(
                    d["source_id"], d["target_id"],
                    weight=d["weight"], edge_type=d["edge_type"],
                )
        except Exception as exc:
            logger.debug("graph_edges read failed: %s", exc)

        # association_edges
        try:
            rows = self._db.execute(
                "SELECT source_fact_id, target_fact_id, weight, "
                "       association_type "
                "FROM association_edges WHERE profile_id = ?",
                (profile_id,),
            )
            for row in rows:
                d = dict(row)
                src, tgt = d["source_fact_id"], d["target_fact_id"]
                if g.has_edge(src, tgt):
                    existing_w = g[src][tgt].get("weight", 0)
                    if d["weight"] > existing_w:
                        g[src][tgt]["weight"] = d["weight"]
                else:
                    g.add_edge(
                        src, tgt,
                        weight=d["weight"],
                        edge_type=d["association_type"],
                    )
        except Exception as exc:
            logger.debug("association_edges read failed: %s", exc)

        return g
