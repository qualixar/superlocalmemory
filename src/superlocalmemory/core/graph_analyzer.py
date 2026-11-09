# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3

"""Graph structural analysis -- PageRank, community detection, centrality.

Reads BOTH graph_edges and association_edges for the full graph picture.
Stores results in fact_importance table.
Called during consolidation (Phase 5), not at query time.

v3.4.1: Added Leiden community detection (optional), TF-IDF community labels,
bridge score detection. Frontend uses Louvain; backend uses Leiden/LP.

Part of Qualixar | Author: Varun Pratap Bhardwaj
License: AGPL-3.0-or-later
"""

from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict
from math import log
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
        """Recompute structural importance for a profile and persist it.

        The computation and the write now live in ``core/graph_metrics``, which
        is the only thing that writes ``fact_importance``. This method keeps its
        shape for its three callers and adds the community labels the dashboard
        reads, which are a presentation concern and not part of the ranking
        signal.

        Two behaviours changed with the move, both deliberate. Isolated facts now
        get a row instead of being skipped, so "no position in the graph" stops
        looking like "not computed yet" to the ranker. And the graph is the
        *visible* graph: the previous version read ``atomic_facts`` and
        ``graph_edges`` raw, so on the author's store it ranked 1,299 facts the
        store is not allowed to return, diluting every real score.
        """
        from superlocalmemory.core.graph_metrics import compute_graph_metrics

        report = compute_graph_metrics(self._db, profile_id)
        if not report.ok:
            logger.warning("Graph metrics: %s", report.summary())
            return {
                "node_count": 0,
                "edge_count": report.edges,
                "community_count": 0,
                "top_5_nodes": [],
                "error": report.error,
            }

        communities: dict[str, int] = {}
        top_5: list[tuple[str, float]] = []
        bridge_count = 0
        top_bridges: list[tuple[str, float]] = []
        try:
            rows = self._db.execute(
                "SELECT fact_id, pagerank_score, community_id, bridge_score "
                "FROM fact_importance WHERE profile_id = ? "
                "ORDER BY pagerank_score DESC",
                (profile_id,),
            )
            ranked = [dict(row) for row in rows]
            for row in ranked:
                if row.get("community_id") is not None:
                    communities[row["fact_id"]] = int(row["community_id"])
            top_5 = [
                (row["fact_id"], round(float(row["pagerank_score"] or 0.0), 4))
                for row in ranked[:5]
            ]
            bridges = [
                (row["fact_id"], float(row.get("bridge_score") or 0.0))
                for row in ranked
            ]
            bridge_count = len([1 for _fid, score in bridges if score > 0.1])
            top_bridges = sorted(bridges, key=lambda item: -item[1])[:5]
        except Exception as exc:
            logger.debug("Reading back graph metrics failed: %s", exc)

        labels: dict[int, str] = {}
        if communities:
            try:
                labels = self.compute_community_labels(profile_id, communities)
                from superlocalmemory.infra.data_root import canonical_data_root
                labels_dir = canonical_data_root()
                labels_dir.mkdir(parents=True, exist_ok=True)
                (labels_dir / f"{profile_id}_community_labels.json").write_text(
                    json.dumps(labels, indent=2),
                    encoding="utf-8",
                )
            except Exception as exc:
                logger.debug("Community labels skipped: %s", exc)

        logger.info("GraphAnalyzer: %s", report.summary())
        return {
            "node_count": report.written,
            "edge_count": report.edges,
            "community_count": report.communities,
            "top_5_nodes": top_5,
            "bridge_count": bridge_count,
            "top_bridge_nodes": [
                (fid, round(score, 4)) for fid, score in top_bridges
            ],
            "community_labels": labels,
            "engine": report.engine,
            "isolated_facts": report.isolated,
            "duration_ms": report.duration_ms,
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

    def detect_communities_louvain(
        self,
        graph: Any = None,
        profile_id: str = "",
    ) -> dict[str, int]:
        """Detect communities via Louvain (modularity-optimizing).

        Higher quality than Label Propagation (deterministic with a seed,
        no giant-community collapse), pure-Python via networkx — no extra
        binary deps. Falls back to Label Propagation if unavailable.
        """

        if graph is None:
            graph = self._build_networkx_graph(profile_id)
        if graph.number_of_nodes() == 0:
            return {}

        undirected = graph.to_undirected()
        try:
            from networkx.algorithms.community import louvain_communities

            communities = louvain_communities(
                undirected, weight="weight", seed=42,
            )
        except Exception as exc:
            logger.debug(
                "Louvain unavailable/failed (%s); using Label Propagation", exc,
            )
            return self.detect_communities(graph, profile_id)

        result: dict[str, int] = {}
        for comm_id, community in enumerate(communities):
            for node in community:
                result[node] = comm_id
        return result

    # ── v3.4.1: Leiden Community Detection ────────────────────────

    def detect_communities_leiden(
        self,
        graph: Any = None,
        profile_id: str = "",
        resolution: float = 1.0,
    ) -> dict[str, int]:
        """Leiden community detection (higher quality than Label Propagation).

        Falls back to detect_communities() (Label Propagation) if
        leidenalg or igraph are not installed.
        """
        if graph is None:
            graph = self._build_networkx_graph(profile_id)
        if graph.number_of_nodes() == 0:
            return {}

        try:
            import igraph
            import leidenalg
        except ImportError:
            logger.info(
                "leidenalg not installed, using Louvain fallback",
            )
            return self.detect_communities_louvain(graph, profile_id)

        # Convert DiGraph -> undirected -> igraph
        undirected = graph.to_undirected()
        node_list = list(undirected.nodes())
        node_index = {n: i for i, n in enumerate(node_list)}

        ig = igraph.Graph(n=len(node_list), directed=False)
        edges = []
        weights = []
        for u, v in undirected.edges():
            if u in node_index and v in node_index:
                edges.append((node_index[u], node_index[v]))
                weights.append(undirected[u][v].get("weight", 1.0))

        ig.add_edges(edges)
        ig.es["weight"] = weights
        ig.simplify(combine_edges={"weight": "max"})

        partition = leidenalg.find_partition(
            ig,
            leidenalg.RBConfigurationVertexPartition,
            resolution_parameter=resolution,
            weights="weight",
        )

        result: dict[str, int] = {}
        for idx, comm_id in enumerate(partition.membership):
            result[node_list[idx]] = comm_id

        logger.info(
            "Leiden detected %d communities (resolution=%.1f)",
            len(set(result.values())), resolution,
        )
        return result

    # ── v3.4.1: TF-IDF Community Labels ─────────────────────────

    def compute_community_labels(
        self,
        profile_id: str,
        communities: dict[str, int],
    ) -> dict[int, str]:
        """Generate human-readable labels via TF-IDF on fact content.

        Returns dict mapping community_id to label string.
        Labels stored in config table for API access.
        """
        if not communities:
            return {}

        # Group fact_ids by community
        comm_facts: dict[int, list[str]] = defaultdict(list)
        for fact_id, comm_id in communities.items():
            comm_facts[comm_id].append(fact_id)

        stopwords = frozenset({
            "the", "a", "an", "is", "was", "were", "are", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "shall", "can",
            "to", "of", "in", "for", "on", "with", "at", "by", "from",
            "as", "into", "through", "during", "before", "after", "above",
            "below", "between", "and", "but", "or", "not", "no", "nor",
            "so", "yet", "both", "either", "neither", "this", "that",
            "these", "those", "it", "its", "they", "them", "their",
            "he", "she", "his", "her", "we", "our", "you", "your",
            "i", "my", "me",
        })

        # Fetch content for each community
        tf_per_comm: dict[int, Counter] = {}
        for comm_id, fact_ids in comm_facts.items():
            placeholders = ",".join("?" * len(fact_ids))
            sql = (
                "SELECT content FROM atomic_facts WHERE fact_id IN ("
                + placeholders
                + ") AND profile_id = ?"
            )
            try:
                rows = self._db.execute(sql, (*fact_ids, profile_id))
                texts = [dict(r).get("content", "") for r in rows]
            except Exception:
                texts = []

            tokens: list[str] = []
            for text in texts:
                for word in text.lower().split():
                    w = word.strip(".,;:!?\"'()[]{}")
                    if len(w) > 2 and w not in stopwords:
                        tokens.append(w)
            tf_per_comm[comm_id] = Counter(tokens)

        num_communities = len(comm_facts)
        labels: dict[int, str] = {}

        if num_communities == 1:
            # Single community: use raw term frequency
            for comm_id, tf in tf_per_comm.items():
                top = [w for w, _ in tf.most_common(3)]
                labels[comm_id] = ", ".join(top) if top else f"Community {comm_id}"
        else:
            # Compute IDF across communities
            doc_freq: Counter = Counter()
            for tf in tf_per_comm.values():
                for term in tf:
                    doc_freq[term] += 1

            for comm_id, tf in tf_per_comm.items():
                scored = []
                for term, count in tf.items():
                    idf = log(1 + num_communities / (1 + doc_freq[term]))
                    scored.append((term, count * idf))
                scored.sort(key=lambda x: x[1], reverse=True)
                top = [w for w, _ in scored[:3]]
                labels[comm_id] = ", ".join(top) if top else f"Community {comm_id}"

        # Store in config table
        try:
            key = "community_labels_" + profile_id
            value = json.dumps(labels)
            self._db.execute(
                "INSERT OR REPLACE INTO config (key, value, updated_at) "
                "VALUES (?, ?, datetime('now'))",
                (key, value),
            )
        except Exception as exc:
            logger.warning("Failed to store community labels: %s", exc)

        return labels

    # ── v3.4.1: Bridge Score Detection ───────────────────────────

    def compute_bridge_scores(self, graph: Any) -> dict[str, float]:
        """Identify bridge nodes via betweenness centrality.

        Returns dict mapping node_id to bridge_score (0.0 to 1.0).
        NOT persisted to DB (no column exists) -- used in summary only.
        """
        import networkx as nx

        if graph.number_of_nodes() <= 2:
            return {}
        return nx.betweenness_centrality(
            graph, weight="weight", normalized=True,
        )

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

        # v3.4.7: Add ALL facts as nodes so isolated facts get base PageRank.
        # Previously, facts without edges were invisible to graph analysis.
        try:
            all_facts = self._db.execute(
                "SELECT fact_id FROM atomic_facts WHERE profile_id = ?",
                (profile_id,),
            )
            for row in all_facts:
                fact_id = dict(row)["fact_id"]
                if fact_id not in g:
                    g.add_node(fact_id)
        except Exception as exc:
            logger.debug("Failed to add isolated fact nodes: %s", exc)

        return g
