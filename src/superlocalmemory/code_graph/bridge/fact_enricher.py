# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory v3.4 — CodeGraph Bridge Module

"""Fact Enricher — append code metadata to fact descriptions.

For MVP: builds enrichment strings from matched nodes but does NOT
write to memory.db.  Returns the enriched description so callers
(MCP tools, event listeners) can decide what to do with it.

This keeps the bridge completely isolated from memory.db writes
per the MVP constraint.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from superlocalmemory.code_graph.models import EdgeKind

if TYPE_CHECKING:
    from superlocalmemory.code_graph.bridge.entity_resolver import MatchedNode
    from superlocalmemory.code_graph.database import CodeGraphDatabase

logger = logging.getLogger(__name__)

MAX_ENRICHMENT_LEN = 500
MAX_NODES_PER_ENRICHMENT = 3


@dataclass(frozen=True)
class EnrichmentResult:
    """Result of enriching a fact description."""
    fact_id: str
    original_description: str
    enriched_description: str
    nodes_used: int


class FactEnricher:
    """Enriches SLM fact descriptions with code graph metadata.

    For MVP, this only constructs enrichment strings from code_graph.db.
    It does NOT read from or write to memory.db.
    """

    def __init__(self, code_graph_db: CodeGraphDatabase) -> None:
        self._db = code_graph_db

    def enrich(
        self,
        fact_id: str,
        matched_nodes: list[MatchedNode],
        original_description: str = "",
    ) -> str:
        """Enrich a fact description with code metadata.

        Args:
            fact_id: SLM fact ID.
            matched_nodes: Nodes matched by EntityResolver.
            original_description: Original fact text to enrich.

        Returns:
            Enriched description string. If no matches, returns
            original_description unchanged.
        """
        if not matched_nodes:
            return original_description

        # Sort by confidence descending, limit to top N
        sorted_nodes = sorted(
            matched_nodes,
            key=lambda n: n.confidence,
            reverse=True,
        )[:MAX_NODES_PER_ENRICHMENT]

        suffix_parts: list[str] = []
        for node in sorted_nodes:
            suffix_part = self._build_node_suffix(node)
            if suffix_part:
                suffix_parts.append(suffix_part)

        if not suffix_parts:
            return original_description

        enrichment_suffix = " ".join(suffix_parts)

        if original_description:
            enriched = f"{original_description} {enrichment_suffix}"
        else:
            enriched = enrichment_suffix

        # Truncate to MAX_ENRICHMENT_LEN
        if len(enriched) > MAX_ENRICHMENT_LEN:
            enriched = enriched[:MAX_ENRICHMENT_LEN - 3] + "..."

        return enriched

    def bulk_enrich(
        self,
        fact_node_pairs: list[tuple[str, list[MatchedNode], str]],
    ) -> list[EnrichmentResult]:
        """Enrich multiple facts. Returns list of EnrichmentResult."""
        results: list[EnrichmentResult] = []
        for fact_id, nodes, description in fact_node_pairs:
            enriched = self.enrich(fact_id, nodes, description)
            results.append(EnrichmentResult(
                fact_id=fact_id,
                original_description=description,
                enriched_description=enriched,
                nodes_used=min(len(nodes), MAX_NODES_PER_ENRICHMENT),
            ))
        return results

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_node_suffix(self, node: MatchedNode) -> str:
        """Build enrichment suffix for a single matched node."""
        try:
            # Get callers count
            callers_count = self._count_edges_to(node.node_id, EdgeKind.CALLS)
            # Get callees count
            callees_count = self._count_edges_from(node.node_id, EdgeKind.CALLS)

            suffix = f"[{node.kind}: {node.file_path}::{node.qualified_name.split('::')[-1] if '::' in node.qualified_name else node.qualified_name}"
            if callers_count > 0:
                suffix += f"; {callers_count} callers"
            if callees_count > 0:
                suffix += f"; calls {callees_count}"
            suffix += "]"
            return suffix
        except Exception:
            logger.debug(
                "Failed to build suffix for node %s", node.node_id,
                exc_info=True,
            )
            return ""

    def _count_edges_to(self, node_id: str, kind: EdgeKind) -> int:
        """Count incoming edges of a specific kind."""
        rows = self._db.execute(
            "SELECT COUNT(*) as cnt FROM graph_edges "
            "WHERE target_node_id = ? AND kind = ?",
            (node_id, kind.value),
        )
        return rows[0]["cnt"] if rows else 0

    def _count_edges_from(self, node_id: str, kind: EdgeKind) -> int:
        """Count outgoing edges of a specific kind."""
        rows = self._db.execute(
            "SELECT COUNT(*) as cnt FROM graph_edges "
            "WHERE source_node_id = ? AND kind = ?",
            (node_id, kind.value),
        )
        return rows[0]["cnt"] if rows else 0
