# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory v3.4 — CodeGraph Bridge Module

"""Bridge between code_graph.db and SLM memory.db.

Five mechanisms cross-pollinate code structure with semantic memory:

1. **Entity Resolver** — Matches code mentions in fact text against
   code graph nodes.  Creates `code_memory_links` entries.
2. **Fact Enricher** — Appends file/function metadata to fact
   descriptions for better auto-invoke precision.
3. **Hebbian Linker** — Creates/strengthens association edges when
   two facts reference code nodes in the same call subgraph.
4. **Event Listeners** — Bidirectional event bus listeners bridging
   `memory.stored` → code linking and `code_graph.node_*` → staleness.
5. **Temporal Checker** — Marks memories about deleted/renamed code
   as temporally stale.

**Hard Rule (HR-1):** The bridge NEVER modifies memory.db schema.
All bridge data lives in code_graph.db via the `code_memory_links` table.
"""

from superlocalmemory.code_graph.bridge.entity_resolver import EntityResolver
from superlocalmemory.code_graph.bridge.event_listeners import BridgeEventListeners
from superlocalmemory.code_graph.bridge.fact_enricher import FactEnricher
from superlocalmemory.code_graph.bridge.hebbian_linker import HebbianLinker
from superlocalmemory.code_graph.bridge.temporal_checker import TemporalChecker

__all__ = [
    "EntityResolver",
    "FactEnricher",
    "HebbianLinker",
    "BridgeEventListeners",
    "TemporalChecker",
]
