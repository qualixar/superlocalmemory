# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Rows keyed by a fact id that no foreign key will remove with the fact.

``ON DELETE CASCADE`` takes the tables that declare it (retention, temporal
validity, access log, polar codes, entity associations, temporal events). These
do not declare one, so every path that deletes a fact has to remove them itself.
Before 4.1.22 ``DatabaseManager.delete_fact`` removed only ``embedding_metadata``
and the graph edges, and four of its callers (enrichment's duplicate drop, the
resurrection undo, the canonical writer, entity erasure) left the rest behind.

What that cost, measured on a real 22k-fact store: 9,237 BM25 token rows for
facts that no longer exist, loaded into the live keyword corpus (it counted
30,090 documents for 22,175 facts); 20 vector-map rows whose vector still
answered nearest-neighbour searches for a deleted memory; 1,290 outcome scores.

Every row removed here is derived from the fact and recomputable; none carries
history. Access history and temporal validity are kept by their own cascades.
The vec0 row itself cannot be removed on this connection (it does not load the
vector extension): once its map and metadata rows are gone it is unreferenced,
and the maintenance sweep removes it (``VectorStore.gc_orphaned_vectors``).
"""

from __future__ import annotations

from typing import Any

#: (table, column naming the fact). Order is irrelevant: all in one transaction.
DERIVED_FACT_ROWS: tuple[tuple[str, str], ...] = (
    ("embedding_metadata", "fact_id"),
    ("vector_row_map", "fact_id"),
    ("bm25_tokens", "fact_id"),
    ("fact_outcome_score", "fact_id"),
    ("activation_cache", "node_id"),
    # The undo record of an automatic data repair quotes the fact's text; it
    # cascades from atomic_facts, and is listed for delete paths without FKs.
    ("data_repair_undo", "fact_id"),
)


def delete_derived_rows(db: Any, fact_id: str) -> None:
    """Remove the fact's non-cascading derived rows. Call inside the delete's
    own transaction, so the fact and its residue go together or not at all.

    A table an old or partial store does not have is skipped; any other error
    propagates and rolls the whole delete back.
    """
    for table, column in DERIVED_FACT_ROWS:
        try:
            db.execute(f"DELETE FROM {table} WHERE {column} = ?", (fact_id,))  # noqa: S608
        except Exception as exc:
            if "no such table" not in str(exc).lower():
                raise


__all__ = ["DERIVED_FACT_ROWS", "delete_derived_rows"]
