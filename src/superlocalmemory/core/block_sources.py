# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V4 | https://qualixar.com | https://varunpratap.com

"""Which facts may become Core Memory: memories, not prompts or bookkeeping.

Core Memory blocks are injected at every session start as "pinned,
high-value" context. They were compiled from the most-read, newest and most
confident facts of any origin, and on the author's store (9 Nov 2026) 14 of
the 15 facts behind them were not memories at all:

* fragments of prompts the user typed to an agent, stored by the auto-capture
  hook with ``metadata.source = "auto-capture"``. Recall already ranks those
  as requests, not facts (retrieval/request_memories.py); a profile block must
  not present them as what the user is or prefers;
* the hooks' own bookkeeping ("session ended 2026-11-09 13:46", "File changed:
  x.py", "[agy] session ended (exit) at ... in config."). SLM already defines
  that: encoding/non_answer.py, the ``non_answer`` flag set on save and by
  migration M062. A row stored before a shape was recognised carries no flag,
  so the text is checked too (``non_answer.is_stub``), with no migration.

A session summary with real content (a branch, commits, the last action) is
not bookkeeping by that definition and stays a source. Everything excluded
here stays stored and searchable; it is only never a block's source.
"""

from __future__ import annotations

from typing import Any, Iterable

from superlocalmemory.encoding.non_answer import is_stub
from superlocalmemory.retrieval.request_memories import REQUEST_SOURCE

#: Rows read per row kept, before the text check drops bookkeeping: a block
#: of 5 reads up to 50 candidates.
OVERSAMPLE = 10

_SOURCE = ("CASE WHEN json_valid(bm.metadata_json) "
           "THEN json_extract(bm.metadata_json, '$.source') END")

#: True for a fact (alias ``f``) that may not be a source: flagged bookkeeping,
#: or derived from a captured prompt.
_NOT_A_SOURCE = (
    "(COALESCE(f.non_answer, 0) = 1 OR EXISTS (SELECT 1 FROM memories bm "
    f"WHERE bm.memory_id = f.memory_id AND COALESCE({_SOURCE}, '') = ?))"
)


def source_filter() -> tuple[str, tuple[str, ...]]:
    """``(" AND NOT ...", params)`` to append to a query over ``atomic_facts f``."""
    return " AND NOT " + _NOT_A_SOURCE, (REQUEST_SOURCE,)


def keep_sources(rows: Iterable[Any], limit: int) -> list[Any]:
    """The first ``limit`` rows whose text is not bookkeeping, order kept."""
    kept = [r for r in rows if not is_stub(str(_content(r)))]
    return kept[:limit]


def not_sources(conn: Any, fact_ids: Iterable[str]) -> set[str]:
    """The ids among ``fact_ids`` that may not be a block's source."""
    ids = [str(i) for i in fact_ids if i]
    if not ids:
        return set()
    marks = ",".join("?" for _ in ids)
    rows = conn.execute(
        f"SELECT f.fact_id, f.content, {_NOT_A_SOURCE} AS excluded "
        f"FROM atomic_facts f WHERE f.fact_id IN ({marks})", (REQUEST_SOURCE, *ids))
    return {str(r[0]) for r in rows if r[2] or is_stub(str(r[1] or ""))}


def _content(row: Any) -> Any:
    try:
        return row["content"]
    except (KeyError, IndexError, TypeError):
        return getattr(row, "content", "")


__all__ = ["OVERSAMPLE", "keep_sources", "not_sources", "source_filter"]
