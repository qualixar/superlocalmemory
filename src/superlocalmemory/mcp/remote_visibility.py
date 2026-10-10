# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V4 | https://qualixar.com | https://varunpratap.com

"""Apply what a remote caller may see (retrieval/remote_view) to the MCP memory tools."""

from __future__ import annotations

from typing import Any, Sequence

from superlocalmemory.mcp.remote_caller import current_remote_key_id, current_remote_media_allowed
from superlocalmemory.retrieval import remote_view, visibility


def current_view() -> str:
    """``""`` for a caller on this computer, else which remote view applies."""
    if current_remote_key_id() is None:
        return ""
    return remote_view.REMOTE_MEDIA if current_remote_media_allowed() else remote_view.REMOTE


def visible_facts(db: Any, profile_id: str, facts: Sequence[Any]) -> list[Any]:
    """``facts`` without what the current caller may not see. The same list for a local caller."""
    ctx = remote_view.context_for(current_view(), db, profile_id)
    if ctx is None or not facts:
        return facts if isinstance(facts, list) else list(facts)
    with visibility.use(ctx):
        hidden = visibility.hidden_among(db, profile_id, [f.fact_id for f in facts])
    return [f for f in facts if f.fact_id not in hidden]


__all__ = ["current_view", "visible_facts"]
