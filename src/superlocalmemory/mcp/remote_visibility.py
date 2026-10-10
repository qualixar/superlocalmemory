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


def hidden_fact_ids(db: Any, profile_id: str, fact_ids: Sequence[str]) -> set[str]:
    """Which of ``fact_ids`` the current caller may not see (none for a local caller)."""
    return remote_view.hidden_among(current_view(), db, profile_id, fact_ids)


def with_view(path: str) -> str:
    """``path`` for the daemon, telling it a remote caller asks. Unchanged for a local caller."""
    view = current_view()
    if not view:
        return path
    return f"{path}{'&' if '?' in path else '?'}{remote_view.VIEW_PARAM}={view}"


__all__ = ["current_view", "hidden_fact_ids", "visible_facts", "with_view"]
