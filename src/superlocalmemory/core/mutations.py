# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later

"""Canonical authorized fact delete and update operations."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("superlocalmemory.audit")


def _context(
    engine: Any,
    operation: str,
    fact_id: str,
    *,
    trusted_actor_id: str,
    source_agent_id: str,
    content_preview: str = "",
) -> tuple[str, dict[str, str]]:
    if not trusted_actor_id:
        raise ValueError("trusted actor identity is required")
    profile_id = engine.profile_id
    context = {
        "operation": operation,
        "agent_id": trusted_actor_id,
        "source_agent_id": source_agent_id,
        "profile_id": profile_id,
        "fact_id": fact_id,
    }
    if content_preview:
        context["content_preview"] = content_preview[:100]
    engine._hooks.run_pre(operation, context)
    return profile_id, context


def delete_fact_authorized(
    engine: Any,
    fact_id: str,
    *,
    trusted_actor_id: str,
    source_agent_id: str,
) -> dict[str, Any]:
    """Authorize, delete one profile-owned fact, then emit post hooks."""
    profile_id, context = _context(
        engine,
        "delete",
        fact_id,
        trusted_actor_id=trusted_actor_id,
        source_agent_id=source_agent_id,
    )
    rows = engine._db.execute(
        "SELECT content FROM atomic_facts "
        "WHERE fact_id = ? AND profile_id = ? LIMIT 1",
        (fact_id, profile_id),
    )
    if not rows:
        return {"ok": False, "error": f"Memory {fact_id} not found"}
    content_preview = dict(rows[0]).get("content", "")[:80]
    engine._db.delete_fact(fact_id)
    engine._hooks.run_post("delete", context)
    logger.info(
        "DELETE fact_id=%s actor=%s source_agent=%s content=%s",
        fact_id[:16], trusted_actor_id, source_agent_id, content_preview,
    )
    return {
        "ok": True,
        "deleted": fact_id,
        "content_preview": content_preview,
    }


def update_fact_authorized(
    engine: Any,
    fact_id: str,
    content: str,
    *,
    trusted_actor_id: str,
    source_agent_id: str,
) -> dict[str, Any]:
    """Authorize a fact update and refresh semantic and lexical indexes."""
    if not content or not content.strip():
        return {"ok": False, "error": "content cannot be empty"}
    content = content.strip()
    profile_id, context = _context(
        engine,
        "update",
        fact_id,
        trusted_actor_id=trusted_actor_id,
        source_agent_id=source_agent_id,
        content_preview=content,
    )
    rows = engine._db.execute(
        "SELECT content FROM atomic_facts "
        "WHERE fact_id = ? AND profile_id = ? LIMIT 1",
        (fact_id, profile_id),
    )
    if not rows:
        return {"ok": False, "error": f"Memory {fact_id} not found"}
    old_content = dict(rows[0]).get("content", "")[:80]
    updates: dict[str, Any] = {"content": content}
    if engine._embedder:
        try:
            embedding = engine._embedder.embed(content)
            if embedding:
                updates["embedding"] = embedding
                fisher_mean, fisher_variance = (
                    engine._embedder.compute_fisher_params(embedding)
                )
                updates["fisher_mean"] = fisher_mean
                updates["fisher_variance"] = fisher_variance
        except Exception as exc:
            logger.warning("UPDATE embedding refresh failed: %s", exc)
    engine._db.update_fact(fact_id, updates)
    retrieval = getattr(engine, "_retrieval_engine", None)
    bm25 = getattr(retrieval, "_bm25", None) if retrieval else None
    if bm25:
        bm25.add(fact_id, content, profile_id)
    engine._hooks.run_post("update", context)
    logger.info(
        "UPDATE fact_id=%s actor=%s source_agent=%s old=%s new=%s",
        fact_id[:16], trusted_actor_id, source_agent_id, old_content, content[:80],
    )
    return {"ok": True, "fact_id": fact_id, "content": content}


__all__ = ["delete_fact_authorized", "update_fact_authorized"]
