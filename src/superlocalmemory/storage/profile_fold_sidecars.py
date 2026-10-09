# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""A deleted profile's rows outside memory.db (see storage/profile_fold.py).

learning.db holds what SLM learned from the deleted profile's use: its ranker,
bandit, signals, evolution settings. Folding that into 'default' would retrain
default on another workspace's queries, so it is DELETED. Receipts, answer-check
history and saved views are purged by their own stores (server/routes/helpers.py
calls them first); their closure and erasure markers are KEPT on purpose so a
late write for the deleted profile is refused.

pending.db holds memories accepted but not yet stored: they are memories, so
they MOVE with the rest. The context cache is a cache: DELETED.

Every learning.db table with a ``profile_id`` column must have a decision here;
an unclassified one stops the delete before memory.db is touched.
"""

from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

DELETE, PURGED, KEEP = "delete", "purged-by-its-store", "keep"

LEARNING_DECISIONS: dict[str, str] = {
    "learning_signals": DELETE, "learning_features": DELETE, "learning_feedback": DELETE,
    "engagement_metrics": DELETE, "learning_model_state": DELETE, "bandit_arms": DELETE,
    "bandit_plays": DELETE, "evolution_config": DELETE, "evolution_llm_cost_log": DELETE,
    "shadow_observations": DELETE,
    "agent_experiences": PURGED, "cognitive_turn_receipts": PURGED,
    "external_evidence_receipts": PURGED, "execution_learning_receipts": PURGED,
    "execution_learning_events": PURGED, "answer_check_events": PURGED, "saved_views": PURGED,
    "agent_receipt_profile_closures": KEEP,  # refuses a late receipt for the deleted profile
    "answer_check_erasures": KEEP,           # the erasure marker of its answer-check history
}


class SidecarFoldError(RuntimeError):
    """A sidecar store cannot be handled; the profile was not deleted."""


def _scoped(conn: sqlite3.Connection) -> list[str]:
    out = []
    for (name,) in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall():
        try:
            if "profile_id" in {r[1] for r in conn.execute(f'PRAGMA table_info("{name}")')}:
                out.append(name)
        except sqlite3.OperationalError:
            continue
    return out


def check_learning(learning_db: Path) -> None:
    """Raise before anything is deleted when learning.db has an unclassified table."""
    if not Path(learning_db).exists():
        return
    with closing(sqlite3.connect(str(learning_db))) as conn:
        unknown = [t for t in _scoped(conn) if t not in LEARNING_DECISIONS]
    if unknown:
        raise SidecarFoldError(f"no recorded decision for learning.db {', '.join(unknown)}; "
                               "the profile was not deleted")


def purge_learned_state(learning_db: Path, profile_id: str) -> int:
    """Delete what was learned from the deleted profile, in one transaction."""
    if not Path(learning_db).exists():
        return 0
    removed = 0
    with closing(sqlite3.connect(str(learning_db))) as conn, conn:
        for table in _scoped(conn):
            if LEARNING_DECISIONS.get(table) == DELETE:
                removed += conn.execute(f'DELETE FROM "{table}" WHERE profile_id = ?',
                                        (profile_id,)).rowcount
    return removed


def move_pending(pending_db: Path, profile_id: str, target: str = "default") -> int:
    """Memories accepted for the deleted profile but not stored yet go to target."""
    if not Path(pending_db).exists():
        return 0
    with closing(sqlite3.connect(str(pending_db))) as conn, conn:
        if "pending_memories" not in _scoped(conn):
            return 0
        return conn.execute("UPDATE pending_memories SET profile_id = ? WHERE profile_id = ?",
                            (target, profile_id)).rowcount


def purge_context_cache(data_root: Path, profile_id: str) -> int:
    """The context cache is per profile; the GDPR erasure's purge, same files."""
    from superlocalmemory.core.context_cache import purge_profile_from_cache_db

    root = Path(data_root)
    candidates = [root / "active_brain_cache.db"]
    try:
        candidates += [child / "active_brain_cache.db" for child in root.iterdir() if child.is_dir()]
    except OSError:
        pass
    return sum(purge_profile_from_cache_db(c, profile_id) for c in candidates if c.exists())


def move_media(data_root: Path, profile_id: str, target: str = "default") -> int:
    """Images and documents of the deleted profile go to target; nothing is made if absent."""
    from superlocalmemory.media import open_media_store

    store = open_media_store(data_root=data_root)
    if store is None:
        return 0
    try:
        return store.move_profile_rows(profile_id, target)
    finally:
        store.close()


__all__ = ["LEARNING_DECISIONS", "SidecarFoldError", "check_learning", "move_pending",
           "move_media", "purge_context_cache", "purge_learned_state"]
