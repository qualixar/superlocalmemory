# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V4 | https://qualixar.com | https://varunpratap.com

"""Apply a :mod:`save_repair` plan: back up, re-file, requeue, backfill.

Everything happens in one transaction on ``memory.db``; every write is guarded
by the state the plan saw, so an operation that changed since (retried,
cancelled) is skipped, never overwritten. A copy of the store is taken first
with SQLite's online backup, next to it, readable by this user only.
"""

from __future__ import annotations

import json
import os
import sqlite3
import tempfile
import time
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

from superlocalmemory.storage.save_repair import (
    REPOINT_ACTIONS,
    RESTORE_ACTIONS,
    STALE_DEAD_LETTERS,
    RefileItem,
    SavePlan,
    _fingerprint,
    count_stale_dead_letters,
)


@dataclass(frozen=True, slots=True)
class SaveRepairResult:
    refiled: int
    skipped: int
    agent_names_backfilled: int
    backup: str
    stale_dead_letters_cleared: int = 0


def backup_store(db_path: Path) -> Path:
    """A consistent copy of ``db_path`` beside it (online backup, 0600)."""
    fd, name = tempfile.mkstemp(
        prefix=f"{db_path.stem}.before-save-repair-{int(time.time())}-",
        suffix=".db", dir=db_path.parent,
    )
    os.close(fd)
    target = Path(name)
    source = sqlite3.connect(str(db_path))
    try:
        copy = sqlite3.connect(str(target))
        try:
            source.backup(copy)
        finally:
            copy.close()
    finally:
        source.close()
    os.chmod(target, 0o600)
    return target


def _memory_for(op: dict, memory_id: str):
    from superlocalmemory.storage.models import MemoryRecord

    metadata = json.loads(op["raw_metadata_json"] or "{}")
    metadata["ingestion_operation_id"] = op["operation_id"]
    return MemoryRecord(
        memory_id=memory_id, profile_id=op["profile_id"], content=op["raw_content"],
        session_id=op["session_id"], session_date=op["session_date"] or op["created_at"][:10],
        speaker=op["speaker"], role=op["role"], metadata=metadata, scope=op["scope"],
        shared_with=json.loads(op["shared_with_json"] or "[]") or None,
    )


def _new_fact(db, op: dict, memory_id: str) -> str:
    """The searchable fact the save should have had, in its own memory."""
    from superlocalmemory.core.ingest_gate import apply_ingest_gate
    from superlocalmemory.core.kind_assignment import assign_kinds
    from superlocalmemory.core.queryable_fact import queryable_fact
    from superlocalmemory.core.source_excerpt import attach_source_pointer

    content = op["raw_content"]
    gate = apply_ingest_gate(content)
    now = datetime.now(UTC).isoformat()
    fact = queryable_fact(
        gate.fact_content, profile_id=op["profile_id"], scope=op["scope"],
        shared_with=json.loads(op["shared_with_json"] or "[]"), session_id=op["session_id"],
        observation_date=op["session_date"] or op["created_at"][:10], created_at=now,
    )
    fact.memory_id = memory_id
    metadata = json.loads(op["raw_metadata_json"] or "{}")
    fact = assign_kinds([fact], metadata=metadata, source_type=op["source_type"],
                        classifier=None, db=db)[0]
    if gate.truncated:
        from superlocalmemory.core.ingest_gate import _MAX_VERBATIM_CHARS

        fact = attach_source_pointer(fact, content, _MAX_VERBATIM_CHARS)
    return db.insert_fact_immutable(fact)


def _restore_text(db, op: dict, item: RefileItem) -> bool:
    """Restore a blanked memory (and its facts) from the operation; False writes nothing.

    The statements are ``store_repair``'s own, shared: every row is verified
    before any is written, each write guarded by the text it replaces. The
    reviewed memory text is checked again against the plan's fingerprint.
    """
    from superlocalmemory.storage.store_repair import restore_blanked_text

    rows = db.execute("SELECT content FROM memories WHERE memory_id=?", (item.memory_id,))
    if not rows or _fingerprint(rows[0]["content"]) != item.memory_fingerprint:
        return False
    return restore_blanked_text(db.execute, item.memory_id, rows[0]["content"],
                                op["raw_content"], item.fact_ids, accept_restored_facts=True)


def _refile(db, item: RefileItem) -> bool:
    rows = db.execute("SELECT * FROM ingestion_operations WHERE operation_id=? "
                      "AND state='failed'", (item.operation_id,))
    if not rows:
        return False
    op = dict(rows[0])
    if item.action in RESTORE_ACTIONS and not _restore_text(db, op, item):
        return False        # nothing was written: the memory or a fact is no longer as reviewed
    if item.action == "add_memory_and_fact":
        db.store_memory(_memory_for(op, item.memory_id))
    fact_ids = list(item.fact_ids) if item.action in REPOINT_ACTIONS else [
        _new_fact(db, op, item.memory_id)]
    db.execute(
        "UPDATE ingestion_operations SET state='queryable', attempt_count=0, next_retry_at=0, "
        "last_error='', lease_owner='', lease_expires_at=0, queryable_fact_ids_json=?, "
        "final_fact_ids_json='[]', derivation_state_json='{}', "
        "updated_at=strftime('%Y-%m-%dT%H:%M:%fZ','now') "
        "WHERE operation_id=? AND state='failed'",
        (json.dumps(fact_ids), item.operation_id))
    db.execute("DELETE FROM dead_letter_operations WHERE original_op_id=?",
               (item.operation_id,))
    return True


def apply(db_path: Path, save_plan: SavePlan, *, backup: bool = True) -> SaveRepairResult:
    """Back up ``db_path``, then apply ``save_plan`` in one transaction.

    ``backup=False`` is for a caller that has already taken the product's shared
    safety copy (Memory health); ``slm ops repair-saves`` always backs up.
    """
    from superlocalmemory.storage.database import DatabaseManager
    from superlocalmemory.storage.save_repair import plan

    copy = backup_store(db_path) if backup else None
    db = DatabaseManager(db_path)
    refiled = skipped = filled = cleared = 0
    try:
        with db.raw_connection() as conn:
            # Hold the writer reservation while rechecking the reviewed evidence
            # and applying it. Another writer cannot change visibility mid-repair.
            conn.execute("BEGIN IMMEDIATE")
            fresh = plan(conn)
            candidates = {item.operation_id: item for item in fresh.refile}
            for item in save_plan.refile:
                current = candidates.get(item.operation_id)
                if current and current.action == "add_memory_and_fact":
                    occupied = conn.execute("SELECT 1 FROM memories WHERE memory_id=?",
                                            (item.memory_id,)).fetchone()
                    current = None if occupied else replace(current, memory_id=item.memory_id)
                if current == item and _refile(db, item):
                    refiled += 1
                else:
                    skipped += 1
            for memory_id, agent in save_plan.attribution:
                if (memory_id, agent) not in fresh.attribution:
                    continue
                db.execute(
                    "UPDATE memories SET metadata_json=json_set(COALESCE(metadata_json,'{}'),"
                    " '$.agent_id', ?, '$.agent_id_source', 'backfill:operation') "
                    "WHERE memory_id=? "
                    "AND COALESCE(json_extract(metadata_json,'$.agent_id'),'')=''",
                    (agent, memory_id))
                filled += int(db.execute("SELECT changes() AS n")[0]["n"])
            # Old failures of saves that later completed (after the re-files above,
            # which delete their own rows, so only genuine leftovers are counted).
            cleared = count_stale_dead_letters(conn)
            db.execute(f"DELETE FROM {STALE_DEAD_LETTERS}")
    finally:
        db.close()
    return SaveRepairResult(refiled, skipped, filled, str(copy) if copy else "", cleared)


__all__ = ["SaveRepairResult", "apply", "backup_store"]
