# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later
"""Deduplication candidates must share visibility and project identity."""

from __future__ import annotations

from superlocalmemory.core.project_identity import project_key
from superlocalmemory.storage.json_columns import is_damaged, load_json_column


def same_context_candidates(db, incoming, candidates):
    visible = [
        fact
        for fact in candidates
        if fact.scope == incoming.scope
        and set(fact.shared_with or []) == set(incoming.shared_with or [])
    ]
    ids = sorted({incoming.memory_id, *(fact.memory_id for fact in visible)} - {""})
    projects = {}
    for start in range(0, len(ids), 400):
        batch = ids[start : start + 400]
        rows = db.execute(
            "SELECT memory_id,metadata_json FROM memories WHERE memory_id IN "
            f"({','.join('?' for _ in batch)})",
            tuple(batch),
        )
        for row in rows:
            if is_damaged(row["metadata_json"]):
                # Unknown context, not "no project": a value only this memory
                # equals, so it is never a candidate for another memory's fact.
                projects[row["memory_id"]] = object()
                continue
            metadata = load_json_column(row["metadata_json"], {}, column="memories.metadata_json",
                                        row=row["memory_id"])
            projects[row["memory_id"]] = project_key(metadata.get("project"))
    want = projects.get(incoming.memory_id)
    return [fact for fact in visible if projects.get(fact.memory_id) == want]
