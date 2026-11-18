# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Erasing a fact removes the versions and proposals that name it (I-11).

The ``atomic_facts`` delete triggers of M080 cover every path that finally
deletes the row. Erasure first writes a tombstone, while the fact still exists;
this function runs at that moment (``core/transactions/tombstones``) so no reader
of versions or proposals can see the fact's words in between.

Compiled pages that cite the fact are scrubbed of those sentences and the
projected Markdown files are rewritten without them (``compiler/wiki.scrub_fact``).

A set still waiting for a decision is removed whole (a reviewer never sees a
diminished unit); from a set already decided only the proposals that name the
fact go, and a set left empty goes with them.
"""

from __future__ import annotations

from typing import Any

from superlocalmemory.compiler import versions
from superlocalmemory.compiler.versions import select_rows


def erase_for_fact(conn: Any, profile_id: str, fact_id: str) -> dict[str, int]:
    """Remove versions and proposals naming ``fact_id``; counts removed. Tolerates old schemas."""
    counts = {"versions": versions.erase_for_fact(conn, profile_id, fact_id), "proposals": 0,
              "pages": 0, "wiki_files": 0}
    try:
        counts["proposals"] = int(select_rows(
            conn, "SELECT COUNT(*) AS n FROM proposals p JOIN proposal_facts f ON "
            "f.proposal_id = p.proposal_id WHERE p.profile_id = ? AND f.fact_id = ?",
            (profile_id, fact_id))[0]["n"])
        select_rows(conn, "DELETE FROM proposal_sets WHERE profile_id = ? AND state = 'proposed' "
                    "AND set_id IN (SELECT p.set_id FROM proposals p JOIN proposal_facts f "
                    "ON f.proposal_id = p.proposal_id WHERE f.fact_id = ?) RETURNING set_id",
                    (profile_id, fact_id))
        select_rows(
            conn, "DELETE FROM proposals WHERE profile_id = ? AND proposal_id IN "
            "(SELECT proposal_id FROM proposal_facts WHERE fact_id = ?) RETURNING proposal_id",
            (profile_id, fact_id))
    except Exception as exc:  # noqa: BLE001 - tables absent on older schemas
        if "no such table" not in str(exc).lower():
            raise
    # Compiled pages that cite the fact lose those sentences, and the projected Markdown files
    # are rewritten without them (D-17, I-11): the folder never keeps what the store erased.
    from superlocalmemory.compiler import wiki

    scrubbed = wiki.scrub_fact(conn, profile_id, fact_id)
    counts["pages"], counts["wiki_files"] = len(scrubbed["pages"]), scrubbed["files"]
    return counts
