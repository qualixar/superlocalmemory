# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""M018 — trust-gated merge columns on atomic_facts (memory.db, deferred).

Adds four columns needed by the belief-update framework alignment
(reference_diag/full_framework_detailed.puml):

    source_agent_id           TEXT    NOT NULL DEFAULT ''
    pending_corroboration     INTEGER NOT NULL DEFAULT 0
    corroboration_agents_json TEXT    NOT NULL DEFAULT '[]'
    intent_flagged            INTEGER NOT NULL DEFAULT 0

``source_agent_id`` records which agent asserted a fact, so per-source trust
(tau(s)) can be looked up at merge time and independent multi-source
corroboration can be detected on first mention. ``pending_corroboration``
marks a fact as quarantined (untrusted merge, or awaiting a second source on
Add). ``corroboration_agents_json`` is the JSON list of distinct agent_ids
that have corroborated the attribute. ``intent_flagged`` marks a fact whose
content was classified as query/directive rather than assertion — still
stored (never silently dropped, see store_pipeline's v3.3.21 invariant) but
forced through quarantine regardless of source trust.

Deferred like M011/M013/M015/M016 because ``atomic_facts`` is bootstrapped
at engine init, not at migration time (``apply_all`` runs before engine
init on daemon startup and would hit "no such table").

Author: Varun Pratap Bhardwaj / Qualixar
"""

from __future__ import annotations

import sqlite3

NAME = "M018_trust_gated_merge"
DB_TARGET = "memory"

DDL = """
BEGIN IMMEDIATE;
ALTER TABLE atomic_facts ADD COLUMN source_agent_id TEXT NOT NULL DEFAULT '';
ALTER TABLE atomic_facts ADD COLUMN pending_corroboration INTEGER NOT NULL DEFAULT 0;
ALTER TABLE atomic_facts ADD COLUMN corroboration_agents_json TEXT NOT NULL DEFAULT '[]';
ALTER TABLE atomic_facts ADD COLUMN intent_flagged INTEGER NOT NULL DEFAULT 0;
CREATE INDEX IF NOT EXISTS idx_facts_pending_corroboration
    ON atomic_facts(profile_id, pending_corroboration);
COMMIT;
"""


def verify(conn: sqlite3.Connection) -> bool:
    """Applied when atomic_facts has all four columns and the new index."""
    info = conn.execute("PRAGMA table_info(atomic_facts)").fetchall()
    if not info:
        return True  # table absent — nothing to verify (shouldn't happen; deferred)
    cols = {r[1] for r in info}
    required = {
        "source_agent_id", "pending_corroboration",
        "corroboration_agents_json", "intent_flagged",
    }
    if not required.issubset(cols):
        return False
    idx = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='index' AND name=?",
        ("idx_facts_pending_corroboration",),
    ).fetchone()
    return idx is not None
