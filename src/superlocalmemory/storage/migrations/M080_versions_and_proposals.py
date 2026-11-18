# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""M080: versions of derived artifacts, and ProposalSets (4.2.30, WS-6, D-09, I-10, I-11).

**Versions first.** A derived artifact (consolidated summary, community summary,
persona, entity profile, memory scene, core memory block) is rewritten whenever
its facts change. Before any rewrite its previous row is copied into
``artifact_versions`` (``compiler/versions.keep_previous``), in the same
transaction, so nothing derived is lost to an overwrite. ``artifact_version_facts``
links a version to the facts its text was built from, so erasing a fact removes
the versions that quote it (I-11).

**ProposalSets.** A unit of proposed changes (retire, merge, edit derived text,
create a page), each with its basis when applied (a certificate id, a rule
certificate id, or a person's approval, I-10) and the inverse of every applied
change, so the whole unit rolls back exactly. ``proposal_facts`` links a
proposal to the facts it names; erasing a fact removes a set still waiting for a
decision whole (a reviewer never sees a diminished unit) and, from a set already
decided, only the proposals that name it, and any set left empty (I-11).

CHECKs hold invariants only (an applied set carries a basis), never a list of
kinds or states, which SQLite could not extend after release (W0-29). The
tables hold ids, hashes and derived text built from the same facts; erasing a
fact erases what quotes it.
"""

from __future__ import annotations

import sqlite3

NAME = "M080_versions_and_proposals"
DB_TARGET = "memory"
#: Additive: new tables an older build never reads.
DOWNGRADE_FLOOR = 51

DDL = """
CREATE TABLE IF NOT EXISTS artifact_versions (
    version_id   TEXT PRIMARY KEY,
    profile_id   TEXT NOT NULL,
    artifact     TEXT NOT NULL CHECK (length(artifact) > 0),
    artifact_key TEXT NOT NULL CHECK (length(artifact_key) > 0),
    seq          INTEGER NOT NULL CHECK (seq >= 1),
    content_hash TEXT NOT NULL,
    row_json     TEXT NOT NULL,
    reason       TEXT NOT NULL CHECK (length(reason) > 0),
    proposal_id  TEXT,
    created_at   TEXT NOT NULL,
    UNIQUE (profile_id, artifact, artifact_key, seq)
);
CREATE INDEX IF NOT EXISTS idx_artifact_versions_profile
    ON artifact_versions (profile_id, artifact, created_at);
CREATE TABLE IF NOT EXISTS artifact_version_facts (
    version_id TEXT NOT NULL,
    fact_id    TEXT NOT NULL,
    PRIMARY KEY (version_id, fact_id)
);
CREATE INDEX IF NOT EXISTS idx_artifact_version_facts_fact
    ON artifact_version_facts (fact_id);
CREATE TABLE IF NOT EXISTS proposal_sets (
    set_id           TEXT PRIMARY KEY,
    profile_id       TEXT NOT NULL,
    title            TEXT NOT NULL DEFAULT '',
    origin           TEXT NOT NULL CHECK (length(origin) > 0),
    state            TEXT NOT NULL CHECK (length(state) > 0),
    basis_kind       TEXT,
    basis_ref        TEXT,
    decided_by       TEXT,
    decided_channel  TEXT,
    created_at       TEXT NOT NULL,
    decided_at       TEXT,
    applied_at       TEXT,
    rolled_back_at   TEXT,
    CHECK ((basis_kind IS NULL) = (basis_ref IS NULL)),
    CHECK (state != 'applied' OR basis_kind IS NOT NULL)
);
CREATE INDEX IF NOT EXISTS idx_proposal_sets_profile
    ON proposal_sets (profile_id, state, created_at);
CREATE TABLE IF NOT EXISTS proposals (
    proposal_id    TEXT PRIMARY KEY,
    set_id         TEXT NOT NULL,
    profile_id     TEXT NOT NULL,
    seq            INTEGER NOT NULL CHECK (seq >= 0),
    kind           TEXT NOT NULL CHECK (length(kind) > 0),
    summary        TEXT NOT NULL DEFAULT '',
    payload_json   TEXT NOT NULL,
    state          TEXT NOT NULL CHECK (length(state) > 0),
    inverse_json   TEXT,
    created_at     TEXT NOT NULL,
    applied_at     TEXT,
    rolled_back_at TEXT,
    UNIQUE (set_id, seq)
);
CREATE INDEX IF NOT EXISTS idx_proposals_set ON proposals (set_id);
CREATE INDEX IF NOT EXISTS idx_proposals_profile ON proposals (profile_id, state);
CREATE TABLE IF NOT EXISTS proposal_facts (
    proposal_id TEXT NOT NULL,
    fact_id     TEXT NOT NULL,
    PRIMARY KEY (proposal_id, fact_id)
);
CREATE INDEX IF NOT EXISTS idx_proposal_facts_fact ON proposal_facts (fact_id)
"""

#: Links go with the row they describe (the profile fold deletes whole tables).
_LINK_TRIGGERS = (
    """
CREATE TRIGGER IF NOT EXISTS trg_artifact_versions_links
AFTER DELETE ON artifact_versions
BEGIN
    DELETE FROM artifact_version_facts WHERE version_id = OLD.version_id;
END
""",
    """
CREATE TRIGGER IF NOT EXISTS trg_proposals_links
AFTER DELETE ON proposals
BEGIN
    DELETE FROM proposal_facts WHERE proposal_id = OLD.proposal_id;
    DELETE FROM proposal_sets WHERE set_id = OLD.set_id
        AND NOT EXISTS (SELECT 1 FROM proposals WHERE set_id = OLD.set_id);
END
""",
    """
CREATE TRIGGER IF NOT EXISTS trg_proposal_sets_members
AFTER DELETE ON proposal_sets
BEGIN
    DELETE FROM proposals WHERE set_id = OLD.set_id;
END
""",
)

#: I-11: erasing a fact removes the versions that quote it and the proposals that name it.
_FACT_TRIGGERS = (
    """
CREATE TRIGGER IF NOT EXISTS trg_artifact_versions_fact_erased
AFTER DELETE ON atomic_facts
BEGIN
    DELETE FROM artifact_versions WHERE version_id IN
        (SELECT version_id FROM artifact_version_facts WHERE fact_id = OLD.fact_id);
END
""",
    """
CREATE TRIGGER IF NOT EXISTS trg_proposals_fact_erased
AFTER DELETE ON atomic_facts
BEGIN
    DELETE FROM proposal_sets WHERE state = 'proposed' AND set_id IN
        (SELECT p.set_id FROM proposals p JOIN proposal_facts f ON f.proposal_id = p.proposal_id
          WHERE f.fact_id = OLD.fact_id);
    DELETE FROM proposals WHERE proposal_id IN
        (SELECT proposal_id FROM proposal_facts WHERE fact_id = OLD.fact_id);
END
""",
)

_TRIGGER_NAMES = (
    "trg_artifact_versions_links", "trg_proposals_links", "trg_proposal_sets_members",
    "trg_artifact_versions_fact_erased", "trg_proposals_fact_erased",
)


def _has_facts(conn: sqlite3.Connection) -> bool:
    return bool(conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='atomic_facts'").fetchone())


def apply(conn: sqlite3.Connection) -> None:
    for statement in DDL.split(";"):
        if statement.strip():
            conn.execute(statement)
    for trigger in _LINK_TRIGGERS:
        conn.execute(trigger)
    if _has_facts(conn):
        for trigger in _FACT_TRIGGERS:
            conn.execute(trigger)


def verify(conn: sqlite3.Connection) -> bool:
    names = {r[0] for r in conn.execute("SELECT name FROM sqlite_master")}
    tables = {"artifact_versions", "artifact_version_facts", "proposal_sets", "proposals",
              "proposal_facts"}
    if not tables <= names:
        return False
    wanted = _TRIGGER_NAMES if _has_facts(conn) else _TRIGGER_NAMES[:3]
    return set(wanted) <= names


def repair(conn: sqlite3.Connection) -> None:
    apply(conn)
