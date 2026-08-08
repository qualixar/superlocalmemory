# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later

"""M039 — normalized scene/fact membership for bounded assignment.

``memory_scenes.fact_ids_json`` remains the public compatibility format. This
additive projection provides the indexed reverse lookup needed to map nearest
fact-vector hits back to candidate scenes without scanning every scene for
every ingested fact. Triggers keep all existing scene write paths synchronized.
"""

from __future__ import annotations

import sqlite3

NAME = "M039_scene_fact_members"
DB_TARGET = "memory"

DDL = """
BEGIN IMMEDIATE;

-- Keep the deferred migration independently safe for partial/legacy installs.
-- On normal daemon startup MemoryEngine has already created this table.
CREATE TABLE IF NOT EXISTS memory_scenes (
    scene_id        TEXT PRIMARY KEY,
    profile_id      TEXT NOT NULL DEFAULT 'default',
    theme           TEXT NOT NULL DEFAULT '',
    fact_ids_json   TEXT NOT NULL DEFAULT '[]',
    entity_ids_json TEXT NOT NULL DEFAULT '[]',
    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
    last_updated    TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (profile_id) REFERENCES profiles(profile_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_scenes_profile ON memory_scenes(profile_id);
CREATE UNIQUE INDEX IF NOT EXISTS uq_scenes_profile_scene
    ON memory_scenes (profile_id, scene_id);
CREATE UNIQUE INDEX IF NOT EXISTS uq_facts_profile_fact
    ON atomic_facts (profile_id, fact_id);

CREATE TABLE IF NOT EXISTS scene_fact_members (
    profile_id TEXT NOT NULL,
    scene_id   TEXT NOT NULL,
    fact_id    TEXT NOT NULL,
    position   INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (scene_id, fact_id),
    FOREIGN KEY (profile_id, scene_id)
        REFERENCES memory_scenes(profile_id, scene_id) ON DELETE CASCADE,
    FOREIGN KEY (profile_id, fact_id)
        REFERENCES atomic_facts(profile_id, fact_id) ON DELETE CASCADE,
    FOREIGN KEY (profile_id) REFERENCES profiles(profile_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_scene_fact_members_lookup
    ON scene_fact_members (profile_id, fact_id, scene_id);
CREATE INDEX IF NOT EXISTS idx_scene_fact_members_order
    ON scene_fact_members (scene_id, position);

CREATE TRIGGER IF NOT EXISTS trg_scene_fact_members_insert
AFTER INSERT ON memory_scenes
BEGIN
    DELETE FROM scene_fact_members WHERE scene_id = NEW.scene_id;
    INSERT OR IGNORE INTO scene_fact_members
        (profile_id, scene_id, fact_id, position)
    SELECT NEW.profile_id, NEW.scene_id, af.fact_id, CAST(member.key AS INTEGER)
    FROM json_each(
        CASE WHEN json_valid(NEW.fact_ids_json)
             THEN NEW.fact_ids_json ELSE '[]' END
    ) AS member
    JOIN atomic_facts AS af
      ON af.fact_id = member.value
     AND af.profile_id = NEW.profile_id;
END;

CREATE TRIGGER IF NOT EXISTS trg_scene_fact_members_update
AFTER UPDATE OF profile_id, fact_ids_json ON memory_scenes
BEGIN
    DELETE FROM scene_fact_members WHERE scene_id = NEW.scene_id;
    INSERT OR IGNORE INTO scene_fact_members
        (profile_id, scene_id, fact_id, position)
    SELECT NEW.profile_id, NEW.scene_id, af.fact_id, CAST(member.key AS INTEGER)
    FROM json_each(
        CASE WHEN json_valid(NEW.fact_ids_json)
             THEN NEW.fact_ids_json ELSE '[]' END
    ) AS member
    JOIN atomic_facts AS af
      ON af.fact_id = member.value
     AND af.profile_id = NEW.profile_id;
END;

INSERT OR IGNORE INTO scene_fact_members
    (profile_id, scene_id, fact_id, position)
SELECT ms.profile_id, ms.scene_id, af.fact_id, CAST(member.key AS INTEGER)
FROM memory_scenes AS ms
JOIN json_each(
    CASE WHEN json_valid(ms.fact_ids_json) THEN ms.fact_ids_json ELSE '[]' END
) AS member
JOIN atomic_facts AS af
  ON af.fact_id = member.value
 AND af.profile_id = ms.profile_id;

COMMIT;
"""


def verify(conn: sqlite3.Connection) -> bool:
    """Verify the table, covering indexes, and synchronization triggers."""
    objects = {
        (str(row[0]), str(row[1]))
        for row in conn.execute(
            "SELECT name, type FROM sqlite_master "
            "WHERE name IN (?, ?, ?, ?, ?, ?, ?)"
            ,
            (
                "scene_fact_members",
                "idx_scene_fact_members_lookup",
                "idx_scene_fact_members_order",
                "trg_scene_fact_members_insert",
                "trg_scene_fact_members_update",
                "uq_scenes_profile_scene",
                "uq_facts_profile_fact",
            ),
        ).fetchall()
    }
    return objects == {
        ("scene_fact_members", "table"),
        ("idx_scene_fact_members_lookup", "index"),
        ("idx_scene_fact_members_order", "index"),
        ("trg_scene_fact_members_insert", "trigger"),
        ("trg_scene_fact_members_update", "trigger"),
        ("uq_scenes_profile_scene", "index"),
        ("uq_facts_profile_fact", "index"),
    }


def repair(conn: sqlite3.Connection) -> None:
    """Restore an accidentally dropped projection and re-backfill it."""
    conn.executescript(DDL)
