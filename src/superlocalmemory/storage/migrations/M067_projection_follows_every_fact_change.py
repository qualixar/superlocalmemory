# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""M067: a fact whose visibility or entities change is queued for the graph and vector copies.

Archive, merge, quarantine and lifecycle changes are plain UPDATEs in several modules, and none
queued a projection, so an archived memory kept its edges, entity bridge and vector in Cozo and
LanceDB (#296: the copy drifted to 4.3 M edges against 2.9 M). One trigger now queues the fact in
the writer's own transaction (``projection_outbox.install_change_trigger``). Deferred: it watches
columns that earlier migrations and engine init add.
"""

from __future__ import annotations

import sqlite3

NAME = "M067_projection_follows_every_fact_change"
DB_TARGET = "memory"
#: Additive (one trigger that only writes the existing queue), so going back stays possible.
DOWNGRADE_FLOOR = 51
DDL = """
-- One trigger, built from the columns present: apply() runs
-- projection_outbox.install_change_trigger.
"""


def apply(conn: sqlite3.Connection) -> None:
    from superlocalmemory.storage.projection_outbox import install_change_trigger

    install_change_trigger(conn)


def verify(conn: sqlite3.Connection) -> bool:
    from superlocalmemory.storage.projection_outbox import change_trigger_installed

    tables = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' "
        "AND name IN ('atomic_facts', 'projection_outbox')")}
    if tables != {"atomic_facts", "projection_outbox"}:
        return True  # nothing to watch yet; apply installs it once both exist
    return change_trigger_installed(conn)


def repair(conn: sqlite3.Connection) -> None:
    apply(conn)
