"""M068: bookkeeping stored before the rule learned its shape gets the flag.

M062 flagged what the stub rule knew in 4.2.2, and saves flag themselves since.
Shapes the rule learned later (agent checkpoints without "[agent]", "the
session ended ...", git-only Stop summaries and the copies derived from them)
were never re-checked on rows already stored: 1,764 + 100 such rows on one real
store, and recall leaves a row out of its answers by the flag. Runs on a
temporary store, never a live one.
"""

from __future__ import annotations

import sqlite3

import pytest

from superlocalmemory.storage import schema
from superlocalmemory.storage._migration_internals import _MODULES
from superlocalmemory.storage.database import DatabaseManager
from superlocalmemory.storage.migration_runner import DEFERRED_MIGRATIONS
from superlocalmemory.storage.migrations import M060_memory_description
from superlocalmemory.storage.migrations import M062_hook_bookkeeping_is_flagged as M062
from superlocalmemory.storage.migrations import M068_newer_bookkeeping_shapes_are_flagged as M068

#: Shapes M062's rule did not know; each one is a real row from the store above.
NEWER_STUBS = (
    "session ended (exit) at 2026-11-07 13:49 in config.",
    "the session ended 2026-11-04 16:39",
    "the user's session ended 2026-10-31 19:02",
    "[Agentic_official] session ended 2026-10-31 19:02 | branch: main"
    " | uncommitted: 2 files changed, 0 insertions(+), 0 deletions(-)",
    "[] session ended 2026-10-29 10:12 | branch: main | files: index.html",
    "2 files changed, 0 insertions(+), 0 deletions(-)",
)
ANSWERS = (
    "[Agentic_official] session ended 2026-10-13 11:01 | branch: main"
    " | uncommitted: 1 file changed | recent: 0545bc9a fix: retry",
    "The user's session ended early because the build broke.",
    "There were 2 files changed in the main branch",
    "Migrating the search index to the new embedding model.",
)


@pytest.fixture
def conn(tmp_path):
    manager = DatabaseManager(tmp_path / "memory.db")
    manager.initialize(schema)
    with manager.raw_connection() as c:
        M060_memory_description.apply(c)
    manager.close()
    connection = sqlite3.connect(tmp_path / "memory.db", isolation_level=None)
    connection.row_factory = sqlite3.Row
    yield connection
    connection.close()


def _add(conn, fact_id, content, non_answer=0, user=None):
    conn.execute("INSERT INTO memories (memory_id, profile_id, content) VALUES (?, 'default', ?)",
                 (f"m-{fact_id}", content))
    conn.execute(
        "INSERT INTO atomic_facts (fact_id, memory_id, profile_id, content, non_answer, "
        "non_answer_user) VALUES (?, ?, 'default', ?, ?, ?)",
        (fact_id, f"m-{fact_id}", content, non_answer, user))


def _flags(conn):
    return {r["fact_id"]: (r["non_answer"], r["non_answer_user"])
            for r in conn.execute("SELECT fact_id, non_answer, non_answer_user FROM atomic_facts")}


def test_registered_after_m062_with_a_floor_and_no_replay():
    assert _MODULES[M068.NAME] is M068
    migration = next(m for m in DEFERRED_MIGRATIONS if m.name == M068.NAME)
    assert {M060_memory_description.NAME, M062.NAME} <= set(migration.dependencies)
    assert M068.DB_TARGET == "memory" and M068.DOWNGRADE_FLOOR == 51
    assert M068.REPAIR_NOT_APPLICABLE.strip()


def test_rows_stored_before_the_rule_learned_their_shape_are_flagged(conn):
    for i, text in enumerate(NEWER_STUBS):
        _add(conn, f"s{i}", text)
    for i, text in enumerate(ANSWERS):
        _add(conn, f"a{i}", text)
    M068.apply(conn)
    flags = _flags(conn)
    assert all(flags[f"s{i}"] == (1, None) for i in range(len(NEWER_STUBS))), flags
    assert all(flags[f"a{i}"] == (0, None) for i in range(len(ANSWERS))), flags


def test_a_row_the_user_decided_about_is_never_touched(conn):
    _add(conn, "said-answer", NEWER_STUBS[3], non_answer=0, user=0)
    _add(conn, "plain", NEWER_STUBS[3])
    M068.apply(conn)
    flags = _flags(conn)
    assert flags["said-answer"] == (0, 0)
    assert flags["plain"] == (1, None)


def test_running_it_twice_changes_nothing_more(conn):
    for i, text in enumerate(NEWER_STUBS):
        _add(conn, f"s{i}", text)
    M068.apply(conn)
    first, before = _flags(conn), conn.total_changes
    M068.apply(conn)
    assert _flags(conn) == first and conn.total_changes == before


def test_the_runner_applies_it_to_a_store_that_predates_it(tmp_path):
    from superlocalmemory.storage import migration_runner as mr
    from tests.test_storage._upgrade_store import clear_snapshots, current_store

    learning, memory = current_store(tmp_path)
    with sqlite3.connect(memory) as c:
        c.row_factory = sqlite3.Row
        assert c.execute("SELECT status FROM migration_log WHERE name=?",
                         (M068.NAME,)).fetchone()["status"] == "complete"
        c.execute("DELETE FROM migration_log WHERE name=?", (M068.NAME,))  # a 4.2.15 store
        _add(c, "s0", NEWER_STUBS[3])
        _add(c, "a0", ANSWERS[0])
    result = mr.apply_deferred(learning, memory)
    clear_snapshots(tmp_path)
    assert M068.NAME in result["applied"] and result["failed"] == [], result
    with sqlite3.connect(memory) as c:
        c.row_factory = sqlite3.Row
        assert _flags(c) == {"s0": (1, None), "a0": (0, None)}
    again = mr.apply_deferred(learning, memory)
    clear_snapshots(tmp_path)
    assert M068.NAME not in again["applied"] and again["failed"] == []
