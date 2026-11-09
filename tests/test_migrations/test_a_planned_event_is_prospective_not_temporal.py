# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""M046 rebuilds atomic_facts, so these tests run it against real tables.

A migration that drops a table is the one place where a test asserting "the
function was called" is worthless. Every test here builds a store with rows,
indexes, triggers and a search index, runs the migration, and then checks that
what came out the other side is the same data under a different name.
"""

from __future__ import annotations

import sqlite3

import pytest

from superlocalmemory.storage.migrations import (
    M046_prospective_memory_has_its_own_name as M046,
)

# A faithful reduction of the shipped table: TEXT primary key (so rowid is
# implicit, which is what makes the search index fragile), the CHECK that has to
# be widened, and the external-content FTS5 index keyed on rowid.
_CREATE = """
CREATE TABLE atomic_facts (
    fact_id     TEXT PRIMARY KEY,
    memory_id   TEXT NOT NULL,
    profile_id  TEXT NOT NULL DEFAULT 'default',
    content     TEXT NOT NULL,
    fact_type   TEXT NOT NULL DEFAULT 'semantic'
                    CHECK (fact_type IN (
                        'episodic', 'semantic', 'opinion', 'temporal'
                    )),
    referenced_date TEXT,
    confidence  REAL NOT NULL DEFAULT 1.0
);
CREATE INDEX idx_af_profile ON atomic_facts (profile_id, fact_type);
CREATE VIRTUAL TABLE atomic_facts_fts USING fts5(
    fact_id UNINDEXED, content,
    content='atomic_facts', content_rowid='rowid'
);
CREATE TRIGGER atomic_facts_fts_insert AFTER INSERT ON atomic_facts BEGIN
    INSERT INTO atomic_facts_fts (rowid, fact_id, content)
    VALUES (new.rowid, new.fact_id, new.content);
END;
CREATE TRIGGER atomic_facts_fts_delete AFTER DELETE ON atomic_facts BEGIN
    INSERT INTO atomic_facts_fts (atomic_facts_fts, rowid, fact_id, content)
    VALUES ('delete', old.rowid, old.fact_id, old.content);
END;
"""

_ROWS = [
    ("f1", "m1", "default", "Dentist appointment on the 4th", "temporal", "2026-09-04", 1.0),
    ("f2", "m1", "default", "Renew the passport before the deadline", "temporal", "2026-10-01", 0.9),
    ("f3", "m2", "default", "Paris is the capital of France", "semantic", None, 1.0),
    ("f4", "m2", "other", "Went to the market on Tuesday", "episodic", None, 0.8),
    ("f5", "m3", "default", "I think the new pipeline is faster", "opinion", None, 0.6),
    ("f6", "m3", "default", "Quarterly review scheduled for March", "temporal", "2026-03-15", 0.7),
]


def _store(path, *, rows=_ROWS) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path))
    conn.executescript(_CREATE)
    conn.executemany(
        "INSERT INTO atomic_facts "
        "(fact_id, memory_id, profile_id, content, fact_type, referenced_date, confidence) "
        "VALUES (?,?,?,?,?,?,?)",
        rows,
    )
    conn.commit()
    return conn


@pytest.fixture
def store(tmp_path):
    conn = _store(tmp_path / "memory.db")
    yield conn
    conn.close()


def _count(conn, sql, *args) -> int:
    return int(conn.execute(sql, args).fetchone()[0])


# ---------------------------------------------------------------------------
# The conversion
# ---------------------------------------------------------------------------

def test_every_planned_event_is_renamed_and_nothing_else_is(store):
    before_total = _count(store, "SELECT COUNT(*) FROM atomic_facts")
    before_planned = _count(
        store, "SELECT COUNT(*) FROM atomic_facts WHERE fact_type='temporal'")
    assert before_planned == 3

    M046.apply(store)

    assert _count(store, "SELECT COUNT(*) FROM atomic_facts") == before_total
    assert _count(
        store, "SELECT COUNT(*) FROM atomic_facts WHERE fact_type='temporal'") == 0
    assert _count(
        store,
        "SELECT COUNT(*) FROM atomic_facts WHERE fact_type='prospective'",
    ) == before_planned
    # The other three types are untouched.
    for kind, n in (("semantic", 1), ("episodic", 1), ("opinion", 1)):
        assert _count(
            store, "SELECT COUNT(*) FROM atomic_facts WHERE fact_type=?", kind
        ) == n


def test_no_column_and_no_row_content_is_lost(store):
    before = store.execute(
        "SELECT fact_id, memory_id, profile_id, content, referenced_date, "
        "confidence FROM atomic_facts ORDER BY fact_id"
    ).fetchall()

    M046.apply(store)

    after = store.execute(
        "SELECT fact_id, memory_id, profile_id, content, referenced_date, "
        "confidence FROM atomic_facts ORDER BY fact_id"
    ).fetchall()
    assert after == before


def test_the_new_value_can_actually_be_written_afterwards(store):
    """The point of the rebuild. Before it, this INSERT is rejected."""
    with pytest.raises(sqlite3.IntegrityError):
        store.execute(
            "INSERT INTO atomic_facts (fact_id, memory_id, content, fact_type) "
            "VALUES ('new1','m9','Book the flight','prospective')"
        )
    store.rollback()

    M046.apply(store)

    store.execute(
        "INSERT INTO atomic_facts (fact_id, memory_id, content, fact_type) "
        "VALUES ('new1','m9','Book the flight','prospective')"
    )
    assert _count(
        store, "SELECT COUNT(*) FROM atomic_facts WHERE fact_id='new1'") == 1


def test_the_old_value_is_rejected_afterwards(store):
    """This is what stops an older process writing the wrong word and winning.

    Without it the constraint would accept both and the old build's memories
    would keep landing under a name nothing looks for.
    """
    M046.apply(store)
    with pytest.raises(sqlite3.IntegrityError):
        store.execute(
            "INSERT INTO atomic_facts (fact_id, memory_id, content, fact_type) "
            "VALUES ('old1','m9','Dentist again','temporal')"
        )


# ---------------------------------------------------------------------------
# The things a rebuild quietly breaks
# ---------------------------------------------------------------------------

def test_the_search_index_still_points_at_the_right_facts(store):
    """The rowid trap.

    The index is external-content FTS5 keyed on rowid, and the primary key is
    TEXT — so the table's rowid is implicit and ``SELECT *`` does not carry it.
    A copy that lets SQLite assign fresh rowids leaves every index entry
    pointing at a different fact. Searches keep working and return the wrong
    rows, and nothing raises.
    """
    M046.apply(store)

    hits = store.execute(
        "SELECT af.fact_id, af.content FROM atomic_facts_fts fts "
        "JOIN atomic_facts af ON af.rowid = fts.rowid "
        "WHERE atomic_facts_fts MATCH 'deadline'"
    ).fetchall()
    assert hits, "the search index found nothing after the rebuild"
    for fact_id, content in hits:
        assert "deadline" in content.lower(), (
            f"the index matched {fact_id} on 'deadline' but its content is "
            f"{content!r} — rowids were reassigned"
        )


def test_rowids_are_preserved_exactly(store):
    before = dict(store.execute("SELECT fact_id, rowid FROM atomic_facts"))
    M046.apply(store)
    after = dict(store.execute("SELECT fact_id, rowid FROM atomic_facts"))
    assert after == before


def test_indexes_and_triggers_come_back(store):
    def names(kind):
        return {
            r[0] for r in store.execute(
                "SELECT name FROM sqlite_master WHERE tbl_name='atomic_facts' "
                "AND type=? AND sql IS NOT NULL", (kind,)
            )
        }

    before_idx, before_trg = names("index"), names("trigger")
    assert before_idx and before_trg

    M046.apply(store)

    assert names("index") == before_idx
    assert names("trigger") == before_trg


def test_the_restored_trigger_still_fires(store):
    """A trigger recreated but broken would only show up on the next write."""
    M046.apply(store)
    store.execute(
        "INSERT INTO atomic_facts (fact_id, memory_id, content, fact_type) "
        "VALUES ('t1','m9','A brand new searchable sentence','semantic')"
    )
    hits = store.execute(
        "SELECT fact_id FROM atomic_facts_fts WHERE atomic_facts_fts MATCH 'searchable'"
    ).fetchall()
    assert ("t1",) in hits


# ---------------------------------------------------------------------------
# Safety properties
# ---------------------------------------------------------------------------

def test_running_it_twice_changes_nothing(store):
    M046.apply(store)
    snapshot = store.execute(
        "SELECT fact_id, fact_type, rowid FROM atomic_facts ORDER BY fact_id"
    ).fetchall()

    M046.apply(store)

    assert store.execute(
        "SELECT fact_id, fact_type, rowid FROM atomic_facts ORDER BY fact_id"
    ).fetchall() == snapshot
    assert M046.verify(store)


def test_verify_fails_before_and_passes_after(store):
    assert not M046.verify(store)
    M046.apply(store)
    assert M046.verify(store)


def test_a_store_without_the_table_is_left_alone(tmp_path):
    """A fresh install runs migrations before the schema exists.

    A migration whose verify() demands rows it cannot create fails forever on a
    new machine — the defect a migration in the previous release shipped with.
    """
    conn = sqlite3.connect(str(tmp_path / "empty.db"))
    try:
        M046.apply(conn)
        assert M046.verify(conn)
    finally:
        conn.close()


def test_a_store_with_no_planned_events_migrates_cleanly(tmp_path):
    conn = _store(tmp_path / "m.db", rows=[r for r in _ROWS if r[4] != "temporal"])
    try:
        M046.apply(conn)
        assert M046.verify(conn)
        assert _count(conn, "SELECT COUNT(*) FROM atomic_facts") == 3
    finally:
        conn.close()


def test_a_rewrite_that_does_not_take_refuses_rather_than_dropping_the_table(tmp_path):
    """The catastrophic path, forced.

    If the constraint rewrite silently produces the ORIGINAL constraint, the
    staging table carries it and the real table is then dropped — every planned
    event lost to a copy that rejects them. The rebuild must stop instead.

    Forced by calling ``_rebuild`` on a table whose definition contains no
    occurrence of the old value, so the rename cannot produce the new one. That
    is the exact condition the guard exists for.
    """
    conn = sqlite3.connect(str(tmp_path / "norewrite.db"))
    try:
        conn.execute(
            "CREATE TABLE atomic_facts ("
            " fact_id TEXT PRIMARY KEY, memory_id TEXT, content TEXT,"
            " fact_type TEXT NOT NULL DEFAULT 'episodic')"
        )
        conn.execute("INSERT INTO atomic_facts VALUES ('f1','m1','a note','episodic')")
        conn.commit()

        with pytest.raises(sqlite3.Error, match="could not rewrite"):
            M046._rebuild(conn)

        # The original table and its row are untouched.
        assert _count(conn, "SELECT COUNT(*) FROM atomic_facts") == 1
        assert conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='atomic_facts'"
        ).fetchone() is not None
    finally:
        conn.close()


def test_a_short_copy_rolls_back_rather_than_committing(store, monkeypatch):
    """If the copy loses rows, nothing is committed.

    Verified by making the count check see a mismatch, which is the guard that
    stands between a partial copy and a dropped original.
    """
    real_count = M046._count
    calls = {"n": 0}

    def lying_count(conn, sql):
        calls["n"] += 1
        # The second count is the staging table's, taken after the copy.
        if calls["n"] == 2:
            return 1
        return real_count(conn, sql)

    monkeypatch.setattr(M046, "_count", lying_count)

    with pytest.raises(sqlite3.Error):
        M046._rebuild(store)

    assert _count(store, "SELECT COUNT(*) FROM atomic_facts") == len(_ROWS)
    assert _count(
        store, "SELECT COUNT(*) FROM atomic_facts WHERE fact_type='temporal'") == 3


def test_it_puts_the_connections_own_pragma_back(store):
    """The migration borrows the connection; it does not own its settings.

    An unconditional ``foreign_keys=ON`` at the end leaves a caller that had
    them off with them on, which changes how every later statement on that
    connection behaves.
    """
    for original in (True, False):
        store.execute(f"PRAGMA foreign_keys={'ON' if original else 'OFF'}")
        assert bool(store.execute("PRAGMA foreign_keys").fetchone()[0]) is original
        # Re-create the pre-migration shape so _rebuild has work to do.
        store.executescript(
            "DROP TABLE IF EXISTS atomic_facts_fts;"
            "DROP TABLE IF EXISTS atomic_facts;"
        )
        store.executescript(_CREATE)
        store.executemany(
            "INSERT INTO atomic_facts (fact_id, memory_id, profile_id, content, "
            "fact_type, referenced_date, confidence) VALUES (?,?,?,?,?,?,?)",
            _ROWS,
        )
        store.commit()

        M046.apply(store)

        assert bool(store.execute("PRAGMA foreign_keys").fetchone()[0]) is original, (
            f"foreign_keys was {original} before the migration and is not after"
        )


# ---------------------------------------------------------------------------
# How the constraint is SPELLED must not change what the migration does
# ---------------------------------------------------------------------------

_CHECK_SPELLINGS = {
    "bare": "CHECK (fact_type IN ('episodic','semantic','opinion','temporal'))",
    "double_quoted": 'CHECK ("fact_type" IN (\'episodic\',\'semantic\',\'opinion\',\'temporal\'))',
    "bracketed": "CHECK ([fact_type] IN ('episodic','semantic','opinion','temporal'))",
    "backticked": "CHECK (`fact_type` IN ('episodic','semantic','opinion','temporal'))",
    "collated": "CHECK (fact_type COLLATE NOCASE IN ('episodic','semantic','opinion','temporal'))",
    "multiline": (
        "CHECK (\n   fact_type   IN (\n 'episodic',\n 'semantic',\n"
        " 'opinion',\n 'temporal'\n )\n)"
    ),
    "absent": "",
    "already_migrated": "CHECK (fact_type IN ('episodic','semantic','opinion','prospective'))",
}


def _store_with_check(path, check: str, *, with_rows: bool) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path))
    conn.execute(
        "CREATE TABLE atomic_facts ("
        " fact_id TEXT PRIMARY KEY, memory_id TEXT, content TEXT,"
        f" fact_type TEXT NOT NULL DEFAULT 'episodic' {check})"
    )
    conn.execute("INSERT INTO atomic_facts VALUES ('s1','m1','a note','episodic')")
    if with_rows and "prospective" not in check:
        conn.execute(
            "INSERT INTO atomic_facts VALUES ('f1','m1','renew passport','temporal')")
    conn.commit()
    return conn


@pytest.mark.parametrize("spelling", sorted(_CHECK_SPELLINGS))
@pytest.mark.parametrize("with_rows", [True, False], ids=["with_rows", "empty"])
def test_the_constraints_spelling_does_not_change_the_outcome(
    tmp_path, spelling, with_rows,
):
    """A quoted identifier used to make this migration corrupt the store.

    The original code decided whether the constraint blocked the new value by
    matching its DDL text with a pattern that required a bare ``fact_type``. On a
    store written ``CHECK ("fact_type" IN (...))`` the pattern missed, and the
    migration took the cheap UPDATE path. Two outcomes, both bad:

    * with rows to convert, the UPDATE raised and the runner retried forever;
    * with none, it committed and ``verify()`` reported success while the table
      went on rejecting the new value — so every planned event stored afterwards
      was lost, and nothing repaired it, because the pattern answered the same
      way every run.

    A constraint's behaviour is now asked of SQLite instead of read out of its
    text, so the spelling cannot matter. This is the test that says so.
    """
    conn = _store_with_check(
        tmp_path / f"{spelling}.db", _CHECK_SPELLINGS[spelling],
        with_rows=with_rows,
    )
    try:
        M046.apply(conn)

        assert M046.verify(conn), f"verify() failed for a {spelling} constraint"
        assert _count(
            conn, "SELECT COUNT(*) FROM atomic_facts WHERE fact_type='temporal'"
        ) == 0, "a planned event was left under the old name"

        expected = 1 if (with_rows and "prospective" not in _CHECK_SPELLINGS[spelling]) else 0
        assert _count(
            conn, "SELECT COUNT(*) FROM atomic_facts WHERE fact_type='prospective'"
        ) == expected

        # The property that matters to the user: a planned event can be stored.
        conn.execute(
            "INSERT INTO atomic_facts VALUES ('p1','m1','book the flight','prospective')")
    finally:
        conn.close()


@pytest.mark.parametrize("spelling", ["double_quoted", "bracketed", "collated"])
def test_the_old_value_is_still_rejected_whatever_the_spelling(tmp_path, spelling):
    """The rebuild has to reach constraints the old code could not even parse."""
    conn = _store_with_check(
        tmp_path / f"{spelling}-rej.db", _CHECK_SPELLINGS[spelling], with_rows=True,
    )
    try:
        M046.apply(conn)
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO atomic_facts VALUES ('o1','m1','old','temporal')")
    finally:
        conn.close()


def test_the_probe_leaves_no_trace(tmp_path):
    """Asking "would this be accepted" must not change anything.

    The question is answered by attempting a write inside a savepoint and rolling
    it back. A savepoint that leaked would either hold a write lock or, worse,
    commit the probe.
    """
    conn = _store_with_check(
        tmp_path / "probe.db", _CHECK_SPELLINGS["bare"], with_rows=True,
    )
    try:
        before = conn.execute(
            "SELECT fact_id, fact_type FROM atomic_facts ORDER BY fact_id"
        ).fetchall()

        assert M046._accepts(conn, "prospective") is False  # blocked pre-migration
        assert M046._accepts(conn, "episodic") is True

        after = conn.execute(
            "SELECT fact_id, fact_type FROM atomic_facts ORDER BY fact_id"
        ).fetchall()
        assert after == before, "the probe changed the data it was asking about"
        assert not conn.in_transaction, "the probe left a transaction open"
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# The rowid trap, made reproducible
#
# The tests above pass even if the copy stops naming `rowid`, because the
# fixture only ever inserts: SQLite hands out 1..N either way, so "preserved"
# and "reassigned" are the same numbers. And the index rebuild at the end
# repairs a wrong index before anything looks at it.
#
# A store that has ever deleted a memory has gaps. That is where reassignment
# shows, and where a failed rebuild leaves searches pointing at the wrong facts
# with nothing raised and verify() reporting success.
# ---------------------------------------------------------------------------

_GAPPED = [
    ("g1", "m1", "default", "alpha unique-aaa", "semantic", None, 1.0),
    ("g2", "m1", "default", "beta unique-bbb", "semantic", None, 1.0),
    ("g3", "m1", "default", "gamma deadline unique-ccc", "temporal", "2026-09-04", 1.0),
    ("g4", "m1", "default", "delta unique-ddd", "semantic", None, 1.0),
    ("g5", "m1", "default", "epsilon unique-eee", "semantic", None, 1.0),
]


@pytest.fixture
def gapped_store(tmp_path):
    """A store whose rowids have holes, as any real store's do."""
    conn = sqlite3.connect(str(tmp_path / "gapped.db"))
    conn.executescript(_CREATE)
    conn.executemany(
        "INSERT INTO atomic_facts (fact_id, memory_id, profile_id, content, "
        "fact_type, referenced_date, confidence) VALUES (?,?,?,?,?,?,?)",
        _GAPPED,
    )
    # Two memories forgotten, which is what makes the rowids non-contiguous.
    conn.execute("DELETE FROM atomic_facts WHERE fact_id IN ('g1','g2')")
    conn.commit()
    yield conn
    conn.close()


def test_the_gapped_fixture_really_is_gapped(gapped_store):
    """Guard the guard: if the holes vanish, the tests below go vacuous again."""
    rowids = [r[0] for r in gapped_store.execute(
        "SELECT rowid FROM atomic_facts ORDER BY rowid")]
    assert rowids == [3, 4, 5], f"expected holes at 1-2, got {rowids}"


def test_rowids_survive_the_rebuild_when_there_are_holes(gapped_store):
    before = dict(gapped_store.execute("SELECT fact_id, rowid FROM atomic_facts"))
    M046.apply(gapped_store)
    after = dict(gapped_store.execute("SELECT fact_id, rowid FROM atomic_facts"))
    assert after == before, (
        f"rowids were reassigned: {before} became {after}. The search index is "
        f"keyed on these, so every entry now points at a different fact."
    )


def test_search_finds_the_right_fact_even_if_the_index_rebuild_fails(
    gapped_store, monkeypatch,
):
    """The failure the module docstring describes, actually constructed.

    The index rebuild is deliberately non-fatal — losing the converted rows to a
    rollback over it would be the worse trade. That makes preserving rowid the
    only thing standing between a store with deletions and a search that
    silently returns the wrong memory. So this disables the rebuild and checks
    the copy alone got it right.
    """
    monkeypatch.setattr(M046, "_rebuild_fts", lambda conn: None)

    M046.apply(gapped_store)

    hits = gapped_store.execute(
        "SELECT af.fact_id, af.content FROM atomic_facts_fts fts "
        "JOIN atomic_facts af ON af.rowid = fts.rowid "
        "WHERE atomic_facts_fts MATCH 'deadline'"
    ).fetchall()
    assert hits, "the search index found nothing at all"
    for fact_id, content in hits:
        assert fact_id == "g3", (
            f"searching 'deadline' returned {fact_id} ({content!r}); the only "
            f"fact containing it is g3. Rowids were reassigned and the index "
            f"now points at the wrong memories."
        )


def test_a_copy_that_drops_rowid_is_caught(gapped_store, monkeypatch):
    """Prove these tests would fail if the fix were removed.

    Without this, "rowid is preserved" rests on the current code happening to
    say so. Here the rowid is deliberately left out of the copy — exactly the
    edit a future refactor might make — and the assertion above must fail.
    """
    monkeypatch.setattr(M046, "_rebuild_fts", lambda conn: None)


    def rebuild_without_rowid(conn):
        # Reproduce _rebuild, minus naming rowid on either side.
        original = M046._table_sql(conn, "atomic_facts")
        staged = original.replace("'temporal'", "'prospective'").replace(
            "atomic_facts", "atomic_facts_m046_new", 1)
        cols = [c for c in M046._columns(conn, "atomic_facts")]
        sel = ", ".join(
            "CASE WHEN fact_type='temporal' THEN 'prospective' ELSE fact_type END"
            if c == "fact_type" else f'"{c}"' for c in cols)
        ins = ", ".join(f'"{c}"' for c in cols)
        for kind, name in M046._trigger_and_index_names(conn):
            conn.execute(f'DROP {kind} IF EXISTS "{name}"')
        conn.execute(staged)
        conn.execute(
            f'INSERT INTO "atomic_facts_m046_new" ({ins}) '
            f"SELECT {sel} FROM atomic_facts")
        conn.execute("DROP TABLE atomic_facts")
        conn.execute('ALTER TABLE "atomic_facts_m046_new" RENAME TO atomic_facts')
        conn.commit()

    rebuild_without_rowid(gapped_store)

    hits = gapped_store.execute(
        "SELECT af.fact_id FROM atomic_facts_fts fts "
        "JOIN atomic_facts af ON af.rowid = fts.rowid "
        "WHERE atomic_facts_fts MATCH 'deadline'"
    ).fetchall()
    wrong = [f for (f,) in hits if f != "g3"]
    assert wrong or not hits, (
        "dropping rowid from the copy did NOT corrupt the index, so the tests "
        "asserting it is preserved cannot fail and prove nothing"
    )


# ---------------------------------------------------------------------------
# Table shapes the shipped schema does not use, but a store might
# ---------------------------------------------------------------------------

def test_a_table_without_a_rowid_still_migrates(tmp_path):
    """Naming rowid in the copy fails outright on such a table.

    The failure is a rollback, so no data is lost — but the migration is then
    marked failed and retried on every start, forever, which is the same
    permanently-stuck state a previous release shipped.

    Such a table cannot carry the external-content search index either, so there
    is no rowid to preserve and nothing is given up by not naming it.
    """
    conn = sqlite3.connect(str(tmp_path / "norowid.db"))
    try:
        conn.execute(
            "CREATE TABLE atomic_facts ("
            " fact_id TEXT PRIMARY KEY, memory_id TEXT, content TEXT,"
            " fact_type TEXT NOT NULL DEFAULT 'episodic'"
            "   CHECK (fact_type IN ('episodic','semantic','opinion','temporal'))"
            ") WITHOUT ROWID"
        )
        conn.execute(
            "INSERT INTO atomic_facts VALUES ('f1','m1','renew passport','temporal')")
        conn.commit()

        M046.apply(conn)

        assert M046.verify(conn)
        assert _count(
            conn, "SELECT COUNT(*) FROM atomic_facts WHERE fact_type='prospective'"
        ) == 1
    finally:
        conn.close()


def test_a_generated_column_does_not_break_the_copy(tmp_path):
    """Inserting into a generated column is an error, so this looked dangerous.

    It is not, and the reason is worth pinning: ``PRAGMA table_info`` does not
    return generated columns at all — they appear only in ``table_xinfo``,
    flagged hidden — so the column list the copy is built from never includes
    one. If that ever changes, this fails rather than a user's upgrade.
    """
    conn = sqlite3.connect(str(tmp_path / "generated.db"))
    try:
        conn.execute(
            "CREATE TABLE atomic_facts ("
            " fact_id TEXT PRIMARY KEY, memory_id TEXT, content TEXT,"
            " fact_type TEXT NOT NULL DEFAULT 'episodic'"
            "   CHECK (fact_type IN ('episodic','semantic','opinion','temporal')))"
        )
        conn.execute(
            "ALTER TABLE atomic_facts ADD COLUMN search_key TEXT "
            "GENERATED ALWAYS AS (lower(content)) VIRTUAL")
        conn.execute(
            "INSERT INTO atomic_facts (fact_id, memory_id, content, fact_type) "
            "VALUES ('f1','m1','Renew Passport','temporal')")
        conn.commit()

        assert "search_key" not in M046._columns(conn, "atomic_facts"), (
            "table_info now returns generated columns; the copy will try to "
            "insert into one and every upgrade on such a store will fail"
        )

        M046.apply(conn)

        assert M046.verify(conn)
        assert conn.execute(
            "SELECT search_key FROM atomic_facts").fetchone()[0] == "renew passport"
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# A rebuild that cannot put the triggers back must not commit
# ---------------------------------------------------------------------------

def test_a_failed_trigger_replay_rolls_the_whole_rebuild_back(store, monkeypatch):
    """Reverses an earlier judgement in this module, which was wrong.

    That judgement was that a missing index is recoverable while the converted
    rows are not, so a failed replay should warn and commit. But a rollback
    loses nothing — the original table is still there and the migration is
    retried. Committing is what costs something: two of the three replayed
    statements are the triggers that keep the search index in step with the
    table, so a store that commits without them stops indexing every memory
    written from then on, and lexical search goes quietly blind to new writes.

    The failure is injected by corrupting the captured DDL rather than by
    patching the connection: ``sqlite3.Connection`` is a C type, and replacing
    its ``execute`` crashes the interpreter outright.
    """
    real_dependents = M046._dependents

    def one_broken_trigger(conn):
        captured = real_dependents(conn)
        assert captured, "nothing to replay, so this test proves nothing"
        return captured[:-1] + ["CREATE TRIGGER not_valid_sql BEGIN SELECT"]

    monkeypatch.setattr(M046, "_dependents", one_broken_trigger)

    with pytest.raises(sqlite3.Error):
        M046.apply(store)
    monkeypatch.undo()

    # Nothing was committed: the original table, its rows and its triggers are
    # all still there, and the migration will be retried.
    assert _count(store, "SELECT COUNT(*) FROM atomic_facts") == len(_ROWS)
    assert _count(
        store, "SELECT COUNT(*) FROM atomic_facts WHERE fact_type='temporal'") == 3
    triggers = {r[0] for r in store.execute(
        "SELECT name FROM sqlite_master WHERE type='trigger' "
        "AND tbl_name='atomic_facts'")}
    assert "atomic_facts_fts_insert" in triggers
    assert not M046.verify(store)


def test_verify_refuses_a_store_whose_index_triggers_are_gone(store):
    """The end state includes being able to keep the search index current.

    A store with the rows converted and the triggers missing satisfies every
    other condition — no old values, constraint correct — while silently not
    indexing anything new. That is not migrated.
    """
    M046.apply(store)
    assert M046.verify(store)

    store.execute("DROP TRIGGER atomic_facts_fts_insert")
    store.execute("DROP TRIGGER atomic_facts_fts_delete")
    store.commit()

    assert not M046.verify(store), (
        "verify() blessed a store that has stopped feeding its search index"
    )


def test_the_runner_waits_for_a_busy_database(tmp_path):
    """A rebuild takes an exclusive lock, so it collides with a live daemon.

    Without a busy timeout SQLite raises immediately, the runner records the
    migration failed, and the upgrade is wedged until someone notices. The
    timeout turns a collision into a wait.
    """
    from superlocalmemory.storage._migration_internals import (
        _MIGRATION_BUSY_TIMEOUT_MS,
        _connect,
    )

    db = tmp_path / "busy.db"
    conn = _connect(db)
    try:
        timeout = conn.execute("PRAGMA busy_timeout").fetchone()[0]
        assert timeout == _MIGRATION_BUSY_TIMEOUT_MS, (
            f"the runner's connection waits {timeout}ms for a locked database; "
            f"expected {_MIGRATION_BUSY_TIMEOUT_MS}ms"
        )
        assert timeout > 0
    finally:
        conn.close()


class TestARebuildSurvivesTriggersThatPointAtTheTable:
    """A trigger elsewhere that joins this table makes the rename fail.

    Since SQLite 3.25, ``ALTER TABLE ... RENAME TO`` reparses every trigger and
    view in the schema so it can fix up their references. The rebuild drops the
    old table first, so at that moment any trigger that joins it names something
    that no longer exists, and the whole statement fails:

        error in trigger trg_scene_fact_members_insert:
        no such table: main.atomic_facts

    Two such triggers ship on ``memory_scenes``. Running this migration on its
    own never reached them, because the migration that creates them had not run
    yet — it took the whole chain against a real archive to find it.
    """

    @staticmethod
    def _store_with_a_trigger_pointing_at_the_table(tmp_path):
        db = tmp_path / "memory.db"
        conn = sqlite3.connect(db)
        conn.executescript(
            """
            CREATE TABLE atomic_facts (
                fact_id TEXT PRIMARY KEY,
                profile_id TEXT NOT NULL DEFAULT 'default',
                memory_id TEXT,
                content TEXT,
                fact_type TEXT NOT NULL DEFAULT 'semantic'
                    CHECK (fact_type IN ('episodic','semantic','opinion','temporal'))
            );
            CREATE TABLE memory_scenes (
                scene_id TEXT PRIMARY KEY, profile_id TEXT, fact_ids_json TEXT);
            CREATE TABLE scene_fact_members (
                profile_id TEXT, scene_id TEXT, fact_id TEXT, position INTEGER);
            CREATE TRIGGER trg_scene_fact_members_insert
            AFTER INSERT ON memory_scenes
            BEGIN
                DELETE FROM scene_fact_members WHERE scene_id = NEW.scene_id;
                INSERT OR IGNORE INTO scene_fact_members
                    (profile_id, scene_id, fact_id, position)
                SELECT NEW.profile_id, NEW.scene_id, af.fact_id, 0
                FROM atomic_facts AS af
                WHERE af.profile_id = NEW.profile_id;
            END;
            """
        )
        conn.execute(
            "INSERT INTO atomic_facts (fact_id, profile_id, memory_id, content, "
            "fact_type) VALUES ('f1','default','m1','a deadline','temporal')"
        )
        conn.commit()
        return conn

    def test_the_rebuild_completes(self, tmp_path) -> None:
        conn = self._store_with_a_trigger_pointing_at_the_table(tmp_path)
        try:
            M046.apply(conn)
            assert M046.verify(conn) is True
            assert conn.execute(
                "SELECT fact_type FROM atomic_facts WHERE fact_id='f1'"
            ).fetchone()[0] == "prospective"
        finally:
            conn.close()

    def test_the_trigger_is_put_back_and_still_works(self, tmp_path) -> None:
        conn = self._store_with_a_trigger_pointing_at_the_table(tmp_path)
        try:
            M046.apply(conn)

            names = {
                r[0] for r in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='trigger'"
                )
            }
            assert "trg_scene_fact_members_insert" in names, (
                "the trigger was dropped to allow the rebuild and never replaced"
            )

            # Replaced is not the same as working.
            conn.execute(
                "INSERT INTO memory_scenes VALUES ('s1','default','[\"f1\"]')"
            )
            conn.commit()
            members = conn.execute(
                "SELECT fact_id FROM scene_fact_members WHERE scene_id='s1'"
            ).fetchall()
            assert members == [("f1",)], (
                "the trigger exists but no longer populates the derived table"
            )
        finally:
            conn.close()
