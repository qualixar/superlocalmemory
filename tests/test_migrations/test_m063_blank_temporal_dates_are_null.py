"""M063: a blank temporal date is stored as NULL, so the slice cache serves.

On the author's store 7 events with observation_date = '' marked the store
uncacheable (M057's non-ISO rule), and every recall re-read 5,000 events.
"""

import sqlite3

from superlocalmemory.storage.migrations import M057_temporal_retrieval_version as m057
from superlocalmemory.storage.migrations import M063_blank_temporal_dates_are_null as m063


def _store():
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE atomic_facts (fact_id TEXT, profile_id TEXT, scope TEXT, "
                 "shared_with TEXT, created_at TEXT)")
    conn.execute("CREATE TABLE temporal_events (event_id TEXT PRIMARY KEY, fact_id TEXT, "
                 "observation_date TEXT, referenced_date TEXT, interval_start TEXT, "
                 "interval_end TEXT)")
    for statement in m057._statements():
        conn.execute(statement)
    return conn


def _uncacheable(conn):
    return conn.execute("SELECT uncacheable FROM temporal_retrieval_version").fetchone()[0]


def test_blank_dates_stop_disabling_the_cache():
    conn = _store()
    conn.execute("INSERT INTO temporal_events VALUES ('e1','f1','',NULL,NULL,NULL)")
    conn.execute("INSERT INTO temporal_events VALUES ('e2','f2','2026-10-13',NULL,NULL,NULL)")
    conn.execute("INSERT INTO temporal_events VALUES ('e3','f3','now',NULL,NULL,NULL)")
    assert _uncacheable(conn) == 2  # the blank and the dynamic 'now'
    m063.apply(conn)
    assert conn.execute("SELECT observation_date FROM temporal_events WHERE event_id='e1'"
                        ).fetchone()[0] is None
    assert _uncacheable(conn) == 1  # 'now' still is dynamic
    m063.apply(conn)  # idempotent
    assert _uncacheable(conn) == 1


def test_a_new_save_stores_a_blank_date_as_null():
    from superlocalmemory.storage.database import _date_or_none

    assert _date_or_none("") is None and _date_or_none("  ") is None
    assert _date_or_none("2026-10-13") == "2026-10-13" and _date_or_none(None) is None
