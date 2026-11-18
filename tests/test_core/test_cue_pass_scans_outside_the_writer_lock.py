# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""WS-7 writer-lock audit: the cue pass scans the store with no lock held and writes in one short hold.

Measured on a copy of the real store (43,888 facts, 833 prospective): the pass held the writer
lock for 2.5 s on every maintenance cycle, almost all of it the scan for plans that name a file
(its first-fact test was a correlated MIN() the planner ran against a full table walk: 3.5 to
10 s in isolation). The scan now runs on a read-only connection, through a query the planner
can drive from the kind index, and the lock is taken for the inserts only.
"""

from __future__ import annotations

import sqlite3

import pytest

from superlocalmemory.core import cue_binding as cb
from tests.test_core.test_cue_triggers import _memory, _rows, store  # noqa: F401 - fixture

#: The first-fact test as it was, kept here as the reference the new query must equal.
_REFERENCE = """
SELECT f.fact_id, f.profile_id, f.content FROM atomic_facts f
 WHERE f.memory_kind = 'prospective' AND f.quarantined = 0 AND f.lifecycle != 'archived'
   AND f.merged_into IS NULL
   AND f.rowid = (SELECT MIN(g.rowid) FROM atomic_facts g WHERE g.memory_id = f.memory_id)
   AND NOT EXISTS (SELECT 1 FROM memory_cues c WHERE c.fact_id = f.fact_id AND c.origin = ?)
 ORDER BY f.rowid DESC LIMIT ?
"""


def test_the_scans_run_with_no_lock_held_and_the_inserts_inside_it(store, tmp_path,  # noqa: F811
                                                                    monkeypatch) -> None:
    _memory(store, cues=["repo:proj"])
    _memory(store, "Edit docs/plan/notes.md before the release.", kind="prospective")
    lock = store._lock
    scans: list[bool] = []
    for name in ("_explicit_rows", "_extracted_rows"):
        real = getattr(cb, name)

        def spy(*args, _real=real, **kwargs):
            scans.append(lock._is_owned())
            return _real(*args, **kwargs)

        monkeypatch.setattr(cb, name, spy)
    inserts: list[bool] = []
    real_bind = cb._bind

    def bind(*args, **kwargs):
        inserts.append(lock._is_owned())
        return real_bind(*args, **kwargs)

    monkeypatch.setattr(cb, "_bind", bind)
    assert cb.run(store, tmp_path / "state") == 2
    assert scans == [False, False]
    assert inserts and all(inserts)


def test_the_new_scan_selects_exactly_what_the_old_one_did(store) -> None:  # noqa: F811
    for i in range(6):
        _memory(store, f"Edit docs/plan{i}/file{i}.md before the release.", kind="prospective",
                sentences=("a.", "b.")[: i % 3])
    _memory(store, "Not a plan, mentions docs/x/y.md", kind="statement")
    first = _memory(store, "Edit docs/z/w.md first.", kind="prospective")
    store.execute("UPDATE atomic_facts SET quarantined = 1 WHERE fact_id = ?", (first,))
    conn = sqlite3.connect(str(store.db_path))
    try:
        want = conn.execute(_REFERENCE, (cb.EXTRACTED, 500)).fetchall()
        got = conn.execute(cb._PROSPECTIVE, (cb.EXTRACTED, 500)).fetchall()
    finally:
        conn.close()
    assert want and got == want


def test_a_fact_erased_between_the_scan_and_the_write_gets_no_cue(store, tmp_path,  # noqa: F811
                                                                   monkeypatch) -> None:
    """The scan no longer holds the lock, so a fact can be erased before its cue is written.
    The cue is not written: an orphan cue would outlive the erasure (I-11)."""
    fact = _memory(store, cues=["repo:proj"])
    real = cb._explicit_rows

    def erase_after_scan(*args, **kwargs):
        out = real(*args, **kwargs)
        store.execute("DELETE FROM atomic_facts WHERE fact_id = ?", (fact,))
        return out

    monkeypatch.setattr(cb, "_explicit_rows", erase_after_scan)
    assert cb.run(store, tmp_path / "state") == 0
    assert _rows(store, "SELECT fact_id FROM memory_cues") == []


def test_the_watermark_is_unchanged_by_the_split(store, tmp_path) -> None:  # noqa: F811
    _memory(store, cues=["repo:one"])
    assert cb.run(store, tmp_path / "state") == 1
    _memory(store, cues=["repo:two"])
    assert cb.run(store, tmp_path / "state") == 1
    assert cb.run(store, tmp_path / "state") == 0


@pytest.mark.parametrize("name", ["bind_explicit", "bind_extracted"])
def test_the_one_connection_entry_points_still_work(store, name) -> None:  # noqa: F811
    _memory(store, cues=["repo:one"])
    _memory(store, "Edit docs/plan/a.md first.", kind="prospective")
    with store.raw_connection() as conn:
        out = getattr(cb, name)(conn)
    assert (out[0] if isinstance(out, tuple) else out) == 1


def test_a_store_before_the_cue_table_is_left_alone(store, tmp_path) -> None:  # noqa: F811
    _memory(store)
    conn = sqlite3.connect(str(store.db_path))
    try:
        conn.execute("DROP TRIGGER IF EXISTS trg_memory_cues_fact_erased")
        conn.execute("DROP TABLE memory_cues")
        conn.commit()
    finally:
        conn.close()
    assert cb.run(store, tmp_path / "state") == 0
