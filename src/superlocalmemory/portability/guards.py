# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""What an imported memory may and may not do to the memory that was already there (D-19, M-15).

Imported facts sit at the bottom of the origin lattice (``imported_file``): they never move a
slot head, and a value that contradicts what a slot already holds opens a dispute rather than
replacing it. This module is the slot resolver's one call for a fact that belongs to an import
run (``truth/slot_resolver.resolve_fact``). It does not depend on ``trust.taint``: the guard for
imported text holds whichever way that flag is set.

* The dispute is relabelled ``origin_below_head``. No judge answers such a dispute (the dispute
  pass skips that reason), so a run causes no judge calls (``JUDGE_CALLS_PER_RUN`` is 0) and
  only a recorded decision settles it (``slm disputes settle``, MCP ``settle_dispute``, the
  dashboard, any caller, D-49).
* A run lists at most ``DISPUTES_PER_RUN`` of them. Past the cap the contradiction is dismissed
  (the imported text is still kept, the slot head is untouched) and the run counts how many
  it left out, so a large import cannot bury the review queue.
* A value older than the head leaves the head where it is whatever the origin, so it is not
  touched here.

Ids and counts only; no memory text is read.
"""

from __future__ import annotations

import sqlite3

from superlocalmemory.portability import runs

HELD = "import_held"
UNLISTED = "import_unlisted"
#: The dispute reason the origin lattice uses for "waits for a recorded decision".
ORIGIN_BELOW_HEAD = "origin_below_head"


def run_of_fact(conn: sqlite3.Connection, fact_id: str) -> tuple[str, int] | None:
    """``(run_id, dispute_cap)`` of the live import run that saved this fact's memory, if any."""
    try:
        row = conn.execute(
            "SELECT r.run_id, r.dispute_cap FROM atomic_facts f "
            "JOIN import_run_items i ON i.memory_id = f.memory_id "
            "JOIN import_runs r ON r.run_id = i.run_id WHERE f.fact_id = ? LIMIT 1",
            (fact_id,)).fetchone()
    except sqlite3.OperationalError:  # a store before M085
        return None
    return (str(row[0]), int(row[1])) if row else None


def hold(conn: sqlite3.Connection, *, fact_id: str, dispute_id: str,
         older_reasons: frozenset[str]) -> str | None:
    """Apply the import rules to a dispute a fact just opened. ``None``: not an import fact."""
    found = run_of_fact(conn, fact_id)
    if found is None:
        return None
    run_id, cap = found
    reason = conn.execute("SELECT reason_code FROM slot_disputes WHERE dispute_id = ?",
                          (dispute_id,)).fetchone()
    if reason is None or reason[0] in older_reasons:
        return None
    opened = conn.execute("SELECT disputes_opened FROM import_runs WHERE run_id = ?",
                          (run_id,)).fetchone()[0]
    if int(opened) >= cap:
        _dismiss(conn, dispute_id)
        runs.bump(conn, run_id, "disputes_unlisted")
        return UNLISTED
    conn.execute("UPDATE slot_disputes SET reason_code = ? WHERE dispute_id = ? AND state = 'open'",
                 (ORIGIN_BELOW_HEAD, dispute_id))
    runs.bump(conn, run_id, "disputes_opened")
    return HELD


def _dismiss(conn: sqlite3.Connection, dispute_id: str) -> None:
    """Close the dispute without a verdict and put the slot's state back in step."""
    from superlocalmemory.truth import slot_store

    slot_id = conn.execute("SELECT slot_id FROM slot_disputes WHERE dispute_id = ?",
                           (dispute_id,)).fetchone()[0]
    slot_store.dismiss_dispute(conn, dispute_id, by="import:cap")
    still_open = conn.execute("SELECT 1 FROM slot_disputes WHERE slot_id = ? AND state = 'open'",
                              (slot_id,)).fetchone()
    conn.execute("UPDATE memory_slots SET state = ? WHERE slot_id = ?",
                 ("disputed" if still_open else "current", slot_id))


__all__ = ["HELD", "ORIGIN_BELOW_HEAD", "UNLISTED", "hold", "run_of_fact"]
