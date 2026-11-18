# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Binding cues to facts (4.2.30, W3-8): the write side of ``core/cue_triggers``.

Two sources, one table (``memory_cues``, M078):

* **Explicit.** ``remember(cues=[...])`` is checked at the door, then travels in
  the memory's metadata under the reserved key ``_slm_cues`` (a caller's own
  metadata can never set it, ``core.metadata_guard``). This pass binds each cue
  to the memory's first fact (the verbatim whole-text copy written before
  enrichment), as an explicit slot is (``truth/slot_claims.py``). The save path
  is untouched; the next maintenance cycle binds, resumably, by fact rowid.
* **Extracted.** A prospective memory (a plan or to-do) whose text names a file
  path gets a ``path`` cue for it, so the plan comes back when that file is
  touched. Deterministic and conservative: only the memory's first fact, only a
  path with a folder part and a file extension, at most ``MAX_EXTRACTED`` per
  fact, and the memory kind must already be ``prospective``. A fact whose kind is
  assigned later is picked up by a later pass (the scan is by kind, not by rowid).

Every write is ``INSERT OR IGNORE`` on a cue id derived from the fact and the cue,
so a pass run twice, or a binding retried, changes nothing.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from superlocalmemory.core.cue_triggers import (
    EXPLICIT,
    EXTRACTED,
    PATH,
    CueError,
    parse_cue,
)

#: Reserved (``_slm_*``): only the checked ``cues`` field of a save can set it.
METADATA_KEY = "_slm_cues"
STATE = "cue-pass.json"
#: Facts examined per pass and extracted cues per fact (routing-only caps).
BATCH = 500
MAX_EXTRACTED = 3
#: Characters of a plan read for paths (routing-only).
_TEXT_CAP = 4000

#: A path with at least one folder part and a file extension, not part of a URL
#: and not the tail of a longer word. Absolute, ``~/`` and relative forms.
_PATH = re.compile(r"(?<![\w/.:~@-])(?:~/|/)?(?:[\w.-]+/)+[\w.-]*\w\.[A-Za-z][A-Za-z0-9]{0,7}\b")

# The memory's first fact: no fact of the same memory has a smaller rowid. Written as a
# NOT EXISTS the index on memory_id answers at its first row; the MIN() form read every fact
# of the memory, per row, and let the planner walk the whole table (WS-7: 2.5 s per cycle on
# 43,888 facts). A fact with no memory id is not a first fact, as before.
_FIRST_FACT = ("f.memory_id IS NOT NULL AND NOT EXISTS (SELECT 1 FROM atomic_facts g "
               "WHERE g.memory_id = f.memory_id AND g.rowid < f.rowid)")
_EXPLICIT = f"""
SELECT f.rowid, f.fact_id, f.profile_id, m.metadata_json
  FROM atomic_facts f JOIN memories m ON m.memory_id = f.memory_id
 WHERE f.rowid > ? AND f.rowid <= ? AND instr(m.metadata_json, ?) > 0 AND {_FIRST_FACT}
 ORDER BY f.rowid LIMIT ?
"""
# The kind is read through its own index (``IN`` over the covering ``idx_facts_memory_kind``):
# as a plain term the planner walked every fact newest first to find the few plans.
_PROSPECTIVE = f"""
SELECT f.fact_id, f.profile_id, f.content
  FROM atomic_facts f
 WHERE f.rowid IN (SELECT rowid FROM atomic_facts WHERE memory_kind = 'prospective')
   AND f.quarantined = 0 AND f.lifecycle != 'archived'
   AND f.merged_into IS NULL AND {_FIRST_FACT}
   AND NOT EXISTS (SELECT 1 FROM memory_cues c WHERE c.fact_id = f.fact_id AND c.origin = ?)
 ORDER BY f.rowid DESC LIMIT ?
"""
# Written only while the fact still exists: the scan runs outside the writer lock, so a fact
# can be erased between it and this write, and a cue for an erased fact would outlive it (I-11).
_INSERT = ("INSERT OR IGNORE INTO memory_cues (cue_id, profile_id, fact_id, kind, pattern, "
           "created_at, origin) SELECT ?, ?, ?, ?, ?, ?, ? "
           "WHERE EXISTS (SELECT 1 FROM atomic_facts WHERE fact_id = ?)")


def cue_id(profile_id: str, fact_id: str, spec: str) -> str:
    return "cue_" + hashlib.sha256(f"{profile_id}\0{fact_id}\0{spec}".encode()).hexdigest()[:20]


def extract_paths(text: str) -> list[str]:
    """File paths a plan's text names, in order, without repeats (at most ``MAX_EXTRACTED``)."""
    found: list[str] = []
    for match in _PATH.finditer((text or "")[:_TEXT_CAP]):
        path = match.group(0)
        if path not in found and len(path) <= 200 and ".." not in path.split("/"):
            found.append(path)
        if len(found) >= MAX_EXTRACTED:
            break
    return found


def _bind(conn: sqlite3.Connection, profile_id: str, fact_id: str, specs: list[str],
          origin: str) -> int:
    now, made = datetime.now(UTC).isoformat(), 0
    for spec in specs:
        try:
            cue = parse_cue(spec)
        except CueError:
            continue  # a stored cue that no door would accept is skipped, never bound
        made += conn.execute(_INSERT, (cue_id(profile_id, fact_id, cue.spec), profile_id, fact_id,
                                       cue.kind, cue.pattern, now, origin, fact_id)).rowcount
    return made


def _stored(metadata_json: object) -> list[str]:
    try:
        value = (json.loads(metadata_json or "{}") or {}).get(METADATA_KEY)
    except (TypeError, ValueError, AttributeError):
        return []
    return [s for s in value if isinstance(s, str)] if isinstance(value, list) else []


def _explicit_rows(conn: sqlite3.Connection, after_rowid: int, limit: int) -> tuple[list, int]:
    """The scan: facts after the watermark whose memory carries cues. Returns ``(rows, top)``."""
    top = conn.execute("SELECT COALESCE(MAX(rowid), 0) FROM atomic_facts").fetchone()[0]
    rows = conn.execute(_EXPLICIT, (int(after_rowid), top, METADATA_KEY, int(limit))).fetchall()
    return rows, top


def _write_explicit(conn: sqlite3.Connection, rows: list) -> int:
    return sum(_bind(conn, profile_id, fact_id, _stored(meta), EXPLICIT)
               for _, fact_id, profile_id, meta in rows)


def bind_explicit(conn: sqlite3.Connection, after_rowid: int = 0, limit: int = BATCH) -> tuple[int, int]:
    """Cues given on a save, bound to the memory's first fact. Returns ``(bound, new watermark)``.

    Reads facts after ``after_rowid`` up to the newest at the start of the call;
    the caller keeps the watermark, so a pass never rescans the whole store.
    """
    try:
        rows, top = _explicit_rows(conn, after_rowid, limit)
    except sqlite3.OperationalError:  # a store before M078, or a fresh one
        return 0, after_rowid
    return _write_explicit(conn, rows), (top if len(rows) < limit else rows[-1][0])


def _extracted_rows(conn: sqlite3.Connection, limit: int) -> list:
    """The scan: prospective memories whose first fact has no extracted cue yet."""
    return conn.execute(_PROSPECTIVE, (EXTRACTED, int(limit))).fetchall()


def _write_extracted(conn: sqlite3.Connection, rows: list) -> int:
    return sum(_bind(conn, profile_id, fact_id, [f"{PATH}:{p}" for p in extract_paths(content)],
                     EXTRACTED) for fact_id, profile_id, content in rows)


def bind_extracted(conn: sqlite3.Connection, limit: int = BATCH) -> int:
    """A ``path`` cue for each file a prospective memory's text names."""
    try:
        rows = _extracted_rows(conn, limit)
    except sqlite3.OperationalError:
        return 0
    return _write_extracted(conn, rows)


def run(db, state_dir: Path) -> int:
    """One pass: explicit cues from the watermark on, then extracted ones. Returns cues bound.

    The watermark moves only after the bindings are committed, so a stop in
    between repeats work (harmless: every insert is idempotent) and never skips it.
    """
    state = Path(state_dir) / STATE
    try:
        after = int(json.loads(state.read_text(encoding="utf-8"))["after_rowid"])
    except (OSError, ValueError, KeyError, TypeError):
        after = 0
    # The scans run on a read-only connection with no writer lock held; the lock is taken only
    # for the inserts (WS-7: the scan used to run inside it, 2.5 s a cycle on a real store).
    from superlocalmemory.truth.invariants import open_read_only

    reader = open_read_only(str(db.db_path))
    try:
        try:
            explicit, top = _explicit_rows(reader, after, BATCH)
            extracted = _extracted_rows(reader, BATCH)
        except sqlite3.OperationalError:  # a store before M078, or a fresh one
            return 0
    finally:
        reader.close()
    last = top if len(explicit) < BATCH else explicit[-1][0]
    with db.raw_connection() as conn:
        made = _write_explicit(conn, explicit) + _write_extracted(conn, extracted)
    if last != after:
        state.parent.mkdir(parents=True, exist_ok=True)
        tmp = state.with_suffix(".json.tmp")
        tmp.write_text(json.dumps({"after_rowid": last}), encoding="utf-8")
        os.replace(tmp, state)
    return made


__all__ = ["METADATA_KEY", "bind_explicit", "bind_extracted", "cue_id", "extract_paths", "run"]
