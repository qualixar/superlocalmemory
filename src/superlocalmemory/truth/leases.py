# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Leases: a memory knows when the file it came from changed (D-38, M070).

A memory saved with ``source_file`` (``remember(source_file=…)`` on MCP, CLI
and HTTP) is fingerprinted at the door, before the file can change: sha256,
size and modification time, kept in reserved metadata ``_slm_lease`` (only the
route sets it, after the caller's reserved keys are stripped). The background
pass (:func:`materialize`) gives each of the memory's facts a lease row.

* Recall calls :func:`check_for_facts` on a read-only connection: one ``stat``
  per leased fact, a hash only when the size is equal but the time moved (a
  touch is not a change), never a write. A changed or missing file shows on
  the result at once as ``source: changed / missing``.
* The background :func:`sweep` writes lease states in bounded batches, oldest
  check first, so the dashboard's "Changed files" list is current.
* A stale lease never retires or hides a memory: it tells the reader to look
  at the file again. :func:`mark_still_true` re-fingerprints after a person or
  agent checked.

File content is never stored or returned: only its hash, size and time.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import stat as stat_mod
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

METADATA_KEY = "_slm_lease"
SOURCE_KIND = "file"
FRESH, CHANGED, MISSING = "fresh", "changed", "missing"
#: Above this size a file is fingerprinted by size and time only.
MAX_HASH_BYTES = 64 * 1024 * 1024
#: Recall hashes at most this much per result before saying "changed".
RECALL_HASH_BYTES = 8 * 1024 * 1024
#: Leased facts one recall checks; the rest wait for the sweep.
RECALL_MAX_CHECKS = 32
MAX_PATH_CHARS = 4096
NEXT_ON_STALE = "reread_source"


class SourceFileError(ValueError):
    """``source_file`` is not a readable local file; nothing was saved."""


@dataclass(frozen=True)
class Fingerprint:
    path: str
    sha256: str  # "" when the file is too large to hash
    size: int
    mtime_ns: int

    def as_meta(self) -> dict:
        return {"path": self.path, "sha256": self.sha256, "size": self.size,
                "mtime_ns": self.mtime_ns, "at": _now()}


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


def _hash(path: Path, limit: int) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
            if f.tell() > limit:
                return ""
    return h.hexdigest()


def fingerprint(source_file: str) -> Fingerprint:
    """The file's fingerprint now. ``source_file`` must be an absolute path to a
    regular file; anything else raises :class:`SourceFileError`."""
    raw = (source_file or "").strip()
    if not raw or len(raw) > MAX_PATH_CHARS or "\x00" in raw:
        raise SourceFileError("source_file must be an absolute path to a file")
    path = Path(raw).expanduser()
    if not path.is_absolute():
        raise SourceFileError("source_file must be an absolute path to a file")
    try:
        st = path.stat()
    except OSError:
        raise SourceFileError("source_file does not exist or cannot be read") from None
    if not stat_mod.S_ISREG(st.st_mode):
        raise SourceFileError("source_file must be a regular file")
    resolved = path.resolve()
    try:
        digest = _hash(resolved, MAX_HASH_BYTES) if st.st_size <= MAX_HASH_BYTES else ""
    except OSError:
        raise SourceFileError("source_file cannot be read") from None
    return Fingerprint(str(resolved), digest, int(st.st_size), int(st.st_mtime_ns))


def _state(locator: str, sha: str, size: int | None, mtime: int | None, *,
           hash_limit: int) -> str:
    try:
        st = os.stat(locator)
    except OSError:
        return MISSING
    if not stat_mod.S_ISREG(st.st_mode):
        return MISSING
    if size is not None and int(st.st_size) != int(size):
        return CHANGED
    if mtime is not None and int(st.st_mtime_ns) == int(mtime):
        return FRESH
    if not sha or st.st_size > hash_limit:
        return CHANGED  # the time moved and nothing cheaper can tell a touch from an edit
    try:
        return FRESH if _hash(Path(locator), hash_limit) == sha else CHANGED
    except OSError:
        return MISSING


def _view(state: str, locator: str, changed_at: str | None = None) -> dict:
    out = {"state": state, "file": locator}
    if state != FRESH:
        out["next"] = NEXT_ON_STALE
        if changed_at:
            out["changed_at"] = changed_at
    return out


def check_for_facts(conn: sqlite3.Connection, fact_ids: list[str]) -> dict[str, dict]:
    """``{fact_id: {"state", "file", ...}}`` for the leased facts among ``fact_ids``.

    Read-only: safe on a ``mode=ro`` connection inside recall.
    """
    ids = [f for f in dict.fromkeys(fact_ids) if f]
    if not ids:
        return {}
    marks = ",".join("?" * len(ids))
    try:
        rows = conn.execute(
            "SELECT fact_id, locator, fingerprint, size_bytes, mtime_ns, changed_at "
            f"FROM memory_leases WHERE source_kind = ? AND fact_id IN ({marks}) "
            "ORDER BY fact_id", (SOURCE_KIND, *ids)).fetchall()
    except sqlite3.OperationalError:  # a store before M070
        return {}
    out: dict[str, dict] = {}
    for fact_id, locator, sha, size, mtime, changed_at in rows[:RECALL_MAX_CHECKS]:
        state = _state(locator, sha, size, mtime, hash_limit=RECALL_HASH_BYTES)
        if out.get(fact_id, {}).get("state", FRESH) == FRESH:  # the worst lease wins
            out[fact_id] = _view(state, locator, changed_at)
    return out


def materialize(conn: sqlite3.Connection, after_rowid: int = 0,
                limit: int = 500) -> tuple[int, int]:
    """Give each fact of a memory saved with ``_slm_lease`` its lease row.

    Walks facts after ``after_rowid`` in rowid order (the caller keeps the
    watermark), so a pass never rescans the whole store. Returns
    ``(leases made, new watermark)``.
    """
    try:
        rows = conn.execute(
            "SELECT f.rowid, f.fact_id, f.profile_id, m.metadata_json FROM atomic_facts f "
            "LEFT JOIN memories m ON m.memory_id = f.memory_id "
            "WHERE f.rowid > ? ORDER BY f.rowid LIMIT ?",
            (int(after_rowid), int(limit))).fetchall()
    except sqlite3.OperationalError:
        return 0, after_rowid
    made, last = 0, after_rowid
    for rowid, fact_id, profile_id, meta_json in rows:
        last = rowid
        if not meta_json or METADATA_KEY not in meta_json:
            continue
        try:
            lease = (json.loads(meta_json) or {}).get(METADATA_KEY)
        except (TypeError, ValueError):
            continue
        if not isinstance(lease, dict) or not isinstance(lease.get("path"), str):
            continue
        at = str(lease.get("at") or _now())
        cur = conn.execute(
            "INSERT OR IGNORE INTO memory_leases (fact_id, profile_id, source_kind, locator, "
            "fingerprint, size_bytes, mtime_ns, state, leased_at, checked_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, 'fresh', ?, ?)",
            (fact_id, profile_id, SOURCE_KIND, lease["path"], str(lease.get("sha256") or ""),
             lease.get("size"), lease.get("mtime_ns"), at, at))
        made += cur.rowcount
    return made, last


def due(conn: sqlite3.Connection, limit: int = 200) -> list[tuple]:
    """The leases checked longest ago (read only)."""
    try:
        return conn.execute(
            "SELECT fact_id, locator, fingerprint, size_bytes, mtime_ns FROM memory_leases "
            "WHERE source_kind = ? ORDER BY checked_at LIMIT ?",
            (SOURCE_KIND, int(limit))).fetchall()
    except sqlite3.OperationalError:
        return []


def recheck(rows: list[tuple]) -> list[tuple[str, str, str]]:
    """``(fact_id, locator, state)`` for each due lease; touches files, never the store,
    so a caller runs it with no database lock held."""
    return [(f, loc, _state(loc, sha, size, mtime, hash_limit=MAX_HASH_BYTES))
            for f, loc, sha, size, mtime in rows]


def record(conn: sqlite3.Connection, checked: list[tuple[str, str, str]]) -> dict[str, int]:
    """Write the states :func:`recheck` found."""
    counts = {"checked": 0, CHANGED: 0, MISSING: 0, FRESH: 0}
    now = _now()
    for fact_id, locator, state in checked:
        counts["checked"] += 1
        counts[state] += 1
        conn.execute(
            "UPDATE memory_leases SET checked_at = ?, "
            "changed_at = CASE WHEN ? = 'fresh' THEN NULL WHEN state != ? THEN ? "
            "ELSE changed_at END, state = ? "
            "WHERE fact_id = ? AND source_kind = ? AND locator = ?",
            (now, state, state, now, state, fact_id, SOURCE_KIND, locator))
    return counts


def sweep(conn: sqlite3.Connection, limit: int = 200) -> dict[str, int]:
    """Re-check the leases checked longest ago and write what changed (one connection)."""
    return record(conn, recheck(due(conn, limit)))


def stale(conn: sqlite3.Connection, profile_id: str, limit: int = 100) -> list[dict]:
    """Memories whose source file changed or went missing, newest change first."""
    try:
        rows = conn.execute(
            "SELECT l.fact_id, f.memory_id, l.locator, l.state, l.changed_at "
            "FROM memory_leases l JOIN atomic_facts f ON f.fact_id = l.fact_id "
            "WHERE l.profile_id = ? AND l.state IN (?, ?) "
            "ORDER BY l.changed_at DESC LIMIT ?",
            (profile_id, CHANGED, MISSING, max(1, min(int(limit), 500)))).fetchall()
    except sqlite3.OperationalError:
        return []
    return [{"fact_id": f, "memory_id": m, **_view(s, loc, c)} for f, m, loc, s, c in rows]


def mark_still_true(conn: sqlite3.Connection, *, profile_id: str, fact_id: str) -> int:
    """A person or agent checked the file: take its fingerprint now as the lease's."""
    rows = conn.execute("SELECT locator FROM memory_leases WHERE profile_id = ? AND fact_id = ? "
                        "AND source_kind = ?", (profile_id, fact_id, SOURCE_KIND)).fetchall()
    now, done = _now(), 0
    for (locator,) in rows:
        try:
            fp = fingerprint(locator)
        except SourceFileError:
            continue  # still missing: nothing to take
        conn.execute(
            "UPDATE memory_leases SET fingerprint = ?, size_bytes = ?, mtime_ns = ?, "
            "state = 'fresh', checked_at = ?, changed_at = NULL "
            "WHERE fact_id = ? AND source_kind = ? AND locator = ?",
            (fp.sha256, fp.size, fp.mtime_ns, now, fact_id, SOURCE_KIND, locator))
        done += 1
    return done


__all__ = ["CHANGED", "FRESH", "Fingerprint", "METADATA_KEY", "MISSING", "SourceFileError",
           "check_for_facts", "due", "fingerprint", "mark_still_true", "materialize", "recheck",
           "record", "stale", "sweep"]
