# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""What changed since a safety copy was taken, and carrying it across a restore.

Restoring a copy taken before an update must not cost the user anything they
did after it. Before the live store is replaced, its difference from the copy
is written out as plain JSON lines (``restore-delta/<ts>/``):

  memories.jsonl    memories added since the copy   -> added back after the restore
  deleted.jsonl     facts / memories forgotten or   -> deleted again in the copy
                    erased since (a recorded intent;
                    a row merely missing comes back)
                    + correction cases a person's    -> closed again in the copy
                    action or ``slm corrections      (type "overtaken": the copy
                    dismiss`` moved to the overtaken  holds them as open)
                    table since, and cases a person
                    put back since (type
                    "overtaken_restored")            -> opened again in the copy
  kind_edits.jsonl  kinds a person confirmed since  -> applied again after the restore
                    (on new memories: matched by    (the newer confirmation wins)
                    the fact's words)
  erasures.jsonl    erasure receipts and tombstones -> erased again in the copy
  profiles.jsonl    profiles created since          -> created in the copy
  meta.json         the live profile list and counts

Deletions and erasures are applied to a STAGING copy of the snapshot, never to
the live store, so erased data never reappears even for a moment and the live
store is replaced in one transaction by an already-correct copy.

Every read of the live store goes through ``mode=ro`` and every read of the
copy through ``immutable=1``: computing a preview changes no byte of either.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any, Iterable

from superlocalmemory.storage._durable_json import (
    read_json, read_jsonl, write_json_atomic, write_jsonl_atomic,
)
from superlocalmemory.storage._restore_intent import (
    deleted_facts_sql, deleted_memories_sql, returning_facts_sql, returning_memories_sql,
)

logger = logging.getLogger(__name__)

_CONFIRMED = ("user", "caller")
_KIND_COLS = ("memory_kind", "memory_kind_source", "memory_kind_confidence",
              "memory_kind_recipe", "memory_kind_at")
#: Ledgers that record an erasure; never deleted by an erasure replay.
_LEDGERS = frozenset({"erasure_receipts", "projection_tombstones", "profiles"})


def open_compare(memory_db: Path, snapshot: Path) -> sqlite3.Connection:
    """Live store as ``main`` (read-only) with the copy attached as ``snap``."""
    conn = sqlite3.connect(f"{Path(memory_db).absolute().as_uri()}?mode=ro", uri=True)
    try:
        conn.row_factory = sqlite3.Row
        conn.execute("ATTACH DATABASE ? AS snap",
                     (f"{Path(snapshot).absolute().as_uri()}?mode=ro&immutable=1",))
    except BaseException:
        # A damaged live store fails here. Closed now, not when the exception
        # (and anything that keeps it, such as a log record) is released: on
        # Windows the open file would stop the restore replacing the store.
        conn.close()
        raise
    return conn


def _tables(conn: sqlite3.Connection, schema: str = "main") -> set[str]:
    return {r[0] for r in conn.execute(
        f"SELECT name FROM {schema}.sqlite_master WHERE type='table'")}


def _cols(conn: sqlite3.Connection, table: str, schema: str = "main") -> list[str]:
    try:
        return [str(r[1]) for r in conn.execute(f"PRAGMA {schema}.table_info('{table}')")]
    except sqlite3.Error:
        return []


def _both(conn: sqlite3.Connection, table: str) -> bool:
    return table in _tables(conn, "main") and table in _tables(conn, "snap")


def _count(conn: sqlite3.Connection, sql: str) -> int:
    return int(conn.execute(sql).fetchone()[0] or 0)


_ADDED_MEMORIES = ("FROM main.memories m WHERE NOT EXISTS "
                   "(SELECT 1 FROM snap.memories s WHERE s.memory_id = m.memory_id)")
#: How a deletion carried in ``deleted.jsonl`` was recorded. Only these are
#: repeated on the copy; see ``_restore_intent``.
INTENT_TOMBSTONE = "tombstone"


def _kind_edit_sql(conn: sqlite3.Connection) -> str | None:
    """Confirmed kinds on facts the copy has, where the copy holds something else.

    An unchanged confirmation is not an edit; one the copy lacks the columns for
    (a copy from before kinds existed) always is.
    """
    if "memory_kind_source" not in _cols(conn, "atomic_facts"):
        return None
    differs = ""
    if "memory_kind_source" in _cols(conn, "atomic_facts", "snap"):
        differs = (" AND (s.memory_kind IS NOT f.memory_kind OR s.memory_kind_source IS NOT "
                   "f.memory_kind_source OR s.memory_kind_at IS NOT f.memory_kind_at)")
    return ("FROM main.atomic_facts f WHERE f.memory_kind_source IN ('user','caller') "
            "AND f.memory_kind IS NOT NULL AND EXISTS "
            f"(SELECT 1 FROM snap.atomic_facts s WHERE s.fact_id = f.fact_id{differs})")


def _added_kind_sql(conn: sqlite3.Connection) -> str | None:
    """Confirmed kinds on facts of memories added after the copy (L1-09).

    Those memories are re-ingested after the restore and get new fact ids, so
    each confirmation is carried with its fact's words and matched to the
    re-ingested fact with the same words (``_restore_reimport``).
    """
    if "memory_kind_source" not in _cols(conn, "atomic_facts"):
        return None
    return ("FROM main.atomic_facts f WHERE f.memory_kind_source IN ('user','caller') "
            "AND f.memory_kind IS NOT NULL AND f.memory_id IN "
            f"(SELECT m.memory_id {_ADDED_MEMORIES})")


def _new_rows_sql(conn: sqlite3.Connection, table: str, keys: Iterable[str]) -> str | None:
    """Rows of ``table`` in the live store that the copy does not have."""
    if table not in _tables(conn, "main"):
        return None
    if table not in _tables(conn, "snap"):
        return f"FROM main.{table} l"
    match = " AND ".join(f"s.{k} = l.{k}" for k in keys)
    return f"FROM main.{table} l WHERE NOT EXISTS (SELECT 1 FROM snap.{table} s WHERE {match})"


def delta_counts(conn: sqlite3.Connection) -> dict[str, int]:
    """Counts for the preview. Reads only."""
    kinds = _kind_edit_sql(conn)
    added_kinds = _added_kind_sql(conn)
    corrections = _new_rows_sql(conn, "correction_cases", ("case_id",))
    receipts = _new_rows_sql(conn, "erasure_receipts", ("erasure_id",))
    tombstones = _new_rows_sql(conn, "projection_tombstones", ("profile_id", "fact_id"))
    return {
        "facts_now": _count(conn, "SELECT COUNT(*) FROM main.atomic_facts"),
        "facts_in_snapshot": _count(conn, "SELECT COUNT(*) FROM snap.atomic_facts"),
        "memories_now": _count(conn, "SELECT COUNT(*) FROM main.memories"),
        "memories_in_snapshot": _count(conn, "SELECT COUNT(*) FROM snap.memories"),
        "added_memories": _count(conn, f"SELECT COUNT(*) {_ADDED_MEMORIES}"),
        "deleted_facts": _count(conn, f"SELECT COUNT(*) {deleted_facts_sql(conn)}"),
        "deleted_memories": _count(conn, f"SELECT COUNT(*) {deleted_memories_sql(conn)}"),
        "returning_facts": _count(conn, f"SELECT COUNT(*) {returning_facts_sql(conn)}"),
        "returning_memories": _count(conn, f"SELECT COUNT(*) {returning_memories_sql(conn)}"),
        "kind_edits": (_count(conn, f"SELECT COUNT(*) {kinds}") if kinds else 0)
                      + (_count(conn, f"SELECT COUNT(*) {added_kinds}") if added_kinds else 0),
        "corrections_lost": _count(conn, f"SELECT COUNT(*) {corrections}") if corrections else 0,
        "erasures": max(
            _count(conn, "SELECT COUNT(*) " + _and(receipts, "l.state='COMPLETE'"))
            if receipts else 0,
            _count(conn, f"SELECT COUNT(*) {tombstones}") if tombstones else 0),
    }


def _and(sql: str, condition: str) -> str:
    return f"{sql} {'AND' if ' WHERE ' in sql else 'WHERE'} {condition}"


def _kind_of_memory(conn: sqlite3.Connection, memory_id: str) -> str | None:
    """The kind every fact of this memory was confirmed as, else None."""
    if "memory_kind_source" not in _cols(conn, "atomic_facts"):
        return None
    rows = conn.execute(
        "SELECT memory_kind, memory_kind_source FROM main.atomic_facts WHERE memory_id=?",
        (memory_id,)).fetchall()
    if not rows or any(r[1] not in _CONFIRMED or not r[0] for r in rows):
        return None
    kinds = {r[0] for r in rows}
    return kinds.pop() if len(kinds) == 1 else None


def _overtaken_since(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    """Cases closed as overtaken (or dismissed) since the copy and still closed.

    The copy holds each of them as an open ``proposed`` case, so a restore would
    bring it back. ``overtake_id`` is local to a store and is not carried.
    """
    if "correction_cases_overtaken" not in _tables(conn, "main"):
        return []
    sql = "SELECT l.* FROM main.correction_cases_overtaken l WHERE l.restored_at IS NULL"
    if "correction_cases_overtaken" in _tables(conn, "snap"):
        sql += (" AND NOT EXISTS (SELECT 1 FROM snap.correction_cases_overtaken s "
                "WHERE s.case_id = l.case_id AND s.overtaken_at = l.overtaken_at)")
    return [{"type": "overtaken", **{k: v for k, v in dict(r).items() if k != "overtake_id"}}
            for r in conn.execute(sql + " ORDER BY l.overtake_id")]


def _restored_since(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    """Cases the copy holds as dismissed that a person put back since.

    A restore carries what the person did after the copy, so putting a case back
    is carried too. A case dismissed AND put back after the copy needs nothing:
    the copy holds it open, which is where the person left it.
    """
    if "correction_cases_overtaken" not in _tables(conn, "main") \
            or "correction_cases_overtaken" not in _tables(conn, "snap"):
        return []
    sql = ("SELECT l.* FROM main.correction_cases_overtaken l WHERE l.restored_at IS NOT NULL "
           "AND EXISTS (SELECT 1 FROM snap.correction_cases_overtaken s WHERE s.case_id = "
           "l.case_id AND s.overtaken_at = l.overtaken_at AND s.restored_at IS NULL) "
           "ORDER BY l.overtake_id")
    return [{"type": "overtaken_restored", **{k: v for k, v in dict(r).items()
                                                if k != "overtake_id"}}
            for r in conn.execute(sql)]


def export_delta(conn: sqlite3.Connection, delta_dir: Path) -> dict[str, int]:
    """Write the delta files (0600). Returns the counts written."""
    delta_dir = Path(delta_dir)
    delta_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    memories = []
    for row in conn.execute(f"SELECT m.* {_ADDED_MEMORIES} ORDER BY m.rowid"):
        item = dict(row)
        item["memory_kind"] = _kind_of_memory(conn, item["memory_id"])
        memories.append(item)
    deleted = [{"type": "fact", "intent": INTENT_TOMBSTONE, **dict(r)} for r in conn.execute(
        f"SELECT s.fact_id, s.profile_id, s.memory_id {deleted_facts_sql(conn)}")]
    deleted += [{"type": "memory", "intent": INTENT_TOMBSTONE, **dict(r)} for r in conn.execute(
        f"SELECT s.memory_id, s.profile_id {deleted_memories_sql(conn)}")]
    deleted += _restored_since(conn) + _overtaken_since(conn)
    kinds_sql = _kind_edit_sql(conn)
    kind_cols = ", ".join("f." + c for c in _KIND_COLS)
    kind_edits = [dict(r) for r in conn.execute(
        f"SELECT f.fact_id, f.profile_id, f.fact_type, {kind_cols} {kinds_sql}")
    ] if kinds_sql else []
    added_sql = _added_kind_sql(conn)
    if added_sql:
        kind_edits += [{**dict(r), "fact_id": None} for r in conn.execute(
            f"SELECT f.memory_id AS added_memory_id, f.profile_id, f.fact_type, f.content, "
            f"(SELECT COUNT(*) FROM main.atomic_facts o WHERE o.memory_id = f.memory_id) "
            f"AS facts_in_memory, {kind_cols} {added_sql}")]
    erasures: list[dict[str, Any]] = []
    receipts = _new_rows_sql(conn, "erasure_receipts", ("erasure_id",))
    if receipts:
        erasures += [{"type": "receipt", **dict(r)} for r in conn.execute(
            f"SELECT l.* {receipts}")]
    tombstones = _new_rows_sql(conn, "projection_tombstones", ("profile_id", "fact_id"))
    if tombstones:
        erasures += [{"type": "tombstone", **dict(r)} for r in conn.execute(
            f"SELECT l.* {tombstones}")]
    new_profiles = _new_rows_sql(conn, "profiles", ("profile_id",))
    profiles = [dict(r) for r in conn.execute(f"SELECT l.* {new_profiles}")] if new_profiles else []
    write_jsonl_atomic(delta_dir / "memories.jsonl", memories)
    write_jsonl_atomic(delta_dir / "deleted.jsonl", deleted)
    write_jsonl_atomic(delta_dir / "kind_edits.jsonl", kind_edits)
    write_jsonl_atomic(delta_dir / "erasures.jsonl", erasures)
    write_jsonl_atomic(delta_dir / "profiles.jsonl", profiles)
    live_profiles = sorted(r[0] for r in conn.execute("SELECT profile_id FROM main.profiles"))
    counts = {"memories": len(memories), "deleted": len(deleted),
              "kind_edits": len(kind_edits), "erasures": len(erasures),
              "profiles": len(profiles)}
    write_json_atomic(delta_dir / "meta.json", {"live_profiles": live_profiles, "counts": counts})
    return counts


def export_empty_delta(delta_dir: Path) -> dict[str, int]:
    """The export when the live store cannot be read: nothing to carry over."""
    delta_dir = Path(delta_dir)
    delta_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    for name in ("memories", "deleted", "kind_edits", "erasures", "profiles"):
        write_jsonl_atomic(delta_dir / f"{name}.jsonl", [])
    counts = dict.fromkeys(("memories", "deleted", "kind_edits", "erasures", "profiles"), 0)
    write_json_atomic(delta_dir / "meta.json", {"live_profiles": [], "counts": counts,
                                                "live_unreadable": True})
    return counts


def delta_from_unreadable_store(delta_dir: Path) -> bool:
    return bool((read_json(Path(delta_dir) / "meta.json") or {}).get("live_unreadable"))


# ---------------------------------------------------------------------------
# Applying the delta to a staging copy (never the live store)
# ---------------------------------------------------------------------------

def _load_vec(conn: sqlite3.Connection) -> bool:
    try:
        from superlocalmemory.storage.sqlite_vectors import load_sqlite_vec_extension

        load_sqlite_vec_extension(conn)
        conn.execute("SELECT vec_version()").fetchone()
        return True
    except Exception:  # noqa: BLE001 - absent extension: vector rows are left
        return False


def _layout(conn: sqlite3.Connection) -> tuple[list[str], set[str]]:
    """(regular tables, virtual table names). Shadow tables are excluded."""
    rows = conn.execute("SELECT name, sql FROM sqlite_master WHERE type='table'").fetchall()
    virtual = {n for n, sql in rows if (sql or "").upper().startswith("CREATE VIRTUAL")}
    regular = [n for n, _sql in rows
               if n not in virtual and not n.startswith("sqlite_")
               and not any(n.startswith(v + "_") for v in virtual)]
    return regular, virtual


def _insert_rows(conn: sqlite3.Connection, table: str, rows: list[dict[str, Any]]) -> int:
    have = set(_cols(conn, table))
    done = 0
    for row in rows:
        cols = [c for c in row if c in have]
        if not cols:
            continue
        cur = conn.execute(
            f"INSERT OR IGNORE INTO {table} ({', '.join(cols)}) "
            f"VALUES ({', '.join('?' for _ in cols)})", [row[c] for c in cols])
        done += max(cur.rowcount, 0)
    return done


def _erase_profiles(conn: sqlite3.Connection, profiles: set[str], live: set[str],
                    vec: bool) -> int:
    """Remove every row of each erased profile (foreign keys off, one transaction)."""
    if not profiles:
        return 0
    regular, virtual = _layout(conn)
    if "fact_embeddings" in virtual and not vec:
        raise RuntimeError("cannot remove an erased profile's vectors: the vector "
                           "extension did not load")
    conn.execute("PRAGMA foreign_keys=OFF")
    conn.execute("BEGIN IMMEDIATE")
    try:
        for pid in sorted(profiles):
            if "fact_embeddings" in virtual:
                conn.execute("DELETE FROM fact_embeddings WHERE profile_id=?", (pid,))
            for table in regular:
                if table in _LEDGERS or "profile_id" not in _cols(conn, table):
                    continue
                conn.execute(f"DELETE FROM {table} WHERE profile_id=?", (pid,))
            if pid not in live:
                conn.execute("DELETE FROM profiles WHERE profile_id=?", (pid,))
        conn.execute("COMMIT")
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    return len(profiles)


def _remove_fact(conn: sqlite3.Connection, fact_id: str, *, erasure: bool,
                 fact_tables: list[str], virtual: set[str], vec: bool) -> bool:
    conn.execute("SAVEPOINT fact")
    try:
        if vec and "fact_embeddings" in virtual and "embedding_metadata" in fact_tables:
            conn.execute("DELETE FROM fact_embeddings WHERE rowid IN (SELECT vec_rowid "
                         "FROM embedding_metadata WHERE fact_id=?)", (fact_id,))
        if "fact_expansion_fts" in virtual:
            conn.execute("DELETE FROM fact_expansion_fts WHERE fact_id=?", (fact_id,))
        if erasure and "correction_cases" in fact_tables:
            # The ledger holds no text, but it RESTRICTs the fact's deletion; an
            # erasure in the live store already removed these rows with the fact.
            cases = ("SELECT case_id FROM correction_cases WHERE predecessor_fact_id=? "
                     "OR successor_fact_id=?")
            if "correction_events" in _tables(conn):
                conn.execute(f"DELETE FROM correction_events WHERE case_id IN ({cases})",
                             (fact_id, fact_id))
            conn.execute("DELETE FROM correction_cases WHERE predecessor_fact_id=? "
                         "OR successor_fact_id=?", (fact_id, fact_id))
        for table in fact_tables:
            if table not in ("correction_cases", "graph_edges"):
                conn.execute(f"DELETE FROM {table} WHERE fact_id=?", (fact_id,))
        if "graph_edges" in fact_tables:
            conn.execute("DELETE FROM graph_edges WHERE source_id=? OR target_id=?",
                         (fact_id, fact_id))
        conn.execute("DELETE FROM atomic_facts WHERE fact_id=?", (fact_id,))
        conn.execute("RELEASE fact")
        return True
    except sqlite3.IntegrityError:
        conn.execute("ROLLBACK TO fact")
        conn.execute("RELEASE fact")
        return False


def _fact_tables(conn: sqlite3.Connection) -> tuple[list[str], set[str]]:
    regular, virtual = _layout(conn)
    out = [t for t in regular
           if t not in ("atomic_facts", "projection_tombstones")
           and "fact_id" in _cols(conn, t)]
    if "correction_cases" in regular:
        out.append("correction_cases")
    if "graph_edges" in regular:
        out.append("graph_edges")
    return out, virtual


def pending_obligation_profiles(present: set[str], data_root: Path) -> set[str]:
    """Of ``present``, the profiles the GDPR obligation ledger says must not come back."""
    if not (Path(data_root) / "backup_obligations.db").exists():
        return set()
    from superlocalmemory.infra.backup_obligations import BackupObligationStore

    store = BackupObligationStore(Path(data_root))
    return {pid for pid in present if pid and store.list_pending_for_profile(pid)}


def erased_profile_ids(delta_dir: Path, data_root: Path, present: set[str]) -> set[str]:
    """Profiles whose data must not come back: completed profile erasures recorded
    in the export, and any of ``present`` the obligation ledger still holds."""
    receipts = [e for e in read_jsonl(Path(delta_dir) / "erasures.jsonl")
                if e.get("type") == "receipt"]
    erased = {r["profile_id"] for r in receipts
              if r.get("subject_type") == "profile" and r.get("state") == "COMPLETE"}
    return erased | pending_obligation_profiles(present, data_root)


def _intended(deleted: list[dict[str, Any]], kind: str) -> list[dict[str, Any]]:
    """Deletions of ``kind`` that carry a recorded intent. An entry without one
    (an export by an earlier build) is not repeated: a loss is never re-applied."""
    return [d for d in deleted if d.get("type") == kind and d.get("intent") == INTENT_TOMBSTONE]


def _holds_pair(conn: sqlite3.Connection, row: dict[str, Any]) -> bool:
    """The copy holds both facts of a case it has no row for, under its profile (and the profile)."""
    tables = _tables(conn)
    if "atomic_facts" not in tables or conn.execute(
            "SELECT 1 FROM correction_cases WHERE case_id=?", (row["case_id"],)).fetchone():
        return False   # no facts to name, or the copy holds the case in some other state
    if "profiles" in tables and not conn.execute(
            "SELECT 1 FROM profiles WHERE profile_id=?", (row["profile_id"],)).fetchone():
        return False
    found = conn.execute(
        "SELECT COUNT(DISTINCT fact_id) FROM atomic_facts WHERE profile_id=? AND fact_id IN (?, ?)",
        (row["profile_id"], row["predecessor_fact_id"], row["successor_fact_id"])).fetchone()
    return found[0] == len({row["predecessor_fact_id"], row["successor_fact_id"]})


def _carry_overtaken(conn: sqlite3.Connection, rows: list[dict[str, Any]]) -> int:
    """Close again, in the copy, the cases that were closed since it was taken.

    A case is moved only if the copy still holds it as an open ``proposed`` case
    between the same two facts. The event log goes with it, the way
    ``overtaken_cases.overtake`` moves it, so no event row is left behind.

    A dismissal of a case the copy does not hold is still kept when the copy holds
    both of its facts: the save that proposed it may come back with the same ids,
    and that row is what refuses the same proposal (``correction_dismissals``). No
    open case is made. If the facts are not in the copy, the memory is re-imported
    under new ids and the row would name nothing, so it is dropped.
    """
    if not rows or "correction_cases" not in _tables(conn):
        return 0
    from superlocalmemory.core.overtaken_cases import ensure_table
    from superlocalmemory.storage.correction_dismissals import DISMISSED_USER_ACTION

    ensure_table(conn)
    carried = 0
    for item in rows:
        row = {k: v for k, v in item.items() if k != "type"}
        key = (row["case_id"], row["predecessor_fact_id"], row["successor_fact_id"])
        is_open = conn.execute(
            "SELECT 1 FROM correction_cases WHERE case_id=? AND predecessor_fact_id=? AND "
            "successor_fact_id=? AND status='proposed'", key).fetchone()
        if not is_open and not (row.get("user_action") == DISMISSED_USER_ACTION
                                and _holds_pair(conn, row)):
            continue
        if conn.execute("SELECT 1 FROM correction_cases_overtaken WHERE case_id=? AND "
                        "overtaken_at=?", (row["case_id"], row["overtaken_at"])).fetchone():
            continue
        if is_open:
            if "correction_events" in _tables(conn):
                conn.execute("DELETE FROM correction_events WHERE case_id=?", (row["case_id"],))
            conn.execute("DELETE FROM correction_cases WHERE case_id=? AND status='proposed'",
                         (row["case_id"],))
        carried += _insert_rows(conn, "correction_cases_overtaken", [row])
    return carried


def _reopen_restored(conn: sqlite3.Connection, rows: list[dict[str, Any]]) -> int:
    """Put back, in the copy, the cases a person put back after it was taken.

    The same ``overtaken_cases.restore`` the person used, so a case whose facts
    are gone, or that was proposed again, is refused and left dismissed. The
    original restore time and actor are kept on the audit row.
    """
    if not rows or "correction_cases" not in _tables(conn):
        return 0
    from superlocalmemory.core.overtaken_cases import RestoreRefused, restore

    reopened = 0
    for row in rows:
        waiting = conn.execute("SELECT overtake_id FROM correction_cases_overtaken WHERE "
                               "case_id=? AND overtaken_at=? AND restored_at IS NULL",
                               (row["case_id"], row["overtaken_at"])).fetchone()
        if waiting is None:
            continue
        try:
            restore(conn, row["case_id"], profile_id=row["profile_id"])
        except RestoreRefused as exc:
            logger.info("restored case %s stays dismissed in the copy: %s",
                        row["case_id"][:12], exc)
            continue
        conn.execute("UPDATE correction_cases_overtaken SET restored_at=?, restored_by=? "
                     "WHERE overtake_id=?", (row["restored_at"], row["restored_by"], waiting[0]))
        reopened += 1
    return reopened


def apply_delta_to_staging(staging: Path, delta_dir: Path, *, data_root: Path) -> dict[str, int]:
    """Erasures, deletions and new profiles applied to the staging copy."""
    delta_dir = Path(delta_dir)
    erasures = read_jsonl(delta_dir / "erasures.jsonl")
    deleted = read_jsonl(delta_dir / "deleted.jsonl")
    profiles = read_jsonl(delta_dir / "profiles.jsonl")
    live = set((read_json(delta_dir / "meta.json") or {}).get("live_profiles") or [])
    receipts = [e for e in erasures if e.get("type") == "receipt"]
    tombstones = [e for e in erasures if e.get("type") == "tombstone"]
    with closing(sqlite3.connect(str(staging), isolation_level=None)) as conn:
        vec = _load_vec(conn)
        present = {r[0] for r in conn.execute("SELECT DISTINCT profile_id FROM atomic_facts")}
        erased_profiles = erased_profile_ids(delta_dir, data_root, present)
        n_profiles = _erase_profiles(conn, erased_profiles, live, vec)
        conn.execute("PRAGMA foreign_keys=ON")
        fact_tables, virtual = _fact_tables(conn)
        counts = {"profiles_erased": n_profiles, "facts_erased": 0, "facts_deleted": 0,
                  "memories_removed": 0, "kept_protected": 0, "profiles_added": 0,
                  "overtaken_carried": 0, "overtaken_reopened": 0}
        conn.execute("BEGIN IMMEDIATE")
        try:
            counts["profiles_added"] = _insert_rows(
                conn, "profiles", [{k: v for k, v in p.items() if k != "type"} for p in profiles])
            # Before any fact is removed: an open case would protect its facts.
            kept = [d for d in deleted if d.get("profile_id") not in erased_profiles]
            # Put back first: a later dismissal of the same case is then carried on top.
            counts["overtaken_reopened"] = _reopen_restored(
                conn, [d for d in kept if d.get("type") == "overtaken_restored"])
            counts["overtaken_carried"] = _carry_overtaken(
                conn, [d for d in kept if d.get("type") == "overtaken"])
            plan = [(t["fact_id"], True) for t in tombstones]
            plan += [(d["fact_id"], False) for d in _intended(deleted, "fact")]
            for fact_id, erasure in plan:
                if not conn.execute("SELECT 1 FROM atomic_facts WHERE fact_id=?",
                                    (fact_id,)).fetchone():
                    continue
                if _remove_fact(conn, fact_id, erasure=erasure, fact_tables=fact_tables,
                                virtual=virtual, vec=vec):
                    counts["facts_erased" if erasure else "facts_deleted"] += 1
                else:
                    counts["kept_protected"] += 1
            memory_ids = [d["memory_id"] for d in _intended(deleted, "memory")]
            memory_ids += [t["memory_id"] for t in tombstones if t.get("memory_id")]
            for memory_id in memory_ids:
                cur = conn.execute(
                    "DELETE FROM memories WHERE memory_id=? AND NOT EXISTS "
                    "(SELECT 1 FROM atomic_facts WHERE memory_id=?)", (memory_id, memory_id))
                counts["memories_removed"] += max(cur.rowcount, 0)
            for table, rows in (("erasure_receipts", receipts),
                                ("projection_tombstones", tombstones)):
                if rows and table in _tables(conn):
                    _insert_rows(conn, table, [{k: v for k, v in r.items() if k != "type"}
                                               for r in rows])
            conn.execute("COMMIT")
        except BaseException:
            conn.execute("ROLLBACK")
            raise
        if conn.execute("PRAGMA quick_check").fetchone()[0] != "ok":
            raise RuntimeError("the prepared copy failed its check")
    return counts


def load_memories(delta_dir: Path) -> list[dict[str, Any]]:
    return read_jsonl(Path(delta_dir) / "memories.jsonl")


def load_kind_edits(delta_dir: Path) -> list[dict[str, Any]]:
    return read_jsonl(Path(delta_dir) / "kind_edits.jsonl")


def metadata_of(row: dict[str, Any]) -> dict[str, Any]:
    raw = row.get("metadata_json") or "{}"
    try:
        meta = json.loads(raw) if isinstance(raw, str) else dict(raw)
    except (TypeError, ValueError):
        meta = {}
    return meta if isinstance(meta, dict) else {}


__all__ = ["INTENT_TOMBSTONE", "apply_delta_to_staging", "delta_counts",
           "delta_from_unreadable_store", "erased_profile_ids", "export_delta",
           "export_empty_delta", "load_kind_edits", "load_memories", "metadata_of",
           "open_compare", "pending_obligation_profiles"]
