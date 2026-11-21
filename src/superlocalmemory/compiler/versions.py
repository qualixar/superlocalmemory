# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Versions first: a derived artifact keeps its previous row before it is overwritten (WS-6).

Every writer of a derived artifact (summaries, persona, entity profiles, scenes,
core memory blocks) calls :func:`keep_previous` inside the transaction that
changes the row, so the previous row is copied before it is lost. The copy is a
full row (``row_json``) plus the ids of the facts its text was built from
(``artifact_version_facts``), so erasing a fact removes every version that quotes
it (I-11).

* An unchanged row is not copied again: a row whose content (volatile timestamp
  columns aside) equals the newest stored version adds nothing, so a nightly
  rebuild that changes nothing cannot push real history out.
* Retention is bounded: the last ``KEEP_LAST`` versions per artifact key (routing
  only: it decides how much history is kept, never whether a memory changes;
  five is enough to step back through several rebuilds and small enough that a
  hub entity rewritten daily costs a few kilobytes). Scenes keep three because
  they are rewritten whenever a member joins.
* A row's BLOB columns (a scene's embedding) are not copied: they are derived
  from the text and rebuilt.
* A store without the table (older schema) copies nothing and writes as before.

Every function takes ``conn``: a ``sqlite3`` connection or the storage manager
(``execute(sql, params)`` returning rows); use :func:`atomic` so the copy and the
overwrite commit or fail together.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

logger = logging.getLogger(__name__)

#: Routing only: how many versions of one artifact key are kept. Never a threshold
#: that changes memory.
KEEP_LAST = 5
SCENE_KEEP_LAST = 3
OVERWRITE, RESTORE, EDIT = "overwrite", "restore", "edit"

_IDENT = re.compile(r"^[a-z_][a-z0-9_]*$")


def _ids(raw: Any) -> list[str]:
    """The ids in a stored JSON list. Bytes that hold UTF-8 text read as that text;
    bytes that do not read as empty (the caller passes the rebuilt ids, ``fact_links``)."""
    try:
        if isinstance(raw, (bytes, bytearray, memoryview)):
            raw = bytes(raw).decode("utf-8")
        value = json.loads(raw) if isinstance(raw, str) else raw
    except (TypeError, ValueError):
        return []
    return [str(v) for v in value] if isinstance(value, list) else []


def _select(conn: Any, sql: str, params: Sequence[Any] = ()) -> list[dict]:
    result = conn.execute(sql, tuple(params))
    if hasattr(result, "fetchall"):  # a sqlite3 cursor
        names = [d[0] for d in result.description or ()]
        return [dict(zip(names, r)) for r in result.fetchall()]
    return [dict(r) for r in result]  # the storage manager's rows


#: Public name for callers that read rows from a connection or the storage manager alike.
select_rows = _select


def _no_table(exc: Exception) -> bool:
    return "no such table" in str(exc).lower()


def _persona_facts(conn: Any, row: Mapping[str, Any]) -> list[str]:
    """A persona stitches community summaries: every fact they cite, now or in a kept version.

    Over-links on purpose: erasing any fact a past community quoted drops the
    persona version, which the persona's next rebuild replaces.
    """
    profile = row["profile_id"]
    found: set[str] = set()
    for r in _select(conn, "SELECT fact_ids_json FROM community_summaries WHERE profile_id = ?",
                     (profile,)):
        found.update(_ids(r["fact_ids_json"]))
    for r in _select(
            conn, "SELECT f.fact_id AS fact_id FROM artifact_version_facts f JOIN "
            "artifact_versions v ON v.version_id = f.version_id WHERE v.profile_id = ? "
            "AND v.artifact = 'community_summary'", (profile,)):
        found.add(str(r["fact_id"]))
    return sorted(found)


def _page_facts(conn: Any, row: Mapping[str, Any]) -> list[str]:
    """The facts a compiled page cites (every sentence's ``facts``), from its body."""
    try:
        sentences = json.loads(row.get("body_json") or "[]")
    except (TypeError, ValueError):
        return []
    found: set[str] = set()
    for sentence in sentences if isinstance(sentences, list) else []:
        found.update(str(f) for f in (sentence.get("facts") or []) if isinstance(sentence, dict))
    return sorted(found)


@dataclass(frozen=True)
class ArtifactSpec:
    """One kind of derived artifact: where it lives, what names it, what may be edited."""

    name: str
    table: str
    key_columns: tuple[str, ...]
    #: Columns that change on every rewrite without the content changing.
    volatile: tuple[str, ...]
    #: Text columns a ProposalSet may edit (``edit_derived``).
    editable: tuple[str, ...]
    facts: Callable[[Any, Mapping[str, Any]], list[str]]
    keep_last: int = KEEP_LAST


ARTIFACTS: dict[str, ArtifactSpec] = {s.name: s for s in (
    ArtifactSpec("entity_profile", "entity_profiles", ("profile_id", "profile_entry_id"),
                 ("last_updated", "last_compiled_at"), ("knowledge_summary", "compiled_truth"),
                 lambda c, r: _ids(r.get("fact_ids_json"))),
    ArtifactSpec("community_summary", "community_summaries", ("profile_id", "community_id"),
                 ("computed_at",), ("summary", "keywords"),
                 lambda c, r: _ids(r.get("fact_ids_json"))),
    ArtifactSpec("persona", "persona_summary", ("profile_id",), ("computed_at",),
                 ("summary", "keywords"), _persona_facts),
    ArtifactSpec("consolidated_summary", "consolidated_summaries", ("profile_id", "summary_id"),
                 ("created_at",), ("content",), lambda c, r: _ids(r.get("source_fact_ids"))),
    ArtifactSpec("scene", "memory_scenes", ("profile_id", "scene_id"), ("last_updated",),
                 ("theme",), lambda c, r: _ids(r.get("fact_ids_json")),
                 keep_last=SCENE_KEEP_LAST),
    ArtifactSpec("core_block", "core_memory_blocks", ("profile_id", "block_type"),
                 ("updated_at", "created_at", "block_id", "version"), ("content",),
                 lambda c, r: _ids(r.get("source_fact_ids"))),
    # A compiled page (D-18) is rebuilt as a whole by ``compiler/pages``; nothing edits its
    # text in place (``editable`` is empty): a person's edit is a proposal on a fact (D-17).
    ArtifactSpec("compiled_page", "compiled_pages", ("profile_id", "page_id"),
                 ("built_at", "set_id", "version"), (), _page_facts),
)}


def spec_for(artifact: str) -> ArtifactSpec:
    try:
        return ARTIFACTS[artifact]
    except KeyError:
        raise ValueError(f"unknown derived artifact {artifact!r}") from None


@contextmanager
def atomic(db: Any) -> Iterator[Any]:
    """Run the copy and the overwrite as one transaction.

    A raw connection, or a manager already inside ``transaction()``, is yielded as
    it is (the caller's transaction covers both); otherwise a transaction opens.
    """
    opener = getattr(db, "transaction", None)
    inside = getattr(db, "in_transaction", None)
    if opener is None or (callable(inside) and inside()):
        yield db
        return
    with opener():
        yield db


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _columns(conn: Any, table: str) -> list[str]:
    return [str(r["name"]) for r in _select(conn, "SELECT name FROM pragma_table_info(?)", (table,))]


BLOB_KEY = "__blob_hex__"


def _record(row: Mapping[str, Any]) -> dict[str, Any]:
    """The row as stored. A BLOB is kept as ``{"__blob_hex__": "<hex>"}``.

    A JSON text column can hold bytes that are not text (a value a writer bound
    as bytes). The version row is where those bytes stay recoverable after the
    live row is rebuilt; ``write_row`` turns the hex back into the same bytes.
    """
    return {k: ({BLOB_KEY: bytes(v).hex()} if isinstance(v, (bytes, bytearray, memoryview)) else v)
            for k, v in row.items()}


def _unrecord(value: Any) -> Any:
    """The inverse of :func:`_record` for one value."""
    if isinstance(value, dict) and set(value) == {BLOB_KEY} and isinstance(value[BLOB_KEY], str):
        return bytes.fromhex(value[BLOB_KEY])
    return value


def content_hash(spec: ArtifactSpec, row: Mapping[str, Any]) -> str:
    kept = {k: v for k, v in _record(row).items() if k not in spec.volatile}
    return hashlib.sha256(json.dumps(kept, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def artifact_key(spec: ArtifactSpec, row: Mapping[str, Any]) -> str:
    return json.dumps([row.get(c) for c in spec.key_columns], default=str, separators=(",", ":"))


def _where(spec: ArtifactSpec, key: Mapping[str, Any] | None, where: str | None,
           params: Sequence[Any]) -> tuple[str, list[Any]]:
    clauses: list[str] = []
    args: list[Any] = []
    for col, value in (key or {}).items():
        if not _IDENT.match(col):
            raise ValueError(f"bad column name {col!r}")
        clauses.append(f"{col} = ?")
        args.append(value)
    if where:
        clauses.append(f"({where})")
        args.extend(params)
    if not clauses:
        raise ValueError("keep_previous needs a key or a where clause: it never copies a whole table")
    return " AND ".join(clauses), args


def keep_previous(conn: Any, artifact: str, key: Mapping[str, Any] | None = None, *,
                  where: str | None = None, params: Sequence[Any] = (),
                  reason: str = OVERWRITE, proposal_id: str | None = None,
                  fact_links: Sequence[str] = ()) -> int:
    """Copy the current row(s) of ``artifact`` into ``artifact_versions``; return versions written.

    Call it inside the transaction that overwrites or deletes the row, before the
    write. ``key`` maps column names to values; ``where`` is an extra SQL
    condition from the calling module's own code (never user input), with
    ``params``. Rows that do not exist yet cost one SELECT and write nothing.
    ``fact_links`` are fact ids the kept version is also linked to: a writer that
    is about to replace a damaged fact list (bytes, not text) passes the ids it
    rebuilt it from, so erasing one of those facts still removes the version
    (which holds the old text, and the damaged bytes).
    """
    spec = spec_for(artifact)
    clause, args = _where(spec, key, where, params)
    try:
        rows = _select(conn, f"SELECT * FROM {spec.table} WHERE {clause}", args)  # noqa: S608
        written = 0
        for row in rows:
            written += _keep_one(conn, spec, row, reason, proposal_id, fact_links)
        return written
    except Exception as exc:  # noqa: BLE001 - older schema without the tables
        if _no_table(exc):
            return 0
        raise


def _keep_one(conn: Any, spec: ArtifactSpec, row: dict, reason: str,
              proposal_id: str | None, fact_links: Sequence[str] = ()) -> int:
    profile, akey = str(row.get("profile_id") or ""), artifact_key(spec, row)
    digest = content_hash(spec, row)
    latest = _select(conn, "SELECT seq, content_hash FROM artifact_versions WHERE profile_id = ? "
                     "AND artifact = ? AND artifact_key = ? ORDER BY seq DESC LIMIT 1",
                     (profile, spec.name, akey))
    if latest and latest[0]["content_hash"] == digest:
        return 0
    seq = (latest[0]["seq"] if latest else 0) + 1
    version_id = uuid.uuid4().hex
    conn.execute(
        "INSERT INTO artifact_versions (version_id, profile_id, artifact, artifact_key, seq, "
        "content_hash, row_json, reason, proposal_id, created_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
        (version_id, profile, spec.name, akey, seq, digest,
         json.dumps(_record(row), sort_keys=True, default=str), reason, proposal_id, _now()))
    for fact_id in sorted({*spec.facts(conn, row), *map(str, fact_links)}):
        conn.execute("INSERT OR IGNORE INTO artifact_version_facts (version_id, fact_id) "
                     "VALUES (?, ?)", (version_id, fact_id))
    conn.execute("DELETE FROM artifact_versions WHERE profile_id = ? AND artifact = ? "
                 "AND artifact_key = ? AND seq <= ?",
                 (profile, spec.name, akey, seq - spec.keep_last))
    return 1


#: A version that quotes a fact whose erasure has begun is not shown (the row itself goes
#: when the erasure completes: I-11).
_NOT_ERASING = (" AND NOT EXISTS (SELECT 1 FROM artifact_version_facts f JOIN "
                "projection_tombstones t ON t.fact_id = f.fact_id AND t.profile_id = v.profile_id "
                "WHERE f.version_id = v.version_id)")


def list_versions(conn: Any, profile_id: str, artifact: str,
                  key: Sequence[Any] | None = None, limit: int = 50) -> list[dict]:
    """Stored versions, newest first, without the row text (ids, hashes, times)."""
    spec = spec_for(artifact)
    sql = ("SELECT version_id, artifact, artifact_key, seq, content_hash, reason, proposal_id, "
           "created_at FROM artifact_versions v WHERE profile_id = ? AND artifact = ?")
    args: list[Any] = [profile_id, spec.name]
    if key is not None:
        sql += " AND artifact_key = ?"
        args.append(json.dumps(list(key), default=str, separators=(",", ":")))
    tail = " ORDER BY created_at DESC, seq DESC LIMIT ?"
    bound = (*args, max(1, min(int(limit), 500)))
    try:
        return _select(conn, sql + _NOT_ERASING + tail, bound)
    except Exception as exc:  # noqa: BLE001
        if "projection_tombstones" in str(exc):  # a store from before erasure tombstones
            return _select(conn, sql + tail, bound)
        if _no_table(exc):
            return []
        raise


def _not_erasing_or_empty(conn: Any) -> str:
    present = _select(conn, "SELECT 1 AS p FROM sqlite_master WHERE type = 'table' "
                      "AND name = 'projection_tombstones'")
    return _NOT_ERASING if present else ""


def get_version(conn: Any, profile_id: str, version_id: str) -> dict | None:
    rows = _select(conn, "SELECT * FROM artifact_versions v WHERE version_id = ? AND profile_id = ?"
                   + _not_erasing_or_empty(conn), (version_id, profile_id))
    if not rows:
        return None
    return {**rows[0], "row": json.loads(rows[0]["row_json"])}


def write_row(conn: Any, spec: ArtifactSpec, row: Mapping[str, Any]) -> None:
    """Write a recorded row back: update it by its key, or insert it when it is gone."""
    columns = set(_columns(conn, spec.table))
    data = {k: _unrecord(v) for k, v in row.items() if k in columns}
    keys = {c: data[c] for c in spec.key_columns if c in data}
    if len(keys) != len(spec.key_columns):
        raise ValueError("the recorded row lacks its key columns")
    where = " AND ".join(f"{c} = ?" for c in keys)
    exists = _select(conn, f"SELECT 1 AS present FROM {spec.table} WHERE {where}",  # noqa: S608
                     tuple(keys.values()))
    if exists:
        sets = [c for c in data if c not in keys]
        if sets:
            conn.execute(f"UPDATE {spec.table} SET " + ", ".join(f"{c} = ?" for c in sets)  # noqa: S608
                         + f" WHERE {where}", (*[data[c] for c in sets], *keys.values()))
        return
    names = list(data)
    conn.execute(f"INSERT INTO {spec.table} ({', '.join(names)}) "  # noqa: S608
                 f"VALUES ({', '.join('?' for _ in names)})", tuple(data[c] for c in names))


def restore_version(conn: Any, profile_id: str, version_id: str, *,
                    proposal_id: str | None = None) -> dict:
    """Put a stored version back as the live row, keeping the row it replaces as a version.

    Refused when any fact the version quotes is gone: such versions are erased
    with the fact (I-11), so this guards a race, not a normal path.
    """
    found = get_version(conn, profile_id, version_id)
    if found is None:
        raise LookupError("no such version in this profile")
    spec = spec_for(found["artifact"])
    row = found["row"]
    keep_previous(conn, spec.name, {c: row[c] for c in spec.key_columns}, reason=RESTORE,
                  proposal_id=proposal_id)
    write_row(conn, spec, row)
    return {"artifact": spec.name, "version_id": version_id, "seq": found["seq"]}


def erase_for_fact(conn: Any, profile_id: str, fact_id: str) -> int:
    """Remove every version that quotes the fact (called at tombstone time; the
    ``atomic_facts`` delete trigger covers every other path). Returns versions removed."""
    try:
        gone = _select(conn, "DELETE FROM artifact_versions WHERE profile_id = ? AND version_id IN "
                       "(SELECT version_id FROM artifact_version_facts WHERE fact_id = ?) "
                       "RETURNING version_id", (profile_id, fact_id))
        return len(gone)
    except Exception as exc:  # noqa: BLE001
        if _no_table(exc):
            return 0
        raise


__all__ = ["ARTIFACTS", "ArtifactSpec", "EDIT", "KEEP_LAST", "OVERWRITE", "RESTORE",
           "SCENE_KEEP_LAST", "artifact_key", "atomic", "content_hash", "erase_for_fact",
           "get_version", "keep_previous", "list_versions", "restore_version", "select_rows",
           "spec_for", "write_row"]
