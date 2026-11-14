# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Merge a verified evidence bundle into a profile that already has memories.

``import_evidence_bundle`` fills an empty profile or replaces one. Merging is
the third case (issue #322): bundle rows are added next to what the target
profile already holds, and nothing already there is changed.

Rules, all fail-closed:

* The bundle is verified first; an invalid bundle writes nothing.
* A row whose key, or any other unique key of its table, already exists in the
  target profile is kept as it is. That is what makes a second merge of the
  same bundle a no-op instead of an overwrite.
* A row whose primary key belongs to another profile cannot be added without
  moving or rewriting that row, so the whole merge stops before writing.
* Any other row the database refuses stops the whole merge; the transaction
  rolls back.
* A preview runs the same inserts inside a transaction that is rolled back, so
  the counts it reports are the counts a real merge would produce.
* A real merge first writes the target profile to ``rollback_dir`` as an
  evidence bundle, the same safety net ``--replace`` has.

Derived state follows the existing import contract: keyword-index rows are
written for the facts that were added (with the same tokenizer the live index
uses), the full-text table follows through its triggers, and embeddings are
left for the daemon's background repair, which fills facts that have none.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from superlocalmemory.core.evidence_bundle import (
    _TABLES,
    _load_jsonl,
    _table_columns,
    _table_exists,
    export_evidence_bundle,
    verify_evidence_bundle,
)

#: The target keeps its own profile record (name, settings, consent).
_KEEP_TARGET_TABLES = frozenset({"profiles"})
_CHUNK = 400


class MergeConflictError(ValueError):
    """The bundle cannot be merged without changing rows that already exist."""


@dataclass(frozen=True, slots=True)
class MergeReport:
    bundle_id: str
    profile_id: str
    executed: bool
    added: dict[str, int] = field(default_factory=dict)
    kept: dict[str, int] = field(default_factory=dict)
    keyword_rows: int = 0
    rollback_dir: str | None = None

    @property
    def facts_added(self) -> int:
        return self.added.get("atomic_facts", 0)


class _Preview(Exception):
    """Raised inside the transaction to roll a preview back."""

    def __init__(self, report: MergeReport) -> None:
        super().__init__("preview")
        self.report = report


def merge_evidence_bundle(
    db: Any,
    bundle: str | Path,
    *,
    target_profile_id: str,
    execute: bool = False,
    rollback_dir: str | Path | None = None,
) -> MergeReport:
    """Add a bundle's rows to ``target_profile_id``; never overwrite a row."""
    root = Path(bundle)
    verified = verify_evidence_bundle(root)
    if not verified.valid:
        raise ValueError("invalid evidence bundle: " + "; ".join(verified.errors))
    if not _profile_exists(db, target_profile_id):
        raise ValueError(
            f"target profile does not exist: {target_profile_id}; "
            "import without --merge to create it"
        )
    if execute and rollback_dir is None:
        raise ValueError("rollback_dir is required for merge --execute")

    plan = [
        (table, key, _load_jsonl(root / filename))
        for filename, table, key in _TABLES
        if table not in _KEEP_TARGET_TABLES and _table_exists(db, table)
    ]
    collisions = _cross_profile_collisions(db, plan, target_profile_id)
    if collisions:
        shown = "; ".join(collisions[:5])
        more = f" (+{len(collisions) - 5} more)" if len(collisions) > 5 else ""
        raise MergeConflictError(
            "these ids already belong to another profile, so merging would move "
            f"or overwrite them: {shown}{more}"
        )

    if execute:
        export_evidence_bundle(db, target_profile_id, rollback_dir)

    try:
        with db.transaction():
            report = _apply(db, plan, target_profile_id, verified.bundle_id, execute)
            if not execute:
                raise _Preview(report)
    except _Preview as preview:
        return preview.report
    return MergeReport(
        bundle_id=report.bundle_id,
        profile_id=report.profile_id,
        executed=True,
        added=report.added,
        kept=report.kept,
        keyword_rows=report.keyword_rows,
        rollback_dir=str(Path(rollback_dir).resolve()) if rollback_dir else None,
    )


def _apply(
    db: Any, plan: list, profile_id: str, bundle_id: str, execute: bool,
) -> MergeReport:
    added: dict[str, int] = {}
    kept: dict[str, int] = {}
    new_facts: list[str] = []
    for table, key, rows in plan:
        columns = _table_columns(db, table)
        unique_keys = _unique_keys(db, table)
        added[table] = kept[table] = 0
        for source in rows:
            row = {name: value for name, value in source.items() if name in columns}
            if "profile_id" in columns:
                row["profile_id"] = profile_id
            if not row:
                continue
            names = sorted(row)
            quoted = ",".join(f'"{name}"' for name in names)
            marks = ",".join("?" for _ in names)
            db.execute(
                f'INSERT OR IGNORE INTO "{table}" ({quoted}) VALUES ({marks})',
                tuple(row[name] for name in names),
            )
            if int(db.execute("SELECT changes() AS n")[0]["n"]):
                added[table] += 1
                if table == "atomic_facts":
                    new_facts.append(str(row[key]))
            elif _already_present(db, table, row, unique_keys, profile_id):
                kept[table] += 1
            else:
                raise ValueError(
                    f"{table} row {row.get(key)!r} was refused by the database; "
                    "nothing was merged"
                )
    keyword_rows = _index_new_facts(db, profile_id, new_facts)
    return MergeReport(
        bundle_id=bundle_id, profile_id=profile_id, executed=execute,
        added=added, kept=kept, keyword_rows=keyword_rows,
    )


def _profile_exists(db: Any, profile_id: str) -> bool:
    if not _table_exists(db, "profiles"):
        return False
    return bool(db.execute(
        "SELECT 1 FROM profiles WHERE profile_id=?", (profile_id,),
    ))


def _cross_profile_collisions(db: Any, plan: list, profile_id: str) -> list[str]:
    found: list[str] = []
    for table, key, rows in plan:
        if "profile_id" not in _table_columns(db, table):
            continue
        ids = [str(row[key]) for row in rows if row.get(key) is not None]
        for start in range(0, len(ids), _CHUNK):
            chunk = ids[start:start + _CHUNK]
            marks = ",".join("?" for _ in chunk)
            hits = db.execute(
                f'SELECT "{key}" AS id FROM "{table}" '
                f'WHERE "{key}" IN ({marks}) AND profile_id != ?',
                (*chunk, profile_id),
            )
            found.extend(f"{table}.{hit['id']}" for hit in hits)
    return found


def _unique_keys(db: Any, table: str) -> list[tuple[str, ...]]:
    """Every unique column set of ``table``, primary key included."""
    keys: list[tuple[str, ...]] = []
    for index in db.execute(f'PRAGMA index_list("{table}")'):
        if not int(index["unique"]) or int(index["partial"]):
            # A partial index only collides inside its WHERE clause; leaving it
            # out means such a refusal is reported as an error, never as "kept".
            continue
        info = db.execute(f'PRAGMA index_info("{index["name"]}")')
        columns = tuple(str(col["name"]) for col in info if col["name"] is not None)
        if columns:
            keys.append(columns)
    for col in db.execute(f'PRAGMA table_info("{table}")'):
        if int(col["pk"]) and (str(col["name"]),) not in keys:
            keys.append((str(col["name"]),))
    return keys


def _already_present(
    db: Any, table: str, row: dict, unique_keys: list[tuple[str, ...]], profile_id: str,
) -> bool:
    scoped = "profile_id" in _table_columns(db, table)
    for columns in unique_keys:
        values = [row.get(name) for name in columns]
        if any(value is None for value in values):
            continue  # NULLs never collide in SQLite
        where = " AND ".join(f'"{name}"=?' for name in columns)
        params: tuple = tuple(values)
        if scoped:
            where += " AND profile_id=?"
            params = (*params, profile_id)
        if db.execute(f'SELECT 1 FROM "{table}" WHERE {where} LIMIT 1', params):
            return True
    return False


def _index_new_facts(db: Any, profile_id: str, fact_ids: list[str]) -> int:
    """Keyword-index rows for added facts, exactly as the live writer stores them."""
    if not fact_ids or not _table_exists(db, "bm25_tokens"):
        return 0
    from superlocalmemory.retrieval.bm25_channel import tokenize

    written = 0
    for start in range(0, len(fact_ids), _CHUNK):
        chunk = fact_ids[start:start + _CHUNK]
        marks = ",".join("?" for _ in chunk)
        rows = db.execute(
            f"SELECT fact_id, content FROM atomic_facts "
            f"WHERE profile_id=? AND fact_id IN ({marks})",
            (profile_id, *chunk),
        )
        for row in rows:
            tokens = tokenize(str(row["content"] or ""))
            if not tokens:
                continue  # no words: owes no keyword entry (see Bm25Owner)
            db.execute(
                "INSERT OR REPLACE INTO bm25_tokens (fact_id, profile_id, tokens) "
                "VALUES (?,?,?)",
                (row["fact_id"], profile_id, json.dumps(tokens)),
            )
            written += 1
    return written


__all__ = ["MergeConflictError", "MergeReport", "merge_evidence_bundle"]
