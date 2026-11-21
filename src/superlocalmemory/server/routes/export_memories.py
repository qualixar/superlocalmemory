# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V4 | https://qualixar.com | https://varunpratap.com

"""The memory export (``GET /api/export``): the rows, written one at a time.

Kept apart from the route so the same writer serves the three formats and so a
failure can be reported before any byte is sent: the route runs this into a
temporary file and answers only when the file is whole.
"""

from __future__ import annotations

import csv
import io
import json
import logging
import sqlite3
from datetime import datetime, timezone
from typing import Any, BinaryIO, Iterator

from superlocalmemory.compliance.export_encoding import (
    ExportOptions,
    dumps,
    encode_row,
    is_vector_column,
)
from superlocalmemory.compliance.export_stream import JsonObjectWriter
from superlocalmemory.storage.erasure_visibility import erasure_clause_for_connection
from superlocalmemory.storage.json_columns import load_memory_metadata

logger = logging.getLogger("superlocalmemory.routes.data_io")

_PAGE_ROWS = 200

# What a marker may say in an export: ids only, never a path or a file name.
_SOURCE_KEYS = ("media_id", "document_id", "page", "part", "source_id")
_SOURCE_TYPES = {"media": "media", "document": "document_page", "folder": "folder"}


def has_column(cursor: Any, table: str, column: str) -> bool:
    """Whether ``table`` carries ``column`` in this database.

    Presence-guarded because ``quarantined`` arrives with a migration and this
    route must keep working on a store the engine has not opened.
    """
    try:
        cursor.execute(f"PRAGMA table_info({table})")
        return any(row["name"] == column for row in cursor.fetchall())
    except Exception:  # noqa: BLE001 -- an export must not fail over a probe
        return False


def source_marks(cursor: Any, profile_id: str) -> dict[str, dict]:
    """The source marker of every fact that came from a picture, a document page or a folder.

    Only their text is exported (never the picture, the document or the folder),
    so the record says where the text came from. Other facts have no marker.
    """
    marks: dict[str, dict] = {}
    try:
        cursor.execute(
            "SELECT memory_id, metadata_json FROM memories WHERE profile_id = ?"
            " AND metadata_json LIKE '%_slm_source%'", (profile_id,))
        for row in cursor.fetchall():
            try:
                raw = (_meta(row["metadata_json"]) or {}).get("_slm_source")
            except ValueError:
                continue
            kind = _SOURCE_TYPES.get(raw.get("type")) if isinstance(raw, dict) else None
            if kind:
                marks[row["memory_id"]] = {
                    "type": kind, **{k: raw[k] for k in _SOURCE_KEYS if k in raw}}
    except Exception:  # noqa: BLE001 -- an export must not fail over a marker
        logger.warning("export: source markers unavailable")
        return {}
    return marks


def _select(cursor: Any, table: str, where: str, params: list,
            options: ExportOptions) -> tuple[str, list]:
    """``SELECT`` naming only the columns this export reads (vectors are never fetched)."""
    cursor.execute(f"PRAGMA table_info({table})")
    names = [row["name"] for row in cursor.fetchall()]
    keep = []
    for name in names:
        if is_vector_column(name) and not options.include_embeddings:
            options.note_omitted(table, name)
        else:
            keep.append(name)
    cols = ", ".join(f'"{n}"' for n in keep)
    return f"SELECT {cols} FROM {table} WHERE {where} ORDER BY created_at", params


def _query(cursor: Any, profile: str, category: str | None, project: str | None,
           options: ExportOptions) -> tuple[str, list, bool]:
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='atomic_facts'")
    use_v3 = cursor.fetchone() is not None
    if use_v3:
        # Withheld summaries are excluded, and this is about IMPORT, not
        # tidiness. Import re-ingests every record through the normal pipeline,
        # which mints a fresh memory with quarantined = 0 -- so an export taken
        # here and imported anywhere would resurrect all model-written rows as
        # though the owner had written them. They are derived artefacts (the
        # consolidator regenerates them from the facts that ARE exported) and
        # their text is kept in consolidated_summaries.
        where, params = "profile_id = ?", [profile]
        where += erasure_clause_for_connection(cursor.connection)
        if has_column(cursor, "atomic_facts", "quarantined"):
            where += " AND COALESCE(quarantined, 0) = 0"
        if category:
            where += " AND fact_type = ?"
            params.append(category)
        if project:
            where += " AND session_id = ?"
            params.append(project)
        sql, params = _select(cursor, "atomic_facts", where, params, options)
        return sql, params, True
    where, params = "profile = ?", [profile]
    if category:
        where += " AND category = ?"
        params.append(category)
    if project:
        where += " AND project_name = ?"
        params.append(project)
    sql, params = _select(cursor, "memories", where, params, options)
    return sql, params, False


def _records(cursor: Any, sql: str, params: list, table: str, options: ExportOptions,
             marks: dict[str, dict]) -> Iterator[dict]:
    cursor.execute(sql, params)
    while True:
        page = cursor.fetchmany(_PAGE_ROWS)
        if not page:
            return
        for row in page:
            record = encode_row(dict(row), table, options)
            mark = marks.get(record.get("memory_id"))
            if mark:
                record["source"] = dict(mark)
            yield record


def write_memories_export(conn: Any, fp: BinaryIO, fmt: str, profile: str, *,
                          category: str | None = None, project_name: str | None = None,
                          include_embeddings: bool = False) -> int:
    """Write the profile's memories to ``fp`` as ``json``, ``jsonl`` or ``csv``.

    ``conn`` is a read-only connection; this sets its row and text factories.
    ``fp`` is a binary file opened ``w+b``. Returns the number of memories written.
    """
    conn.row_factory = sqlite3.Row
    # TEXT that is not valid UTF-8 can be stored; it must not stop an export.
    conn.text_factory = lambda raw: raw.decode("utf-8", "replace")
    options = ExportOptions(include_embeddings=include_embeddings)
    cursor = conn.cursor()
    sql, params, use_v3 = _query(cursor, profile, category, project_name, options)
    marks = source_marks(cursor, profile) if use_v3 else {}
    # A second cursor: the first reads memory markers and column names above, and
    # this one streams the facts.
    rows = _records(conn.cursor(), sql, params, "atomic_facts" if use_v3 else "memories",
                    options, marks)
    if fmt == "jsonl":
        count = 0
        for record in rows:
            fp.write(dumps(record).encode("utf-8") + b"\n")
            count += 1
        return count
    if fmt == "csv":
        return _write_csv(fp, rows, cursor, use_v3, options)
    writer = JsonObjectWriter(fp)
    writer.begin()
    writer.value("version", "3.0.0")
    writer.value("exported_at", datetime.now(timezone.utc).isoformat())
    writer.value("filters", {"category": category, "project_name": project_name})
    count = writer.array("memories", rows)
    writer.value("total_memories", count)
    writer.value("embeddings", "included_base64" if include_embeddings else "omitted")
    writer.value("export_manifest", options.manifest())
    writer.end()
    return count


def _write_csv(fp: BinaryIO, rows: Iterator[dict], cursor: Any, use_v3: bool,
               options: ExportOptions) -> int:
    """CSV with a header fixed by the table's columns, not by whichever row came first."""
    table = "atomic_facts" if use_v3 else "memories"
    cursor.execute(f"PRAGMA table_info({table})")
    header = [r["name"] for r in cursor.fetchall()
              if options.include_embeddings or not is_vector_column(r["name"])]
    if use_v3:
        header.append("source")
    text = io.TextIOWrapper(fp, encoding="utf-8", newline="", write_through=True)
    try:
        writer = csv.DictWriter(text, fieldnames=header, extrasaction="ignore")
        writer.writeheader()
        count = 0
        for record in rows:
            writer.writerow({k: (json.dumps(v) if isinstance(v, (dict, list)) else v)
                             for k, v in record.items()})
            count += 1
        text.flush()
        return count
    finally:
        text.detach()  # the caller owns (and closes) the binary file


__all__ = ["has_column", "source_marks", "write_memories_export"]


_meta = load_memory_metadata
