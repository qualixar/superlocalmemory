# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V4 | https://qualixar.com | https://varunpratap.com

"""The right-of-access / portability export of one profile (GDPR Art. 15 and 20).

One reader, two ways out. :meth:`ProfileExport.sections` yields the export a
section at a time, reading each table through a read-only snapshot in pages;
:meth:`ProfileExport.as_dict` collects it (the in-process API) and
:meth:`ProfileExport.write` streams it to a file without ever holding it.

What is in it: every table of ``memory.db`` that carries a ``profile_id`` (found
from the live schema, so a new table cannot be missed), the profile record, the
code graph, the picture/document/folder records of ``media.db``, what the system
learned about how the person works (``learning.db``), the hash-chained audit
records about them (``audit_chain.db``) and any save still waiting in
``pending.db``.

What is deliberately not in it, and why, is written into the export itself
(``export_manifest``): binary and vector columns unless asked for (derived
numbers, not what the person wrote), search indexes (rebuilt from the exported
text), and installation bookkeeping (schema versions, other users' credentials).
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO, Iterator

from superlocalmemory.compliance.export_encoding import (
    SEARCH_INDEX_REASON,
    SYSTEM_TABLE_REASON,
    ExportOptions,
    encode_row,
    encode_value,
    is_vector_column,
)
from superlocalmemory.compliance.export_stream import JsonObjectWriter

logger = logging.getLogger(__name__)

#: Rows fetched per page. Small enough that a page of embeddings stays modest.
PAGE_ROWS = 200

#: Friendly export keys -> canonical table names (the stable Art. 20 contract).
EXPORT_ALIASES = {
    "facts": "atomic_facts",
    "entities": "canonical_entities",
    "edges": "graph_edges",
    "feedback": "feedback_records",
    "scenes": "memory_scenes",
}

_FTS_SUFFIXES = ("_fts", "_fts_data", "_fts_idx", "_fts_content", "_fts_docsize", "_fts_config")

Section = tuple[str, str, Any]


def _quote(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _lenient_text(raw: bytes) -> str:
    """A TEXT value that is not valid UTF-8 must not stop the export."""
    return raw.decode("utf-8", "replace")


def _open_readonly(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True, timeout=5,
                           check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.text_factory = _lenient_text
    return conn


def _paged(cursor: sqlite3.Cursor) -> Iterator[sqlite3.Row]:
    while True:
        page = cursor.fetchmany(PAGE_ROWS)
        if not page:
            return
        yield from page


def _readable_columns(conn: sqlite3.Connection, table: str, options: ExportOptions) -> list[str]:
    """The table's columns that this export reads; vector columns are not read at all."""
    keep: list[str] = []
    for row in conn.execute(f"PRAGMA table_info({_quote(table)})"):
        name = row[1]
        if is_vector_column(name) and not options.include_embeddings:
            options.note_omitted(table, name)
        else:
            keep.append(name)
    return keep


def table_rows(conn: sqlite3.Connection, table: str, options: ExportOptions, *,
               where: str = "", params: tuple[Any, ...] = (), order: str = "") -> Iterator[dict]:
    """The encoded rows of ``table``, a page at a time.

    The query is prepared before the first row is returned, so a missing table
    or column raises here rather than half-way through a file.
    """
    columns = _readable_columns(conn, table, options)
    if not columns:
        raise sqlite3.OperationalError(f"no readable columns in {table}")
    sql = f"SELECT {', '.join(_quote(c) for c in columns)} FROM {_quote(table)}"
    if where:
        sql += f" WHERE {where}"
    if order:
        sql += f" ORDER BY {order}"
    cursor = conn.execute(sql, params)
    return (encode_row(dict(row), table, options) for row in _paged(cursor))


def sidecar_tables(path: Path, profile_id: str, options: ExportOptions, *,
                   only: tuple[str, ...] | None = None) -> dict[str, list[dict]] | None:
    """The rows of one profile from a side database; ``None`` when there is no file.

    Scoped by ``profile_id`` wherever the table has one and skipped where it has
    none: a table with no workspace column holds nothing attributable to one
    workspace, and returning it would hand another workspace's rows to this
    person. Every row, never a sample.
    """
    if not path.exists():
        return None
    out: dict[str, list[dict]] = {}
    try:
        conn = _open_readonly(path)
    except sqlite3.Error as exc:
        logger.warning("export: %s could not be opened: %s", path.name, type(exc).__name__)
        options.skipped_tables[path.name] = type(exc).__name__
        return None
    try:
        names = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")]
        for table in names:
            if table.endswith(_FTS_SUFFIXES) or (only and table not in only):
                continue
            try:
                info = conn.execute(f"PRAGMA table_info({_quote(table)})")
                if "profile_id" not in {r[1] for r in info}:
                    continue
                rows = list(table_rows(conn, table, options, where="profile_id = ?",
                                       params=(profile_id,)))
            except sqlite3.Error as exc:
                logger.warning("export: %s.%s skipped: %s", path.name, table, type(exc).__name__)
                options.skipped_tables[f"{path.name}:{table}"] = type(exc).__name__
                continue
            if rows:
                out[table] = rows
    finally:
        conn.close()
    return out


def code_graph(path: Path, options: ExportOptions, *,
               owned_fact_ids: list[str] | None = None) -> dict[str, list[dict]] | None:
    """``code_graph.db`` for one person (repo paths, file and symbol names identify one at work).

    The graph has no owner column. ``owned_fact_ids=None`` means the person is
    the only profile on this computer, so the whole graph is theirs. Otherwise
    only what their own memories point at is exported: the links, the symbols
    they link to and the files of those symbols. The rest of a shared graph
    describes the source code and may name another person's repositories; a
    portability export must not hand it over.
    """
    if not path.exists():
        return None
    out: dict[str, list[dict]] = {}
    try:
        conn = _open_readonly(path)
    except sqlite3.Error as exc:
        logger.warning("export: code_graph.db read failed: %s", type(exc).__name__)
        options.skipped_tables[path.name] = type(exc).__name__
        return None
    try:
        if owned_fact_ids is None:
            _whole_graph(conn, path, options, out)
        else:
            _linked_graph(conn, path, options, out, owned_fact_ids)
    finally:
        conn.close()
    return out or None


def _whole_graph(conn: sqlite3.Connection, path: Path, options: ExportOptions,
                 out: dict[str, list[dict]]) -> None:
    names = [r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")]
    for table in names:
        if table.endswith(_FTS_SUFFIXES):
            continue
        try:
            # Every row, a page at a time: a fixed LIMIT once cut a large
            # repository's graph at 10,000 rows and said nothing.
            out[table] = list(table_rows(conn, table, options))
        except sqlite3.Error as exc:
            logger.warning("export: code_graph.%s skipped: %s", table, type(exc).__name__)
            options.skipped_tables[f"{path.name}:{table}"] = type(exc).__name__


#: (table, how its rows are selected given the linked node ids / file paths)
_LINKED_SELECTORS = (
    ("code_memory_links", "slm_fact_id IN (SELECT value FROM json_each(?))", "facts"),
    ("graph_nodes", "node_id IN (SELECT code_node_id FROM code_memory_links "
                    "WHERE slm_fact_id IN (SELECT value FROM json_each(?)))", "facts"),
    ("graph_files", "file_path IN (SELECT file_path FROM graph_nodes WHERE node_id IN "
                    "(SELECT code_node_id FROM code_memory_links "
                    "WHERE slm_fact_id IN (SELECT value FROM json_each(?))))", "facts"),
)


def _linked_graph(conn: sqlite3.Connection, path: Path, options: ExportOptions,
                  out: dict[str, list[dict]], owned: list[str]) -> None:
    import json

    names = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")}
    for table, where, _kind in _LINKED_SELECTORS:
        if table not in names or "code_memory_links" not in names:
            continue
        try:
            out[table] = list(table_rows(conn, table, options, where=where,
                                         params=(json.dumps(owned),)))
        except sqlite3.Error as exc:
            logger.warning("export: code_graph.%s skipped: %s", table, type(exc).__name__)
            options.skipped_tables[f"{path.name}:{table}"] = type(exc).__name__
    left_out = sorted(n for n in names
                      if n not in out and not n.endswith(_FTS_SUFFIXES))
    if left_out:
        options.not_exported.append({"tables": left_out, "reason": (
            "the code graph is shared by every workspace on this computer and records no "
            "owner. Only the links from your memories to code, and the symbols and files they "
            "point at, are exported; the rest describes source code and may name another "
            "person's repositories")})


class ProfileExport:
    """Reads one profile's data for an export; see the module docstring."""

    def __init__(self, db: Any, profile_id: str, tables: list[str], *,
                 data_root: Path | None, options: ExportOptions | None = None,
                 code_graph_facts: list[str] | None = None) -> None:
        self._db = db
        self._profile = profile_id
        self._tables = tables
        self._root = data_root
        self.options = options or ExportOptions()
        #: ``None`` exports the whole code graph (the person is the only profile);
        #: a list restricts it to what those facts link to.
        self._code_graph_facts = code_graph_facts

    # -- sections -----------------------------------------------------------

    def sections(self) -> Iterator[Section]:
        """``(key, "rows" | "object", payload)`` in file order."""
        yield from self._memory_db_sections()
        if self._root is None:
            return
        graph = code_graph(self._root / "code_graph.db", self.options,
                           owned_fact_ids=self._code_graph_facts)
        if graph is not None:
            yield "code_graph", "object", graph
        from superlocalmemory.compliance.gdpr_media_export import export_media
        media = export_media(self._root, self._profile, self._db, self.options)
        if media is not None:
            yield "media", "object", media
        learned = sidecar_tables(self._root / "learning.db", self._profile, self.options)
        if learned is not None:
            yield "learning_signals", "object", learned
        chain = sidecar_tables(self._root / "audit_chain.db", self._profile, self.options,
                               only=("audit_chain",))
        if chain and chain.get("audit_chain"):
            yield "audit_chain", "rows", chain["audit_chain"]
        mesh = sidecar_tables(self._root / "mesh.db", self._profile, self.options)
        if mesh:
            yield "mesh", "object", mesh
        waiting = sidecar_tables(self._root / "pending.db", self._profile, self.options,
                                 only=("pending_memories",))
        if waiting and waiting.get("pending_memories"):
            yield "pending_saves", "rows", waiting["pending_memories"]

    def _memory_db_sections(self) -> Iterator[Section]:
        db_path = getattr(self._db, "db_path", None)
        if db_path is None or not Path(db_path).exists():
            yield from self._fallback_sections()
            return
        conn = _open_readonly(Path(db_path))
        try:
            conn.execute("BEGIN")  # one consistent snapshot for every table
            for table in self._tables:
                yield from self._one_table(conn, table, table, "profile_id = ?")
            yield from self._one_table(conn, "profiles", "profile_record", "profile_id = ?",
                                       empty_on_error=True)
            self._describe_exclusions(conn)
        finally:
            conn.close()

    def _one_table(self, conn: sqlite3.Connection, table: str, key: str, where: str, *,
                   empty_on_error: bool = False) -> Iterator[Section]:
        try:
            rows = table_rows(conn, table, self.options, where=where, params=(self._profile,))
        except sqlite3.Error as exc:
            logger.warning("export: table %s skipped: %s", table, type(exc).__name__)
            self.options.skipped_tables[table] = type(exc).__name__
            if empty_on_error:
                yield key, "rows", []
            return
        yield key, "rows", rows

    def _fallback_sections(self) -> Iterator[Section]:
        """A database object with no file behind it (a test double): materialised reads."""
        for table in self._tables:
            try:
                found = self._db.execute(
                    f"SELECT * FROM {_quote(table)} WHERE profile_id = ?", (self._profile,))
            except Exception as exc:  # noqa: BLE001
                self.options.skipped_tables[table] = type(exc).__name__
                continue
            yield table, "rows", (encode_row(dict(r), table, self.options) for r in found)
        try:
            record = self._db.execute(
                "SELECT * FROM profiles WHERE profile_id = ?", (self._profile,))
        except Exception:  # noqa: BLE001
            record = []
        yield "profile_record", "rows", (
            encode_row(dict(r), "profiles", self.options) for r in record)

    def _describe_exclusions(self, conn: sqlite3.Connection) -> None:
        exported = set(self._tables) | {"profiles"}
        index, system, receipts = [], [], []
        for (name,) in conn.execute("SELECT name FROM sqlite_master WHERE type='table'"):
            if name in exported or name.startswith("sqlite_"):
                continue
            if name == "erasure_receipts":
                receipts.append(name)
            elif "fts" in name or name.startswith("fact_embeddings") or "vector" in name:
                index.append(name)
            else:
                system.append(name)
        notes = self.options.not_exported
        if index:
            notes.append({"tables": sorted(index), "reason": SEARCH_INDEX_REASON})
        if system:
            notes.append({"tables": sorted(system), "reason": SYSTEM_TABLE_REASON})
        if receipts:
            notes.append({"tables": receipts, "reason": (
                "the operator's tamper-evident proof that an erasure took place; it "
                "holds hashes, not your text, and must survive the erasure it proves")})

    # -- outputs ------------------------------------------------------------

    def as_dict(self) -> dict:
        """The whole export as one dict (small stores and the in-process API)."""
        data: dict[str, Any] = {"profile_id": self._profile, "exported_at": _now()}
        for key, kind, payload in self.sections():
            data[key] = list(payload) if kind == "rows" else encode_value(
                payload, include_binary=self.options.include_embeddings)
        data["total_items"] = sum(len(v) for v in data.values() if isinstance(v, list))
        for friendly, table in EXPORT_ALIASES.items():
            if table in data:
                data[friendly] = data[table]
        data["export_manifest"] = self.options.manifest()
        return data

    def write(self, fp: BinaryIO) -> dict[str, Any]:
        """Stream the export to ``fp`` (opened ``w+b``); returns what was written."""
        writer = JsonObjectWriter(fp)
        writer.begin()
        writer.value("profile_id", self._profile)
        writer.value("exported_at", _now())
        counts: dict[str, int] = {}
        for key, kind, payload in self.sections():
            if kind == "rows":
                counts[key] = writer.array(key, payload)
            else:
                writer.value(key, encode_value(
                    payload, include_binary=self.options.include_embeddings))
        total = sum(counts.values())
        writer.value("total_items", total)
        for friendly, table in EXPORT_ALIASES.items():
            if writer.has_array(table):
                writer.alias(friendly, table)
        writer.value("export_manifest", self.options.manifest())
        writer.end()
        fp.flush()
        return {"bytes": fp.tell(), "total_items": total, "tables": counts}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


__all__ = ["EXPORT_ALIASES", "ProfileExport", "code_graph", "sidecar_tables", "table_rows"]
