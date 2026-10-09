# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""The ``sources``, ``source_files`` tables of media.db. Only the sources package touches them."""

from __future__ import annotations

import json
import uuid
from typing import Any, Sequence

from superlocalmemory.media.store_jobs import utc_stamp


def entries_of(row: dict[str, Any]) -> list[dict[str, Any]]:
    """The memories of a file row: ``{"m": memory_id, "f": [fact ids], "v": version, "sup": when}``."""
    try:
        found = json.loads(row.get("memory_ids_json") or "[]")
    except ValueError:
        return []
    return [e for e in found if isinstance(e, dict)]


def memory_entries(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The memories this file owns. ``shared_m`` / ``shared_doc`` entries point at things it does not own."""
    return [e for e in entries if "m" in e]


class SourceStore:
    def __init__(self, media: Any) -> None:
        self._m = media

    # -- sources -----------------------------------------------------------------
    def create_source(self, profile_id: str, kind: str, root_path: str, display_name: str,
                      include_types: Sequence[str], source_id: str | None = None) -> str:
        """Add an active source, or return the id of the live one for this root."""
        with self._m._write() as conn:
            row = conn.execute("SELECT source_id FROM sources WHERE profile_id = ? AND root_path = ?"
                               " AND state != 'removed'", (profile_id, root_path)).fetchone()
            if row:
                return row[0]
            source_id = source_id or uuid.uuid4().hex
            conn.execute(
                "INSERT INTO sources(source_id, profile_id, kind, root_path, display_name,"
                " include_types_json, state, remote_visible, watch, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?, 'active', 0, 1, ?)",
                (source_id, profile_id, kind, root_path, display_name, json.dumps(list(include_types)),
                 utc_stamp()))
        return source_id

    def get_source(self, source_id: str) -> dict[str, Any] | None:
        row = self._m._read().execute("SELECT * FROM sources WHERE source_id = ?", (source_id,)).fetchone()
        return dict(row) if row else None

    def list_sources(self, profile_id: str | None = None, *, states: Sequence[str] | None = None
                     ) -> list[dict[str, Any]]:
        sql, args = "SELECT * FROM sources WHERE state != 'removed'", []
        if profile_id is not None:
            sql += " AND profile_id = ?"
            args.append(profile_id)
        if states:
            sql += f" AND state IN ({','.join('?' * len(states))})"
            args += list(states)
        return [dict(r) for r in self._m._read().execute(sql + " ORDER BY created_at", args)]

    def set_state(self, source_id: str, state: str, *, stats: dict[str, Any] | None = None,
                  scanned: bool = False) -> None:
        sets, args = ["state = ?"], [state]
        if stats is not None:
            sets.append("last_scan_stats_json = ?")
            args.append(json.dumps(stats))
        if scanned:
            sets.append("last_scan_at = ?")
            args.append(utc_stamp())
        with self._m._write() as conn:
            conn.execute(f"UPDATE sources SET {', '.join(sets)} WHERE source_id = ?", (*args, source_id))

    # -- files -------------------------------------------------------------------
    def files(self, source_id: str, states: Sequence[str] | None = None) -> list[dict[str, Any]]:
        sql, args = "SELECT * FROM source_files WHERE source_id = ?", [source_id]
        if states:
            sql += f" AND state IN ({','.join('?' * len(states))})"
            args += list(states)
        return [dict(r) for r in self._m._read().execute(sql + " ORDER BY relpath", args)]

    def get_file(self, source_id: str, relpath: str) -> dict[str, Any] | None:
        row = self._m._read().execute("SELECT * FROM source_files WHERE source_id = ? AND relpath = ?",
                                      (source_id, relpath)).fetchone()
        return dict(row) if row else None

    def put_file(self, source_id: str, relpath: str, **fields: Any) -> None:
        """Insert or update a file row; ``fields`` are column values (entries as a list)."""
        if "entries" in fields:
            fields["memory_ids_json"] = json.dumps(fields.pop("entries"))
        existing = self.get_file(source_id, relpath)
        row = {"size": 0, "mtime_ns": 0, "file_id": None, "sha256": None, "state": "pending",
               "reason": None, "memory_ids_json": "[]", "document_id": None, "media_id": None,
               "tombstoned_at": None}
        row.update({k: v for k, v in (existing or {}).items() if k in row})
        row.update(fields)
        row["updated_at"] = utc_stamp()
        cols = ["source_id", "relpath", *row]
        with self._m._write() as conn:
            conn.execute(
                f"INSERT OR REPLACE INTO source_files({', '.join(cols)}) VALUES ({','.join('?' * len(cols))})",
                (source_id, relpath, *row.values()))

    def repoint(self, source_id: str, old: str, new: str, **fields: Any) -> None:
        """A moved file keeps its row (and memories) under the new path."""
        row = self.get_file(source_id, old)
        if row is None:
            return
        with self._m._write() as conn:
            conn.execute("DELETE FROM source_files WHERE source_id = ? AND relpath = ?", (source_id, old))
        keep = {k: row[k] for k in ("size", "mtime_ns", "file_id", "sha256", "state", "reason",
                                    "memory_ids_json", "document_id", "media_id")}
        keep.update(fields)
        self.put_file(source_id, new, **keep)

    def release_shared(self, source_id: str, sha256: str | None, except_relpath: str) -> int:
        """Queue the other copies of these bytes that borrowed the owner's save to be saved afresh."""
        if not sha256:
            return 0
        with self._m._write() as conn:
            return conn.execute(
                "UPDATE source_files SET state = 'pending' WHERE source_id = ? AND sha256 = ?"
                " AND reason = 'shared' AND relpath != ? AND state IN ('indexed', 'pending')",
                (source_id, sha256, except_relpath)).rowcount

    def next_save_n(self, source_id: str, relpath: str) -> int:
        """How many times this path has been saved, counting this one. Never reset, not even by a purge."""
        with self._m._write() as conn:
            conn.execute(
                "INSERT INTO source_save_counters(source_id, relpath, n) VALUES (?, ?, 1)"
                " ON CONFLICT(source_id, relpath) DO UPDATE SET n = n + 1", (source_id, relpath))
            return conn.execute("SELECT n FROM source_save_counters WHERE source_id = ? AND relpath = ?",
                                (source_id, relpath)).fetchone()[0]

    def delete_file(self, source_id: str, relpath: str) -> None:
        with self._m._write() as conn:
            conn.execute("DELETE FROM source_files WHERE source_id = ? AND relpath = ?", (source_id, relpath))

    def delete_source_rows(self, source_id: str) -> None:
        with self._m._write() as conn:
            for table in ("source_files", "source_links", "source_save_counters"):
                conn.execute(f"DELETE FROM {table} WHERE source_id = ?", (source_id,))
            conn.execute("UPDATE sources SET state = 'removed' WHERE source_id = ?", (source_id,))

    def counts(self, source_id: str) -> dict[str, int]:
        rows = self._m._read().execute(
            "SELECT state, COUNT(*) FROM source_files WHERE source_id = ? GROUP BY state", (source_id,))
        return {r[0]: r[1] for r in rows}

    # -- jobs --------------------------------------------------------------------
    def queue_scan(self, profile_id: str, source_id: str) -> dict[str, Any]:
        """Queue a scan unless one is already waiting or running for this source."""
        for job in self._m.list_jobs(profile_id, ["queued", "running"]):
            if job["kind"] == "source_scan" and json.loads(job["payload_json"]).get("source_id") == source_id:
                return job
        job_id = self._m.enqueue_job(profile_id, "source_scan", 0, {"source_id": source_id})
        return self._m.get_job(job_id)
