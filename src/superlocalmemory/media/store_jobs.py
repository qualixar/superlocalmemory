# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Background-job rows of the media store: a small lease-based queue."""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Sequence

_FMT = "%Y-%m-%dT%H:%M:%S.%fZ"
_FINAL = ("done", "failed", "cancelled")


def utc_stamp(offset_s: float = 0.0) -> str:
    """A sortable UTC timestamp; same fixed width everywhere so text compares as time."""
    return (datetime.now(timezone.utc) + timedelta(seconds=offset_s)).strftime(_FMT)


class JobsMixin:
    """Mixed into MediaStore; needs its ``_write()`` and ``_read()`` helpers."""

    def enqueue_job(self, profile_id: str, kind: str, total: int = 0) -> str:
        job_id, now = uuid.uuid4().hex, utc_stamp()
        with self._write() as conn:
            conn.execute(
                "INSERT INTO jobs(job_id, profile_id, kind, state, done, total, created_at, updated_at)"
                " VALUES (?, ?, ?, 'queued', 0, ?, ?, ?)",
                (job_id, profile_id, kind, int(total), now, now),
            )
        return job_id

    def claim_job(self, owner: str, lease_s: float = 60,
                  kinds: Sequence[str] | None = None) -> dict[str, Any] | None:
        """Take one queued job (or a running one whose lease ran out); None if none."""
        now = utc_stamp()
        marks = ",".join("?" * len(kinds)) if kinds else ""
        kind_sql = f" AND kind IN ({marks})" if kinds else ""
        with self._write() as conn:
            row = conn.execute(
                "SELECT job_id FROM jobs WHERE (state = 'queued' OR (state = 'running' AND lease_until < ?))"
                + kind_sql + " ORDER BY created_at LIMIT 1", (now, *(kinds or ())),
            ).fetchone()
            if row is None:
                return None
            conn.execute(
                "UPDATE jobs SET state = 'running', lease_owner = ?, lease_until = ?, updated_at = ?"
                " WHERE job_id = ?", (owner, utc_stamp(lease_s), now, row[0]),
            )
        return self.get_job(row[0])

    def progress_job(self, job_id: str, owner: str, done: int, total: int | None = None) -> bool:
        """Record progress; False (and nothing written) when this owner no longer holds the job."""
        sets, args = ["done = ?", "updated_at = ?"], [int(done), utc_stamp()]
        if total is not None:
            sets.append("total = ?")
            args.append(int(total))
        with self._write() as conn:
            return conn.execute(
                f"UPDATE jobs SET {', '.join(sets)} WHERE job_id = ? AND lease_owner = ? AND state = 'running'",
                (*args, job_id, owner)).rowcount == 1

    def renew_lease(self, job_id: str, owner: str, lease_s: float = 60) -> bool:
        with self._write() as conn:
            return conn.execute(
                "UPDATE jobs SET lease_until = ?, updated_at = ? WHERE job_id = ? AND lease_owner = ?"
                " AND state = 'running'", (utc_stamp(lease_s), utc_stamp(), job_id, owner)).rowcount == 1

    def finish_job(self, job_id: str, owner: str, state: str, error: str | None = None) -> bool:
        """Finish a job this owner holds; False when the lease was lost to someone else."""
        if state not in _FINAL:
            raise ValueError(f"not a final job state: {state}")
        with self._write() as conn:
            return conn.execute(
                "UPDATE jobs SET state = ?, error = ?, lease_owner = NULL, lease_until = NULL,"
                " updated_at = ? WHERE job_id = ? AND lease_owner = ? AND state = 'running'",
                (state, error, utc_stamp(), job_id, owner)).rowcount == 1

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        row = self._read().execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        return dict(row) if row else None

    def list_jobs(self, profile_id: str, states: Sequence[str] | None = None) -> list[dict[str, Any]]:
        sql, args = "SELECT * FROM jobs WHERE profile_id = ?", [profile_id]
        if states:
            sql += f" AND state IN ({','.join('?' * len(states))})"
            args += list(states)
        rows = self._read().execute(sql + " ORDER BY created_at", args).fetchall()
        return [dict(r) for r in rows]

    def purge_jobs(self, older_than_days: int = 30) -> int:
        """Forget finished jobs untouched for that long; queued/running ones stay."""
        cutoff = utc_stamp(-older_than_days * 86400)
        with self._write() as conn:
            return conn.execute(
                f"DELETE FROM jobs WHERE state IN ({','.join('?' * len(_FINAL))}) AND updated_at < ?",
                (*_FINAL, cutoff)).rowcount
