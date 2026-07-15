# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Bounded, content-free operational diagnostics stored on this machine.

The store accepts only enumerated operations and normalized client families.
It persists daily counters, latency buckets, coarse error classes, and client
family transitions. Export is a deliberate local file operation; this module
has no automatic reporting mechanism.
"""

from __future__ import annotations

import json
import math
import os
import sqlite3
import tempfile
from collections.abc import Callable, Iterable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from superlocalmemory.infra.data_root import state_path

SCHEMA_VERSION = 1
RETENTION_DAYS = 31
MAX_COUNTER = 2**63 - 1

_OPERATIONS = frozenset(
    {
        "activation",
        "remember",
        "recall",
        "observe",
        "session_open",
        "session_close",
    }
)
_CLIENT_FAMILIES = (
    "claude",
    "codex",
    "cursor",
    "windsurf",
    "copilot",
    "vscode",
    "jetbrains",
    "dashboard",
    "hook",
    "mcp",
    "api",
    "cli",
)
_LATENCY_BOUNDS_MS = (5, 25, 100, 500, 2_000, 10_000, 60_000)
_ERROR_CLASSES = frozenset(
    {"timeout", "authorization", "validation", "unavailable", "internal"}
)


def normalize_client(value: str | None) -> str:
    """Collapse an arbitrary actor label to a fixed non-identifying family."""
    candidate = str(value or "").strip().lower()
    for family in _CLIENT_FAMILIES:
        if family in candidate:
            return family
    return "other"


def classify_error(error: BaseException | None) -> str | None:
    """Map an exception type to a fixed class without inspecting its message."""
    if error is None:
        return None
    if isinstance(error, TimeoutError):
        return "timeout"
    if isinstance(error, PermissionError):
        return "authorization"
    if isinstance(error, (TypeError, ValueError)):
        return "validation"
    if isinstance(error, (ConnectionError, FileNotFoundError)):
        return "unavailable"
    return "internal"


def _latency_bucket(duration_ms: float | int | None) -> str | None:
    if duration_ms is None:
        return None
    try:
        value = float(duration_ms)
    except (TypeError, ValueError, OverflowError):
        return "invalid"
    if not math.isfinite(value) or value < 0:
        return "invalid"
    for bound in _LATENCY_BOUNDS_MS:
        if value <= bound:
            return f"le_{bound}"
    return "gt_60000"


class LocalDiagnostics:
    """SQLite-backed bounded aggregate diagnostics.

    The schema intentionally has no free-text value column. Every persisted
    dimension comes from a finite allow-list, so caller payloads cannot enter
    the database or an export by accident.
    """

    def __init__(
        self,
        db_file: str | Path,
        *,
        clock: Callable[[], datetime] | None = None,
        retention_days: int = RETENTION_DAYS,
    ) -> None:
        if retention_days < 1 or retention_days > RETENTION_DAYS:
            raise ValueError(f"retention_days must be between 1 and {RETENTION_DAYS}")
        self._db_file = Path(db_file)
        self._clock = clock or (lambda: datetime.now(UTC))
        self._retention_days = retention_days

    def _connect(self) -> sqlite3.Connection:
        self._db_file.parent.mkdir(parents=True, exist_ok=True)
        if os.name == "posix":
            os.chmod(self._db_file.parent, 0o700)
        conn = sqlite3.connect(str(self._db_file), timeout=5.0)
        conn.execute(
            "CREATE TABLE IF NOT EXISTS diagnostic_aggregates ("
            "day TEXT NOT NULL, group_name TEXT NOT NULL, "
            "dimension TEXT NOT NULL, bucket TEXT NOT NULL, "
            "value INTEGER NOT NULL CHECK(value >= 0), "
            "PRIMARY KEY(day, group_name, dimension, bucket))"
        )
        conn.commit()
        if os.name == "posix":
            os.chmod(self._db_file, 0o600)
        return conn

    def _today(self) -> datetime:
        current = self._clock()
        if current.tzinfo is None:
            current = current.replace(tzinfo=UTC)
        return current.astimezone(UTC)

    @staticmethod
    def _increment(
        conn: sqlite3.Connection,
        day: str,
        group_name: str,
        dimension: str,
        bucket: str,
    ) -> None:
        conn.execute(
            "INSERT INTO diagnostic_aggregates "
            "(day, group_name, dimension, bucket, value) VALUES (?,?,?,?,1) "
            "ON CONFLICT(day, group_name, dimension, bucket) DO UPDATE SET "
            "value=MIN(diagnostic_aggregates.value + 1, ?)",
            (day, group_name, dimension, bucket, MAX_COUNTER),
        )

    def record(
        self,
        operation: str,
        *,
        client: str | None = None,
        duration_ms: float | int | None = None,
        error: BaseException | None = None,
        source_clients: Iterable[str] = (),
    ) -> None:
        """Record one operation using only fixed aggregate dimensions."""
        if operation not in _OPERATIONS:
            raise ValueError(f"unsupported diagnostic operation: {operation}")
        current = self._today()
        day = current.date().isoformat()
        cutoff = (current.date() - timedelta(days=self._retention_days - 1)).isoformat()
        client_family = normalize_client(client)
        status = "error" if error is not None else "success"
        duration_bucket = _latency_bucket(duration_ms)
        error_class = classify_error(error)
        transitions = {
            f"{normalize_client(source)}_to_{client_family}"
            for source in source_clients
            if normalize_client(source) != client_family
        }

        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute("DELETE FROM diagnostic_aggregates WHERE day < ?", (cutoff,))
            self._increment(conn, day, "operations", operation, status)
            self._increment(conn, day, "clients", client_family, operation)
            if duration_bucket is not None:
                self._increment(conn, day, "latency_ms", operation, duration_bucket)
            if error_class is not None:
                self._increment(conn, day, "errors", operation, error_class)
            if operation == "activation":
                self._increment(conn, day, "activations", client_family, "count")
            for transition in sorted(transitions):
                self._increment(conn, day, "handoffs", transition, "count")
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
        finally:
            conn.close()

    def export_payload(self) -> dict[str, Any]:
        """Return a deterministic, sanitized aggregate document."""
        days: dict[str, dict[str, Any]] = {}
        if self._db_file.is_file():
            conn = self._connect()
            try:
                rows = conn.execute(
                    "SELECT day, group_name, dimension, bucket, value "
                    "FROM diagnostic_aggregates "
                    "ORDER BY day, group_name, dimension, bucket"
                ).fetchall()
            finally:
                conn.close()
            for day, group_name, dimension, bucket, value in rows:
                entry = days.setdefault(str(day), {"day": str(day)})
                group = entry.setdefault(str(group_name), {})
                dimension_values = group.setdefault(str(dimension), {})
                dimension_values[str(bucket)] = int(value)
        return {
            "schema_version": SCHEMA_VERSION,
            "retention_days": self._retention_days,
            "privacy": {
                "local_only": True,
                "reporting": "manual_export_only",
                "content_values": False,
                "query_values": False,
                "fact_identifiers": False,
                "filesystem_locations": False,
                "identity_values": False,
            },
            "days": [days[key] for key in sorted(days)],
        }

    def export_json(self, destination: str | Path) -> dict[str, Any]:
        """Atomically write a deterministic manual export with mode 0600."""
        payload = self.export_payload()
        output = Path(destination)
        output.parent.mkdir(parents=True, exist_ok=True)
        data = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
        temporary_name = ""
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb", dir=output.parent, prefix=f".{output.name}.", delete=False,
            ) as temporary:
                temporary_name = temporary.name
                os.chmod(temporary_name, 0o600)
                temporary.write(data)
                temporary.flush()
                os.fsync(temporary.fileno())
            os.replace(temporary_name, output)
            if os.name == "posix":
                os.chmod(output, 0o600)
        finally:
            if temporary_name:
                Path(temporary_name).unlink(missing_ok=True)
        return payload


def default_diagnostics() -> LocalDiagnostics:
    """Return the diagnostics store for the active local state root."""
    return LocalDiagnostics(state_path("diagnostics.db"))


def record_operation(
    operation: str,
    *,
    client: str | None = None,
    duration_ms: float | int | None = None,
    error: BaseException | None = None,
    source_clients: Iterable[str] = (),
) -> None:
    """Best-effort hot-path adapter; diagnostics never affect correctness."""
    try:
        default_diagnostics().record(
            operation,
            client=client,
            duration_ms=duration_ms,
            error=error,
            source_clients=source_clients,
        )
    except Exception:
        return


def record_recall(db: Any, response: Any, *, client: str | None) -> None:
    """Record recall and aggregate cross-client provenance transitions."""
    source_clients: list[str] = []
    fact_ids = [
        str(result.fact.fact_id)
        for result in getattr(response, "results", ())
        if getattr(getattr(result, "fact", None), "fact_id", None)
    ]
    profile_id = ""
    if getattr(response, "results", None):
        profile_id = str(getattr(response.results[0].fact, "profile_id", ""))
    if fact_ids and profile_id:
        try:
            marks = ",".join("?" for _ in fact_ids)
            rows = db.execute(
                f"SELECT DISTINCT created_by FROM provenance "
                f"WHERE profile_id=? AND fact_id IN ({marks})",
                (profile_id, *fact_ids),
            )
            source_clients = [str(dict(row).get("created_by", "")) for row in rows]
        except Exception:
            source_clients = []
    record_operation(
        "recall",
        client=client,
        duration_ms=getattr(response, "retrieval_time_ms", None),
        source_clients=source_clients,
    )


__all__ = [
    "LocalDiagnostics",
    "classify_error",
    "default_diagnostics",
    "normalize_client",
    "record_operation",
    "record_recall",
]
