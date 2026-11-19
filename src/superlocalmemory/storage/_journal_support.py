# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later

"""Constants, errors and pure helpers of the admission journal.

Split out of ``storage/admission_journal.py`` (final audit F-15, 19 Nov 2026) so that module stays
under 800 lines. ``storage/admission_journal`` re-exports every name here, so every existing import
keeps working unchanged.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import time
from collections.abc import Mapping
from typing import Any

from superlocalmemory.storage.journal_writer import AdmissionJournalUnavailable, is_sqlite_busy

_MAX_COMMAND_BYTES = 256 * 1024
_MAX_RECEIPT_BYTES = 16 * 1024
_MAX_METADATA_DEPTH = 8
_IDEMPOTENCY_KEY = re.compile(r"^[A-Za-z0-9._:-]{1,256}$")
_STATES = frozenset({"prepared", "dispatched", "committed", "rejected"})
#: Error code of an entry set aside because its command cannot be read back
#: (corrupt bytes, or written under another machine key). The row and its
#: original encrypted bytes are kept untouched for an operator.
UNREADABLE_COMMAND = "UNREADABLE_COMMAND"

_JOURNAL_DDL = """
CREATE TABLE IF NOT EXISTS admission_journal (
    journal_id TEXT PRIMARY KEY,
    idempotency_key TEXT NOT NULL,
    request_hash TEXT NOT NULL,
    profile_id TEXT NOT NULL,
    command_json TEXT NOT NULL,
    state TEXT NOT NULL CHECK (
        state IN ('prepared','dispatched','committed','rejected')
    ),
    canonical_operation_id TEXT,
    canonical_commit_sequence INTEGER,
    error_code TEXT,
    receipt_json TEXT,
    created_at_ms INTEGER NOT NULL,
    updated_at_ms INTEGER NOT NULL,
    UNIQUE(profile_id, idempotency_key)
);
CREATE INDEX IF NOT EXISTS idx_admission_replay
    ON admission_journal(state, updated_at_ms);
"""

_JOURNAL_TABLE_DDL = _JOURNAL_DDL.split(";", 1)[0]
_JOURNAL_REPLAY_INDEX_DDL = (
    "CREATE INDEX IF NOT EXISTS idx_admission_replay "
    "ON admission_journal(state, updated_at_ms)"
)


class IdempotencyConflict(ValueError):
    """The key belongs to a different immutable remember request."""


class AdmissionAuthorizationError(PermissionError):
    """The actor is not entitled to submit the requested profile and scope."""


class AdmissionPayloadError(ValueError):
    """The admission body is invalid, oversized, or cannot be encoded safely."""


#: The code every deterministic refusal carried before 4.2.18, when the
#: reason was not recorded. The only producer was "ingestion produced no
#: queryable facts" (an ingest-policy refusal). Since 4.2.18 no code path
#: writes it bare: each refusal names its reason (``COMMAND_REJECTED:<why>``).
UNEXPLAINED_REJECTION = "COMMAND_REJECTED"


class TerminalAdmissionError(RuntimeError):
    """A replayable command was deterministically rejected after journaling."""

    def __init__(self, error_code: str = "COMMAND_REJECTED") -> None:
        if not error_code or len(error_code) > 128:
            raise ValueError("error_code is required and bounded")
        super().__init__(error_code)
        self.error_code = error_code


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
            "utf-8"
        )
    except (TypeError, ValueError) as exc:
        raise AdmissionPayloadError("remember command must be JSON serializable") from exc


def _request_hash(payload: Mapping[str, Any]) -> str:
    """The idempotency hash of one remember command, bound to its stable caller."""
    from superlocalmemory.storage.idempotency_identity import stable_request_payload

    return hashlib.sha256(_canonical_bytes(stable_request_payload(payload))).hexdigest()


def _receipt_json(receipt: Mapping[str, Any]) -> str:
    if not isinstance(receipt, Mapping):
        raise ValueError("receipt must be an object")
    _validate_json(dict(receipt), "receipt")
    _reject_raw_content(receipt)
    rendered = _canonical_bytes(dict(receipt))
    if len(rendered) > _MAX_RECEIPT_BYTES:
        raise ValueError("receipt exceeds journal receipt limit")
    return rendered.decode("utf-8")


def _validate_json(value: Any, label: str, depth: int = 0) -> None:
    if depth > _MAX_METADATA_DEPTH:
        raise AdmissionPayloadError(f"{label} nesting exceeds limit")
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise AdmissionPayloadError(f"{label} keys must be strings")
            _validate_json(child, label, depth + 1)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _validate_json(child, label, depth + 1)
    elif not isinstance(value, (str, int, float, bool, type(None))):
        raise AdmissionPayloadError(f"{label} must be JSON serializable")


def _reject_raw_content(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if key.casefold() in {
                "content",
                "content_preview",
                "raw_content",
                "memory_content",
                "source_content",
            }:
                raise ValueError("receipt must not include raw memory content")
            _reject_raw_content(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _reject_raw_content(child)


def _now_ms() -> int:
    return int(time.time() * 1000)


def _remaining_seconds(deadline: float | None) -> float:
    if deadline is None:
        return 1.0
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise AdmissionJournalUnavailable("admission journal deadline expired")
    return remaining


_is_sqlite_busy = is_sqlite_busy


def _has_unique_index(
    conn: sqlite3.Connection, table: str, columns: tuple[str, ...]
) -> bool:
    """Return whether SQLite enforces exactly these columns as a unique key."""
    for index in conn.execute(f"PRAGMA index_list({table})").fetchall():
        if not index[2]:
            continue
        names = tuple(
            row[2] for row in conn.execute(f"PRAGMA index_info({index[1]})").fetchall()
        )
        if names == columns:
            return True
    return False


def _create_journal_schema(conn: sqlite3.Connection) -> None:
    conn.execute(_JOURNAL_TABLE_DDL)
    conn.execute(_JOURNAL_REPLAY_INDEX_DDL)
