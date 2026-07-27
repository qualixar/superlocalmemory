# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com

"""Separate FULL-synchronous journal for durable remember admission.

The journal is intentionally not a memory store.  Its only mutable state is a
small encrypted replay command and the canonical receipt associated with it.
Canonical facts, FTS, graph, vectors, and model work stay outside this module.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import re
import sqlite3
import time
import uuid
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generator, Protocol

from cryptography.exceptions import InvalidTag

_MAX_COMMAND_BYTES = 256 * 1024
_MAX_RECEIPT_BYTES = 16 * 1024
_MAX_METADATA_DEPTH = 8
_IDEMPOTENCY_KEY = re.compile(r"^[A-Za-z0-9._:-]{1,256}$")
_STATES = frozenset({"prepared", "dispatched", "committed", "rejected"})

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


class TerminalAdmissionError(RuntimeError):
    """A replayable command was deterministically rejected after journaling."""

    def __init__(self, error_code: str = "COMMAND_REJECTED") -> None:
        if not error_code or len(error_code) > 128:
            raise ValueError("error_code is required and bounded")
        super().__init__(error_code)
        self.error_code = error_code


class CommandCodec(Protocol):
    """Existing product encryption policy injected by the runtime.

    The journal deliberately owns no key derivation or cryptographic primitive.
    This prevents it from creating a second, incompatible encryption policy.
    """

    def encrypt(self, plaintext: bytes) -> bytes: ...

    def decrypt(self, ciphertext: bytes) -> bytes: ...


@dataclass(frozen=True, slots=True)
class Actor:
    """Bounded authorization context supplied by the authenticated boundary."""

    principal_id: str
    allowed_profiles: frozenset[str]
    allowed_scopes: frozenset[str]
    trusted: bool = True

    def permits(self, profile_id: str, scope: str) -> bool:
        return self.trusted and profile_id in self.allowed_profiles and scope in self.allowed_scopes


@dataclass(frozen=True, slots=True)
class RememberRequest:
    """Immutable canonical input for a single remember admission."""

    content: str
    profile_id: str
    source_type: str
    idempotency_key: str
    metadata: Mapping[str, Any] = field(default_factory=dict)
    scope: str = "personal"
    shared_with: tuple[str, ...] = ()
    trusted_actor_id: str = ""
    session_id: str = ""
    session_date: str = ""
    speaker: str = ""
    role: str = "user"

    def __post_init__(self) -> None:
        if not isinstance(self.content, str) or not self.content.strip():
            raise AdmissionPayloadError("content is required")
        for name in ("profile_id", "source_type"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise AdmissionPayloadError(f"{name} is required")
        if not isinstance(self.idempotency_key, str) or not _IDEMPOTENCY_KEY.fullmatch(
            self.idempotency_key
        ):
            raise AdmissionPayloadError("idempotency_key must be 1-256 safe characters")
        if self.scope not in {"personal", "project", "shared", "global"}:
            raise AdmissionPayloadError(f"unsupported scope: {self.scope}")
        if not isinstance(self.metadata, Mapping):
            raise AdmissionPayloadError("metadata must be an object")
        metadata = dict(self.metadata)
        _validate_json(metadata, "metadata")
        object.__setattr__(self, "metadata", metadata)
        object.__setattr__(self, "shared_with", tuple(self.shared_with))

    def canonical_payload(self) -> dict[str, Any]:
        return {
            "content": self.content,
            "idempotency_key": self.idempotency_key,
            "metadata": dict(self.metadata),
            "profile_id": self.profile_id,
            "role": self.role,
            "scope": self.scope,
            "session_date": self.session_date,
            "session_id": self.session_id,
            "shared_with": list(self.shared_with),
            "source_type": self.source_type,
            "speaker": self.speaker,
            "trusted_actor_id": self.trusted_actor_id,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> RememberRequest:
        return cls(
            content=str(payload["content"]),
            profile_id=str(payload["profile_id"]),
            source_type=str(payload["source_type"]),
            idempotency_key=str(payload["idempotency_key"]),
            metadata=dict(payload.get("metadata") or {}),
            scope=str(payload.get("scope") or "personal"),
            shared_with=tuple(payload.get("shared_with") or ()),
            trusted_actor_id=str(payload.get("trusted_actor_id") or ""),
            session_id=str(payload.get("session_id") or ""),
            session_date=str(payload.get("session_date") or ""),
            speaker=str(payload.get("speaker") or ""),
            role=str(payload.get("role") or "user"),
        )


@dataclass(frozen=True, slots=True)
class AdmissionEntry:
    """Content-free journal metadata safe for status and recovery decisions."""

    journal_id: str
    idempotency_key: str
    request_hash: str
    profile_id: str
    state: str
    canonical_operation_id: str | None
    canonical_commit_sequence: int | None
    error_code: str | None
    created_at_ms: int
    updated_at_ms: int
    original_receipt: dict[str, Any] | None = None


PreparedAdmission = AdmissionEntry


class AdmissionJournal:
    """Synchronous idempotency journal stored independently from ``memory.db``."""

    def __init__(self, path: str | Path, *, codec: CommandCodec) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._codec = codec
        self._initialize()

    def prepare(self, request: RememberRequest, actor: Actor) -> PreparedAdmission:
        """Durably prepare one encrypted replay command before dispatch."""
        if not actor.principal_id.strip() or not actor.permits(request.profile_id, request.scope):
            raise AdmissionAuthorizationError(
                "actor is not authorized for requested profile or scope"
            )
        if request.trusted_actor_id and request.trusted_actor_id != actor.principal_id:
            raise AdmissionAuthorizationError(
                "trusted actor does not match authenticated principal"
            )

        payload = request.canonical_payload()
        plaintext = _canonical_bytes(payload)
        if len(plaintext) > _MAX_COMMAND_BYTES:
            raise AdmissionPayloadError("remember command exceeds journal payload limit")
        encrypted = self._codec.encrypt(plaintext)
        if not isinstance(encrypted, bytes) or not encrypted:
            raise AdmissionPayloadError("configured command codec returned no ciphertext")
        command_json = json.dumps({"ciphertext_b64": base64.b64encode(encrypted).decode("ascii")})
        request_hash = hashlib.sha256(plaintext).hexdigest()
        now = _now_ms()
        journal_id = uuid.uuid4().hex

        with self._connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                existing = conn.execute(
                    "SELECT * FROM admission_journal "
                    "WHERE profile_id=? AND idempotency_key=?",
                    (request.profile_id, request.idempotency_key),
                ).fetchone()
                if existing is not None:
                    entry = self._entry_from_row(existing)
                    if entry.request_hash != request_hash:
                        raise IdempotencyConflict(
                            "idempotency key belongs to a different immutable request"
                        )
                    conn.commit()
                    return entry
                conn.execute(
                    "INSERT INTO admission_journal "
                    "(journal_id, idempotency_key, request_hash, profile_id, command_json, state, "
                    "created_at_ms, updated_at_ms) VALUES (?, ?, ?, ?, ?, 'prepared', ?, ?)",
                    (
                        journal_id,
                        request.idempotency_key,
                        request_hash,
                        request.profile_id,
                        command_json,
                        now,
                        now,
                    ),
                )
                row = conn.execute(
                    "SELECT * FROM admission_journal WHERE journal_id=?", (journal_id,)
                ).fetchone()
                conn.commit()
            except BaseException:
                conn.rollback()
                raise
        assert row is not None
        return self._entry_from_row(row)

    def request_for(self, entry: AdmissionEntry) -> RememberRequest:
        """Decrypt the minimal replay body only immediately before canonical work."""
        with self._connection() as conn:
            row = conn.execute(
                "SELECT command_json FROM admission_journal WHERE journal_id=?", (entry.journal_id,)
            ).fetchone()
        if row is None:
            raise KeyError(entry.journal_id)
        try:
            encoded = json.loads(str(row["command_json"]))["ciphertext_b64"]
            plaintext = self._codec.decrypt(base64.b64decode(encoded, validate=True))
            payload = json.loads(plaintext.decode("utf-8"))
        except (
            KeyError,
            TypeError,
            UnicodeDecodeError,
            ValueError,
            binascii.Error,
            json.JSONDecodeError,
            InvalidTag,
        ) as exc:
            raise AdmissionPayloadError(
                "journal command cannot be decrypted by the configured policy"
            ) from exc
        return RememberRequest.from_payload(payload)

    def mark_dispatched(self, journal_id: str) -> AdmissionEntry:
        # A concurrent retry may observe ``prepared`` and then lose the race to
        # another caller that commits the same idempotent command.  Treat that
        # terminal state as a successful no-op so the retry can return the
        # canonical receipt instead of surfacing a false transition failure.
        return self._transition(
            journal_id,
            target="dispatched",
            allowed={"prepared", "dispatched", "committed"},
        )

    def mark_rejected(self, journal_id: str, error_code: str) -> AdmissionEntry:
        if not error_code or len(error_code) > 128:
            raise ValueError("error_code is required and bounded")
        return self._transition(
            journal_id,
            target="rejected",
            allowed={"prepared", "dispatched", "rejected"},
            error_code=error_code,
        )

    def mark_committed(self, journal_id: str, receipt: Mapping[str, Any]) -> AdmissionEntry:
        receipt_json = _receipt_json(receipt)
        data = json.loads(receipt_json)
        operation_id = data.get("operation_id")
        commit_sequence = data.get("commit_sequence")
        if operation_id is not None and not isinstance(operation_id, str):
            raise ValueError("receipt operation_id must be a string")
        if commit_sequence is not None and not isinstance(commit_sequence, int):
            raise ValueError("receipt commit_sequence must be an integer")
        return self._transition(
            journal_id,
            target="committed",
            allowed={"prepared", "dispatched", "rejected", "committed"},
            receipt_json=receipt_json,
            operation_id=operation_id,
            commit_sequence=commit_sequence,
        )

    def get(self, journal_id: str) -> AdmissionEntry:
        with self._connection() as conn:
            row = conn.execute(
                "SELECT * FROM admission_journal WHERE journal_id=?", (journal_id,)
            ).fetchone()
        if row is None:
            raise KeyError(journal_id)
        return self._entry_from_row(row)

    def get_by_idempotency_key(
        self, profile_id: str, idempotency_key: str
    ) -> AdmissionEntry | None:
        """Return a profile-scoped retry record, never a cross-profile match."""
        with self._connection() as conn:
            row = conn.execute(
                "SELECT * FROM admission_journal WHERE profile_id=? AND idempotency_key=?",
                (profile_id, idempotency_key),
            ).fetchone()
        return self._entry_from_row(row) if row is not None else None

    def count(self) -> int:
        with self._connection() as conn:
            return int(conn.execute("SELECT COUNT(*) FROM admission_journal").fetchone()[0])

    def replay_pending(
        self,
        find_canonical_receipt: Callable[[AdmissionEntry], Mapping[str, Any] | None],
        dispatch: Callable[[AdmissionEntry, RememberRequest], Mapping[str, Any]],
        *,
        profile_id: str | None = None,
    ) -> int:
        """Resolve crash-surviving entries without duplicate canonical writes.

        A daemon runtime is bound to one active profile at a time. Pending
        commands for other profiles remain durable until that profile is
        rebound; dispatching them through the wrong profile writer would make
        one abandoned command prevent the daemon from starting.
        """
        with self._connection() as conn:
            if profile_id is None:
                rows = conn.execute(
                    "SELECT * FROM admission_journal "
                    "WHERE state IN ('prepared', 'dispatched') "
                    "ORDER BY created_at_ms, journal_id"
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM admission_journal "
                    "WHERE state IN ('prepared', 'dispatched') AND profile_id=? "
                    "ORDER BY created_at_ms, journal_id",
                    (profile_id,),
                ).fetchall()
        recovered = 0
        for row in rows:
            entry = self._entry_from_row(row)
            try:
                canonical = find_canonical_receipt(entry)
                if canonical is None:
                    canonical = dispatch(entry, self.request_for(entry))
            except TerminalAdmissionError as exc:
                self.mark_rejected(entry.journal_id, exc.error_code)
                recovered += 1
                continue
            self.mark_committed(entry.journal_id, canonical)
            recovered += 1
        return recovered

    def _transition(
        self,
        journal_id: str,
        *,
        target: str,
        allowed: set[str],
        error_code: str | None = None,
        receipt_json: str | None = None,
        operation_id: str | None = None,
        commit_sequence: int | None = None,
    ) -> AdmissionEntry:
        with self._connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                row = conn.execute(
                    "SELECT * FROM admission_journal WHERE journal_id=?", (journal_id,)
                ).fetchone()
                if row is None:
                    raise KeyError(journal_id)
                previous = self._entry_from_row(row)
                if previous.state not in allowed:
                    if previous.state == "committed":
                        raise ValueError("journal entry is already committed")
                    raise ValueError(f"illegal admission transition {previous.state} -> {target}")
                if previous.state == "committed":
                    conn.commit()
                    return previous
                conn.execute(
                    "UPDATE admission_journal SET "
                    "state=?, "
                    "canonical_operation_id=COALESCE(?, canonical_operation_id), "
                    "canonical_commit_sequence=COALESCE(?, canonical_commit_sequence), "
                    "error_code=?, "
                    "receipt_json=COALESCE(?, receipt_json), "
                    "updated_at_ms=? WHERE journal_id=? AND state=?",
                    (
                        target,
                        operation_id,
                        commit_sequence,
                        error_code,
                        receipt_json,
                        _now_ms(),
                        journal_id,
                        previous.state,
                    ),
                )
                updated = conn.execute(
                    "SELECT * FROM admission_journal WHERE journal_id=?", (journal_id,)
                ).fetchone()
                conn.commit()
            except BaseException:
                conn.rollback()
                raise
        assert updated is not None
        return self._entry_from_row(updated)

    def _initialize(self) -> None:
        with self._connection() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            if self._has_legacy_global_idempotency_key(conn):
                self._upgrade_legacy_schema(conn)
            else:
                _create_journal_schema(conn)

    @staticmethod
    def _has_legacy_global_idempotency_key(conn: sqlite3.Connection) -> bool:
        table = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='admission_journal'"
        ).fetchone()
        if table is None:
            return False
        return _has_unique_index(conn, "admission_journal", ("idempotency_key",))

    @staticmethod
    def _upgrade_legacy_schema(conn: sqlite3.Connection) -> None:
        """Replace only the provisional global-key table without losing journals."""
        conn.execute("SAVEPOINT admission_journal_profile_key_upgrade")
        try:
            conn.execute("ALTER TABLE admission_journal RENAME TO admission_journal_legacy")
            conn.execute("DROP INDEX IF EXISTS idx_admission_replay")
            _create_journal_schema(conn)
            conn.execute(
                "INSERT INTO admission_journal("
                "journal_id, idempotency_key, request_hash, profile_id, command_json, state, "
                "canonical_operation_id, canonical_commit_sequence, error_code, receipt_json, "
                "created_at_ms, updated_at_ms"
                ") SELECT journal_id, idempotency_key, request_hash, profile_id, command_json, "
                "state, canonical_operation_id, canonical_commit_sequence, error_code, "
                "receipt_json, "
                "created_at_ms, updated_at_ms FROM admission_journal_legacy"
            )
            conn.execute("DROP TABLE admission_journal_legacy")
        except BaseException:
            conn.execute("ROLLBACK TO admission_journal_profile_key_upgrade")
            conn.execute("RELEASE admission_journal_profile_key_upgrade")
            raise
        conn.execute("RELEASE admission_journal_profile_key_upgrade")

    @contextmanager
    def _connection(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(str(self.path), timeout=1.0)
        try:
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA synchronous=FULL")
            conn.execute("PRAGMA foreign_keys=ON")
            conn.execute("PRAGMA busy_timeout=1000")
            yield conn
        finally:
            conn.close()

    @staticmethod
    def _entry_from_row(row: sqlite3.Row) -> AdmissionEntry:
        receipt_raw = row["receipt_json"]
        receipt = json.loads(receipt_raw) if receipt_raw else None
        return AdmissionEntry(
            journal_id=str(row["journal_id"]),
            idempotency_key=str(row["idempotency_key"]),
            request_hash=str(row["request_hash"]),
            profile_id=str(row["profile_id"]),
            state=str(row["state"]),
            canonical_operation_id=row["canonical_operation_id"],
            canonical_commit_sequence=row["canonical_commit_sequence"],
            error_code=row["error_code"],
            created_at_ms=int(row["created_at_ms"]),
            updated_at_ms=int(row["updated_at_ms"]),
            original_receipt=receipt,
        )


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
            "utf-8"
        )
    except (TypeError, ValueError) as exc:
        raise AdmissionPayloadError("remember command must be JSON serializable") from exc


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
