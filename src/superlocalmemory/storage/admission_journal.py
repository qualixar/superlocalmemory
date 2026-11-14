# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com

"""Separate journal for durable remember admission.

The journal is intentionally not a memory store.  Its only mutable state is a
small encrypted replay command and the canonical receipt associated with it.
Durable prepare and terminal transitions are FULL-synchronous; the advisory
dispatched marker may use NORMAL because both replay states are equivalent.
Canonical facts, FTS, graph, vectors, and model work stay outside this module.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import logging
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

from superlocalmemory.storage.journal_writer import (
    AdmissionJournalOverloaded,  # noqa: F401 - re-exported for callers
    AdmissionJournalUnavailable,
    GroupCommitWriter,
    ReadPool,
    is_sqlite_busy,
)

logger = logging.getLogger("superlocalmemory.storage.admission_journal")

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
    #: What this save retires once committed (``remember(..., replaces=)``).
    #: Journaled with the save so a deferred or replayed commit applies it.
    replaces: str = ""

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
        if not isinstance(self.replaces, str):
            raise AdmissionPayloadError("replaces must be a string id")
        if not isinstance(self.metadata, Mapping):
            raise AdmissionPayloadError("metadata must be an object")
        metadata = dict(self.metadata)
        _validate_json(metadata, "metadata")
        object.__setattr__(self, "metadata", metadata)
        object.__setattr__(self, "shared_with", tuple(self.shared_with))

    def canonical_payload(self) -> dict[str, Any]:
        payload = {
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
        if self.replaces:
            # Only when set: every request hash journaled before this field
            # existed is unchanged, so retries across an upgrade still match.
            payload["replaces"] = self.replaces
        return payload

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
            replaces=str(payload.get("replaces") or ""),
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
        # The daemon is the sole journal owner, but many HTTP/MCP request
        # threads prepare and transition entries concurrently. One writer
        # thread group-commits their tiny mutations (one FULL-synchronous
        # COMMIT per batch); reads use a bounded pool of persistent
        # connections. No save opens a connection of its own.
        self._writer = GroupCommitWriter(self.path)
        self._readers = ReadPool(self.path)

    def close(self) -> None:
        """Finish queued journal mutations and release every connection.

        Not final: the next operation reopens what it needs.
        """
        self._writer.close()
        self._readers.close()

    def prepare(
        self,
        request: RememberRequest,
        actor: Actor,
        *,
        deadline: float | None = None,
    ) -> PreparedAdmission:
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
        # Bound to the stable caller, not to this daemon start's capability
        # (storage/idempotency_identity.py): a retry after a restart is the
        # same request; a different caller's is not.
        request_hash = _request_hash(payload)
        now = _now_ms()
        journal_id = uuid.uuid4().hex

        # No separate read first: the key is checked inside the writer's own
        # transaction. A reader on this path starts read transactions while
        # the writer commits every few milliseconds, and SQLite then makes
        # it retry for its WAL snapshot with growing sleeps - seconds, under
        # a burst. A retry costs one read-only operation in the next batch.
        def insert(conn: sqlite3.Connection) -> AdmissionEntry:
            existing = conn.execute(
                "SELECT * FROM admission_journal "
                "WHERE profile_id=? AND idempotency_key=?",
                (request.profile_id, request.idempotency_key),
            ).fetchone()
            if existing is not None:
                entry = self._entry_from_row(existing)
                if entry.request_hash != request_hash and not self._stored_request_matches(
                    existing["command_json"], request_hash,
                ):
                    raise IdempotencyConflict(
                        "idempotency key belongs to a different immutable request"
                    )
                return entry
            conn.execute(
                "INSERT INTO admission_journal "
                "(journal_id, idempotency_key, request_hash, profile_id, "
                "command_json, state, created_at_ms, updated_at_ms) "
                "VALUES (?, ?, ?, ?, ?, 'prepared', ?, ?)",
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
                "SELECT * FROM admission_journal WHERE journal_id=?", (journal_id,),
            ).fetchone()
            return self._entry_from_row(row)

        # Durable when this returns; on AdmissionJournalUnavailable it is
        # guaranteed not to have been written (cancel before commit).
        return self._writer.submit(insert, deadline=deadline)

    def pending_entries(self, profile_id: str | None = None) -> list[AdmissionEntry]:
        """Every accepted entry not yet committed or rejected, oldest first."""
        sql = "SELECT * FROM admission_journal WHERE state IN ('prepared', 'dispatched')"
        params: tuple[str, ...] = ()
        if profile_id is not None:
            sql += " AND profile_id=?"
            params = (profile_id,)
        with self._read_connection() as conn:
            rows = conn.execute(sql + " ORDER BY created_at_ms, journal_id", params).fetchall()
        return [self._entry_from_row(row) for row in rows]

    def request_for(
        self,
        entry: AdmissionEntry,
        *,
        deadline: float | None = None,
    ) -> RememberRequest:
        """Decrypt the minimal replay body only immediately before canonical work."""
        with self._read_connection(deadline=deadline) as conn:
            row = conn.execute(
                "SELECT command_json FROM admission_journal WHERE journal_id=?", (entry.journal_id,)
            ).fetchone()
        if row is None:
            raise KeyError(entry.journal_id)
        request = RememberRequest.from_payload(self._decrypt_command(row["command_json"]))
        _remaining_seconds(deadline)
        return request

    def _decrypt_command(self, command_json: Any) -> dict[str, Any]:
        try:
            encoded = json.loads(str(command_json))["ciphertext_b64"]
            plaintext = self._codec.decrypt(base64.b64decode(encoded, validate=True))
            payload = json.loads(plaintext.decode("utf-8"))
            if not isinstance(payload, dict):
                raise TypeError("journal command is not an object")
            return payload
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

    def _stored_request_matches(self, command_json: Any, request_hash: str) -> bool:
        """Whether a row journaled before 4.1.22 holds this same request.

        Those rows hashed the per-start daemon actor, so their stored hash can
        never equal the stable one. Their command is re-hashed the stable way;
        a command that cannot be read back is never treated as a match.
        """
        try:
            return _request_hash(self._decrypt_command(command_json)) == request_hash
        except AdmissionPayloadError:
            return False

    def mark_dispatched(
        self,
        journal_id: str,
        *,
        deadline: float | None = None,
        known_prepared: bool = False,
    ) -> AdmissionEntry:
        # A concurrent retry may observe ``prepared`` and then lose the race to
        # another caller that commits the same idempotent command.  Treat that
        # terminal state as a successful no-op so the retry can return the
        # canonical receipt instead of surfacing a false transition failure.
        if not known_prepared:
            existing = self._get_entry(journal_id, deadline=deadline)
            if existing.state in {"dispatched", "committed"}:
                return existing
        return self._transition(
            journal_id,
            target="dispatched",
            allowed={"prepared", "dispatched", "committed"},
            deadline=deadline,
        )

    def mark_rejected(
        self,
        journal_id: str,
        error_code: str,
        *,
        deadline: float | None = None,
    ) -> AdmissionEntry:
        if not error_code or len(error_code) > 128:
            raise ValueError("error_code is required and bounded")
        return self._transition(
            journal_id,
            target="rejected",
            allowed={"prepared", "dispatched", "rejected"},
            error_code=error_code,
            deadline=deadline,
        )

    def reopen_unexplained_rejection(
        self,
        journal_id: str,
        *,
        deadline: float | None = None,
    ) -> AdmissionEntry | None:
        """Return a pre-4.2.18 refusal to ``prepared`` so it is judged once more.

        Issue #319: a refusal is cached under its idempotency key, and a
        folder retry re-sends the same keys (that is what keeps the parts
        already saved from being saved twice). A part refused by an older,
        stricter policy therefore stayed refused forever. Only an entry whose
        code is the bare ``UNEXPLAINED_REJECTION`` reopens, and a second
        refusal records its reason, so each entry is re-judged at most once.
        Returns the reopened entry, or ``None`` when nothing was reopened.
        """
        def update(conn: sqlite3.Connection) -> AdmissionEntry | None:
            cursor = conn.execute(
                "UPDATE admission_journal SET state='prepared', error_code=NULL, "
                "updated_at_ms=? WHERE journal_id=? AND state='rejected' AND error_code=?",
                (_now_ms(), journal_id, UNEXPLAINED_REJECTION),
            )
            if cursor.rowcount != 1:
                return None
            row = conn.execute(
                "SELECT * FROM admission_journal WHERE journal_id=?", (journal_id,)
            ).fetchone()
            return self._entry_from_row(row)

        return self._writer.submit(update, deadline=deadline)

    def mark_committed(
        self,
        journal_id: str,
        receipt: Mapping[str, Any],
        *,
        deadline: float | None = None,
    ) -> AdmissionEntry:
        receipt_json = _receipt_json(receipt)
        data = json.loads(receipt_json)
        operation_id = data.get("operation_id")
        commit_sequence = data.get("commit_sequence")
        if operation_id is not None and not isinstance(operation_id, str):
            raise ValueError("receipt operation_id must be a string")
        if commit_sequence is not None and not isinstance(commit_sequence, int):
            raise ValueError("receipt commit_sequence must be an integer")
        # Already committed is answered inside the writer's transaction
        # (see prepare for why there is no separate read on this path).
        return self._transition(
            journal_id,
            target="committed",
            allowed={"prepared", "dispatched", "rejected", "committed"},
            receipt_json=receipt_json,
            operation_id=operation_id,
            commit_sequence=commit_sequence,
            deadline=deadline,
        )

    def get(self, journal_id: str) -> AdmissionEntry:
        return self._get_entry(journal_id)

    def get_by_idempotency_key(
        self, profile_id: str, idempotency_key: str
    ) -> AdmissionEntry | None:
        """Return a profile-scoped retry record, never a cross-profile match."""
        return self._get_by_idempotency_key(profile_id, idempotency_key)

    def count(self) -> int:
        with self._read_connection() as conn:
            return int(conn.execute("SELECT COUNT(*) FROM admission_journal").fetchone()[0])

    def replay_pending(
        self,
        find_canonical_receipt: Callable[[AdmissionEntry], Mapping[str, Any] | None],
        dispatch: Callable[[AdmissionEntry, RememberRequest], Mapping[str, Any]],
        *,
        profile_id: str | None = None,
        after_commit: Callable[
            [AdmissionEntry, RememberRequest | None, Mapping[str, Any]], None
        ] | None = None,
    ) -> int:
        """Resolve crash-surviving entries without duplicate canonical writes.

        ``profile_id`` limits recovery to one profile. ``after_commit`` runs
        once the canonical write exists and BEFORE the journal records it, so
        follow-up work it does (a requested replacement) is retried by the
        next recovery if the process dies in between. It receives the decoded
        request when dispatch needed it, else ``None``.
        """
        recovered = 0
        for entry in self.pending_entries(profile_id):
            request: RememberRequest | None = None
            try:
                canonical = find_canonical_receipt(entry)
                if canonical is None:
                    request = self.request_for(entry)
                    canonical = dispatch(entry, request)
                if after_commit is not None:
                    after_commit(entry, request, canonical)
            except TerminalAdmissionError as exc:
                self.mark_rejected(entry.journal_id, exc.error_code)
                recovered += 1
                continue
            except AdmissionPayloadError:
                # One unreadable entry must never block the others (or the
                # daemon's start): set it aside, keep its bytes, carry on.
                self.quarantine(entry.journal_id)
                recovered += 1
                continue
            except Exception as exc:  # noqa: BLE001 - one entry, not the whole replay
                # Busy or failing writer: the entry stays pending and durable;
                # the caller hands what is left to its background committer.
                logger.warning(
                    "pending remember %s not recovered yet (%s); it stays queued",
                    entry.journal_id, type(exc).__name__,
                )
                continue
            self.mark_committed(entry.journal_id, canonical)
            recovered += 1
        return recovered

    def quarantine(
        self, journal_id: str, *, deadline: float | None = None,
    ) -> AdmissionEntry:
        """Set aside an entry whose command cannot be read; nothing is deleted."""
        logger.error(
            "a saved memory cannot be read back by this machine's key and was set "
            "aside (journal entry %s); its original bytes are kept", journal_id,
        )
        return self.mark_rejected(journal_id, UNREADABLE_COMMAND, deadline=deadline)

    def quarantined_count(self) -> int:
        """How many saves were set aside as unreadable."""
        with self._read_connection() as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM admission_journal WHERE state='rejected' "
                "AND error_code=?", (UNREADABLE_COMMAND,),
            ).fetchone()
        return int(row[0])

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
        deadline: float | None = None,
    ) -> AdmissionEntry:
        def update(conn: sqlite3.Connection) -> AdmissionEntry:
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
            return self._entry_from_row(updated)

        return self._writer.submit(update, deadline=deadline)

    def _get_entry(
        self,
        journal_id: str,
        *,
        deadline: float | None = None,
    ) -> AdmissionEntry:
        with self._read_connection(deadline=deadline) as conn:
            row = conn.execute(
                "SELECT * FROM admission_journal WHERE journal_id=?", (journal_id,)
            ).fetchone()
        if row is None:
            raise KeyError(journal_id)
        return self._entry_from_row(row)

    def _get_by_idempotency_key(
        self,
        profile_id: str,
        idempotency_key: str,
        *,
        deadline: float | None = None,
    ) -> AdmissionEntry | None:
        with self._read_connection(deadline=deadline) as conn:
            row = conn.execute(
                "SELECT * FROM admission_journal WHERE profile_id=? AND idempotency_key=?",
                (profile_id, idempotency_key),
            ).fetchone()
        return self._entry_from_row(row) if row is not None else None

    def _initialize(self) -> None:
        with self._connection() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            if self._has_legacy_global_idempotency_key(conn):
                self._upgrade_legacy_schema(conn)
            else:
                _create_journal_schema(conn)

    @contextmanager
    def _read_connection(
        self,
        *,
        deadline: float | None = None,
    ) -> Generator[sqlite3.Connection, None, None]:
        """Borrow a pooled journal reader without leaking SQLite lock errors."""
        with self._readers.connection(deadline=deadline) as conn:
            yield conn

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
    def _connection(
        self,
        *,
        timeout: float = 1.0,
    ) -> Generator[sqlite3.Connection, None, None]:
        bounded_timeout = max(0.001, timeout)
        busy_timeout_ms = max(1, int(bounded_timeout * 1_000))
        conn = sqlite3.connect(str(self.path), timeout=bounded_timeout)
        try:
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA synchronous=FULL")
            conn.execute("PRAGMA foreign_keys=ON")
            conn.execute(f"PRAGMA busy_timeout={busy_timeout_ms}")
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
