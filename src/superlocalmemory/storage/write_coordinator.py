# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com

"""Daemon-owned, cross-process coordinator for canonical SQLite writes.

This module implements the 3.8.6 first migration step: one daemon can claim
one ``memory.db`` path, and all in-process work submitted to that owner is
serialised by a single connection-owning thread.  It intentionally does not
create an IPC protocol.  CLI, MCP, and dashboard clients must reach this
coordinator through the authenticated daemon boundary as later migration work
removes their direct writes.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
import uuid
from collections import deque
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType, ModuleType
from typing import Any, Callable, Literal, Mapping

from superlocalmemory.core.file_lock import LockHeldError, exclusive_lock
from superlocalmemory.storage.write_lock import get_write_lock

try:
    import portalocker as _portalocker_import
except ImportError:  # pragma: no cover - pinned production dependency
    _portalocker: ModuleType | None = None
else:
    _portalocker = _portalocker_import


class WriteCoordinatorError(RuntimeError):
    """Base class for canonical writer failures."""


class OwnershipRequiredError(WriteCoordinatorError):
    """Raised when a process has not claimed canonical write ownership."""


class QueueOverloadedError(WriteCoordinatorError):
    """Raised when bounded coordinator capacity is exhausted."""


class WriteDeadlineExceededError(WriteCoordinatorError):
    """Raised before a queued write can start within its caller deadline."""


class CommandConflictError(WriteCoordinatorError):
    """A durable command id was reused for a different immutable request."""


class CommandRejectedError(WriteCoordinatorError):
    """A deterministic command cannot succeed if replayed unchanged."""

    def __init__(self, error_code: str = "COMMAND_REJECTED") -> None:
        super().__init__("canonical command was deterministically rejected")
        self.error_code = error_code


class Lane(StrEnum):
    """Scheduling lanes, ordered to protect foreground memory operations."""

    FOREGROUND = "foreground"
    CONTROL = "control"
    BACKGROUND = "background"


class CommandKind(StrEnum):
    """Durable command families accepted by the canonical writer.

    A command is intentionally more specific than an SQL statement.  The
    coordinator can therefore make the receipt part of the same SQLite commit
    and safely replay an acknowledged request without calling its handler.
    """

    ADMISSION = "admission"
    DELETE_FACT = "delete_fact"
    UPDATE_FACT = "update_fact"
    ARCHIVE_FACT = "archive_fact"
    MERGE_FACT = "merge_fact"
    SET_FACT_SCOPE = "set_fact_scope"


JsonValue = None | bool | int | float | str | tuple["JsonValue", ...] | Mapping[str, "JsonValue"]


def _freeze_json(value: Any) -> JsonValue:
    """Make JSON-shaped command/receipt data immutable before it crosses lanes."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze_json(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item) for item in value)
    raise TypeError("command payloads and receipts must be JSON-compatible")


def _thaw_json(value: JsonValue) -> Any:
    """Return ordinary JSON-compatible values for stable receipt encoding."""
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class WriteCommand:
    """An immutable request that can be committed and replayed exactly once."""

    command_id: str
    kind: CommandKind
    payload: Mapping[str, JsonValue] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        if not self.command_id or not isinstance(self.command_id, str):
            raise ValueError("command_id must be a non-empty string")
        object.__setattr__(self, "kind", CommandKind(self.kind))
        frozen = _freeze_json(dict(self.payload))
        if not isinstance(frozen, Mapping):  # pragma: no cover - dict input is enforced above
            raise TypeError("command payload must be an object")
        object.__setattr__(self, "payload", frozen)

    @classmethod
    def create(
        cls,
        kind: CommandKind,
        payload: Mapping[str, Any] | None = None,
        *,
        command_id: str | None = None,
    ) -> "WriteCommand":
        """Create a command with a caller-supplied or generated idempotency key."""
        resolved_id = str(uuid.uuid4()) if command_id is None else command_id
        return cls(resolved_id, CommandKind(kind), payload or {})


@dataclass(frozen=True, slots=True)
class WriteResult:
    """An immutable receipt produced by a typed command handler."""

    command_id: str
    kind: CommandKind
    receipt: Mapping[str, JsonValue] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        if not self.command_id or not isinstance(self.command_id, str):
            raise ValueError("command_id must be a non-empty string")
        object.__setattr__(self, "kind", CommandKind(self.kind))
        frozen = _freeze_json(dict(self.receipt))
        if not isinstance(frozen, Mapping):  # pragma: no cover - dict input is enforced above
            raise TypeError("command receipt must be an object")
        object.__setattr__(self, "receipt", frozen)

    @classmethod
    def from_receipt(
        cls,
        command: WriteCommand,
        receipt: Mapping[str, Any] | None = None,
    ) -> "WriteResult":
        """Bind a handler receipt to the command being processed."""
        return cls(command.command_id, command.kind, receipt or {})


@dataclass(frozen=True, slots=True)
class WriteCapability:
    """A worker-thread-only capability for DatabaseManager binding.

    This is an in-process authority boundary, not a security credential.  Its
    coordinator identity token, exact resolved database path, and worker
    identity keep ordinary manager code from attaching a foreign connection.
    """

    db_path: Path
    owner_id: str
    worker_ident: int
    _issuer: Any = field(repr=False, compare=False)
    _token: object = field(repr=False, compare=False)

    def _validate(self, db_path: Path) -> None:
        issuer = self._issuer
        if getattr(issuer, "_capability_token", None) is not self._token:
            raise WriteCoordinatorError("untrusted coordinator capability")
        if getattr(issuer, "owner_id", None) != self.owner_id:
            raise WriteCoordinatorError("coordinator capability identity mismatch")
        if getattr(issuer, "db_path", None) != self.db_path:
            raise WriteCoordinatorError("coordinator capability database identity mismatch")
        if getattr(issuer, "_worker_ident", None) != self.worker_ident:
            raise WriteCoordinatorError("coordinator capability worker identity mismatch")
        if threading.get_ident() != self.worker_ident:
            raise WriteCoordinatorError("coordinator capability used outside its worker thread")
        if db_path.expanduser().resolve() != self.db_path:
            raise WriteCoordinatorError("coordinator capability targets a different database")


CommandHandler = Callable[[sqlite3.Connection, WriteCapability, WriteCommand], WriteResult]


_Priority = Literal["foreground", "control", "background"]
_MAX_QUEUE_DEPTH = 4_096
_FOREGROUND_BURST = 8
_SQLITE_BUSY_CODES = {sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED}


@dataclass(slots=True)
class _Execution:
    sql: str | None
    parameters: tuple[Any, ...]
    lane: Lane
    deadline: float
    command: WriteCommand | None = None
    completion: threading.Event = field(default_factory=threading.Event)
    rows: list[sqlite3.Row] | None = None
    result: WriteResult | None = None
    error: BaseException | None = None
    cancelled: bool = False


class WriteCoordinator:
    """Own one writable ``memory.db`` connection for a daemon lifetime.

    ``claim_ownership`` is deliberately separate from construction so a caller
    can report an already-running daemon without attempting an SQLite open.
    ``execute`` exists only as the migration adapter for small bounded storage
    commands.  Product code will use typed commands as writer families move to
    this coordinator.
    """

    def __init__(
        self,
        db_path: str | Path,
        *,
        owner_id: str | None = None,
        max_queue_depth: int = _MAX_QUEUE_DEPTH,
    ) -> None:
        if max_queue_depth < 1:
            raise ValueError("max_queue_depth must be at least one")
        self._db_path = Path(db_path).expanduser().resolve()
        self._owner_id = owner_id or str(uuid.uuid4())
        self._lock_path = self._db_path.with_name(f"{self._db_path.name}.writer.lock")
        # Expand-migrate-contract compatibility: legacy in-process writers
        # already serialize on this per-path RLock. The coordinator must join
        # that same critical section until every background/control writer has
        # moved behind typed commands; otherwise the new admission connection
        # can race the materializer connection and reintroduce SQLITE_BUSY.
        self._process_write_lock = get_write_lock(self._db_path)
        self._max_queue_depth = max_queue_depth
        self._ownership_context: AbstractContextManager[int] | None = None
        self._lock_fd: int | None = None
        self._condition = threading.Condition()
        self._queues: dict[Lane, deque[_Execution]] = {
            Lane.FOREGROUND: deque(),
            Lane.CONTROL: deque(),
            Lane.BACKGROUND: deque(),
        }
        self._queued_count = 0
        self._foreground_served = 0
        self._stopping = False
        self._worker: threading.Thread | None = None
        self._lease_release_pending = False
        self._lease_release_reaper: threading.Thread | None = None
        self._worker_ready = threading.Event()
        self._worker_error: BaseException | None = None
        self._worker_ident: int | None = None
        self._capability_token = object()
        self._handlers: dict[CommandKind, CommandHandler] = {}

    @property
    def db_path(self) -> Path:
        """Resolved canonical database path owned by this coordinator."""
        return self._db_path

    @property
    def owner_id(self) -> str:
        """Opaque daemon instance identifier recorded in the ownership lease."""
        return self._owner_id

    def claim_ownership(self) -> bool:
        """Claim the cross-platform owner lease without waiting on another daemon."""
        if _portalocker is None:
            raise WriteCoordinatorError("portalocker is required for canonical writer ownership")
        if self._ownership_context is not None:
            if self._lease_release_pending:
                raise WriteCoordinatorError("canonical writer is still shutting down")
            return True
        self._lock_path.parent.mkdir(parents=True, exist_ok=True)
        context = exclusive_lock(self._lock_path, timeout_s=0.0)
        try:
            fd = context.__enter__()
        except LockHeldError:
            return False
        except OSError as exc:
            raise WriteCoordinatorError(
                f"could not claim canonical writer lock for {self._db_path}"
            ) from exc

        self._ownership_context = context
        self._lock_fd = fd
        try:
            self._write_owner_metadata(fd)
        except BaseException:
            self.release_ownership()
            raise
        return True

    def release_ownership(self) -> None:
        """Stop the worker and release the ownership lease, if held."""
        try:
            self.stop()
        except WriteCoordinatorError:
            self._release_lease_after_worker_exit()
            raise
        self._release_lease_if_worker_stopped()

    def start(self) -> None:
        """Start the sole connection-owning worker after a successful claim."""
        self._require_ownership()
        with self._condition:
            worker = self._worker
            if worker is None:
                if self._lease_release_pending:
                    raise WriteCoordinatorError("canonical writer is still shutting down")
                self._stopping = False
                self._worker_error = None
                self._worker_ready.clear()
                worker = threading.Thread(
                    target=self._run,
                    name=f"slm-write-coordinator-{self._owner_id[:8]}",
                    daemon=True,
                )
                self._worker = worker
                worker.start()
            else:
                if self._stopping:
                    raise WriteCoordinatorError("canonical writer is stopping")
        if not self._worker_ready.wait(timeout=2.0):
            self.stop()
            raise WriteCoordinatorError("canonical writer did not start within two seconds")
        if self._worker_error is not None:
            error = self._worker_error
            self.stop()
            raise WriteCoordinatorError("canonical writer could not open memory.db") from error

    def stop(self, deadline_s: float = 2.0) -> None:
        """Stop accepting work and wait briefly for the sole writer thread."""
        worker = self._worker
        if worker is None:
            return
        with self._condition:
            self._stopping = True
            self._condition.notify_all()
        worker.join(timeout=max(0.0, deadline_s))
        if worker.is_alive():
            raise WriteCoordinatorError("canonical writer did not stop before its deadline")
        self._worker = None

    def _release_lease_after_worker_exit(self) -> None:
        """Release a requested lease only after its live worker has terminated."""
        with self._condition:
            worker = self._worker
            if worker is None or not worker.is_alive() or self._lease_release_pending:
                return
            self._lease_release_pending = True
            reaper = threading.Thread(
                target=self._reap_worker_then_release_lease,
                args=(worker,),
                name=f"slm-write-lease-reaper-{self._owner_id[:8]}",
                daemon=True,
            )
            self._lease_release_reaper = reaper
        reaper.start()

    def _reap_worker_then_release_lease(self, worker: threading.Thread) -> None:
        """Wait without a shutdown deadline, then relinquish a drained lease."""
        worker.join()
        self._release_lease_if_worker_stopped(expected_worker=worker)

    def _release_lease_if_worker_stopped(
        self,
        *,
        expected_worker: threading.Thread | None = None,
    ) -> None:
        """Release ownership only when no writer thread can still use the DB."""
        with self._condition:
            worker = self._worker
            if worker is not None:
                if expected_worker is not None and worker is not expected_worker:
                    return
                if worker.is_alive():
                    return
                self._worker = None
            context = self._ownership_context
            self._ownership_context = None
            self._lock_fd = None
            self._lease_release_pending = False
            self._lease_release_reaper = None
        if context is not None:
            context.__exit__(None, None, None)

    def execute(
        self,
        sql: str,
        parameters: tuple[Any, ...] = (),
        *,
        priority: _Priority | Lane = Lane.FOREGROUND,
        timeout: float = 1.0,
    ) -> list[sqlite3.Row]:
        """Execute one bounded statement through the daemon-owned connection.

        The migration adapter rejects empty SQL and expired requests.  It is
        intentionally not an escape hatch for slow work or unbounded batches.
        """
        if not isinstance(sql, str) or not sql.strip():
            raise ValueError("sql must be a non-empty statement")
        if timeout <= 0:
            raise ValueError("timeout must be greater than zero")
        lane = self._coerce_lane(priority)
        self.start()
        item = _Execution(
            sql=sql,
            parameters=tuple(parameters),
            lane=lane,
            deadline=time.monotonic() + timeout,
        )
        self._enqueue(item)
        remaining = max(0.0, item.deadline - time.monotonic())
        if not item.completion.wait(remaining):
            with self._condition:
                item.cancelled = True
            raise WriteDeadlineExceededError("canonical write exceeded its caller deadline")
        if item.error is not None:
            raise item.error
        return item.rows or []

    def register_handler(self, kind: CommandKind, handler: CommandHandler) -> None:
        """Register the sole handler for a durable command family.

        Handler registration is a daemon-start concern.  Refusing changes
        after the worker starts avoids a request observing a partially changed
        command dispatch table.
        """
        if not callable(handler):
            raise TypeError("command handler must be callable")
        command_kind = CommandKind(kind)
        with self._condition:
            if self._worker is not None:
                raise WriteCoordinatorError("command handlers must be registered before start")
            if command_kind in self._handlers:
                raise WriteCoordinatorError(f"handler already registered for {command_kind.value}")
            self._handlers[command_kind] = handler

    def submit(
        self,
        command: WriteCommand,
        *,
        priority: _Priority | Lane = Lane.FOREGROUND,
        timeout: float = 1.0,
    ) -> WriteResult:
        """Run a typed command and persist its receipt in the same commit."""
        if not isinstance(command, WriteCommand):
            raise TypeError("command must be a WriteCommand")
        if timeout <= 0:
            raise ValueError("timeout must be greater than zero")
        lane = self._coerce_lane(priority)
        self.start()
        item = _Execution(
            sql=None,
            parameters=(),
            lane=lane,
            deadline=time.monotonic() + timeout,
            command=command,
        )
        self._enqueue(item)
        remaining = max(0.0, item.deadline - time.monotonic())
        if not item.completion.wait(remaining):
            with self._condition:
                item.cancelled = True
            raise WriteDeadlineExceededError("canonical write exceeded its caller deadline")
        if item.error is not None:
            raise item.error
        if item.result is None:  # pragma: no cover - defensive worker invariant
            raise WriteCoordinatorError("canonical command completed without a receipt")
        return item.result

    def _enqueue(self, item: _Execution) -> None:
        with self._condition:
            if self._stopping:
                raise WriteCoordinatorError("canonical writer is stopping")
            if self._queued_count >= self._max_queue_depth:
                raise QueueOverloadedError("canonical writer queue is full")
            self._queues[item.lane].append(item)
            self._queued_count += 1
            self._condition.notify()

    def _run(self) -> None:
        self._worker_ident = threading.get_ident()
        try:
            conn = self._open_connection()
        except BaseException as exc:
            self._worker_error = exc
            self._worker_ready.set()
            return
        self._worker_ready.set()
        try:
            while True:
                item = self._next_item()
                if item is None:
                    return
                self._execute_item(conn, item)
        finally:
            self._worker_ident = None
            conn.close()

    def _open_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path), timeout=1.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA busy_timeout=1000")
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _next_item(self) -> _Execution | None:
        with self._condition:
            while self._queued_count == 0 and not self._stopping:
                self._condition.wait()
            if self._queued_count == 0:
                return None

            lane = self._select_lane()
            item = self._queues[lane].popleft()
            self._queued_count -= 1
            return item

    def _select_lane(self) -> Lane:
        foreground = self._queues[Lane.FOREGROUND]
        control = self._queues[Lane.CONTROL]
        background = self._queues[Lane.BACKGROUND]
        can_continue_foreground = self._foreground_served < _FOREGROUND_BURST
        if foreground and (can_continue_foreground or not (control or background)):
            self._foreground_served += 1
            return Lane.FOREGROUND
        if control:
            self._foreground_served = 0
            return Lane.CONTROL
        if background:
            self._foreground_served = 0
            return Lane.BACKGROUND
        self._foreground_served = 0
        return Lane.FOREGROUND

    def _execute_item(self, conn: sqlite3.Connection, item: _Execution) -> None:
        if item.cancelled or time.monotonic() >= item.deadline:
            item.error = WriteDeadlineExceededError("canonical write expired before execution")
            item.completion.set()
            return
        try:
            with self._process_write_lock:
                synchronous = "FULL" if item.lane is Lane.FOREGROUND else "NORMAL"
                conn.execute(f"PRAGMA synchronous={synchronous}")
                conn.execute("BEGIN IMMEDIATE")
                if item.command is not None:
                    item.result = self._execute_command(conn, item.command)
                else:
                    if item.sql is None:  # pragma: no cover - execution invariant
                        raise WriteCoordinatorError("missing coordinator SQL command")
                    cursor = conn.execute(item.sql, item.parameters)
                    item.rows = cursor.fetchall()
                conn.commit()
        except WriteCoordinatorError as exc:
            conn.rollback()
            item.error = exc
        except sqlite3.Error as exc:
            conn.rollback()
            if self._is_busy(exc):
                item.error = QueueOverloadedError("canonical writer is temporarily busy")
            else:
                item.error = WriteCoordinatorError("canonical write command was rejected")
                item.error.__cause__ = exc
        except BaseException as exc:
            conn.rollback()
            item.error = WriteCoordinatorError("canonical write command failed")
            item.error.__cause__ = exc
        finally:
            item.completion.set()

    def _execute_command(self, conn: sqlite3.Connection, command: WriteCommand) -> WriteResult:
        """Dispatch one command and atomically append its immutable receipt."""
        payload = _thaw_json(command.payload)
        if not isinstance(payload, dict):
            raise WriteCoordinatorError("command payload must be an object")
        request_hash = _required_text(payload, "request_hash")
        profile_id = _required_text(payload, "profile_id")
        idempotency_key = _required_text(payload, "idempotency_key")
        existing = conn.execute(
            "SELECT command_kind, request_hash, profile_id, idempotency_key, "
            "receipt_json FROM write_commits WHERE command_id = ?",
            (command.command_id,),
        ).fetchone()
        if existing is not None:
            if existing["command_kind"] != command.kind.value:
                raise CommandConflictError("command id was already committed with a different kind")
            if (
                existing["request_hash"] != request_hash
                or existing["profile_id"] != profile_id
                or existing["idempotency_key"] != idempotency_key
            ):
                raise CommandConflictError(
                    "command id was already committed for a different request"
                )
            try:
                receipt = json.loads(existing["receipt_json"])
            except (TypeError, json.JSONDecodeError) as exc:
                raise WriteCoordinatorError("stored command receipt is invalid") from exc
            if not isinstance(receipt, dict):
                raise WriteCoordinatorError("stored command receipt is not an object")
            return WriteResult(command.command_id, command.kind, receipt)

        handler = self._handlers.get(command.kind)
        if handler is None:
            raise WriteCoordinatorError(f"no handler registered for {command.kind.value}")
        worker_ident = self._worker_ident
        if worker_ident is None:  # pragma: no cover - worker-only call
            raise WriteCoordinatorError("typed command dispatch requires the writer worker")
        capability = WriteCapability(
            self._db_path,
            self._owner_id,
            worker_ident,
            self,
            self._capability_token,
        )
        result = handler(conn, capability, command)
        if not isinstance(result, WriteResult):
            raise WriteCoordinatorError("command handler must return WriteResult")
        if result.command_id != command.command_id or result.kind is not command.kind:
            raise WriteCoordinatorError(
                "command handler returned a receipt for a different command"
            )
        receipt = _thaw_json(result.receipt)
        if not isinstance(payload, dict) or not isinstance(receipt, dict):
            raise WriteCoordinatorError("admission payload and receipt must be objects")
        _reject_receipt_memory_content(receipt)
        journal_id = _required_text(payload, "journal_id")
        operation_id = _required_text(receipt, "operation_id")
        next_sequence = int(
            conn.execute(
                "SELECT COALESCE(MAX(commit_sequence), 0) + 1 FROM write_commits"
            ).fetchone()[0]
        )
        receipt.setdefault("commit_sequence", next_sequence)
        committed_result = WriteResult(command.command_id, command.kind, receipt)
        receipt_json = json.dumps(
            receipt,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        conn.execute(
            "INSERT INTO write_commits("
            "commit_sequence, command_id, journal_id, command_kind, request_hash, "
            "profile_id, idempotency_key, operation_id, receipt_json, committed_at"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                next_sequence,
                command.command_id,
                journal_id,
                command.kind.value,
                request_hash,
                profile_id,
                idempotency_key,
                operation_id,
                receipt_json,
                time.time(),
            ),
        )
        return committed_result

    def _write_owner_metadata(self, fd: int) -> None:
        metadata = {
            "owner_id": self._owner_id,
            "pid": os.getpid(),
            "database": str(self._db_path),
            "claimed_at_ms": int(time.time() * 1000),
        }
        payload = json.dumps(metadata, sort_keys=True).encode("utf-8") + b"\n"
        os.ftruncate(fd, 0)
        os.lseek(fd, 0, os.SEEK_SET)
        os.write(fd, payload)
        os.fsync(fd)

    def _require_ownership(self) -> None:
        if self._ownership_context is None:
            raise OwnershipRequiredError("canonical writer ownership has not been claimed")

    @staticmethod
    def _coerce_lane(priority: _Priority | Lane) -> Lane:
        try:
            return Lane(priority)
        except ValueError as exc:
            raise ValueError(f"unknown coordinator priority: {priority}") from exc

    @staticmethod
    def _is_busy(error: sqlite3.Error) -> bool:
        code = getattr(error, "sqlite_errorcode", None)
        return code in _SQLITE_BUSY_CODES


def _required_text(value: Mapping[str, Any], key: str) -> str:
    candidate = value.get(key)
    if not isinstance(candidate, str) or not candidate:
        raise WriteCoordinatorError(f"admission command is missing {key}")
    return candidate


def _reject_receipt_memory_content(value: Any) -> None:
    """Keep the immutable command ledger free of deleted or edited memory text."""
    if isinstance(value, Mapping):
        for key, child in value.items():
            if key.casefold() in {
                "content",
                "content_preview",
                "raw_content",
                "memory_content",
                "source_content",
            }:
                raise WriteCoordinatorError(
                    "immutable command receipts must contain metadata only"
                )
            _reject_receipt_memory_content(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _reject_receipt_memory_content(child)


__all__ = [
    "CommandConflictError",
    "CommandKind",
    "CommandRejectedError",
    "Lane",
    "OwnershipRequiredError",
    "QueueOverloadedError",
    "WriteCapability",
    "WriteCommand",
    "WriteCoordinator",
    "WriteCoordinatorError",
    "WriteDeadlineExceededError",
    "WriteResult",
]
