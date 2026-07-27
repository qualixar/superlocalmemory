# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Daemon-owned bridge from journaled remember requests to one SQLite writer.

The public HTTP boundary authenticates and runs trust policy before it calls
this module.  This runtime then uses a separate FULL-synchronous journal to
make the request replayable, and submits a typed command to the sole daemon
writer.  The command transaction creates only the M018 operation plus its
immediately-queryable projection; model and enrichment work remain with the
existing background materializer.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
import uuid
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from superlocalmemory.core.ingestion_command import (
    IngestionCommand,
    IngestionOperation,
    IngestionOperationRepository,
    IngestionRejectedError,
    IngestionRequest,
    MaterializationResult,
)
from superlocalmemory.core.remember_admission import (
    RememberAdmissionCommand,
    RememberReceipt,
    RememberService,
)
from superlocalmemory.storage.admission_codec import MachineKeyCommandCodec
from superlocalmemory.storage.admission_journal import (
    Actor,
    AdmissionEntry,
    AdmissionJournal,
    AdmissionPayloadError,
    RememberRequest,
    TerminalAdmissionError,
)
from superlocalmemory.storage.database import DatabaseManager
from superlocalmemory.storage.write_coordinator import (
    CommandConflictError,
    CommandKind,
    CommandRejectedError,
    OwnershipRequiredError,
    WriteCommand,
    WriteCoordinator,
    WriteCoordinatorError,
    WriteResult,
)

QueryableWriter = Callable[[IngestionRequest, str], list[str]]
Materializer = Callable[
    [IngestionOperation],
    list[str] | tuple[str, ...] | MaterializationResult,
]


class CanonicalRememberUnavailable(RuntimeError):
    """The daemon cannot accept a bounded canonical remember request."""


class CanonicalMutationConflict(ValueError):
    """A mutation retry key was reused for different immutable input."""


_MUTATION_IDEMPOTENCY_KEY = re.compile(r"^[A-Za-z0-9._:-]{1,256}$")


def validate_deterministic_admission(
    content: str,
    *,
    max_verbatim_chars: int = 24_000,
    max_ingest_bytes: int = 1_048_576,
) -> None:
    """Reject deterministic non-evidence before it can enter the journal.

    The immediate writer checks these gates as a defence in depth measure. At
    the daemon boundary they must run first: a rejected payload has no
    queryable receipt, so recording it as dispatched would create replay work
    that can never commit.
    """
    from superlocalmemory.core.engine_ingestion import content_passes_admission
    from superlocalmemory.core.ingest_gate import apply_ingest_gate

    if not content_passes_admission(content):
        raise AdmissionPayloadError("content rejected by deterministic admission policy")
    if apply_ingest_gate(
        content,
        max_verbatim_chars=max_verbatim_chars,
        max_ingest_bytes=max_ingest_bytes,
    ).rejected:
        raise AdmissionPayloadError("content rejected by deterministic ingest policy")


class _CoordinatorAdapter:
    """Translate the journal service's narrow protocol into typed commands."""

    def __init__(self, coordinator: WriteCoordinator) -> None:
        self._coordinator = coordinator

    def submit(
        self, command: RememberAdmissionCommand, *, wait_ms: int,
    ) -> Mapping[str, Any]:
        payload = {
            "journal_id": command.journal_id,
            "request_hash": command.request_hash,
            "profile_id": command.profile_id,
            "idempotency_key": command.idempotency_key,
            "request": command.request.canonical_payload(),
        }
        try:
            result = self._coordinator.submit(
                WriteCommand.create(
                    CommandKind.ADMISSION,
                    payload,
                    command_id=command.journal_id,
                ),
                timeout=max(0.001, wait_ms / 1000),
            )
        except CommandRejectedError as exc:
            raise TerminalAdmissionError(exc.error_code) from exc
        return {"state": "committed", "receipt": dict(result.receipt)}


class CanonicalRememberRuntime:
    """One daemon lifetime of journal-first, coordinator-owned admission."""

    def __init__(
        self,
        *,
        db: DatabaseManager,
        profile_id: str,
        writer: QueryableWriter,
        journal_path: str | Path,
        materialize: Materializer | None = None,
        owner_id: str | None = None,
    ) -> None:
        if not profile_id:
            raise ValueError("profile_id is required")
        self._db = db
        self._profile_id = profile_id
        self._writer = writer
        self._materialize = materialize or _materialization_is_not_available
        self._binding_lock = threading.RLock()
        self.coordinator = WriteCoordinator(db.db_path, owner_id=owner_id)
        self.journal = AdmissionJournal(
            journal_path,
            codec=MachineKeyCommandCodec(Path(journal_path).with_name("admission-key.bin")),
        )
        self._service = RememberService(self.journal, _CoordinatorAdapter(self.coordinator))
        self._started = False

    @classmethod
    def for_engine(cls, engine: Any) -> "CanonicalRememberRuntime":
        """Create the daemon boundary from an initialized engine only."""
        from superlocalmemory.core.engine_ingestion import build_immediate_admission_handler
        from superlocalmemory.infra.data_root import state_path

        db = engine._db
        store_config = getattr(engine._config, "store", None)
        return cls(
            db=db,
            profile_id=engine._profile_id,
            writer=build_immediate_admission_handler(
                db,
                profile_id=engine._profile_id,
                max_verbatim_chars=getattr(store_config, "max_verbatim_chars", 24_000),
                max_ingest_bytes=getattr(store_config, "max_ingest_bytes", 1_048_576),
            ),
            journal_path=state_path("admission_journal.db"),
        )

    def start(self) -> None:
        """Claim writer ownership, install the handler, then recover journal work."""
        if self._started:
            return
        if not self.coordinator.claim_ownership():
            raise CanonicalRememberUnavailable(
                "another daemon owns the canonical memory writer"
            )
        try:
            self.coordinator.register_handler(CommandKind.ADMISSION, self._handle_admission)
            self.coordinator.register_handler(CommandKind.DELETE_FACT, self._handle_mutation)
            self.coordinator.register_handler(CommandKind.UPDATE_FACT, self._handle_mutation)
            self.coordinator.register_handler(CommandKind.ARCHIVE_FACT, self._handle_mutation)
            self.coordinator.register_handler(CommandKind.MERGE_FACT, self._handle_mutation)
            self.coordinator.register_handler(CommandKind.SET_FACT_SCOPE, self._handle_mutation)
            self.coordinator.start()
            self.replay_pending()
        except BaseException:
            self.coordinator.release_ownership()
            raise
        self._started = True

    @property
    def ready(self) -> bool:
        worker = self.coordinator._worker
        return bool(
            self._started
            and self.coordinator._ownership_context is not None
            and worker is not None
            and worker.is_alive()
        )

    def stop(self) -> None:
        """Release the daemon writer lease after callers and workers have stopped."""
        self._started = False
        self.coordinator.release_ownership()

    def rebind_engine(self, engine: Any) -> None:
        """Atomically follow a drained daemon mode/profile transition."""
        from superlocalmemory.core.engine_ingestion import build_immediate_admission_handler

        db = engine._db
        if db.db_path.expanduser().resolve() != self.coordinator.db_path:
            raise CanonicalRememberUnavailable(
                "reconfigured engine targets a different canonical database"
            )
        profile_id = str(engine._profile_id)
        if not profile_id:
            raise CanonicalRememberUnavailable("reconfigured engine has no profile")
        store_config = getattr(getattr(engine, "_config", None), "store", None)
        writer = build_immediate_admission_handler(
            db,
            profile_id=profile_id,
            max_verbatim_chars=getattr(store_config, "max_verbatim_chars", 24_000),
            max_ingest_bytes=getattr(store_config, "max_ingest_bytes", 1_048_576),
        )
        with self._binding_lock:
            previous = (self._db, self._profile_id, self._writer)
            self._db = db
            self._profile_id = profile_id
            self._writer = writer
        try:
            self.replay_pending()
        except BaseException:
            with self._binding_lock:
                self._db, self._profile_id, self._writer = previous
            raise

    def remember(
        self, request: RememberRequest, actor: Actor, *, deadline_ms: int = 2_000,
    ) -> RememberReceipt:
        """Journal then commit one bounded queryable admission receipt."""
        if not self._started:
            raise CanonicalRememberUnavailable("canonical remember writer is not ready")
        if deadline_ms < 1 or deadline_ms > 2_000:
            raise ValueError("deadline_ms must be between 1 and 2000")
        try:
            return self._service.remember(request, actor, deadline_ms=deadline_ms)
        except (OwnershipRequiredError, WriteCoordinatorError) as exc:
            raise CanonicalRememberUnavailable(
                "canonical remember is temporarily unavailable"
            ) from exc

    def replay_pending(self) -> int:
        """Finish prepared/dispatched journal entries before publishing readiness."""
        if self.coordinator._ownership_context is None:
            raise CanonicalRememberUnavailable("canonical writer ownership is required")

        def find(entry: AdmissionEntry) -> Mapping[str, Any] | None:
            rows = self.coordinator.execute(
                "SELECT request_hash, receipt_json FROM write_commits "
                "WHERE profile_id=? AND idempotency_key=?",
                (entry.profile_id, entry.idempotency_key),
                timeout=1.0,
            )
            if not rows:
                return None
            if rows[0]["request_hash"] != entry.request_hash:
                raise TerminalAdmissionError("IDEMPOTENCY_CONFLICT")
            receipt = json.loads(rows[0]["receipt_json"])
            return receipt if isinstance(receipt, dict) else None

        def dispatch(entry, request: RememberRequest) -> Mapping[str, Any]:
            command = RememberAdmissionCommand(
                journal_id=entry.journal_id,
                request_hash=entry.request_hash,
                request=request,
                profile_id=entry.profile_id,
                idempotency_key=entry.idempotency_key,
            )
            return _CoordinatorAdapter(self.coordinator).submit(command, wait_ms=2_000)["receipt"]

        try:
            return self.journal.replay_pending(
                find,
                dispatch,
                profile_id=self._profile_id,
            )
        except (WriteCoordinatorError, ValueError, json.JSONDecodeError) as exc:
            raise CanonicalRememberUnavailable("pending remember recovery failed") from exc

    def delete_fact(
        self, profile_id: str, fact_id: str, *, idempotency_key: str | None = None,
    ) -> Mapping[str, Any]:
        """Hard-delete one profile-owned fact through the sole writer."""
        return self._submit_mutation(
            CommandKind.DELETE_FACT,
            profile_id,
            {"fact_id": fact_id},
            idempotency_key=idempotency_key,
        )

    def update_fact(
        self,
        profile_id: str,
        fact_id: str,
        updates: Mapping[str, Any],
        *,
        idempotency_key: str | None = None,
    ) -> Mapping[str, Any]:
        """Apply deterministic fact fields after policy/model work completes."""
        return self._submit_mutation(
            CommandKind.UPDATE_FACT,
            profile_id,
            {"fact_id": fact_id, "updates": _json_roundtrip(updates)},
            idempotency_key=idempotency_key,
        )

    def archive_fact(
        self, profile_id: str, fact_id: str, *, idempotency_key: str | None = None,
    ) -> Mapping[str, Any]:
        """Archive a fact and its restore payload in one bounded transaction."""
        return self._submit_mutation(
            CommandKind.ARCHIVE_FACT,
            profile_id,
            {"fact_id": fact_id},
            idempotency_key=idempotency_key,
        )

    def merge_fact(
        self,
        profile_id: str,
        fact_id: str,
        kept_fact_id: str,
        *,
        idempotency_key: str | None = None,
    ) -> Mapping[str, Any]:
        """Record and apply a same-profile merge through the sole writer."""
        return self._submit_mutation(
            CommandKind.MERGE_FACT,
            profile_id,
            {
                "fact_id": fact_id,
                "kept_fact_id": kept_fact_id,
            },
            idempotency_key=idempotency_key,
        )

    def set_fact_scope(
        self,
        profile_id: str,
        fact_id: str,
        scope: str,
        shared_with: list[str],
        *,
        idempotency_key: str | None = None,
    ) -> Mapping[str, Any]:
        """Set a validated scope without bypassing profile isolation."""
        return self._submit_mutation(
            CommandKind.SET_FACT_SCOPE,
            profile_id,
            {"fact_id": fact_id, "scope": scope, "shared_with": shared_with},
            idempotency_key=idempotency_key,
        )

    def _submit_mutation(
        self,
        kind: CommandKind,
        profile_id: str,
        payload: Mapping[str, Any],
        *,
        idempotency_key: str | None,
    ) -> Mapping[str, Any]:
        if not self._started:
            raise CanonicalRememberUnavailable("canonical mutation writer is not ready")
        if not profile_id:
            raise ValueError("profile_id is required")
        key = idempotency_key or str(uuid.uuid4())
        if not _MUTATION_IDEMPOTENCY_KEY.fullmatch(key):
            raise ValueError("idempotency key must be 1-256 safe characters")
        canonical = _json_roundtrip(payload)
        request_hash = hashlib.sha256(
            json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        scoped_key = (
            f"mutation:{kind.value}:"
            + hashlib.sha256(
                f"{profile_id}\0{key}".encode("utf-8")
            ).hexdigest()
        )
        command = WriteCommand(
            command_id=scoped_key,
            kind=kind,
            payload={
                **canonical,
                "journal_id": scoped_key,
                "request_hash": request_hash,
                "profile_id": profile_id,
                "idempotency_key": scoped_key,
            },
        )
        try:
            return dict(self.coordinator.submit(command, timeout=2.0).receipt)
        except CommandConflictError as exc:
            raise CanonicalMutationConflict(
                "idempotency key belongs to a different mutation request"
            ) from exc
        except (OwnershipRequiredError, WriteCoordinatorError) as exc:
            raise CanonicalRememberUnavailable(
                "canonical mutation is temporarily unavailable"
            ) from exc

    def _handle_admission(
        self,
        conn,
        capability,
        command: WriteCommand,
    ) -> WriteResult:
        """Project a journal command under the coordinator's sole transaction."""
        payload = command.payload
        raw_request = payload.get("request")
        if not isinstance(raw_request, Mapping):
            raise ValueError("admission command request is missing")
        request = RememberRequest.from_payload(raw_request)
        with self._binding_lock:
            db = self._db
            if request.profile_id != self._profile_id:
                raise ValueError("admission command targets a different profile")
            ingestion_request = IngestionRequest(
                content=request.content,
                profile_id=request.profile_id,
                source_type=request.source_type,
                idempotency_key=request.idempotency_key,
                metadata=dict(request.metadata),
                scope=request.scope,
                shared_with=request.shared_with,
                trusted_actor_id=request.trusted_actor_id,
                session_id=request.session_id,
                session_date=request.session_date,
                speaker=request.speaker,
                role=request.role,
            )
            # There is deliberately no validate_admission argument here. The
            # HTTP trust hook ran before journal.prepare, and no hook/model or
            # projection code can enter the coordinator transaction.
            with db._bind_coordinator_connection(conn, capability):
                command_impl = IngestionCommand(
                    IngestionOperationRepository(db),
                    write_queryable=self._writer,
                    materialize=self._materialize,
                )
                try:
                    receipt = command_impl.submit(ingestion_request)
                except IngestionRejectedError as exc:
                    raise CommandRejectedError() from exc
        return WriteResult.from_receipt(
            command,
            {
                "operation_id": receipt.operation_id,
                "pending_id": receipt.operation_id,
                "fact_ids": list(receipt.fact_ids),
                "count": len(receipt.fact_ids),
                "status": "queryable",
                "materialization_state": receipt.state.value,
            },
        )

    def _handle_mutation(self, conn, capability, command: WriteCommand) -> WriteResult:
        """Run only deterministic SQLite mutation statements under the writer."""
        payload = command.payload
        profile_id = _payload_text(payload, "profile_id")
        with self._binding_lock:
            if profile_id != self._profile_id:
                raise ValueError("mutation command targets a different profile")
            with self._db._bind_coordinator_connection(conn, capability):
                receipt = _execute_mutation(self._db, command.kind, profile_id, payload)
        receipt["operation_id"] = f"mutation:{command.kind.value}:{command.command_id}"
        return WriteResult.from_receipt(command, receipt)


def _materialization_is_not_available(_operation: IngestionOperation) -> list[str]:
    """Guard against accidental inline enrichment on the canonical path."""
    raise RuntimeError("canonical remember materialization belongs to the background worker")


_FACT_UPDATE_COLUMNS = frozenset({"content", "embedding", "fisher_mean", "fisher_variance"})


def _json_roundtrip(value: Mapping[str, Any]) -> dict[str, Any]:
    """Reject non-command data before it reaches the writer thread."""
    try:
        decoded = json.loads(json.dumps(value, sort_keys=True, ensure_ascii=False))
    except (TypeError, ValueError) as exc:
        raise ValueError("mutation payload must be JSON-compatible") from exc
    if not isinstance(decoded, dict):  # pragma: no cover - Mapping input is enforced
        raise ValueError("mutation payload must be an object")
    return decoded


def _payload_text(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"mutation command is missing {key}")
    return value


def _fact_row(db: DatabaseManager, fact_id: str, profile_id: str) -> Mapping[str, Any] | None:
    rows = db.execute(
        "SELECT * FROM atomic_facts WHERE fact_id = ? AND profile_id = ? LIMIT 1",
        (fact_id, profile_id),
    )
    return dict(rows[0]) if rows else None


def _execute_mutation(
    db: DatabaseManager,
    kind: CommandKind,
    profile_id: str,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Dispatch the finite mutation set; no policy, hooks, models, or I/O."""
    fact_id = _payload_text(payload, "fact_id")
    if kind is CommandKind.DELETE_FACT:
        return _delete_fact(db, fact_id, profile_id)
    if kind is CommandKind.UPDATE_FACT:
        return _update_fact(db, fact_id, profile_id, payload)
    if kind is CommandKind.ARCHIVE_FACT:
        return _archive_fact(db, fact_id, profile_id, payload)
    if kind is CommandKind.MERGE_FACT:
        return _merge_fact(db, fact_id, profile_id, payload)
    if kind is CommandKind.SET_FACT_SCOPE:
        return _set_fact_scope(db, fact_id, profile_id, payload)
    raise ValueError(f"unsupported mutation command {kind.value}")


def _delete_fact(db: DatabaseManager, fact_id: str, profile_id: str) -> dict[str, Any]:
    row = _fact_row(db, fact_id, profile_id)
    if row is None:
        return {"ok": False, "operation_id": f"delete:{fact_id}", "fact_id": fact_id}
    db.delete_fact(fact_id, profile_id=profile_id)
    return {
        "ok": True,
        "operation_id": f"delete:{fact_id}",
        "deleted": fact_id,
    }


def _update_fact(
    db: DatabaseManager, fact_id: str, profile_id: str, payload: Mapping[str, Any],
) -> dict[str, Any]:
    updates = payload.get("updates")
    if not isinstance(updates, Mapping) or not isinstance(updates.get("content"), str):
        raise ValueError("update command requires content")
    row = _fact_row(db, fact_id, profile_id)
    if row is None:
        return {"ok": False, "operation_id": f"update:{fact_id}", "fact_id": fact_id}
    safe = {
        key: _thaw_command_value(value)
        for key, value in updates.items()
        if key in _FACT_UPDATE_COLUMNS
    }
    if set(safe) != set(updates):
        raise ValueError("update command contains unsupported fields")
    db.update_fact(fact_id, safe, profile_id=profile_id)
    return {
        "ok": True,
        "operation_id": f"update:{fact_id}",
        "fact_id": fact_id,
    }


def _archive_fact(
    db: DatabaseManager, fact_id: str, profile_id: str, payload: Mapping[str, Any],
) -> dict[str, Any]:
    row = _fact_row(db, fact_id, profile_id)
    if row is None:
        return {"ok": False, "operation_id": f"archive:{fact_id}", "fact_id": fact_id}
    command_key = _payload_text(payload, "idempotency_key")
    archive_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{command_key}:archive"))
    archived_at = datetime.now(timezone.utc).isoformat()
    archive_payload = {
        key: row.get(key)
        for key in (
            "fact_id",
            "content",
            "canonical_entities_json",
            "importance",
            "confidence",
            "created_at",
        )
    }
    db.execute(
        "INSERT INTO memory_archive "
        "(archive_id, fact_id, profile_id, payload_json, archived_at, reason) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (
            archive_id,
            fact_id,
            profile_id,
            json.dumps(archive_payload),
            archived_at,
            "user_forget_dashboard",
        ),
    )
    db.execute(
        "UPDATE atomic_facts SET archive_status = 'archived' "
        "WHERE fact_id = ? AND profile_id = ?",
        (fact_id, profile_id),
    )
    return {
        "ok": True,
        "operation_id": f"archive:{fact_id}",
        "fact_id": fact_id,
        "archived_at": archived_at,
    }


def _merged_fact_ids(
    db: DatabaseManager, fact_id: str, kept: str, profile_id: str,
) -> set[str]:
    rows = db.execute(
        "SELECT fact_id FROM atomic_facts "
        "WHERE fact_id IN (?, ?) AND profile_id = ?",
        (fact_id, kept, profile_id),
    )
    return {row["fact_id"] for row in rows}


def _merge_fact(
    db: DatabaseManager, fact_id: str, profile_id: str, payload: Mapping[str, Any],
) -> dict[str, Any]:
    kept = _payload_text(payload, "kept_fact_id")
    if kept == fact_id:
        raise ValueError("cannot merge a fact into itself")
    found = _merged_fact_ids(db, fact_id, kept, profile_id)
    if fact_id not in found or kept not in found:
        return {"ok": False, "operation_id": f"merge:{fact_id}:{kept}", "fact_id": fact_id}
    command_key = _payload_text(payload, "idempotency_key")
    merge_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{command_key}:merge"))
    merged_at = datetime.now(timezone.utc).isoformat()
    db.execute(
        "INSERT INTO memory_merge_log "
        "(merge_id, profile_id, canonical_fact_id, merged_fact_id, "
        "cosine_sim, entity_jaccard, merged_at, reversible) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (merge_id, profile_id, kept, fact_id, None, None, merged_at, 1),
    )
    db.execute(
        "UPDATE atomic_facts SET merged_into = ?, archive_status = 'merged', "
        "archive_reason = 'user_merge_dashboard' "
        "WHERE fact_id = ? AND profile_id = ?",
        (kept, fact_id, profile_id),
    )
    return {
        "ok": True,
        "operation_id": f"merge:{fact_id}:{kept}",
        "merged": fact_id,
        "into": kept,
        "merged_at": merged_at,
    }


def _set_fact_scope(
    db: DatabaseManager, fact_id: str, profile_id: str, payload: Mapping[str, Any],
) -> dict[str, Any]:
    scope = _payload_text(payload, "scope")
    shared_with = payload.get("shared_with")
    if scope not in {"personal", "shared", "global"} or not isinstance(shared_with, tuple):
        raise ValueError("invalid scope command")
    if _fact_row(db, fact_id, profile_id) is None:
        return {"ok": False, "operation_id": f"scope:{fact_id}", "fact_id": fact_id}
    values = [str(item) for item in shared_with]
    db.update_fact(fact_id, {"scope": scope, "shared_with": values}, profile_id=profile_id)
    return {
        "ok": True,
        "operation_id": f"scope:{fact_id}",
        "fact_id": fact_id,
        "scope": scope,
        "shared_with": values,
    }


def _thaw_command_value(value: Any) -> Any:
    """Restore coordinator-frozen JSON lists before DatabaseManager serializes them."""
    if isinstance(value, Mapping):
        return {key: _thaw_command_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_command_value(item) for item in value]
    return value


__all__ = [
    "CanonicalRememberRuntime",
    "CanonicalRememberUnavailable",
    "validate_deterministic_admission",
]
