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
import logging
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
from superlocalmemory.core.transactions.concrete_owners import (
    REQUIRED_ADMISSION_OWNERS,
)
from superlocalmemory.core.transactions.obligations import ObligationLedger
from superlocalmemory.core.transactions.owners import (
    ObligationKind,
    OperationContext,
)
from superlocalmemory.storage.admission_codec import MachineKeyCommandCodec
from superlocalmemory.storage.admission_journal import (
    Actor,
    AdmissionEntry,
    AdmissionJournal,
    AdmissionJournalUnavailable,
    AdmissionPayloadError,
    RememberRequest,
    TerminalAdmissionError,
)
from superlocalmemory.storage.database import DatabaseManager
from superlocalmemory.storage.generation_fence import (
    admitted_epoch,
    clear_admission_epoch,
    record_admission_epoch,
)
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

logger = logging.getLogger("superlocalmemory.core.remember_runtime")
_OBLIGATION_LEDGER = ObligationLedger()


def _obligation_schema_present(conn) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='projection_obligations'"
    ).fetchone()
    if row is None:
        return False
    columns = {r[1] for r in conn.execute("PRAGMA table_info(projection_obligations)")}
    return "context_digest" in columns


class CanonicalRememberUnavailable(RuntimeError):
    """The daemon cannot accept a bounded canonical remember request."""


class DaemonAlreadyServing(RuntimeError):
    """A healthy SLM daemon is already serving; this instance should exit 0.

    Raised by ``CanonicalRememberRuntime.start()`` (H2) when the writer claim
    fails AND a health-verified daemon is responding on the configured port.
    Caught by ``unified_daemon.py`` lifespan → ``sys.exit(0)``.
    """


class CanonicalMutationConflict(WriteCoordinatorError, ValueError):
    """A mutation retry key was reused for different immutable input."""


_MUTATION_IDEMPOTENCY_KEY = re.compile(r"^[A-Za-z0-9._:-]{1,256}$")

# 4.1.14 audit: bound on cached per-profile admission handlers. Profile
# counts are small; the cap is a backstop against unbounded growth, not a
# tuner — eviction is plain FIFO and a rebuild is one indexed SELECT plus
# a handler construction.
_ROUTED_WRITERS_CAP = 32


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


# ---------------------------------------------------------------------------
# H2 — graceful single-instance helpers (keep all I/O in stdlib; no imports
# from the server layer to avoid circular dependencies with core).
# ---------------------------------------------------------------------------

def _get_daemon_port() -> int:
    """Return the HTTP port this daemon instance listens on.

    Reads ``SLM_DAEMON_PORT`` from the environment (set by ``unified_daemon``
    at startup) and falls back to the conventional default of 8765.
    """
    import os

    raw = os.environ.get("SLM_DAEMON_PORT", "")
    try:
        return int(raw)
    except (ValueError, TypeError):
        return 8765


def _slm_health_check(port: int) -> bool:
    """Return True iff a healthy SLM daemon responds on *port* within 2 s."""
    import urllib.request

    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{port}/health", timeout=2
        ) as resp:
            return int(resp.status) == 200
    except Exception:
        return False


def _boot_self_heal(data_dir: Path) -> None:
    """Run the H1 stale-artifact reaper.  Fail-soft — never blocks startup."""
    try:
        from superlocalmemory.infra.self_heal import reap_stale_artifacts

        report = reap_stale_artifacts(data_dir)
        if report["removed"]:
            logger.info(
                "remember_runtime self-heal: removed %d stale artifact(s)",
                len(report["removed"]),
            )
    except Exception as exc:
        logger.debug("remember_runtime self-heal failed (non-fatal): %s", exc)


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
        max_verbatim_chars: int = 24_000,
        max_ingest_bytes: int = 1_048_576,
    ) -> None:
        if not profile_id:
            raise ValueError("profile_id is required")
        self._db = db
        self._profile_id = profile_id
        self._writer = writer
        # Writer bounds retained so handlers built later for routed profiles
        # (per-request profile routing) carry the same deterministic limits
        # the active-profile handler was built with.
        self._max_verbatim_chars = int(max_verbatim_chars)
        self._max_ingest_bytes = int(max_ingest_bytes)
        # Per-request profile routing: admission handlers bound to non-active
        # profiles, keyed by profile_id. Built on first use, cleared on rebind.
        self._routed_writers: dict[str, QueryableWriter] = {}
        self._materialize = materialize or _materialization_is_not_available
        self._binding_lock = threading.RLock()
        self._generation = 0
        self.coordinator = WriteCoordinator(db.db_path, owner_id=owner_id)
        self.journal = AdmissionJournal(
            journal_path,
            codec=MachineKeyCommandCodec(Path(journal_path).with_name("admission-key.bin")),
        )
        self._service = RememberService(self.journal, _CoordinatorAdapter(self.coordinator))
        self._started = False
        self._obligation_schema_ok: bool | None = None

    @classmethod
    def for_engine(cls, engine: Any) -> "CanonicalRememberRuntime":
        """Create the daemon boundary from an initialized engine only."""
        from superlocalmemory.core.engine_ingestion import build_immediate_admission_handler
        from superlocalmemory.infra.data_root import state_path

        db = engine._db
        store_config = getattr(engine._config, "store", None)
        max_verbatim_chars = getattr(
            store_config, "max_verbatim_chars", 24_000,
        )
        max_ingest_bytes = getattr(
            store_config, "max_ingest_bytes", 1_048_576,
        )
        return cls(
            db=db,
            profile_id=engine._profile_id,
            writer=build_immediate_admission_handler(
                db,
                profile_id=engine._profile_id,
                max_verbatim_chars=max_verbatim_chars,
                max_ingest_bytes=max_ingest_bytes,
            ),
            journal_path=state_path("admission_journal.db"),
            max_verbatim_chars=max_verbatim_chars,
            max_ingest_bytes=max_ingest_bytes,
        )

    def start(self) -> None:
        """Claim writer ownership, install the handler, then recover journal work.

        H2 / G-01+G-05 — bounded graceful single-instance:

        When ``claim_ownership()`` fails (another process holds the portalocker
        flock), we enter a bounded retry loop (≤ 5 attempts, ~1 s between each,
        total ≤ ~6 s) rather than immediately crashing or silently giving up.
        This handles the race between a healthy daemon's HTTP-server bind and our
        health check — the holder may be alive but not yet responding.

        Per-iteration logic:
          (a) ``claim_ownership()`` succeeds → break and proceed (holder died).
          (b) ``_slm_health_check(port)`` is True → raise ``DaemonAlreadyServing``
              (caught by ``unified_daemon.py`` lifespan → ``sys.exit(0)``).
          (c) First iteration only → run ``_boot_self_heal()`` to clear
              provably-dead metadata artifacts, then continue.

        After the loop exhausts all attempts without claiming the lock, we know
        a live process is holding the portalocker flock.  Raise
        ``DaemonAlreadyServing`` (NOT ``CanonicalRememberUnavailable``) so the
        daemon exits cleanly rather than crashing with a traceback.

        INVARIANT: ``claim_ownership()`` (portalocker OS flock) is the sole
        writer-integrity gate.  We never bypass it, never signal or kill a live
        process.
        """
        import time as _time

        if self._started:
            return
        if not self.coordinator.claim_ownership():
            port = _get_daemon_port()
            _self_heal_done = False
            _MAX_ATTEMPTS = 5

            for _attempt in range(_MAX_ATTEMPTS):
                if _attempt > 0:
                    _time.sleep(1.0)

                # (a) Re-try the claim — holder may have died in the gap.
                if self.coordinator.claim_ownership():
                    break  # claimed → proceed to handler registration

                # (b) Health-verified owner → exit gracefully.
                if _slm_health_check(port):
                    raise DaemonAlreadyServing(
                        f"another healthy SLM daemon is already serving"
                        f" on port {port}"
                    )

                # (c) First iteration only: clear provably-dead stale artifacts.
                if not _self_heal_done:
                    logger.info(
                        "remember_runtime: writer claim failed, no healthy"
                        " daemon on port %d; running self-heal (attempt %d/%d)",
                        port, _attempt + 1, _MAX_ATTEMPTS,
                    )
                    _boot_self_heal(self.coordinator.db_path.parent)
                    _self_heal_done = True
            else:
                # Loop exhausted: a live process holds the portalocker flock.
                # Exit cleanly — never crash with a traceback.
                raise DaemonAlreadyServing(
                    f"another SLM daemon holds the writer lock after"
                    f" {_MAX_ATTEMPTS} attempts on port {port}"
                )
        try:
            self.coordinator.register_handler(CommandKind.ADMISSION, self._handle_admission)
            self.coordinator.register_handler(CommandKind.DELETE_FACT, self._handle_mutation)
            self.coordinator.register_handler(CommandKind.UPDATE_FACT, self._handle_mutation)
            self.coordinator.register_handler(CommandKind.PROPOSE_CORRECTION, self._handle_mutation)
            self.coordinator.register_handler(CommandKind.APPLY_CORRECTION, self._handle_mutation)
            self.coordinator.register_handler(CommandKind.REJECT_CORRECTION, self._handle_mutation)
            self.coordinator.register_handler(
                CommandKind.ROLLBACK_CORRECTION, self._handle_mutation,
            )
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
        max_verbatim_chars = getattr(
            store_config, "max_verbatim_chars", 24_000,
        )
        max_ingest_bytes = getattr(
            store_config, "max_ingest_bytes", 1_048_576,
        )
        writer = build_immediate_admission_handler(
            db,
            profile_id=profile_id,
            max_verbatim_chars=max_verbatim_chars,
            max_ingest_bytes=max_ingest_bytes,
        )
        with self._binding_lock:
            previous = (
                self._db, self._profile_id, self._writer,
                self._max_verbatim_chars, self._max_ingest_bytes,
            )
            self._db = db
            self._profile_id = profile_id
            self._writer = writer
            self._max_verbatim_chars = max_verbatim_chars
            self._max_ingest_bytes = max_ingest_bytes
            # Routed handlers close over the previous binding; drop them so
            # the next routed request rebuilds against the new one.
            self._routed_writers.clear()
            self._generation += 1
        try:
            self.replay_pending()
        except BaseException:
            with self._binding_lock:
                # Roll back the WHOLE binding, not just the writer triple: a
                # stale limits pair or a routed-handler cache built against
                # the failed binding would outlive the rebind that never
                # happened.
                (
                    self._db, self._profile_id, self._writer,
                    self._max_verbatim_chars, self._max_ingest_bytes,
                ) = previous
                self._routed_writers.clear()
                self._generation -= 1
            raise

    def remember(
        self, request: RememberRequest, actor: Actor, *, deadline_ms: int = 2_000,
    ) -> RememberReceipt:
        """Journal then commit one bounded queryable admission receipt."""
        if not self._started:
            raise CanonicalRememberUnavailable("canonical remember writer is not ready")
        if deadline_ms < 1 or deadline_ms > 2_000:
            raise ValueError("deadline_ms must be between 1 and 2000")
        with self._binding_lock:
            admitted = self._generation
        record_admission_epoch(request.profile_id, request.idempotency_key, admitted)
        try:
            return self._service.remember(request, actor, deadline_ms=deadline_ms)
        except (
            AdmissionJournalUnavailable,
            OwnershipRequiredError,
            WriteCoordinatorError,
        ) as exc:
            # Name which of the three it was. They are not interchangeable and
            # they call for different responses: an unavailable journal is I/O
            # or lock contention and worth retrying, lost ownership means
            # another writer holds the lease, and a coordinator error is the
            # same type the generation fence raises to reject a stale epoch.
            # Collapsing all three into one string makes a spurious fence
            # rejection indistinguishable from a transient disk stall, for the
            # operator reading a log and for a caller deciding whether to
            # retry. Only the class name is included: it is the whole of the
            # discriminating information and carries no request content.
            raise CanonicalRememberUnavailable(
                "canonical remember is temporarily unavailable "
                f"({type(exc).__name__})"
            ) from exc
        finally:
            clear_admission_epoch(request.profile_id, request.idempotency_key)

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

    def create_correction_successor(
        self,
        profile_id: str,
        fact_id: str,
        successor_fact_id: str,
        content: str,
        *,
        embedding: list[float] | None = None,
        fisher_mean: list[float] | None = None,
        fisher_variance: list[float] | None = None,
        trusted_actor_id: str = "canonical-runtime",
        idempotency_key: str | None = None,
    ) -> Mapping[str, Any]:
        """Atomically create an immutable, review-required successor case."""
        return self._submit_mutation(
            CommandKind.PROPOSE_CORRECTION,
            profile_id,
            {
                "fact_id": fact_id,
                "successor_fact_id": successor_fact_id,
                "content": content,
                "embedding": _json_roundtrip({"value": embedding})["value"],
                "fisher_mean": _json_roundtrip({"value": fisher_mean})["value"],
                "fisher_variance": _json_roundtrip({"value": fisher_variance})["value"],
                "trusted_actor_id": trusted_actor_id,
            },
            idempotency_key=idempotency_key,
        )

    def transition_correction(
        self,
        profile_id: str,
        case_id: str,
        *,
        action: str,
        expected_version: int,
        actor_id: str,
        event_valid_until: str | None = None,
        idempotency_key: str | None = None,
    ) -> Mapping[str, Any]:
        """Apply, reject, or roll back one server-authorized correction case."""
        command = {
            "apply": CommandKind.APPLY_CORRECTION,
            "reject": CommandKind.REJECT_CORRECTION,
            "rollback": CommandKind.ROLLBACK_CORRECTION,
        }.get(action)
        if command is None:
            raise ValueError("correction action must be apply, reject, or rollback")
        return self._submit_mutation(
            command,
            profile_id,
            {
                "case_id": case_id,
                "expected_version": expected_version,
                "trusted_actor_id": actor_id,
                "event_valid_until": event_valid_until,
            },
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
        except CanonicalMutationConflict:
            # A deterministic lifecycle conflict is not a writer outage.  Let
            # HTTP/CLI/MCP report the actionable 409/validation result.
            raise
        except CommandConflictError as exc:
            raise CanonicalMutationConflict(
                "idempotency key belongs to a different mutation request"
            ) from exc
        except (OwnershipRequiredError, WriteCoordinatorError) as exc:
            raise CanonicalRememberUnavailable(
                "canonical mutation is temporarily unavailable"
            ) from exc

    def _profile_exists_locked(self, profile_id: str) -> bool:
        """Whether ``profile_id`` still names a live profile row.

        Caller must hold ``self._binding_lock``. A single indexed point
        lookup — cheap enough to run on every routed admission.
        """
        return bool(self._db.execute(
            "SELECT 1 AS one FROM profiles WHERE profile_id = ?", (profile_id,),
        ))

    def _routed_writer_locked(self, profile_id: str) -> QueryableWriter:
        """Return the admission handler bound to a non-active profile.

        Per-request profile routing: a request carrying a different
        profile_id must reach a handler bound to THAT profile, never a 409.
        The handler is built once per profile and cached under the binding
        lock. Fail-closed on a profile that does not exist, so not even a
        journal replay can invent one.

        4.1.14 audit: the cache is BOUNDED (FIFO eviction past the cap —
        profile counts are small; the cap is a backstop, not a tuner) and
        every hit revalidates existence, so a profile deleted after its
        first route fails closed instead of admitting from a stale
        handler.

        Caller must hold ``self._binding_lock``.
        """
        writer = self._routed_writers.get(profile_id)
        if writer is not None:
            if self._profile_exists_locked(profile_id):
                return writer
            del self._routed_writers[profile_id]
        from superlocalmemory.core.engine_ingestion import (
            build_immediate_admission_handler,
        )

        if not self._profile_exists_locked(profile_id):
            from superlocalmemory.core.ingestion_command import (
                UnknownProfileError,
            )
            raise UnknownProfileError(
                "admission command targets an unknown profile"
            )
        writer = build_immediate_admission_handler(
            self._db,
            profile_id=profile_id,
            max_verbatim_chars=self._max_verbatim_chars,
            max_ingest_bytes=self._max_ingest_bytes,
        )
        self._routed_writers[profile_id] = writer
        while len(self._routed_writers) > _ROUTED_WRITERS_CAP:
            self._routed_writers.pop(next(iter(self._routed_writers)))
        return writer

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
            # Per-request profile routing: the active-profile handler serves
            # the daemon's binding; any other existing profile gets (and
            # caches) a handler bound to itself.
            if request.profile_id == self._profile_id:
                writer = self._writer
            else:
                writer = self._routed_writer_locked(request.profile_id)
            expected = admitted_epoch(request.profile_id, request.idempotency_key)
            if expected is not None and expected != self._generation:
                raise ValueError("admission command epoch is stale")
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
                    write_queryable=writer,
                    materialize=self._materialize,
                )
                try:
                    receipt = command_impl.submit(ingestion_request)
                except IngestionRejectedError as exc:
                    raise CommandRejectedError() from exc
                self._record_projection_obligations(conn, request, receipt)
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

    def _record_projection_obligations(self, conn, request, receipt) -> None:
        """Record per-owner APPLY obligations for the admitted facts.

        Fail-closed: raises RuntimeError if M033 is absent.  This is
        intentional — a successful remember() without a corresponding
        obligation record would make projection audits permanently incomplete.
        Back-compat: installs without M033 must run migrations first.

        Schema negative cache: ``_obligation_schema_ok`` is re-checked on
        every call while False so that hot migrations (M033 applied while the
        runtime is running) are detected without a restart.  Once the schema
        is confirmed present the value is cached True for the lifetime of the
        runtime instance.

        Cross-process erasure fence: distributed, multi-writer erasure fencing
        is intentionally SCOPED OUT of V4.  Single-process SQLite WAL provides
        adequate isolation for the current deployment model.  See
        docs/architecture/erasure-fence-deferred.md for the deferred design.
        """
        fact_ids = tuple(getattr(receipt, "fact_ids", ()) or ())
        if not fact_ids:
            return
        # Re-check while False so a hot M033 migration is picked up without
        # requiring a daemon restart.  Cache True permanently once confirmed.
        if not self._obligation_schema_ok:
            self._obligation_schema_ok = _obligation_schema_present(conn)
        if not self._obligation_schema_ok:
            raise RuntimeError(
                "projection_obligations schema is absent; "
                "run migrations (M033) before ingesting facts"
            )
        context = OperationContext(
            operation_id=receipt.operation_id,
            profile_id=request.profile_id,
            subject_id=receipt.operation_id,
            fact_ids=fact_ids,
        )
        _OBLIGATION_LEDGER.record_many(
            conn, context, REQUIRED_ADMISSION_OWNERS, ObligationKind.APPLY,
        )

    def _handle_mutation(self, conn, capability, command: WriteCommand) -> WriteResult:
        """Run only deterministic SQLite mutation statements under the writer."""
        payload = command.payload
        profile_id = _payload_text(payload, "profile_id")
        with self._binding_lock:
            if profile_id != self._profile_id:
                raise ValueError("mutation command targets a different profile")
            with self._db._bind_coordinator_connection(conn, capability):
                receipt = _execute_mutation(
                    self._db, command.kind, profile_id, payload, connection=conn
                )
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
    *,
    connection: Any,
) -> dict[str, Any]:
    """Dispatch the finite mutation set; no policy, hooks, models, or I/O."""
    if kind is CommandKind.PROPOSE_CORRECTION:
        return _propose_correction_successor(db, profile_id, payload, connection=connection)
    if kind in {
        CommandKind.APPLY_CORRECTION,
        CommandKind.REJECT_CORRECTION,
        CommandKind.ROLLBACK_CORRECTION,
    }:
        return _transition_correction(db, kind, profile_id, payload, connection=connection)
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
    # M042 deliberately retains immutable predecessor/successor history.
    # SQLite's FK would reject a direct delete, but exposing that as a generic
    # writer outage turns a real lifecycle conflict into a misleading 503.
    # A dedicated erasure workflow owns ledger removal; ordinary forget never
    # deletes a fact that is part of immutable correction history.
    try:
        protected = db.execute(
            "SELECT 1 FROM correction_cases "
            "WHERE profile_id=? AND (predecessor_fact_id=? OR successor_fact_id=?) LIMIT 1",
            (profile_id, fact_id, fact_id),
        )
    except Exception as exc:
        if "no such table" in str(exc).lower():
            protected = []
        else:
            raise
    if protected:
        raise CanonicalMutationConflict(
            "fact is protected by correction history; resolve the correction before forgetting it"
        )
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


def _propose_correction_successor(
    db: DatabaseManager,
    profile_id: str,
    payload: Mapping[str, Any],
    *,
    connection: Any,
) -> dict[str, Any]:
    """Create successor and M042 proposal in one canonical writer transaction.

    System time is represented by the new fact's ``created_at`` and temporal
    knowledge anchor.  Event-time fields are copied exactly from the
    predecessor because an edit payload has no trustworthy event-time claim.
    """
    fact_id = _payload_text(payload, "fact_id")
    source = _fact_row(db, fact_id, profile_id)
    if source is None:
        return {"ok": False, "operation_id": f"correction:{fact_id}", "fact_id": fact_id}
    successor_id = _payload_text(payload, "successor_fact_id")
    content = _payload_text(payload, "content").strip()
    if not content:
        raise ValueError("correction successor content cannot be empty")
    if successor_id == fact_id:
        raise ValueError("correction successor must differ from predecessor")

    from superlocalmemory.storage.models import AtomicFact, FactType, MemoryLifecycle, SignalType

    def _list_payload(key: str) -> list[float] | None:
        value = payload.get(key)
        if value is None:
            return None
        if not isinstance(value, (list, tuple)) or not all(
            isinstance(v, (int, float)) for v in value
        ):
            raise ValueError(f"correction successor {key} must be numeric or null")
        return [float(v) for v in value]

    now = datetime.now(timezone.utc).isoformat()
    successor = AtomicFact(
        fact_id=successor_id,
        memory_id=str(source.get("memory_id") or ""),
        profile_id=profile_id,
        scope=str(source.get("scope") or "personal"),
        shared_with=json.loads(source["shared_with"]) if source.get("shared_with") else None,
        content=content,
        fact_type=FactType(str(source.get("fact_type") or "semantic")),
        entities=json.loads(source["entities_json"]) if source.get("entities_json") else [],
        canonical_entities=(
            json.loads(source["canonical_entities_json"])
            if source.get("canonical_entities_json") else []
        ),
        observation_date=source.get("observation_date"),
        referenced_date=source.get("referenced_date"),
        interval_start=source.get("interval_start"),
        interval_end=source.get("interval_end"),
        confidence=float(source.get("confidence") or 0.0),
        importance=float(source.get("importance") or 0.0),
        evidence_count=1,
        access_count=0,
        source_turn_ids=(
            json.loads(source["source_turn_ids_json"])
            if source.get("source_turn_ids_json") else []
        ),
        session_id=str(source.get("session_id") or ""),
        embedding=_list_payload("embedding"),
        fisher_mean=_list_payload("fisher_mean"),
        fisher_variance=_list_payload("fisher_variance"),
        lifecycle=MemoryLifecycle.ACTIVE,
        langevin_position=None,
        emotional_valence=float(source.get("emotional_valence") or 0.0),
        emotional_arousal=float(source.get("emotional_arousal") or 0.0),
        signal_type=SignalType(str(source.get("signal_type") or "factual")),
        created_at=now,
    )
    persisted_id = db.insert_fact_immutable(successor)
    from superlocalmemory.storage.correction_cases import (
        CorrectionActor,
        propose_on_connection,
    )

    trusted_actor_id = _payload_text(payload, "trusted_actor_id")
    actor = CorrectionActor(
        actor_id=trusted_actor_id,
        actor_kind="host_authenticated",
        trust_tier="trusted",
    )
    case_id = uuid.uuid5(
        uuid.NAMESPACE_URL,
        f"slm-correction:{profile_id}:{fact_id}:{persisted_id}",
    ).hex
    case = propose_on_connection(
        connection,
        case_id=case_id,
        profile_id=profile_id,
        scope=str(source.get("scope") or "personal"),
        predecessor_fact_id=fact_id,
        successor_fact_id=persisted_id,
        reason_code="direct_content_correction",
        actor=actor,
        idempotency_key=_payload_text(payload, "idempotency_key"),
        is_profile_active=lambda candidate: candidate == profile_id,
        is_actor_trusted=lambda candidate: candidate == actor,
    )
    return {
        "ok": True,
        "operation_id": f"correction:{fact_id}:{persisted_id}",
        "predecessor_fact_id": fact_id,
        "successor_fact_id": persisted_id,
        "case_id": case.case_id,
        "status": case.status,
        "version": case.version,
        "created_at": now,
    }


def _transition_correction(
    db: DatabaseManager,
    kind: CommandKind,
    profile_id: str,
    payload: Mapping[str, Any],
    *,
    connection: Any,
) -> dict[str, Any]:
    """Advance one correction case under the same receipt transaction."""
    from superlocalmemory.storage.correction_cases import (
        CorrectionActor,
        transition_on_connection,
    )

    case_id = _payload_text(payload, "case_id")
    version = payload.get("expected_version")
    if not isinstance(version, int) or isinstance(version, bool) or version < 0:
        raise ValueError("correction expected_version must be a non-negative integer")
    actor = CorrectionActor(
        actor_id=_payload_text(payload, "trusted_actor_id"),
        actor_kind="host_authenticated",
        trust_tier="trusted",
    )
    transitions = {
        CommandKind.APPLY_CORRECTION: ("proposed", "applied", True),
        CommandKind.REJECT_CORRECTION: ("proposed", "rejected", False),
        CommandKind.ROLLBACK_CORRECTION: ("applied", "rolled_back", True),
    }
    from_status, to_status, mutate_temporal = transitions[kind]
    case = transition_on_connection(
        connection,
        case_id=case_id,
        expected_version=version,
        actor=actor,
        operation_id=_payload_text(payload, "idempotency_key"),
        from_status=from_status,
        to_status=to_status,
        mutate_temporal=mutate_temporal,
        event_valid_until=payload.get("event_valid_until"),
        is_profile_active=lambda candidate: candidate == profile_id,
        is_actor_trusted=lambda candidate: candidate == actor,
    )
    return {
        "ok": True,
        "operation_id": f"correction:{to_status}:{case.case_id}",
        "case_id": case.case_id,
        "predecessor_fact_id": case.predecessor_fact_id,
        "successor_fact_id": case.successor_fact_id,
        "status": case.status,
        "version": case.version,
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
    "DaemonAlreadyServing",
    "validate_deterministic_admission",
]
