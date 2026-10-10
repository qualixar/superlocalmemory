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
import sqlite3
import threading
import uuid
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from superlocalmemory.core.ingestion_command import (
    IngestionCommand,
    IngestionOperation,
    IngestionOperationRepository,
    IngestionRejectedError,
    IngestionRequest,
    MaterializationResult,
)
from superlocalmemory.core.deferred_admission import DeferredCommitter
from superlocalmemory.core.mutation_routing import MutationTarget, classify_target
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
    _thaw_json,
)

QueryableWriter = Callable[[IngestionRequest, str], list[str]]
Materializer = Callable[
    [IngestionOperation],
    list[str] | tuple[str, ...] | MaterializationResult,
]

logger = logging.getLogger("superlocalmemory.core.remember_runtime")
_OBLIGATION_LEDGER = ObligationLedger()

#: How long one deferred commit attempt may wait for the writer before the
#: committer backs off and tries again. Short enough that stop() stays prompt.
_DEFERRED_COMMIT_WAIT_MS = 5_000


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


class CanonicalRememberBusy(CanonicalRememberUnavailable):
    """The save was refused before anything was written; retrying is safe.

    Raised when the admission journal is saturated (its queue is full or the
    caller's budget ran out while queued). Nothing about this save was stored.
    """

    def __init__(self, message: str, *, retry_after_seconds: int = 1) -> None:
        super().__init__(message)
        self.retry_after_seconds = max(1, int(retry_after_seconds))


class DaemonAlreadyServing(RuntimeError):
    """A healthy SLM daemon is already serving; this instance should exit 0.

    Raised by ``CanonicalRememberRuntime.start()`` (H2) when the writer claim
    fails AND a health-verified daemon is responding on the configured port.
    Caught by ``unified_daemon.py`` lifespan → ``sys.exit(0)``.
    """


class CanonicalMutationConflict(WriteCoordinatorError, ValueError):
    """A mutation retry key was reused for different immutable input."""


class MutationNotRoutable(CanonicalMutationConflict):
    """A mutation of a kind that may not leave the active profile named another.

    A refusal by design (``core/mutation_routing.py``), not an outage: retrying
    can never succeed, so it must not be reported as "temporarily unavailable".
    """


class MutationTargetMissing(WriteCoordinatorError, LookupError):
    """What a mutation names is not there: a "not found", never an outage.

    Raised, not returned, so the writer commits nothing and the caller's retry
    key stays unspent for when the target exists. Not a conflict either: the
    generic HTTP mapping turns it into a 404.
    """


class UnknownMutationProfile(MutationTargetMissing):
    """A routed mutation named a profile that does not exist (any more)."""


class CaseNotInProfile(MutationTargetMissing):
    """A correction review named a case that is not in the profile reviewed.

    One answer whether the case is missing or another profile's, so review
    cannot probe other profiles.
    """


_MUTATION_IDEMPOTENCY_KEY = re.compile(r"^[A-Za-z0-9._:-]{1,256}$")
#: Most facts one kind change may cover.
_MAX_KIND_ITEMS = 200

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
    """Return True iff an SLM daemon of this account and data folder answers
    on *port* within 2 s.

    Any healthy answer used to count: on a computer shared by several
    accounts, another account's daemon on the default port made this daemon
    conclude its own namespace was already served, and exit.
    """
    import json
    import urllib.request

    from superlocalmemory.infra.daemon_identity import health_is_same_account

    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{port}/health", timeout=2
        ) as resp:
            if int(resp.status) != 200:
                return False
            health = json.loads(resp.read().decode("utf-8"))
    except Exception:
        return False
    return isinstance(health, dict) and health_is_same_account(health)


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
        except WriteCoordinatorError as exc:
            if _caused_by_unknown_profile(exc):
                # Final, not contention: a deleted profile never comes back,
                # so retrying would loop forever. Recorded as a rejection.
                raise TerminalAdmissionError("UNKNOWN_PROFILE") from exc
            raise
        return {"state": "committed", "receipt": dict(result.receipt)}


def _journal_was_busy(error: BaseException) -> bool:
    """True when the journal refused for overload or time, not for storage."""
    from superlocalmemory.storage.journal_writer import (
        _BUSY_MESSAGE,
        AdmissionJournalOverloaded,
        is_sqlite_busy,
    )

    if isinstance(error, AdmissionJournalOverloaded) or str(error) == _BUSY_MESSAGE:
        return True
    cause = error.__cause__
    if cause is None:
        return True  # raised by the journal itself, without a storage error
    return isinstance(cause, sqlite3.Error) and is_sqlite_busy(cause)


def _caused_by_unknown_profile(error: BaseException) -> bool:
    from superlocalmemory.core.ingestion_command import UnknownProfileError

    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        if isinstance(current, UnknownProfileError):
            return True
        seen.add(id(current))
        current = current.__cause__ or current.__context__
    return False


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
        # Commits remembers that were accepted while the writer was busy.
        self._deferred = DeferredCommitter(self.journal, self._commit_deferred)
        self._started = False
        # Mutations may run as soon as the writer does - during start()'s own
        # recovery too, which applies replacements of replayed saves.
        self._writer_open = False
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
            self.coordinator.register_handler(CommandKind.SET_FACT_KIND, self._handle_mutation)
            self.coordinator.register_handler(
                CommandKind.REPLACE_BY_CALLER, self._handle_mutation,
            )
            self.coordinator.start()
            self._writer_open = True
            self.replay_pending()
        except BaseException:
            self._writer_open = False
            self.coordinator.release_ownership()
            raise
        if self._deferred.stopped:  # a restart of this same runtime
            self._deferred = DeferredCommitter(self.journal, self._commit_deferred)
        self._started = True
        # Everything still pending - saves for other profiles, and any the
        # synchronous replay above could not finish - is committed in the
        # background, so no accepted save waits for a rebind or a restart.
        self._hand_pending_to_committer()

    def _hand_pending_to_committer(self) -> int:
        """Queue every pending journal entry, of every profile, for commit.

        Never raises: the entries are durable whatever happens here, and the
        next start recovers anything that could not be queued now.
        """
        try:
            entries = self.journal.pending_entries()
        except Exception as exc:  # noqa: BLE001 - durable regardless
            logger.warning(
                "pending saves could not be listed (%s); they stay durable and "
                "are recovered at the next start", type(exc).__name__,
            )
            return 0
        for entry in entries:
            self._deferred.defer(entry)
        return len(entries)

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
        # Before the lease goes: an in-flight deferred commit finishes or
        # fails cleanly, and anything still queued stays in the journal for
        # replay_pending at the next start.
        self._deferred.stop()
        self._writer_open = False
        self.coordinator.release_ownership()
        # Drains queued journal marks, then frees the writer thread and the
        # pooled connections; a later start reopens them.
        self.journal.close()

    @property
    def deferred_count(self) -> int:
        """Remembers accepted under contention and not yet committed."""
        return self._deferred.pending

    def admission_status(self) -> dict[str, int]:
        """Counts an operator needs: saves still being indexed, saves set aside."""
        try:
            unreadable = self.journal.quarantined_count()
        except Exception as exc:  # noqa: BLE001 - status must answer
            logger.warning("admission journal status unavailable (%s)", type(exc).__name__)
            unreadable = -1
        return {"saves_waiting": self.deferred_count, "unreadable_saves": unreadable}

    def wait_for_deferred(self, timeout: float) -> bool:
        """Block until every accepted remember is committed; False on timeout."""
        return self._deferred.wait_idle(timeout)

    def _commit_deferred(
        self, entry: AdmissionEntry, request: RememberRequest,
    ) -> Mapping[str, Any]:
        command = RememberAdmissionCommand(
            journal_id=entry.journal_id,
            request_hash=entry.request_hash,
            request=request,
            profile_id=entry.profile_id,
            idempotency_key=entry.idempotency_key,
        )
        receipt = _CoordinatorAdapter(self.coordinator).submit(
            command, wait_ms=_DEFERRED_COMMIT_WAIT_MS,
        )["receipt"]
        # Before the journal marks it committed, so a crash in between replays
        # the replacement too (it is idempotent, keyed on the operation id).
        self._apply_replacement(request, receipt)
        return receipt

    def _apply_replacement(
        self, request: RememberRequest, receipt: Mapping[str, Any],
    ) -> None:
        """Retire what a save named in ``replaces``, after a background commit.

        A foreground save is answered by the daemon, which applies (and
        reports) its own replacement; a save accepted under contention or
        recovered after a restart has no caller waiting, so it is applied
        here. Never raises: the save stands either way, and the outcome is
        logged and returned to whoever resends the same idempotency key.
        """
        if not request.replaces:
            return
        from types import SimpleNamespace

        from superlocalmemory.core.remember_replaces import replace_after_save

        with self._binding_lock:
            db, active = self._db, self._profile_id
        result = replace_after_save(
            self, SimpleNamespace(_db=db), replaces=request.replaces,
            profile_id=request.profile_id,
            successor_fact_ids=list(receipt.get("fact_ids") or ()),
            operation_id=str(receipt.get("operation_id") or ""),
            trusted_actor_id=request.trusted_actor_id,
            routed=request.profile_id != active,
        )
        if result.get("ok"):
            logger.info("a deferred save retired %d fact(s) it replaces",
                        len(result.get("fact_ids") or ()))
        else:
            logger.warning("a deferred save did not replace what it named: %s",
                           result.get("reason"))

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
        # Everything that can fail (building the handler) is done above, so
        # the swap below is all-or-nothing and needs no rollback.
        with self._binding_lock:
            self._db = db
            self._profile_id = profile_id
            self._writer = writer
            self._max_verbatim_chars = max_verbatim_chars
            self._max_ingest_bytes = max_ingest_bytes
            # Routed handlers close over the previous binding; drop them so
            # the next routed request rebuilds against the new one.
            self._routed_writers.clear()
            self._generation += 1
        # Pending saves (for this profile or any other) are handed to the
        # background committer rather than replayed here: a switch must not
        # wait on, or fail because of, a writer that is busy right now.
        self._hand_pending_to_committer()

    def remember(
        self,
        request: RememberRequest,
        actor: Actor,
        *,
        deadline_ms: int = 2_000,
        accept_after_ms: int | None = None,
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
            return self._service.remember(
                request, actor, deadline_ms=deadline_ms, defer=self._deferred.defer,
                accept_after_ms=accept_after_ms,
            )
        except AdmissionJournalUnavailable as exc:
            # Only the journal prepare can raise this out of the service (a
            # busy journal after prepare is answered "accepted"), and the
            # journal guarantees a refused prepare was not written. A full
            # queue or a spent budget is overload; anything else (a failed
            # COMMIT: full disk, I/O error) is storage and must say so.
            if not _journal_was_busy(exc):
                raise CanonicalRememberUnavailable(
                    "this save could not be written to disk "
                    f"({type(exc.__cause__ or exc).__name__}); nothing was saved. "
                    "Check free disk space and that the SLM data folder is writable."
                ) from exc
            raise CanonicalRememberBusy(
                "too many saves are arriving at once; this one was not saved",
                retry_after_seconds=getattr(exc, "retry_after_seconds", 1),
            ) from exc
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
                after_commit=self._after_replayed_commit,
            )
        except (WriteCoordinatorError, ValueError, json.JSONDecodeError) as exc:
            raise CanonicalRememberUnavailable("pending remember recovery failed") from exc

    def _after_replayed_commit(
        self,
        entry: AdmissionEntry,
        request: RememberRequest | None,
        receipt: Mapping[str, Any],
    ) -> None:
        if request is None:
            # Found already committed: read the command back only to learn
            # whether it carried a replacement still to apply. The save
            # itself stands even if the command can no longer be read.
            try:
                request = self.journal.request_for(entry)
            except AdmissionPayloadError:
                logger.error(
                    "a recovered save's command cannot be read back, so any "
                    "replacement it asked for was not applied (%s)", entry.journal_id,
                )
                return
        self._apply_replacement(request, receipt)

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

    def set_fact_kinds(
        self,
        profile_id: str,
        items: Sequence[tuple[str, str]],
        *,
        idempotency_key: str | None = None,
    ) -> Mapping[str, Any]:
        """Set the kind of 1-200 facts in this profile, as the user's choice.

        Every kind is checked before anything is submitted: one unknown kind
        rejects the whole request rather than changing some of the facts.
        """
        from superlocalmemory.storage.memory_kinds import parse_kind

        pairs = list(items or ())
        if not 1 <= len(pairs) <= _MAX_KIND_ITEMS:
            raise ValueError(f"set between 1 and {_MAX_KIND_ITEMS} facts at a time")
        checked: list[list[str]] = []
        for fact_id, value in pairs:
            kind = parse_kind(value)
            if not isinstance(fact_id, str) or not fact_id or kind is None:
                raise ValueError(f"not a fact id and memory kind: {fact_id!r}, {value!r}")
            checked.append([fact_id, kind.value])
        return self._submit_mutation(
            CommandKind.SET_FACT_KIND,
            profile_id,
            {"items": checked},
            idempotency_key=idempotency_key,
        )

    def replace_by_caller(
        self,
        profile_id: str,
        replaces: str,
        successor_fact_id: str,
        *,
        trusted_actor_id: str,
        idempotency_key: str,
    ) -> Mapping[str, Any]:
        """Retire what ``replaces`` names in favour of a just-saved fact.

        One writer transaction proposes and applies a correction case per
        retired fact (see ``core/remember_replaces.py``).
        """
        return _thaw_command_value(self._submit_mutation(
            CommandKind.REPLACE_BY_CALLER,
            profile_id,
            {
                "replaces": replaces,
                "successor_fact_id": successor_fact_id,
                "trusted_actor_id": trusted_actor_id,
            },
            idempotency_key=idempotency_key,
        ))

    def _submit_mutation(
        self,
        kind: CommandKind,
        profile_id: str,
        payload: Mapping[str, Any],
        *,
        idempotency_key: str | None,
    ) -> Mapping[str, Any]:
        if not self._writer_open:
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
            # 5 s, not 2: one background commit can stall ~3 s under disk load,
            # and an edit now waits off the request loop (server/routes/memories).
            return dict(self.coordinator.submit(command, timeout=5.0).receipt)
        except (CanonicalMutationConflict, MutationTargetMissing):
            raise  # deterministic, not an outage: HTTP/CLI/MCP report it as such
        except CommandConflictError as exc:
            raise CanonicalMutationConflict(
                "idempotency key belongs to a different mutation request"
            ) from exc
        except (OwnershipRequiredError, WriteCoordinatorError) as exc:
            from superlocalmemory.core.writer_refusals import refusal_or_outage
            raise refusal_or_outage(exc) from exc  # a ledger refusal is a 409, not a 503

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
                metadata=_thaw_json(request.metadata),  # nested records were frozen in the queue
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
        """Run only deterministic SQLite mutation statements under the writer.

        A command for another profile than the active one is refused unless its
        kind may be routed there (``core/mutation_routing.py``).
        """
        payload = command.payload
        profile_id = _payload_text(payload, "profile_id")
        with self._binding_lock:
            target = classify_target(conn, command.kind, profile_id, self._profile_id)
            if target is MutationTarget.NOT_ROUTABLE:
                raise MutationNotRoutable(
                    f"{command.kind.value} cannot target a profile other than the active one")
            if target is MutationTarget.UNKNOWN_PROFILE:
                raise UnknownMutationProfile(f"profile {profile_id!r} does not exist")
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
    if kind is CommandKind.SET_FACT_KIND:
        return _set_fact_kinds(db, profile_id, payload, connection=connection)
    if kind is CommandKind.REPLACE_BY_CALLER:
        from superlocalmemory.core.remember_replaces import apply_replacement

        return apply_replacement(connection, profile_id, payload)
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


def _set_fact_kinds(
    db: DatabaseManager,
    profile_id: str,
    payload: Mapping[str, Any],
    *,
    connection: Any,
) -> dict[str, Any]:
    """Apply a user's kind choice inside the coordinator's transaction."""
    from superlocalmemory.storage.memory_kind_store import MemoryKindStore
    from superlocalmemory.storage.memory_kinds import parse_kind

    if not db.has_memory_kind_columns():
        return {"ok": False, "operation_id": "set_fact_kind", "changed": 0,
                "reason": "memory kinds are not available on this store yet"}
    items = []
    for pair in payload.get("items") or ():
        kind = parse_kind(pair[1]) if len(pair) == 2 else None
        if kind is not None:
            items.append((str(pair[0]), kind))
    results = MemoryKindStore(db).set_kinds(connection, profile_id, items, actor="user")
    # A fact owned by another profile comes back "not found" and is untouched.
    return {"ok": True, "operation_id": "set_fact_kind",
            "changed": sum(1 for r in results if r.get("ok")),
            "facts": [dict(r) for r in results]}


def _delete_fact(db: DatabaseManager, fact_id: str, profile_id: str) -> dict[str, Any]:
    row = _fact_row(db, fact_id, profile_id)
    if row is None:
        return {"ok": False, "operation_id": f"delete:{fact_id}", "fact_id": fact_id}
    # M042 deliberately retains immutable predecessor/successor history.
    # SQLite's FK would reject a direct delete, but exposing that as a generic
    # writer outage turns a real lifecycle conflict into a misleading 503.
    # A dedicated erasure workflow owns ledger removal; ordinary forget never
    # deletes a fact that is part of immutable correction history.
    # 4.1.22: a pending MACHINE proposal is overtaken by the user's delete, in
    # this same transaction (core/overtaken_cases.py); any other case refuses.
    from superlocalmemory.core import overtaken_cases as _ot
    from superlocalmemory.core.correction_protection import blocking_cases, protection_message

    blocked = blocking_cases(db, profile_id, fact_id)
    if blocked:
        raise CanonicalMutationConflict(protection_message(blocked))
    _ot.overtake(db, _ot.cases_naming(db, [fact_id]), user_action="delete",
                 actor_id=f"canonical-writer:{profile_id}", operation_id=f"delete:{fact_id}")
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
    from superlocalmemory.core.open_correction import refuse_second_open_case
    refuse_second_open_case(connection, profile_id, fact_id)  # a 409, never a false 503

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
        # A corrected fact keeps its kind once the edit is applied, never before:
        # the kind columns stay untyped here (LLD I6), a proposal is no one's
        # confirmed say-so, and ``_transition_correction`` copies the
        # predecessor's current kind onto this successor only when applied.
    )
    persisted_id = db.insert_fact_immutable(successor)
    from superlocalmemory.storage.correction_cases import (
        CorrectionActor,
        propose_on_connection,
    )

    trusted_actor_id = _payload_text(payload, "trusted_actor_id")
    from superlocalmemory.core import overtaken_cases as _ot  # the user's edit wins

    _ot.overtake(connection, _ot.cases_naming(connection, [fact_id], predecessor_only=True,
                                              profile_id=profile_id),
                 user_action="update", actor_id=trusted_actor_id,
                 operation_id=f"correction:{fact_id}:{persisted_id}")
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
    owner = connection.execute(
        "SELECT profile_id FROM correction_cases WHERE case_id = ?", (case_id,),
    ).fetchone()
    if owner is None or owner[0] != profile_id:
        raise CaseNotInProfile("correction case not found in this profile")
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
    if kind is CommandKind.APPLY_CORRECTION:
        _carry_over_confirmed_kind(
            connection, profile_id, case.predecessor_fact_id, case.successor_fact_id,
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


_KIND_CARRYOVER_COLUMNS = (
    "memory_kind", "memory_kind_source", "memory_kind_confidence",
    "memory_kind_recipe", "memory_kind_at",
)


def _carry_over_confirmed_kind(
    connection: Any, profile_id: str, predecessor_fact_id: str, successor_fact_id: str,
) -> None:
    """On apply, a successor inherits the predecessor's *current* kind - never before.

    Proposing a correction leaves its successor untyped (see
    ``_propose_correction_successor``): a suggestion no person has reviewed
    must not drive standing-rule injection or any other kind-gated behaviour
    (LLD I6). Applying is the one event that makes a successor's kind
    official, inside the same transaction that supersedes the predecessor, so
    the two never disagree. Reading the predecessor's columns at apply time
    (rather than at proposal time) also means a kind confirmed after the
    proposal was opened is still honoured. Columns the store does not have
    yet (an older database) are silently skipped - a kind is never the reason
    a review action fails.
    """
    try:
        columns = {row[1] for row in
                  connection.execute("PRAGMA table_info(atomic_facts)").fetchall()}
        present = [c for c in _KIND_CARRYOVER_COLUMNS if c in columns]
        if not present:
            return
        row = connection.execute(
            f"SELECT {', '.join(present)} FROM atomic_facts "
            "WHERE fact_id=? AND profile_id=?",
            (predecessor_fact_id, profile_id),
        ).fetchone()
        if row is None:
            return
        assignments = ", ".join(f"{c}=?" for c in present)
        connection.execute(
            f"UPDATE atomic_facts SET {assignments} WHERE fact_id=? AND profile_id=?",
            (*(row[c] for c in present), successor_fact_id, profile_id),
        )
    except Exception as exc:  # noqa: BLE001 - applying a correction must never fail on this
        logger.warning("correction apply: kind carry-over skipped (%s)", type(exc).__name__)


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
